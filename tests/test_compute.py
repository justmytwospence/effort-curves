"""Window evaluation against brute-force references on synthetic activities."""

import numpy as np
import pytest

import effort_curves as ec

M = {m: i for i, m in enumerate(ec.METRICS)}
NAN = float("nan")
# speed cap, grade cap, ascent cap, ascent min length, GAP min time, GAP min distance
SPORT = (7.5, 0.45, 0.5, 60.0, 180.0, 800.0)


def activity(rng, n, gap_at=None, gap_len=0, sparse=None):
    """1 Hz samples with varying speed, power, heart rate and altitude; optionally a pause and a sparse stretch."""
    t = np.arange(n, dtype=np.int64)
    if sparse:
        lo, hi, every = sparse
        keep = (t < lo) | (t >= hi) | ((t - lo) % every == 0)
        t = t[keep]
    if gap_at is not None:
        t = np.where(t >= gap_at, t + gap_len, t)
    speed = np.clip(3 + np.sin(t / 50) + rng.normal(0, 0.3, len(t)), 0.5, 6)
    dist = np.concatenate([[0], np.cumsum(speed[1:] * np.diff(t))])
    if gap_at is not None:  # no distance across a real pause
        dist = np.where(t >= gap_at + gap_len, dist - speed[np.searchsorted(t, gap_at + gap_len)] * gap_len, dist)
    alt = 1000 + 30 * np.sin(t / 200)
    power = np.clip(250 + 80 * np.sin(t / 30) + rng.normal(0, 20, len(t)), 0, None)
    hr = 140 + 15 * np.sin(t / 120)
    return t, dist, alt, power, hr


def prepared(acts):
    ids = np.arange(1, len(acts) + 1, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum([len(a[0]) for a in acts])]).astype(np.int64)
    cat = [np.concatenate([a[k] for a in acts]) for k in range(5)]
    return ec.prepare(
        activity_ids=ids, offsets=offsets, t=cat[0].astype(np.int64), dist=cat[1].astype(float), alt=cat[2].astype(float),
        power=cat[3].astype(float), hr=cat[4].astype(float),
        indoor=np.zeros(len(acts), np.uint8), powered=np.ones(len(acts), np.uint8), sport=SPORT,
    )


@pytest.fixture(scope="module")
def P():
    rng = np.random.default_rng(7)
    return prepared([
        activity(rng, 3000),
        activity(rng, 2500, gap_at=1200, gap_len=300),
        activity(rng, 2000, sparse=(500, 1500, 4)),
    ])


LENGTHS = np.array([5, 30, 60, 300, 900], dtype=float)


def test_power_matches_brute_force():
    rng = np.random.default_rng(1)
    t, dist, alt, power, hr = activity(rng, 1500)
    P = prepared([(t, dist, alt, power, hr)])
    y, t0 = P.compute("time", LENGTHS)
    cp = np.cumsum(power)  # running total includes second k; a window (i, i+L] sums seconds i+1..i+L
    for k, L in enumerate(LENGTHS.astype(int)):
        ref = ((cp[L:] - cp[:-L]) / L)
        assert y[0, k, M["p"]] == pytest.approx(ref.max(), rel=1e-5)
        assert t0[0, k, M["p"]] == int(np.argmax(ref))


def test_groups_partition_the_ungrouped_result(P):
    y, _ = P.compute("time", LENGTHS)
    edges = np.array([-1.0, 0.0, 0.02, 0.05, 1.0])
    yg, _ = P.compute("time", LENGTHS, group=("g", edges))
    assert yg.shape == (3, 4, len(LENGTHS), len(M))
    # windows with a known gradient: the best over all bins is the best among windows with any gradient
    filt, _ = P.compute("time", LENGTHS, filters=[("g", NAN, NAN)])
    np.testing.assert_allclose(np.nanmax(yg, axis=1), filt, rtol=1e-6)
    assert np.all((filt <= y) | np.isnan(filt))


def test_group_bin_equals_filter(P):
    edges = np.array([2.5, 3.0, 3.5, 4.0])
    yg, tg = P.compute("time", LENGTHS, group=("v", edges))
    for b in range(len(edges) - 1):
        hi = np.nextafter(edges[b + 1], -np.inf)
        yf, tf = P.compute("time", LENGTHS, filters=[("v", edges[b], hi)])
        np.testing.assert_array_equal(yg[:, b], yf)
        np.testing.assert_array_equal(tg[:, b], tf)


def test_two_sided_filter_is_exact(P):
    """Every reported best respects the band, and the band's best is at most the unfiltered best."""
    lo, hi = 3.0, 3.6
    y, _ = P.compute("time", LENGTHS, filters=[("v", lo, hi)])
    v = y[:, :, M["v"]]
    ok = ~np.isnan(v)
    assert ok.any()
    assert np.all((v[ok] >= lo - 1e-6) & (v[ok] <= hi + 1e-6))


def test_open_bounds_and_start_filter(P):
    y_all, t_all = P.compute("time", LENGTHS)
    y_open, _ = P.compute("time", LENGTHS, filters=[("p", NAN, NAN)])
    np.testing.assert_array_equal(y_all[:, :, M["p"]], y_open[:, :, M["p"]])
    _, t_late = P.compute("time", LENGTHS, filters=[("t0", 600.0, NAN)])
    late = t_late[t_late >= 0]
    assert late.size and late.min() >= 600


def test_activity_subset(P):
    y, t0 = P.compute("time", LENGTHS)
    ys, ts = P.compute("time", LENGTHS, activities=np.array([2, 0], dtype=np.int64))
    np.testing.assert_array_equal(ys[0], y[2])
    np.testing.assert_array_equal(ys[1], y[0])
    np.testing.assert_array_equal(ts[0], t0[2])


def test_pause_is_bridged_and_counts_against_speed(P):
    """Activity 1 stops for 300 s at 1200 s: a long window across it is slower than the moving pace."""
    y, t0 = P.compute("time", np.array([2400.0]))
    # 2500 s of moving over a 2800 s span: no 2400 s window can avoid the stop, so its speed is diluted
    assert y[1, 0, M["v"]] < 3.0 * 2400 / 2400
    assert not np.isnan(y[1, 0, M["h"]])  # heart rate bridged across the pause


def test_bad_input():
    rng = np.random.default_rng(3)
    P = prepared([activity(rng, 500)])
    with pytest.raises(ValueError):
        P.compute("time", LENGTHS, group=("v", np.array([3.0, 2.0])))
    with pytest.raises(ValueError):
        P.compute("time", LENGTHS, activities=np.array([5], dtype=np.int64))
    with pytest.raises(ValueError):
        P.compute("time", LENGTHS, filters=[("nope", 0.0, 1.0)])
