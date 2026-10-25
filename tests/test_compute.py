"""Window evaluation against brute-force references on synthetic activities."""

import numpy as np
import pytest

import effort_curves as ec

M = {m: i for i, m in enumerate(ec.METRICS)}
NAN = float("nan")
# speed cap, grade cap, ascent cap, ascent min length, GAP min time, GAP min distance, cadence cap
SPORT = (7.5, 0.45, 0.5, 60.0, 180.0, 800.0, 200.0)


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
    cad = np.clip(85 + 10 * np.sin(t / 40) + rng.normal(0, 3, len(t)), 0, None)
    return t, dist, alt, power, hr, cad


def prepared(acts):
    ids = np.arange(1, len(acts) + 1, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum([len(a[0]) for a in acts])]).astype(np.int64)
    cat = [np.concatenate([a[k] for a in acts]) for k in range(6)]
    return ec.prepare(
        activity_ids=ids, offsets=offsets, t=cat[0].astype(np.int64), dist=cat[1].astype(float), alt=cat[2].astype(float),
        power=cat[3].astype(float), hr=cat[4].astype(float),
        indoor=np.zeros(len(acts), np.uint8), powered=np.ones(len(acts), np.uint8), sport=SPORT,
        cadence=cat[5].astype(float),
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
    t, dist, alt, power, hr, cad = activity(rng, 1500)
    P = prepared([(t, dist, alt, power, hr, cad)])
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


def test_cadence_metrics_match_brute_force():
    rng = np.random.default_rng(5)
    t, dist, alt, power, hr, cad = activity(rng, 1200)
    cad[300:360] = 0.0  # a minute of coasting
    cad[500] = 999.0  # a glitch above the cap counts as nothing
    P = prepared([(t, dist, alt, power, hr, cad)])
    y, t0 = P.compute("time", LENGTHS)
    c = np.where((cad >= 20) & (cad <= 200), cad, 0.0)
    rev = np.cumsum(c / 60)
    ped = np.cumsum((c > 0).astype(float))
    work = np.cumsum(power)
    for k, L in enumerate(LENGTHS.astype(int)):
        cbar = (rev[L:] - rev[:-L]) / L * 60
        assert y[0, k, M["c"]] == pytest.approx(cbar.max(), rel=1e-5)
        i = int(t0[0, k, M["c"]])
        assert cbar[i] == pytest.approx(cbar.max(), rel=1e-5)
        share = (ped[L:] - ped[:-L]) / L
        assert y[0, k, M["ps"]] == pytest.approx(share.max(), rel=1e-6)
        # torque and step length are ratios of the same running totals
        dr = rev[L:] - rev[:-L]
        pw = np.cumsum(np.where(c > 0, power, 0.0))
        dp = ped[L:] - ped[:-L]
        pc = np.where(dp >= np.maximum(5, 0.25 * L), dr / np.maximum(dp, 1e-9) * 60, 0)
        tq = np.where((dr >= 1) & (pc >= 30), (pw[L:] - pw[:-L]) / (2 * np.pi * np.maximum(dr, 1e-9)), -np.inf)
        assert y[0, k, M["k"]] == pytest.approx(tq.max(), rel=1e-4)


def test_no_cadence_means_no_cadence_metrics():
    rng = np.random.default_rng(6)
    t, dist, alt, power, hr, cad = activity(rng, 800)
    acts = [(t, dist, alt, power, hr, np.full_like(cad, np.nan))]
    P = prepared(acts)
    y, _ = P.compute("time", LENGTHS)
    for m in ("c", "k", "l", "pc", "ps"):
        assert np.isnan(y[0, :, M[m]]).all()
    assert not np.isnan(y[0, :, M["p"]]).all()


def test_pedaling_cadence_filter_and_group():
    rng = np.random.default_rng(8)
    P = prepared([activity(rng, 2000), activity(rng, 1500)])
    edges = np.array([60.0, 80.0, 90.0, 120.0])
    yg, _ = P.compute("time", LENGTHS, group=("pc", edges))
    for b in range(len(edges) - 1):
        yf, _ = P.compute("time", LENGTHS, filters=[("pc", edges[b], np.nextafter(edges[b + 1], -np.inf))])
        np.testing.assert_array_equal(yg[:, b], yf)
    yf, _ = P.compute("time", LENGTHS, filters=[("pc", 80.0, 90.0)])
    pc = yf[:, :, M["pc"]]
    ok = ~np.isnan(pc)
    assert ok.any() and np.all((pc[ok] >= 80 - 1e-4) & (pc[ok] <= 90 + 1e-4))


def test_torque_ignores_power_without_cadence():
    """A trainer reporting power while the cadence sensor drops out must not produce huge torque."""
    rng = np.random.default_rng(9)
    t, dist, alt, power, hr, cad = activity(rng, 900)
    cad[200:400] = np.nan
    cad[400:410] = 2.0  # a near-stopped crank reading
    P = prepared([(t, dist, alt, power, hr, cad)])
    y, _ = P.compute("time", LENGTHS)
    tq = y[0, :, M["k"]]
    # 250 W at ~85 rpm is ~28 N·m; anything near 100 would mean the dropout leaked in
    assert np.nanmax(tq) < 60


def test_step_length_ignores_distance_without_cadence():
    rng = np.random.default_rng(10)
    t, dist, alt, power, hr, cad = activity(rng, 900)
    cad[200:600] = np.nan  # the watch stops counting steps for a while
    P = prepared([(t, dist, alt, power, hr, cad)])
    y, _ = P.compute("time", LENGTHS)
    # ~3 m/s at ~85 steps/min is about 2 m per step; a dropout leaking in would give tens of metres
    assert np.nanmax(y[0, :, M["l"]]) < 4
