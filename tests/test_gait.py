"""Gait (running or walking, per second stepping) against a reference written out by hand."""

import numpy as np
import pytest

import effort_curves as ec

NAN = float("nan")
SPORT = (7.5, 0.45, 0.5, 60.0, 180.0, 800.0, 220.0)
THR = 145.0


def cadence(n, rng):
    """Running at about 175, walking at about 112, with short blips that must not start a bout, a dropout (no
    cadence) and a stretch under 20 (not stepping)."""
    c = 175 + rng.normal(0, 3, n)
    c[300:305] = 110  # a 5 s blip: still running
    c[500:700] = 112 + rng.normal(0, 3, 200)  # walking
    c[600:603] = 172  # a 3 s blip: still walking
    c[900:920] = NAN  # no cadence
    c[1000:1010] = 10  # under 20: not stepping
    c[1010:1016] = 110  # 6 s of walking, after the gap: not a bout
    return c


def prepare(cad):
    n = len(cad)
    t = np.arange(n, dtype=np.int64)
    return ec.prepare(
        activity_ids=np.array([1], np.int64), offsets=np.array([0, n], np.int64), t=t, dist=t * 3.0,
        alt=np.full(n, 1000.0), power=np.full(n, NAN), hr=np.full(n, 140.0), indoor=np.zeros(1, np.uint8),
        powered=np.zeros(1, np.uint8), sport=SPORT, cadence=cad,
    )


def reference_labels(cad, thr, bout):
    """1 running, 0 walking, -1 not stepping: the same rule, written plainly."""
    step = np.isfinite(cad) & (cad >= 20) & (cad <= SPORT[6])
    lab = np.full(len(cad), -1)
    state, pend = None, []
    for g in np.flatnonzero(step):
        r = int(cad[g] >= thr)
        if state is None:
            state = r
            lab[g] = r
        elif r == state:
            pend = []
            lab[g] = r
        else:
            lab[g] = state
            pend.append(g)
            if len(pend) >= bout:
                lab[pend] = r
                state, pend = r, []
    return lab


@pytest.fixture(scope="module")
def case():
    cad = cadence(1600, np.random.default_rng(3))
    P = prepare(cad)
    P.set_gait(THR, 10)
    return cad, P, reference_labels(cad, THR, 10)


def test_labels_follow_bouts(case):
    cad, P, lab = case
    assert P.gait == THR
    # one-second windows binned by start: bin i holds second i + 1's gait
    n = len(cad)
    s = P.seconds_in("t0", np.arange(0.0, n, 1.0), 1.0, by_gait=True)
    assert s.shape == (1, n - 1, 2)
    got = np.where(s[0, :, 1] > 0, 1, np.where(s[0, :, 0] > 0, 0, -1))
    np.testing.assert_array_equal(got[: n - 2], lab[1 : n - 1])
    # the blips stay in the bout around them, the walk is a bout
    assert set(lab[300:305]) == {1} and set(lab[600:603]) == {0} and set(lab[510:700]) == {0}
    assert set(lab[1010:1016]) == {1}


@pytest.mark.parametrize("L", [10, 30, 120])
def test_running_share_matches_a_count(case, L):
    cad, P, lab = case
    n = len(cad)
    stepping = np.concatenate([[0], np.cumsum(lab >= 0)])
    running = np.concatenate([[0], np.cumsum(lab == 1)])
    ref = []
    for i in range(n - L):
        ped = stepping[i + L + 1] - stepping[i + 1]  # seconds i + 1 .. i + L
        ref.append((running[i + L + 1] - running[i + 1]) / ped if ped >= max(5, 0.25 * L) else NAN)
    ref = np.array(ref)
    edges = np.linspace(0, 1.0000001, 11)
    got = P.seconds_in("rs", edges, float(L))[0]
    want, _ = np.histogram(ref[np.isfinite(ref)], bins=edges)
    np.testing.assert_array_equal(got, want)


def test_split_adds_up_to_the_seconds_stepping(case):
    cad, P, lab = case
    e = np.arange(0.0, 1600.0, 100.0)
    plain = P.seconds_in("t0", e, 30.0)
    split = P.seconds_in("t0", e, 30.0, by_gait=True)
    assert np.all(split.sum(axis=2) <= plain)
    # bins whose starts' first seconds all step: every start counts once; the dropouts' bins lose exactly those seconds
    tot = split.sum(axis=2)[0]
    ok = [k for k in range(len(e) - 1) if e[k + 1] <= 899 or e[k] >= 1010]
    np.testing.assert_array_equal(tot[ok], plain[0, ok])
    # starts 900-999 are seconds 901-1000 (19 without cadence, one under 20); starts 1000-1099 lose seconds 1001-1009
    assert plain[0, 9] - tot[9] == 20 and plain[0, 10] - tot[10] == 9


def test_clearing_gait():
    cad = cadence(1200, np.random.default_rng(5))
    P = prepare(cad)
    y, _ = P.compute("time", np.array([60.0]))
    assert np.all(np.isnan(y[..., ec.METRICS.index("rs")]))
    P.set_gait(THR)
    y, _ = P.compute("time", np.array([60.0]))
    assert y[0, 0, ec.METRICS.index("rs")] == pytest.approx(1.0)
    P.set_gait(NAN)
    assert np.isnan(P.gait)
    assert P.seconds_in("t0", np.array([0.0, 1e6]), 30.0, by_gait=True).sum() == 0


def test_cadence_seconds_is_a_histogram(case):
    cad, P, _ = case
    e = np.arange(0.0, 241.0, 1.0)
    h = P.cadence_seconds(e)
    step = np.isfinite(cad) & (cad >= 20) & (cad <= SPORT[6])
    assert h.sum() == step.sum()
    want, _ = np.histogram(cad[step], bins=e)
    assert np.abs(h - want).sum() <= 2 * 4  # values a hair off an edge may land in the neighbour


def test_threshold_finds_the_valley():
    e = np.arange(40.0, 241.0, 1.0)
    c = (e[:-1] + e[1:]) / 2
    two = 2000 * np.exp(-0.5 * ((c - 178) / 5) ** 2) + 60 * np.exp(-0.5 * ((c - 108) / 10) ** 2)
    thr = ec.gait_threshold(two, e)
    assert 125 < thr < 160
    one = 2000 * np.exp(-0.5 * ((c - 178) / 5) ** 2)
    assert np.isnan(ec.gait_threshold(one, e))
    with pytest.raises(ValueError):
        ec.gait_threshold(two, e[:-1])
