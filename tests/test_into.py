"""The frontier over where a window starts, against every start's own window."""

import numpy as np
import pytest

import effort_curves as ec
from test_compute import NAN, activity, prepared

M = {m: i for i, m in enumerate(ec.METRICS)}


@pytest.fixture(scope="module")
def P():
    rng = np.random.default_rng(5)
    return prepared([
        activity(rng, 2400),
        activity(rng, 2000, gap_at=900, gap_len=240),
        activity(rng, 1800, sparse=(400, 1200, 3)),
    ])


def per_start(P, basis, length, metric, rows, filters=()):
    """Every start's own window: (row, start second, value), from compute grouped by 1 s of t0."""
    edges = np.arange(0, 3200, dtype=float)
    y, t = P.compute(basis, np.array([float(length)]), filters=list(filters), group=("t0", edges), activities=np.array(rows, dtype=np.int64))
    out = []
    for r in range(len(rows)):
        for b in range(len(edges) - 1):
            v = y[r, b, 0, M[metric]]
            if not np.isnan(v):
                out.append((r, float(edges[b]), float(v), int(t[r, b, 0, M[metric]])))
    return out


def reference(points):
    """The staircase from the far end (ties in x: the best), then efforts by activity and overlapping windows."""
    pts = sorted(points, key=lambda p: (-p[1], -p[2]))
    stair, held = [], None
    for p in pts:
        if held is None or p[2] > held:
            held = p[2]
            stair.append(p)
    return stair[::-1]


@pytest.mark.parametrize("metric,basis,length", [("v", "time", 300), ("p", "time", 60), ("v", "dist", 1000)])
def test_staircase_matches_every_start(P, metric, basis, length):
    rows = [0, 1, 2]
    ref = reference(per_start(P, basis, length, metric, rows))
    e, p = P.frontier_into(basis, float(length), metric, "t0", np.array(rows, dtype=np.int64), [rows])
    got = list(zip(p["x"].tolist(), p["y"].tolist()))
    assert len(got) == len(ref) >= 3
    for (gx, gy), r in zip(got, ref):
        assert gx == r[1]
        assert gy == pytest.approx(r[2], rel=1e-6)
    # efforts tile the staircase in order, each holding from the previous one's last x
    assert list(e["effort"]) == list(range(len(e["effort"])))
    assert np.isnan(e["from_x"][0])
    assert list(e["from_x"][1:]) == list(e["x"][:-1])
    last = {int(k): x for k, x in zip(p["effort"], p["x"])}
    assert [last[i] for i in range(len(e["x"]))] == list(e["x"])


def test_filters_and_sets(P):
    flt = [("h", 135.0, NAN)]
    for rows in ([0, 2], [1]):
        ref = reference(per_start(P, "time", 120, "v", rows, flt))
        idx = [0, 1, 2]
        e, p = P.frontier_into("time", 120.0, "v", "t0", np.array(idx, dtype=np.int64), [rows], filters=flt)
        assert [(x, round(y, 4)) for x, y in zip(p["x"].tolist(), p["y"].tolist())] == [(r[1], round(r[2], 4)) for r in ref]


def test_distance_into_holds_at_least_d(P):
    """x in metres: each staircase point is the best window starting at least that far in."""
    e, p = P.frontier_into("time", 300.0, "v", "d0", np.array([0, 1, 2], dtype=np.int64), [[0, 1, 2]])
    for x, y in list(zip(p["x"].tolist(), p["y"].tolist()))[::7]:
        best, _ = P.compute("time", np.array([300.0]), filters=[("d0", x, NAN)])
        assert np.nanmax(best[:, 0, M["v"]]) == pytest.approx(y, rel=1e-6)


def per_start_xy(P, basis, length, xm, ym, rows):
    """Every start's own window: (row, its x metric, its y metric)."""
    edges = np.arange(0, 3200, dtype=float)
    y, _ = P.compute(basis, np.array([float(length)]), group=("t0", edges), activities=np.array(rows, dtype=np.int64))
    out = []
    for r in range(len(rows)):
        for b in range(len(edges) - 1):
            xv, yv = y[r, b, 0, M[xm]], y[r, b, 0, M[ym]]
            if not (np.isnan(xv) or np.isnan(yv)):
                out.append((r, float(xv), float(yv)))
    return out


@pytest.mark.parametrize("direction", ["le", "ge"])
def test_metric_dimension_both_directions(P, direction):
    """x is the window's own heart rate: the best pace among windows at most (le) or at least (ge) that heart rate."""
    rows = [0, 1, 2]
    pts = per_start_xy(P, "time", 300, "h", "v", rows)
    s = -1 if direction == "le" else 1
    e, p = P.frontier_into("time", 300.0, "v", "h", np.array(rows, dtype=np.int64), [rows], direction=direction)
    got = list(zip(p["x"].tolist(), p["y"].tolist()))
    assert len(got) >= 3
    hx = np.array([s * x for _, x, _ in pts])  # the reference's x is float32: compare with a margin
    hy = np.array([y for _, _, y in pts])
    tol = 1e-3
    for gx, gy in got:
        # on the staircase: no strictly harder window is faster
        harder = hx > s * gx + tol
        assert not np.any(hy[harder] > gy + 1e-5)
        # and it is a real window
        assert np.any((np.abs(hx - s * gx) <= tol) & (np.abs(hy - gy) <= 1e-5))
    # every window is covered: some staircase point at least as hard is at least as fast
    gx_ = np.array([s * x for x, _ in got]); gy_ = np.array([y for _, y in got])
    for x, y in zip(hx, hy):
        assert np.any((gx_ >= x - tol) & (gy_ >= y - 1e-5))
    # efforts come out in the dimension's own sense
    if direction == "le":
        assert all(a > b for a, b in zip(e["x"][:-1], e["x"][1:]))


def test_bad_dim(P):
    with pytest.raises(ValueError):
        P.frontier_into("time", 60.0, "v", "zz", np.array([0], dtype=np.int64), [[0]])
    with pytest.raises(ValueError):
        P.frontier_into("time", 60.0, "v", "h", np.array([0], dtype=np.int64), [[0]], direction="up")
