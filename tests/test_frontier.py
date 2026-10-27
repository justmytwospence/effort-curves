"""The frontier as efforts, against a reference built from the exhaustive curve at every integer length."""

import numpy as np
import pytest

import effort_curves as ec
from test_compute import NAN, activity, prepared

M = {m: i for i, m in enumerate(ec.METRICS)}


@pytest.fixture(scope="module")
def P():
    rng = np.random.default_rng(11)
    return prepared([
        activity(rng, 2400),
        activity(rng, 2000, gap_at=900, gap_len=240),
        activity(rng, 1800, sparse=(400, 1200, 3)),
        activity(rng, 1500),
    ])


def grid(lo, hi, ratio):
    """Integer lengths about `ratio` apart."""
    out = [lo]
    while out[-1] < hi:
        out.append(max(out[-1] + 1, int(round(out[-1] * ratio))))
    return np.array(out[:-1], dtype=float)


def reference(y, t0, lengths, rows):
    """Efforts of the frontier of `rows` from per-length bests (rows, lengths): envelope, efficient lengths, overlap."""
    best = np.full(len(lengths), np.nan)
    who = np.full(len(lengths), -1)
    for r in rows:
        take = ~np.isnan(y[r]) & (np.isnan(best) | (y[r] > best))
        best[take], who[take] = y[r][take], r
    out = []
    held = np.nan
    eff = []
    for k in range(len(lengths) - 1, -1, -1):
        if not np.isnan(best[k]) and (np.isnan(held) or best[k] > held):
            held = best[k]
            eff.append(k)
    for k in sorted(eff):
        r = who[k]
        s = (t0[r, k], t0[r, k] + lengths[k])
        if out and out[-1]["row"] == r and s[0] < out[-1]["span"][1] and s[1] > out[-1]["span"][0]:
            d = out[-1]
            d.update(len=lengths[k], t0=int(t0[r, k]), y=float(best[k]), span=(min(d["span"][0], s[0]), max(d["span"][1], s[1])))
        else:
            out.append({"row": int(r), "len": lengths[k], "t0": int(t0[r, k]), "y": float(best[k]), "span": s})
    return out


def efforts(d, s=0, c=0):
    keep = (d["set"] == s) & (d["cell"] == c)
    return [{k: d[k][keep][i] for k in d} for i in range(int(keep.sum()))]


def test_refined_frontier_matches_every_integer_length(P):
    coarse = grid(5, 1400, 1.04)
    fine = np.arange(5, coarse[-1] + 1, dtype=float)
    yf, tf = P.compute("time", fine)
    ref = reference(yf[..., M["v"]], tf[..., M["v"]], fine, range(4))
    y, t0 = P.compute("time", coarse)
    acts = np.arange(4, dtype=np.int64)
    d = P.frontier("time", "v", coarse, y[:, None, :, M["v"]].copy(), t0[:, None, :, M["v"]].copy(), acts, [[0, 1, 2, 3]])
    got = efforts(d)
    assert len(ref) >= 3
    assert [(e["row"], e["len"]) for e in got] == [(e["row"], e["len"]) for e in ref]
    for g, r in zip(got, ref):
        assert g["t0"] == r["t0"]
        assert g["y"] == pytest.approx(r["y"], rel=1e-6)
    # each effort holds the frontier from the previous one's full length
    assert np.isnan(got[0]["from_len"])
    assert [g["from_len"] for g in got[1:]] == [g["len"] for g in got[:-1]]


def test_grid_resolution_without_refine(P):
    coarse = grid(5, 1400, 1.04)
    y, t0 = P.compute("time", coarse)
    ref = reference(y[..., M["p"]], t0[..., M["p"]], coarse, range(4))
    d = P.frontier("time", "p", coarse, y[:, None, :, M["p"]].copy(), t0[:, None, :, M["p"]].copy(),
                   np.arange(4, dtype=np.int64), [[0, 1, 2, 3]], refine=False)
    assert [(e["row"], e["len"], e["t0"]) for e in efforts(d)] == [(e["row"], e["len"], e["t0"]) for e in ref]


def test_single_rows_give_each_activitys_own_efforts(P):
    coarse = grid(5, 1400, 1.04)
    y, t0 = P.compute("time", coarse)
    d = P.frontier("time", "v", coarse, y[:, None, :, M["v"]].copy(), t0[:, None, :, M["v"]].copy(),
                   np.arange(4, dtype=np.int64), [[0], [1], [2], [3]], refine=False)
    for r in range(4):
        ref = reference(y[..., M["v"]], t0[..., M["v"]], coarse, [r])
        assert [(e["len"], e["t0"]) for e in efforts(d, s=r)] == [(e["len"], e["t0"]) for e in ref]
        assert set(efforts(d, s=r)[i]["row"] for i in range(len(ref))) == {r}


def test_cells_and_filters_are_honoured_when_refining(P):
    edges = np.array([-1.0, 0.0, 1.0])
    coarse = grid(5, 900, 1.05)
    fine = np.arange(5, coarse[-1] + 1, dtype=float)
    flt = [("h", 135.0, NAN)]
    y, t0 = P.compute("time", coarse, filters=flt, group=("g", edges))
    d = P.frontier("time", "v", coarse, y[..., M["v"]].copy(), t0[..., M["v"]].copy(), np.arange(4, dtype=np.int64),
                   [[0, 1, 2, 3]], filters=flt, group=("g", edges))
    yf, tf = P.compute("time", fine, filters=flt, group=("g", edges))
    for c in range(2):
        ref = reference(yf[:, c, :, M["v"]], tf[:, c, :, M["v"]], fine, range(4))
        assert [(e["row"], e["len"]) for e in efforts(d, c=c)] == [(e["row"], e["len"]) for e in ref]


def test_distance_handoffs_are_exact(P):
    coarse = grid(100, 4000, 1.03)
    fine = np.arange(100, coarse[-1] + 1, dtype=float)
    yf, _ = P.compute("dist", fine)
    v = yf[..., M["v"]]
    env = np.nanmax(v, axis=0)
    held = np.maximum.accumulate(np.nan_to_num(env, nan=-1)[::-1])[::-1]
    y, t0 = P.compute("dist", coarse)
    d = P.frontier("dist", "v", coarse, y[:, None, :, M["v"]].copy(), t0[:, None, :, M["v"]].copy(),
                   np.arange(4, dtype=np.int64), [[0, 1, 2, 3]])
    got = efforts(d)
    assert len(got) >= 2
    for e in got[:-1]:
        k = int(e["len"]) - 100
        # a handoff: the effort's own window is the frontier at its full length, and nothing that long is longer
        assert e["y"] == pytest.approx(env[k], rel=1e-6)
        assert held[k + 1] < e["y"]


def test_bad_shapes(P):
    coarse = grid(5, 600, 1.1)
    y, t0 = P.compute("time", coarse)
    with pytest.raises(ValueError):
        P.frontier("time", "v", coarse, y[:, None, :-1, 1].copy(), t0[:, None, :-1, 1].copy(), np.arange(4, dtype=np.int64), [[0]])
    with pytest.raises(ValueError):
        P.frontier("time", "v", coarse, y[:, None, :, 1].copy(), t0[:, None, :, 1].copy(), np.arange(4, dtype=np.int64), [[9]])


@pytest.mark.parametrize("metric", ["v", "p", "x"])
def test_coarse_grid_is_refined_exactly(P, metric):
    """A 10 % grid: long steps are refined by subdivision, and still match every integer length."""
    coarse = grid(5, 1800, 1.10)
    fine = np.arange(5, coarse[-1] + 1, dtype=float)
    yf, tf = P.compute("time", fine)
    ref = reference(yf[..., M[metric]], tf[..., M[metric]], fine, range(4))
    y, t0 = P.compute("time", coarse)
    d = P.frontier("time", metric, coarse, y[:, None, :, M[metric]].copy(), t0[:, None, :, M[metric]].copy(),
                   np.arange(4, dtype=np.int64), [[0, 1, 2, 3]])
    assert [(e["row"], e["len"], e["t0"]) for e in efforts(d)] == [(e["row"], e["len"], e["t0"]) for e in ref]


def test_filters_without_groups_match_every_integer_length(P):
    """Filters alone take the single-metric path; it must agree with the full evaluation."""
    coarse = grid(5, 1200, 1.06)
    fine = np.arange(5, coarse[-1] + 1, dtype=float)
    flt = [("g", -0.02, 0.03), ("h", 130.0, NAN)]
    yf, tf = P.compute("time", fine, filters=flt)
    y, t0 = P.compute("time", coarse, filters=flt)
    for metric in ("v", "p"):
        ref = reference(yf[..., M[metric]], tf[..., M[metric]], fine, range(4))
        d = P.frontier("time", metric, coarse, y[:, None, :, M[metric]].copy(), t0[:, None, :, M[metric]].copy(),
                       np.arange(4, dtype=np.int64), [[0, 1, 2, 3]], filters=flt)
        assert [(e["row"], e["len"], e["t0"]) for e in efforts(d)] == [(e["row"], e["len"], e["t0"]) for e in ref]
