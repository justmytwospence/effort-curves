"""Benchmark the Rust prototype against the current cube (built by current_cube.py into out/current).

Per sport: build time, stored size, agreement with the cube at the cube's 24 durations / 19 distances.
For Run also: exhaustive vs strided accuracy, single-activity update time, and a two-sided gradient band query.
"""

import json
import os
import sys
import time

import duckdb
import numpy as np
import polars as pl

import effort_curves as ec
from current_cube import OUT as CUR
from load import NB, dist_grid, envelope, gap_factors, load, sport_params, time_grid

MI = {m: i for i, m in enumerate(ec.METRICS)}
COMPARE = ["p", "v", "x", "a", "h"]
EPS = 0.01
HR_BRIDGE = int(os.environ.get("HR_BRIDGE", "600"))


def clock(f, *a, **k):
    t = time.perf_counter()
    r = f(*a, **k)
    return r, time.perf_counter() - t


def cube_table(con, sport):
    con.execute(f"create or replace table cube as select * from '{CUR / sport}/*.parquet'")


def cube_env(con, basis):
    """Per-activity best per length with the notebook's monotone envelope, for each compared metric."""
    cols = ", ".join(f"max({m}) over w as {m}" for m in COMPARE)
    return con.sql(f"""
        with per as (select activity_id, len, {', '.join(f'max({m}) as {m}' for m in COMPARE)}
                     from cube where basis = '{basis}' group by all)
        select activity_id, len, {cols} from per
        window w as (partition by activity_id order by len desc rows between unbounded preceding and current row)
    """).pl()


def rust_long(ids, grid, env, lens):
    idx = [int(np.searchsorted(grid, l)) for l in lens]
    assert np.allclose(grid[idx], lens)
    rows = []
    for m in COMPARE:
        v = env[:, idx, MI[m]]
        rows.append(pl.DataFrame({
            "activity_id": np.repeat(ids, len(lens)), "len": np.tile(np.array(lens, dtype=np.int64), len(ids)),
            "metric": m, "rust": v.reshape(-1).astype(np.float64),
        }))
    return pl.concat(rows)


def agreement(cube, rust):
    c = cube.unpivot(index=["activity_id", "len"], on=COMPARE, variable_name="metric", value_name="cube")
    j = c.join(rust, on=["activity_id", "len", "metric"], how="inner").filter(
        pl.col("cube").is_not_null() & pl.col("rust").is_not_nan() & (pl.col("cube") > 0)
    )
    j = j.with_columns(rel=pl.col("rust") / pl.col("cube") - 1)
    per_metric = j.group_by("metric").agg(
        n=pl.len(),
        median=pl.col("rel").median(),
        p05=pl.col("rel").quantile(0.05),
        p95=pl.col("rel").quantile(0.95),
        within_1pct=(pl.col("rel").abs() <= 0.01).mean(),
        within_5pct=(pl.col("rel").abs() <= 0.05).mean(),
    ).sort("metric")
    best = j.group_by("metric", "len").agg(cube=pl.col("cube").max(), rust=pl.col("rust").max()).with_columns(
        rel=pl.col("rust") / pl.col("cube") - 1
    ).sort("metric", "len")
    return per_metric, best, j


def breakpoints(env):
    """Rows needed to store each activity's envelope exactly: lengths where it steps down, per metric."""
    nxt = np.concatenate([env[:, 1:, :], np.full_like(env[:, :1, :], np.nan)], axis=1)
    step = ~np.isnan(env) & ((env > nxt) | np.isnan(nxt))
    return int(step.sum())


def bench_sport(sport, report, deep=False):
    r = report[sport] = {"hr_bridge": HR_BRIDGE}
    d, r["load_s"] = clock(load, sport)
    P, r["prepare_s"] = clock(ec.prepare, **d, sport=sport_params(sport), gap=gap_factors(sport), hr_bridge=HR_BRIDGE)
    r["activities"], r["segments"], r["seconds"] = len(d["activity_ids"]), P.n_segments, P.seconds
    tg, dg = time_grid(), dist_grid()
    r["time_lengths"], r["dist_lengths"] = len(tg), len(dg)
    (yt, tt), r["time_compute_s"] = clock(P.compute, "time", tg, EPS)
    (yd, td), r["dist_compute_s"] = clock(P.compute, "dist", dg, EPS)
    r["build_total_s"] = r["load_s"] + r["prepare_s"] + r["time_compute_s"] + r["dist_compute_s"]
    et, ed = envelope(yt), envelope(yd)
    r["stored_rows"] = breakpoints(et) + breakpoints(ed)

    con = duckdb.connect()
    con.execute("SET threads = 6")
    cube_table(con, sport)
    r["cube_cells"] = con.sql("select count(*) from cube").fetchone()[0]
    r["cube_build_s"] = json.loads((CUR / "timings.json").read_text()).get(sport)
    ids = P.activity_ids
    at, best_t, jt = agreement(cube_env(con, "time"), rust_long(ids, tg, et, NB["DURATIONS"]))
    ad, best_d, jd = agreement(cube_env(con, "dist"), rust_long(ids, dg, ed, NB["DISTANCES"]))
    print(f"\n=== {sport}: {json.dumps(r)}")
    print("agreement with cube, per-activity envelope, time basis (rel = rust/cube - 1):")
    print(at)
    print("agreement, distance basis:")
    print(ad)
    with pl.Config(tbl_rows=60):
        print("sport-level Best, time basis (pace m/s, power W):")
        print(best_t.filter(pl.col("metric").is_in(["v", "p", "x"])))
        print("sport-level Best, distance basis:")
        print(best_d.filter(pl.col("metric").is_in(["v"])))
    # worst disagreements, to explain
    worst = jt.filter(pl.col("metric") == "v").sort(pl.col("rel").abs(), descending=True).head(8)
    print("largest per-activity speed disagreements (time basis):")
    print(worst)

    if not deep:
        return
    # exhaustive (every start) vs strided, unfiltered
    (ye, _), r["time_exhaustive_s"] = clock(P.compute, "time", tg, 0.0)
    ee = envelope(ye)
    nonneg = np.array([m != "g" for m in ec.METRICS])  # gradient can be negative: no ratio bound
    ok = ~np.isnan(ee) & ~np.isnan(et) & (ee > 0) & nonneg[None, None, :]
    r["stride_error"] = {}
    for i, m in enumerate(ec.METRICS):
        o = ok[:, :, i]
        if o.any():
            rel = et[:, :, i][o] / ee[:, :, i][o] - 1
            r["stride_error"][m] = {"max": round(float(-rel.min()), 4), "p99": round(float(-np.quantile(rel, 0.01)), 4)}

    # single-activity updates: median and longest activity
    sizes = np.diff(d["offsets"])
    for name, a in (("median", int(np.argsort(sizes)[len(sizes) // 2])), ("longest", int(np.argmax(sizes)))):
        aid = int(ids[a])
        (one, t_load) = clock(load, sport, [aid])
        P1, t_prep = clock(ec.prepare, **one, sport=sport_params(sport), gap=gap_factors(sport), hr_bridge=HR_BRIDGE)
        _, t_time = clock(P1.compute, "time", tg, EPS)
        _, t_dist = clock(P1.compute, "dist", dg, EPS)
        r[f"update_{name}"] = {"seconds": int(sizes[a]), "load_s": t_load, "prepare_s": t_prep, "time_s": t_time, "dist_s": t_dist}

    # two-sided gradient band 4-8 %, pace by duration: cube query vs on-demand scan
    lens24 = np.array(NB["DURATIONS"], dtype=np.float64)
    q = """select len, max(v) as v from (select activity_id, len, max(v) as v from cube
           where basis = 'time' and gb between 4 and 7 group by all) group by len order by len"""
    con.sql(q).pl()  # warm
    cube_band, r["band_cube_query_s"] = clock(lambda: con.sql(q).pl())
    filt = [("g", 0.04, 0.08 - 1e-12)]
    (yb24, _), r["band_scan_24_exhaustive_s"] = clock(P.compute, "time", lens24, 0.0, filt)
    (_, _), r["band_scan_24_eps_s"] = clock(P.compute, "time", lens24, EPS, filt)
    (ybf, _), r["band_scan_full_exhaustive_s"] = clock(P.compute, "time", tg, 0.0, filt)
    (_, _), r["band_scan_full_eps_s"] = clock(P.compute, "time", tg, EPS, filt)
    band = cube_band.join(pl.DataFrame({"len": lens24.astype(np.int64), "rust": np.nanmax(yb24[:, :, MI["v"]], axis=0).astype(float)}), on="len")
    print("two-sided band 4-8 %, best speed by duration (no envelope), cube vs exhaustive scan:")
    with pl.Config(tbl_rows=30):
        print(band.with_columns(rel=pl.col("rust") / pl.col("v") - 1))
    print(f"\n=== {sport} deep: " + json.dumps({k: v for k, v in r.items() if k.startswith(("time_ex", "stride", "update", "band"))}, indent=1))


if __name__ == "__main__":
    sports = sys.argv[1:] or ["Run", "Ride", "Hike", "Skimo", "Skate ski"]
    report = {}
    for s in sports:
        bench_sport(s, report, deep=(s in ("Run", "Hike")))
    (CUR.parent / f"report-hr{HR_BRIDGE}.json").write_text(json.dumps(report, indent=1))
    print("\nSUMMARY")
    print(pl.DataFrame([{"sport": s, **{k: v for k, v in r.items() if not isinstance(v, dict)}} for s, r in report.items()]))
