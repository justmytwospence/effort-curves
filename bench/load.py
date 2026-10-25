"""Load one sport's raw samples from the host's Garmin snapshot into effort_curves.prepare inputs.

Sport definitions, caps and the grade-adjustment model come from the Data cubes notebook, so both
pipelines see the same activities and limits.
"""

import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from current_cube import SNAPSHOT, OUT as CURRENT_OUT, load_cube_defs

NB = load_cube_defs()
GAP_MODEL = CURRENT_OUT / "cube" / "gap_model.parquet"


def sport_params(sport):
    return (
        float(NB["SPEED_CAP"][sport]),
        float(NB["GRADE_CAP"][sport]),
        float(NB["ASCENT_CAP"]),
        float(NB["ASCENT_MIN_LEN"]),
        float(NB["GAP_MIN_TIME"]),
        float(NB["GAP_MIN_DIST"]),
    )


def gap_factors(sport):
    if sport not in NB["GAP_SPORTS"]:
        return None
    m = pl.read_parquet(GAP_MODEL).sort("gp")
    assert m["gp"].to_list() == list(range(-35, 36))
    return m["r"].to_numpy().astype(np.float64)


def load(sport, activity_ids=None):
    """Raw samples for a sport (or a subset of its activities), sorted by (activity, t)."""
    con = duckdb.connect()
    con.execute("SET threads = 6")
    con.execute(f"ATTACH '{SNAPSHOT}' AS garmin (READ_ONLY)")
    only = f"and a.activity_id in ({', '.join(map(str, activity_ids))})" if activity_ids else ""
    indoor = NB["INDOOR"]
    df = con.sql(f"""
        with raw as (
            select r.activity_id, epoch(r.timestamp)::bigint as ts, r.distance, r.altitude, r.power, r.heart_rate,
                   a.activity_type in {indoor} as indoor
            from garmin.fit_record r join garmin.activity a using (activity_id)
            where {NB['sport_sql']('a')} = '{sport}' and a.deleted_at is null and r.timestamp is not null {only}
        )
        select activity_id, (ts - min(ts) over (partition by activity_id))::bigint as t,
               distance, altitude, power::double as power, heart_rate::double as hr, indoor
        from raw
        qualify row_number() over (partition by activity_id, ts order by distance desc nulls last) = 1
        order by activity_id, t
    """).pl()
    df = df.with_columns(pl.col("distance", "altitude", "power", "hr").cast(pl.Float64).fill_null(np.nan))
    acts = df.group_by("activity_id", maintain_order=True).agg(
        n=pl.len(), indoor=pl.col("indoor").first(), powered=pl.col("power").is_not_nan().any()
    )
    offsets = np.concatenate([[0], np.cumsum(acts["n"].to_numpy())]).astype(np.int64)
    return dict(
        activity_ids=acts["activity_id"].to_numpy().astype(np.int64),
        offsets=offsets,
        t=df["t"].to_numpy().astype(np.int64),
        dist=df["distance"].to_numpy(),
        alt=df["altitude"].to_numpy(),
        power=df["power"].to_numpy(),
        hr=df["hr"].to_numpy(),
        indoor=acts["indoor"].to_numpy().astype(np.uint8),
        powered=acts["powered"].to_numpy().astype(np.uint8),
    )


def time_grid(max_s=60 * 60 * 15, ratio=1.01):
    """Every second to 100 s (1 s is >= 1 % there), then 1 % steps; plus the notebook's durations."""
    g = set(range(1, 101))
    x = 100.0
    while x < max_s:
        x *= ratio
        g.add(int(round(x)))
    g.update(NB["DURATIONS"])
    return np.array(sorted(g), dtype=np.float64)


def dist_grid(lo=50.0, hi=200_000.0, ratio=1.01):
    g = set()
    x = lo
    while x < hi:
        g.add(round(x, 1))
        x *= ratio
    g.update(float(d) for d in NB["DISTANCES"])
    return np.array(sorted(g), dtype=np.float64)


def envelope(y):
    """Best at this length or any longer one, per activity and metric (axis 1 is length, ascending)."""
    return np.fmax.accumulate(y[:, ::-1, :], axis=1)[:, ::-1, :]
