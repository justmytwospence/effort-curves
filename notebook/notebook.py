# /// script
# requires-python = ">=3.10,<3.15"
#
# [tool.marimo-studio]
# default = "dashboard"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", auto_download=["html"])

with app.setup:
    import marimo as mo
    import os
    import time
    import datetime as dt
    from pathlib import Path
    import duckdb
    import numpy as np
    import polars as pl
    import altair as alt
    import shutil
    import json


@app.cell(hide_code=True)
def intro():
    mo.md(r"""
    # Data cubes

    Best efforts and their spread, by any dimension: pick what to maximise and what to plot it against in the sidebar, then click a point on the Best line to see where it happened.
    """)
    return


@app.cell(hide_code=True)
def connect():
    con = duckdb.connect()
    con.execute("SET threads = 6")
    con.execute("SET memory_limit = '16GB'")
    _token = os.environ["GARMIN_SNAPSHOT_TOKEN"]
    con.execute(
        "CREATE OR REPLACE SECRET garmin_snapshot (TYPE http, SCOPE 'http://garmin-mcp:8000', "
        f"EXTRA_HTTP_HEADERS MAP {{'Authorization': 'Bearer {_token}'}})"
    )
    con.execute("ATTACH IF NOT EXISTS 'http://garmin-mcp:8000/snapshot.duckdb' AS garmin (READ_ONLY)")
    con.execute("SET TimeZone = 'America/Denver'")
    # Strava's mirror, only to link an activity to its Strava page (matched by start time); optional
    _strava_token = os.environ.get("STRAVA_SNAPSHOT_TOKEN")
    HAS_STRAVA = False
    if _strava_token:
        try:
            con.execute(
                "CREATE OR REPLACE SECRET strava_snapshot (TYPE http, SCOPE 'http://strava-mcp:8000', "
                f"EXTRA_HTTP_HEADERS MAP {{'Authorization': 'Bearer {_strava_token}'}})"
            )
            con.execute("ATTACH IF NOT EXISTS 'http://strava-mcp:8000/snapshot.duckdb' AS strava (READ_ONLY)")
            HAS_STRAVA = True
        except Exception:
            pass


    def q(sql):
        return con.sql(sql).pl()

    return HAS_STRAVA, con, q


@app.cell(hide_code=True)
def about():
    ABOUT_MD = mo.md(r"""
    Every recorded second of every activity (Garmin's FIT records) is put on a 1 Hz clock as running totals of distance, work, heartbeats, climbing and grade-adjusted distance, so any stretch of an activity is the difference of two rows. The curves are computed by [effort-curves](https://github.com/justmytwospence/effort-curves), a small Rust engine.

    **Gaps are bridged, not cut.** Pauses and smart-recording gaps stay inside the activity and their time counts against any effort that spans them. Distance and altitude are interpolated across a gap (a real pause records no distance, so pace there is zero), power is zero, and heart rate is interpolated between the readings either side (up to 10 minutes). A second faster than the sport's speed cap (a GPS jump or a vehicle) counts as no movement, and a barometer jump as no climbing.

    **Stored curves.** For each activity the best power, speed, gradient, heart rate, vertical speed and grade-adjusted pace are kept for every duration from 1 s to 100 s and in 1 % steps beyond (to 15 h), and for every distance from 100 m in 1 % steps (to 200 km). A best held for longer counts at every shorter length too, so each curve only falls as the windows lengthen, and between two 1 % steps it cannot move by more than 1 %. New activities are computed on their own (well under a second each) and added; the stored curves feed the unfiltered duration and distance views directly.

    **Filters, bands and value axes are exact.** Any narrowed slider, a band split or a value on the x axis (gradient, pace, power, heart rate) runs a scan of every window of the selected activities, with each window's own average tested against the exact bounds. A slider left at its end means no limit on that side. Value axes are drawn in bins (1 % gradient, 0.2 m/s, 20 W, 5 bpm).

    The numbers check out: per-activity best power matches the previous cube (which itself matched Garmin's mean-max power to a median of 0.15 %) to within 1 % for 97 % of activities and lengths; the differences come from smart-recording activities the old cube could not measure and GPS glitches it let through.

    **Cadence** is pedal revolutions per minute on a bike and steps per minute on foot (Garmin records one foot, so steps are twice its number). The curves count it two ways. On the y axis it is the whole window's cadence, with coasting and stops as zero, so the best cadence held for at least a duration behaves like any other curve. As an x axis, filter or band it is the cadence while pedaling or stepping, so a descent with a minute of coasting still lands at the cadence you actually turned; such a window needs at least a quarter of its time pedaling, and the **pedaling share** slider (stepping on foot) narrows to windows that were mostly pedaled, which keeps coasting descents out of power by gradient. **Torque** is the work per pedal revolution (newton-metres at the crank), and **step length** the distance per step. Rides without a cadence sensor have no cadence views.

    **Grade-adjusted pace** comes from your own strap-recorded runs. In steady stretches the notebook compares the speed you hold on each 1 % of gradient with your flat speed at the same heart rate, giving a factor per gradient: about 0.77 at +5 %, 0.58 at +10 %, 0.36 at +20 %; on descents you speed up to about 1.07 at -8 %, are back to flat speed at -12 %, and slow down again beyond it (0.70 at -20 %, 0.47 at -30 %), so steep descents are credited for how much they slow you. Each second's distance is divided by the factor for its gradient, so a window's grade-adjusted pace is the flat pace of the same effort. Windows shorter than 3 minutes or 800 m get no grade-adjusted pace, since short altitude glitches swing it too much, and the first minute of each run is left unadjusted while the barometer settles. The model is refitted whenever the run curves rebuild.
    """)
    return (ABOUT_MD,)


@app.cell(hide_code=True)
def cube_defs():
    os.environ.setdefault("RAYON_NUM_THREADS", "6")  # the sandbox's CPU quota; read when the engine first runs
    import effort_curves as ec

    SPORT_TYPES = {
        "Run": ["running", "treadmill_running", "track_running", "trail_running"],
        "Ride": ["cycling", "road_biking", "gravel_cycling", "mountain_biking", "indoor_cycling", "virtual_ride"],
        "Hike": ["hiking", "walking", "mountaineering", "snow_shoe_ws"],
        "Skimo": ["backcountry_skiing"],
        "Skate ski": ["skate_skiing_ws"],
        # left out: classic cross-country (one activity) and resort skiing (lift-assisted descents would swamp every speed best)
    }
    # named lengths: the "Over" choices for value axes, and always on the curve grids so headline numbers land exactly
    DURATIONS = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600, 900, 1200, 1800, 2700, 3600, 5400, 7200, 10800, 14400, 18000]
    DISTANCES = [100, 200, 400, 800, 1000, 1609, 2000, 3000, 5000, 8047, 10000, 15000, 16093, 21097, 30000, 42195, 50000, 80467, 100000]
    # bin widths of the value axes and slider steps: time-into 5 min, distance-into 0.5 mi, gradient 1 %, speed 0.2 m/s, power 20 W, heart rate 5 bpm
    BINS = {"tb": 300, "db": 804.672, "gb": 0.01, "vb": 0.2, "pb": 20, "hb": 5, "cb": 5, "sb": 0.05}
    CUBE_DIR = "cube"
    CUBE_SCHEMA = 11  # bump when the curves' columns or the engine's rules change: every sport rebuilds once


    def cube_files(sport):
        return f"'{CUBE_DIR}/{sport}/*.parquet'"


    # fastest believable speed (m/s): a faster second is a GPS jump (or a vehicle) and counts as no movement
    SPEED_CAP = {"Run": 7.5, "Hike": 5.0, "Ride": 30.0, "Skimo": 35.0, "Skate ski": 20.0}
    # steepest believable net gradient of a window; steeper means bad altitude
    GRADE_CAP = {"Run": 0.45, "Hike": 0.8, "Ride": 0.25, "Skimo": 0.8, "Skate ski": 0.4}
    # fastest believable climb (m/s of ascent, ~5,900 ft/h); a faster second is a barometer jump and counts as no climbing
    ASCENT_CAP = 0.5
    # ascent rate only means something over a minute or more: shorter windows are barometer jitter
    ASCENT_MIN_LEN = 60
    # no real terrain indoors: their altitude is missing or simulated
    INDOOR = ("indoor_cycling", "virtual_ride", "treadmill_running")
    # cadence: pedal revolutions per minute on a bike, steps per minute on foot (FIT counts one foot, so steps = 2 x cadence)
    CADENCE_SCALE = {"Run": 2, "Hike": 2, "Ride": 1, "Skimo": 1, "Skate ski": 1}
    CADENCE_UNIT = {"Run": "spm", "Hike": "spm", "Ride": "rpm", "Skimo": "/min", "Skate ski": "/min"}
    # highest believable cadence: a faster second is a sensor glitch and counts as no pedaling or stepping
    CADENCE_CAP = {"Run": 260, "Hike": 220, "Ride": 200, "Skimo": 200, "Skate ski": 200}
    # what the stored curves keep per activity (the engine's pedaling cadence and share serve filters and axes only)
    STORED_METRICS = ["p", "v", "g", "h", "a", "x", "c", "k", "l"]

    # grade-adjusted pace, from your own runs: at the same heart rate, how fast you move on each gradient relative to flat
    GAP_SPORTS = {"Run"}
    GAP_MODEL_PATH = f"{CUBE_DIR}/gap_model.parquet"
    # shorter windows amplify altitude glitches (steep grades multiply speed up to 4x): no GAP below these
    GAP_MIN_TIME, GAP_MIN_DIST = 180, 800


    def _grid(lo, hi, ratio, every_to, named):
        """Every unit up to `every_to`, then steps of `ratio`, plus the named lengths."""
        _g = set(range(lo, every_to + 1))
        _x = float(every_to)
        while _x < hi:
            _x *= ratio
            _g.add(int(round(_x)))
        _g.update(named)
        return np.array(sorted(_g), dtype=np.float64)


    # what is stored per activity: every second to 100 s, then 1 % steps (the curve can't move more than 1 % between them)
    TIME_GRID = _grid(1, 15 * 3600, 1.01, 100, DURATIONS)
    DIST_GRID = _grid(100, 200_000, 1.01, 100, DISTANCES)
    def _thin(grid, lo, ratio, named):
        """Lengths of a stored grid at least `ratio` apart (from `lo` up), plus the named ones: every one of them is a stored length."""
        _keep, _last = [], None
        for _v in grid:
            if _v >= lo and (_last is None or _v >= _last * ratio or _v in named):
                _keep.append(_v)
                _last = _v
        return np.array(_keep, dtype=np.float64)


    # what a filtered scan and the chart use: about 4 % steps, plenty for a line, taken from the stored grids
    SCAN_TIME = _thin(TIME_GRID, 5, 1.04, set(DURATIONS))
    SCAN_DIST = _thin(DIST_GRID, 100, 1.04, set(DISTANCES))


    def fit_gap_model(bc):
        """Speed on each 1 % gradient relative to flat at equal heart rate, from strap-recorded runs.

        60 s centred windows give speed and gradient; heart rate is the average of the next 40 s (it lags effort) and
        only steady stretches count (heart-rate spread under 6 bpm). Per heart-rate band the median speed on each
        gradient is divided by the flat one; the bands are averaged by sample count, then lightly smoothed."""
        bc.execute("""
            create or replace temp table gap_s as
            with strap as (
                select activity_id from garmin.fit_device_info group by 1
                having bool_or(antplus_device_type = 'heart_rate' or (source_type = 'bluetooth_low_energy' and device_type = '1')
                               or garmin_product ilike 'hrm%')
            ),
            r as (
                select r.activity_id, epoch(r.timestamp)::bigint as ts, r.distance as d, r.altitude as alt, r.heart_rate as hr
                from garmin.fit_record r join garmin.activity a using (activity_id) join strap using (activity_id)
                where a.activity_type in ('running', 'trail_running') and a.deleted_at is null and r.distance is not null and r.altitude is not null
                qualify row_number() over (partition by r.activity_id, epoch(r.timestamp)::bigint order by r.distance desc) = 1
            ),
            g as (
                select activity_id, ts,
                       lead(d, 30) over w - lag(d, 30) over w as dd, lead(ts, 30) over w - lag(ts, 30) over w as dt,
                       avg(alt) over (partition by activity_id order by ts rows between 10 preceding and 10 following) as ca,
                       avg(hr) over (partition by activity_id order by ts rows between current row and 40 following) as hr_lag,
                       stddev(hr) over (partition by activity_id order by ts rows between 30 preceding and 30 following) as hr_sd
                from r window w as (partition by activity_id order by ts)
            )
            select dd / dt as v, (lead(ca, 30) over w - lag(ca, 30) over w) / dd as grade, hr_lag, hr_sd, dd
            from g where dt = 60 window w as (partition by activity_id order by ts)
        """)
        _df = bc.sql("""
            select round(grade * 100)::int as gp, (floor(hr_lag / 5) * 5)::int as hrb, median(v) as v, count(*) as n
            from gap_s where dd > 30 and grade between -0.35 and 0.35 and hr_sd < 6 and hr_lag between 120 and 185 and v between 0.6 and 7
            group by all
        """).pl()
        _flat = _df.filter(pl.col("gp") == 0).select("hrb", pl.col("v").alias("v0"), pl.col("n").alias("n0"))
        _rel = _df.join(_flat, on="hrb").filter((pl.col("n") >= 40) & (pl.col("n0") >= 200)).with_columns(r=pl.col("v") / pl.col("v0"))
        _m = (
            _rel.group_by("gp").agg(r=(pl.col("r") * pl.col("n")).sum() / pl.col("n").sum(), n=pl.col("n").sum())
            .sort("gp")
            .with_columns(r=pl.col("r").rolling_mean(3, center=True, min_samples=1))
            .with_columns(r=pl.when(pl.col("gp") == 0).then(1.0).otherwise(pl.col("r")))
        )
        return pl.DataFrame({"gp": list(range(-35, 36))}).join(_m, on="gp", how="left").with_columns(
            pl.col("r").fill_null(strategy="forward").fill_null(strategy="backward"), pl.col("n").fill_null(0)
        )


    def sport_sql(alias):
        """Broad sport of an activity: by its Garmin type, except that a name saying which kind of skiing wins (two backcountry days were saved as a trail run and a hike)."""
        _case = " ".join(f"when {alias}.activity_type in ({', '.join(repr(t) for t in types)}) then '{s}'" for s, types in SPORT_TYPES.items())
        return f"(case when {alias}.activity_name ilike '%backcountry skiing%' then 'Skimo' when lower({alias}.activity_name) like '%skate ski%' then 'Skate ski' {_case} end)"


    def garmin_connection():
        """A separate DuckDB connection for loading, so a build never competes with the explorer's queries."""
        bc = duckdb.connect()
        bc.execute("SET threads = 6")
        _token = os.environ["GARMIN_SNAPSHOT_TOKEN"]
        bc.execute(
            "CREATE OR REPLACE SECRET garmin_snapshot (TYPE http, SCOPE 'http://garmin-mcp:8000', "
            f"EXTRA_HTTP_HEADERS MAP {{'Authorization': 'Bearer {_token}'}})"
        )
        bc.execute("ATTACH 'http://garmin-mcp:8000/snapshot.duckdb' AS garmin (READ_ONLY)")
        return bc


    def load_samples(bc, sport, ids=None):
        """Every recorded second of a sport's activities (or just `ids`), as the engine's input arrays."""
        _only = f"and a.activity_id in ({', '.join(map(str, ids))})" if ids else ""
        _df = bc.sql(f"""
            with raw as (
                select r.activity_id, epoch(r.timestamp)::bigint as ts, r.distance, r.altitude, r.power, r.heart_rate,
                       (r.cadence + coalesce(r.fractional_cadence, 0)) * {CADENCE_SCALE[sport]} as cadence,
                       a.activity_type in {INDOOR} as indoor
                from garmin.fit_record r join garmin.activity a using (activity_id)
                where {sport_sql('a')} = '{sport}' and a.deleted_at is null and r.timestamp is not null {_only}
            )
            select activity_id, (ts - min(ts) over (partition by activity_id))::bigint as t,
                   distance::double as distance, altitude::double as altitude, power::double as power, heart_rate::double as hr,
                   cadence::double as cadence, indoor
            from raw
            qualify row_number() over (partition by activity_id, ts order by distance desc nulls last) = 1
            order by activity_id, t
        """).pl()
        _df = _df.with_columns(pl.col("distance", "altitude", "power", "hr", "cadence").fill_null(np.nan))
        _acts = _df.group_by("activity_id", maintain_order=True).agg(
            n=pl.len(), indoor=pl.col("indoor").first(), powered=pl.col("power").is_not_nan().any()
        )
        return dict(
            activity_ids=_acts["activity_id"].to_numpy().astype(np.int64),
            offsets=np.concatenate([[0], np.cumsum(_acts["n"].to_numpy())]).astype(np.int64),
            t=_df["t"].to_numpy().astype(np.int64),
            dist=_df["distance"].to_numpy(),
            alt=_df["altitude"].to_numpy(),
            power=_df["power"].to_numpy(),
            hr=_df["hr"].to_numpy(),
            cadence=_df["cadence"].to_numpy(),
            indoor=_acts["indoor"].to_numpy().astype(np.uint8),
            powered=_acts["powered"].to_numpy().astype(np.uint8),
        )


    def prepare(bc, sport, ids=None):
        """The engine's 1 Hz running totals for a sport's activities (gaps bridged, glitches dropped per second)."""
        _d = load_samples(bc, sport, ids)
        _gap = None
        if sport in GAP_SPORTS and Path(GAP_MODEL_PATH).exists():
            _m = pl.read_parquet(GAP_MODEL_PATH).sort("gp")
            _gap = _m["r"].to_numpy().astype(np.float64)
        _params = (float(SPEED_CAP[sport]), float(GRADE_CAP[sport]), float(ASCENT_CAP), float(ASCENT_MIN_LEN), float(GAP_MIN_TIME), float(GAP_MIN_DIST), float(CADENCE_CAP[sport]))
        return ec.prepare(**_d, sport=_params, gap=_gap), _d


    def curves_frame(P):
        """Each activity's best of every metric at every stored length, by time and by distance: one row per activity and length."""
        _out = []
        for _basis, _grid in (("time", TIME_GRID), ("dist", DIST_GRID)):
            _y, _t = P.compute(_basis, _grid, 0.01)
            _a, _k, _m = _y.shape
            _cols = {"activity_id": np.repeat(P.activity_ids, _k), "basis": [_basis] * (_a * _k), "len": np.tile(_grid.astype(np.int32), _a)}
            for _name in STORED_METRICS:
                _i = ec.METRICS.index(_name)
                _cols[_name] = _y[:, :, _i].reshape(-1).astype(np.float64)
                _cols[f"t_{_name}"] = _t[:, :, _i].reshape(-1).astype(np.int64)
            _f = pl.DataFrame(_cols).with_columns(
                [pl.col(m).fill_nan(None) for m in STORED_METRICS]
                + [pl.when(pl.col(f"t_{m}") >= 0).then(pl.col(f"t_{m}")).alias(f"t_{m}") for m in STORED_METRICS]
            )
            _out.append(_f.filter(pl.any_horizontal([pl.col(m).is_not_null() for m in STORED_METRICS])))
        return pl.concat(_out)


    def slider_stats(d):
        """Slider ranges for a sport, in the units of the bins (BINS): from activity spans, 60 s stretches and every heart-rate and power sample."""
        _off = d["offsets"]
        _t, _dist, _alt = d["t"], d["dist"], d["alt"]
        _last = _off[1:] - 1
        _dur = _t[_last].astype(float)
        _far = np.array([np.nanmax(_dist[a:b]) if np.isfinite(_dist[a:b]).any() else 0.0 for a, b in zip(_off[:-1], _off[1:])])
        # 60 s stretches, every 10th sample: speed and gradient
        _act = np.repeat(np.arange(len(_off) - 1), np.diff(_off))
        _key = _act.astype(np.int64) * 10_000_000 + _t
        _i = np.arange(0, len(_t), 10)
        _j = np.minimum(np.searchsorted(_key, _key[_i] + 60), len(_t) - 1)
        _ok = (_act[_j] == _act[_i]) & (_t[_j] - _t[_i] >= 50)
        _i, _j = _i[_ok], _j[_ok]
        _dd = _dist[_j] - _dist[_i]
        with np.errstate(divide="ignore", invalid="ignore"):
            _v = _dd / (_t[_j] - _t[_i])
            _g = np.where(_dd > 20, (_alt[_j] - _alt[_i]) / _dd, np.nan)

        def _q(x, p):
            x = x[np.isfinite(x)]
            return float(np.quantile(x, p)) if x.size else None

        def _b(x, w):
            return None if x is None else int(np.floor(x / w))

        _p, _h = d["power"], d["hr"]
        _c = d["cadence"]
        _c = _c[np.isfinite(_c) & (_c > 0)]
        _h = np.where((_h >= 30) & (_h <= 230), _h, np.nan)
        return {
            "has_power": bool(np.isfinite(_p).any()),
            "tb": _b(_q(_dur, 0.995), BINS["tb"]), "db": _b(_q(_far, 0.995), BINS["db"]),
            "glo": _b(_q(_g, 0.005), BINS["gb"]), "ghi": _b(_q(_g, 0.995), BINS["gb"]),
            "vlo": _b(_q(_v[_v > 0.3], 0.002), BINS["vb"]), "vhi": _b(_q(_v, 0.998), BINS["vb"]),
            "pb": _b(_q(_p[_p > 0], 0.998), BINS["pb"]),
            "hlo": _b(_q(_h, 0.002), BINS["hb"]), "hhi": _b(_q(_h, 0.998), BINS["hb"]),
            "has_cadence": bool(_c.size), "clo": _b(_q(_c, 0.005), BINS["cb"]), "chi": _b(_q(_c, 0.995), BINS["cb"]),
        }


    def build_curves(sport, path, ids=None):
        """Compute and save a sport's curves: all activities into a fresh directory, or only `ids`, added as one more file.

        Returns the slider ranges when all activities were built (None for an addition)."""
        bc = garmin_connection()
        if ids is None and sport in GAP_SPORTS:
            fit_gap_model(bc).write_parquet(GAP_MODEL_PATH)
        P, _d = prepare(bc, sport, ids)
        bc.close()
        _f = curves_frame(P)
        _path = Path(path)
        if ids is None:
            shutil.rmtree(_path, ignore_errors=True)
        _path.mkdir(parents=True, exist_ok=True)
        # files stay under the Hub's 64 MiB per-file workspace cap so the curves survive the session
        _stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        for _n, _start in enumerate(range(0, max(_f.height, 1), 400_000)):
            _f.slice(_start, 400_000).write_parquet(_path / f"curves-{_stamp}-{_n}.parquet", compression="zstd")
        return slider_stats(_d) if ids is None else None

    # Display table: what every chart and animation reads. Each activity's curve with "held for at least" already
    # applied, at the chart's lengths (about 8 % apart), with the activity's date, so any frame (a window, a year) is
    # one grouping and no query recomputes the envelope.
    DISPLAY_TIME = _thin(SCAN_TIME, 5, 1.075, set(DURATIONS))
    DISPLAY_DIST = _thin(SCAN_DIST, 100, 1.075, set(DISTANCES))
    DISP_SCHEMA = 1
    DISP_DIR = Path(CUBE_DIR, "_display")


    def _activity_days(sport):
        """Each activity's local date as a day number (days since 1970-01-01)."""
        bc = garmin_connection()
        _d = bc.sql(f"""
            select activity_id, (start_time_local::date - date '1970-01-01')::int as d
            from garmin.activity a where {sport_sql('a')} = '{sport}' and deleted_at is null
        """).pl()
        bc.close()
        return _d


    def build_display(sport):
        """Write the sport's display table: activity, basis, length, day, and per stored metric its envelope value (y_),
        where the effort that set it starts (t_) and how long that effort is (s_)."""
        DISP_DIR.mkdir(parents=True, exist_ok=True)
        con_d = duckdb.connect()
        con_d.execute("SET threads = 6")
        con_d.register("dates", _activity_days(sport).to_arrow())
        _env = ", ".join(
            f"max({m}) over w as y_{m}, arg_max(t_{m}, {m}) over w as t_{m}, arg_max(len, {m}) over w as s_{m}" for m in STORED_METRICS
        )
        _parts = []
        for _basis, _grid in (("time", DISPLAY_TIME), ("dist", DISPLAY_DIST)):
            _show = ", ".join(str(int(v)) for v in _grid)
            _parts.append(con_d.sql(f"""
                with env as (
                    select activity_id, len::int as xb, {_env} from {cube_files(sport)} where basis = '{_basis}'
                    window w as (partition by activity_id order by len desc rows between unbounded preceding and current row)
                )
                select e.activity_id, '{_basis}' as basis, e.xb, dates.d, e.* exclude (activity_id, xb)
                from env e join dates using (activity_id) where e.xb in ({_show})
            """).pl())
        _tab = pl.concat(_parts).with_columns(
            [pl.col(f"y_{m}").cast(pl.Float32) for m in STORED_METRICS]
            + [pl.col(f"t_{m}").cast(pl.Int32) for m in STORED_METRICS] + [pl.col(f"s_{m}").cast(pl.Int32) for m in STORED_METRICS]
        )
        _tab.write_parquet(DISP_DIR / f"{sport}.parquet", compression="zstd")
        (DISP_DIR / f"{sport}.json").write_text(json.dumps({"schema": DISP_SCHEMA}))
        return _tab


    def display_file(sport):
        """The sport's display table on disk, built if missing or from an older schema."""
        _meta = DISP_DIR / f"{sport}.json"
        if not (_meta.exists() and (DISP_DIR / f"{sport}.parquet").exists() and json.loads(_meta.read_text()).get("schema") == DISP_SCHEMA):
            build_display(sport)
        return DISP_DIR / f"{sport}.parquet"


    # Window positions: the precomputed frames of the most common animation (the default 365-day window slid a week at
    # a time, no lines, no filters), read straight from disk. Every other animation is computed when chosen.
    POS_DAYS, POS_STEP = 365, 7
    POS_PCTS = [5, 10, 25, 50, 75, 90, 95]
    POS_SCHEMA = 2
    POS_DIR = Path(CUBE_DIR, "_positions")


    def build_positions(sport, excluded):
        """Best (with its effort) and the fixed percentiles per window position, length, metric and basis, for one sport."""
        POS_DIR.mkdir(parents=True, exist_ok=True)
        con_p = duckdb.connect()
        con_p.execute("SET threads = 6")
        con_p.execute(f"create table disp as select * from '{display_file(sport)}'")
        # positions end on the last day with a recorded activity (the date window's default end), a week apart
        _hi, _lo = con_p.sql("select max(d), min(d) from disp").fetchall()[0]
        _n = (_hi - _lo) // POS_STEP + 1
        _qs = ", ".join(str(p / 100) for p in POS_PCTS)
        _excl = f"and activity_id not in ({', '.join(map(str, excluded))})" if excluded else ""
        _out = []
        for _basis in ("time", "dist"):
            for _m in STORED_METRICS:
                _f = con_p.sql(f"""
                    with cur as (select activity_id, xb, d, y_{_m} as y, t_{_m} as t, s_{_m} as src from disp
                                 where basis = '{_basis}' and y_{_m} is not null {_excl}),
                    pos as (select {_hi} - {POS_STEP} * k as e from range(0, {_n}) r(k))
                    select '{_m}' as metric, '{_basis}' as basis, p.e, c.xb, count(*)::int as n,
                           max(c.y) as best, arg_max(c.activity_id, c.y) as aid, arg_max(c.t, c.y)::int as t, arg_max(c.src, c.y)::int as src,
                           quantile_cont(c.y, [{_qs}]) as q
                    from cur c join pos p on c.d > p.e - {POS_DAYS} and c.d <= p.e
                    group by all
                """).pl()
                if _f.height:
                    _out.append(_f.with_columns([pl.col("q").list.get(i).alias(f"q{p}") for i, p in enumerate(POS_PCTS)]).drop("q"))
        _tab = pl.concat(_out) if _out else pl.DataFrame()
        _tab.write_parquet(POS_DIR / f"{sport}.parquet", compression="zstd")
        (POS_DIR / f"{sport}.json").write_text(json.dumps({"schema": POS_SCHEMA, "excluded": sorted(excluded), "hi": _hi}))
        return _tab


    def load_positions(sport, excluded):
        """The sport's position table, rebuilt if missing, from an older schema, or made with another exclude list."""
        _meta = POS_DIR / f"{sport}.json"
        _ok = _meta.exists() and (POS_DIR / f"{sport}.parquet").exists()
        if _ok:
            _m = json.loads(_meta.read_text())
            _ok = _m.get("schema") == POS_SCHEMA and _m.get("excluded") == sorted(excluded)
        return pl.read_parquet(POS_DIR / f"{sport}.parquet") if _ok else build_positions(sport, excluded)


    return (
        BINS,
        CADENCE_CAP,
        CADENCE_SCALE,
        CADENCE_UNIT,
        CUBE_DIR,
        CUBE_SCHEMA,
        DISPLAY_DIST,
        DISPLAY_TIME,
        DISTANCES,
        DURATIONS,
        GAP_SPORTS,
        POS_DAYS,
        POS_PCTS,
        POS_STEP,
        SPEED_CAP,
        SPORT_TYPES,
        build_curves,
        build_display,
        build_positions,
        cube_files,
        display_file,
        ec,
        garmin_connection,
        load_positions,
        prepare,
        sport_sql,
    )


@app.cell(hide_code=True)
def rebuild():
    rebuild = mo.ui.run_button(label="Rebuild all cubes")
    return (rebuild,)


@app.cell(hide_code=True)
def cube_status(
    CUBE_DIR,
    CUBE_SCHEMA,
    SPORT_TYPES,
    build_curves,
    build_display,
    build_positions,
    con,
    cube_files,
    q,
    rebuild,
    sport_sql,
):
    _ = rebuild.value
    _dir = Path(CUBE_DIR)
    _dir.mkdir(exist_ok=True)
    # which activities have second-by-second data: fit_session has one small row per activity FIT file,
    # so this check costs milliseconds instead of a scan of every record
    _candidates = q(f"""
        select {sport_sql('a')} as sport, list(distinct s.activity_id order by s.activity_id) as ids
        from garmin.fit_session s join garmin.activity a using (activity_id)
        where a.deleted_at is null group by 1
    """)
    _ids_by_sport = dict(zip(_candidates["sport"], _candidates["ids"].to_list()))
    _rows = []
    for _sport in SPORT_TYPES:
        _path = _dir / _sport
        _manifest = _path / "manifest.json"
        _ids = sorted(_ids_by_sport.get(_sport, []))
        _known = json.loads(_manifest.read_text()) if _manifest.exists() and any(_path.glob("*.parquet")) else {}
        _have = set(_known.get("ids", []))
        _new = [a for a in _ids if a not in _have]
        _t = time.time()
        _status = "up to date"
        # a schema change, a deleted activity or the button rebuilds everything; new activities are computed on their own and added
        if rebuild.value or _known.get("schema") != CUBE_SCHEMA or not _have.issubset(_ids):
            with mo.status.spinner(title=f"Building {_sport} curves from {len(_ids)} activities"):
                _stats = build_curves(_sport, str(_path))
            _known = {"ids": _ids, "schema": CUBE_SCHEMA, "stats": _stats}
            _status = f"rebuilt in {time.time() - _t:.0f} s"
        elif _new:
            with mo.status.spinner(title=f"Adding {len(_new)} new {_sport} activities"):
                build_curves(_sport, str(_path), ids=_new)
            _known["ids"] = _ids
            _status = f"added {len(_new)} in {time.time() - _t:.1f} s"
        if _status != "up to date":
            _counts = q(f"select count(distinct activity_id) as a, count(*) as n from {cube_files(_sport)}").row(0)
            _known.update(activities=_counts[0], rows=_counts[1], built=dt.datetime.now().isoformat(timespec="minutes"))
            _manifest.write_text(json.dumps(_known))
            # the live chart's window positions follow the curves (and the exclude list in force)
            _excl_file = _dir / "excluded.json"
            build_display(_sport)
            build_positions(_sport, json.loads(_excl_file.read_text()) if _excl_file.exists() else [])
        _rows.append(
            {
                "sport": _sport,
                "activities": _known["activities"],
                "curve points": _known["rows"],
                "size MB": round(sum(f.stat().st_size for f in _path.glob("*.parquet")) / 1e6, 1),
                "built": _known["built"],
                "status": _status,
            }
        )
    cube_status = pl.DataFrame(_rows)
    CUBE_MANIFESTS = {s: json.loads((_dir / s / "manifest.json").read_text()) for s in SPORT_TYPES}
    acts = q("""
        select activity_id, activity_name as name, start_time_local::date as date, activity_type
        from garmin.activity where deleted_at is null
    """)

    # sports already loaded into memory this session, and their raw running totals for filtered scans (cleared whenever the curves change)
    SPORT_CACHE = {}
    SCANNERS = {}
    SCAN_LOCK = __import__("threading").Lock()  # one load per sport, whoever asks first
    con.register("acts", acts)
    return CUBE_MANIFESTS, SCANNERS, SCAN_LOCK, SPORT_CACHE, acts, cube_status


@app.cell(hide_code=True)
def hr_quality(CUBE_DIR, con, cube_status, q):
    # Heart-rate data quality, one row per activity, cached next to the cubes and extended as activities arrive.
    # Two levels for the "Heart-rate data" control:
    #   strap      a chest strap or armband was connected (an ANT+ or Bluetooth heart-rate sensor in the FIT file)
    #   plausible  nothing obviously wrong, judged against your own strap-recorded activities
    _ = cube_status
    HRQ_VERSION = 1
    _path = Path(CUBE_DIR) / "hr_quality.parquet"
    _bins_path = Path(CUBE_DIR) / "hr_effort.parquet"
    _old = pl.read_parquet(_path) if _path.exists() else None
    if _old is not None and (_old["version"][0] if _old.height else HRQ_VERSION) != HRQ_VERSION:
        _old = None
    _all = q("select distinct activity_id from garmin.fit_session where activity_id is not null")["activity_id"].to_list()
    _have = set(_old["activity_id"].to_list()) if _old is not None else set()
    _new = [a for a in _all if a not in _have]
    if _new:
        _ids = ", ".join(map(str, _new))
        with mo.status.spinner(title=f"Checking heart-rate data of {len(_new)} activities"):
            _metrics = q(f"""
                with r as (
                    select activity_id, idx, heart_rate as hr,
                           cadence > 60 and abs(heart_rate - (cadence + coalesce(fractional_cadence, 0)) * 2) <= 4 as lk,
                           heart_rate - lag(heart_rate) over w as dhr,
                           heart_rate = lag(heart_rate, 30) over w and heart_rate = lag(heart_rate, 15) over w as flat
                    from garmin.fit_record where activity_id in ({_ids})
                    window w as (partition by activity_id order by idx)
                ),
                g as (select activity_id, lk, idx - row_number() over (partition by activity_id, lk order by idx) as grp from r),
                locks as (select activity_id, max(n) as longest_lock from (select activity_id, grp, count(*) as n from g where lk group by all) group by 1)
                select activity_id, count(*) as samples,
                       avg((hr is not null)::int) as coverage,
                       avg((abs(dhr) >= 15)::int) filter (where dhr is not null) as spikes,
                       avg(flat::int) filter (where hr is not null) as stuck,
                       quantile_cont(hr, 0.999) as hr_top,
                       avg((hr < 40)::int) filter (where hr is not null) as too_low,
                       coalesce(any_value(l.longest_lock), 0) as longest_lock
                from r left join locks l using (activity_id) group by 1
            """)
            _strap = q(f"""
                select activity_id, bool_or(antplus_device_type = 'heart_rate'
                    or (source_type = 'bluetooth_low_energy' and device_type = '1')
                    or garmin_product ilike 'hrm%') as strap
                from garmin.fit_device_info where activity_id in ({_ids}) group by 1
            """)
            # heart rate by effort (power in 20 W steps, else speed in 0.25 m/s steps), to compare against strap data
            _bins = q(f"""
                select r.activity_id, s.sport, year(r.timestamp) as y, r.power is not null as pw,
                       case when r.power is not null then round(r.power / 20) else round(r.speed * 4) end as eb,
                       median(r.heart_rate) as hr_med, count(*) as k
                from garmin.fit_record r join garmin.fit_session s using (activity_id)
                where r.activity_id in ({_ids}) and r.heart_rate is not null and r.speed > 1.5
                      and s.sport in ('running', 'cycling', 'cross_country_skiing', 'hiking', 'mountaineering')
                group by all having count(*) >= 30
            """)
        _fresh = (
            pl.DataFrame({"activity_id": _new})
            .join(_metrics, on="activity_id", how="left")
            .join(_strap, on="activity_id", how="left")
            .with_columns(strap=pl.col("strap").fill_null(False), version=pl.lit(HRQ_VERSION))
        )
        _raw = _fresh if _old is None else pl.concat([_old, _fresh], how="diagonal_relaxed")
        _raw.write_parquet(_path)
        _bins_all = _bins if (_old is None or not _bins_path.exists()) else pl.concat([pl.read_parquet(_bins_path), _bins], how="diagonal_relaxed")
        _bins_all.write_parquet(_bins_path)
    else:
        _raw = _old
        _bins_all = pl.read_parquet(_bins_path) if _bins_path.exists() else None

    # thresholds come from your strap-recorded activities: anything beyond what the strap itself ever shows is suspect
    _sraw = _raw.filter(pl.col("strap") & pl.col("samples").is_not_null())
    HR_MAX = float(_sraw["hr_top"].quantile(0.99)) if _sraw.height else 200.0
    # effort check: each activity's heart rate at a given pace or power against the strap's, same sport and year
    _resid = pl.DataFrame(schema={"activity_id": pl.Int64, "effort_gap": pl.Float64})
    if _bins_all is not None and _bins_all.height:
        _sid = set(_sraw["activity_id"].to_list())
        _ref = (
            _bins_all.filter(pl.col("activity_id").is_in(_sid))
            .group_by("sport", "y", "pw", "eb").agg(ref=pl.col("hr_med").median(), n_ref=pl.len())
            .filter(pl.col("n_ref") >= 5)
        )
        _resid = (
            _bins_all.join(_ref, on=["sport", "y", "pw", "eb"])
            .group_by("activity_id")
            .agg(effort_gap=((pl.col("hr_med") - pl.col("ref")) * pl.col("k")).sum() / pl.col("k").sum(), k=pl.col("k").sum())
            .filter(pl.col("k") >= 300).select("activity_id", "effort_gap")
        )
    _q = _raw.join(_resid, on="activity_id", how="left").with_columns(
        reasons=pl.concat_list(
            pl.when(pl.col("samples").is_null() | (pl.col("coverage") < 0.8)).then(pl.lit("gaps in the heart rate")),
            pl.when(pl.col("spikes") > 0.006).then(pl.lit("frequent spikes")),
            pl.when(pl.col("stuck") > 0.3).then(pl.lit("stuck on one value")),
            pl.when((pl.col("hr_top") > HR_MAX + 8) | (pl.col("too_low") > 0.005)).then(pl.lit("impossible values")),
            pl.when(~pl.col("strap") & (pl.col("longest_lock") >= 300)).then(pl.lit("locked onto cadence")),
            pl.when(pl.col("effort_gap") < -30).then(pl.lit("far too low for the effort")),
            pl.when(pl.col("effort_gap") > 25).then(pl.lit("far too high for the effort")),
        ).list.drop_nulls()
    ).with_columns(plausible=pl.col("reasons").list.len() == 0)
    con.register("hrq_df", _q.select("activity_id", "strap", "plausible").to_arrow())
    con.execute("create or replace table hrq as select * from hrq_df")
    HRQ = {r["activity_id"]: r for r in _q.select("activity_id", "strap", "plausible", "reasons").iter_rows(named=True)}
    HRQ_SUMMARY = _q.select(
        strap=pl.col("strap").sum(), plausible=pl.col("plausible").sum(), flagged=(~pl.col("plausible")).sum(), total=pl.len()
    ).row(0, named=True)
    HRQ_SUMMARY
    return (HRQ,)


@app.cell(hide_code=True)
def sport(SPORT_TYPES, get_sport, set_sport):
    sport = mo.ui.dropdown(options=list(SPORT_TYPES), value=get_sport() if get_sport() in SPORT_TYPES else "Run", label="", full_width=True, on_change=set_sport)
    return (sport,)


@app.cell(hide_code=True)
def ui_state(CUBE_DIR):
    # remembered choices, so changing one control (or the sport) never resets the others
    get_sport, set_sport = mo.state("Run")
    get_x, set_x = mo.state("len_time")
    get_length, set_length = mo.state("5 min")
    get_y, set_y = mo.state(None)
    get_dates, set_dates = mo.state(None)
    get_filters, set_filters = mo.state({})
    get_lines, set_lines = mo.state("none")
    get_pick, set_pick = mo.state(None)  # (series, xb) of the clicked chart point
    get_pcts, set_pcts = mo.state("100, 90, 75, 50, 25, 10")
    # animation: what it steps through, the window's step, and the frame on screen
    get_anim, set_anim = mo.state("none")
    get_step, set_step = mo.state("Month")
    get_frame, set_frame = mo.state(None)
    # the frame on screen, readable without making the chart rebuild on every scrub (the chart takes it as its start)
    FRAME_NOW = {"i": None}
    _excluded_file = Path(CUBE_DIR, "excluded.json")
    get_excluded, set_excluded = mo.state(json.loads(_excluded_file.read_text()) if _excluded_file.exists() else [])
    # which chart a pick was made on: the drill-down closes once the chart is rebuilt for another view
    PICK_GEN = {"gen": None, "pick": None, "frame": None}
    # the stretch dragged on the drill-down's charts, measured, for its point on the main chart
    get_bdot, set_bdot = mo.state(None)
    return (
        FRAME_NOW,
        PICK_GEN,
        get_anim,
        get_bdot,
        get_dates,
        get_excluded,
        get_filters,
        get_frame,
        get_length,
        get_lines,
        get_pcts,
        get_pick,
        get_sport,
        get_step,
        get_x,
        get_y,
        set_anim,
        set_bdot,
        set_dates,
        set_excluded,
        set_filters,
        set_frame,
        set_length,
        set_lines,
        set_pcts,
        set_pick,
        set_sport,
        set_step,
        set_x,
        set_y,
    )


@app.cell(hide_code=True)
def frame_report(set_frame):
    # The dashboard reports the animation's frame here ("<frame>:<nonce>") when its scrubber is let go or Play stops, so
    # the headline and the drill-down follow the chart; while you drag or play, only the chart moves
    frame_report = mo.ui.text(
        value="", label="", on_change=lambda v: set_frame(int(v.split(":")[0])) if v.split(":")[0].isdigit() else None
    )
    frame_report
    return


@app.function(hide_code=True)
def control_row(label, el):
    """A control with its label in a fixed column, so every widget (and every range slider) lines up."""
    return mo.hstack(
        [mo.Html(f'<div style="font-size:12px;font-weight:600;line-height:1.25;color:#1e293b;white-space:normal">{label}</div>'), el],
        justify="start",
        align="center",
        gap=0.5,
        widths=[1, 2.7],
    )


@app.cell(hide_code=True)
def sport_cube(
    CUBE_MANIFESTS,
    SCANNERS,
    SCAN_LOCK,
    SPORT_CACHE,
    con,
    cube_files,
    cube_status,
    display_file,
    garmin_connection,
    prepare,
    q,
    re,
    sport,
):
    _ = cube_status  # the curves must exist before anything reads them
    # each sport's curves are loaded into memory once per session, with its slider ranges, so switching back is instant
    if sport.value not in SPORT_CACHE:
        _t = "cube_" + re.sub(r"\W+", "_", sport.value.lower())
        con.execute(f"create or replace table {_t} as select * from {cube_files(sport.value)}")
        _stats = CUBE_MANIFESTS[sport.value]["stats"]
        _span = q(f"select min(a.date) as lo, max(a.date) as hi from acts a semi join (select distinct activity_id from {_t}) c using (activity_id)").row(0)
        # the display table (enveloped curves at the chart's lengths, dated) beside it: what every chart reads
        _dt = "disp_" + re.sub(r"\W+", "_", sport.value.lower())
        con.execute(f"create or replace table {_dt} as select * from '{display_file(sport.value)}'")
        SPORT_CACHE[sport.value] = (_t, _stats, _span, _dt)
    cur_table, CUBE_STATS, CUBE_SPAN, disp_table = SPORT_CACHE[sport.value]


    def _load_scanner(sp):
        with SCAN_LOCK:
            if sp not in SCANNERS:
                _bc = garmin_connection()
                SCANNERS[sp] = prepare(_bc, sp)[0]
                _bc.close()
        return SCANNERS[sp]


    def scanner():
        """The sport's raw running totals for filtered and grouped scans, loaded on first use (a few seconds) and kept."""
        if sport.value in SCANNERS:
            return SCANNERS[sport.value]
        with mo.status.spinner(title=f"Loading every {sport.value} recording for filtered curves"):
            return _load_scanner(sport.value)


    def warm_scanner(sp):
        """Start loading a sport's recordings in the background (nothing if they are loaded or loading)."""
        if sp not in SCANNERS and not SCAN_LOCK.locked():
            __import__("threading").Thread(target=_load_scanner, args=(sp,), daemon=True).start()


    return CUBE_SPAN, CUBE_STATS, cur_table, disp_table, scanner, warm_scanner


@app.cell(hide_code=True)
def dims(CUBE_STATS, GAP_SPORTS, sport):
    SPEED_SPORT = {"Ride", "Skimo", "Skate ski"}
    DIMS = {
        "len_time": "Duration",
        "len_dist": "Distance",
        "tb": "Time into activity",
        "db": "Distance into activity",
        "gb": "Gradient",
        "vb": "Speed" if sport.value in SPEED_SPORT else "Pace",
        "pb": "Power",
        "hb": "Heart rate",
    }
    METRICS = {"p": "Power", "v": "Speed" if sport.value in SPEED_SPORT else "Pace", "g": "Gradient", "h": "Heart rate (sustained)", "a": "Vertical speed"}
    METRIC_DIM = {"p": "pb", "v": "vb", "g": "gb", "h": "hb", "a": "-", "x": "-", "c": "cb", "k": "-", "l": "-"}  # vertical speed, GAP, torque and step length have no filter dimension
    if sport.value in GAP_SPORTS:
        METRICS["x"] = "Grade-adjusted pace"
    _has_power = CUBE_STATS["has_power"]
    # cadence: the x axis, filter and bands use cadence while pedaling or stepping; the y axis the whole window's cadence
    if CUBE_STATS.get("has_cadence"):
        DIMS["cb"] = "Cadence"
        DIMS["sb"] = "Pedaling share" if sport.value == "Ride" else "Stepping share"
        METRICS["c"] = "Cadence"
        if sport.value == "Ride" and _has_power:
            METRICS["k"] = "Torque"
        if sport.value in ("Run", "Hike"):
            METRICS["l"] = "Step length"
    if not _has_power:
        DIMS.pop("pb")
        METRICS.pop("p")
    return DIMS, METRICS, METRIC_DIM, SPEED_SPORT


@app.cell(hide_code=True)
def y_axis(METRICS, get_y, set_y, sport):
    _y = get_y() if get_y() in METRICS else ("p" if sport.value == "Ride" else "v")
    y_axis = mo.ui.dropdown(options={v: k for k, v in METRICS.items()}, value=METRICS[_y], label="", full_width=True, on_change=set_y)
    return (y_axis,)


@app.cell(hide_code=True)
def x_axis(DIMS, get_x, set_x):
    _x = get_x() if get_x() in DIMS and get_x() not in ("tb", "db", "sb") else "len_time"
    x_axis = mo.ui.dropdown(
        options={v: k for k, v in DIMS.items() if k not in ("tb", "db", "sb")},
        value=DIMS[_x],
        label="", full_width=True,
        on_change=set_x,
    )
    return (x_axis,)


@app.cell(hide_code=True)
def length(DISTANCES, DURATIONS, get_length, set_length):
    def fmt_len(basis, n):
        if basis == "time":
            h, rem = divmod(n, 3600)
            m, s = divmod(rem, 60)
            return " ".join(p for p in (f"{h} h" if h else "", f"{m} min" if m else "", f"{s} s" if s else "") if p)
        named = {1609: "1 mi", 8047: "5 mi", 16093: "10 mi", 21097: "half marathon", 42195: "marathon", 80467: "50 mi"}
        return named.get(n, f"{n} m" if n < 1000 else f"{n / 1000:g} km")


    _opts = {f"{fmt_len('time', n)}": ("time", n) for n in DURATIONS} | {f"{fmt_len('dist', n)}": ("dist", n) for n in DISTANCES}
    length = mo.ui.dropdown(options=_opts, value=get_length() if get_length() in _opts else "5 min", label="", full_width=True, on_change=set_length)
    return fmt_len, length


@app.cell(hide_code=True)
def stat():
    stat = mo.ui.radio(options=["Best", "Typical (median)", "Percentile"], value="Best", label="", inline=True)
    return (stat,)


@app.cell(hide_code=True)
def pct():
    pct = mo.ui.slider(start=50, stop=99, value=90, step=1, label="", full_width=True, show_value=True, debounce=True)
    return (pct,)


@app.cell(hide_code=True)
def date_window(CUBE_SPAN, get_dates):
    # the one date window every control reads and writes (quick ranges, pickers, slider)
    _lo, _hi = CUBE_SPAN
    # the last 12 months until another window is chosen; "all" (set by presets) is every activity on record
    _w = (_lo, _hi) if get_dates() == "all" else get_dates() or (_hi - dt.timedelta(days=365), _hi)
    DATE_WINDOW = (max(min(_w[0], _hi), _lo), max(min(_w[1], _hi), _lo))
    if DATE_WINDOW[0] > DATE_WINDOW[1]:
        DATE_WINDOW = (_lo, _hi)
    return (DATE_WINDOW,)


@app.cell(hide_code=True)
def dates(CUBE_SPAN, DATE_WINDOW, set_dates):
    # Slide window: one handle, the window's end, a week per step; the window keeps its length. Its value is the end as a day
    # number, so the dashboard page can pass it to the live chart while it is dragged (the notebook hears it on release).
    _span = CUBE_SPAN
    WEEKS = [(_span[0] + dt.timedelta(weeks=i)) for i in range((_span[1] - _span[0]).days // 7 + 2)]
    _w0, _w1 = DATE_WINDOW
    _L = (_w1 - _w0).days
    _epoch = dt.date(1970, 1, 1)
    _hi_d = (_span[1] - _epoch).days
    _first = min((_span[0] - _epoch).days + _L, _hi_d)
    # ends a whole number of weeks back from the last activity, so they land on the live chart's precomputed positions
    _start = _hi_d - 7 * ((_hi_d - _first) // 7)
    _end = (_w1 - _epoch).days
    _value = min(max(_start + 7 * round((_end - _start) / 7), _start), _hi_d)
    dates = mo.ui.slider(
        start=_start, stop=max(_hi_d, _start + 7), step=7, value=_value, label="", full_width=True, debounce=True, show_value=False,
        on_change=lambda v: set_dates((_epoch + dt.timedelta(days=int(v) - _L), _epoch + dt.timedelta(days=int(v)))),
    )
    return


@app.cell(hide_code=True)
def date_picker(CUBE_SPAN, DATE_WINDOW, set_dates):
    date_picker = mo.ui.date_range(
        start=CUBE_SPAN[0], stop=CUBE_SPAN[1], value=DATE_WINDOW, label="",
        on_change=lambda v: set_dates((v[0], v[1])),
    )
    return (date_picker,)


@app.cell(hide_code=True)
def quick_range(CUBE_SPAN, set_dates):
    def _quick(name):
        _end = CUBE_SPAN[1]
        _today = dt.date.today()
        return {
            "Last 4 weeks": (_end - dt.timedelta(weeks=4), _end),
            "Last 3 months": (_end - dt.timedelta(days=91), _end),
            "Last 6 months": (_end - dt.timedelta(days=182), _end),
            "Last 12 months": (_end - dt.timedelta(days=365), _end),
            "This year": (dt.date(_today.year, 1, 1), _end),
            "Last year": (dt.date(_today.year - 1, 1, 1), dt.date(_today.year - 1, 12, 31)),
            "All": CUBE_SPAN,
        }[name]


    quick_range = mo.ui.dropdown(
        options=["Last 4 weeks", "Last 3 months", "Last 6 months", "Last 12 months", "This year", "Last year", "All"],
        value="Last 12 months",
        label="", full_width=True,
        on_change=lambda v: set_dates(_quick(v)) if v else None,
    )
    return (quick_range,)


@app.cell(hide_code=True)
def lines_by(DIMS, get_lines, set_lines):
    LINE_GROUPS = {
        "None (spread of activities)": "none",
        "Period: by year": "year",
        "Period: same window last year": "lastyear",
        "Period: previous window": "prev",
        **{f"Band: {DIMS[k]}": k for k in ("tb", "db", "gb", "vb", "pb", "hb", "cb") if k in DIMS},
    }
    _g = get_lines() if get_lines() in LINE_GROUPS.values() else "none"
    lines_by = mo.ui.dropdown(
        options=LINE_GROUPS,
        value=next(k for k, v in LINE_GROUPS.items() if v == _g),
        label="", full_width=True,
        on_change=set_lines,
    )
    return (lines_by,)


@app.cell(hide_code=True)
def animate(
    DIMS,
    FRAME_NOW,
    get_anim,
    get_step,
    set_anim,
    set_frame,
    set_step,
):
    # Animate by: a third way to slice the cube, after the axes and the lines. The chart steps through frames: windows
    # sliding through time, calendar years, or the bands of a dimension.
    def _anim(v):
        FRAME_NOW["i"] = None
        set_frame(None)
        set_anim(v)


    _opts = {
        "None": "none", "Sliding window": "window", "Year": "year",
        **{DIMS[k]: k for k in ("tb", "db", "gb", "vb", "pb", "hb", "cb") if k in DIMS},
    }
    _a = get_anim() if get_anim() in _opts.values() else "none"
    animate_by = mo.ui.dropdown(options=_opts, value=next(k for k, v in _opts.items() if v == _a), label="", full_width=True, on_change=_anim)
    anim_step = mo.ui.dropdown(options=["Week", "Month", "Quarter", "Year"], value=get_step(), label="", full_width=True, on_change=set_step)
    return anim_step, animate_by


@app.cell(hide_code=True)
def hr_involved(
    FILTER_FULL,
    METRIC_DIM,
    animate_by,
    filters,
    lines_by,
    x_axis,
    y_axis,
):
    # heart-rate data quality matters only when heart rate is on an axis, in the lines, animated or narrowed by a filter
    HR_INVOLVED = "hb" in (x_axis.value, METRIC_DIM[y_axis.value], lines_by.value, animate_by.value) or ("hb" in FILTER_FULL and tuple(filters.value["hb"]) != FILTER_FULL["hb"])
    return (HR_INVOLVED,)


@app.cell(hide_code=True)
def pct_list(get_pcts, set_pcts):
    pct_list = mo.ui.text(value=get_pcts(), label="", full_width=True, on_change=set_pcts)
    return (pct_list,)


@app.cell(hide_code=True)
def preset(
    FILTER_FULL,
    FRAME_NOW,
    anim_step,
    animate_by,
    filters,
    get_dates,
    get_length,
    lines_by,
    set_anim,
    set_dates,
    set_filters,
    set_frame,
    set_length,
    set_lines,
    set_pick,
    set_sport,
    set_step,
    set_x,
    set_y,
    sport,
    x_axis,
    y_axis,
):
    # every view a preset can set, in full: sport, axes, lines, window length, filters and the period. Ordered by sport.
    # period "all" is every activity on record (for lines by year, and for data that is sparse, such as running power);
    # otherwise the last 12 months.
    PRESETS = {
        "Run · Pace curve": dict(sport="Run", y="v", x="len_time", lines="none"),
        "Run · Best efforts by distance": dict(sport="Run", y="v", x="len_dist", lines="none"),
        "Run · Grade-adjusted pace curve": dict(sport="Run", y="x", x="len_time", lines="none"),
        "Run · Pace curve through the years": dict(sport="Run", y="v", x="len_time", lines="none", animate="window", step="Month"),
        "Run · Pace curve vs a year earlier": dict(sport="Run", y="v", x="len_time", lines="lastyear"),
        "Run · Durability: pace by time into the run": dict(sport="Run", y="v", x="len_time", lines="tb", filters={"tb": [0, 240]}),
        "Run · Pace curve by time into the run": dict(sport="Run", y="v", x="len_time", lines="none", animate="tb", filters={"tb": [0, 240]}),
        "Run · Pace by gradient, year by year": dict(sport="Run", y="v", x="gb", length="5 min", lines="none", animate="year", period="all"),
        "Run · Pace by gradient": dict(sport="Run", y="v", x="gb", length="5 min", lines="none"),
        "Run · Flat pace by heart rate, by year": dict(sport="Run", y="v", x="hb", length="10 min", lines="year", filters={"gb": [-2, 2]}, period="all"),
        "Run · Heart-rate ceiling by year": dict(sport="Run", y="h", x="len_time", lines="year", period="all"),
        "Run · Power by pace": dict(sport="Run", y="p", x="vb", length="5 min", lines="none", period="all"),
        "Run · Pace by cadence": dict(sport="Run", y="v", x="cb", length="5 min", lines="none"),
        "Run · Step length by pace": dict(sport="Run", y="l", x="vb", length="5 min", lines="none"),
        "Ride · Power curve": dict(sport="Ride", y="p", x="len_time", lines="none"),
        "Ride · Power curve by year": dict(sport="Ride", y="p", x="len_time", lines="year", period="all"),
        "Ride · Power curve by gradient band": dict(sport="Ride", y="p", x="len_time", lines="none", animate="gb", period="all"),
        "Ride · Power by gradient": dict(sport="Ride", y="p", x="gb", length="5 min", lines="none", period="all"),
        "Ride · Power by gradient, lines by cadence": dict(sport="Ride", y="p", x="gb", length="5 min", lines="cb", filters={"cb": [60, 120], "sb": [90, 100]}, period="all"),
        "Ride · Power by cadence": dict(sport="Ride", y="p", x="cb", length="5 min", lines="none", filters={"sb": [90, 100]}, period="all"),
        "Ride · Torque curve": dict(sport="Ride", y="k", x="len_time", lines="none", period="all"),
        "Hike · Climbing rate": dict(sport="Hike", y="a", x="len_time", lines="none"),
        "Hike · Pace by gradient": dict(sport="Hike", y="v", x="gb", length="10 min", lines="none"),
        "Skimo · Climbing rate": dict(sport="Skimo", y="a", x="len_time", lines="none", period="all"),
    }


    CUSTOM = "Custom view"


    def _matches(p):
        """Whether the controls show exactly this preset's view right now."""
        if (p["sport"], p["y"], p["x"], p["lines"]) != (sport.value, y_axis.value, x_axis.value, lines_by.value):
            return False
        if p["x"] not in ("len_time", "len_dist") and p.get("length", "5 min") != get_length():
            return False
        if p.get("animate", "none") != animate_by.value:
            return False
        if p.get("animate") == "window" and p.get("step", "Month") != anim_step.value:
            return False
        if get_dates() != ("all" if p.get("period") == "all" else None):
            return False
        # narrowed filters only, clamped to the sliders' ranges the way the sliders clamp them
        _want = {}
        for _k, _v in p.get("filters", {}).items():
            if _k in FILTER_FULL:
                _lo, _hi = FILTER_FULL[_k]
                _c = (min(max(_v[0], _lo), _hi), min(max(_v[1], _lo), _hi))
                if _c != (_lo, _hi):
                    _want[_k] = _c
        _have = {k: tuple(v) for k, v in filters.value.items() if tuple(v) != tuple(FILTER_FULL[k])}
        return _want == _have


    def _apply_preset(name):
        if not name or name == CUSTOM:
            return
        _p = PRESETS[name]
        set_sport(_p["sport"])
        set_y(_p["y"])
        set_x(_p["x"])
        set_lines(_p["lines"])
        if "length" in _p:
            set_length(_p["length"])
        # a preset's filters replace that sport's remembered ones; every other filter goes back to full range
        set_filters(lambda s: {**s, _p["sport"]: _p.get("filters", {})})
        set_dates("all" if _p.get("period") == "all" else None)
        set_pick(None)  # a picked point belongs to the previous view
        set_anim(_p.get("animate", "none"))
        set_step(_p.get("step", "Month"))
        FRAME_NOW["i"] = None
        set_frame(None)


    # the dropdown shows the preset the view currently matches, and "Custom view" as soon as any control departs from it
    _current = next((n for n, p in PRESETS.items() if _matches(p)), CUSTOM)
    preset = mo.ui.dropdown(options=[CUSTOM, *PRESETS], value=_current, label="", full_width=True, on_change=_apply_preset)
    return (preset,)


@app.cell(hide_code=True)
def filters(
    CADENCE_UNIT,
    CUBE_STATS,
    DIMS,
    SPEED_SPORT,
    get_filters,
    set_filters,
    sport,
):
    _b = CUBE_STATS
    _prev = get_filters().get(sport.value, {})
    _sliders = {}
    FILTER_FULL = {}
    FILTER_LABELS = {}


    def _rs(key, lo, hi, step, label):
        FILTER_FULL[key] = (lo, hi)
        FILTER_LABELS[key] = label
        _v = _prev.get(key, (lo, hi))
        _v = [min(max(_v[0], lo), hi), min(max(_v[1], lo), hi)]
        _sliders[key] = mo.ui.range_slider(start=lo, stop=hi, step=step, value=_v, label="", show_value=True, debounce=True, full_width=True)


    # one slider per dimension for the whole sport; the axes only decide which are shown and applied
    _rs("tb", 0, (_b["tb"] + 1) * 5, 5, "Time in (min)")
    _rs("db", 0, (_b["db"] + 1) * 0.5, 0.5, "Distance in (mi)")
    _rs("gb", _b["glo"], _b["ghi"] + 1, 1, "Gradient (%)")
    if sport.value in SPEED_SPORT:
        _rs("vb", round(_b["vlo"] * 0.2 * 2.23694), round((_b["vhi"] + 1) * 0.2 * 2.23694), 1, "Speed (mph)")
    else:
        _rs("vb", max(3.0, round(26.8224 / ((_b["vhi"] + 1) * 0.2) * 4) / 4), min(40.0, round(26.8224 / (max(_b["vlo"], 1) * 0.2) * 4) / 4), 0.25, "Pace (min/mi)")
    if "pb" in DIMS:
        _rs("pb", 0, (_b["pb"] + 1) * 20, 20, "Power (W)")
    if _b["hlo"] is not None:
        _rs("hb", _b["hlo"] * 5, (_b["hhi"] + 1) * 5, 5, "Heart rate (bpm)")
    if _b.get("has_cadence"):
        _rs("cb", _b["clo"] * 5, (_b["chi"] + 1) * 5, 5, f"Cadence ({CADENCE_UNIT[sport.value]})")
        _rs("sb", 0, 100, 5, ("Pedaling" if sport.value == "Ride" else "Stepping") + " (% of window)")
    filters = mo.ui.dictionary(_sliders, on_change=lambda v: set_filters(lambda s: {**s, sport.value: v}))
    return FILTER_FULL, FILTER_LABELS, filters


@app.cell(hide_code=True)
def hr_data():
    HR_LEVELS = {"All activities": "any", "Plausible heart rate": "plausible", "Chest strap or armband": "strap"}
    hr_data = mo.ui.dropdown(options=HR_LEVELS, value="All activities", label="", full_width=True)
    return (hr_data,)


@app.cell(hide_code=True)
def control_rows(
    CADENCE_UNIT,
    DATE_WINDOW,
    FILTER_FULL,
    FILTER_LABELS,
    METRIC_DIM,
    SPEED_SPORT,
    animate_by,
    date_picker,
    filters,
    fmt_len,
    length,
    lines_by,
    preset,
    quick_range,
    set_filters,
    sport,
    x_axis,
    y_axis,
):
    _w0, _w1 = DATE_WINDOW
    _g = lines_by.value
    _hide = {x_axis.value, METRIC_DIM[y_axis.value]}
    _speedy = sport.value in SPEED_SPORT
    _value_axis = x_axis.value not in ("len_time", "len_dist")

    # active filters in words: a slider left at one end is open on that side
    _SHORT = {
        "tb": ("time in", " min"), "db": ("distance in", " mi"), "gb": ("gradient", "%"),
        "vb": ("speed", " mph") if _speedy else ("pace", " /mi"), "pb": ("power", " W"), "hb": ("heart rate", " bpm"),
        "cb": ("cadence", f" {CADENCE_UNIT[sport.value]}"), "sb": ("pedaling" if sport.value == "Ride" else "stepping", "% of the window"),
    }


    def _fv(k, v):
        return f"{int(v)}:{int(round(v % 1 * 60)):02d}" if (k == "vb" and not _speedy) else f"{v:g}"


    ACTIVE_FILTERS = {}
    for _k, (_lo, _hi) in filters.value.items():
        if _k in _hide or _k not in FILTER_FULL:
            continue
        _flo, _fhi = FILTER_FULL[_k]
        _open_lo, _open_hi = _lo <= _flo, _hi >= _fhi
        if _open_lo and _open_hi:
            continue
        _name, _u = _SHORT[_k]
        ACTIVE_FILTERS[_k] = (
            f"{_name} ≥ {_fv(_k, _lo)}{_u}" if _open_hi
            else f"{_name} ≤ {_fv(_k, _hi)}{_u}" if _open_lo
            else f"{_name} {_fv(_k, _lo)}–{_fv(_k, _hi)}{_u}"
        )
    FILTER_SUMMARY = ", ".join(ACTIVE_FILTERS.values())
    WINDOW_TEXT = f"{fmt_len(*length.value)} windows" if _value_axis else ""


    def _reset(keys):
        def _do(_):
            set_filters(lambda s: {**s, sport.value: {**s.get(sport.value, {}), **{k: list(FILTER_FULL[k]) for k in keys}}})
        return _do




    def _filter_row(k):
        _label = FILTER_LABELS[k]
        if k not in ACTIVE_FILTERS:
            return control_row(_label, filters[k])
        # narrowed: accent label and a reset beside the slider
        _btn = mo.ui.button(label="Reset", kind="neutral", on_click=_reset([k]))
        return control_row(
            f'<span style="color:#1f4e9c">● {_label}</span>',
            mo.hstack([filters[k], _btn], justify="start", align="center", gap=0.3, widths=[1, 0]),
        )


    _shown = [k for k in FILTER_FULL if k not in _hide]
    _filter_head = mo.hstack(
        [
            mo.Html('<div style="font-size:12px;font-weight:600;color:#1e293b;margin-top:6px">Only count windows where…</div>'),
            *([mo.ui.button(label="Reset all", kind="neutral", on_click=_reset(list(ACTIVE_FILTERS)))] if ACTIVE_FILTERS else []),
        ],
        justify="space-between",
        align="end",
    )
    # one list of labelled rows, shared by the notebook's sidebar and the Studio dashboard's rail
    CONTROL_ROWS = [
        control_row("Preset", preset),
        control_row("Sport", sport),
        control_row("Y axis", y_axis),
        control_row("X axis", x_axis),
        control_row("Lines", lines_by),
        control_row("Animate by", animate_by),
        control_row("Period", quick_range),
        control_row("From – to", date_picker),
        # the chart's own controls (window length, statistic, percentiles, heart-rate data, step, slide window) are in
        # chart_controls, above the chart
        _filter_head,
        *[_filter_row(k) for k in _shown],
    ]
    return CONTROL_ROWS, FILTER_SUMMARY, WINDOW_TEXT


@app.cell(hide_code=True)
def dot_cloud_cell():
    # The dot cloud under the curve (in the row above the chart): None, or one of three principled subsets of the efforts
    # behind it
    dot_cloud = mo.ui.dropdown(
        {"None": "none", "Strongest length": "strong", "Records as set": "records", "Near the best": "near"},
        value="None",
    )
    return (dot_cloud,)


@app.cell(hide_code=True)
def chart_controls_cell(
    HR_INVOLVED,
    anim_step,
    animate_by,
    dot_cloud,
    hr_data,
    length,
    lines_by,
    pct,
    pct_list,
    stat,
    x_axis,
):
    # Controls that belong to the chart on screen rather than to the cube, in a row above the chart and only when they
    # apply, so the controls panel is the same whatever chart it makes
    _g = lines_by.value


    def _tool(label, el, width=None):
        _el = mo.Html(f'<div style="width:{width}px">{el}</div>') if width else el
        return mo.hstack(
            [mo.Html(f'<span style="font-size:12px;font-weight:600;color:#1e293b;white-space:nowrap">{label}</span>'), _el],
            justify="start", align="center", gap=0.4,
        )


    _tools = [
        *([_tool("Window", length)] if x_axis.value not in ("len_time", "len_dist") else []),
        *([_tool("Each line is the", stat)] if _g != "none" else []),
        *([_tool("Percentile", pct, 200)] if _g != "none" and stat.value == "Percentile" else []),
        *([_tool("Percentiles", pct_list, 190)] if _g == "none" else []),
        *([_tool("Heart-rate data", hr_data)] if HR_INVOLVED else []),
        *([_tool("Step", anim_step, 110)] if animate_by.value == "window" else []),
        _tool("Dots", dot_cloud, 170),
    ]
    chart_controls = mo.hstack(_tools, justify="start", align="center", gap=1.4, wrap=True) if _tools else mo.Html("")
    chart_controls
    return


@app.cell(hide_code=True)
def controls(CONTROL_ROWS):
    mo.sidebar([mo.md("### Data cubes"), *CONTROL_ROWS], width="360px")
    return


@app.cell(hide_code=True)
def viewport():
    # the dashboard reports the size of its chart area here ("1300x880", CSS px), so the charts are drawn to fit it exactly
    viewport = mo.ui.text(value="", label="")
    viewport
    return (viewport,)


@app.cell(hide_code=True)
def dash_layout(viewport):
    import re
    # chart sizes for the Studio dashboard, from the chart area the page reports (fallback: a laptop screen).
    # Closed: the curve fills the area. Open: curve and drill-down split it evenly.
    _m = re.match(r"(\d+)x(\d+)", viewport.value or "")
    _W, _H = (int(_m[1]), int(_m[2])) if _m else (1300, 880)
    _half = _H // 2
    _map = max(180, _half - 78)  # square map: half the height, less the drill-down header and padding
    DASH = {
        "curve_h": max(220, _H - 128),  # less the legend above and the axis below
        "curve_h_open": max(120, _half - 128),
        "map": _map,
        "streams_w": max(360, _W - _map - 150),  # less the map, the gap between them and the streams' y axes
        "plain_w": max(360, _W - 110),  # streams alone, for activities without GPS
    }
    return DASH, re


@app.cell(hide_code=True)
def controls_panel(CONTROL_ROWS):
    # the same controls for the Studio dashboard's rail
    controls_panel = mo.vstack(CONTROL_ROWS, gap=0.45)
    return


@app.cell(hide_code=True)
def curve(
    BINS,
    CADENCE_CAP,
    CADENCE_UNIT,
    CUBE_SPAN,
    DATE_WINDOW,
    DIMS,
    DISPLAY_DIST,
    DISPLAY_TIME,
    FILTER_FULL,
    HR_INVOLVED,
    METRIC_DIM,
    POS_DAYS,
    POS_PCTS,
    POS_STEP,
    SPEED_CAP,
    SPEED_SPORT,
    acts,
    anim_step,
    animate_by,
    con,
    cur_table,
    disp_table,
    ec,
    filters,
    get_excluded,
    hr_data,
    length,
    lines_by,
    load_positions,
    pct,
    pct_list,
    q,
    scanner,
    sport,
    stat,
    x_axis,
    y_axis,
):
    _ = acts  # registered in DuckDB by the cube cell; referencing it orders this cell after that one
    _x, _ym, _g = x_axis.value, y_axis.value, lines_by.value
    _ydim = METRIC_DIM[_ym]
    _speedy = sport.value in SPEED_SPORT
    _w0, _w1 = DATE_WINDOW
    _len_days = (_w1 - _w0).days + 1
    _len_axis = _x in ("len_time", "len_dist")
    _A = animate_by.value  # none | window | year | a band dimension
    _T0 = time.time()

    # periods: which dates feed which line
    if _g == "lastyear":
        _periods = [("Selected", _w0, _w1), ("A year earlier", _w0 - dt.timedelta(days=365), _w1 - dt.timedelta(days=365))]
    elif _g == "prev":
        _periods = [("Selected", _w0, _w1), (f"Previous {_len_days // 7} weeks", _w0 - dt.timedelta(days=_len_days), _w0 - dt.timedelta(days=1))]
    else:
        _periods = [("Selected", _w0, _w1)]
    _period_sql = " union all ".join(f"select '{n}' as series, date '{a}' as lo, date '{b}' as hi" for n, a, b in _periods)


    def _unit(k, v):
        if k == "vb" and not _speedy:
            return f"{int(v)}:{int(round(v % 1 * 60)):02d}"
        return f"{v:g}"


    _UNITS = {"tb": " min in", "db": " mi in", "gb": "%", "pb": " W", "hb": " bpm", "vb": " mph" if _speedy else " /mi", "cb": f" {CADENCE_UNIT[sport.value]}", "sb": "%"}
    # each slider dimension in the engine: which window value it tests, and its display units -> engine units
    _ENGINE = {"tb": "t0", "db": "d0", "gb": "g", "vb": "v", "pb": "p", "hb": "h", "cb": "pc", "sb": "ps"}


    def _to_engine(k, v):
        if k == "tb":
            return v * 60
        if k == "db":
            return v * 1609.344
        if k in ("gb", "sb"):
            return v / 100
        if k == "vb":
            return v / 2.23694 if _speedy else (26.8224 / v if v > 0 else float("inf"))
        return float(v)


    def _range(k, lo, hi):
        """Display range -> engine (lo, hi), NaN for an open end. Pace runs backwards: slow paces are low speeds."""
        _a, _b = _to_engine(k, lo), _to_engine(k, hi)
        return (_b, _a) if (k == "vb" and not _speedy) else (_a, _b)


    # band widths in round numbers, finest first (display units)
    _LADDER = {
        "tb": [10, 15, 30, 60, 120], "db": [0.5, 1, 2, 5, 10], "gb": [1, 2, 2.5, 5, 10, 20],
        "vb": [1, 2, 5, 10] if _speedy else [0.25, 0.5, 1, 2, 5], "pb": [10, 20, 25, 50, 100], "hb": [5, 10, 20],
        "cb": [2, 5, 10, 20], "sb": [5, 10, 25],
    }


    def _band_edges(k, cap):
        """Bands across the dimension's filter range (itself the data's range unless narrowed), of the finest round
        width on the dimension's ladder that keeps them to `cap`: edges (display units) and names."""
        _lo, _hi = filters.value[k]
        _count = lambda w: np.ceil((_hi - np.floor(_lo / w) * w) / w - 1e-9)
        _step = next((w for w in _LADDER[k] if _count(w) <= cap), _LADDER[k][-1])
        _start = np.floor(_lo / _step) * _step
        _e = [round(float(_start + i * _step), 4) for i in range(int(np.ceil((_hi - _start) / _step - 1e-9)) + 1)]
        _e[0], _e[-1] = max(_e[0], _lo), min(_e[-1], _hi)
        # an end band cut to half a width or less by the filter range joins its neighbour
        if len(_e) > 3 and _e[1] - _e[0] <= _step / 2:
            _e.pop(1)
        if len(_e) > 3 and _e[-1] - _e[-2] <= _step / 2:
            _e.pop(-2)
        # a dash between two numbers reads as a minus once either is negative, so those say "to", with a real minus sign
        _sep = lambda a, b: " to " if min(a, b) < 0 else "–"
        _num = lambda v: _unit(k, v).replace("-", "−")
        return _e, [f"{_num(_e[i])}{_sep(_e[i], _e[i + 1])}{_num(_e[i + 1])}{_UNITS[k]}" for i in range(len(_e) - 1)]


    _anim_clash = _A in _ENGINE and _A in (_x, _ydim, _g)
    _period_clash = _A in ("window", "year") and _g in ("lastyear", "prev")
    AXES_CLASH = _x == _ydim or (_g in _ENGINE and _g in (_x, _ydim)) or _anim_clash or _period_clash
    CLASH_MSG = (
        f"**{DIMS[_x]}** is on both axes: pick a different x axis." if _x == _ydim
        else f"Animation is by **{DIMS.get(_A, _A)}**, which is already an axis or the lines: animate by something else." if _anim_clash
        else "Lines by period compare fixed windows, and the animation moves through time: pick one of the two." if _period_clash
        else f"Lines are grouped by **{DIMS.get(_g, _g)}**, which is already an axis: group by something else."
    )

    if _g == "year":
        SERIES_KIND, LEGEND_TITLE = "ordered", "Year"
    elif _g in ("lastyear", "prev"):
        SERIES_KIND, LEGEND_TITLE = "categorical", None
        SERIES_ORDER = [p[0] for p in _periods]
    elif _g in _ENGINE:
        SERIES_KIND, LEGEND_TITLE = "ordered", DIMS[_g]
        # the filter is the outer range; bands of a round width split it, at most 10 lines (thin end bands go after the scan)
        _edges, _names = _band_edges(_g, 10)
        _n = len(_names)
        SERIES_ORDER = _names
    else:
        SERIES_KIND, LEGEND_TITLE = ("ordered", "Statistic") if _g == "none" else ("categorical", None)

    # statistics: one line per percentile when grouping by statistic, else the Statistic control
    if _g == "none":
        _qs = []
        for _tok in pct_list.value.replace(";", ",").split(","):
            try:
                _v = float(_tok)
            except ValueError:
                continue
            if 0 <= _v <= 100 and _v not in _qs:
                _qs.append(_v)
        _qs = sorted(_qs or [100, 50], reverse=True)
        _plan = [("Best" if v == 100 else "Lowest" if v == 0 else "Median" if v == 50 else f"p{v:g}", v / 100) for v in _qs]
        SERIES_ORDER = [n for n, _ in _plan]
        SERIES_Q = dict(_plan)
    else:
        SERIES_Q = {}
        _plan = [(None, 1.0 if stat.value == "Best" else 0.5 if stat.value.startswith("Typical") else pct.value / 100)]

    # filters: exact bounds on each window's own value; an end left at the slider's extreme is open
    _filters = []
    for _k, (_flo, _fhi) in filters.value.items():
        if _k in (_x, _ydim):
            continue  # an axis dimension is not filtered
        _full_lo, _full_hi = FILTER_FULL[_k]
        _open_lo, _open_hi = _flo <= _full_lo, _fhi >= _full_hi
        if _open_lo and _open_hi:
            continue
        _a, _b = _range(_k, _flo, _fhi)
        if _k == "vb" and not _speedy:
            _open_lo, _open_hi = _open_hi, _open_lo
        _filters.append((_ENGINE[_k], float("nan") if _open_lo else _a, float("nan") if _open_hi else _b))

    # activities that feed the lines. A time animation (window, year) runs through every date on record; otherwise the
    # date periods apply. Then the exclude list and, when heart rate takes part, its data quality.
    _epoch = dt.date(1970, 1, 1)
    if _A in ("window", "year"):
        _sel = acts.with_columns(series=pl.lit("Selected"))
    else:
        _sel = acts.join(pl.DataFrame({"series": [p[0] for p in _periods], "lo": [p[1] for p in _periods], "hi": [p[2] for p in _periods]}), how="cross")
        _sel = _sel.filter(pl.col("date").is_between(pl.col("lo"), pl.col("hi")))
    if get_excluded():
        _sel = _sel.filter(~pl.col("activity_id").is_in(get_excluded()))
    if HR_INVOLVED and hr_data.value != "any":
        _sel = _sel.filter(pl.col("activity_id").is_in(q(f"select activity_id from hrq where {hr_data.value}")["activity_id"].to_list()))
    if _g == "year":
        _sel = _sel.with_columns(series=pl.col("date").dt.year().cast(pl.String))
    elif _g not in ("lastyear", "prev"):
        _sel = _sel.with_columns(series=pl.lit("Selected"))
    _sel = _sel.with_columns(d=(pl.col("date") - pl.lit(_epoch)).dt.total_days().cast(pl.Int32))

    # Per-activity values. Stored route: an unfiltered duration or distance curve, read from the display table (curves
    # enveloped at the chart's lengths). Scan route: anything filtered, banded, animated by a band or against a value
    # axis, from every window of the selected activities with exact bounds.
    SCAN = not (_len_axis and not _filters and _g not in _ENGINE and _A not in _ENGINE)
    _basis = ("time" if _x == "len_time" else "dist") if _len_axis else length.value[0]
    _mi = ec.METRICS.index(_ym)
    _scan_s = None
    _EMPTY = {"series": pl.String, "activity_id": pl.Int64, "fb": pl.Int32, "xb": pl.Int64, "y": pl.Float64, "t_start": pl.Int64, "src0": pl.Int64}
    _abands = []
    if _A in _ENGINE:
        # at most 15 frames; on a value axis each line band and animation band is its own pass, so with both the
        # animation gets coarser to keep the passes near 40
        _ae, _an = _band_edges(_A, max(3, 40 // _n) if (not _len_axis and _g in _ENGINE) else 15)
        _abands = [(_an[i], _range(_A, _ae[i], _ae[i + 1])) for i in range(len(_an))]
        # the scrubber's two ends: the range the bands cover, not the first and last band
        FRAME_ENDS = [f"{_unit(_A, v).replace('-', chr(0x2212))}{_UNITS[_A]}" for v in (_ae[0], _ae[-1])]
    else:
        FRAME_ENDS = None
    if AXES_CLASH or not _sel.height:
        _per = pl.DataFrame(schema=_EMPTY)
    elif not SCAN:
        con.register("sel_df", _sel.select("series", "activity_id").to_arrow())
        _per = q(f"""
            select s.series, c.activity_id, 0::int as fb, c.xb::bigint as xb, c.y_{_ym}::double as y, c.t_{_ym}::bigint as t_start, c.s_{_ym}::bigint as src0
            from {disp_table} c join sel_df s using (activity_id)
            where c.basis = '{_basis}' and c.y_{_ym} is not null
        """)
    else:
        _P = scanner()
        _index = {int(a): i for i, a in enumerate(_P.activity_ids)}
        # only activities that record the metric at all (power, heart rate): the others cannot contribute a window
        _has = q(f"select distinct activity_id from {cur_table} where {_ym} is not null")["activity_id"].to_list()
        _sel = _sel.filter(pl.col("activity_id").is_in(list(_index)) & pl.col("activity_id").is_in(_has))
        _uniq = _sel["activity_id"].unique(maintain_order=True).to_list()
        _ix = np.array([_index[a] for a in _uniq], dtype=np.int64)
        _lengths = (DISPLAY_TIME if _basis == "time" else DISPLAY_DIST) if _len_axis else np.array([float(length.value[1])])
        # line bands and animation bands, in engine units
        _lbands = []
        if _g in _ENGINE:
            _be = [_range(_g, _edges[i], _edges[i + 1]) for i in range(_n)]
            _lbands = [(_names[i], _be[i]) for i in range(_n)]

        def _cuts_of(bands):
            """Bin edges for a grouped pass over contiguous bands; the outer edges stay open (the dimension's filter bounds them)."""
            _c = sorted({e for _b in bands for e in _b[1]})
            return _c, np.array([-1e12, *_c[1:-1], 1e12])

        # up to two splits go in the engine's grouped pass at once (the x bins on a value axis, the line bands, the
        # animation bands, in that order); a third, if there is one, is a pass per band, filtered to it
        _roles = []
        if not _len_axis:
            _w = BINS[_x]
            _lo_b, _hi_b = {"gb": (-60, 60), "vb": (0, int(np.ceil(SPEED_CAP[sport.value] / _w)) + 1), "pb": (0, 150), "hb": (6, 47), "cb": (0, int(CADENCE_CAP[sport.value] / _w) + 1)}[_x]
            _roles.append(("x", _ENGINE[_x], (np.arange(_lo_b, _hi_b + 1) * _w).astype(np.float64)))
        if _lbands:
            _lcuts, _le = _cuts_of(_lbands)
            _roles.append(("lines", _ENGINE[_g], _le))
        if _abands:
            _acuts, _ae2 = _cuts_of(_abands)
            _roles.append(("anim", _ENGINE[_A], _ae2))
        _grouped, _looped = _roles[:2], _roles[2:]
        _gnames = [r[0] for r in _grouped]
        _lloop = _lbands if (_lbands and "lines" not in _gnames) else [(None, None)]
        _aloop = list(enumerate(_abands)) if (_abands and "anim" not in _gnames) else [(None, None)]
        _t = time.time()
        _parts = []

        def _rows(series, fb, yy, tt, xs):
            _a, _k = yy.shape
            return pl.DataFrame({
                "series": [series] * (_a * _k) if series is not None else [None] * (_a * _k),
                "activity_id": np.repeat(np.array(_uniq, dtype=np.int64), _k), "fb": np.full(_a * _k, fb if fb is not None else 0, dtype=np.int32),
                "xb": np.tile(np.asarray(xs).astype(np.int64), _a),
                "y": yy.reshape(-1).astype(np.float64), "t_start": tt.reshape(-1).astype(np.int64),
            }, schema_overrides={"series": pl.String}).filter(pl.col("y").is_not_nan())

        # the bins of each grouped role: (index in the engine's output, line name or None, frame or None, x bins or None)
        def _bins(role):
            if role == "x":
                return [(None, None, None, "x")]
            if role == "lines":
                return [(_lcuts.index(_a), _name, None, None) for _name, (_a, _b) in _lbands]
            return [(_acuts.index(_a), None, _k2, None) for _k2, (_name, (_a, _b)) in enumerate(_abands)]

        for _ln, _lb in _lloop:
            for _ai, _ab in _aloop:
                _f = list(_filters)
                if _lb is not None:
                    _f.append((_ENGINE[_g], _lb[0], np.nextafter(_lb[1], -np.inf)))
                if _ab is not None:
                    _f.append((_ENGINE[_A], _ab[1][0], np.nextafter(_ab[1][1], -np.inf)))
                _kw = {}
                if _grouped:
                    _kw["group"] = (_grouped[0][1], _grouped[0][2])
                if len(_grouped) > 1:
                    _kw["group2"] = (_grouped[1][1], _grouped[1][2])
                _y, _t0 = _P.compute(_basis, _lengths, 0.0, _f, activities=_ix, **_kw)
                _y, _t0 = _y[..., _mi], _t0[..., _mi]
                # walk the grouped axes: each combination of bins is one (line, frame) curve; x bins stay as the length axis
                _combos = [((), _ln, _ai, None)]
                for _role in _gnames:
                    _combos = [(_ix0 + ((_b,) if _b is not None else (slice(None),)), _nm if _nm is not None else _l0, _fr if _fr is not None else _a0, _xs or _x0)
                               for (_ix0, _l0, _a0, _x0) in _combos for (_b, _nm, _fr, _xs) in _bins(_role)]
                for _sel_ix, _name, _frame, _xs in _combos:
                    _yy, _tt = _y[(slice(None), *_sel_ix)], _t0[(slice(None), *_sel_ix)]
                    if _xs == "x":
                        # x bins are the curve's own axis: (activities, x bins) after the other picks
                        _parts.append(_rows(_name, _frame, _yy.reshape(_yy.shape[0], -1), _tt.reshape(_tt.shape[0], -1), np.arange(_lo_b, _hi_b)))
                    else:
                        _parts.append(_rows(_name, _frame, _yy, _tt, _lengths))
        _scan_s = time.time() - _t
        _per = pl.concat(_parts).join(_sel.select(pl.col("series").alias("pseries"), "activity_id"), on="activity_id")
        _per = _per.with_columns(series=pl.coalesce("series", "pseries"), src0=pl.col("xb")).select(list(_EMPTY))
    _per = _per.join(_sel.select("activity_id", "d").unique("activity_id"), on="activity_id", how="left")


    def _thin_ends(counts):
        """The run of bands to keep: thin bands at either end (fewer than 3 activities, or under 5 % of the busiest band) go."""
        _min = max(3, 0.05 * max(counts, default=0))
        _i0, _i1 = 0, len(counts)
        while _i0 < _i1 and counts[_i0] < _min:
            _i0 += 1
        while _i1 > _i0 and counts[_i1 - 1] < _min:
            _i1 -= 1
        return (_i0, _i1) if _i1 > _i0 else (0, len(counts))


    # line bands: every line kept has data behind it
    if _g in _ENGINE and _per.height:
        _la = dict(_per.group_by("series").agg(pl.col("activity_id").n_unique()).iter_rows())
        _l0, _l1 = _thin_ends([_la.get(_nm, 0) for _nm in _names])
        if (_l0, _l1) != (0, len(_names)):
            _per = _per.filter(pl.col("series").is_in(_names[_l0:_l1]))
            _names, _edges = _names[_l0:_l1], _edges[_l0:_l1 + 1]
            _n = len(_names)
            SERIES_ORDER = _names

    # Monotone on duration and distance axes: each activity's best at a length is the best held for at least that long
    # (best of this length and every longer one), filtered or not. The display table already holds it; scans get it here.
    _T1 = time.time()
    _T2 = None
    MONOTONE = _len_axis and SCAN
    _env = "max(y) over w" if MONOTONE else "y"
    _env_t = "arg_max(t_start, y) over w" if MONOTONE else "t_start"
    _env_src = "arg_max(src0, y) over w" if MONOTONE else "src0"
    _env_window = "window w as (partition by series, fb, activity_id order by xb desc rows between unbounded preceding and current row)" if MONOTONE else ""

    # Frames: what the animation steps through. Each frame is a set of activities (a window, a year) or a band; every
    # frame's Best (with its effort), percentiles and count per line and length are computed here at once, so moving
    # between frames in the browser only picks rows.
    _probs = sorted({p / 100 for p in POS_PCTS} | {v / 100 for v in (_qs if _g == "none" else []) if 0 <= v < 100}
                    | ({pct.value / 100} if _g != "none" and stat.value == "Percentile" else set()))
    _L = (_w1 - _w0).days
    _hi_d, _lo_d = (CUBE_SPAN[1] - _epoch).days, (CUBE_SPAN[0] - _epoch).days
    _steps = {"Week": 7, "Month": 28, "Quarter": 91, "Year": 364}
    _n_series = max(1, len(SERIES_ORDER) if _g != "year" else int(_sel["series"].n_unique()))
    _n_x = len(DISPLAY_TIME if _basis == "time" else DISPLAY_DIST) if _len_axis else 120
    FRAME_NOTE = ""
    _frame_tbl, _kept, _join = None, None, None
    _fast = (
        not SCAN and _g == "none" and _len_axis and abs(_L - POS_DAYS) <= 1 and not (HR_INVOLVED and hr_data.value != "any")
        and _A == "window" and anim_step.value == "Week" and (_hi_d - (_w1 - _epoch).days) % POS_STEP == 0
    )
    if AXES_CLASH or not _per.height:
        FRAME_KIND, FRAMES, FRAME_LABELS, FRAME_ACTS, FRAME_DEFAULT = "none", pl.DataFrame(), [], [], 0
    elif _fast:
        # the precomputed positions of the default window: the Slide window moves through them
        _pos = load_positions(sport.value, get_excluded()).filter((pl.col("metric") == _ym) & (pl.col("basis") == _basis)).sort("e")
        _ends = sorted(_pos["e"].unique().to_list())
        _fi = {e: i for i, e in enumerate(_ends)}
        FRAMES = _pos.with_columns(f=pl.col("e").replace_strict(_fi, return_dtype=pl.Int32), s=pl.lit(0, pl.Int32)).drop("metric", "basis", "e")
        FRAME_KIND = "window"
        FRAME_LABELS = [f"{_epoch + dt.timedelta(days=e - POS_DAYS + 1):%b %d, %Y} – {_epoch + dt.timedelta(days=e):%b %d, %Y}" for e in _ends]
        _na = dict(FRAMES.group_by("f").agg(pl.col("n").max()).iter_rows())
        FRAME_ACTS = [_na.get(i, 0) for i in range(len(_ends))]
        FRAME_DEFAULT = _fi.get((_w1 - _epoch).days, len(_ends) - 1)
        _frame_tbl = pl.DataFrame({"f": list(range(len(_ends))), "e": _ends})
        _join = f"join frame_df fr on p.d > fr.e - {POS_DAYS} and p.d <= fr.e"
    else:
        con.register("per_act0_df", _per.to_arrow())
        con.execute(f"""
            create or replace temp table per_act as
            select series, fb, activity_id, xb, d, {_env} as y, {_env_t} as t_start, {_env_src} as src_xb
            from per_act0_df {_env_window}
        """)
        if _A == "window":
            # window ends a step apart back from the last activity; coarser steps when the frames would get too heavy
            _step = _steps[anim_step.value]
            _order = list(_steps.values())
            while ((_hi_d - _lo_d) // _step + 1) * _n_series * _n_x > 80_000 and _step != _order[-1]:
                _step = _order[_order.index(_step) + 1]
            if _step != _steps[anim_step.value]:
                FRAME_NOTE = f"stepping by {next(k for k, v in _steps.items() if v == _step).lower()} (finer steps would be too much data)"
            _ends = [_hi_d - _step * k for k in range((_hi_d - _lo_d) // _step + 1)][::-1]
            _frame_tbl = pl.DataFrame({"f": list(range(len(_ends))), "e": _ends})
            con.register("frame_df", _frame_tbl.to_arrow())
            _join = f"join frame_df fr on p.d > fr.e - {_L} and p.d <= fr.e"
            FRAME_LABELS = [f"{_epoch + dt.timedelta(days=e - _L + 1):%b %d, %Y} – {_epoch + dt.timedelta(days=e):%b %d, %Y}" for e in _ends]
            FRAME_DEFAULT = len(_ends) - 1
        elif _A == "year":
            _years = sorted(_sel["date"].dt.year().unique().to_list())
            _frame_tbl = pl.DataFrame({"f": list(range(len(_years))), "yr": _years})
            con.register("frame_df", _frame_tbl.to_arrow())
            _join = "join frame_df fr on year(date '1970-01-01' + p.d) = fr.yr"
            FRAME_LABELS = [str(y) for y in _years]
            FRAME_DEFAULT = len(_years) - 1
        elif _A in _ENGINE:
            _join = "join (select distinct fb as f from per_act) fr on p.fb = fr.f"
            FRAME_LABELS = [b[0] for b in _abands]
            FRAME_DEFAULT = 0
        else:
            _join = "join (select 0 as f) fr on true"
            FRAME_LABELS = ["Selected"]
            FRAME_DEFAULT = 0
        FRAME_KIND = "window" if _A == "window" else "year" if _A == "year" else "band" if _A in _ENGINE else "none"
        _sidx = {n: i for i, n in enumerate(SERIES_ORDER)} if _g not in ("year", "none") else None
        _T2 = time.time()
        _agg = q(f"""
            select fr.f::int as f, p.series, p.xb::int as xb, count(*)::int as n, max(p.y) as best,
                   arg_max(p.activity_id, p.y) as aid, arg_max(p.t_start, p.y)::int as t, arg_max(p.src_xb, p.y)::int as src,
                   quantile_cont(p.y, [{', '.join(str(x) for x in _probs)}]) as q
            from per_act p {_join}
            group by all
        """)
        _acts_f = q(f"select fr.f::int as f, count(distinct p.activity_id)::int as a from per_act p {_join} group by 1")
        FRAME_ACTS = [dict(zip(_acts_f["f"], _acts_f["a"])).get(i, 0) for i in range(len(FRAME_LABELS))]
        if _g == "year":
            SERIES_ORDER = sorted(_agg["series"].unique().to_list())
        _sidx = {n: i for i, n in enumerate(SERIES_ORDER)} if _g != "none" else {}
        FRAMES = _agg.with_columns(
            [pl.col("q").list.get(i).alias(f"q{round(p * 100, 6):g}") for i, p in enumerate(_probs)]
        ).drop("q").with_columns(
            s=(pl.col("series").replace_strict(_sidx, default=0, return_dtype=pl.Int32) if _sidx else pl.lit(0, pl.Int32))
        ).drop("series")
    # a frame with fewer than 3 activities (a Best line from one or two, no spread) is skipped, so the scrubber and Play
    # jump over gaps in the record; a band animation also drops the thin bands at its ends, and its scrubber's ends say
    # where the bands with data are. If every frame is that thin, they all stay.
    if FRAME_KIND != "none" and FRAME_ACTS:
        _keep = [i for i, a in enumerate(FRAME_ACTS) if a >= 3]
        if FRAME_KIND == "band":
            _f0, _f1 = _thin_ends(FRAME_ACTS)
            _keep = [i for i in _keep if _f0 <= i < _f1]
        if 0 < len(_keep) < len(FRAME_LABELS):
            _new = {old: new for new, old in enumerate(_keep)}
            _kept = _keep
            FRAMES = FRAMES.filter(pl.col("f").is_in(_keep)).with_columns(f=pl.col("f").replace_strict(_new, return_dtype=pl.Int32))
            FRAME_LABELS = [FRAME_LABELS[i] for i in _keep]
            FRAME_ACTS = [FRAME_ACTS[i] for i in _keep]
            FRAME_DEFAULT = _new[max([i for i in _keep if i <= FRAME_DEFAULT] or [_keep[0]])]
            if FRAME_KIND == "band":
                FRAME_ENDS = [f"{_unit(_A, v).replace('-', chr(0x2212))}{_UNITS[_A]}" for v in (_ae[_keep[0]], _ae[_keep[-1] + 1])]
    # nothing to draw at all: say why rather than show an empty chart
    EMPTY_MSG = (
        f"No {sport.value.lower()} activity in this period has **{DIMS[_A].lower()}** data to animate by: widen the period or animate by something else."
        if _A in _ENGINE else "No activities match this view: widen the period or the filters."
    )
    FRAME_PROBS = _probs
    # what the dot cloud reads: every activity's curve, how activities fall into frames, and the frames kept
    CLOUD_IN = None if (AXES_CLASH or not FRAMES.height or _join is None) else {
        "per": _per, "env": MONOTONE, "join": _join, "frames": _frame_tbl, "kept": _kept,
    }
    _T3 = time.time()
    CURVE_TIMING = {"route": "scan" if SCAN else "stored", "scan_s": _scan_s,
                    "stages_s": {"select_and_scan": round(_T1 - _T0, 2), "envelope": round((_T2 or _T3) - _T1, 2), "frames": round(_T3 - (_T2 or _T3), 2)}, "activities": int(_per["activity_id"].n_unique()) if _per.height else 0, "frames": len(FRAME_LABELS)}
    if _g in ("lastyear", "prev"):
        SERIES_ORDER = [p[0] for p in _periods]


    # colours that match what the lines mean
    def _hex(c):
        return [int(c[i:i + 2], 16) for i in (1, 3, 5)]


    def _mix(a, b, t):
        return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(_hex(a), _hex(b)))


    def _seq(n, light, dark):
        return [_mix(light, dark, i / max(n - 1, 1)) for i in range(n)] if n > 1 else [dark]


    def _div(values, center, neg, pos, mid="#9e9e9e"):
        # each side scaled on its own, so the extremes on both sides get the darkest shade
        up = max((v - center for v in values if v > center), default=1) or 1
        down = max((center - v for v in values if v < center), default=1) or 1
        out = []
        for v in values:
            t = (v - center) / (up if v > center else down)
            out.append(mid if abs(t) < 1e-9 else _mix(mid, pos if t > 0 else neg, 0.45 + 0.55 * abs(t)))
        return out


    _n_series = len(SERIES_ORDER)
    if _g == "none":
        # diverging from the median: above 50 toward blue (best darkest), below toward orange
        SERIES_COLORS = _div(_qs, 50, "#7f3b08", "#08306b")
    elif _g == "gb":
        # downhill blue, uphill red, flat grey
        SERIES_COLORS = _div([(_edges[i] + _edges[i + 1]) / 2 for i in range(_n)], 0, "#08519c", "#99000d")
    elif _g in ("tb", "db"):
        SERIES_COLORS = _seq(_n_series, "#9ecae1", "#08306b") if _g == "tb" else _seq(_n_series, "#bcbddc", "#3f007d")
    elif _g in ("pb", "vb", "hb", "cb"):
        # more intense = darker; for pace, lower numbers are faster, so reverse
        _c = _seq(_n_series, "#fdae6b", "#7f2704")
        SERIES_COLORS = _c[::-1] if (_g == "vb" and not _speedy) else _c
    elif _g == "year":
        SERIES_COLORS = _seq(_n_series, "#a1d99b", "#00441b")
    elif _g in ("lastyear", "prev"):
        SERIES_COLORS = ["#1f4e9c", "#a0a0a0"]
    else:
        SERIES_COLORS = ["#1f4e9c"] * max(_n_series, 1)
    return (
        AXES_CLASH,
        CLASH_MSG,
        CLOUD_IN,
        EMPTY_MSG,
        FRAMES,
        FRAME_ACTS,
        FRAME_DEFAULT,
        FRAME_ENDS,
        FRAME_KIND,
        FRAME_LABELS,
        FRAME_NOTE,
        LEGEND_TITLE,
        SERIES_COLORS,
        SERIES_ORDER,
        SERIES_Q,
    )


@app.cell(hide_code=True)
def frame_curve(FRAMES, SERIES_ORDER, SERIES_Q, acts, lines_by, pct, stat):
    # The curve of one frame, in the shape the headline cards, the drill-down and the fallback chart read
    def frame_curve(i):
        g = lines_by.value
        schema = {"series": pl.String, "xb": pl.Int32, "y": pl.Float64, "n": pl.Int64, "activity_id": pl.Int64, "t_start": pl.Int64, "src_xb": pl.Int32}
        parts = []
        if FRAMES.height:
            fr = FRAMES.filter(pl.col("f") == i)
            qcol = lambda qv: f"q{round(qv * 100, 6):g}"
            best = lambda name, src: src.select(
                series=name, xb=pl.col("xb"), y=pl.col("best").cast(pl.Float64), n=pl.col("n"), activity_id=pl.col("aid"), t_start=pl.col("t"), src_xb=pl.col("src")
            )
            summ = lambda name, src, qv: src.filter(pl.col("n") >= 3).select(
                series=name, xb=pl.col("xb"), y=pl.col(qcol(qv)).cast(pl.Float64), n=pl.col("n"),
                activity_id=pl.lit(None, pl.Int64), t_start=pl.lit(None, pl.Int64), src_xb=pl.lit(None, pl.Int32)
            )
            if g == "none":
                for name, qv in SERIES_Q.items():
                    parts.append(best(pl.lit("Best"), fr) if qv == 1.0 else summ(pl.lit(name), fr, qv))
            else:
                name = pl.col("s").replace_strict(dict(enumerate(SERIES_ORDER)), default="", return_dtype=pl.String)
                qv = 1.0 if stat.value == "Best" else 0.5 if stat.value.startswith("Typical") else pct.value / 100
                parts.append(best(name, fr) if qv == 1.0 else summ(name, fr, qv))
        return (
            pl.concat([p.cast(schema) for p in parts]) if parts else pl.DataFrame(schema=schema)
        ).join(acts.select("activity_id", "name", "date"), on="activity_id", how="left").sort("series", "xb")


    return (frame_curve,)


@app.cell(hide_code=True)
def cloud(
    CLOUD_IN,
    FRAME_LABELS,
    SERIES_ORDER,
    acts,
    con,
    dot_cloud,
    lines_by,
    q,
    x_axis,
):
    # The dot cloud: a principled few of the efforts behind the curve, one dot each, for the frame on screen.
    # - Strongest length: every activity once, at the length where it ranks highest among the frame's activities. The rank
    #   counts over n + 1, so topping a handful of activities counts for less than topping many.
    # - Records as set: each effort that beat every earlier one in the frame at its length, at the time, once there are
    #   at least 3 earlier ones to beat (the first few efforts of a frame are always records).
    # - Near the best: efforts within 3 % of the frame's Best at their length (the Best itself is the line).
    # Computed from the same per-activity curves as the frames, so the dots follow the filters, the lines and the frames.
    CLOUD, CLOUD_NAMES, CLOUD_WHY, CLOUD_NOTE, CLOUD_TIMING = None, None, "''", "", None
    _mode = dot_cloud.value
    if _mode != "none" and CLOUD_IN is not None and CLOUD_IN["per"].height:
        _ci = CLOUD_IN
        _T = time.time()
        con.register("cloud_per0", _ci["per"].select("series", "fb", "activity_id", "xb", "d", "y", "t_start", "src0").to_arrow())
        if _ci["env"]:
            # each activity's best at a length is the best held for at least that long, as the curve counts it
            _ysql = "max(y) over w as y, arg_max(t_start, y) over w as t_start, arg_max(src0, y) over w as src_xb"
            _wsql = "window w as (partition by series, fb, activity_id order by xb desc rows between unbounded preceding and current row)"
        else:
            _ysql, _wsql = "y, t_start, src0 as src_xb", ""
        con.execute(f"""
            create or replace temp table cloud_env as
            select series, fb, activity_id, xb, d, {_ysql} from cloud_per0 where y is not null and not isnan(y) {_wsql}
        """)
        if _ci["frames"] is not None:
            con.register("cloud_frame_df", _ci["frames"].to_arrow())
        _join = _ci["join"].replace("frame_df", "cloud_frame_df").replace("from per_act)", "from cloud_env)")
        if lines_by.value == "none":
            _ssql, _sjoin = "0", ""
        else:
            con.register("cloud_series", pl.DataFrame({"series": SERIES_ORDER, "s": list(range(len(SERIES_ORDER)))}).to_arrow())
            _ssql, _sjoin = "sm.s", "join cloud_series sm on sm.series = p.series"
        _b = f"""
            select fr.f::int as f, {_ssql}::int as s, p.activity_id, p.xb::int as xb, p.d, p.y, p.t_start::int as t, p.src_xb::int as src
            from cloud_env p {_join} {_sjoin}
        """
        _sql = {
            "strong": f"""
                with b as ({_b}), r as (
                    select *, count(*) over w as n, rank() over (partition by f, s, xb order by y) as rk, y / nullif(median(y) over w, 0) as rel
                    from b window w as (partition by f, s, xb)
                )
                select *, n - rk + 1 as pos, rk / (n + 1.0) as score from r where n >= 3
                qualify row_number() over (partition by f, s, activity_id order by rk / (n + 1.0) desc, rel desc nulls last, xb) = 1
            """,
            "records": f"""
                with b as ({_b}), r as (
                    select *, max(y) over e as prev, count(*) over e as earlier, y / nullif(max(y) over (partition by f, s, xb), 0) as keep
                    from b window e as (partition by f, s, xb order by d, activity_id rows between unbounded preceding and 1 preceding)
                )
                select *, (y - prev) / nullif(abs(prev), 0) as score from r where earlier >= 3 and y > prev
            """,
            "near": f"""
                with b as ({_b}), r as (
                    select *, max(y) over w as best, count(*) over w as n from b window w as (partition by f, s, xb)
                )
                select *, (best - y) / nullif(abs(best), 0) as gap, -(best - y) / nullif(abs(best), 0) as score
                from r where n >= 3 and y < best and y >= best - 0.03 * abs(best)
            """,
        }[_mode]
        _c = q(_sql)
        # the frames the chart kept (thin ones are skipped), renumbered as the chart numbers them
        if _ci["kept"] is not None:
            _map = {old: new for new, old in enumerate(_ci["kept"])}
            _c = _c.filter(pl.col("f").is_in(list(_map))).with_columns(f=pl.col("f").replace_strict(_map, return_dtype=pl.Int32))
        # bounded: the chart carries every frame's dots, so past this many each frame keeps its most notable ones
        _MAX = 15_000
        if _c.height > _MAX:
            _per_f = max(20, _MAX // max(len(FRAME_LABELS), 1))
            # most notable: the strongest ranks, the records nearest the frame's Best, the efforts nearest it
            _c = _c.sort("keep" if "keep" in _c.columns else "score", descending=True, nulls_last=True).group_by("f", maintain_order=True).head(_per_f)
            CLOUD_NOTE = f"the {_per_f} most notable per frame"
        # numbers only per dot (the chart carries every frame's): the reason as k1, k2, put in words by the browser, and the
        # activities' names in a table of their own
        _here = "length" if x_axis.value in ("len_time", "len_dist") else "point"
        _k = {
            "strong": (pl.col("pos"), pl.col("n")),
            "records": ((pl.col("score") * 100).round(1), pl.lit(0)),
            "near": ((pl.col("gap") * 100).round(1), pl.lit(0)),
        }[_mode]
        CLOUD_WHY = {
            "strong": f"'Strongest {_here}: #' + datum.k1 + ' of ' + datum.k2 + ' here'",
            "records": "'Record when set: ' + format(datum.k1, '.1f') + '% past the earlier best'",
            "near": "format(datum.k1, '.1f') + '% off the Best'",
        }[_mode]
        CLOUD = _c.select(
            pl.col("f").cast(pl.Int32), pl.col("s").cast(pl.Int32), pl.col("xb").cast(pl.Int64), pl.col("y").round(4),
            "activity_id", "t", "src", _k[0].alias("k1"), _k[1].alias("k2"),
        )
        CLOUD_NAMES = acts.filter(pl.col("activity_id").is_in(CLOUD["activity_id"].unique().to_list())).select(
            "activity_id", "name", pl.col("date").dt.strftime("%b %d, %Y").alias("date")
        )
        CLOUD_TIMING = round(time.time() - _T, 2)
    return CLOUD, CLOUD_NAMES, CLOUD_WHY


@app.cell(hide_code=True)
def curve_now(
    CHART_GEN,
    FRAMES,
    FRAME_DEFAULT,
    FRAME_LABELS,
    FRAME_NOW,
    PICK_GEN,
    frame_curve,
    get_frame,
    lines_by,
    set_pick,
):
    # The frame on screen, as the curve every other cell reads (headline cards, drill-down): the scrubber's frame, or the
    # animation's default, clamped to the frames there are
    _nf = len(FRAME_LABELS)
    _fi = get_frame()
    FRAME_I = _fi if (_fi is not None and 0 <= _fi < _nf) else FRAME_DEFAULT
    FRAME_NOW["i"] = FRAME_I
    # a drill-down open on another frame: keep its activity if it is on this frame's Best line (at its nearest point),
    # else close it, so the panel never shows an activity the chart does not
    _pk = PICK_GEN.get("pick")
    if _pk is not None and PICK_GEN.get("gen") == CHART_GEN and PICK_GEN.get("frame") not in (None, FRAME_I):
        _mine = FRAMES.filter((pl.col("f") == FRAME_I) & (pl.col("aid") == _pk[2])) if (len(_pk) == 5 and lines_by.value == "none") else FRAMES.head(0)
        if _mine.height:
            _m = _mine.with_columns(_d=(pl.col("xb") - _pk[1]).abs()).sort("_d").row(0, named=True)
            _keep = ("Best", int(_m["xb"]), int(_m["aid"]), int(_m["t"]), int(_m["src"]))
            PICK_GEN.update(pick=_keep, frame=FRAME_I)
            set_pick(_keep)
        else:
            PICK_GEN.update(pick=None, frame=None)
            set_pick(None)
    curve = frame_curve(FRAME_I)
    return FRAME_I, curve


@app.cell(hide_code=True)
def chart(
    AXES_CLASH,
    CADENCE_UNIT,
    CLIENT_SPEC,
    DASH,
    DATE_WINDOW,
    DIMS,
    FRAME_DEFAULT,
    LEGEND_TITLE,
    METRICS,
    SERIES_COLORS,
    SERIES_ORDER,
    SERIES_Q,
    SPEED_SPORT,
    fmt_len,
    frame_curve,
    length,
    lines_by,
    pct,
    sport,
    stat,
    x_axis,
    y_axis,
):
    # the notebook's chart: only built when the live chart does not cover the view
    chart_spec = None
    if CLIENT_SPEC is None:
        _x = x_axis.value
        _ym = "v" if y_axis.value == "x" else y_axis.value  # grade-adjusted pace is drawn exactly like pace
        _speedy = sport.value in SPEED_SPORT
        # the default frame: this chart is drawn by the notebook, so it does not follow the scrubber
        _d = frame_curve(FRAME_DEFAULT).with_columns(
            x=(
                pl.col("xb") if _x in ("len_time", "len_dist")
                else (pl.col("xb") + 0.5) * 5 if _x == "tb"
                else (pl.col("xb") + 0.5) * 0.5 if _x == "db"
                else pl.col("xb") + 0.5 if _x == "gb"
                else (pl.col("xb") + 0.5) * 20 if _x == "pb"
                else (pl.col("xb") + 0.5) * 5 if _x in ("hb", "cb")
                else ((pl.col("xb") + 0.5) * 0.2 * 2.23694 if _speedy else 26.8224 / ((pl.col("xb") + 0.5) * 0.2))
            ),
            yv=(
                pl.col("y") if _ym == "p"
                else pl.col("y") * 100 if _ym == "g"
                else pl.col("y") if _ym == "h"
                else pl.col("y") * 3600 * 3.28084 if _ym == "a"
                else pl.col("y") if _ym in ("c", "k")
                else pl.col("y") * 39.3701 if _ym == "l"
                else (pl.col("y") * 2.23694 if _speedy else 26.8224 / pl.col("y"))
            ),
            started=(pl.col("t_start") / 60).round(1),
        )
        if _x not in ("len_time", "len_dist") and _d.height:
            # on a value axis (heart rate, pace, power, gradient, time or distance in), keep only the stretch where at
            # least 3 activities contribute: past it, Best rests on one or two outings, usually bad sensor data
            _ok = _d.group_by("xb").agg(pl.col("n").max()).filter(pl.col("n") >= 3)["xb"]
            if _ok.len():
                _d = _d.filter(pl.col("xb").is_between(_ok.min(), _ok.max()))
        if _x not in ("len_time", "len_dist") and _d.height:
            # empty bins become null rows so lines break instead of bridging gaps
            _grid = _d.select("series").unique().join(pl.DataFrame({"xb": pl.int_range(_d["xb"].min(), _d["xb"].max() + 1, eager=True).cast(_d["xb"].dtype)}), how="cross")
            _d = _grid.join(_d, on=["series", "xb"], how="left")
        if _x == "vb" and not _speedy:
            _d = _d.filter(pl.col("x") < 30)
        _pace_expr = "floor(datum.value) + ':' + (round((datum.value % 1) * 60) < 10 ? '0' : '') + round((datum.value % 1) * 60)"
        _dur_expr = "datum.value < 60 ? datum.value + 's' : datum.value < 3600 ? round(datum.value / 60) + 'm' : (datum.value / 3600) + 'h'"
        _dist_expr = "datum.value < 1000 ? datum.value + ' m' : datum.value == 1609 ? '1 mi' : datum.value == 21097 ? 'half' : datum.value == 42195 ? 'mara' : (datum.value / 1000) + ' km'"
        _xtitle = {"len_time": "Duration", "len_dist": "Distance", "tb": "Minutes into activity", "db": "Miles into activity", "gb": "Gradient (%)", "pb": "Power (W)", "hb": "Heart rate (bpm)", "vb": "Speed (mph)" if _speedy else "Pace (min/mi)", "cb": f"Cadence ({CADENCE_UNIT[sport.value]})"}[_x]
        if _x == "len_time":
            _xenc = alt.X("x:Q", title=_xtitle, scale=alt.Scale(type="log"), axis=alt.Axis(values=[5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200, 18000], labelExpr=_dur_expr))
        elif _x == "len_dist":
            _xenc = alt.X("x:Q", title=_xtitle, scale=alt.Scale(type="log"), axis=alt.Axis(values=[100, 400, 1000, 1609, 5000, 10000, 21097, 42195, 100000], labelExpr=_dist_expr))
        elif _x == "vb" and not _speedy:
            _xenc = alt.X("x:Q", title=_xtitle, scale=alt.Scale(zero=False), axis=alt.Axis(labelExpr=_pace_expr))
        else:
            _xenc = alt.X("x:Q", title=_xtitle, scale=alt.Scale(zero=False))

        # x: exactly the span of the data, no dead space either side
        _xd = _d.drop_nulls("yv")["x"] if _d.height else None
        if _xd is not None and _xd.len():
            _xenc.scale = alt.Scale(domain=[float(_xd.min()), float(_xd.max())], nice=False, **({"type": "log"} if _x in ("len_time", "len_dist") else {"zero": False}))
        _ytitle = {"p": "Power (W)", "g": "Gradient (%)", "h": "Heart rate (bpm)", "a": "Vertical speed (ft/h)", "v": "Speed (mph)" if _speedy else "Pace (min/mi)", "c": f"Cadence ({CADENCE_UNIT[sport.value]})", "k": "Torque (N·m)", "l": "Step length (in)"}[_ym]
        if y_axis.value == "x":
            _ytitle = "Grade-adjusted pace (min/mi)"
        # units on a second line of the axis title
        _ytitle2 = [_ytitle[:_ytitle.index(" (")], _ytitle[_ytitle.index(" (") + 1:]] if " (" in _ytitle else _ytitle
        _yenc = alt.Y("yv:Q", title=_ytitle2, scale=alt.Scale(zero=False, reverse=(_ym == "v" and not _speedy)), axis=alt.Axis(labelExpr=_pace_expr) if (_ym == "v" and not _speedy) else alt.Undefined)
        _stat_txt = stat.value if stat.value != "Percentile" else f"{pct.value}th percentile"
        _len_txt = "" if _x in ("len_time", "len_dist") else f" over {fmt_len(*length.value)}"
        # hover tooltip: only the fields a point has (summary points carry no activity), with readable values.
        # On the nearest-point overlay a hovered item's row is datum.datum, on the points themselves it is datum.
        _D = "(isValid(datum.datum) ? datum.datum : datum)"


        def _k(s):
            return "'" + str(s).replace("'", "\\'") + "'"


        _common = f"{_k(LEGEND_TITLE or 'Line')}: {_D}.series, {_k(_xtitle)}: {_D}.xlab, {_k(_ytitle)}: {_D}.ylab"
        _TIP = (
            f"isValid({_D}.name) ? {{{_common}, 'Activity': {_D}.name, 'Date': timeFormat(toDate({_D}.date), '%b %d, %Y'), "
            f"'Starts': {_D}.started + ' min in'}} : {{{_common}, 'Activities': {_D}.n}}"
        )
        _d = _d.with_columns(clk=pl.col("activity_id").is_not_null() if "activity_id" in _d.columns else pl.lit(False))
        # a hand cursor over points that open an activity: the hovered tuple carries clk, and Vega's built-in cursor signal follows it
        _cursor = alt.param(name="cursor", expr="length(data('hover_store')) && data('hover_store')[0].values[2] ? 'pointer' : 'default'")
        _base = alt.Chart(_d).encode(
            x=_xenc,
            y=_yenc,
            color=alt.Color("series:N", sort=SERIES_ORDER, legend=alt.Legend(title=LEGEND_TITLE, orient="top"), scale=alt.Scale(domain=SERIES_ORDER, range=SERIES_COLORS)),
            tooltip=alt.value({"expr": _TIP}),
        )
        # percentiles as a fan chart: symmetric pairs (10/90, 25/75, ...) become shaded ribbons,
        # the rest (best, median, lowest, unpaired) stay lines
        _rib = None
        _ribbons = alt.Chart(_d.head(0)).mark_area().encode(x=_xenc, y=_yenc)
        _paired = set()
        if lines_by.value == "none" and _d.height:
            _qv = {n: round(v * 100, 6) for n, v in SERIES_Q.items()}
            _pairs = [(lo, hi) for hi, hv in _qv.items() for lo, lv in _qv.items() if 50 < hv < 100 and abs(lv - (100 - hv)) < 1e-9]
            _paired = {n for p in _pairs for n in p}
            _rib = []
            # where Best runs beyond the spread (fewer than 3 activities), every band collapses onto the Best point,
            # so the shading tapers into Best instead of stopping short
            _bx = _d.filter(pl.col("series") == "Best").select("xb", "x", pl.col("yv").alias("best"))

            def _edges(_lo_n, _hi_n):
                _lo_d = _d.filter(pl.col("series") == _lo_n).select("xb", "x", pl.col("yv").alias("lo"))
                _hi_d = _d.filter(pl.col("series") == _hi_n).select("xb", pl.col("yv").alias("hi"))
                if not _bx.height:
                    return _lo_d.join(_hi_d, on="xb", how="inner")
                return (
                    _bx.join(_lo_d.drop("x"), on="xb", how="left").join(_hi_d, on="xb", how="left")
                    .with_columns(lo=pl.coalesce("lo", "best"), hi=pl.coalesce("hi", "best"))
                    .drop("best")
                )

            for _lo_n, _hi_n in _pairs:
                _rib.append(_edges(_lo_n, _hi_n).with_columns(band=pl.lit(f"{_lo_n}–{_hi_n}"), width=pl.lit(_qv[_hi_n] - _qv[_lo_n])))
            # shade on out to Best: from the highest percentile below 100 up to the best effort
            _top_q = max((v for v in _qv.values() if 50 < v < 100), default=None)
            if "Best" in _qv and _top_q is not None:
                _top = next(n for n, v in _qv.items() if v == _top_q)
                _rib.append(_edges(_top, "Best").with_columns(band=pl.lit(f"{_top}–Best"), width=pl.lit(999.0)))
            if _rib:
                _rib = pl.concat(_rib).sort("width", descending=True)
                _bands = _rib["band"].unique(maintain_order=True).to_list()
                _ribbons = (
                    alt.Chart(_rib)
                    .mark_area(interpolate="linear", invalid="break-paths-filter-domains", clip=True)
                    .encode(
                        # progressively shaded: the band out to Best is faintest, and the nested bands stack darker toward
                        # the median, so every percentile edge shows without a line of its own
                        opacity=alt.Opacity("band:N", scale=alt.Scale(domain=_bands, range=[0.1 if b.endswith("Best") else 0.2 for b in _bands]), legend=None),
                        x=_xenc,
                        y=alt.Y("lo:Q", title=_ytitle2, scale=alt.Scale(zero=False, reverse=(_ym == "v" and not _speedy)), axis=alt.Axis(labelExpr=_pace_expr) if (_ym == "v" and not _speedy) else alt.Undefined),
                        y2="hi:Q",
                        fill=alt.Fill("band:N", sort=_bands, scale=alt.Scale(domain=_bands, range=["#4a78b5"] * len(_bands)), legend=alt.Legend(title="Spread of activities", orient="top", symbolOpacity=0.5)),
                        tooltip=["band:N", alt.Tooltip("x:Q", title=_xtitle, format=".2f")],
                    )
                )
            _line_names = [n for n in SERIES_ORDER if n not in _paired]
            _base = _base.encode(
                color=alt.Color("series:N", sort=_line_names, scale=alt.Scale(domain=_line_names, range=[SERIES_COLORS[SERIES_ORDER.index(n)] for n in _line_names]), legend=alt.Legend(title="Statistic", orient="top"))
            )


        # y: fit the part that matters. In the spread view that is Best and the median (the ribbons are clipped);
        # with grouped lines, every line. A little padding either side.
        _core = _d.filter(pl.col("series").is_in(["Best", "Median"])) if lines_by.value == "none" else _d
        _core = _core.drop_nulls("yv")["yv"]
        if _core.len():
            _pad = (_core.max() - _core.min()) * 0.06 or 1
            _ydom = [float(_core.min() - _pad), float(_core.max() + _pad)]
            _rev = _ym == "v" and not _speedy
            _yax = alt.Axis(labelExpr=_pace_expr) if _rev else alt.Undefined
            _yenc = alt.Y("yv:Q", title=_ytitle2, scale=alt.Scale(domain=_ydom, reverse=_rev, nice=False), axis=_yax)
            if isinstance(_rib, pl.DataFrame):
                _ribbons = _ribbons.encode(y=alt.Y("lo:Q", title=_ytitle2, scale=alt.Scale(domain=_ydom, reverse=_rev, nice=False), axis=_yax))

        # crosshair: the point nearest the pointer (any curve) gets rules to both axes and labelled values
        def _fmt(kind, v):
            if v is None:
                return None
            if kind == "dur":
                return fmt_len("time", int(v))
            if kind == "dist":
                return fmt_len("dist", int(v))
            if kind == "pace":
                return f"{int(v)}:{int(round(v % 1 * 60)):02d} /mi"
            return {"grade": f"{v:.0f}%", "power": f"{v:.0f} W", "speed": f"{v:.1f} mph", "min": f"{v:.0f} min", "mi": f"{v:.1f} mi", "bpm": f"{v:.0f} bpm", "vam": f"{v:,.0f} ft/h", "cad": f"{v:.0f} {CADENCE_UNIT[sport.value]}", "torque": f"{v:.0f} N·m", "len": f"{v:.0f} in"}[kind]


        _xkind = {"len_time": "dur", "len_dist": "dist", "gb": "grade", "pb": "power", "vb": "speed" if _speedy else "pace", "tb": "min", "db": "mi", "hb": "bpm", "cb": "cad"}[_x]
        _ykind = {"p": "power", "g": "grade", "h": "bpm", "a": "vam", "v": "speed" if _speedy else "pace", "c": "cad", "k": "torque", "l": "len"}[_ym]
        _d = _d.with_columns(
            xlab=pl.col("x").map_elements(lambda v: _fmt(_xkind, v), return_dtype=pl.String),
            ylab=pl.col("yv").map_elements(lambda v: _fmt(_ykind, v), return_dtype=pl.String),
        )
        # pointermove + clear on leaving the chart: hovering ON a point (under the highlight ring) no longer flickers
        _near = alt.selection_point(name="hover", nearest=True, on="pointermove", clear="view:mouseleave", fields=["series", "xb", "clk"], empty=False)
        _base = _base.properties(data=_d) if False else alt.Chart(_d).encode(x=_xenc, y=_yenc, color=_base.encoding.color, tooltip=_base.encoding.tooltip)
        # percentiles that edge a ribbon are drawn as thin lines in the ribbon colour (so p90 shows against the band above it)
        # and keep invisible hover points; the rest (best, median, unpaired) are full lines
        _paired_names = sorted(_paired)
        _line_src = _base.transform_filter(alt.FieldOneOfPredicate(field="series", oneOf=[n for n in SERIES_ORDER if n not in _paired])) if _paired else _base
        _line = _line_src.mark_line(strokeWidth=2.2, invalid="break-paths-filter-domains", clip=True)
        _edge = (
            _base.transform_filter(alt.FieldOneOfPredicate(field="series", oneOf=_paired_names))
            .mark_line(strokeWidth=1, opacity=0.75, invalid="break-paths-filter-domains", clip=True)
            .encode(color=alt.value("#4a78b5"), detail="series:N")
        ) if _paired else None
        _pick = alt.selection_point(name="pick", on="click", fields=["series", "xb"], empty=False)
        # dots only where a point is one activity's effort (Best): summary lines (median, percentiles) stay plain lines,
        # with invisible points underneath so hovering them still shows the tooltip
        _pts = (_base.transform_filter("isValid(datum.activity_id)") if _paired else _base).mark_circle(size=40, clip=True).encode(
            opacity=alt.condition("isValid(datum.activity_id)", alt.value(1), alt.value(0))
        ).add_params(_near, _pick)
        # off the Best points, hovering anywhere shows every statistic at that length (one tooltip per x position)
        _xrule = _xsel = None
        if _paired and _d.height:
            _wide = _d.select("xb", "x", "xlab", "series", "ylab").pivot(on="series", index=["xb", "x", "xlab"], values="ylab")
            _cols = [n for n in SERIES_ORDER if n in _wide.columns]
            _xhov = alt.selection_point(name="xhov", nearest=True, on="pointermove", clear="view:mouseleave", encodings=["x"], empty=False)
            _xsel = alt.Chart(_wide).mark_point(opacity=0).encode(x=_xenc).add_params(_xhov)
            _xrule = alt.Chart(_wide).mark_rule(color="#94a3b8", strokeWidth=1).encode(
                x=_xenc, tooltip=[alt.Tooltip("xlab:N", title=_xtitle)] + [alt.Tooltip(f"{n}:N", title=n) for n in _cols]
            ).transform_filter(_xhov)
        # the clicked best effort gets a ring, drawn by the browser from the click itself (no kernel round trip)
        _picked = _base.mark_circle(size=320, filled=False, stroke="#f97316", strokeWidth=3, opacity=1).transform_filter(_pick).transform_filter("isValid(datum.activity_id)")
        _hl = _base.mark_circle(size=150, stroke="white", strokeWidth=2, opacity=1).transform_filter(_near)
        if _paired:
            _hl = _hl.encode(color=alt.condition(alt.FieldOneOfPredicate(field="series", oneOf=_paired_names), alt.value("#4a78b5"), _base.encoding.color))
        _vrule = alt.Chart(_d).mark_rule(strokeDash=[4, 3], color="gray").encode(x=_xenc).transform_filter(_near)
        _hrule = alt.Chart(_d).mark_rule(strokeDash=[4, 3], color="gray").encode(y=_yenc).transform_filter(_near)
        _xtext = alt.Chart(_d).mark_text(align="left", dx=5, dy=-6, fontSize=12, fontWeight="bold").encode(x=_xenc, y=alt.value(420), text="xlab:N").transform_filter(_near)
        _ytext = alt.Chart(_d).mark_text(align="left", dx=6, dy=-7, fontSize=12, fontWeight="bold").encode(x=alt.value(0), y=_yenc, text="ylab:N").transform_filter(_near)

        _chart = (
            alt.layer(*[l for l in (_ribbons, _xsel, _xrule, _vrule, _hrule, _line, _pts, _hl, _picked, _xtext, _ytext) if l is not None]).add_params(_cursor)
            .properties(width="container", height=DASH["curve_h"], title=alt.Title(f"{sport.value}: {(_stat_txt.lower() + ' ') if lines_by.value != 'none' else ''}{METRICS[y_axis.value].lower()} by {DIMS[_x].lower()}{_len_txt}{(', lines by ' + (LEGEND_TITLE or 'period').lower()) if lines_by.value != 'none' else ', spread of activities'}", subtitle=f"{DATE_WINDOW[0]:%b %d, %Y} to {DATE_WINDOW[1]:%b %d, %Y}; click a best-effort point to see where it happened" + ("" if stat.value == "Best" else "; points backed by fewer than 3 activities are hidden")))
            .configure_view(strokeWidth=0)
            .configure_axis(gridOpacity=0.3, domain=False, labelFontSize=11, titleFontSize=12, titleFontWeight="normal")
            .configure_title(anchor="start", fontSize=16, subtitleFontSize=12, offset=10)
        )
        # the dashboard's top bar carries the title; the chart keeps only its legend
        _chart.title = alt.Undefined
        # height from the chart area the page reports (DASH): a container height resolves to 0 in the dashboard's output host
        chart_spec = None if AXES_CLASH else _chart
    return (chart_spec,)


@app.cell(hide_code=True)
def client_chart(
    AXES_CLASH,
    CADENCE_UNIT,
    CLOUD,
    CLOUD_NAMES,
    CLOUD_WHY,
    DIMS,
    FRAMES,
    FRAME_ACTS,
    FRAME_DEFAULT,
    FRAME_ENDS,
    FRAME_KIND,
    FRAME_LABELS,
    FRAME_NOW,
    LEGEND_TITLE,
    SERIES_COLORS,
    SERIES_ORDER,
    SERIES_Q,
    SPEED_SPORT,
    acts,
    animate_by,
    dot_cloud,
    lines_by,
    pct,
    sport,
    stat,
    x_axis,
    y_axis,
):
    # The chart: every view is drawn by the browser from the frames table (one frame when nothing is animated), so moving
    # the scrubber or playing only filters rows there, with no round trip to the notebook. The frame on screen is a Vega
    # parameter bound to the page's #live-win input, which the scrubber and the Play button write.
    CLIENT_SPEC = None
    LIVE_INFO = {"live": False}
    # a new chart for each run of this cell: a pick made on an earlier one no longer points at anything on screen
    CHART_GEN = time.time_ns()
    _x, _ym, _g = x_axis.value, y_axis.value, lines_by.value
    if not AXES_CLASH and FRAMES.height:
        _F = len(FRAME_LABELS)
        _i0 = FRAME_NOW["i"] if (FRAME_NOW["i"] is not None and 0 <= FRAME_NOW["i"] < _F) else FRAME_DEFAULT
        _len_axis = _x in ("len_time", "len_dist")
        _speedy = sport.value in SPEED_SPORT
        _k = "v" if _ym == "x" else _ym
        _rev = _k == "v" and not _speedy
        # raw value -> what the y axis shows, as a Vega expression and in Python
        _YX = {"p": "({0})", "g": "({0})*100", "h": "({0})", "a": "({0})*3600*3.28084", "c": "({0})", "k": "({0})", "l": "({0})*39.3701",
               "v": "({0})*2.23694" if _speedy else "26.8224/({0})"}[_k]
        _yv = lambda f: _YX.format(f)
        _ypy = {"p": lambda s: s, "g": lambda s: s * 100, "h": lambda s: s, "a": lambda s: s * 3600 * 3.28084, "c": lambda s: s, "k": lambda s: s,
                "l": lambda s: s * 39.3701, "v": (lambda s: s * 2.23694) if _speedy else (lambda s: 26.8224 / s)}[_k]
        _lab = lambda e: (f"(floor({e}) + ':' + (round(({e} % 1) * 60) < 10 ? '0' : '') + round(({e} % 1) * 60))" if _rev else f"format({e}, ',.1f')")
        _qn = lambda p: f"q{round(p, 6):g}"  # percent -> column

        # what is drawn: the spread of activities (ribbons, median, Best), or one line per series of the chosen statistic
        _none = _g == "none"
        if _none:
            _inner = sorted(round(v * 100, 6) for v in SERIES_Q.values() if 0 <= v < 1)
            _qcols = [_qn(p) for p in _inner]
        else:
            _qv = 1.0 if stat.value == "Best" else 0.5 if stat.value.startswith("Typical") else pct.value / 100
            _vcol = "best" if _qv == 1.0 else _qn(_qv * 100)
            _qcols = [] if _qv == 1.0 else [_vcol]
        _rows = FRAMES.select(["f", "s", "xb", "n", "best", "aid", "t", "src", *[c for c in _qcols if c in FRAMES.columns]]).with_columns(
            [pl.col(c).round(4) for c in ["best", *[c for c in _qcols if c in FRAMES.columns]]]
        )

        # x: duration or distance on a log scale, or a value axis at its bins' centres (over the bins 3+ activities reach)
        _pace_expr = "floor(datum.value) + ':' + (round((datum.value % 1) * 60) < 10 ? '0' : '') + round((datum.value % 1) * 60)"
        if _len_axis:
            _basis = "time" if _x == "len_time" else "dist"
            _xcalc = "datum.xb"
            _xs = _rows["xb"]
            _xaxis = (alt.Axis(values=[5, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600, 7200, 18000],
                               labelExpr="datum.value < 60 ? datum.value + 's' : datum.value < 3600 ? round(datum.value / 60) + 'm' : (datum.value / 3600) + 'h'")
                      if _basis == "time" else
                      alt.Axis(values=[100, 400, 1000, 1609, 5000, 10000, 21097, 42195, 100000],
                               labelExpr="datum.value < 1000 ? datum.value + ' m' : datum.value == 1609 ? '1 mi' : datum.value == 21097 ? 'half' : datum.value == 42195 ? 'mara' : (datum.value / 1000) + ' km'"))
            _xdom_of = lambda r: [float(r["xb"].min()), float(r["xb"].max())]
            _X = lambda dom: alt.X("x:Q", title="Duration" if _basis == "time" else "Distance", scale=alt.Scale(type="log", domain=dom, nice=False), axis=_xaxis)
            _xl = ("datum.xb < 60 ? datum.xb + ' s' : datum.xb < 3600 ? format(datum.xb / 60, '.1f') + ' min' : format(datum.xb / 3600, '.2f') + ' h'"
                   if _basis == "time" else "datum.xb < 1000 ? datum.xb + ' m' : format(datum.xb / 1000, '.2f') + ' km'")
        else:
            _w = {"gb": 1, "pb": 20, "hb": 5, "cb": 5}.get(_x)
            if _x == "vb":
                _xcalc = "(datum.xb + 0.5) * 0.2 * 2.23694" if _speedy else "26.8224 / ((datum.xb + 0.5) * 0.2)"
                _xpy = (lambda b: (b + 0.5) * 0.2 * 2.23694) if _speedy else (lambda b: 26.8224 / ((b + 0.5) * 0.2))
            else:
                _xcalc = f"(datum.xb + 0.5) * {_w}"
                _xpy = lambda b: (b + 0.5) * _w
            def _xdom_of(r):
                """The bins 3+ activities reach (all bins when none do), at their centres."""
                _d = r.filter(pl.col("n") >= 3)["xb"]
                _xv = [_xpy(_d.min()), _xpy(_d.max())] if _d.len() else [_xpy(r["xb"].min()), _xpy(r["xb"].max())]
                if _x == "vb" and not _speedy:
                    _xv = [min(_xv), min(max(_xv), 30.0)]  # paces slower than 30:00 /mi are walking breaks
                return sorted(float(v) for v in _xv)
            _xtitle = {"gb": "Gradient (%)", "pb": "Power (W)", "hb": "Heart rate (bpm)", "cb": f"Cadence ({CADENCE_UNIT[sport.value]})",
                       "vb": "Speed (mph)" if _speedy else "Pace (min/mi)"}[_x]
            _X = lambda dom: alt.X("x:Q", title=_xtitle, scale=alt.Scale(domain=dom, nice=False, zero=False),
                       axis=alt.Axis(labelExpr=_pace_expr) if (_x == "vb" and not _speedy) else alt.Undefined)
            _xl = f"format({_xcalc}, ',.1f')" if not (_x == "vb" and not _speedy) else _lab(_xcalc)

        # y: one range for every frame (from frames with 3+ activities, trimmed), so motion on screen means change
        _yt = {"p": "Power (W)", "g": "Gradient (%)", "h": "Heart rate (bpm)", "a": "Vertical speed (ft/h)", "v": "Speed (mph)" if _speedy else "Pace (min/mi)",
               "c": f"Cadence ({CADENCE_UNIT[sport.value]})", "k": "Torque (N·m)", "l": "Step length (in)"}[_k]
        if _ym == "x":
            _yt = "Grade-adjusted pace (min/mi)"
        _yt = [_yt[:_yt.index(" (")], _yt[_yt.index(" (") + 1:]] if " (" in _yt else _yt
        def _ydom_of(r):
            """Best (and the median) over the x range 3+ activities reach, trimmed at 0.5 / 99.5 % and padded."""
            _dn = r.filter(pl.col("n") >= 3)
            if not _len_axis:
                _d = _dn["xb"]
                _dn = _dn.filter(pl.col("xb").is_between(_d.min(), _d.max())) if _d.len() else _dn
            _ysrc = [_dn["best"], _dn[_qn(50)]] if (_none and _qn(50) in _dn.columns) else [_dn["best"]] if _none else [_dn[_vcol]]
            _yy = pl.concat([_ypy(s.cast(pl.Float64)) for s in _ysrc]).drop_nulls()
            if not _yy.len():
                _yy = _ypy(r["best"].cast(pl.Float64)).drop_nulls()
            if not _yy.len():
                return [0.0, 1.0]
            _ylo, _yhi = _yy.quantile(0.005), _yy.quantile(0.995)
            _pad = (_yhi - _ylo) * 0.06 or 1
            return [float(_ylo - _pad), float(_yhi + _pad)]

        # An animation keeps one range for every frame, so motion on screen means change; otherwise (the Slide window)
        # each frame fits its own data, the browser picking the frame's range from a parameter as the window slides
        _fit_each = animate_by.value == "none" and _F > 1
        if _fit_each:
            _byf = _rows.partition_by("f", as_dict=True)
            _empty = _rows.head(0)
            _xd = alt.param(name="xdom", value=[_xdom_of(_byf.get((i,), _empty)) if (i,) in _byf else _xdom_of(_rows) for i in range(_F)])
            _yd = alt.param(name="ydom", value=[_ydom_of(_byf[(i,)]) if (i,) in _byf else _ydom_of(_rows) for i in range(_F)])
            _XDOM, _YDOM = alt.ExprRef("xdom[round(frame - 100000)]"), alt.ExprRef("ydom[round(frame - 100000)]")
        else:
            _XDOM, _YDOM = _xdom_of(_rows), _ydom_of(_rows)
        _X = _X(_XDOM)
        _yscale = alt.Scale(domain=_YDOM, reverse=_rev, nice=False)
        _yaxis = alt.Axis(labelExpr=_pace_expr) if _rev else alt.Undefined
        _Y = lambda f: alt.Y(f"{f}:Q", title=_yt, scale=_yscale, axis=_yaxis)

        _frame = alt.param(name="frame", value=100000 + _i0, bind=alt.BindDirect(element="#live-win"))
        # the point the drill-down shows (frame * 100000 + length; -1 for none), written by the page from the drill-down
        _kpick = alt.param(name="kpick", value=-1, bind=alt.BindDirect(element="#pick-key"))
        # a stretch dragged on the drill-down's charts ("x|y|label", raw y), written by the page
        _bdot = alt.param(name="bdot", value="", bind=alt.BindDirect(element="#brush-dot"))
        # the frames table is attached once, at the top of the chart, and every layer reads it from there: attached per
        # layer, marimo stores a copy per layer in the kernel's shared memory (64 MB), which a few renders fill
        _base = alt.Chart().transform_filter("datum.f == frame - 100000").transform_calculate(x=_xcalc)
        if _F > 1:
            # Between two frames (the frame is fractional while playing or dragging) the chart is a blend of both, weighted
            # by how far it has moved, so the curves glide. What belongs to one frame (the effort behind a Best point, the
            # frame a click picks) comes from the nearer one, and an x bin only one of them has appears at the halfway point.
            _fr = "(frame - 100000)"
            _vals = ["best", "n", *[c for c in _qcols if c in FRAMES.columns]]
            _base = (
                alt.Chart()
                .transform_filter(f"datum.f == floor({_fr}) || datum.f == floor({_fr}) + 1")
                .transform_calculate(w=f"datum.f == floor({_fr}) ? 1 - ({_fr} - floor({_fr})) : {_fr} - floor({_fr})")
                .transform_calculate(**{f"w_{c}": f"isValid(datum.{c}) ? datum.w : 0" for c in _vals},
                                     **{f"y_{c}": f"isValid(datum.{c}) ? datum.w * datum.{c} : 0" for c in _vals})
                .transform_aggregate(sw="sum(w)", near="argmax(w)", groupby=["s", "xb"],
                                     **{f"sw_{c}": f"sum(w_{c})" for c in _vals}, **{f"sy_{c}": f"sum(y_{c})" for c in _vals})
                .transform_filter(f"datum.sw > 0.5 || (datum.sw == 0.5 && datum.near.f == floor({_fr}))")
                .transform_calculate(**{c: f"datum.sw_{c} > 0 ? datum.sy_{c} / datum.sw_{c} : null" for c in _vals})
                .transform_calculate(n="round(datum.n)", aid="datum.near.aid", t="datum.near.t", src="datum.near.src", f="datum.near.f", x=_xcalc)
            )
        _names = acts.filter(pl.col("activity_id").is_in(_rows["aid"].unique().to_list())).select(
            pl.col("activity_id").alias("aid"), "name", pl.col("date").dt.strftime("%b %d, %Y").alias("date")
        )
        _pick = alt.selection_point(name="pick", on="click", fields=["series", "xb", "activity_id", "t", "src", "f"], empty=False)
        _xhov = alt.selection_point(name="xhov", nearest=True, on="pointermove", clear="view:mouseleave", encodings=["x"], empty=False)
        _layers = []

        def _cloud_layers(color):
            """The dot cloud, in one layer (its data attached once): faint, under the Best line, the picked dot solid."""
            if CLOUD is None or not CLOUD.height:
                return []
            _sn = "[" + ", ".join(json.dumps(n) for n in SERIES_ORDER) + "][datum.s]" if SERIES_ORDER else "''"
            _fr = "(frame - 100000)"
            _key = "'' + datum.activity_id" if dot_cloud.value == "strong" else "datum.activity_id + ':' + datum.xb"
            # x moves on the axis's own scale: log for durations and distances
            _lx, _ux = ("log(datum.x)", "exp(datum.sx / datum.sw)") if _len_axis else ("datum.x", "datum.sx / datum.sw")
            _cc = (
                alt.Chart(CLOUD)
                .transform_filter(f"datum.f == floor({_fr}) || datum.f == floor({_fr}) + 1")
                .transform_calculate(x=_xcalc, yv=_yv("datum.y"), key=_key,
                                     w=f"datum.f == floor({_fr}) ? 1 - ({_fr} - floor({_fr})) : {_fr} - floor({_fr})")
                .transform_calculate(wx=f"datum.w * {_lx}", wy="datum.w * datum.yv")
                .transform_aggregate(sw="sum(w)", sx="sum(wx)", sy="sum(wy)", near="argmax(w)", groupby=["key"])
                .transform_filter("datum.sw > 0")
                .transform_calculate(
                    x=_ux, yv="datum.sy / datum.sw", op="0.3 * datum.sw",
                    f="datum.near.f", s="datum.near.s", xb="datum.near.xb", y="datum.near.y", activity_id="datum.near.activity_id",
                    t="datum.near.t", src="datum.near.src", k1="datum.near.k1", k2="datum.near.k2",
                )
                .transform_calculate(
                    series="'Cloud'", sname=_sn, ylab=_lab(_yv("datum.y")), xl=_xl,
                    on="'c:' + datum.f + ':' + datum.xb + ':' + datum.activity_id == kpick", why=CLOUD_WHY,
                )
                .transform_lookup(lookup="activity_id", from_=alt.LookupData(CLOUD_NAMES, "activity_id", ["name", "date"]))
            )
            _cpick = alt.selection_point(name="cpick", on="click", fields=["series", "xb", "activity_id", "t", "src", "f"], empty=False)
            _tips = [alt.Tooltip("name:N", title="Activity"), alt.Tooltip("date:N", title="Date"), alt.Tooltip("xl:N", title="At"),
                     alt.Tooltip("ylab:N", title="Value"), alt.Tooltip("why:N", title="Why")]
            _on = "datum.on"
            return [
                _cc.mark_circle(clip=True, strokeWidth=1.5).encode(
                    x=_X, y=_Y("yv"),
                    color=alt.condition(_on, alt.value("#f97316"), color if color is not None else alt.value("#334155")),
                    opacity=alt.condition(_on, alt.value(1), alt.Opacity("op:Q", scale=None)),
                    size=alt.condition(_on, alt.value(110), alt.value(22)),
                    stroke=alt.condition(_on, alt.value("white"), alt.value(None)),
                    tooltip=_tips,
                ).add_params(_cpick)
            ]

        if _none:
            # ribbons, faintest out to Best and darker toward the median; every percentile edge shows as a change of shade
            _pairs = sorted([(100 - p, p) for p in _inner if p > 50 and (100 - p) in _inner], key=lambda t: -(t[1] - t[0]))
            _top = max((p for p in _inner if p > 50), default=None)
            _calc = {}
            if _top is not None:
                _calc.update(lo_top=_yv(f"datum.{_qn(_top)}"), hi_top=_yv("datum.best"))
            for _i, (_lo_p, _hi_p) in enumerate(_pairs):
                _calc.update({f"lo{_i}": _yv(f"datum.{_qn(_lo_p)}"), f"hi{_i}": _yv(f"datum.{_qn(_hi_p)}")})
            if 50 in _inner:
                _calc["yv"] = _yv(f"datum.{_qn(50)}")
            _stats = _base.transform_filter("datum.n >= 3").transform_calculate(**_calc) if _calc else None
            if _top is not None:
                _layers.append(_stats.mark_area(color="#4a78b5", opacity=0.1, clip=True).encode(x=_X, y=_Y("lo_top"), y2="hi_top:Q"))
            for _i, _pp in enumerate(_pairs):
                _layers.append(_stats.mark_area(color="#4a78b5", opacity=0.2, clip=True).encode(x=_X, y=_Y(f"lo{_i}"), y2=f"hi{_i}:Q"))
            if 50 in _inner:
                _layers.append(_stats.mark_line(color="#9e9e9e", strokeWidth=2.2, clip=True).encode(x=_X, y=_Y("yv")))
            # off the Best points, hovering anywhere shows every statistic at that x
            _tcalc = {"xl": _xl, "t_best": _lab(_yv("datum.best"))}
            _tips = [alt.Tooltip("xl:N", title=_X.title if isinstance(_X.title, str) else "x"), alt.Tooltip("t_best:N", title="Best")]
            for _p in sorted(_inner, reverse=True):
                _tcalc[f"t_{_qn(_p)}"] = _lab(_yv(f"datum.{_qn(_p)}"))
                _tips.append(alt.Tooltip(f"t_{_qn(_p)}:N", title="Median" if _p == 50 else "Lowest" if _p == 0 else f"p{_p:g}"))
            _tips.append(alt.Tooltip("n:Q", title="Activities"))
            _tstats = _base.transform_filter("datum.n >= 3").transform_calculate(**_tcalc)
            _layers.append(_tstats.mark_point(opacity=0).encode(x=_X).add_params(_xhov))
            _layers.append(_tstats.mark_rule(color="#94a3b8", strokeWidth=1).encode(x=_X, tooltip=_tips).transform_filter(_xhov))
            _layers += _cloud_layers(None)
            # Best: the activity that set it, clickable for the drill-down
            _best = (
                _base.transform_calculate(activity_id="datum.aid", yv=_yv("datum.best"), series="'Best'", ylab=_lab(_yv("datum.best")))
                .transform_lookup(lookup="aid", from_=alt.LookupData(_names, "aid", ["name", "date"]))
            )
            _layers.append(_best.mark_line(color="#08306b", strokeWidth=2.2, clip=True).encode(x=_X, y=_Y("yv")))
            _layers.append(_best.mark_circle(size=36, color="#08306b", clip=True).encode(
                x=_X, y=_Y("yv"), tooltip=[alt.Tooltip("name:N", title="Activity"), alt.Tooltip("date:N", title="Date"), alt.Tooltip("ylab:N", title="Best")]
            ).add_params(_pick))
            # the focused effort: a subtle crosshair to both axes, under its ring
            _focus = _best.transform_filter("datum.f * 100000 + datum.xb == kpick")
            _hair = dict(color="#f97316", strokeWidth=1, strokeDash=[3, 3], opacity=0.45)
            _layers.append(_focus.mark_rule(**_hair).encode(x=_X))
            _layers.append(_focus.mark_rule(**_hair).encode(y=_Y("yv")))
            _layers.append(_focus.mark_circle(size=320, filled=False, stroke="#f97316", strokeWidth=3).encode(x=_X, y=_Y("yv")))
        else:
            # one line per series, coloured as the lines mean (years, bands, periods)
            _sname = "[" + ", ".join(json.dumps(n) for n in SERIES_ORDER) + "][datum.s]"
            _color = alt.Color("series:N", sort=SERIES_ORDER, scale=alt.Scale(domain=SERIES_ORDER, range=SERIES_COLORS[:len(SERIES_ORDER)] or ["#1f4e9c"]),
                               legend=alt.Legend(title=LEGEND_TITLE, orient="top"))
            _lines = _base.transform_filter("datum.n >= 3" if _vcol != "best" else "true").transform_calculate(
                series=_sname, yv=_yv(f"datum.{_vcol}"), activity_id="datum.aid" if _vcol == "best" else "null", ylab=_lab(_yv(f"datum.{_vcol}"))
            )
            _layers.append(_lines.mark_line(strokeWidth=2.2, clip=True).encode(x=_X, y=_Y("yv"), color=_color, detail="series:N"))
            # hovering anywhere shows every line's value at that x
            _wide = _lines.transform_calculate(sk="'s' + datum.s").transform_pivot("sk", value="ylab", groupby=["xb", "x"]).transform_calculate(xl=_xl)
            _tips = [alt.Tooltip("xl:N", title="x")] + [alt.Tooltip(f"s{i}:N", title=n) for i, n in enumerate(SERIES_ORDER)]
            _layers.append(_wide.mark_point(opacity=0).encode(x=_X).add_params(_xhov))
            _layers.append(_wide.mark_rule(color="#94a3b8", strokeWidth=1).encode(x=_X, tooltip=_tips).transform_filter(_xhov))
            _layers += _cloud_layers(alt.Color("sname:N", scale=alt.Scale(domain=SERIES_ORDER, range=SERIES_COLORS[:len(SERIES_ORDER)] or ["#1f4e9c"]), legend=None))
            if _vcol == "best":
                _lb = _lines.transform_lookup(lookup="aid", from_=alt.LookupData(_names, "aid", ["name", "date"]))
                _layers.append(_lb.mark_circle(size=30, clip=True).encode(
                    x=_X, y=_Y("yv"), color=_color,
                    tooltip=[alt.Tooltip("series:N", title=LEGEND_TITLE or "Line"), alt.Tooltip("name:N", title="Activity"), alt.Tooltip("date:N", title="Date"), alt.Tooltip("ylab:N", title="Best")],
                ).add_params(_pick))
                _layers.append(_lb.mark_circle(size=320, filled=False, stroke="#f97316", strokeWidth=3).encode(x=_X, y=_Y("yv")).transform_filter(_pick))
        _bd = (
            alt.Chart(pl.DataFrame({"z": [0]})).transform_filter("bdot != ''")
            .transform_calculate(x="toNumber(split(bdot, '|')[0])", yr="toNumber(split(bdot, '|')[1])", label="split(bdot, '|')[2]")
            .transform_calculate(yv=_yv("datum.yr"), ylab=_lab(_yv("datum.yr")))
        )
        # its crosshair to both axes, as subtle as the focused effort's
        _bd_hair = dict(color="#f97316", strokeWidth=1, strokeDash=[2, 3], opacity=0.5)
        _layers.append(_bd.mark_rule(**_bd_hair).encode(x=_X))
        _layers.append(_bd.mark_rule(**_bd_hair).encode(y=_Y("yv")))
        _layers.append(
            _bd.mark_point(shape="diamond", filled=True, size=170, color="#f97316", stroke="white", strokeWidth=1.5, opacity=1, clip=True)
            .encode(x=_X, y=_Y("yv"), tooltip=[alt.Tooltip("label:N", title="Stretch"), alt.Tooltip("ylab:N", title="Value")])
        )
        if _F > 1:
            # the frame's name and activity count, written by the browser as frames change
            _fl = pl.DataFrame({"f": list(range(_F)), "fl": [f"{lab} · {a} activities" for lab, a in zip(FRAME_LABELS, FRAME_ACTS)]})
            _layers.append(
                alt.Chart(_fl).transform_filter("datum.f == round(frame - 100000)")
                .mark_text(align="left", baseline="top", fontSize=13, fontWeight="bold", color="#1f4e9c")
                .encode(x=alt.value(8), y=alt.value(4), text="fl:N")
            )
        # height from the page: it sizes the chart's box to the panel, so opening the drill-down only resizes it there
        CLIENT_SPEC = (
            alt.layer(*_layers, data=_rows)
            .add_params(_frame, _kpick, _bdot, *([_xd, _yd] if _fit_each else []))
            .properties(width="container", height="container")
            .configure_view(strokeWidth=0)
            .configure_axis(gridOpacity=0.3, domain=False, labelFontSize=11, titleFontSize=12, titleFontWeight="normal")
        )
        # the scrubber beside Play: what it moves through, the frame names and its two ends (a window by its end month)
        _end = lambda l: l.split(" – ")[-1][:3] + " " + l[-4:] if FRAME_KIND == "window" else l
        LIVE_INFO = {
            "live": _F > 1, "lo": 100000, "hi": 100000 + _F - 1, "step": 1, "end": 100000 + _i0, "labels": FRAME_LABELS,
            "anim": animate_by.value != "none",
            "title": {"window": "Window", "year": "Year"}.get(FRAME_KIND, DIMS.get(animate_by.value, "")),
            "ends": FRAME_ENDS or [_end(FRAME_LABELS[0]), _end(FRAME_LABELS[-1])],
        }
    return CHART_GEN, CLIENT_SPEC


@app.cell(hide_code=True)
def chart_view(
    AXES_CLASH,
    CHART_GEN,
    CLASH_MSG,
    CLIENT_SPEC,
    EMPTY_MSG,
    FRAMES,
    FRAME_NOW,
    PICK_GEN,
    chart_spec,
    set_frame,
    set_pick,
):
    _SEEN = {}


    def _on_chart(_value):
        # remember the clicked point in notebook state, so the chart can re-render shorter without losing it. A click
        # lands in one of two selections (a Best point, or a dot of the cloud): the one that changed is the click
        _sels = chart_view.selections or {}
        _hit = [n for n in ("cpick", "pick") if (_sels.get(n) or {}).get("series") and (_sels.get(n) or {}).get("xb") and _sels.get(n) != _SEEN.get(n)]
        _SEEN.update({n: _sels.get(n) for n in ("pick", "cpick")})
        if not _hit:
            return
        _sel = _sels[_hit[0]]
        if _sel.get("activity_id"):
            # the frame chart: the click names the effort and the frame it was in, and the notebook follows that frame
            if _sel.get("f") is not None and len(_sel["f"]):
                FRAME_NOW["i"] = int(_sel["f"][0])  # the frame's index, as the chart's rows carry it
                set_frame(FRAME_NOW["i"])
            _new = (_sel["series"][0], int(_sel["xb"][0]), int(_sel["activity_id"][0]), int(_sel["t"][0]), int(_sel["src"][0]))
        else:
            _new = (_sel["series"][0], _sel["xb"][0])
        PICK_GEN.update(gen=CHART_GEN, pick=_new, frame=FRAME_NOW["i"])
        # set without reading the pick back: a cell that reads it re-runs on every click, and this one holds the chart
        set_pick(_new)


    # the live chart (window slid in the browser) where it applies, else the notebook's chart
    _spec = None if (AXES_CLASH or not FRAMES.height) else CLIENT_SPEC if CLIENT_SPEC is not None else chart_spec
    # JSON, not marimo's default Arrow: Vega-Lite parses some fields in place (x as a number), and rows read from Arrow are
    # read-only, so those datasets failed to load and the chart drew nothing
    with alt.data_transformers.enable("marimo_json"):
        chart_view = None if _spec is None else mo.ui.altair_chart(_spec, chart_selection=False, legend_selection=False, on_change=_on_chart)
    # marimo filters the chart's data by every selection it sends (a hover, a click) to make the element's value. Our
    # selections name fields the chart computes (series, x), so that filter raised and the click never reached _on_chart;
    # the handler reads the selection itself, so there is nothing to filter
    if chart_view is not None:
        chart_view.dataframe = None
    mo.md(CLASH_MSG if AXES_CLASH else EMPTY_MSG) if _spec is None else chart_view
    return


@app.cell(hide_code=True)
def key_nav(
    CHART_GEN,
    FRAMES,
    FRAME_DEFAULT,
    FRAME_NOW,
    PICK_GEN,
    lines_by,
    set_pick,
):
    # Left and right arrows: the previous or next point along the Best line of the frame on screen (Lines: None), with or
    # without an animation (up and down move the frames, in the page). The page reports the key here
    # ("next:<point on screen>:<nonce>", the point as frame * 100000 + length, a cloud dot as c:frame:length:activity, -1
    # for none; a cloud dot steps from its length); the pick
    # moves from the point on screen (not from a pick still on its way there) to the neighbouring dot, whichever activity
    # set it, and the drill-down follows.
    def _step(v):
        _dir = v.split(":")[0]
        if _dir not in ("next", "prev") or lines_by.value != "none" or not FRAMES.height:
            return
        _fr = FRAME_NOW["i"] if FRAME_NOW["i"] is not None else FRAME_DEFAULT
        _best = FRAMES.filter((pl.col("f") == _fr) & pl.col("aid").is_not_null()).sort("xb")
        if not _best.height:
            return
        _rows = _best.rows(named=True)
        _parts = v.split(":")
        _sp = _parts[1:-1]
        _shown = int(_sp[0]) if len(_sp) == 1 and _sp[0].lstrip("-").isdigit() else -1
        _cur = None
        if len(_sp) == 4 and _sp[0] == "c" and _sp[1].isdigit() and int(_sp[1]) == _fr and _sp[2].isdigit():
            # from a cloud dot: the nearest Best points either side of its length
            _xb = int(_sp[2])
            _nxt = [i for i, r in enumerate(_rows) if (r["xb"] > _xb if _dir == "next" else r["xb"] < _xb)]
            if not _nxt:
                return
            _r = _rows[_nxt[0] if _dir == "next" else _nxt[-1]]
            _new = ("Best", int(_r["xb"]), int(_r["aid"]), int(_r["t"]), int(_r["src"]))
            PICK_GEN.update(gen=CHART_GEN, pick=_new, frame=_fr)
            set_pick(_new)
            return
        if _shown >= 0 and _shown // 100000 == _fr:
            _row = next((r for r in _rows if r["xb"] == _shown % 100000), None)
            _cur = None if _row is None else ("Best", _row["xb"], _row["aid"], _row["t"], _row["src"])
        if _cur is None:
            _i = 0 if _dir == "next" else len(_rows) - 1
        else:
            _at = next((i for i, r in enumerate(_rows) if r["xb"] == _cur[1]), None)
            if _at is None:
                _i = 0 if _dir == "next" else len(_rows) - 1
            else:
                _i = _at + (1 if _dir == "next" else -1)
                if not 0 <= _i < len(_rows):
                    return  # already at the end of the curve that way
        _r = _rows[_i]
        _new = ("Best", int(_r["xb"]), int(_r["aid"]), int(_r["t"]), int(_r["src"]))
        PICK_GEN.update(gen=CHART_GEN, pick=_new, frame=_fr)
        set_pick(_new)


    key_report = mo.ui.text(value="", label="", on_change=_step)
    key_report
    return


@app.cell(hide_code=True)
def brush_nav(
    CHART_GEN,
    PICK_GEN,
    SPEED_SPORT,
    ec,
    scanner,
    set_bdot,
    sport,
    x_axis,
    y_axis,
):
    # A stretch dragged on the drill-down's charts, as a point on the main chart: the page reports the dragged range
    # here ("<start min>,<end min>:<nonce>", or "clear:<nonce>"), and this measures exactly that window of the open
    # activity with the engine (the same way the curve measures every window), placed the way the chart's x axis reads:
    # its length on a duration axis, the distance it covers on a distance axis, its own value on a value axis.
    def _on_brush(v):
        _r = v.split(":")[0]
        if _r == "clear":
            set_bdot(None)
            return
        try:
            _a, _b = sorted(float(x) for x in _r.split(","))
        except ValueError:
            return
        _pk = PICK_GEN["pick"] if PICK_GEN["gen"] == CHART_GEN else None
        if not _pk or len(_pk) != 5:
            return
        _t0, _t1 = int(round(_a * 60)), int(round(_b * 60))
        if _t1 - _t0 < 5:
            set_bdot(None)
            return
        _P = scanner()
        _at = {int(a): i for i, a in enumerate(_P.activity_ids)}.get(int(_pk[2]))
        if _at is None:
            return
        _one = np.array([_at], dtype=np.int64)
        _start = [("t0", float(_t0), float(_t0))]  # the window that starts exactly where the drag did
        _w = _P.compute("time", np.array([float(_t1 - _t0)]), 0.0, _start, activities=_one)[0][0, 0]
        _x, _mi = x_axis.value, ec.METRICS.index(y_axis.value)
        if _x == "len_time":
            _xv, _yv = float(_t1 - _t0), float(_w[_mi])
        elif _x == "len_dist":
            _d = float(_w[ec.METRICS.index("v")]) * (_t1 - _t0)
            _wd = _P.compute("dist", np.array([_d]), 0.0, _start, activities=_one)[0][0, 0] if np.isfinite(_d) and _d > 0 else None
            _xv, _yv = _d, (float(_wd[_mi]) if _wd is not None else float("nan"))
        else:
            _speedy = sport.value in SPEED_SPORT
            _val = float(_w[ec.METRICS.index({"gb": "g", "vb": "v", "pb": "p", "hb": "h", "cb": "pc"}[_x])])
            _xv = {"gb": _val * 100, "vb": (_val * 2.23694 if _speedy else (26.8224 / _val if _val > 0 else float("nan")))}.get(_x, _val)
            _yv = float(_w[_mi])
        _n = _t1 - _t0
        _len = f"{_n // 3600}:{_n % 3600 // 60:02d}:{_n % 60:02d}" if _n >= 3600 else f"{_n // 60}:{_n % 60:02d}"
        set_bdot({"gen": CHART_GEN, "aid": int(_pk[2]), "x": _xv, "y": _yv,
                  "label": f"Dragged stretch: {_len} from {_t0 // 60}:{_t0 % 60:02d} in"})


    brush_report = mo.ui.text(value="", label="", on_change=_on_brush)
    brush_report
    return


@app.cell(hide_code=True)
def brush_dot_cell(CHART_GEN, get_bdot, get_pick, x_axis):
    # The dragged stretch's point, for the page to hand to the main chart ("x|y|label|log or lin", the last for how the
    # page glides it to the next position: along a log axis or a linear one); empty once the drill-down shows
    # another activity or the chart is rebuilt, or when the metric is not defined for that stretch
    _b, _p = get_bdot(), get_pick()
    _ok = (_b is not None and _b["gen"] == CHART_GEN and _p is not None and len(_p) == 5 and _p[2] == _b["aid"]
           and np.isfinite(_b["x"]) and np.isfinite(_b["y"]))
    _scale = "log" if x_axis.value in ("len_time", "len_dist") else "lin"
    brush_dot = mo.Html(f'<span data-bdot="{_b["x"]:.6g}|{_b["y"]:.6g}|{_b["label"]}|{_scale}"></span>' if _ok else '<span data-bdot=""></span>')
    brush_dot
    return


@app.cell(hide_code=True)
def headline(
    CADENCE_UNIT,
    DATE_WINDOW,
    DIMS,
    FILTER_SUMMARY,
    FRAME_ACTS,
    FRAME_I,
    FRAME_KIND,
    FRAME_LABELS,
    FRAME_NOTE,
    METRICS,
    SPEED_SPORT,
    WINDOW_TEXT,
    curve,
    lines_by,
    sport,
    x_axis,
    y_axis,
):
    # live numbers for the dashboard view's header cards
    _speedy = sport.value in SPEED_SPORT
    _ym = "v" if y_axis.value == "x" else y_axis.value


    def _fmt_y(y):
        if y is None:
            return "–"
        if _ym == "v":
            return f"{y * 2.23694:.1f} mph" if _speedy else f"{int(26.8224 / y)}:{int(26.8224 / y % 1 * 60):02d} /mi"
        return {"p": f"{y:.0f} W", "g": f"{y * 100:.1f}%", "h": f"{y:.0f} bpm", "a": f"{y * 3600 * 3.28084:,.0f} ft/h",
                "c": f"{y:.0f} {CADENCE_UNIT[sport.value]}", "k": f"{y:.0f} N·m", "l": f"{y * 39.3701:.0f} in"}[_ym]


    def _at(series, xb):
        _r = curve.filter((pl.col("series") == series) & (pl.col("xb") == xb))
        return _fmt_y(_r["y"][0]) if _r.height else "–"


    _marks = [(300, "5 min"), (1200, "20 min"), (3600, "1 h")] if x_axis.value == "len_time" else [(1609, "1 mi"), (5000, "5 km"), (21097, "Half")] if x_axis.value == "len_dist" else []
    _n = FRAME_ACTS[FRAME_I] if FRAME_ACTS else (curve["n"].max() if curve.height else 0)
    headline = {
        "title": f"{sport.value}: {METRICS[y_axis.value].lower()} by {DIMS[x_axis.value].lower()}" + (f" · {WINDOW_TEXT}" if WINDOW_TEXT else ""),
        # the top bar's second line: dates, then the filters that narrow which windows count
        "window": (FRAME_LABELS[FRAME_I] if FRAME_KIND != "none" and FRAME_LABELS else f"{DATE_WINDOW[0]:%b %d, %Y} – {DATE_WINDOW[1]:%b %d, %Y}")
        + (f" · {FRAME_NOTE}" if FRAME_NOTE else "") + (f" · only {FILTER_SUMMARY}" if FILTER_SUMMARY else ""),
        "activities": int(_n or 0),
        "cards": [{"label": f"Best · {lab}", "value": _at("Best", xb), "sub": f"median {_at('Median', xb)}" if lines_by.value == "none" else ""} for xb, lab in _marks],
    }
    return


@app.cell(hide_code=True)
def drill_cache():
    DRILL_CACHE = {}  # drill-downs already built, keyed by the picked point
    return (DRILL_CACHE,)


@app.cell(hide_code=True)
def drill_close(set_pick):
    drill_close = mo.ui.button(label="Close", on_click=lambda _: set_pick(None))
    return (drill_close,)


@app.cell(hide_code=True)
def drill(
    CADENCE_SCALE,
    CADENCE_UNIT,
    CHART_GEN,
    DASH,
    DRILL_CACHE,
    FRAME_I,
    HAS_STRAVA,
    HRQ,
    PICK_GEN,
    SPEED_SPORT,
    acts,
    curve,
    drill_close,
    fmt_len,
    get_pick,
    length,
    q,
    set_pick,
    sport,
    warm_scanner,
    x_axis,
):
    class VegaLiteSpec:
        """A finished Vega-Lite spec (a dict) shown as a chart, skipping Altair's copying and validation."""

        MIME = "application/vnd.vegalite.v%s+json" % alt.SCHEMA_VERSION.lstrip("v").split(".")[0]

        def __init__(self, spec):
            self.spec = spec

        def _mime_(self):
            return self.MIME, json.dumps(self.spec, default=str)


    def _hms(seconds):
        """Elapsed time as 4:05 or 1:02:03."""
        seconds = int(round(seconds or 0))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


    # a pick made on an earlier chart (the view has changed since) points at nothing on screen: drop it, closing the drawer
    if get_pick() is not None and PICK_GEN["gen"] != CHART_GEN:
        set_pick(None)
    # a click on a chart point picks it; only best-effort points name an activity
    _pk = get_pick() if PICK_GEN["gen"] == CHART_GEN else None
    _sel = {"series": [_pk[0]], "xb": [_pk[1]]} if _pk else None
    _r, _msg = None, mo.md("_Click a best-effort point on the chart to see where it happened._")
    if _sel and _sel.get("series") and _sel.get("xb"):
        _m = curve.filter((pl.col("series") == _sel["series"][0]) & (pl.col("xb") == _sel["xb"][0]) & pl.col("activity_id").is_not_null())
        if _m.height:
            _r = _m.row(0, named=True)
        else:
            _msg = mo.md("_That point summarises many activities, so there is no single effort to show: click a point on the Best line._")
    if _pk and len(_pk) == 5:
        # a click on the live chart names the effort itself (its window may differ from the notebook's curve)
        _p = _pk
        _r = {"series": _p[0], "xb": _p[1], "activity_id": _p[2], "t_start": _p[3], "src_xb": _p[4]}
    _key = None if _r is None else (sport.value, x_axis.value, length.value, _r["series"], _r["xb"], _r["activity_id"], _r["t_start"])
    # the dashboard view shows its drill-down drawer while this is set, and reopens it when it changes
    drill_key = "" if _key is None else "|".join(map(str, _key))
    if _r is None:
        _out = _msg
    elif (_key, tuple(DASH.values())) in DRILL_CACHE:
        _out = DRILL_CACHE[(_key, tuple(DASH.values()))]  # hovering re-runs this cell; an unchanged pick costs nothing
    else:
        warm_scanner(sport.value)
        _aid, _t0 = int(_r["activity_id"]), int(_r["t_start"])
        # where the monotone envelope lifted a point, the effort is the longer window that set it
        _len = int(_r["src_xb"]) if _r.get("src_xb") is not None else int(_r["xb"])
        _basis, _n = ("time", _len) if x_axis.value == "len_time" else ("dist", _len) if x_axis.value == "len_dist" else length.value
        _rec = q(f"""
            select epoch(timestamp)::bigint as ts, position_lat as lat, position_long as lon, distance, altitude * 3.28084 as elev_ft,
                   speed, heart_rate as hr, power,
                   case when cadence > 0 then (cadence + coalesce(fractional_cadence, 0)) * {CADENCE_SCALE[sport.value]} end as cad
            from garmin.fit_record where activity_id = {_aid}
            qualify row_number() over (partition by epoch(timestamp)::bigint order by distance desc nulls last) = 1
            order by ts
        """)
        _rec = _rec.with_columns(t=pl.col("ts") - pl.col("ts").min(), minute=(pl.col("ts") - pl.col("ts").min()) / 60)
        # gradient at each moment: climb over distance across about 30 s around it (in %), from the smoothed altitude
        _am = (pl.col("elev_ft") / 3.28084).rolling_mean(15, center=True, min_samples=5)
        _dd = pl.col("distance").shift(-15) - pl.col("distance").shift(15)
        _rec = _rec.with_columns(_am.alias("_am")).with_columns(
            grade=pl.when(_dd > 10).then(((pl.col("_am").shift(-15) - pl.col("_am").shift(15)) / _dd * 100).clip(-40, 40))
        ).drop("_am")
        # the window, located the same way the cube measured it: by seconds, or by distance from its start
        if _basis == "time":
            _t1 = _t0 + _n
        else:
            _d0 = _rec.filter(pl.col("t") >= _t0)["distance"].drop_nulls()
            _after = _rec.filter(pl.col("distance") >= (_d0[0] if _d0.len() else 0) + _n)
            _t1 = int(_after["t"][0]) if _after.height else int(_rec["t"].max())
        _win = _rec.filter(pl.col("t").is_between(_t0, _t1))
        _act = acts.filter(pl.col("activity_id") == _aid).row(0, named=True)
        _speedy = sport.value in SPEED_SPORT
        _mean_speed = (_win["distance"].max() - _win["distance"].min()) / max(_t1 - _t0, 1) if _win.height else None
        _gain = _win["elev_ft"].diff().clip(lower_bound=0).sum() if _win.height else 0
        _facts = [
            f"{fmt_len(_basis, _n)} at {_hms(_t0)}–{_hms(_t1)} in",
            (f"{_mean_speed * 2.23694:.1f} mph" if _speedy else f"{int(26.8224 / _mean_speed)}:{int(26.8224 / _mean_speed % 1 * 60):02d} /mi") if _mean_speed else None,
            (f"{_win['hr'].mean():.0f} bpm" + (" (strap" if HRQ.get(_aid, {}).get("strap") else " (wrist") + (": " + ", ".join(HRQ[_aid]["reasons"]) if HRQ.get(_aid, {}).get("reasons") else "") + ")") if _win["hr"].drop_nulls().len() else None,
            f"{_win['power'].mean():.0f} W" if _win["power"].drop_nulls().len() else None,
            f"{_win['cad'].mean():.0f} {CADENCE_UNIT[sport.value]}" if _win["cad"].drop_nulls().len() else None,
            f"+{_gain:,.0f} ft" if _gain else None,
        ]
        # links to the activity itself: Garmin by its id; Strava by the activity that starts within two minutes of it
        _links = [f'<a href="https://connect.garmin.com/modern/activity/{_aid}" target="_blank" rel="noopener">Garmin ↗</a>']
        if HAS_STRAVA:
            _sv = q(f"""
                select s.id from strava.activity s, garmin.activity g
                where g.activity_id = {_aid} and s.deleted_at is null
                  and abs(epoch(s.start_date) - epoch(g.start_time_gmt)) <= 120
                order by abs(epoch(s.start_date) - epoch(g.start_time_gmt)) limit 1
            """)
            if _sv.height:
                _links.append(f'<a href="https://www.strava.com/activities/{_sv["id"][0]}" target="_blank" rel="noopener">Strava ↗</a>')
        _pm = f"c:{FRAME_I}:{int(_r['xb'])}:{int(_r['activity_id'])}" if _r["series"] == "Cloud" else FRAME_I * 100000 + int(_r["xb"])
        _pick_mark = f'<span hidden data-pick="{_pm}"></span>'
        _header = mo.md(_pick_mark + f"**{_act['name']}** · {_act['date']:%A, %B %d, %Y} · " + " · ".join(f for f in _facts if f) + " · " + " · ".join(_links))

        # one Altair chart for streams + map, so brushing links them in the browser with no kernel round-trips
        _speedy = sport.value in SPEED_SPORT
        # recording gaps are drawn the way the curves count them: no movement and no power while stopped (pace at the
        # slow edge, power at zero), heart rate and elevation carried straight across (heart rate for gaps up to 10 minutes)
        _st = _rec.with_columns(
            pace=pl.when(pl.col("speed") > 0.5).then(pl.col("speed") * 2.23694 if _speedy else 26.8224 / pl.col("speed")),
        ).gather_every(max(1, _rec.height // 700))
        _powered = _rec["power"].drop_nulls().len() > 0
        _cadenced = _rec["cad"].drop_nulls().len() > 0
        _fill = []
        _r = _rec.select("t", "elev_ft", "hr").to_dicts()
        for _k in range(len(_r) - 1):
            _a, _b = _r[_k], _r[_k + 1]
            if _b["t"] - _a["t"] <= 10:
                continue
            # points every few seconds across the gap (at most 120), carrying the bridged values: the charts draw the
            # pause as before, and a press or hover inside it lands on the moment under the pointer, not a pause's end
            _gap = _b["t"] - _a["t"]
            _k = min(120, max(2, _gap // 5))
            for _tt in sorted({_a["t"] + 1, _b["t"] - 1, *(_a["t"] + round(_gap * (i + 1) / (_k + 1)) for i in range(_k))}):
                _f = (_tt - _a["t"]) / (_b["t"] - _a["t"])
                _row = {"t": _tt, "minute": _tt / 60, "speed": 0.0, "power": 0.0 if _powered else None, "cad": 0.0 if _cadenced else None}
                for _c in ("elev_ft", "hr"):
                    _ok = _a[_c] is not None and _b[_c] is not None and (_c != "hr" or _b["t"] - _a["t"] <= 600)
                    _row[_c] = _a[_c] + (_b[_c] - _a[_c]) * _f if _ok else None
                _fill.append(_row)
        if _fill:
            _st = pl.concat([_st, pl.DataFrame(_fill)], how="diagonal_relaxed").sort("t")
        if _powered:
            _st = _st.with_columns(pl.col("power").fill_null(0))
        if _cadenced:
            _st = _st.with_columns(pl.col("cad").fill_null(0))  # coasting and stops: no pedaling or stepping
        # axis ranges from moving seconds only; stopped seconds then sit on the slow edge (pace) or at zero (speed)
        _dom = {"pace": (_st["pace"].quantile(0.01), _st["pace"].quantile(0.99))}
        _st = _st.with_columns(pace=pl.col("pace").fill_null(0.0 if _speedy else _dom["pace"][1]))
        _st = _st.with_columns(elapsed=pl.col("t").map_elements(_hms, return_dtype=pl.String))
        _has_gps = _st.drop_nulls(["lat", "lon"]).height > 10
        # sized for a full-width marimo page (~1,450 px): one Vega view (needed for linked brushing) cannot stretch to its container
        _MAP = DASH["map"]  # px, square
        _STREAM_W = DASH["streams_w"]
        if _has_gps:
            # Web Mercator world pixels at a zoom where the route fills the square; basemap tiles share that grid
            def _wx(lon, z):
                return (lon + 180) / 360 * 256 * 2 ** z

            def _wy(lat, z):
                _r = np.radians(lat)
                return (1 - np.log(np.tan(_r) + 1 / np.cos(_r)) / np.pi) / 2 * 256 * 2 ** z

            _g = _st.drop_nulls(["lat", "lon"])
            _lon, _lat = _g["lon"].to_numpy(), _g["lat"].to_numpy()
            _z = 17
            while _z > 3 and max(_wx(_lon.max(), _z) - _wx(_lon.min(), _z), _wy(_lat.min(), _z) - _wy(_lat.max(), _z)) > _MAP * 0.85:
                _z -= 1
            _st = _st.with_columns(
                px=pl.when(pl.col("lon").is_not_null()).then((pl.col("lon") + 180) / 360 * 256 * 2 ** _z),
                py=pl.when(pl.col("lat").is_not_null()).then(
                    (1 - (pl.col("lat").radians().tan() + 1 / pl.col("lat").radians().cos()).log() / np.pi) / 2 * 256 * 2 ** _z
                ),
            )
            _cx = (_st["px"].min() + _st["px"].max()) / 2
            _cy = (_st["py"].min() + _st["py"].max()) / 2
            _h = _MAP / 2
            _xd, _yd = [_cx - _h, _cx + _h], [_cy - _h, _cy + _h]
            _tiles = pl.DataFrame(
                [
                    {"tx": tx * 256, "ty": ty * 256, "url": f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{_z}/{ty}/{tx}"}
                    for tx in range(int(_xd[0] // 256), int(_xd[1] // 256) + 1)
                    for ty in range(int(_yd[0] // 256), int(_yd[1] // 256) + 1)
                ]
            )

        # The chart as a plain Vega-Lite spec. Built through Altair (objects, deep copies, serialisation) it took about
        # 1.5 s a click; as dicts it is a few milliseconds. One view for streams and map, so brushing and hovering link
        # them in the browser with no kernel round trips.
        _S = {"name": "streams"}
        _datasets = {}

        def _inline(name, rows):
            _datasets[name] = rows
            return {"name": name}

        _specs = [s for s in [("elev_ft", "Elevation (ft)", "#90be6d"), ("pace", "Speed (mph)" if _speedy else "Pace (min/mi)", "#277da1"), ("hr", "Heart rate", "#f94144"), ("power", "Power (W)", "#8d5a97"), ("cad", f"Cadence ({CADENCE_UNIT[sport.value]})", "#f3722c")] if _st[s[0]].drop_nulls().len() >= 10]
        # hover linking: the moment under the pointer, on a stream (nearest in time) or on the map (nearest point)
        _hover = [{"param": "hov", "empty": False}] if _specs else []
        if _has_gps:
            _hover.append({"param": "mhov", "empty": False})
        _HOVER = _hover[0] if len(_hover) == 1 else {"or": _hover}
        # elapsed time as 45m / 1h 20m; only the bottom panel carries the axis labels and title
        _hm = "datum.value < 60 ? round(datum.value) + 'm' : floor(datum.value / 60) + 'h' + (round(datum.value % 60) ? ' ' + round(datum.value % 60) + 'm' : '')"
        _tmax = float(_st["minute"].max())
        # the streams share the map's height, however many of them there are
        _ph = max(60, (_MAP - 42 - 6 * (len(_specs) - 1)) // max(len(_specs), 1))
        _pw = _STREAM_W if _has_gps else DASH["plain_w"]
        _shade = {"data": _inline("shade", [{"a": _t0 / 60, "b": _t1 / 60}]), "mark": {"type": "rect", "color": "#f97316", "opacity": 0.15},
                  "encoding": {"x": {"field": "a", "type": "quantitative"}, "x2": {"field": "b"}}}
        # the window dragged on the streams: one window, the same in every panel (the page runs the drag and sets "win",
        # "start,end" in elapsed minutes, so dragging inside it moves it and dragging elsewhere draws a new one)
        _WIN = "win != '' && datum.minute >= toNumber(split(win, ',')[0]) && datum.minute <= toNumber(split(win, ',')[1])"
        _window = {"data": _inline("winrow", [{"z": 0}]), "mark": {"type": "rect", "color": "#1f4e9c", "opacity": 0.14},
                   "transform": [{"filter": "win != ''"}, {"calculate": "toNumber(split(win, ',')[0])", "as": "wa"}, {"calculate": "toNumber(split(win, ',')[1])", "as": "wb"}],
                   "encoding": {"x": {"field": "wa", "type": "quantitative"}, "x2": {"field": "wb"}}}
        _ix_views, _panels = [], []
        # one tooltip for every panel: all the streams at the moment under the pointer, wherever it is in a panel (the
        # hover is by x, and the tooltip rides on the hover's nearest-point areas)
        _tips = [{"field": "elapsed", "title": "Elapsed", "type": "nominal"}]
        _tip_calc = []
        for _col, _title, _color in _specs:
            if _col == "pace" and not _speedy:
                _tip_calc.append({"calculate": "datum.pace == null ? null : floor(datum.pace) + ':' + (floor((datum.pace % 1) * 60) < 10 ? '0' : '') + floor((datum.pace % 1) * 60)", "as": "pace_l"})
                _tips.append({"field": "pace_l", "title": "Pace (min/mi)", "type": "nominal"})
            else:
                _tips.append({"field": _col, "title": _title, "type": "quantitative", "format": ",.1f" if _col == "pace" else ",.0f"})
            if _col == "elev_ft" and "grade" in _st.columns and _st["grade"].drop_nulls().len() >= 10:
                _tips.append({"field": "grade", "title": "Gradient (%)", "type": "quantitative", "format": ".1f"})
        for _i, (_col, _title, _color) in enumerate(_specs):
            _last = _i == len(_specs) - 1
            _lo, _hi = _dom.get(_col) or (_st[_col].quantile(0.01), _st[_col].quantile(0.99))
            _x = {"axis": {"labelExpr": _hm, "labels": _last, "ticks": _last}, "field": "minute", "scale": {"domain": [0, _tmax], "nice": False},
                  "title": "Elapsed time" if _last else None, "type": "quantitative"}
            _y = {"field": _col, "scale": {"clamp": True, "domain": [_lo, _hi], "reverse": _col == "pace" and not _speedy},
                  "title": [_title[:_title.index(" (")], _title[_title.index(" (") + 1:]] if " (" in _title else _title, "type": "quantitative"}
            # one invisible layer per panel carries both selections (drag brush and hover)
            _layers = [
                _shade,
                _window,
                {"data": _S, "mark": {"type": "line", "color": _color, "strokeWidth": 1.2}, "encoding": {"x": _x, "y": _y}},
                {"data": _S, "mark": {"type": "circle", "opacity": 0, "size": 80}, "encoding": {"x": _x, "y": _y, "tooltip": _tips}, "name": f"ix_{_col}",
                 **({"transform": _tip_calc} if _tip_calc else {})},
                {"data": _S, "mark": {"type": "rule", "color": "#111827", "strokeWidth": 1}, "encoding": {"x": _x}, "transform": [{"filter": _HOVER}]},
                {"data": _S, "mark": {"type": "circle", "color": "#111827", "size": 45}, "transform": [{"filter": _HOVER}], "encoding": {"x": _x, "y": _y}},
            ]
            _ix_views.append(f"ix_{_col}")
            if _has_gps:
                # moments inside a box drawn on the map light up here
                _layers.append({"data": _S, "mark": {"type": "circle", "color": "#e63946", "size": 14}, "encoding": {"x": _x, "y": _y}, "transform": [{"filter": {"param": "mapbrush", "empty": False}}]})
            _panel = {"layer": _layers}
            if _col == "elev_ft" and _st["grade"].drop_nulls().len() >= 10:
                # gradient behind the elevation line, on its own axis at the right: red uphill, green downhill
                _gm = float(max(5.0, min(40.0, _st["grade"].abs().max() or 5.0)))  # the steepest shown, so no area is cut off
                _gy = lambda f: {"axis": {"grid": False, "orient": "right"}, "field": f, "scale": {"clamp": True, "domain": [-_gm, _gm]}, "title": ["Gradient", "(%)"], "type": "quantitative"}
                _panel = {"layer": [
                    {"data": _S, "layer": [
                        {"mark": {"type": "area", "color": "#e76f51", "opacity": 0.35, "clip": True}, "encoding": {"x": _x, "y": _gy("gu"), "y2": {"datum": 0}}, "transform": [{"calculate": "datum.grade > 0 ? datum.grade : 0", "as": "gu"}]},
                        {"mark": {"type": "area", "color": "#2a9d8f", "opacity": 0.35, "clip": True}, "encoding": {"x": _x, "y": _gy("gd"), "y2": {"datum": 0}}, "transform": [{"calculate": "datum.grade < 0 ? datum.grade : 0", "as": "gd"}]},
                    ]},
                    _panel,
                ], "resolve": {"scale": {"y": "independent"}}}
            _panel.update(width=_pw, height=_ph)
            _panels.append(_panel)
        _params = [{"name": "win", "value": "", "bind": {"element": "#brush-win"}}]
        if _ix_views:
            _params += [
                {"name": "hov", "select": {"type": "point", "clear": "pointerout", "encodings": ["x"], "nearest": True, "on": "pointerover"}, "views": _ix_views},
            ]
            # where a drag starts, in elapsed minutes, written to the page (#bs) for the main chart's point (the page adds
            # how far the pointer moved); one per panel, since a bound selection over several views only listens to the first
            for _v in _ix_views:
                _params += [
                    {"name": f"bs_{_v}", "select": {"type": "point", "encodings": ["x"], "nearest": True, "on": "pointerdown", "clear": False}, "bind": {"element": "#bs"}, "views": [_v]},
                    {"name": f"hm_{_v}", "select": {"type": "point", "encodings": ["x"], "nearest": True, "on": "pointermove", "clear": "pointerout"}, "bind": {"element": "#hm"}, "views": [_v]},
                ]
        _body = {"vconcat": _panels, "spacing": 6} if _panels else None

        if _has_gps:
            _mx = lambda f: {"axis": None, "field": f, "scale": {"domain": _xd, "nice": False, "zero": False}, "type": "quantitative"}
            _my = lambda f: {"axis": None, "field": f, "scale": {"domain": _yd, "nice": False, "reverse": True, "zero": False}, "type": "quantitative"}
            _MX, _MY = _mx("px"), _my("py")
            _scale = _MAP / (_xd[1] - _xd[0])
            _order = {"field": "t", "type": "quantitative"}
            # hovering the map: where the route passes a spot more than once (out and back, laps) the nearest recorded
            # point could be any pass, so the linked moment jumped between them. One point per 6-pixel cell instead,
            # always the same pass: the one inside the highlighted effort, else the earliest.
            _cell = 6 / _scale
            _hover_pts = (
                _st.drop_nulls(["px", "py"])
                .with_columns(cx=(pl.col("px") / _cell).floor(), cy=(pl.col("py") / _cell).floor(),
                              away=~pl.col("t").is_between(_t0, _t1))
                .sort("away", "t")
                .unique(subset=["cx", "cy"], keep="first")
                .select(pl.col("t"), pl.col("px").round(1), pl.col("py").round(1))
                .sort("t")
            )
            _map_layers = [
                {"data": _inline("tiles", _tiles.to_dicts()), "mark": {"type": "image", "align": "left", "baseline": "top", "clip": True, "height": 256 * _scale + 1, "width": 256 * _scale + 1},
                 "encoding": {"url": {"field": "url", "type": "nominal"}, "x": _mx("tx"), "y": _my("ty")}},
                {"data": _S, "mark": {"type": "line", "clip": True, "color": "#4a5568", "strokeWidth": 2.5}, "encoding": {"order": _order, "x": _MX, "y": _MY}},
                {"data": _S, "mark": {"type": "line", "clip": True, "color": "#f97316", "strokeWidth": 5}, "encoding": {"order": _order, "x": _MX, "y": _MY},
                 "transform": [{"filter": f"datum.t >= {_t0} && datum.t <= {_t1}"}]},
            ]
            if _ix_views:
                _map_layers.append({"data": _S, "mark": {"type": "circle", "clip": True, "color": "#e63946", "size": 22}, "encoding": {"x": _MX, "y": _MY},
                                    "transform": [{"filter": _WIN}]})
            _map_layers += [
                {"data": _inline("start", _st.drop_nulls("px").head(1).select("px", "py").to_dicts()), "mark": {"type": "circle", "clip": True, "color": "#2d6a4f", "size": 110, "stroke": "white", "strokeWidth": 2},
                 "encoding": {"x": _MX, "y": _MY}},
                {"data": _inline("maphover", _hover_pts.to_dicts()), "mark": {"type": "circle", "clip": True, "opacity": 0, "size": 80}, "encoding": {"x": _MX, "y": _MY}, "name": "map_ix"},
                {"data": _S, "mark": {"type": "circle", "clip": True, "color": "#111827", "size": 130, "stroke": "white", "strokeWidth": 2}, "encoding": {"x": _MX, "y": _MY},
                 "transform": [{"filter": _HOVER}]},
                {"data": _inline("credit", [{"x": _xd[1] - 4, "y": _yd[1] - 4, "t": "Esri World Topo"}]),
                 "mark": {"type": "text", "align": "right", "baseline": "bottom", "color": "#475569", "fontSize": 9, "opacity": 0.8},
                 "encoding": {"text": {"field": "t", "type": "nominal"}, "x": _mx("x"), "y": _my("y")}},
            ]
            _params += [
                {"name": "mapbrush", "select": {"type": "interval", "encodings": ["x", "y"]}, "views": ["map_ix"]},
                {"name": "mhov", "select": {"type": "point", "clear": "pointerout", "fields": ["t"], "nearest": True, "on": "pointerover"}, "views": ["map_ix"]},
            ]
            _map = {"layer": _map_layers, "height": _MAP, "width": _MAP}
            _body = {"hconcat": [_body, _map], "spacing": 24} if _body is not None else _map
        if _body is not None:
            # only the columns the charts use, rounded: a smaller page to send and parse
            _keep = {"t": 0, "minute": 3, "elapsed": None, "seg": None, "elev_ft": 0, "pace": 2, "hr": 0, "power": 0, "cad": 0, "grade": 1, "px": 1, "py": 1}
            _ship = _st.select([pl.col(c).round(d) if d is not None else pl.col(c) for c, d in _keep.items() if c in _st.columns])
            _datasets["streams"] = _ship.fill_nan(None).to_dicts()
            _view = VegaLiteSpec({
                "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
                "config": {"view": {"continuousWidth": 300, "continuousHeight": 300, "strokeWidth": 0, "fill": "transparent"}, "axis": {"domain": False, "gridOpacity": 0.3},
                           "title": {"anchor": "start", "fontSize": 13, "subtitleFontSize": 11}},
                **_body, "params": _params, "datasets": _datasets,
            })
        else:
            _view = mo.md("_No GPS or streams for this activity._")

        _mpp = (_tmax / _pw) if (_specs and _pw) else 0
        _out = mo.vstack([mo.hstack([_header, drill_close], justify="space-between", align="center"), _view,
                          mo.Html(f'<span hidden data-mpp="{_mpp:.6g}" data-tmax="{(_tmax if _specs else 0):.6g}"></span>')], gap=1)
        DRILL_CACHE[(_key, tuple(DASH.values()))] = _out
    _out
    return


@app.cell(hide_code=True)
def excluded(CUBE_DIR, acts, get_excluded, set_excluded):
    # the exclude list lives next to the cubes, so it survives sessions; untick an activity to bring it back
    Path(CUBE_DIR, "excluded.json").write_text(json.dumps(get_excluded()))
    _names = acts.filter(pl.col("activity_id").is_in(get_excluded()))
    _opts = {f"{r['name']} ({r['date']:%b %d, %Y})": r["activity_id"] for r in _names.iter_rows(named=True)}
    excluded_view = mo.ui.multiselect(options=_opts, value=list(_opts), label=f"Excluded activities ({len(_opts)}); untick to restore", on_change=lambda v: set_excluded(sorted(v)))
    return (excluded_view,)


@app.cell(hide_code=True)
def footer(ABOUT_MD, cube_status, excluded_view, rebuild):
    mo.accordion(
        {
            "About and data": mo.vstack(
                [
                    ABOUT_MD,
                    mo.ui.table(cube_status, selection=None, label="Cubes (rebuilt automatically when new activities appear)"),
                    rebuild,
                    excluded_view if len(excluded_view.options) else mo.md("_No activities excluded._"),
                ],
                gap=1,
            )
        }
    )
    return


if __name__ == "__main__":
    app.run()
