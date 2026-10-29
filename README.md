# effort-curves (prototype)

Mean-max effort curves from 1 Hz activity samples, in Rust with Python bindings. A prototype for replacing the Data cubes notebook's fixed-length cube (24 durations, 19 distances) with curves at every second up to 100 s and 1 % steps beyond.

## Model

- **Gaps are bridged, never split.** Pauses and smart-recording gaps stay inside the activity, and their elapsed time counts against any window spanning them. Distance and altitude are interpolated across a gap (a real pause records no distance, so pace there is zero), power is zero, and heart rate is interpolated between the readings either side (up to `hr_bridge` seconds, default 600).
- **Running totals.** Each activity becomes a 1 Hz grid of running totals (distance, work, heartbeats, gross ascent, grade-adjusted distance). Every window's average is a difference of two rows.
- **Glitches** are removed per sample: a second faster than the sport's speed cap, or climbing faster than the ascent cap, counts as no movement or no climbing. Gradient keeps a window-level cap.
- **Strides.** A window of length L starts every max(1, floor(eps * L)) seconds, plus the window ending at the activity's end. A distance window advances by eps times its own duration. Together with a 1 % length grid, the per-activity "best at this length or longer" curve of every metric with a non-negative running total (power, speed, grade-adjusted pace, vertical speed) stays within about 1 % of the exhaustive one.
- **Metrics** per window: `p` power, `v` speed, `g` net gradient, `h` heart rate, `a` vertical speed, `x` grade-adjusted speed, `c` cadence over the whole window (coasting and stops count as zero, so it keeps the running-total guarantee), `k` crank torque (work per radian of pedal revolution), `l` distance per cycle (step length), `pc` cadence while pedaling or stepping, `ps` share of the window spent pedaling or stepping. Cadence is given in cycles per minute (pedal revolutions, or steps); seconds above the sport's cadence cap count as none.
- **Filters** (`(dim, lo, hi)` on any metric, t0 = seconds in, d0 = metres in; NaN = open end) are applied per window, so two-sided bands are exact. That is the on-demand scan.
- **Groups** (`group=(dim, edges)`) put each window into a bin of one dimension and keep the best per bin: value x-axes (pace by gradient) and band lines (by time into the activity) in one pass.
  `group2=(dim, edges)` bins a second dimension at the same time, so every combination of two splits (band lines and animation bands, or a value axis and band lines) comes out of one pass, shaped (activities, groups, groups2, lengths, metrics).
- **Activities** (`activities=` indices) restrict a scan to a selection, such as a date window.
- **Metrics** (`metrics=[...]`) return only the metrics asked for: a fine-binned pass for one metric is a tenth of the size.
- **Range frames** (`range_frames(y, t0, width)`): from a fine-binned grid, the best over every run of `width` consecutive bins (a sliding maximum), for an animation whose frame is a range of a dimension.
- **Time spent** (`Prepared.seconds_in(dim, edges, length=30, filters=[], activities=None)`): per activity, the seconds in each bin of a dimension (heart-rate or gradient zones, time or distance into the activity), each second valued as the `length`-second window starting at it, among windows passing the filters: the histogram under a range animation.
- **Gait** (`Prepared.cadence_seconds(edges)`, `ec.gait_threshold(seconds, edges)`, `Prepared.set_gait(threshold, min_bout=10)`): each second stepping is running (cadence at or above the threshold) or walking, in bouts of at least `min_bout` seconds. The threshold is the valley between the two cadence peaks of the recordings' own time-weighted histogram (NaN for a single gait). Windows then have `rs`, the share of their stepping that was running, as a metric, filter or group, and `seconds_in(..., by_gait=True)` splits time spent into [not running, running].
- **Frontier as efforts** (`Prepared.frontier(basis, metric, lengths, y, t0, activities, sets, ...)`) turns a grid of per-activity bests into the efforts that make up the frontier of each set of activities (and each group cell): an effort is one stretch of one activity that holds the best (of that length or longer) over a range of lengths, from the previous effort's full length up to its own longest window. With `refine` (default) the efforts and their handoffs are exact to the second or metre: grid steps that could hide anything are evaluated at every integer length, subdividing long ones, for the activities that could matter there. Which activities could matter is decided by a strict bound for power, speed, grade-adjusted speed, vertical speed and cadence on the time basis (a window lies inside a longer one), and for speed on the distance basis (a stretch contains a shorter one). For the other metrics only the handoffs are refined. A single set gives that activity's own efforts. On Run, one frontier takes about 10-40 ms over 678 activities.
- **Frontier over a window property** (`Prepared.frontier_into(basis, length, metric, dim, activities, sets, ..., direction)`): for windows of one length, the best among those at least as hard as x on a dimension with a harder direction: where the window starts (seconds `t0` or metres `d0` in; later is harder, `ge`) or one of its own metrics (heart rate: lower is harder for the same pace, `le`; gradient: steeper, `ge`). Later is harder, so it is a staircase read from the far end, exact by construction (every start of every activity, under the filters), split into efforts as above. A durability curve: on Run, about 15-40 ms for one set.

## Build

No Rust toolchain on the host: `./build.sh` builds the abi3 manylinux wheel in the pinned `ghcr.io/pyo3/maturin` container, for x86-64-v3 CPUs (`TARGET_CPU=native` to tune it to the build machine). Releases attach that wheel.

```sh
./build.sh
uv venv -p 3.12 .venv && uv pip install -p .venv duckdb numpy polars pyarrow dist/*.whl
```

## Test

```sh
uv pip install -p .venv pytest && .venv/bin/python -m pytest tests
```

## Benchmark

`bench/current_cube.py` times the notebook's own `build_cube` against the host's Garmin snapshot (into `out/current`). `bench/bench.py` builds the curves, compares them with that cube, and times single-activity updates and a two-sided gradient band. `HR_BRIDGE` sets the heart-rate bridging limit and `RAYON_NUM_THREADS` the thread count.

## License

[Functional Source License, Version 1.1, ALv2 Future License](LICENSE.md) (FSL-1.1-ALv2). You may use, modify and self-host it for any purpose except offering a competing commercial product or service; each release becomes available under Apache-2.0 two years after it ships.

Outside contributions require agreeing to a contributor license agreement; open an issue before sending a pull request.
