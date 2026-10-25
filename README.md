# effort-curves (prototype)

Mean-max effort curves from 1 Hz activity samples, in Rust with Python bindings. A prototype for replacing the Data cubes notebook's fixed-length cube (24 durations, 19 distances) with curves at every second up to 100 s and 1 % steps beyond.

## Model

- **Gaps are bridged, never split.** Pauses and smart-recording gaps stay inside the activity, and their elapsed time counts against any window spanning them. Distance and altitude are interpolated across a gap (a real pause records no distance, so pace there is zero), power is zero, and heart rate is interpolated between the readings either side (up to `hr_bridge` seconds, default 600).
- **Running totals.** Each activity becomes a 1 Hz grid of running totals (distance, work, heartbeats, gross ascent, grade-adjusted distance). Every window's average is a difference of two rows.
- **Glitches** are removed per sample: a second faster than the sport's speed cap, or climbing faster than the ascent cap, counts as no movement or no climbing. Gradient keeps a window-level cap.
- **Strides.** A window of length L starts every max(1, floor(eps * L)) seconds, plus the window ending at the activity's end. A distance window advances by eps times its own duration. Together with a 1 % length grid, the per-activity "best at this length or longer" curve of every metric with a non-negative running total (power, speed, grade-adjusted pace, vertical speed) stays within about 1 % of the exhaustive one.
- **Filters** (`(dim, lo, hi)` on p v g h a x, t0 = seconds in, d0 = metres in; NaN = open end) are applied per window, so two-sided bands are exact. That is the on-demand scan.

## Build

No Rust toolchain on the host: `./build.sh` builds the abi3 wheel in the pinned `ghcr.io/pyo3/maturin` container, tuned for this CPU (`target-cpu=native`).

```sh
./build.sh
uv venv -p 3.12 .venv && uv pip install -p .venv duckdb numpy polars pyarrow dist/*.whl
```

## Benchmark

`bench/current_cube.py` times the notebook's own `build_cube` against the host's Garmin snapshot (into `out/current`). `bench/bench.py` builds the curves, compares them with that cube, and times single-activity updates and a two-sided gradient band. `HR_BRIDGE` sets the heart-rate bridging limit and `RAYON_NUM_THREADS` the thread count.
