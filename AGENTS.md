# effort-curves: agent guidance

The Rust engine (PyO3 bindings) for exact mean-max effort curves, and its main consumer, the MarimoHub notebook "Data cubes". The engine is meant to move into garmin-mcp (Python) and strava-mcp (TypeScript), so keep logic in Rust rather than in the notebook where both would need it.

## Engine

- Build with `./build.sh`: the pinned maturin container, since the host has no Rust toolchain. Install the wheel into `.venv` with `uv pip install -p .venv --reinstall dist/<wheel>`, then test with `.venv/bin/python -m pytest -q tests`.
- Tests compare against brute-force references: every start, every integer length. Keep that standard for new features.
- `Prepared.frontier` must match the every-integer-length reference exactly wherever its bounds are strict:
  - time basis: p v a x c;
  - distance basis: speed.
  - For the other metrics, only the handoffs are refined; say so where it matters.
- Real-data checks use `bench/load.py` (the host's Garmin snapshot), for example a frontier of 30 runs against `compute` at every integer length.
- Release:
  1. Bump `Cargo.toml` and `pyproject.toml` together.
  2. Build, then test.
  3. Commit (conventional) and tag `vX.Y.Z`.
  4. Push `main` and the tag.
  5. `gh release create vX.Y.Z dist/<wheel>`.

  Never replace a released asset; cut a new patch release instead. A local rebuild is not byte-identical to the release.

## The notebook

Its MarimoHub IDs:
- project `proj-04vwegq2rf4cdtza`
- notebook `nb-he1qvfcq81feym9f`
- app: `https://marimo.spencerboucher.com/projects/proj-04vwegq2rf4cdtza/notebooks/nb-he1qvfcq81feym9f/app`

The workspace is on the host at `~/homelab/marimo/data/projects/proj-04vwegq2rf4cdtza/notebooks/nb-he1qvfcq81feym9f/workspace/`, which the sandbox sees as `/workspace`.

- **Editing the notebook:**
  - Edit only through a live session with `marimo._code_mode` (`ctx.edit_cell`, `create_cell`, `run_cell`, `set_ui_value`). Follow the marimo-pair skill (`~/homelab/hermes/skills/marimo-pair/SKILL.md`) and marimo-style (`~/.agents/skills/marimo-style/SKILL.md`).
  - Never edit `notebook.py` directly.
  - Deliver one finished piece, then stop for feedback.
- **Editing the dashboard page:**
  - The page is `studio/notebook/dashboard/index.html`. The host copy is not writable, so write it from inside a session.
  - Then run `marimo_studio.agent.current_workspace().view("dashboard")`, with `validate(level="static")` and `build()`.
  - Also check that every page script parses (node `new Function`) and that no top-level `let`/`const` is declared twice.
- **Engine dependency:**
  1. In the sandbox, run `uv remove --no-sync effort-curves`, then `uv add --no-sync --raw "effort-curves @ <release wheel URL>"`.
  2. Stop the edit session and start a new one to pick up the new version.
- **Stored curves** (`cube/`):
  - They are every start (eps 0) at 1 % lengths, and the frontier is measured against them.
  - Bump `CUBE_SCHEMA` when the stored columns or the engine's rules change. Every sport rebuilds once, along with its display table and positions, which takes about 2.5 min.
  - Keep precomputation on disk, not in session caches.

## Sessions: never stop the app

- Use one edit session of your own. Stopping it saves a version.
- **Never stop app sessions.** Stopping one shows every open tab "App stopped", and the user has to click Restart.
- MarimoHub handles versions itself (`MARIMOHUB_APP_MAX_SESSIONS_PER_VERSION: 1`, see `~/homelab/marimo/docker-compose.yaml`):
  - saving a version marks older app sessions as draining;
  - an open tab keeps its session;
  - a reload gets the new version;
  - a drained session retires once it has had no visitors for the app idle timeout (8 h).
- After a save, tell the user to reload; do not restart anything for them.
- To keep memory bounded (an app sandbox holds about 1.4 GB), at most two app sessions should run: the newest, and the one before it, which may be the user's open tab. On a save you may stop app sessions older than those two. Never stop either of them.
  - `list_catalog` shows only the IDs and status, so record which app sessions exist after each save to know their order.

## Testing in the app

- The user allows the NUC's telebrowser for debugging this app; it is not allowed for other marimo work.
- If it is signed out, sign in yourself.
- Load the app URL. The dashboard is in an iframe whose `contentDocument` is reachable.
  - Synthetic `pointermove` events work for hovering and tooltips.
  - Real clicks and drags need an invisible overlay `<button>` in the top document, clicked by snapshot uid.
- It draws about 2 animation frames a second, so glides and transitions cannot be judged there:
  - render Vega specs offline instead (`vl_convert` in this repo's `.venv`);
  - or replace the iframe's `requestAnimationFrame` with a 16 ms timer to log the values.
- Say what was and was not tested in the app.

## Checkpoints

`notebook/` holds a restore point: `notebook.py`, `pyproject.toml`, `uv.lock` and the dashboard page. It is not the source of truth. Refresh it only when asked, from the workspace, as a `chore:` commit, and note the MarimoHub version ID in `notebook/README.md`.
