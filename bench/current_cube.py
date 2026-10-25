"""Time the notebook's current cube build against the host's read-only Garmin snapshot.

Extracts the `cube_defs` cell from the Data cubes notebook, swaps the HTTP snapshot attach for the local
file, and runs `build_cube` per sport into out/current/<sport>.
"""

import ast
import json
import os
import sys
import time
from pathlib import Path

import duckdb

NOTEBOOK = Path(
    "/home/spencer/homelab/marimo/data/projects/proj-04vwegq2rf4cdtza/notebooks/nb-he1qvfcq81feym9f/workspace/notebook.py"
)
SNAPSHOT = "/home/spencer/homelab/garmin-mcp/data/snapshot/garmin.duckdb"
OUT = Path(__file__).resolve().parent.parent / "out" / "current"


def load_cube_defs():
    tree = ast.parse(NOTEBOOK.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cube_defs")
    body = [s for s in fn.body if not isinstance(s, ast.Return)]
    mod = ast.Module(body=body, type_ignores=[])
    ns = {}
    exec("import os, time, shutil, json\nimport duckdb\nimport numpy as np\nimport polars as pl\nfrom pathlib import Path", ns)
    exec(compile(mod, "cube_defs", "exec"), ns)

    def _build_connection():
        path = "/tmp/cube_build.duckdb"
        Path(path).unlink(missing_ok=True)
        bc = duckdb.connect(path)
        bc.execute("SET threads = 6")
        bc.execute("SET memory_limit = '12GB'")
        bc.execute("SET preserve_insertion_order = false")
        bc.execute(f"ATTACH '{SNAPSHOT}' AS garmin (READ_ONLY)")
        bc.execute("SET TimeZone = 'America/Denver'")
        return bc

    ns["_build_connection"] = _build_connection
    return ns


if __name__ == "__main__":
    sports = sys.argv[1:] or ["Run", "Ride", "Hike", "Skimo", "Skate ski"]
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUT)  # build_cube writes cube/gap_model.parquet relative to cwd
    Path("cube").mkdir(exist_ok=True)
    ns = load_cube_defs()
    results = {}
    for sport in sports:
        t = time.time()
        ns["build_cube"](sport, str(OUT / sport))
        results[sport] = round(time.time() - t, 1)
        print(f"{sport}: {results[sport]} s", flush=True)
    (OUT / "timings.json").write_text(json.dumps(results))
