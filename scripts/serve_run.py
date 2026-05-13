"""Serve a saved run in the zoom viewer without resimulating.

    uv run python scripts/serve_run.py                   # list available runs
    uv run python scripts/serve_run.py <name>            # serve latest of <name>
    uv run python scripts/serve_run.py <parquet-path>    # serve that exact run
"""
from __future__ import annotations

import sys
from pathlib import Path

from neuro import list_runs, load_latest, load_run

if len(sys.argv) == 1:
    runs = list_runs()
    if not runs:
        print("No runs under output/")
        sys.exit(1)
    print(f"{'name':<40} {'saved_at':<20} parquet")
    for r in runs:
        print(f"{r.name:<40} {r.saved_at:<20} {r.parquet}")
    sys.exit(0)

arg = sys.argv[1]
run = load_run(arg) if Path(arg).suffix == ".parquet" else load_latest(arg)
print(f"serving {run.parquet}")
run.serve()
