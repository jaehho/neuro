"""Open the same zoom-adaptive viewer as ``baseline.py``, on one or more sweep cells.

    uv run python scripts/view_sweep_cells.py

Edit ``SWEEP`` and ``CELLS`` below. ``CELLS`` is a list of ``(r_pre, r_target)``
tuples in Hz; the script snaps each pair to its nearest grid cell using
``output/<SWEEP>/summary.parquet``.

  - 1 cell  → http://127.0.0.1:8050/      (identical to ``run.serve()`` in baseline.py)
  - N cells → http://127.0.0.1:8050/ … :8050+N-1/   (one tab each, all live)

Ctrl-C in the terminal tears every server down at once.
"""
from __future__ import annotations

import threading
from pathlib import Path

import polars as pl

from neuro import load_latest

# ── Edit me ──────────────────────────────────────────────────────────
SWEEP = "low-rate-sweep"

# (r_pre, r_target) in Hz. Add or remove tuples to compare more or fewer cells.
CELLS: list[tuple[float, float]] = [
    (15.0, 15.0),   # diagonal: post tracks target
    (10.0, 20.0),   # top-left:  input too weak, post < target
    (20.0, 10.0),   # bot-right: input too strong, post > target
]
# ─────────────────────────────────────────────────────────────────────


def _nearest_cell(summary: pl.DataFrame, r_pre: float, r_target: float) -> tuple[int, int]:
    row = (
        summary
        .with_columns(((pl.col("r_pre") - r_pre) ** 2
                       + (pl.col("r_target") - r_target) ** 2).alias("_d2"))
        .sort("_d2")
        .row(0, named=True)
    )
    return int(row["i"]), int(row["j"])


def _latest_sweep_run(sweep_dir: Path) -> Path:
    if not sweep_dir.is_dir():
        raise FileNotFoundError(
            f"{sweep_dir} does not exist — run the sweep first, e.g.\n"
            f"    uv run python experiments/{SWEEP.replace('-', '_')}.py"
        )
    candidates = sorted(d for d in sweep_dir.iterdir()
                        if d.is_dir() and (d / "summary.parquet").exists())
    if not candidates:
        raise FileNotFoundError(
            f"No <ts>/summary.parquet under {sweep_dir} — run the sweep first, e.g.\n"
            f"    uv run python experiments/{SWEEP.replace('-', '_')}.py"
        )
    return candidates[-1]   # YYYYMMDD_HHMMSS dirs sort chronologically


def main() -> None:
    sweep_run_dir = _latest_sweep_run(Path("output") / SWEEP)
    sweep_ts = sweep_run_dir.name
    summary = pl.read_parquet(sweep_run_dir / "summary.parquet")
    print(f"sweep:   {sweep_run_dir}")

    targets = []
    for r_pre, r_target in CELLS:
        i, j = _nearest_cell(summary, r_pre, r_target)
        run = load_latest(f"{SWEEP}/{sweep_ts}/cell_{i:02d}_{j:02d}")
        targets.append((r_pre, r_target, i, j, run))

    base_port = 8050
    print()
    for k, (r_pre_q, r_target_q, i, j, run) in enumerate(targets):
        port = base_port + k
        print(f"  cell_{i:02d}_{j:02d}  "
              f"r_pre={run.params.r_pre[0]:.2f}  r_target={run.params.r_target:.2f}  "
              f"(asked {r_pre_q}, {r_target_q})  →  http://127.0.0.1:{port}/")
    print()

    # Daemon-serve all but the last; block on the last in the main thread so
    # Ctrl-C kills the whole process (daemon servers die with it).
    for k, (_, _, _, _, run) in enumerate(targets[:-1]):
        port = base_port + k
        threading.Thread(
            target=lambda r=run, p=port: r.serve(host="127.0.0.1", port=p),
            daemon=True,
        ).start()
    targets[-1][-1].serve(host="127.0.0.1", port=base_port + len(targets) - 1)


if __name__ == "__main__":
    main()
