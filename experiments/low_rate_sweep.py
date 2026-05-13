"""Low-rate r_pre × r_target sweep on a 22×22 grid (10-20 Hz, 0.5 Hz steps).

Linear-gated rule (M = R = target − r_post). Probes the operational
learning range, well below the refractory ceiling (1/τ_ref ≈ 333 Hz).

Each invocation creates a self-contained sweep dir
``output/low-rate-sweep/<YYYYMMDD_HHMMSS>/`` containing ``summary.parquet``,
``summary.png``, ``base_params.json``, and one ``cell_iJ_jJ/`` subdir per
grid point. Inspect a cell with
``load_latest("low-rate-sweep/<sweep-ts>/cell_03_07").serve()``.

    uv run python experiments/low_rate_sweep.py
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

from neuro import Params, simulate

SWEEP = "low-rate-sweep"
OUT = Path("output") / SWEEP

BASE = Params()

X_GRID = np.arange(10.0, 25.0, 0.5)   # r_pre (Hz):    0,1,…,20
Y_GRID = np.arange(10.0, 25.0, 0.5)   # r_target (Hz): 0,1,…,20


def _silent(it):
    return it


def run_cell(args: tuple[int, int, float, float, str]) -> dict:
    """Run one (r_pre, r_target) cell and return its summary row."""
    i, j, x, y, sweep_ts = args
    p = replace(BASE, r_pre=(float(x),), r_target=float(y))
    name = f"{SWEEP}/{sweep_ts}/cell_{i:02d}_{j:02d}"
    run = simulate(p, name=name, progress=_silent)

    spk = pl.read_parquet(run.spikes).filter(pl.col("spike_type") == "post")
    df = pl.read_parquet(run.parquet, columns=["t", "w1"])
    t_end = float(df["t"][-1])
    window_s = 10.0 * (p.rate_window if p.rate_mode == "window" else p.tau_r_post)
    late_start = max(t_end - window_s, 0.0)
    late = spk["t"].to_numpy()
    late = late[late >= late_start]
    r_post_late = float(len(late)) / max(t_end - late_start, 1e-6)

    return {
        "i": i, "j": j,
        "r_pre": float(x), "r_target": float(y),
        "r_post_late": r_post_late,
        "abs_error": abs(r_post_late - float(y)),
        "w_final": float(df["w1"][-1]),
        "duration_s": run.duration_s,
    }


def _heatmap(df: pl.DataFrame, png_path: Path) -> None:
    nx, ny = len(X_GRID), len(Y_GRID)
    err = np.full((ny, nx), np.nan)
    for r in df.iter_rows(named=True):
        err[int(r["i"]), int(r["j"])] = r["abs_error"]

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.pcolormesh(X_GRID, Y_GRID, err, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label="|<r_post> − target| (Hz)")
    ax.set_xlabel("r_pre (Hz)")
    ax.set_ylabel("r_target (Hz)")
    ax.set_xticks(X_GRID[::2])
    ax.set_yticks(Y_GRID[::2])
    ax.set_title(f"low-rate sweep ({nx}×{ny}, T={BASE.T}s)")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    sweep_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = OUT / sweep_ts
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "base_params.json").write_text(
        json.dumps(asdict(BASE), indent=2, sort_keys=True)
    )

    args_list = [
        (i, j, float(X_GRID[j]), float(Y_GRID[i]), sweep_ts)
        for i, j in itertools.product(range(len(Y_GRID)), range(len(X_GRID)))
    ]

    procs = max(1, mp.cpu_count() - 2)
    with mp.Pool(procs) as pool:
        rows = list(tqdm(pool.imap_unordered(run_cell, args_list),
                         total=len(args_list), desc="cells"))

    df = pl.DataFrame(rows).sort(["i", "j"])
    summary_pq = sweep_dir / "summary.parquet"
    summary_png = sweep_dir / "summary.png"
    df.write_parquet(summary_pq)
    _heatmap(df, summary_png)
    print(f"  {summary_pq}")
    print(f"  {summary_png}")
