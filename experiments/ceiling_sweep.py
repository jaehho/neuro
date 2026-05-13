"""Ceiling sweep: r_pre × r_target probing the 1/τ_ref refractory bound.

Linear-gated rule (M = R = target − r_post). Wide log-spaced ranges so
the heatmap shows a saturation band at r_target ≈ 333 Hz (= 1/τ_ref
with τ_ref = 3 ms) where the post can't fire faster.

Each invocation creates a self-contained sweep dir
``output/ceiling-sweep/<YYYYMMDD_HHMMSS>/`` containing ``summary.parquet``,
``summary.png``, ``base_params.json``, and one ``cell_iJ_jJ/`` subdir per
grid point. Inspect a cell with
``load_latest("ceiling-sweep/<sweep-ts>/cell_03_07").serve()``.

    uv run python experiments/ceiling_sweep.py
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

SWEEP = "ceiling-sweep"
OUT = Path("output") / SWEEP

BASE = Params()

X_GRID = np.geomspace(5.0, 500.0, 10)   # r_pre (Hz)
Y_GRID = np.geomspace(5.0, 500.0, 10)   # r_target (Hz); crosses 1/τ_ref ≈ 333 Hz


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
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r_pre (Hz)")
    ax.set_ylabel("r_target (Hz)")
    ax.axhline(1.0 / BASE.tau_ref, color="white", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(f"ceiling sweep ({nx}×{ny}, T={BASE.T}s)")
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
