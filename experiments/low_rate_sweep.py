"""Low-rate r_pre × r_target sweep on a 21×21 grid (0-20 Hz, 1 Hz steps).

Linear-gated rule (M = R = target − r_post). Probes the operational
learning range, well below the refractory ceiling (1/τ_ref ≈ 333 Hz).

Each cell is a full ``simulate(p, name=…)`` run, inspectable via
``load_latest("low-rate-sweep/cell_03_07").serve()``. The summary parquet
and matplotlib heatmap go to ``output/low-rate-sweep/summary.{parquet,png}``.

    uv run python experiments/low_rate_sweep.py
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

from neuro import Params, simulate

SWEEP = "low-rate-sweep"
OUT = Path("output") / SWEEP

BASE = Params(
    T=60.0, dt=1e-4, seed=1, record_every=1e-3,
    n_pre=1,
    r_pre=20.0,                         # placeholder, overridden below
    poisson=True,

    # Linear-gated three-factor rule
    M_rule="gated",
    R_rule="target_rate_linear",

    rate_mode="window",
    rate_window=0.5,

    w0=5.0,
    wmax=10.0,
    eta_plus=1e-4,
    eta_minus=1e-4,
)

X_GRID = np.arange(0.0, 21.0, 0.25)   # r_pre (Hz):    0,1,…,20
Y_GRID = np.arange(0.0, 21.0, 0.25)   # r_target (Hz): 0,1,…,20


def _silent(it):
    return it


def run_cell(args: tuple[int, int, float, float]) -> dict:
    """Run one (r_pre, r_target) cell and return its summary row."""
    i, j, x, y = args
    p = replace(BASE, r_pre=(float(x),), r_target=float(y))
    name = f"{SWEEP}/cell_{i:02d}_{j:02d}"
    run = simulate(p, name=name, progress=_silent)

    spk = pl.read_parquet(run.spikes).filter(pl.col("spike_type") == "post")
    df = pl.read_parquet(run.parquet, columns=["t", "w1"])
    t_end = float(df["t"][-1])
    half = max(t_end / 2.0, 1.0)
    late = spk["t"].to_numpy()
    late = late[late >= half]
    r_post_late = float(len(late)) / max(t_end - half, 1e-6)

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
    OUT.mkdir(parents=True, exist_ok=True)
    args_list = [
        (i, j, X_GRID[j], Y_GRID[i])
        for i, j in itertools.product(range(len(Y_GRID)), range(len(X_GRID)))
    ]

    procs = max(1, mp.cpu_count() - 2)
    with mp.Pool(procs) as pool:
        rows = list(tqdm(pool.imap_unordered(run_cell, args_list),
                         total=len(args_list), desc="cells"))

    df = pl.DataFrame(rows).sort(["i", "j"])
    summary_pq = OUT / "summary.parquet"
    summary_png = OUT / "summary.png"
    df.write_parquet(summary_pq)
    _heatmap(df, summary_png)
    print(f"  {summary_pq}")
    print(f"  {summary_png}")
