"""Ceiling sweep: r_pre × r_target probing the 1/τ_ref refractory bound.

Linear-gated rule (M = R = target − r_post). Wide log-spaced ranges so
the heatmap shows a saturation band at r_target ≈ 333 Hz (= 1/τ_ref
with τ_ref = 3 ms) where the post can't fire faster.

Each cell is a full ``simulate(p, name=…)`` run, so any cell is
inspectable later via ``load_latest("ceiling-sweep/cell_03_07").serve()``.
The sweep summary is written to ``output/ceiling-sweep/summary.parquet``
plus a matplotlib heatmap at ``summary.png``.

    uv run python experiments/ceiling_sweep.py
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

SWEEP = "ceiling-sweep"
OUT = Path("output") / SWEEP

BASE = Params(
    T=60.0, dt=1e-4, seed=1, record_every=1e-3,
    n_pre=1,
    r_pre=20.0,                         # placeholder, overridden below
    poisson=True,

    # Linear-gated three-factor rule
    M_rule="gated",
    R_rule="target_rate_linear",

    rate_mode="window",                 # window estimator stays steady when post fires fast
    rate_window=0.5,

    w0=5.0,                             # higher initial weight encourages saturation
    wmax=10.0,
    eta_plus=1e-4,
    eta_minus=1e-4,
)

X_GRID = np.geomspace(5.0, 1500.0, 14)   # r_pre (Hz)
Y_GRID = np.geomspace(5.0, 1000.0, 14)   # r_target (Hz); crosses 1/τ_ref ≈ 333 Hz


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
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r_pre (Hz)")
    ax.set_ylabel("r_target (Hz)")
    ax.axhline(1.0 / BASE.tau_ref, color="white", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(f"ceiling sweep ({nx}×{ny}, T={BASE.T}s)")
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
