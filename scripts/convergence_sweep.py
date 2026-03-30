#!/usr/bin/env python3
"""2D convergence sweep over (r_pre_rate, alpha).

Runs the neuromodulated STDP model across a grid of presynaptic spike
frequencies and target rate ratios, classifying each point as converged,
silent, saturated, oscillating, rate-mismatched, or numerically divergent.

Run:  uv run python scripts/convergence_sweep.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from neuro.sim import Params, simulate

OUT = Path("output")

# ── Grid definition ───────────────────────────────────────────────────

R_PRE_RATES = np.concatenate([
    np.geomspace(1, 10, 5),       # log-spaced low end (silent-synapse boundary)
    np.linspace(20, 200, 15),     # linear high end
])
ALPHAS = np.linspace(0.05, 2.0, 20)

# ── Convergence thresholds ────────────────────────────────────────────

W_SILENT = 0.01
W_SATURATED = 9.9       # wmax - 0.1
W_STD_OSCILLATING = 1.0
REL_RATE_ERR = 0.3

# ── Helpers ───────────────────────────────────────────────────────────

def _run(T: float = 10.0, **kw):
    p = Params(T=T, method="rk4", record_every=1e-3, **kw)
    rec = simulate(p)
    return p, rec


def _metrics(rec: dict, p: Params) -> dict:
    t = rec["t"]
    half = p.T / 2
    ps = rec["post_spike_times"]
    late = ps[ps >= half]
    rate = len(late) / (p.T - half) if p.T > half else 0.0
    target = p.alpha * p.r_pre_rate

    wl = rec["w"][t >= half]
    w_mean = float(np.mean(wl)) if len(wl) else 0.0
    w_std = float(np.std(wl)) if len(wl) else 0.0

    has_nan = False
    has_inf = False
    for key in ("w", "R", "M", "R_bar", "E", "V", "r_post"):
        arr = rec[key]
        if np.any(np.isnan(arr)):
            has_nan = True
        if np.any(np.isinf(arr)):
            has_inf = True

    rate_err = abs(rate - target)
    rel_err = rate_err / max(target, 1e-6)

    return dict(
        post_rate=rate,
        target_rate=target,
        rate_error=rate_err,
        relative_rate_error=rel_err,
        w_final=float(rec["w"][-1]),
        w_mean=w_mean,
        w_std=w_std,
        n_post=len(ps),
        has_nan=has_nan,
        has_inf=has_inf,
    )


def _classify(m: dict) -> str:
    if m["has_nan"] or m["has_inf"]:
        return "nan_inf"
    if m["n_post"] == 0 or m["w_final"] < W_SILENT:
        return "silent"
    if m["w_final"] > W_SATURATED:
        return "saturated"
    if m["w_std"] > W_STD_OSCILLATING:
        return "oscillating"
    if m["relative_rate_error"] > REL_RATE_ERR:
        return "rate_mismatch"
    return "converged"


# ── Plotting ──────────────────────────────────────────────────────────

CLASSES = ["converged", "silent", "saturated", "oscillating",
           "rate_mismatch", "nan_inf", "exception"]
CLASS_COLORS = ["#2a9d8f", "#264653", "#e76f51", "#e9c46a",
                "#f4a261", "#1a1a2e", "#aaaaaa"]
CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}


def _reshape(results: list[dict], key: str, nr: int, na: int, default=np.nan):
    grid = np.full((nr, na), default)
    for k, m in enumerate(results):
        grid[k // na, k % na] = m.get(key, default)
    return grid


def plot_classification(results: list[dict], nr: int, na: int) -> None:
    grid = np.full((nr, na), len(CLASSES) - 1, dtype=int)
    for k, m in enumerate(results):
        grid[k // na, k % na] = CLASS_MAP.get(m["status"], len(CLASSES) - 1)

    cmap = ListedColormap(CLASS_COLORS)
    bounds = np.arange(len(CLASSES) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.pcolormesh(ALPHAS, R_PRE_RATES, grid, cmap=cmap, norm=norm,
                       shading="nearest")

    cbar = fig.colorbar(im, ax=ax, ticks=range(len(CLASSES)))
    cbar.ax.set_yticklabels(CLASSES)

    ax.set_xlabel(r"$\alpha$ (target rate ratio)", fontsize=12)
    ax.set_ylabel(r"$r_{\rm pre}$ (Hz)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Convergence classification", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "convergence_map.png", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT / 'convergence_map.png'}")


def plot_heatmap(results: list[dict], key: str, nr: int, na: int,
                 *, title: str, fname: str, log: bool = False) -> None:
    grid = _reshape(results, key, nr, na)

    fig, ax = plt.subplots(figsize=(12, 8))
    kw: dict = {}
    if log:
        floor = np.nanmin(grid[grid > 0]) if np.any(grid > 0) else 1e-10
        grid = np.where(grid > 0, grid, floor)
        kw["norm"] = LogNorm(vmin=floor, vmax=np.nanmax(grid))

    cmap = plt.cm.viridis.copy()  # type: ignore[attr-defined]
    cmap.set_bad("gray", 0.3)

    im = ax.pcolormesh(ALPHAS, R_PRE_RATES, grid, cmap=cmap,
                       shading="nearest", **kw)
    fig.colorbar(im, ax=ax, label=key)

    ax.set_xlabel(r"$\alpha$ (target rate ratio)", fontsize=12)
    ax.set_ylabel(r"$r_{\rm pre}$ (Hz)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    print(f"  -> {OUT / fname}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(exist_ok=True)

    nr, na = len(R_PRE_RATES), len(ALPHAS)
    total = nr * na
    results: list[dict] = []

    print(f"Convergence sweep: {nr} x {na} = {total} simulations")
    t0 = time.time()

    for i, rpr in enumerate(R_PRE_RATES):
        for j, alpha in enumerate(ALPHAS):
            idx = i * na + j + 1
            label = f"[{idx}/{total}] r_pre={rpr:6.1f} Hz  alpha={alpha:.3f}"
            print(f"  {label}", end="", flush=True)

            try:
                p, rec = _run(r_pre_rate=float(rpr), alpha=float(alpha))
                m = _metrics(rec, p)
                m["r_pre_rate"] = float(rpr)
                m["alpha"] = float(alpha)
                m["status"] = _classify(m)
                m["error"] = ""
            except Exception as ex:
                m = dict(
                    r_pre_rate=float(rpr),
                    alpha=float(alpha),
                    post_rate=np.nan,
                    target_rate=float(alpha * rpr),
                    rate_error=np.nan,
                    relative_rate_error=np.nan,
                    w_final=np.nan,
                    w_mean=np.nan,
                    w_std=np.nan,
                    n_post=0,
                    has_nan=True,
                    has_inf=False,
                    status="exception",
                    error=str(ex),
                )

            results.append(m)
            print(f"  -> {m['status']}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed / total:.2f}s per sim)\n")

    # ── Save parquet ──────────────────────────────────────────────
    table = pa.Table.from_pylist(results)
    pq_path = OUT / "convergence_sweep.parquet"
    pq.write_table(table, pq_path, compression="zstd")
    print(f"Results -> {pq_path}")

    # ── Plots ─────────────────────────────────────────────────────
    plot_classification(results, nr, na)
    plot_heatmap(results, "w_final", nr, na,
                 title="Final weight", fname="convergence_w_final.png")
    plot_heatmap(results, "rate_error", nr, na, log=True,
                 title="Rate error  |r_post - alpha * r_pre|",
                 fname="convergence_rate_error.png")
    plot_heatmap(results, "w_std", nr, na, log=True,
                 title="Weight stability (std in 2nd half)",
                 fname="convergence_w_stability.png")

    # ── Summary ───────────────────────────────────────────────────
    print("\nConvergence Summary")
    print("=" * 40)
    from collections import Counter
    counts = Counter(m["status"] for m in results)
    for cls in CLASSES:
        n = counts.get(cls, 0)
        pct = 100.0 * n / total
        print(f"  {cls:16s}  {n:4d} / {total}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
