"""2D sweep over (r_pre, r_target) probing the firing-rate ceiling.

The post-synaptic LIF neuron has an absolute refractory period of
tau_ref = 3 ms, so its hard firing-rate ceiling is 1/tau_ref ≈ 333 Hz.
This script sweeps r_target above and below that ceiling to test what
the linear-gated three-factor rule does when the target is unreachable.

Configuration
-------------
- reward_signal = "target_rate_linear"  (M = R = target − r_post)
- neuromod_type = "gated"                (no R̄ baseline subtraction)
- n_pre = 1, single Poisson presynaptic neuron, no I_ext
- rate_mode = "window" with W = 0.5 s

Heatmap value: |late-half mean(r_post) − r_target| in Hz.

Each cell uses StreamingConvergence to break the simulation loop early
once r_post has held steady; cells that don't converge within T_max are
flagged. Outputs go to output/sweeps/<hash>.{parquet,json,png}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

from neuro import ConvergenceCriterion, Params, StreamingConvergence, simulate


# ── Sweep configuration ────────────────────────────────────────────

OUT_DIR = Path("output/sweeps")

T_MAX = 60.0          # max simulated seconds per cell
DT = 1e-4
SEED = 1
W0 = 5.0              # initial weight; high to encourage saturation
WMAX = 10.0
ETA = 1e-4
RATE_WINDOW = 0.5
RECORD_EVERY = 1e-3

# Convergence detector for early-stop and "did it run long enough?"
CRITERION = ConvergenceCriterion(
    window=4.0,        # 4 s half-window → 8 s comparison span
    rel_tol=0.05,
    abs_tol=0.5,
    consecutive=3,
    min_t=10.0,
    check_interval=1.0,
)

# 1/tau_ref for the default Params(tau_ref=3 ms) — used only for the plot guideline.
TAU_REF = 0.003
HARD_CEILING_HZ = 1.0 / TAU_REF  # ≈ 333.3 Hz


def _silent(it):
    return it


def _run_one(args: tuple[int, int, float, float]) -> dict:
    i, j, r_pre, r_target = args
    p = Params(
        T=T_MAX, dt=DT, method="rk4", seed=SEED,
        n_pre=1,
        r_pre_rates=(r_pre,), poisson=True,
        w0=(W0,), wmax=WMAX, eta_plus=ETA, eta_minus=ETA,
        reward_signal="target_rate_linear",
        neuromod_type="gated",
        target_func="fixed",
        r_target=r_target,
        rate_mode="window", rate_window=RATE_WINDOW,
        record_every=RECORD_EVERY,
    )
    detector = StreamingConvergence(criterion=CRITERION, target=r_target)
    t0 = time.monotonic()
    rec = simulate(p, progress=_silent, early_stop=detector)
    elapsed = time.monotonic() - t0

    post_times = np.asarray(rec["post_spike_times"])
    if detector.converged_at is not None:
        t_end = detector.converged_at
        half = max(t_end / 2.0, 1.0)
    else:
        t_end = T_MAX
        half = T_MAX / 2.0
    late = post_times[post_times >= half]
    span = max(t_end - half, 1e-6)
    r_post_late = float(len(late)) / span

    return {
        "i": i, "j": j,
        "r_pre": float(r_pre),
        "r_target": float(r_target),
        "r_post_late": r_post_late,
        "err": abs(r_post_late - r_target),
        "w_final": float(rec["w1"][-1]),
        "converged": detector.converged_at is not None,
        "t_converged": float(detector.converged_at) if detector.converged_at is not None else float("nan"),
        "t_end": float(t_end),
        "wall_s": elapsed,
    }


def _build_grids(n_grid: int, r_pre_range: tuple[float, float], r_target_range: tuple[float, float]):
    r_pre_grid = np.linspace(r_pre_range[0], r_pre_range[1], n_grid)
    # log-spaced r_target so the cluster around the ceiling gets resolution
    r_target_grid = np.geomspace(r_target_range[0], r_target_range[1], n_grid)
    return r_pre_grid, r_target_grid


def run_sweep(n_grid: int, r_pre_range: tuple[float, float], r_target_range: tuple[float, float], procs: int) -> dict:
    r_pre_grid, r_target_grid = _build_grids(n_grid, r_pre_range, r_target_range)
    cells = [
        (i, j, float(r_pre_grid[j]), float(r_target_grid[i]))
        for i in range(n_grid) for j in range(n_grid)
    ]

    rows: list[dict] = []
    if procs <= 1:
        for c in tqdm(cells, desc="cells"):
            rows.append(_run_one(c))
    else:
        with mp.Pool(procs) as pool:
            for row in tqdm(pool.imap_unordered(_run_one, cells), total=len(cells), desc="cells"):
                rows.append(row)

    df = pl.DataFrame(rows).sort(["r_target", "r_pre"])
    return {
        "df": df,
        "r_pre_grid": r_pre_grid,
        "r_target_grid": r_target_grid,
        "n_grid": n_grid,
    }


def build_manifest(n_grid: int, r_pre_grid: np.ndarray, r_target_grid: np.ndarray) -> dict:
    return {
        "version": 1,
        "kind": "target_ceiling_sweep",
        "x_var": "r_pre",
        "y_var": "r_target",
        "x_grid": [float(v) for v in r_pre_grid],
        "y_grid": [float(v) for v in r_target_grid],
        "n_grid": n_grid,
        "T_max": T_MAX, "dt": DT, "method": "rk4", "seed": SEED,
        "n_pre": 1, "poisson": True,
        "reward_signal": "target_rate_linear",
        "neuromod_type": "gated",
        "rate_mode": "window", "rate_window": RATE_WINDOW,
        "w0": W0, "wmax": WMAX, "eta": ETA,
        "tau_ref": TAU_REF,
        "hard_ceiling_hz": HARD_CEILING_HZ,
        "convergence": {
            "window": CRITERION.window,
            "rel_tol": CRITERION.rel_tol,
            "abs_tol": CRITERION.abs_tol,
            "consecutive": CRITERION.consecutive,
            "min_t": CRITERION.min_t,
            "check_interval": CRITERION.check_interval,
        },
    }


def hash_of(manifest: dict) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_outputs(result: dict, manifest: dict, h: str, out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = result["df"]
    pq_path = out_dir / f"{h}.parquet"
    pq_tmp = pq_path.with_suffix(".parquet.tmp")
    df.write_parquet(str(pq_tmp))
    pq_tmp.replace(pq_path)

    meta = {"hash": h, "saved_at": datetime.now().isoformat(timespec="seconds"), "parquet": str(pq_path), **manifest}
    json_path = out_dir / f"{h}.json"
    _atomic_write_bytes(json_path, json.dumps(meta, indent=2, sort_keys=True).encode())

    # Heatmap
    n = result["n_grid"]
    r_pre_grid = result["r_pre_grid"]
    r_target_grid = result["r_target_grid"]
    err = np.full((n, n), np.nan)
    converged = np.zeros((n, n), dtype=bool)
    for row in df.iter_rows(named=True):
        # find indices via grid lookup
        j = int(np.argmin(np.abs(r_pre_grid - row["r_pre"])))
        i = int(np.argmin(np.abs(r_target_grid - row["r_target"])))
        err[i, j] = row["err"]
        converged[i, j] = row["converged"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    ax = axes[0]
    im = ax.pcolormesh(r_pre_grid, r_target_grid, err, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label="|r_post − r_target| (Hz)")
    ax.axhline(HARD_CEILING_HZ, color="#ff4488", linestyle="--", linewidth=1.5, label=f"1/τ_ref ≈ {HARD_CEILING_HZ:.0f} Hz")
    ax.set_yscale("log")
    ax.set_xlabel("r_pre (Hz)")
    ax.set_ylabel("r_target (Hz, log)")
    ax.set_title("error: |r_post − r_target|")
    ax.legend(loc="lower right", fontsize=9)

    ax2 = axes[1]
    im2 = ax2.pcolormesh(r_pre_grid, r_target_grid, converged.astype(int), cmap="RdYlGn", vmin=0, vmax=1, shading="auto")
    fig.colorbar(im2, ax=ax2, label="converged (1) / didn't (0)")
    ax2.axhline(HARD_CEILING_HZ, color="#000000", linestyle="--", linewidth=1.5)
    ax2.set_yscale("log")
    ax2.set_xlabel("r_pre (Hz)")
    ax2.set_ylabel("r_target (Hz, log)")
    ax2.set_title(f"convergence flag ({converged.sum()}/{converged.size})")

    png_path = out_dir / f"{h}.png"
    png_tmp = png_path.with_suffix(".png.tmp")
    fig.savefig(str(png_tmp), dpi=120, format="png")
    plt.close(fig)
    png_tmp.replace(png_path)

    return pq_path, json_path, png_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-grid", type=int, default=12, help="grid points per axis")
    ap.add_argument("--r-pre-min", type=float, default=10.0)
    ap.add_argument("--r-pre-max", type=float, default=400.0)
    ap.add_argument("--r-target-min", type=float, default=20.0)
    ap.add_argument("--r-target-max", type=float, default=600.0)
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    r_pre_range = (args.r_pre_min, args.r_pre_max)
    r_target_range = (args.r_target_min, args.r_target_max)

    r_pre_grid, r_target_grid = _build_grids(args.n_grid, r_pre_range, r_target_range)
    manifest = build_manifest(args.n_grid, r_pre_grid, r_target_grid)
    h = hash_of(manifest)

    pq_path = args.out_dir / f"{h}.parquet"
    json_path = args.out_dir / f"{h}.json"
    png_path = args.out_dir / f"{h}.png"
    if pq_path.exists() and json_path.exists() and png_path.exists():
        print(f"Already done: {h} (re-run by deleting {args.out_dir}/{h}.*)")
        return

    print(f"Sweep: {args.n_grid}×{args.n_grid} = {args.n_grid ** 2} cells")
    print(f"  r_pre ∈ [{r_pre_range[0]}, {r_pre_range[1]}] Hz")
    print(f"  r_target ∈ [{r_target_range[0]}, {r_target_range[1]}] Hz (geomspace)")
    print(f"  hard ceiling 1/τ_ref ≈ {HARD_CEILING_HZ:.1f} Hz")
    print(f"  T_max={T_MAX}s with early-stop convergence detector")
    print(f"  procs={args.procs}, hash={h}")

    t0 = time.monotonic()
    result = run_sweep(args.n_grid, r_pre_range, r_target_range, args.procs)
    elapsed = time.monotonic() - t0

    pq, js, png = write_outputs(result, manifest, h, args.out_dir)
    n_conv = int(result["df"]["converged"].sum())
    print(f"Done in {elapsed:.1f}s. Converged: {n_conv}/{args.n_grid ** 2}")
    print(f"  {pq}")
    print(f"  {js}")
    print(f"  {png}")


if __name__ == "__main__":
    main()
