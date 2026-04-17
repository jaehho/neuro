"""Sweep every 2D pair of sweep variables for the baseline path.

For each pair (x_var, y_var) in C(variables, 2), run a 2D grid sweep
and save three artifacts in output/sweeps/<hash>.{parquet,json,png}:
  - parquet: long-form DataFrame, one row per grid cell
  - json:    manifest (axes, grids, frozen values, seed, hash, saved_at)
  - png:     error heatmap with grid-aligned green outlines on converged cells

Hashing mirrors the notebook's save cell; if all three artifacts for a
given hash already exist, that pair is skipped (idempotent).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

from neuro.sim import Params, simulate


# Per-variable scalar (frozen) value and sweep range. Keep in sync with
# notebooks/rate_sweep_2d.py defaults so hashes are comparable across entrypoints.
VARS: dict[str, dict[str, object]] = {
    "r_pre":        {"frozen": 20.0,    "range": (5.0, 80.0)},
    "r_target":     {"frozen": 10.0,    "range": (2.0, 30.0)},
    "w0":           {"frozen": 2.0,     "range": (0.5, 8.0)},
    "W":            {"frozen": 0.5,     "range": (0.1, 1.0)},
    "eta_plus":     {"frozen": 0.0001,  "range": (0.00001, 0.001)},
    "eta_minus":    {"frozen": 0.0001,  "range": (0.00001, 0.001)},
    "wmax":         {"frozen": 10.0,    "range": (1.0, 20.0)},
    "tau_e":        {"frozen": 0.5,     "range": (0.05, 2.0)},
    "tau_Rbar":     {"frozen": 5.0,     "range": (0.5, 20.0)},
    "reward_delay": {"frozen": 1.0,     "range": (0.0, 3.0)},
}

T = 20.0
DT = 1e-4
SEED = 1
OUT_DIR = Path("output/sweeps")


def _silent(it):
    return it


def run_one_sweep(x_var: str, y_var: str, n_grid: int) -> dict:
    x_lo, x_hi = VARS[x_var]["range"]
    y_lo, y_hi = VARS[y_var]["range"]
    x_grid = np.linspace(x_lo, x_hi, n_grid)
    y_grid = np.linspace(y_lo, y_hi, n_grid)
    frozen = {k: float(v["frozen"]) for k, v in VARS.items()}

    err = np.full((n_grid, n_grid), np.nan)
    rate = np.full((n_grid, n_grid), np.nan)
    wfin = np.full((n_grid, n_grid), np.nan)
    thr = np.full((n_grid, n_grid), np.nan)

    half = T / 2.0
    pairs = [(i, j) for i in range(n_grid) for j in range(n_grid)]
    for i, j in tqdm(pairs, desc=f"{x_var} x {y_var}", leave=False, mininterval=0.5):
        vals = dict(frozen)
        vals[x_var] = float(x_grid[j])
        vals[y_var] = float(y_grid[i])
        p = Params(
            T=T, dt=DT, method="rk4", seed=SEED,
            n_pre=1,
            r_pre_rates=(vals["r_pre"],), poisson=False,
            w0=(vals["w0"],),
            reward_signal="target_rate", target_func="fixed",
            r_target=vals["r_target"],
            neuromod_type="covariance",
            rate_mode="window", rate_window=vals["W"],
            eta_plus=vals["eta_plus"], eta_minus=vals["eta_minus"],
            wmax=vals["wmax"],
            tau_e=vals["tau_e"], tau_Rbar=vals["tau_Rbar"],
            reward_delay=vals["reward_delay"],
            record_every=1e-3,
        )
        rec = simulate(p, progress=_silent)
        post_times = rec["post_spike_times"]
        late = post_times[post_times >= half]
        r_late = float(len(late)) / (T - half)
        rate[i, j] = r_late
        err[i, j] = abs(r_late - vals["r_target"])
        wfin[i, j] = float(rec["w1"][-1])
        thr[i, j] = 1.0 / vals["W"]

    return {
        "x_grid": x_grid, "y_grid": y_grid,
        "err": err, "rate": rate, "wfin": wfin, "thr": thr,
        "converged": err < thr,
        "frozen": frozen,
    }


def build_manifest(x_var: str, y_var: str, x_grid: np.ndarray, y_grid: np.ndarray,
                   frozen: dict[str, float]) -> dict:
    return {
        "version": 1,
        "x_var": x_var,
        "y_var": y_var,
        "x_grid": [float(v) for v in x_grid],
        "y_grid": [float(v) for v in y_grid],
        "T": T, "dt": DT, "method": "rk4", "seed": SEED,
        "n_pre": 1, "poisson": False,
        "reward_signal": "target_rate",
        "target_func": "fixed",
        "neuromod_type": "covariance",
        "rate_mode": "window",
        "frozen_effective": {
            k: float(v) for k, v in frozen.items() if k not in {x_var, y_var}
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
    x_var, y_var = manifest["x_var"], manifest["y_var"]

    # parquet (long-form DataFrame, one row per cell)
    rows = []
    ny, nx = result["err"].shape
    for i in range(ny):
        for j in range(nx):
            rows.append({
                x_var: float(result["x_grid"][j]),
                y_var: float(result["y_grid"][i]),
                "r_post_late": float(result["rate"][i, j]),
                "err": float(result["err"][i, j]),
                "w_final": float(result["wfin"][i, j]),
                "threshold": float(result["thr"][i, j]),
                "converged": bool(result["converged"][i, j]),
            })
    df = pl.DataFrame(rows).sort([y_var, x_var])

    pq_path = out_dir / f"{h}.parquet"
    pq_tmp = pq_path.with_suffix(".parquet.tmp")
    df.write_parquet(str(pq_tmp))
    pq_tmp.replace(pq_path)

    # json sidecar
    meta = {
        "hash": h,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "parquet": str(pq_path),
        **manifest,
    }
    json_path = out_dir / f"{h}.json"
    _atomic_write_bytes(json_path, json.dumps(meta, indent=2, sort_keys=True).encode())

    # png plot
    png_path = out_dir / f"{h}.png"
    x_grid = result["x_grid"]
    y_grid = result["y_grid"]
    err = result["err"]
    thr = result["thr"]
    converged = result["converged"]

    fig, ax = plt.subplots(figsize=(10, 6))
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])
    extent = [
        float(x_grid[0]) - dx / 2, float(x_grid[-1]) + dx / 2,
        float(y_grid[0]) - dy / 2, float(y_grid[-1]) + dy / 2,
    ]
    vmax = max(4.0 * float(np.nanmean(thr)), float(np.nanmax(err)))
    im = ax.imshow(err, origin="lower", extent=extent, aspect="auto",
                   cmap="viridis", vmin=0.0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="|r_post - r_target| (Hz)")
    for i, j in zip(*np.where(converged)):
        ax.add_patch(mpatches.Rectangle(
            (float(x_grid[j]) - dx / 2, float(y_grid[i]) - dy / 2),
            dx, dy, fill=False, edgecolor="#00ff88", linewidth=1.8,
        ))
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    tmin, tmax = float(np.nanmin(thr)), float(np.nanmax(thr))
    tstr = (f"1/W = {tmin:.3g} Hz" if abs(tmax - tmin) < 1e-9
            else f"1/W in [{tmin:.3g}, {tmax:.3g}] Hz")
    n_conv = int(np.sum(converged))
    ax.set_title(f"{x_var} x {y_var} — {n_conv}/{converged.size} converged (|err| < {tstr})")
    fig.tight_layout()

    png_tmp = png_path.with_suffix(".png.tmp")
    fig.savefig(str(png_tmp), dpi=120, format="png")
    plt.close(fig)
    png_tmp.replace(png_path)

    return pq_path, json_path, png_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-grid", type=int, default=10, help="grid points per axis")
    ap.add_argument("--pairs", type=int, default=None, help="only first N pairs (smoke test)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    all_pairs = list(itertools.combinations(VARS.keys(), 2))
    if args.pairs is not None:
        all_pairs = all_pairs[:args.pairs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    n_total = len(all_pairs)
    print(f"[{datetime.now():%H:%M:%S}] {n_total} pairs, {args.n_grid}x{args.n_grid} grid each "
          f"(T={T}s, dt={DT}, seed={SEED})")
    print(f"[{datetime.now():%H:%M:%S}] output dir: {args.out_dir}")

    n_skipped = 0
    n_run = 0

    for idx, (x_var, y_var) in enumerate(all_pairs, 1):
        t0 = time.monotonic()

        # Build grids and manifest up-front so we can hash before running.
        x_grid = np.linspace(*VARS[x_var]["range"], args.n_grid)
        y_grid = np.linspace(*VARS[y_var]["range"], args.n_grid)
        frozen = {k: float(v["frozen"]) for k, v in VARS.items()}
        manifest = build_manifest(x_var, y_var, x_grid, y_grid, frozen)
        h = hash_of(manifest)

        pq = args.out_dir / f"{h}.parquet"
        js = args.out_dir / f"{h}.json"
        pg = args.out_dir / f"{h}.png"
        if pq.exists() and js.exists() and pg.exists():
            print(f"[{datetime.now():%H:%M:%S}] [{idx}/{n_total}] {x_var} x {y_var}: "
                  f"skip (hash {h})")
            n_skipped += 1
            continue

        print(f"[{datetime.now():%H:%M:%S}] [{idx}/{n_total}] {x_var} x {y_var}: "
              f"running (hash {h})...", flush=True)
        result = run_one_sweep(x_var, y_var, n_grid=args.n_grid)
        # Recompute hash from the actual grids (identical by construction).
        manifest = build_manifest(x_var, y_var, result["x_grid"], result["y_grid"], result["frozen"])
        h = hash_of(manifest)
        pq_path, json_path, png_path = write_outputs(result, manifest, h, args.out_dir)

        elapsed = time.monotonic() - t0
        total = time.monotonic() - start
        n_conv = int(np.sum(result["converged"]))
        print(f"[{datetime.now():%H:%M:%S}]     done in {elapsed:.1f}s (total {total:.1f}s), "
              f"{n_conv}/{result['converged'].size} converged, saved {png_path.name}",
              flush=True)
        n_run += 1

    total = time.monotonic() - start
    print(f"[{datetime.now():%H:%M:%S}] all done in {total:.1f}s "
          f"({n_run} run, {n_skipped} skipped)")


if __name__ == "__main__":
    main()
