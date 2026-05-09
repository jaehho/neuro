"""2D parameter sweep machinery.

A sweep iterates over a grid of (x_var, y_var) values, with everything
else held at a frozen baseline.  Cells route through ``cached_simulate``
so each cell's full trace lives in ``output/<cell_hash>.parquet`` and is
shared with single-run cache entries — drill into any cell directly.

Outputs:
  output/<cell_hash>.parquet              per-cell trace (shared cache)
  output/sweeps/<sweep_hash>.json         manifest (axes, frozen, cell hashes)
  output/sweeps/<sweep_hash>.parquet      summary table, one row per cell
  output/sweeps/<sweep_hash>.png          error heatmap
"""
from __future__ import annotations

import hashlib
import html
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

from neuro.cache import cached_simulate, params_hash
from neuro.convergence import (
    ConvergenceCriterion,
    MultiConvergence,
    StreamingConvergence,
    check_steady_state,
)
from neuro.params import Params


# ── Variable application ──────────────────────────────────────────

_PER_SYNAPSE_FIELDS = {"r_pre_rates", "w0", "I_s0", "x_pre0", "E0"}
_PER_SYNAPSE_ALIASES = {"r_pre": "r_pre_rates"}
_SCALAR_ALIASES = {"W": "rate_window"}


def apply_var(p: Params, name: str, value: float) -> Params:
    """Return a Params with the named knob set to *value*.

    Recognised names:
      - any direct Params field
      - 'r_pre' -> broadcasts to r_pre_rates
      - 'w0', 'I_s0', 'x_pre0', 'E0' -> broadcast scalars to per-synapse tuples
      - 'W'    -> rate_window
      - 'eta'  -> eta_plus = eta_minus = value
    """
    if name == "eta":
        return replace(p, eta_plus=value, eta_minus=value)
    target = _PER_SYNAPSE_ALIASES.get(name, _SCALAR_ALIASES.get(name, name))
    if target in _PER_SYNAPSE_FIELDS:
        return replace(p, **{target: tuple(float(value) for _ in range(p.n_pre))})
    if not hasattr(p, target):
        raise ValueError(f"Unknown sweep variable {name!r} (no Params field {target!r})")
    return replace(p, **{target: value})


def sweep_var_choices() -> list[str]:
    """Field names (incl. aliases) that ``apply_var`` accepts."""
    fields_list = sorted(set(p.name for p in Params.__dataclass_fields__.values()))
    return fields_list + ["r_pre", "W", "eta"]


# ── Sweep specification ───────────────────────────────────────────

@dataclass
class SweepSpec:
    x_var: str
    y_var: str
    x_grid: list[float]
    y_grid: list[float]
    x_log: bool
    y_log: bool
    base: Params
    criterion: ConvergenceCriterion        # rate criterion (flat + on-target)
    weight_criterion: ConvergenceCriterion | None = None  # optional weight-flatness criterion
    target_fixed: float | None = None

    def cells(self) -> list[tuple[int, int, float, float]]:
        return [
            (i, j, float(self.x_grid[j]), float(self.y_grid[i]))
            for i in range(len(self.y_grid))
            for j in range(len(self.x_grid))
        ]

    def n_cells(self) -> int:
        return len(self.x_grid) * len(self.y_grid)


# ── Per-cell run (picklable for multiprocessing) ──────────────────

def _resolve_target(meta: dict, x_val: float, y_val: float, p: Params) -> float | None:
    if meta.get("target_fixed") is not None:
        return float(meta["target_fixed"])
    if meta["y_var"] == "r_target":
        return float(y_val)
    if meta["x_var"] == "r_target":
        return float(x_val)
    return float(p.r_target)


def _silent(it):
    return it


def _build_detector(rate_crit: ConvergenceCriterion,
                    weight_crit: ConvergenceCriterion | None,
                    target: float | None):
    """Build the early-stop detector: rate (flat+on-target) AND weight-flat."""
    rate_det = StreamingConvergence(criterion=rate_crit, target=target, signal="r_post")
    if weight_crit is None:
        return rate_det
    weight_det = StreamingConvergence(criterion=weight_crit, signal="w")
    return MultiConvergence(rate_det, weight_det)


def _run_one(args: tuple) -> dict:
    (i, j, x_val, y_val, base_kwargs, rate_crit_kwargs,
     weight_crit_kwargs, sweep_meta, cache_kwargs) = args
    p = Params(**base_kwargs)
    p = apply_var(p, sweep_meta["x_var"], x_val)
    p = apply_var(p, sweep_meta["y_var"], y_val)

    rate_crit = ConvergenceCriterion(**rate_crit_kwargs)
    weight_crit = ConvergenceCriterion(**weight_crit_kwargs) if weight_crit_kwargs else None
    target = _resolve_target(sweep_meta, x_val, y_val, p)
    detector = _build_detector(rate_crit, weight_crit, target)
    cell_hash = params_hash(p, early_stop=detector)

    t0 = time.monotonic()
    rec = cached_simulate(
        p,
        cache_dir=Path(cache_kwargs["cache_dir"]),
        chunk_rows=cache_kwargs["chunk_rows"],
        progress=_silent,
        early_stop=detector,
        quiet=True,
    )
    wall = time.monotonic() - t0

    pq = pl.read_parquet(rec["parquet_path"], columns=["t", "r_post", "w1"])
    spk = pl.read_parquet(rec["parquet_spikes_path"]).filter(pl.col("spike_type") == "post")
    t_arr = pq["t"].to_numpy()
    r_arr = pq["r_post"].to_numpy()
    w_arr = pq["w1"].to_numpy()
    post_times = spk["t"].to_numpy()

    rate_out = check_steady_state(t_arr, r_arr, target=target, criterion=rate_crit)
    weight_out = (
        check_steady_state(t_arr, w_arr, target=None, criterion=weight_crit)
        if weight_crit is not None else {"converged": True, "delta": 0.0}
    )
    t_end = float(t_arr[-1]) if len(t_arr) > 0 else 0.0
    half = max(t_end / 2.0, 1.0)
    late = post_times[post_times >= half]
    span = max(t_end - half, 1e-6)
    r_post_late = float(len(late)) / span

    return {
        "i": i, "j": j,
        "cell_hash": cell_hash,
        sweep_meta["x_var"]: float(x_val),
        sweep_meta["y_var"]: float(y_val),
        "r_post_late": r_post_late,
        "abs_error": float(rate_out.get("abs_error", abs(r_post_late - (target or 0.0)))),
        "rate_delta": float(rate_out.get("delta", float("nan"))),
        "weight_delta": float(weight_out.get("delta", float("nan"))),
        "w_final": float(w_arr[-1]) if len(w_arr) else float("nan"),
        "rate_flat": bool(rate_out.get("flat", False)),
        "rate_on_target": bool(rate_out.get("on_target", False)),
        "weight_flat": bool(weight_out.get("converged", False)),
        "converged": bool(rate_out.get("converged", False) and weight_out.get("converged", False)),
        "t_end": t_end,
        "wall_s": wall,
    }


# ── Driver ────────────────────────────────────────────────────────

def run_sweep(
    spec: SweepSpec,
    *,
    procs: int = 1,
    cache_dir: Path = Path("output"),
    chunk_rows: int = 100_000,
) -> pl.DataFrame:
    base_kwargs = asdict(spec.base)
    for k, v in list(base_kwargs.items()):
        if isinstance(v, list):
            base_kwargs[k] = tuple(v)

    sweep_meta = {
        "x_var": spec.x_var,
        "y_var": spec.y_var,
        "target_fixed": spec.target_fixed,
    }
    cache_kwargs = {"cache_dir": str(cache_dir), "chunk_rows": chunk_rows}
    rate_crit_kwargs = asdict(spec.criterion)
    weight_crit_kwargs = asdict(spec.weight_criterion) if spec.weight_criterion else None

    args_list = [
        (i, j, x, y, base_kwargs, rate_crit_kwargs, weight_crit_kwargs, sweep_meta, cache_kwargs)
        for (i, j, x, y) in spec.cells()
    ]

    rows: list[dict] = []
    if procs <= 1:
        for a in tqdm(args_list, desc="cells"):
            rows.append(_run_one(a))
    else:
        with mp.Pool(procs) as pool:
            for r in tqdm(pool.imap_unordered(_run_one, args_list), total=len(args_list), desc="cells"):
                rows.append(r)

    return pl.DataFrame(rows).sort([spec.y_var, spec.x_var])


# ── Manifest, hashing, output ─────────────────────────────────────

def build_manifest(spec: SweepSpec, df: pl.DataFrame) -> dict:
    base_dict = asdict(spec.base)
    return {
        "version": 3,
        "kind": "sweep_2d",
        "x_var": spec.x_var,
        "y_var": spec.y_var,
        "x_grid": list(spec.x_grid),
        "y_grid": list(spec.y_grid),
        "x_log": spec.x_log,
        "y_log": spec.y_log,
        "base_params": base_dict,
        "criterion": asdict(spec.criterion),
        "weight_criterion": asdict(spec.weight_criterion) if spec.weight_criterion else None,
        "target_fixed": spec.target_fixed,
        "n_cells": len(df),
        "cell_hashes": df.sort(["i", "j"])["cell_hash"].to_list(),
    }


def hash_manifest(manifest: dict) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_outputs(spec: SweepSpec, df: pl.DataFrame, out_dir: Path) -> tuple[Path, Path, Path, str]:
    manifest = build_manifest(spec, df)
    h = hash_manifest({k: v for k, v in manifest.items() if k != "cell_hashes"})
    out_dir.mkdir(parents=True, exist_ok=True)

    pq_path = out_dir / f"{h}.parquet"
    pq_tmp = pq_path.with_suffix(".parquet.tmp")
    df.write_parquet(str(pq_tmp))
    pq_tmp.replace(pq_path)

    meta = {"hash": h, "saved_at": datetime.now().isoformat(timespec="seconds"), **manifest}
    json_path = out_dir / f"{h}.json"
    _atomic_write_bytes(json_path, json.dumps(meta, indent=2, sort_keys=True).encode())

    nx, ny = len(spec.x_grid), len(spec.y_grid)
    err = np.full((ny, nx), np.nan)
    converged = np.zeros((ny, nx), dtype=bool)
    for row in df.iter_rows(named=True):
        err[int(row["i"]), int(row["j"])] = row["abs_error"]
        converged[int(row["i"]), int(row["j"])] = row["converged"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    ax = axes[0]
    im = ax.pcolormesh(spec.x_grid, spec.y_grid, err, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label="|<r_post> − target| (Hz)")
    if spec.x_log: ax.set_xscale("log")
    if spec.y_log: ax.set_yscale("log")
    ax.set_xlabel(spec.x_var); ax.set_ylabel(spec.y_var)
    n_conv = int(converged.sum())
    ax.set_title(f"abs_error  ({n_conv}/{converged.size} converged)")

    ax2 = axes[1]
    im2 = ax2.pcolormesh(spec.x_grid, spec.y_grid, converged.astype(int),
                         cmap="RdYlGn", vmin=0, vmax=1, shading="auto")
    fig.colorbar(im2, ax=ax2, label="converged (1) / didn't (0)")
    if spec.x_log: ax2.set_xscale("log")
    if spec.y_log: ax2.set_yscale("log")
    ax2.set_xlabel(spec.x_var); ax2.set_ylabel(spec.y_var)
    ax2.set_title("convergence flag")

    png_path = out_dir / f"{h}.png"
    png_tmp = png_path.with_suffix(".png.tmp")
    fig.savefig(str(png_tmp), dpi=120, format="png")
    plt.close(fig)
    png_tmp.replace(png_path)

    return pq_path, json_path, png_path, h


def build_grid(lo: float, hi: float, n: int, log: bool) -> list[float]:
    if log:
        return [float(v) for v in np.geomspace(lo, hi, n)]
    return [float(v) for v in np.linspace(lo, hi, n)]


# ── Index page (contact sheet of all sweeps) ──────────────────────

def build_index(sweep_dir: Path) -> Path:
    """Render output/sweeps/index.html: matrix of (y_var, x_var) thumbnails."""
    manifests = []
    for p in sorted(sweep_dir.glob("*.json")):
        try:
            m = json.loads(p.read_text())
            # Accept any manifest with x_var/y_var/hash — covers legacy and v2 formats.
            if "x_var" in m and "y_var" in m and "hash" in m:
                manifests.append(m)
        except json.JSONDecodeError:
            continue
    if not manifests:
        raise SystemExit(f"No sweep manifests in {sweep_dir}")

    # Group by (x_var, y_var); pick most recent saved_at for each pair.
    by_pair: dict[tuple[str, str], dict] = {}
    for m in manifests:
        key = (m["x_var"], m["y_var"])
        prev = by_pair.get(key)
        if prev is None or m.get("saved_at", "") > prev.get("saved_at", ""):
            by_pair[key] = m

    seen = set()
    for x, y in by_pair:
        seen.add(x); seen.add(y)
    order = sorted(seen)

    rows: list[str] = []
    head = "".join(f"<th class='vh top'>{html.escape(v)}</th>" for v in order)
    rows.append(f"<tr><th class='corner'></th>{head}</tr>")
    for y in order:
        cells = [f"<th class='vh side'>{html.escape(y)}</th>"]
        for x in order:
            m = by_pair.get((x, y))
            if m is None:
                cells.append("<td class='empty'></td>")
                continue
            png = f"{m['hash']}.png"
            js = f"{m['hash']}.json"
            tip = f"{x} × {y}  |  hash={m['hash']}  |  saved {m.get('saved_at', '?')}"
            cells.append(
                "<td>"
                f"<a href='{html.escape(png)}' title='{html.escape(tip)}'>"
                f"<img src='{html.escape(png)}' loading='lazy' alt='{html.escape(x)} x {html.escape(y)}'>"
                f"</a>"
                f"<div class='cap'>"
                f"<span class='lbl'>{html.escape(x)} × {html.escape(y)}</span>"
                f"<a href='{html.escape(js)}'>json</a>"
                f"</div>"
                "</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    doc = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><title>2D sweep contact sheet</title>
<style>
  body {{ margin:0; padding:0; background:#111; color:#eee; font:13px/1.4 system-ui, sans-serif; }}
  header {{ padding:12px 16px; border-bottom:1px solid #333; }}
  header h1 {{ margin:0; font-size:15px; font-weight:600; }}
  .wrap {{ overflow:auto; padding:8px; }}
  table {{ border-collapse:separate; border-spacing:3px; }}
  th, td {{ padding:0; }}
  th.vh {{ background:#1c1c1c; color:#ffd66b; padding:6px 10px;
          font:12px ui-monospace, monospace; white-space:nowrap; }}
  th.vh.top {{ position:sticky; top:0; z-index:2; }}
  th.vh.side {{ position:sticky; left:0; z-index:2; text-align:right; min-width:92px; }}
  td {{ background:#1c1c1c; vertical-align:top; }}
  td.empty {{ background:#161616; }}
  td img {{ display:block; width:240px; height:auto; }}
  td .cap {{ padding:3px 6px; font:11px ui-monospace, monospace; color:#bbb;
            border-top:1px solid #252525; display:flex; justify-content:space-between; gap:8px; }}
  td .cap a {{ color:#8cf; text-decoration:none; }}
</style></head><body>
<header><h1>2D sweep contact sheet: {html.escape(str(sweep_dir))} — {len(by_pair)} pairs</h1></header>
<div class='wrap'><table>
{chr(10).join(rows)}
</table></div></body></html>
"""
    out = sweep_dir / "index.html"
    out.write_text(doc, encoding="utf-8")
    return out
