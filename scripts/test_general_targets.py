#!/usr/bin/env python3
"""Test convergence of general target functions for neuromodulated STDP.

Runs the model with reward_signal="target_rate" for several target function
types (transcendental, affine, polynomial), measures actual vs target post
rates, and plots convergence diagnostics.

Run:  uv run python scripts/test_general_targets.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuro.sim import Params, simulate, _compute_target_r_post

OUT = Path("output")

# ── Target function configurations ───────────────────────────────────

TARGET_CONFIGS: dict[str, tuple[str, dict]] = {
    "linear (baseline)": ("linear", {"a": 0.5}),
    "affine: 0.3r + 2":  ("affine", {"a": 0.3, "b": 2.0}),
    "quadratic":          ("quadratic", {"a": -0.03, "b": 0.8, "c": 0.5}),
    "sqrt":               ("sqrt", {"a": 1.5, "b": 0.5}),
    "log":                ("log", {"a": 1.5, "b": 1.5}),
    "sin":                ("sin", {"a": 2.0, "b": 5.0, "c": 0.3}),
    "power: r^0.6":       ("power", {"a": 1.0, "b": 0.5, "c": 0.6}),
}

R_PRE_RATES = [10.0, 20.0, 40.0]

# ── Convergence thresholds (from convergence_sweep.py) ───────────────

W_SILENT = 0.01
W_SATURATED = 9.9
W_STD_OSCILLATING = 1.0
REL_RATE_ERR = 0.3


def _run(target_func: str, target_func_params: dict, r_pre_rate: float = 20.0, T: float = 20.0) -> tuple[Params, dict]:
    p = Params(
        T=T,
        method="rk4",
        record_every=1e-3,
        reward_signal="target_rate",
        target_func=target_func,
        target_func_params=json.dumps(target_func_params),
        r_pre_rates=r_pre_rate,
    )
    rec = simulate(p)
    return p, rec


def _target_rate_hz(p: Params, r_pre_rate: float) -> float:
    """Compute the expected postsynaptic firing rate in Hz."""
    r_pre_trace = r_pre_rate * p.tau_r
    target_trace = _compute_target_r_post(p, r_pre_trace)
    return max(target_trace, 0.0) / p.tau_r


def _metrics(rec: dict, p: Params) -> dict:
    t = rec["t"]
    half = p.T / 2
    ps = rec["post_spike_times"]
    late = ps[ps >= half]
    actual_rate = len(late) / (p.T - half) if p.T > half else 0.0
    target_rate = _target_rate_hz(p, p.r_pre_rates[0])

    wl = rec["w1"][t >= half]
    w_mean = float(np.mean(wl)) if len(wl) else 0.0
    w_std = float(np.std(wl)) if len(wl) else 0.0

    has_nan = any(np.any(np.isnan(rec[k])) for k in ("w", "R", "M", "R_bar", "E", "V", "r_post"))
    has_inf = any(np.any(np.isinf(rec[k])) for k in ("w", "R", "M", "R_bar", "E", "V", "r_post"))

    rate_err = abs(actual_rate - target_rate)
    rel_err = rate_err / max(target_rate, 1e-6)

    return dict(
        actual_rate=actual_rate,
        target_rate=target_rate,
        rate_error=rate_err,
        relative_rate_error=rel_err,
        w_final=float(rec["w1"][-1]),
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


# ── Figure 1: Target function curves ────────────────────────────────

def plot_target_curves() -> None:
    r_pre = np.linspace(0.1, 20, 200)
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, (func, coeffs) in TARGET_CONFIGS.items():
        p = Params(target_func=func, target_func_params=json.dumps(coeffs))
        targets = [max(_compute_target_r_post(p, float(r)), 0.0) for r in r_pre]
        ax.plot(r_pre, targets, label=name, linewidth=2)

    ax.set_xlabel("r_pre (rate trace)", fontsize=12)
    ax.set_ylabel("target r_post (rate trace)", fontsize=12)
    ax.set_title("Target functions: r_post = f(r_pre)", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=10, color="gray", linestyle="--", alpha=0.5, label="r_pre at 20 Hz")
    fig.tight_layout()
    fig.savefig(OUT / "general_target_curves.png", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT / 'general_target_curves.png'}")


# ── Figure 2: Time-series panels ────────────────────────────────────

def plot_time_series() -> None:
    names = list(TARGET_CONFIGS.keys())
    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n), squeeze=False)

    for i, name in enumerate(names):
        func, coeffs = TARGET_CONFIGS[name]
        print(f"  Running {name} at 20 Hz ...", end="", flush=True)
        t0 = time.time()
        p, rec = _run(func, coeffs, r_pre_rates=20.0)
        m = _metrics(rec, p)
        status = _classify(m)
        elapsed = time.time() - t0
        print(f"  {status} ({elapsed:.1f}s)")

        t = rec["t"]

        # Weight evolution
        ax_w = axes[i, 0]
        ax_w.plot(t, rec["w1"], linewidth=0.8, color="#2a9d8f")
        ax_w.set_ylabel("w", fontsize=10)
        ax_w.set_title(f"{name}  [{status}]", fontsize=11, fontweight="bold")
        ax_w.grid(True, alpha=0.3)

        # Actual r_post vs target
        ax_r = axes[i, 1]
        ax_r.plot(t, rec["r_post"], linewidth=0.8, color="#264653", label="r_post")
        target_trace = max(_compute_target_r_post(p, p.r_pre_rates[0] * p.tau_r), 0.0)
        ax_r.axhline(y=target_trace, color="#e76f51", linestyle="--", linewidth=1.5, label=f"target = {target_trace:.2f}")
        ax_r.set_ylabel("r_post", fontsize=10)
        ax_r.legend(fontsize=8, loc="upper right")
        ax_r.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)", fontsize=10)
    axes[-1, 1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle("Convergence with general target functions (r_pre = 20 Hz)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "general_target_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT / 'general_target_timeseries.png'}")


# ── Figure 3: Actual vs target scatter ──────────────────────────────

def plot_actual_vs_target() -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(TARGET_CONFIGS)))  # type: ignore[attr-defined]

    all_results: list[dict] = []

    for ci, (name, (func, coeffs)) in enumerate(TARGET_CONFIGS.items()):
        for rpr in R_PRE_RATES:
            print(f"  Running {name} at {rpr:.0f} Hz ...", end="", flush=True)
            t0 = time.time()
            p, rec = _run(func, coeffs, r_pre_rates=rpr)
            m = _metrics(rec, p)
            status = _classify(m)
            elapsed = time.time() - t0
            print(f"  {status} ({elapsed:.1f}s)")

            m["name"] = name
            m["r_pre_rate"] = rpr
            m["status"] = status
            all_results.append(m)

            marker = "o" if status == "converged" else "x"
            ax.scatter(
                m["target_rate"], m["actual_rate"],
                color=colors[ci], marker=marker, s=80, zorder=3,
                label=f"{name} @ {rpr:.0f} Hz" if rpr == R_PRE_RATES[0] else None,
            )

    # Diagonal line
    lims = [0, max(m["target_rate"] for m in all_results) * 1.2]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="perfect match")

    ax.set_xlabel("Target post rate (Hz)", fontsize=12)
    ax.set_ylabel("Actual post rate (Hz)", fontsize=12)
    ax.set_title("Convergence: actual vs target firing rate", fontsize=14)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "general_target_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT / 'general_target_scatter.png'}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Name':<22s} {'r_pre':>5s} {'Target':>8s} {'Actual':>8s} {'RelErr':>8s} {'Status':<14s}")
    print("-" * 80)
    for m in all_results:
        print(f"{m['name']:<22s} {m['r_pre_rate']:5.0f} {m['target_rate']:8.1f} {m['actual_rate']:8.1f} {m['relative_rate_error']:8.2f} {m['status']:<14s}")
    print("=" * 80)

    n_converged = sum(1 for m in all_results if m["status"] == "converged")
    print(f"\nConverged: {n_converged}/{len(all_results)}")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(exist_ok=True)

    print("=== Figure 1: Target function curves ===")
    plot_target_curves()

    print("\n=== Figure 2: Time-series panels (20 Hz) ===")
    plot_time_series()

    print("\n=== Figure 3: Actual vs target scatter (multi-rate) ===")
    plot_actual_vs_target()

    print(f"\nAll plots saved to {OUT}/")


if __name__ == "__main__":
    main()
