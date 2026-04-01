#!/usr/bin/env python3
"""
Compare neuromodulator types and reward signals for neuromodulated STDP.

Two orthogonal axes (Frémaux & Gerstner 2016):
  1. neuromod_type: how M is derived from R (covariance, gated, surprise, constant)
  2. reward_signal: what R measures (target_rate, biofeedback)

Produces:
  1. Time-series overlay (w, r_post, E, M) across neuromod types
  2. Summary bar chart (final rate and weight)
  3. Per-configuration detail panels (R, R_bar, M dynamics)
  4. Biofeedback comparison across neuromod types
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuro.sim import Params, simulate

OUT = Path("output")
OUT.mkdir(exist_ok=True)

# ── Configuration matrix ────────────────────────────────────────────
#
# Compare neuromod types with target_rate reward (self-supervisory demo)
# and biofeedback reward (proper three-factor paradigm).

NEUROMOD_CONFIGS = {
    "covariance + target_rate": dict(
        neuromod_type="covariance",
        reward_signal="target_rate",
        target_func="fixed",
        r_target=5.0,
    ),
    "gated + target_rate": dict(
        neuromod_type="gated",
        reward_signal="target_rate",
        target_func="fixed",
        r_target=5.0,
    ),
    "surprise": dict(
        neuromod_type="surprise",
        reward_signal="target_rate",
        target_func="fixed",
        r_target=5.0,
    ),
    "constant (non-modulated)": dict(
        neuromod_type="constant",
        reward_signal="target_rate",
        target_func="fixed",
        r_target=5.0,
    ),
}

BIOFEEDBACK_CONFIGS = {
    "biofeedback + covariance": dict(
        neuromod_type="covariance",
        reward_signal="biofeedback",
        reward_delay=0.5,
        reward_amount=1.0,
        reward_tau=0.2,
    ),
    "biofeedback + gated": dict(
        neuromod_type="gated",
        reward_signal="biofeedback",
        reward_delay=0.5,
        reward_amount=1.0,
        reward_tau=0.2,
    ),
}


def _run(overrides, T=10.0, record_every=1e-3):
    p = Params(T=T, method="rk4", record_every=record_every, **overrides)
    rec = simulate(p)
    return p, rec


def _metrics(rec, p):
    t = rec["t"]
    half = p.T / 2
    ps = rec["post_spike_times"]
    late = ps[ps >= half]
    rate = len(late) / (p.T - half) if p.T > half else 0.0
    wl = rec["w1"][t >= half]
    return dict(
        post_rate=rate,
        w_final=float(rec["w1"][-1]),
        w_mean=float(np.mean(wl)) if len(wl) else 0.0,
        w_std=float(np.std(wl)) if len(wl) else 0.0,
        n_post=len(ps),
    )


def _verify(rec, p, name):
    """Sanity checks: no NaN/Inf, weight in bounds."""
    for key in ("w1", "w2", "R", "M", "R_bar", "E1", "E2"):
        arr = rec[key]
        if np.any(np.isnan(arr)):
            print(f"    WARNING: NaN in {key} for {name}")
            return False
        if np.any(np.isinf(arr)):
            print(f"    WARNING: Inf in {key} for {name}")
            return False
    for wkey in ("w1", "w2"):
        w = rec[wkey]
        if np.any(w < -1e-9) or np.any(w > p.wmax + 1e-9):
            print(f"    WARNING: {wkey} out of [0, wmax] for {name}")
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Neuromodulator Type Comparison (Frémaux Eq. 14)")
    print("=" * 60)

    results = {}
    for configs, label in [(NEUROMOD_CONFIGS, "neuromod"), (BIOFEEDBACK_CONFIGS, "biofeedback")]:
        for name, overrides in configs.items():
            print(f"  {name:35s}", end="", flush=True)
            try:
                p, rec = _run(overrides)
                ok = _verify(rec, p, name)
                m = _metrics(rec, p)
                results[name] = (p, rec, m)
                status = "OK" if ok else "WARN"
                print(f"  [{status}]  rate={m['post_rate']:6.1f} Hz  "
                      f"w_final={m['w_final']:.4f}  spikes={m['n_post']}")
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()
                results[name] = None

    valid = {n: d for n, d in results.items() if d is not None}

    # ── Plot 1: Time-series overlay ──────────────────────────────────

    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    panels = [
        ("w1",     "Synaptic weight w1(t)"),
        ("r_post", "Post-synaptic rate trace r_post(t)"),
        ("E1",     "Eligibility trace E1(t)"),
        ("M",      "Modulation signal M(t)"),
    ]
    for ax, (key, title) in zip(axes, panels):
        for name, (p, rec, m) in valid.items():
            ax.plot(rec["t"], rec[key], label=name, alpha=0.8, linewidth=0.8)
        ax.set(ylabel=key, title=title)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    path = OUT / "neuromod_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  -> {path}")

    # ── Plot 2: Summary bar charts ───────────────────────────────────

    names = list(valid.keys())
    rates = [valid[n][2]["post_rate"] for n in names]
    weights = [valid[n][2]["w_final"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.barh(names, rates, color="tab:blue", alpha=0.7)
    ax1.axvline(10.0, color="red", ls="--", alpha=0.5, label="10 Hz ref")
    ax1.set(xlabel="Post rate (Hz, 2nd half)",
            title="Equilibrium firing rate by configuration")
    ax1.legend()
    ax1.grid(True, alpha=0.2, axis="x")

    ax2.barh(names, weights, color="tab:orange", alpha=0.7)
    ax2.axvline(2.0, color="gray", ls="--", alpha=0.5, label="w0=2.0")
    ax2.set(xlabel="w_final", title="Final weight by configuration")
    ax2.legend()
    ax2.grid(True, alpha=0.2, axis="x")

    fig.tight_layout()
    path = OUT / "neuromod_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")

    # ── Plot 3: Per-config detail panels (R, R_bar, M) ───────────────

    n = len(valid)
    fig, axes = plt.subplots(n, 3, figsize=(18, 3 * n), sharex=True)
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (name, (p, rec, m)) in enumerate(valid.items()):
        t = rec["t"]
        axes[i, 0].plot(t, rec["R"], linewidth=0.5)
        axes[i, 0].set(ylabel="R", title=f"{name}: Reward R(t)")
        axes[i, 0].grid(True, alpha=0.15)

        axes[i, 1].plot(t, rec["R_bar"], linewidth=0.5)
        axes[i, 1].set(ylabel="R_bar", title=f"{name}: R_bar(t)")
        axes[i, 1].grid(True, alpha=0.15)

        axes[i, 2].plot(t, rec["M"], linewidth=0.5)
        axes[i, 2].set(ylabel="M", title=f"{name}: Modulation M(t)")
        axes[i, 2].grid(True, alpha=0.15)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[-1, 2].set_xlabel("Time (s)")
    fig.tight_layout()
    path = OUT / "neuromod_details.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")

    print(f"\n{'=' * 60}")
    print(f"  All plots in {OUT.resolve()}")
    print(f"{'=' * 60}")
