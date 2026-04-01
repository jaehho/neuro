#!/usr/bin/env python3
"""
Deep analysis of the neuromodulated STDP model.

Generates:
  1. Spectrograms of all state signals (time-frequency decomposition)
  2. Derivative time-series (numerical dX/dt for each variable)
  3. RK4 convergence verification (error vs dt, estimated order)
  4. Parameter sensitivity sweeps (alpha, eta, tau_e, tau_Rbar, w0, R_m)
  5. Initial-condition sensitivity (weight trajectories from different w0)
  6. Euler vs RK4 trajectory comparison
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram as _spectrogram

from neuro.sim import Params, simulate

OUT = Path("output")
OUT.mkdir(exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────

def _run(T=10.0, method="rk4", record_every=1e-4, **kw):
    p = Params(T=T, method=method, record_every=record_every, **kw)
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
        target=p.alpha * p.r_pre_rates[0],
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. SPECTROGRAMS
# ═══════════════════════════════════════════════════════════════════════

def plot_spectrograms(rec, p):
    """Time-frequency decomposition of each recorded signal.

    Uses scipy.signal.spectrogram (STFT).  Window sizes and frequency
    ceilings are tuned per variable: fast variables (V, I_s) get short
    windows for temporal resolution; slow variables (w, R_bar) get long
    windows for frequency resolution.
    """
    t = rec["t"]
    dt = float(t[1] - t[0]) if len(t) > 1 else p.dt
    fs = 1.0 / dt

    specs = [
        ("V",       256,  500),
        ("I_s",     256,  500),
        ("x_pre",   512,  200),
        ("y_post",  512,  200),
        ("E",      1024,   50),
        ("r_pre",  1024,  100),
        ("r_post", 1024,  100),
        ("w",      2048,   20),
        ("R",      1024,   50),
        ("M",      1024,   50),
    ]

    fig, axes = plt.subplots(len(specs), 1, figsize=(16, 2.8 * len(specs)),
                             sharex=True)
    for ax, (key, nperseg, fmax) in zip(axes, specs):
        sig = np.asarray(rec[key], dtype=np.float64)
        n = min(nperseg, len(sig) // 4)
        if n < 16:
            ax.text(0.5, 0.5, "insufficient data",
                    transform=ax.transAxes, ha="center")
            ax.set_title(key)
            continue
        f, ts, Sxx = _spectrogram(sig, fs=fs, nperseg=n, noverlap=n // 2)
        mask = f <= fmax
        db = 10 * np.log10(Sxx[mask] + 1e-30)
        ax.pcolormesh(ts, f[mask], db, shading="gouraud", cmap="viridis")
        ax.set_ylabel("Hz")
        ax.set_title(key)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "spectrograms.png", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT / 'spectrograms.png'}")


# ═══════════════════════════════════════════════════════════════════════
# 2. DERIVATIVES
# ═══════════════════════════════════════════════════════════════════════

def plot_derivatives(rec, p):
    """Numerical dX/dt for each state variable (central finite difference)."""
    t = rec["t"]
    dt = float(t[1] - t[0]) if len(t) > 1 else p.dt
    keys = ["V", "I_s", "x_pre", "y_post", "E",
            "r_pre", "r_post", "R_bar", "w", "M"]

    fig, axes = plt.subplots(len(keys), 1,
                             figsize=(16, 2.2 * len(keys)), sharex=True)
    for ax, k in zip(axes, keys):
        ax.plot(t, np.gradient(rec[k], dt), linewidth=0.3, alpha=0.8)
        ax.set_ylabel(f"d{k}/dt")
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Numerical derivatives (central finite-difference)", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "derivatives.png", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT / 'derivatives.png'}")


# ═══════════════════════════════════════════════════════════════════════
# 3. RK4 CONVERGENCE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

def verify_rk4():
    """Compare Euler vs RK4 error convergence rates.

    Uses a very-fine-dt RK4 run as the reference truth.  Expects Euler
    to show O(dt) convergence and RK4 to show O(dt^4), modulo the
    spike-event discontinuities that reduce effective order.
    """
    T = 1.0
    # Reference at dt = 10 us
    _, rec_ref = _run(T=T, record_every=1e-5, dt=1e-5, method="rk4")
    V_ref = rec_ref["V"][-1]
    w_ref = rec_ref["w"][-1]

    dts = [5e-4, 2e-4, 1e-4, 5e-5]
    errs = {k: [] for k in ("euler_V", "rk4_V", "euler_w", "rk4_w")}

    for dt_val in dts:
        for m in ("euler", "rk4"):
            _, r = _run(T=T, dt=dt_val, method=m, record_every=dt_val)
            errs[f"{m}_V"].append(abs(r["V"][-1] - V_ref))
            errs[f"{m}_w"].append(abs(r["w"][-1] - w_ref))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    da = np.array(dts)
    for ax, v in [(ax1, "V"), (ax2, "w")]:
        ee = np.array(errs[f"euler_{v}"])
        er = np.array(errs[f"rk4_{v}"])
        ax.loglog(dts, ee, "o-", label="Euler")
        ax.loglog(dts, er, "s-", label="RK4")
        if ee[0] > 0:
            ax.loglog(da, ee[0] * (da / dts[0]) ** 1,
                      ":", alpha=0.4, label="O(dt)")
        if er[0] > 0:
            ax.loglog(da, max(er[0], 1e-15) * (da / dts[0]) ** 4,
                      ":", alpha=0.4, label="O(dt^4)")
        ax.set(xlabel="dt (s)", ylabel=f"|err {v}|",
               title=f"Convergence - {v}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "rk4_convergence.png", dpi=150)
    plt.close(fig)

    for name, e in errs.items():
        if e[0] > 1e-15 and e[-1] > 1e-15:
            order = np.log(e[0] / e[-1]) / np.log(dts[0] / dts[-1])
            print(f"  {name:10s} -> estimated order = {order:.2f}")
        else:
            print(f"  {name:10s} -> errors near machine eps")
    print(f"  -> {OUT / 'rk4_convergence.png'}")


# ═══════════════════════════════════════════════════════════════════════
# 4. PARAMETER SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════

def sensitivity():
    """Sweep key parameters and measure post-rate and final weight.

    Each simulation is T=10 s (2 x tau_Rbar) so slow variables
    approximately reach steady-state.  Metrics are measured over
    the second half to discard initial transients.

    Theoretically interesting parameters:
      alpha     - reward landscape shape; controls equilibrium rate ratio
      eta_plus  - LTP step; too small => no learning; too large => instability
      tau_e     - eligibility window; must span reward delay or credit lost
      tau_Rbar  - baseline speed; too fast => M ~0; too slow => noisy M
      w0        - initial weight; below a critical value the neuron is silent
                  and E=0 forever (silent synapse problem)
      R_m       - membrane resistance; sets gain of synaptic drive on V
    """
    T = 10.0
    sweeps = {
        "alpha":    np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9]),
        "eta_plus": np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3]),
        "tau_e":    np.array([0.1, 0.2, 0.5, 1.0, 2.0]),
        "tau_Rbar": np.array([1.0, 2.0, 5.0, 10.0, 20.0]),
        "w0":       np.array([0.5, 1.0, 2.0, 5.0, 8.0]),
        "R_m":      np.array([10.0, 20.0, 50.0, 80.0, 120.0]),
    }

    results = {}
    for pname, vals in sweeps.items():
        print(f"    {pname}", end="", flush=True)
        outs = []
        for v in vals:
            try:
                p, rec = _run(T=T, record_every=1e-3, **{pname: float(v)})
                outs.append(_metrics(rec, p))
            except Exception as ex:
                print(f" [FAIL@{v}: {ex}]", end="")
                outs.append(dict(post_rate=np.nan, w_final=np.nan,
                                 target=np.nan))
            print(".", end="", flush=True)
        results[pname] = (vals, outs)
        print()

    n = len(sweeps)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.2 * n))
    for i, (pname, (vals, outs)) in enumerate(results.items()):
        rates = [o["post_rate"] for o in outs]
        targets = [o.get("target", np.nan) for o in outs]
        wf = [o["w_final"] for o in outs]

        axes[i, 0].plot(vals, rates, "o-", label="post rate")
        axes[i, 0].plot(vals, targets, "--", alpha=0.5, label="target")
        axes[i, 0].set(xlabel=pname, ylabel="Hz",
                       title=f"Post-synaptic rate vs {pname}")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.2)

        axes[i, 1].plot(vals, wf, "s-", color="tab:orange")
        axes[i, 1].set(xlabel=pname, ylabel="w_final",
                       title=f"Final weight vs {pname}")
        axes[i, 1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT / "sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT / 'sensitivity.png'}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# 5. INITIAL-CONDITION SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════

def ic_sensitivity():
    """Show whether different initial weights converge to the same equilibrium.

    Key question: is there a critical w0 below which the neuron stays silent
    and learning is impossible? (the "silent synapse" fixed point)
    """
    w0s = [0.5, 1.0, 2.0, 5.0, 8.0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for w0 in w0s:
        _, rec = _run(T=10.0, record_every=1e-3, w0=w0)
        ax1.plot(rec["t"], rec["w1"], label=f"w0={w0}")
        ax2.plot(rec["t"], rec["r_post"], label=f"w0={w0}")
    ax1.set(ylabel="w(t)", title="Weight trajectories from different initial conditions")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    ax2.set(xlabel="Time (s)", ylabel="r_post(t)",
            title="Post-synaptic rate traces")
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "ic_sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT / 'ic_sensitivity.png'}")


# ═══════════════════════════════════════════════════════════════════════
# 6. EULER vs RK4 TRAJECTORY COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def euler_vs_rk4():
    """Overlay Euler and RK4 trajectories to visualise method differences."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for method in ("euler", "rk4"):
        _, rec = _run(T=2.0, method=method, record_every=1e-4)
        axes[0].plot(rec["t"], rec["V"], label=method,
                     linewidth=0.5, alpha=0.7)
        axes[1].plot(rec["t"], rec["w1"], label=method)
        axes[2].plot(rec["t"], rec["E1"], label=method)
    axes[0].set(ylabel="V (mV)", title="Membrane potential")
    axes[1].set(ylabel="w", title="Synaptic weight")
    axes[2].set(xlabel="Time (s)", ylabel="E", title="Eligibility trace")
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.2)
    fig.suptitle("Euler vs RK4 trajectories (dt = 0.1 ms)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "euler_vs_rk4.png", dpi=150)
    plt.close(fig)
    print(f"  -> {OUT / 'euler_vs_rk4.png'}")


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  Neuromodulated STDP - Deep Analysis")
    print("=" * 55)

    print("\n[1/6] Reference simulation (T=10 s, RK4, dt=0.1 ms)...")
    p0, rec0 = _run(T=10.0, record_every=1e-4)
    m0 = _metrics(rec0, p0)
    print(f"  Samples:     {len(rec0['t']):,}")
    print(f"  Pre spikes:  {len(rec0['pre_spike_times']):,}")
    print(f"  Post spikes: {len(rec0['post_spike_times']):,}")
    print(f"  Post rate (2nd half): {m0['post_rate']:.1f} Hz  "
          f"(target {m0['target']:.1f} Hz)")
    print(f"  Final weight: {m0['w_final']:.4f}")
    print(f"  Weight std (2nd half): {m0['w_std']:.4f}")

    print("\n[2/6] Spectrograms...")
    plot_spectrograms(rec0, p0)

    print("\n[3/6] Derivatives...")
    plot_derivatives(rec0, p0)

    print("\n[4/6] RK4 convergence...")
    verify_rk4()

    print("\n[5/6] Parameter sensitivity...")
    sensitivity()

    print("\n[6/6] IC sensitivity + Euler vs RK4...")
    ic_sensitivity()
    euler_vs_rk4()

    print(f"\n{'=' * 55}")
    print(f"  All plots saved to {OUT.resolve()}")
    print(f"{'=' * 55}")
