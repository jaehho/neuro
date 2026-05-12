"""Baseline 2-pre → 1-post run with target-rate reward, covariance neuromod.

The canonical Fetz-style demo: a single neuron learning to track a fixed
target firing rate via three-factor STDP. Run with

    uv run python experiments/baseline.py
"""
from __future__ import annotations

from neuro import Params, simulate

p = Params(
    # Simulation
    T=60.0, dt=1e-4, seed=1, record_every=1e-3,

    # Pre input (Poisson, 20 Hz)
    n_pre=1,
    r_pre=20.0,
    poisson=True,

    # Rate estimation
    rate_mode="exp",
    rate_window=0.5,
    tau_r_post=0.5,

    # Three-factor rule
    M_rule="covariance",
    R_rule="target_rate",
    r_target=10.0,
    tau_Rbar=5.0,

    # Weights
    w0=2.0,
    wmax=10.0,
    eta_plus=1e-4,
    eta_minus=1e-4,
)

if __name__ == "__main__":
    run = simulate(p, name="baseline")
    print(f"  parquet: {run.parquet}")
    print(f"  duration: {run.duration_s:.1f}s, rows: {run.rows_written}")
    run.serve()  # opens http://127.0.0.1:8050/
