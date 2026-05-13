"""Baseline 1-pre → 1-post run with default Params.

The canonical Fetz-style demo: a single neuron learning to track a fixed
target firing rate via three-factor STDP. Run with

    uv run python experiments/baseline.py
"""
from __future__ import annotations

from neuro import Params, simulate

p = Params(r_pre=(21.0,), r_target=10)

if __name__ == "__main__":
    run = simulate(p, name="baseline")
    print(f"  parquet: {run.parquet}")
    print(f"  duration: {run.duration_s:.1f}s, rows: {run.rows_written}")
    run.serve()  # opens http://127.0.0.1:8050/
