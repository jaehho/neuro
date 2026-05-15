"""Scratch experiment from low-rate-sweep/20260513_110517/cell_29_00."""
from __future__ import annotations

from neuro import Params, simulate

p = Params(
    T=10000,
    r_pre=(10.0,),
    r_target=23.5,
)

if __name__ == "__main__":
    run = simulate(p, name="scratch")
    print(f"  parquet: {run.parquet}")
    print(f"  duration: {run.duration_s:.1f}s, rows: {run.rows_written}")
    run.serve()
