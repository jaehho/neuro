#!/usr/bin/env bash
# Run every C(10, 2) = 45 variable pair as a 16x16 2D sweep.
# Resumable via content-addressed hashing: already-completed pairs are skipped.
# Outputs: output/sweeps/<hash>.{parquet,json,png}
set -euo pipefail

cd "$(dirname "$0")/.."
exec uv run python -u scripts/sweep_all_pairs.py \
    --n-grid 16 \
    --out-dir output/sweeps \
    "$@"
