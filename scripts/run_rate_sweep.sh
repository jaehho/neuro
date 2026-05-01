#!/usr/bin/env bash
# Focused 20x20 sweep of r_pre × r_target on the baseline path.
# Resumable via content-addressed hashing: already-completed hash is skipped.
# Outputs: output/sweeps/<hash>.{parquet,json,png}
set -euo pipefail

cd "$(dirname "$0")/.."
exec uv run python -u scripts/sweep_all_pairs.py \
    --n-grid 20 \
    --only r_pre,r_target \
    --out-dir output/sweeps \
    "$@"
