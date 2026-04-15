# CLAUDE.md

## Project

Neuromodulated STDP simulation: pre-synaptic neurons → 1 post-synaptic LIF neuron, implementing three-factor learning rules from Frémaux & Gerstner (2016).

The **verified path** is 1 pre → 1 post with `target_rate` reward and `covariance` neuromodulator (`notebooks/verified.py`). Everything else (N-pre, biofeedback, contingent, gated/surprise/constant neuromod, rate-estimator comparisons, parameter sweeps) is exploratory and lives in `notebooks/exploratory.py`.

References: `docs/references.bib`. Write-ups: `docs/*.typ`.

## Commands

```bash
uv sync                                       # install (uv + hatchling)
uv run marimo edit notebooks/verified.py      # interactive verified path
uv run marimo edit notebooks/exploratory.py   # interactive exploratory work
uv run neuro --help                           # CLI sim runner: options and defaults
uv run pytest                                 # tests
typst compile docs/<name>.typ                 # build a write-up (PDFs are gitignored)
```

## Architecture

Single simulation module: `src/neuro/sim.py` — read `Params` for the parameter surface and `simulate()` for the event-driven/continuous loop.

- **Two orthogonal axes**: `neuromod_type` (how M derives from R) and `reward_signal` (what R measures). They combine freely.
- **Per-synapse broadcasting**: scalar `Params` fields are broadcast to length `n_pre`; tuples are per-synapse.
- **State layout**: a flat state vector with shared state followed by per-synapse blocks. Index helpers live in `sim.py` — use them rather than hardcoding offsets.
- **Synapse 0 is the "target"** (reward-paired) for contingent mode.
- **Recording**: `Recorder` hierarchy (Memory / HDF5 / Parquet / Multi). Use `ParquetRecorder` for long runs to avoid OOM (in-memory recording at sub-ms `dt` over hours blows up).
- **Caching**: `cache.py` (`cached_simulate`) hashes `Params` and reuses prior runs from `output/runs.db`.
- **Visualization**: matplotlib only (in notebooks). The plotly + HTTP server code in `sim.py` is legacy and pending removal — see `TODO.md`.

## Key design decisions

- **R is external in three-factor literature.** `target_rate` is a self-supervisory demonstration — `biofeedback` and `contingent` are the proper paradigms.
- **Weight update rule**: `dw_i/dt = M(t) · E_i(t)` — global modulator times per-synapse eligibility. Spatial credit assignment works because all synapses see the same M, but only the reward-paired synapse has high E when reward arrives.
- **Integration**: RK4 is O(dt⁴) for smooth ODE between spikes; spike events use threshold-crossing interpolation.
- **1-indexed keys** (`w1`, `w2`, …) match neuroscience convention and preserve compatibility with saved data.

## Conventions

- Tabular work uses **polars**, not pandas.
- New analysis goes in `notebooks/exploratory.py`. Promote to `notebooks/verified.py` only after the result is reproduced and reviewed.
- Long simulations: stream to parquet and downsample for plotting (see notebooks for the pattern).
