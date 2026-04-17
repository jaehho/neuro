# CLAUDE.md

## Project

Neuromodulated STDP simulation: pre-synaptic neurons → 1 post-synaptic LIF neuron, implementing three-factor learning rules from Frémaux & Gerstner (2016).

The **baseline path** is 1 pre → 1 post with `target_rate` reward and `covariance` neuromodulator (`notebooks/baseline.py`). Each exploratory notebook under `notebooks/` varies one axis from that baseline — run `ls notebooks/` to see the current set.

References: `docs/references.bib`. Write-ups: `docs/*.typ`.

## Commands

```bash
uv sync                                       # install (uv + hatchling)
uv run marimo edit notebooks/baseline.py      # interactive baseline path
uv run marimo edit notebooks/<topic>.py       # see ls notebooks/ for topics
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
- **Visualization**: plotly in notebooks for trajectory plots (hover, client-side pan/zoom); matplotlib for grid/heatmap analysis (`imshow`) where plotly's scattergl doesn't help. The standalone zoom-adaptive plotly viewer in `sim.py::serve_zoom_adaptive_plot` is the right tool for drilling into multi-million-point parquet runs — marimo captures box-select, not scroll-zoom, and can't mutate widgets after creation, so it's not a substitute.

## Where work belongs

- **CLI (`uv run neuro`)** — long runs (minutes to hours). Streams to parquet via `ParquetRecorder`, registers in the cache. Launch `--plot-backend server` for the zoom-adaptive viewer on `127.0.0.1:8050` with live parquet re-decimation on every scroll-zoom.
- **Notebooks** — short/medium runs (seconds to a couple minutes) and analysis of CLI-produced parquets. `cached_simulate` cache-hits whatever the CLI already ran with matching `Params`.
- **Standalone viewer** — extended drill-down into one parquet; keep it open in a separate tab while iterating in a notebook.

## Key design decisions

- **R is external in three-factor literature.** `target_rate` is a self-supervisory demonstration — `biofeedback` and `contingent` are the proper paradigms.
- **Weight update rule**: `dw_i/dt = M(t) · E_i(t)` — global modulator times per-synapse eligibility. Spatial credit assignment works because all synapses see the same M, but only the reward-paired synapse has high E when reward arrives.
- **Integration**: RK4 is O(dt⁴) for smooth ODE between spikes; spike events use threshold-crossing interpolation.
- **1-indexed keys** (`w1`, `w2`, …) match neuroscience convention and preserve compatibility with saved data.

## Conventions

- Tabular work uses **polars**, not pandas.
- New analysis: create `notebooks/<topic>.py` with its own topic-scoped marimo app. Each notebook should vary one axis from `baseline.py` and document that axis in its docstring. Keep the reactive graph small — don't mix orthogonal analyses in one file (stale variable-name collisions bite).
- Long simulations: stream to parquet and downsample for plotting (see notebooks for the pattern).
