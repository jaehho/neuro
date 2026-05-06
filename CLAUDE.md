# CLAUDE.md

## Project

Neuromodulated STDP simulation: pre-synaptic neurons → 1 post-synaptic LIF neuron, implementing three-factor learning rules from Frémaux & Gerstner (2016).

The **baseline path** is 1 pre → 1 post with `target_rate` reward and `covariance` neuromodulator (`notebooks/baseline.py`). The unified write-up is `docs/main.typ`; references are in `docs/references.bib`.

## Commands

```bash
uv sync                                       # install (uv + hatchling)
uv run marimo edit notebooks/baseline.py      # interactive baseline path
uv run neuro --help                           # CLI: long sim + zoom-adaptive viewer
uv run pytest                                 # tests
typst compile --root . docs/main.typ          # build the write-up (PDFs are gitignored)
```

## Architecture

The simulation package `src/neuro/` is split into focused modules:

- `params.py` — `Params` dataclass, state-vector layout, key generation. Read this first to see the parameter surface.
- `dynamics.py` — RHS, Euler/RK4 step, threshold-crossing interpolation, reward/modulation rules, target-function family, state packing.
- `simulate.py` — the event-driven / continuous main loop (`simulate()`). Optional `early_stop=StreamingConvergence(...)` breaks out once `r_post` settles.
- `recording.py` — `Recorder` hierarchy (Memory / HDF5 / Parquet / Multi). Use `ParquetRecorder` for long runs to avoid OOM.
- `convergence.py` — `ConvergenceCriterion`, `StreamingConvergence` (online early-stop), `check_steady_state` (post-hoc "did this run actually settle?").
- `io.py` — Parquet/HDF5/Polars loaders with min/max envelope downsampling for multi-million-point recordings.
- `plotting.py` — matplotlib summary, plotly summary, and the zoom-adaptive HTTP server (`serve_zoom_adaptive_plot`) that re-decimates from disk on each pan/zoom.
- `cli.py` — typer CLI entry point (`neuro = "neuro.cli:app"` in pyproject).
- `cache.py` — content-addressed simulation cache backed by `output/runs.db`.
- `sim.py` — compatibility shim re-exporting the previous flat namespace; existing `from neuro.sim import …` imports keep working. Prefer `from neuro import Params, simulate` for new code.

Cross-cutting facts:
- **Two orthogonal axes**: `neuromod_type` (how M derives from R) and `reward_signal` (what R measures). They combine freely.
- **Per-synapse broadcasting**: scalar `Params` fields are broadcast to length `n_pre`; tuples are per-synapse.
- **State layout**: flat numpy vector, shared state then per-synapse blocks. Use the index helpers (`_I_s_idx`, `_W_idx`, …) — never hardcode offsets.
- **Synapse 0 is the "target"** (reward-paired) for contingent mode.

## Where work belongs

- **CLI (`uv run neuro`)** — long runs (minutes to hours). Streams to parquet via `ParquetRecorder`, registers in the cache. Launch `--plot-backend server` for the zoom-adaptive viewer on `127.0.0.1:8050` with live parquet re-decimation on every scroll-zoom.
- **Scripts (`scripts/<name>.py`)** — sweeps and other batch simulations. Examples: `sweep_target_ceiling.py` (2D `r_pre` × `r_target` ceiling sweep), `sweep_all_pairs.py`. Outputs land in `output/sweeps/<hash>.{parquet,json,png}` keyed by manifest hash so re-running is idempotent.
- **Notebooks** — quick scratchwork + visualisation only. **Do not put real simulations in notebooks** — they take a long time to run and marimo doesn't persist outputs across kernels. Use a notebook to load a parquet that a script produced and explore it interactively.
- **Standalone viewer** — extended drill-down into one parquet; keep it open in a separate tab while iterating in a notebook.

## Key design decisions

- **R is external in three-factor literature.** `target_rate` is a self-supervisory demonstration — `biofeedback` and `contingent` are the proper paradigms. The 1-pre baseline maps to Fetz-style operant conditioning of a single neuron; mathematically the rule is REINFORCE with a value baseline (Williams 1992; Frémaux & Gerstner §4.3). See `docs/main.typ` Introduction.
- **Weight update rule**: `dw_i/dt = M(t) · E_i(t)` — global modulator times per-synapse eligibility. Spatial credit assignment works because all synapses see the same M, but only the reward-paired synapse has high E when reward arrives.
- **Integration**: RK4 is O(dt⁴) for smooth ODE between spikes; spike events use threshold-crossing interpolation. Effective order on the full state is below 4 because of spike discontinuities.
- **Hard firing-rate ceiling**: `1/tau_ref ≈ 333 Hz` with default `tau_ref = 3 ms`. The post can't fire faster than refractory allows; a *driveable* soft ceiling well below that bites first when input drive is limited.
- **1-indexed keys** (`w1`, `w2`, …) match neuroscience convention and preserve compatibility with saved data.

## Conventions

- Tabular work uses **polars**, not pandas.
- Long simulations: stream to parquet and downsample for plotting (see scripts for the pattern).
- Sweep cells: pass a `StreamingConvergence` to `simulate(early_stop=...)` to short-circuit once `r_post` is steady, and check `detector.converged_at is not None` to flag cells that didn't settle within `T_max`.
