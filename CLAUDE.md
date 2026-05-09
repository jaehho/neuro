# CLAUDE.md

## Project

Neuromodulated STDP simulation: pre-synaptic neurons → 1 post-synaptic LIF neuron, implementing three-factor learning rules from Frémaux & Gerstner (2016).

The **baseline path** is 1 pre → 1 post with `target_rate` reward and `covariance` neuromodulator. Run it via `uv run neuro run` (defaults match the baseline). The unified write-up is `docs/main.typ`; references are in `docs/references.bib`.

## Commands

```bash
uv sync                                       # install (uv + hatchling)
uv run neuro --help                           # CLI: run, list, show, sweep
uv run neuro run                              # interactive single-run wizard
uv run neuro list                             # browse cached runs
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

- **CLI (`uv run neuro`)** — primary interface. Subcommands: `run` (single sim, all Params exposed, `--interactive` wizard or `--config FILE`), `list`/`show <hash>` (browse + view cached runs), `sweep run/show/cell/index` (2D sweeps and drill-down), `config init [run|sweep]` (write a TOML with all defaults + descriptions), `cache merge` (integrate a remote runs.db). Cells route through `cached_simulate` so sweep cells share `output/runs.db` with single runs. The zoom-adaptive HTTP viewer (`127.0.0.1:8050`) re-decimates parquet on every scroll-zoom.
- **Examples (`examples/*.toml`)** — starter configs (`baseline-run`, `credit-assignment-run`, `ceiling-sweep`). Copy and edit, then pass via `--config`.
- **Notebooks** — quick scratchwork + visualisation only. **Do not put real simulations in notebooks** — they take a long time to run and marimo doesn't persist outputs across kernels. Use a notebook to load a parquet that the CLI produced and explore it interactively.

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
