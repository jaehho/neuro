# CLAUDE.md

## Project

Neuromodulated STDP simulation: pre-synaptic neurons → 1 post-synaptic LIF neuron, implementing three-factor learning rules from Frémaux & Gerstner (2016). The unified write-up is `docs/main.typ`; references are in `docs/references.bib`.

## Workflow

Each experiment is one Python file under `experiments/` that builds a `Params` and calls `simulate(p, name=...)`. Runs are saved to a per-name run journal:

```
output/<name>/<YYYYMMDD_HHMMSS>.parquet           series
output/<name>/<YYYYMMDD_HHMMSS>.spikes.parquet    spike events
output/<name>/<YYYYMMDD_HHMMSS>.json              sidecar (Params + metadata)
```

Reload a past run with `load_latest("<name>")` and inspect with `.serve()` (zoom-adaptive HTTP viewer at 127.0.0.1:8050) or `.df()` for in-memory analysis.

```bash
uv sync
uv run python experiments/baseline.py            # one experiment per file
uv run pytest
typst compile --root . docs/main.typ             # build the write-up (PDFs gitignored)
```

There is no CLI and no config-file workflow — the experiment file *is* the config. There is no run cache; the journal is the source of truth and rerunning recomputes.

## Architecture

`src/neuro/` — the simulation package. Read in this order to onboard:

1. `params.py` — `Params` dataclass + state-vector layout. The parameter surface lives here; everything else is downstream.
2. `dynamics.py` — RHS, Euler/RK4 step, threshold-crossing interpolation, reward (`_reward`) and modulator (`_modulation`) rules, state packing. **The math.** Read the module docstring first — the equations are stated up front.
3. `simulate.py` — main event-driven / continuous loop, the `Run` dataclass, and journal helpers (`load_run`, `load_latest`, `list_runs`).
4. `recording.py` — `ParquetRecorder` (streaming write) + `spike_parquet_path` convention.
5. `convergence.py` — `ConvergenceCriterion`, `StreamingConvergence` (online early-stop), `check_steady_state` (post-hoc). **Note**: the early-stop is currently flagged for correction in `TODO.md`.
6. `io.py` — parquet loading + min/max envelope downsampling for the plot viewer.
7. `plotting.py` — `serve_zoom(parquet, params)`: stacked-panel figure that re-decimates from disk on every pan/zoom event.

Cross-cutting facts:

- **Two orthogonal axes** of three-factor rule: `M_rule` (how M derives from R) ∈ {`covariance`, `gated`} and `R_rule` (what R measures) ∈ {`target_rate`, `target_rate_linear`}. They combine freely.
- **Per-synapse broadcasting**: scalar `Params` fields are broadcast to length `n_pre`; tuples are per-synapse.
- **State layout**: flat numpy vector, shared state (`V, y_post, r_post, R_bar`) then per-synapse blocks `(I_s, x_pre, E, w)`. Use the index helpers (`_I_s_idx(i)`, `_W_idx(i)`, …) — never hardcode offsets.
- **Integrator**: `simulate()` always uses RK4. `_advance_state` retains a `method=` parameter for the `test_rk4.py::TestConvergenceOrder` validation tests; nothing else uses it.

## Where work belongs

- **`experiments/*.py`** — one Python file per experiment. Build a `Params`, call `simulate(p, name=…)`, optionally `run.serve()`. Sweeps go here too; see `experiments/ceiling_sweep.py` for the `multiprocessing.Pool` + heatmap pattern.
- **`notebooks/*.py`** — marimo notebooks for **scratchwork + visualisation only**. Long simulations belong in `experiments/`, not notebooks (marimo doesn't persist outputs across kernels). Use a notebook to load a parquet and explore it interactively.
- **`scripts/*.py`** — standalone Python utilities (e.g. RK4 nonlinear validation).

## Key design decisions

- **R is external in three-factor literature.** Both kept reward signals (`target_rate`, `target_rate_linear`) are self-supervisory demonstrations that close the loop on `r_post` itself; the 1-pre baseline maps to Fetz-style operant conditioning of a single neuron. Mathematically the rule is REINFORCE with a value baseline (Williams 1992; Frémaux & Gerstner §4.3). See `docs/main.typ` Introduction.
- **Weight update rule**: `dw_i/dt = M(t) · E_i(t)` — global modulator times per-synapse eligibility. Spatial credit assignment works because all synapses see the same M, but only the reward-paired synapse has high E when reward arrives.
- **Integration**: RK4 is O(dt⁴) for the smooth ODE between spikes; spike events use threshold-crossing interpolation. Effective order on the full state is below 4 because of spike discontinuities.
- **Hard firing-rate ceiling**: `1/tau_ref ≈ 333 Hz` with default `tau_ref = 3 ms`. The post can't fire faster than refractory allows; a *driveable* soft ceiling well below that bites first when input drive is limited.
- **1-indexed series keys** (`w1`, `w2`, …) match neuroscience convention and preserve compatibility with saved data.

## Conventions

- Tabular work uses **polars**, not pandas.
- Long simulations stream to parquet; the zoom viewer re-decimates from disk on every scroll-zoom, so plotting million-row recordings stays responsive.
- Sweep cells live under `output/<sweep-name>/cell_iJ_jJ/`. The summary parquet (heatmap data) and PNG live at `output/<sweep-name>/summary.{parquet,png}`.
- Before writing a plot or analysis script, check whether `simulate.py` or `plotting.py` already exposes what you need.
