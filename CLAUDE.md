# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Neuromodulated STDP simulation: N pre-synaptic neurons → 1 post-synaptic LIF neuron, implementing three-factor learning rules from Frémaux & Gerstner (2016). The simulation explores how different neuromodulator types (covariance, gated, surprise, constant) and reward signals (target_rate, biofeedback, contingent) affect synaptic weight dynamics and spatial credit assignment.

## Commands

```bash
# Install (uses uv with hatchling backend)
uv sync

# Run simulation (typer CLI, entry point: neuro)
uv run neuro                                    # defaults: 100s, rk4, target_rate, covariance, 2 pre
uv run neuro --n-pre 1                          # single pre-neuron
uv run neuro --n-pre 5 --reward-signal contingent
uv run neuro --help                             # all CLI options

# Run tests
uv run pytest                                   # all tests (85)
uv run pytest tests/test_rk4.py                 # RK4 integrator tests
uv run pytest tests/test_general_target.py      # reward/modulation tests
uv run pytest tests/test_n_pre.py               # N-pre generalization tests
uv run pytest -k "test_rk4_decay"               # single test by name

# Analysis scripts (write plots to output/)
uv run python scripts/analysis.py               # spectrograms, convergence, sensitivity
uv run python scripts/analysis_regimes.py       # neuromod type comparison
uv run python scripts/test_general_targets.py   # target function sweeps

# Build paper (Typst)
typst compile docs/main.typ docs/main.pdf
```

## Architecture

Everything lives in a single simulation module `src/neuro/sim.py`:

- **`Params` dataclass**: all model parameters. `n_pre` controls the number of pre-synaptic neurons (default 2). Per-synapse fields (`r_pre_rates`, `w0`, `I_s0`, `x_pre0`, `r_pre0`, `E0`) are tuples of length `n_pre`; scalars are broadcast automatically. Two orthogonal configuration axes: `neuromod_type` (how M derives from R) and `reward_signal` (what R measures).

- **State vector** (4 + n_pre×5 elements): flat numpy array. Shared state: `[V, y_post, r_post, R_bar]`. Per-synapse block (×n_pre): `[I_s, x_pre, E, r_pre, w]`. Index helpers: `_I_s_idx(i)`, `_W_idx(i)`, etc. Backward-compatible aliases `V_IDX`, `W1_IDX`, etc. exist for n_pre=2.

- **ODE system**: `_smooth_rhs()` loops over N synapses for decays and weight updates. Membrane voltage sums all weighted currents. `_advance_state()` wraps Euler or RK4. Spike jumps are in `simulate()`, not the RHS.

- **`simulate()`**: hybrid event-driven / continuous loop. Synapse 0 is the "target" (reward-paired) for contingent mode. Dynamic key generation: `series_keys(n_pre)`, `spike_keys(n_pre)`.

- **Recording**: `Recorder` hierarchy (Memory/HDF5/Parquet/Multi), keyed dynamically (e.g., `w1`..`wN`, `E1`..`EN`). Parquet stores spikes in `.spikes.parquet`.

- **Visualization**: `matplotlib`, `plotly`, and `server` (HTTP with zoom-adaptive resampling). All adapt to n_pre.

## Key design decisions

- The reward signal R is always *external* in the three-factor literature. `target_rate` is a self-supervisory demonstration — `biofeedback` and `contingent` are the proper paradigms.
- `neuromod_type` and `reward_signal` are orthogonal axes.
- Weight update rule: `dw_i/dt = M(t) · E_i(t)` — global modulator times per-synapse eligibility. Spatial credit assignment: all synapses see the same M, but only the reward-paired synapse (index 0) has high E when reward arrives.
- RK4 is O(dt⁴) for smooth ODE between spikes; spike events use threshold-crossing interpolation O(dt²).
- Keys are 1-indexed (`w1`, `w2`, ...) to match neuroscience convention and preserve compatibility with existing saved data.
