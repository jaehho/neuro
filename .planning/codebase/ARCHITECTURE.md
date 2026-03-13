# Architecture

**Analysis Date:** 2026-03-13

## Pattern Overview

**Overall:** Layered Pipeline Architecture with Pluggable Storage

This is a neurobiological simulation system with three distinct architectural layers:
1. **Simulation Engine** (Core dynamics)
2. **Recording/Storage Abstraction** (Flexible persistence)
3. **Visualization Pipeline** (Multi-backend rendering)

Key Characteristics:
- Simulation-agnostic data persistence via abstract `Recorder` interface
- Dual-mode plotting: static export (HTML) and adaptive server (live zoom)
- Envelope downsampling for visualization at arbitrary resolution limits
- Spike event tracking parallel to continuous state series

## Layers

**Simulation Core:**
- Purpose: Computes neurobiological dynamics using numerical integration (Euler or RK4)
- Location: `model/main.py` lines 466-597 (`simulate()` function)
- Contains: State initialization, spike detection, synaptic/plasticity updates, neuron models
- Depends on: `Params` dataclass, numpy arrays
- Used by: Entire application; entry point for all workflows

**Recording Abstraction:**
- Purpose: Decouples simulation from storage backend (memory, HDF5, Parquet)
- Location: `model/main.py` lines 137-349 (Recorder interface and implementations)
- Contains: `Recorder` abstract base, `MemoryRecorder`, `HDF5Recorder`, `ParquetRecorder`, `MultiRecorder`
- Depends on: h5py, pyarrow, numpy
- Used by: `simulate()`, enables test isolation and production flexibility

**Data Loading & Resampling:**
- Purpose: Efficiently loads partial data from disk and applies envelope downsampling
- Location: `model/main.py` lines 600-931 (frame loading, envelope extraction)
- Contains: Polars DataFrame loaders, HDF5/Parquet readers, envelope bucket methods
- Depends on: h5py, polars, pyarrow
- Used by: Visualization pipeline

**Visualization Pipeline:**
- Purpose: Renders time series and pairwise trajectory views in interactive/static HTML
- Location: `model/main.py` lines 943-1240 (plotting functions and server)
- Contains: Plotly figure builders, matplotlib fallback, HTTP server for adaptive zoom
- Depends on: plotly, matplotlib
- Used by: Entry point when `--plot-backend` is set

**Trajectory Analysis:**
- Purpose: Secondary analysis module for pairwise state variable projections
- Location: `model/trajectories.py` (standalone script)
- Contains: Pairwise variable combinations, subplots, time-color mapping
- Depends on: plotly, imports from `main.py`
- Used by: Optional analysis workflow

## Data Flow

**Simulation→Recording→Plotting (Full Stack):**

1. `parse_args()` at `model/main.py:1245` reads command-line config
2. `simulate(Params, hdf5_path, parquet_path)` at line 466:
   - Initializes state vector `y` via `_pack_state()`
   - Builds recorder via `_build_recorder()` (selects backend)
   - Loops over timesteps 0 to n=T/dt:
     - Detects pre-spike events (periodic at `1.0/(r_pre_rate*dt)` period)
     - Checks for post-spike via voltage threshold crossing
     - Updates continuous state via `_smooth_rhs()` (Euler) or RK4
     - Records sampled row every `record_every` interval
   - Returns finalized recording (dict or path metadata)
3. Plotting dispatches based on `--plot-backend`:
   - `"matplotlib"`: requires in-memory dict, calls `plot_all_in_one_figure_matplotlib()` at line 957
   - `"plotly"`: calls `plot_all_in_one_plotly()` at line 1077, which internally calls `load_time_series_frame()` (envelope downsampling)
   - `"server"`: calls `serve_zoom_adaptive_plot()` at line 1096, spawns HTTP server that dynamically resamples for zoom ranges

**Alternative: Trajectory Analysis:**

1. `trajectories.py` imports `Params`, `simulate()`, `load_plot_frame()` from `main`
2. Calls `load_plot_frame()` with specific columns: ["t"] + VAR_NAMES
3. Loads via envelope method if exceeding `max_points`
4. Extracts all pairwise combinations of 9 state variables (36 pairs total)
5. Renders 6×6 grid with time color mapping

**State Management:**

- **Simulation state**: Numpy array `y` of 9 floats (V, I_s, x_pre, y_post, E, r_pre, r_post, R_bar, w)
- **Recording state**: Abstracted behind Recorder interface; parallel spike list storage
- **Visualization state**: Polars DataFrames (in-memory slices); spike times as separate dicts
- **Server state**: Stateless HTTP handler; recomputes figures on-demand from disk

## Key Abstractions

**Recorder Interface (Polymorphic Storage):**
- Purpose: Enable simulation without knowledge of persistence backend
- Examples: `MemoryRecorder`, `HDF5Recorder`, `ParquetRecorder`, `MultiRecorder`
- Pattern: Abstract base with `append(row)`, `append_spike(key, value)`, `finalize()` contract
- Allows: Simultaneous multi-backend recording (e.g., both HDF5 and Parquet in single run)

**State Vector Packing:**
- Purpose: Efficient numerical integration (single numpy array vs. dict with strings)
- Location: `_pack_state()` at line 357
- Pattern: 9-element numpy array with module-level index constants (V_IDX, I_S_IDX, etc.)
- Trade-off: Speed vs. readability; hardcoded indices require alignment

**Envelope Downsampling:**
- Purpose: Visualize multi-million-point datasets at fixed 40k points
- Location: `_envelope_bucket_size()` at line 122; `_add_bucket_extrema_indices()` at line 688
- Pattern: Divide data into buckets, track min/max indices per variable, render extrema
- Result: Preserves visual peaks while reducing memory by 50-100x

**Frame Loading Dispatch:**
- Purpose: Unified interface for memory, HDF5, and Parquet sources
- Location: `load_time_series_frame()` at line 869, `load_plot_frame()` at line 894
- Pattern: Type-check input (dict/Path/str) and delegate to backend-specific loader
- Fallback: Returns finalization result dict metadata when source is simulation output

## Entry Points

**Main Simulation:**
- Location: `model/main.py:1260` (`if __name__ == "__main__"`)
- Triggers: `python model/main.py [--hdf5 PATH] [--parquet PATH] [--plot-backend {matplotlib,plotly,server}] ...`
- Responsibilities:
  - Parses command-line arguments
  - Instantiates `Params` (hardcoded defaults at line 1263, override via future CLI flags)
  - Runs `simulate()`
  - Dispatches to appropriate plotting backend
  - Serves HTTP or saves static HTML

**Trajectory Analysis:**
- Location: `model/trajectories.py:87` (`if __name__ == "__main__"`)
- Triggers: `python model/trajectories.py [--hdf5 PATH | --parquet PATH] [--output-html PATH] [--max-plot-points N]`
- Responsibilities:
  - Loads recording from disk
  - Computes pairwise projections
  - Renders and exports trajectory visualization

## Error Handling

**Strategy:** Optional dependency imports with runtime checks

**Patterns:**
- Try-except on all external library imports at module load (lines 14-38)
- Graceful fallback: Set module to `None` if missing
- Runtime check before use: `if h5py is None: raise ImportError(...)`
- Advantage: Users can run basic simulation without optional deps (h5py, plotly, polars)
- Trade-off: Deferred errors (user discovers missing dependencies late)

**State Validation:**
- `Params.method` check at line 472: must be "euler" or "rk4"
- `--plot-backend server` requires `--hdf5` or `--parquet` at line 1277
- `max_points <= 0` validation in downsampling functions

## Cross-Cutting Concerns

**Logging:**
- Minimal; only `print()` statements for server startup (line 1238) and final output paths (trajectories.py line 96)
- No structured logging framework; appropriate for single-script tool

**Validation:**
- Input validation at CLI parsing (argparse choices)
- State bounds: Voltage clipped to [0, wmax], refractory period clamped to [0, tau_ref]
- Data type enforcement via numpy float64 throughout

**Authentication:**
- Not applicable; local-only tool with no external auth

**Performance Optimization:**
- Streaming Parquet writes to avoid OOM on long simulations
- Envelope downsampling mandatory for plotting (reduces 1M points → 40k)
- HDF5 chunking (default 100k rows) for efficient disk I/O
- Polars lazy evaluation for Parquet frame filtering

---

*Architecture analysis: 2026-03-13*
