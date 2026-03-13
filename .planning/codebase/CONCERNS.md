# Codebase Concerns

**Analysis Date:** 2026-03-13

## Memory Management

**Memory Pre-allocation Limits:**
- Issue: `MemoryRecorder` pre-allocates full capacity at initialization (`np.zeros(capacity, dtype=np.float64)`). For long simulations (T=100s with dt=1e-4), this allocates ~1.2M rows × 17 variables × 8 bytes = ~160MB. Larger simulations can cause OOM.
- Files: `src/model/main.py:150` (MemoryRecorder.__init__)
- Impact: Simulations with T > 1000s using default memory recorder will fail with MemoryError
- Fix approach: Consider lazy allocation, chunked memory allocation, or enforce disk-backed recorders for long simulations. Add validation in simulate() to warn/error when capacity exceeds reasonable memory limits.

**Parquet Spike Resizing:**
- Issue: `append_spike()` in `HDF5Recorder` resizes datasets one-by-one per spike (`ds.resize((idx + 1,))` at line 216). With potentially thousands of spikes, this causes repeated expensive disk I/O operations.
- Files: `src/model/main.py:213-218` (HDF5Recorder.append_spike)
- Impact: Spike recording becomes bottleneck for simulations with high spike rates; slows down simulation runtime
- Fix approach: Batch spike writes with a buffer before resizing, similar to `ParquetRecorder`'s chunk-based buffering at line 289.

## Numerical Precision Issues

**Period Steps Calculation Fragility:**
- Issue: `period_steps = max(1, round(1.0 / (p.r_pre_rate * p.dt)))` (line 477). If `r_pre_rate * dt` is very small (e.g., rate=0.01, dt=1e-4), division and rounding can cause precision loss. `period_steps` controls pre-spike timing.
- Files: `src/model/main.py:477`
- Impact: Small timing errors accumulate across simulation steps; spike timing becomes slightly inaccurate, affecting learning dynamics
- Fix approach: Use higher-precision arithmetic or add a check to ensure `period_steps >= 1` after rounding, validate against expected spike count post-simulation.

**Refractory Period Edge Case:**
- Issue: `ref_remaining` is tracked as float but compared with floating-point arithmetic (`ref_remaining <= 0.0`, `ref_remaining - p.dt`). Due to floating-point rounding, transitions between refractory and non-refractory states can be off by one timestep.
- Files: `src/model/main.py:524-535` (Euler method), `src/model/main.py:556-583` (RK4 method)
- Impact: Spike-generation logic may misbehave near refractory period boundaries; neuron could spike during refractory period or delay spike incorrectly
- Fix approach: Use epsilon-based comparisons or track refractory state as integer step counter instead of continuous float.

**Weight Clipping Placement:**
- Issue: Weight `w` is clipped twice: in `_smooth_rhs()` at line 382 and in `_advance_state()` at line 422. In the Euler method, clipping happens once per step at line 551. RK4 method clips in `_advance_state()` only. Inconsistent clipping between methods can lead to different weight trajectories.
- Files: `src/model/main.py:382,422,551`
- Impact: Different integration methods produce different synaptic weight evolution; results not comparable across method parameter
- Fix approach: Standardize to single clipping location per step (e.g., only in finalize state packing or only in RK4 advance).

## Plotting & Visualization

**Large Asset File Committed:**
- Issue: `plotly.min.js` (4.7MB) is committed to git repository at `/home/jaeho/neuro/plotly.min.js`. This inflates repository size unnecessarily.
- Files: `/home/jaeho/neuro/plotly.min.js`
- Impact: Repository slow to clone/pull; waste of storage; versioning confusion
- Fix approach: Remove from git, add to `.gitignore`, reference plotly from CDN in `write_plotly_html()` at line 946 instead, or use a build script to download at runtime.

**No Validation on Plotting Input:**
- Issue: `build_all_in_one_plotly_figure()` (line 1002) and other plotting functions assume columns exist without validating. If recording is incomplete or missing columns, Polars will raise cryptic errors downstream.
- Files: `src/model/main.py:1012-1014`
- Impact: User-facing error messages unhelpful; hard to debug
- Fix approach: Add early validation of column presence in `load_time_series_frame()` and `load_plot_frame()` before processing.

## Integration Method Divergence

**Euler vs RK4 State Update Differences:**
- Issue: Euler method (lines 509-553) updates state variables sequentially with explicit stepping, while RK4 method (lines 555-583) uses vector-based RK4 integration. Post-spike handling differs: Euler applies `y_post += 1.0` and `E += eta_plus * ...` before exponential decay, while RK4 applies updates to `y_mid` before continuing. This causes different spike-triggered plasticity magnitudes.
- Files: `src/model/main.py:509-553` (Euler), `src/model/main.py:555-583` (RK4)
- Impact: Euler and RK4 simulations produce significantly different weight trajectories despite same parameter set; incomparable results
- Fix approach: Unify spike handling logic between methods, apply identical post-spike updates regardless of integration method.

## Simulation Initialization

**Hardcoded Initial Conditions in Main:**
- Issue: Default simulation at lines 1263-1275 uses specific initial conditions from a prior analysis/optimization. These are not representative of default behavior and override `Params` defaults.
- Files: `src/model/main.py:1263-1275`
- Impact: Running script without arguments uses non-default initialization, confusing users expecting Params defaults; makes reproducibility harder
- Fix approach: Move hardcoded values into a separate config file or load from optional flag; use `Params()` defaults for main() unless explicitly overridden.

## Scaling Limits

**O(n) Spike Resizing in HDF5:**
- Issue: Each spike append resizes HDF5 dataset one spike at a time (line 216). For simulations with 10,000+ spikes, this is O(n) resize operations against disk-backed storage.
- Files: `src/model/main.py:213-218`
- Impact: HDF5 spike recording time scales linearly with spike count; can dominate runtime for high-activity neurons
- Fix approach: Batch spike writes (buffer 1000 spikes before resize), similar to Parquet's chunk approach.

**Envelope Sampling Complexity:**
- Issue: `_collect_parquet_envelope_frame()` (lines 804-866) creates row indices and aggregations across all data columns. For many columns and large datasets, this becomes expensive.
- Files: `src/model/main.py:826-845`
- Impact: Plotting large Parquet files with many variables becomes slow; potential memory spike when creating index set
- Fix approach: Implement streaming envelope calculation or lazy column selection.

## Testing & Validation Gaps

**No Spike Count Validation:**
- Issue: Simulations produce spikes, but there's no built-in check that spike counts match expected rate. If `period_steps` calculation is wrong, extra/missing spikes go unnoticed.
- Files: `src/model/main.py:477,498`
- Impact: Silent correctness errors; wrong spike rates propagate downstream to analysis
- Fix approach: Add optional post-simulation spike count validation against expected `T / (1.0 / r_pre_rate)`.

**No Numerical Stability Checks:**
- Issue: State variables can grow unbounded (e.g., `I_s`, `x_pre`, `r_pre` accumulate on spikes with no normalization). No checks for NaN/inf during simulation.
- Files: `src/model/main.py:501-503` (spike increments)
- Impact: Silent numerical failures; variables become NaN/inf without error, corrupting output silently
- Fix approach: Add periodic health checks (e.g., after each spike, verify all variables are finite).

## Dependencies at Risk

**Optional Dependency Import Pattern:**
- Issue: Multiple packages are imported as optional (`h5py`, `polars`, `pyarrow`, `plotly`). Missing imports raise ImportError only when that code path is used. If user specifies `--hdf5` but h5py is uninstalled, error occurs after potentially long simulation.
- Files: `src/model/main.py:14-38`
- Impact: Poor user experience; simulation starts then crashes during finalization
- Fix approach: Check all required dependencies upfront in `main()` before running expensive simulation.

## Code Organization

**Large Single File:**
- Issue: `main.py` is 1317 lines containing simulation engine, recording infrastructure, I/O, plotting, and HTTP server. No clear module boundaries.
- Files: `src/model/main.py`
- Impact: Difficult to navigate, test, and maintain; mixed concerns
- Fix approach: Refactor into modules: `simulator.py`, `recorder.py`, `plotting.py`, `server.py`.

**Incomplete Docstrings:**
- Issue: Most functions lack docstrings (e.g., `_smooth_rhs`, `_advance_state`, `_pack_state`). Comments minimal.
- Files: `src/model/main.py` throughout
- Impact: Hard for users/developers to understand equations, expected parameter ranges, and mathematical assumptions
- Fix approach: Add comprehensive docstrings with equation references, parameter ranges, and return value descriptions.

---

*Concerns audit: 2026-03-13*
