# Testing Patterns

**Analysis Date:** 2026-03-13

## Test Framework

**Current State:**
- No test framework configured
- No test files detected in codebase
- No test configuration files (pytest.ini, conftest.py, tox.ini, etc.)
- No test dependencies in `pyproject.toml`

**Recommendation:**
- Testing infrastructure is absent
- To add testing, select a framework and add to dependencies
- Recommended: pytest (lightweight, simple, widely used for scientific Python)

---

## Test Organization (Not Implemented)

**Where to Place Tests:**
- Proposal: `tests/` directory at project root
- Parallel structure to `model/` directory
- Test files: `tests/test_main.py`, `tests/test_trajectories.py`

**Naming Convention:**
- If adopted: `test_*.py` files (pytest standard)
- Functions: `test_*()` pattern

---

## Code Structure for Testing

The codebase has several components suitable for unit testing:

**Core Simulation Engine:**
- Location: `model/main.py`
- Entry point: `simulate(p: Params, ...)`
- Testable functions:
  - `_sampling_stride(total_rows, max_points)` - pure function with clear inputs/outputs
  - `_reward_terms(r_pre, r_post, R_bar, alpha)` - mathematical computation
  - `_pack_state()` - state vector construction
  - `_smooth_rhs()` - ODE right-hand side
  - `_advance_state()` - numerical integration step
  - `_crossing_fraction()` - threshold detection logic

**Data I/O:**
- `MemoryRecorder`, `HDF5Recorder`, `ParquetRecorder` - recorder implementations
- `polars_frame_from_hdf5()`, `polars_frame_from_parquet()` - file reading
- `read_spike_times_hdf5()`, `read_spike_times_parquet()` - spike data reading

**Plotting:**
- `build_all_in_one_plotly_figure()` - generates plots without displaying
- `_plotly_values()` - finite value handling

---

## Testing Strategy (Not Yet Implemented)

### Unit Tests

**What to Test:**

1. **Mathematical functions** (pure, deterministic):
   ```python
   def test_reward_terms_basic():
       R, M = _reward_terms(r_pre=1.0, r_post=1.0, R_bar=0.0, alpha=0.5)
       assert R == -(1.0 - 0.5 * 1.0) ** 2  # Expected: -0.25
       assert M == -0.25 - 0.0
   ```

2. **State management**:
   ```python
   def test_pack_state():
       y = _pack_state(V=-65.0, I_s=0.0, x_pre=0.0, y_post=0.0,
                       E=0.0, r_pre=0.0, r_post=0.0, R_bar=0.0, w=2.0)
       assert len(y) == 9
       assert y[V_IDX] == -65.0
       assert y[W_IDX] == 2.0
   ```

3. **Threshold crossing detection**:
   ```python
   def test_crossing_fraction_below_threshold():
       frac = _crossing_fraction(v0=-60.0, v1=-40.0, threshold=-50.0)
       assert 0.0 <= frac <= 1.0  # Linear interpolation

   def test_crossing_fraction_no_cross():
       frac = _crossing_fraction(v0=-60.0, v1=-55.0, threshold=-50.0)
       assert frac is None  # No crossing
   ```

4. **Data filtering**:
   ```python
   def test_filter_spike_times():
       spikes = {"pre_spike_times": np.array([0.1, 0.5, 1.0, 1.5])}
       filtered = _filter_spike_times(spikes, x0=0.4, x1=1.2)
       assert np.allclose(filtered["pre_spike_times"], [0.5, 1.0])
   ```

### Integration Tests

**What to Test:**

1. **Recorder pipeline** - use all recorder types with small simulations
2. **File I/O round-trip** - write to HDF5/Parquet, read back, verify data
3. **Frame loading** - load_plot_frame with various downsampling factors
4. **Spike detection** - verify pre/post spike times logged correctly

### End-to-End Tests

**What to Test:**

1. **Full simulation** - run complete `simulate()` with small parameters, verify output shape and ranges
2. **Plotting** - generate plots without display, verify figure structure
3. **CLI workflows** - test main.py and trajectories.py with various argument combinations

---

## Mocking Strategy (Not Yet Implemented)

**If tests are added, mock these:**

1. **Optional dependencies:**
   ```python
   # For testing without h5py installed
   @pytest.mark.skipif(h5py is None, reason="h5py not installed")
   def test_hdf5_recorder():
       ...
   ```

2. **File I/O in unit tests:**
   - Mock `Path.exists()`, `Path.read_bytes()`, etc.
   - Use `tempfile` for actual file operations in integration tests

3. **HTTP server** (`serve_zoom_adaptive_plot`):
   - Mock requests using `unittest.mock.patch`
   - Or use pytest-httpserver for actual server testing

4. **Large arrays:**
   - Generate small test arrays instead of simulating large systems
   - Use `numpy.testing.assert_allclose()` for floating-point comparisons

---

## Test Data Fixtures

**Recommended fixtures** (if tests are implemented):

```python
# tests/conftest.py or tests/fixtures.py

import numpy as np
import pytest

@pytest.fixture
def small_params():
    """Minimal params for fast tests"""
    return Params(T=0.1, dt=1e-4, record_every=1e-4)

@pytest.fixture
def small_recording():
    """Small in-memory recording for output validation"""
    return {
        "t": np.linspace(0, 1, 100),
        "V": np.random.randn(100) - 65.0,
        "w": np.ones(100) * 2.0,
        # ... other required fields
    }

@pytest.fixture
def temp_hdf5(tmp_path):
    """Temporary HDF5 file for testing"""
    path = tmp_path / "test_recording.h5"
    # Write test data
    yield path
    # Cleanup automatic via tmp_path
```

---

## Integration Test Pattern

**Example structure** (if tests are implemented):

```python
def test_simulate_to_hdf5_to_read():
    """Full pipeline: simulate -> save HDF5 -> read back"""
    params = Params(T=0.5, dt=1e-4, record_every=1e-4)

    # Simulate and save
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = f"{tmpdir}/test.h5"
        rec = simulate(params, hdf5_path=hdf5_path)

        # Verify file was created
        assert Path(hdf5_path).exists()

        # Read back and validate
        frame = polars_frame_from_hdf5(hdf5_path, lazy=False)
        assert len(frame) > 0
        assert all(col in frame.columns for col in SERIES_KEYS)
```

---

## Coverage Considerations

**No coverage tool configured.**

If pytest is added, install pytest-cov:
```bash
uv add --dev pytest pytest-cov
```

Run coverage:
```bash
pytest --cov=model --cov-report=html
```

---

## Current Testing Capability

The codebase **lacks test coverage** but contains testable logic:

| Component | Testability | Priority |
|-----------|-------------|----------|
| `_reward_terms()` | High (pure function) | High |
| `_pack_state()` | High (pure function) | High |
| `_crossing_fraction()` | High (pure function, edge cases) | High |
| `simulate()` | Medium (complex, integration test) | Medium |
| `MemoryRecorder` | High (self-contained) | Medium |
| `load_plot_frame()` | Medium (requires files) | Medium |
| `build_all_in_one_plotly_figure()` | Low (heavy side effects) | Low |
| `serve_zoom_adaptive_plot()` | Low (HTTP server) | Low |

---

## Manual Testing Evidence

No automated tests found, but codebase includes:

1. **Default parameters** demonstrating expected values:
   ```python
   params = Params(
       T=100,
       method="rk4",
       V0=-62.39967779660166,
       # ... validated parameter set
   )
   ```

2. **CLI argument validation**:
   ```python
   if args.plot_backend == "server" and not (args.parquet or args.hdf5):
       raise ValueError("The server backend requires `--parquet` or `--hdf5` ...")
   ```

3. **Output visualization files** (simulation.png, trajectories.png in repo) suggest manual testing of plotting functionality

---

*Testing analysis: 2026-03-13*
