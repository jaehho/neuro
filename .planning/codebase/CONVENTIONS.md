# Coding Conventions

**Analysis Date:** 2026-03-13

## Naming Patterns

**Files:**
- Snake case: `main.py`, `trajectories.py`
- Descriptive names indicating purpose or domain

**Functions:**
- Snake case: `_sampling_stride()`, `simulate()`, `plot_all_in_one_figure_matplotlib()`
- Private/internal functions prefixed with underscore: `_smooth_rhs()`, `_pack_state()`, `_advance_state()`
- Public API functions without underscore prefix

**Variables:**
- Snake case for all local variables: `total_rows`, `max_points`, `V_reset`, `tau_m`
- Single or double letter variables for mathematical quantities: `V` (voltage), `w` (weight), `E` (eligibility trace), `R` (reward), `M` (modulated reward), `dt` (timestep), `t` (time)
- Index constants in UPPER_CASE with suffix `_IDX`: `V_IDX = 0`, `I_S_IDX = 1`, `X_PRE_IDX = 2`, `W_IDX = 8`
- Global constant lists in UPPER_CASE: `SERIES_KEYS`, `SPIKE_KEYS`

**Types:**
- Class names in PascalCase: `Params`, `Recorder`, `MemoryRecorder`, `HDF5Recorder`, `ParquetRecorder`, `MultiRecorder`, `Handler`
- Type hints using modern Python 3.12+ syntax: `str | Path`, `dict[str, float]`, `list[str]`

## Code Style

**Formatting:**
- No explicit formatter detected (no `.prettierrc`, `.flake8`, or similar config)
- Import statement in file: `from __future__ import annotations` (enables modern type hint syntax)

**Linting:**
- No ESLint or Python linter detected in config files
- Follows conventional Python PEP8-style conventions implicitly

**Imports:**
- Organized by standard library, third-party, local modules
- Standard library imports first: `argparse`, `json`, `dataclasses`, `http`, `pathlib`, `urllib`
- Third-party imports second: `matplotlib`, `numpy`, `h5py`, `polars`, `pyarrow`, `plotly`
- Local imports last: `from main import ...` in `trajectories.py`
- Optional imports wrapped in try/except blocks for graceful degradation when packages not installed

## Import Organization

**Order:**
1. `from __future__ import annotations` (future behavior)
2. Standard library imports (`argparse`, `json`, `dataclasses`, `http`, `pathlib`, `urllib`)
3. Third-party imports (`matplotlib`, `numpy`, `h5py`, `polars`, `pyarrow`, `plotly`)
4. Conditional imports wrapped in try/except blocks
5. Local module imports

**Optional Dependencies Pattern:**
```python
try:
    import h5py
except ImportError:
    h5py = None

try:
    import polars as pl
except ImportError:
    pl = None
```

**Path Aliases:**
- Not used; imports are explicit with full module paths

## Error Handling

**Patterns:**

1. **ImportError for optional dependencies:**
   ```python
   if h5py is None:
       raise ImportError("h5py is required for HDF5 output. Install it with `uv add h5py`.")
   ```
   - Check if module is None (set in except block)
   - Raise ImportError with helpful installation instructions

2. **ValueError for invalid parameters:**
   ```python
   if max_points <= 0:
       raise ValueError("max_points must be positive.")
   ```
   - Validate numeric parameters against business logic constraints

3. **Unknown method validation:**
   ```python
   if method not in {"euler", "rk4"}:
       raise ValueError("Params.method must be either 'euler' or 'rk4'.")
   ```

4. **File existence checks:**
   ```python
   if not source.exists():
       raise FileNotFoundError(f"Plot source does not exist: {source}")
   ```

5. **NotImplementedError for abstract methods:**
   ```python
   def append(self, row: dict[str, float]) -> None:
       raise NotImplementedError
   ```

6. **Type checking:**
   ```python
   if isinstance(result, dict):
       merged.update(result)
   ```

## Logging

**Framework:** `print()` and standard output only

**Patterns:**
- Console output via `print()` for informational messages
- No structured logging library detected
- Example: `print(f"Served zoom-adaptive plot at http://{host}:{port}")`
- Silent operations: Handler classes override `log_message()` to return None

## Comments

**When to Comment:**
- Minimal comments in codebase; code is self-documenting
- One commented-out line: `# params = Params(T=2000, method="rk4")` (reference to alternate parameter configuration)

**JSDoc/TSDoc:**
- Not used; codebase is Python, not JavaScript/TypeScript
- No docstrings detected in examined code
- Type hints serve as inline documentation

## Function Design

**Size:**
- Functions range from 3 lines (`_reward_terms()`) to 200+ lines (`serve_zoom_adaptive_plot()`, `simulate()`)
- Simulation core logic concentrated in `simulate()` function (130+ lines)
- Smaller utility functions decompose specific concerns

**Parameters:**
- Functions with 3-9 parameters typical
- Use keyword-only arguments for behavioral flags: `def _smooth_rhs(..., *, voltage_active: bool)`
- Default parameter values for optional behavior: `def load_plot_frame(..., max_points: int = 40_000)`
- Type hints required for all parameters

**Return Values:**
- Functions return single values or tuples: `tuple[float, float]` for `_reward_terms()`
- Can return None: `float | None` in `_crossing_fraction()`
- Recorder interface returns dict with metadata: `dict[str, str | int]`
- Functions document return type with type hints

## Module Design

**Exports:**
- No `__all__` declaration detected
- `main.py` exports classes (`Params`, `Recorder`, subclasses), functions for simulation and plotting
- `trajectories.py` imports from `main` module: `from main import Params, _plotly_values, load_plot_frame, simulate, write_plotly_html`
- Private functions use underscore prefix; public API exposed without prefix

**Barrel Files:**
- Not used; direct imports from modules

## Special Patterns

**Dataclass for Configuration:**
```python
@dataclass
class Params:
    T: float = 20.0
    dt: float = 1e-4
    # ... 30+ parameters with sensible defaults
```
- Uses `@dataclass` decorator for parameter container
- All fields have default values
- Used throughout codebase for configuration passing

**Abstract Base Class Pattern:**
```python
class Recorder:
    def append(self, row: dict[str, float]) -> None:
        raise NotImplementedError
    def finalize(self):
        raise NotImplementedError
```
- Base class with interface methods raising NotImplementedError
- Subclasses: `MemoryRecorder`, `HDF5Recorder`, `ParquetRecorder`

**Adapter Pattern:**
```python
class MultiRecorder(Recorder):
    def __init__(self, recorders: list[Recorder]):
        self.recorders = recorders

    def append(self, row: dict[str, float]) -> None:
        for recorder in self.recorders:
            recorder.append(row)
```
- Allows multiple recorders to operate in parallel

---

*Convention analysis: 2026-03-13*
