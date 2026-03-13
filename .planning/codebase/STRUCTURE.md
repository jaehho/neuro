# Codebase Structure

**Analysis Date:** 2026-03-13

## Directory Layout

```
/home/jaeho/neuro/
├── model/               # Core simulation and visualization code
│   ├── main.py         # Main simulation, recording backends, plotting
│   ├── trajectories.py # Pairwise state trajectory visualization
│   └── README.md       # Data handling and plotting documentation
├── paper/              # Documentation/reference (not analyzed)
├── .planning/          # GSD planning artifacts (auto-generated)
│   └── codebase/      # Architecture analysis documents
├── .git/               # Version control
├── pyproject.toml      # Project metadata and dependencies
├── .python-version     # Python version specification
├── uv.lock            # Locked dependency versions (uv package manager)
├── README.md          # (Empty, placeholder)
├── .gitignore         # Git ignore rules
├── simulation.png     # Output: matplotlib simulation plot
├── trajectories.png   # Output: trajectory visualization
└── plotly.min.js      # Asset: Plotly library (static asset for server mode)
```

## Directory Purposes

**`model/`:**
- Purpose: All executable code for simulation, data I/O, and visualization
- Contains: Python simulation engine, recording backends, plotting pipelines
- Key files: `main.py` (1317 lines), `trajectories.py` (97 lines)

**`paper/`:**
- Purpose: Research documentation or notes (not examined)
- Contains: Likely LaTeX, markdown, or reference papers
- Key files: Unknown; not analyzed

**`.planning/codebase/`:**
- Purpose: GSD (Generative System Development) analysis documents
- Contains: ARCHITECTURE.md, STRUCTURE.md (this file), future CONVENTIONS.md, TESTING.md
- Key files: Analysis artifacts for code generation tools

**`.git/`:**
- Purpose: Version control repository
- Contains: Commit history, branch tracking
- Key files: Objects, refs, logs

## Key File Locations

**Entry Points:**
- `model/main.py:1260`: Main simulation orchestration (`if __name__ == "__main__"`)
- `model/trajectories.py:87`: Trajectory analysis tool (`if __name__ == "__main__"`)

**Core Simulation Logic:**
- `model/main.py:72-114`: `Params` dataclass (model configuration)
- `model/main.py:466-597`: `simulate()` function (main integration loop)
- `model/main.py:371-425`: `_advance_state()` (RK4/Euler numerical methods)
- `model/main.py:351-354`: `_reward_terms()` (neuromodulation math)

**Recording Backends:**
- `model/main.py:137-167`: `MemoryRecorder` (in-RAM storage)
- `model/main.py:169-227`: `HDF5Recorder` (compressed disk storage)
- `model/main.py:229-302`: `ParquetRecorder` (columnar streaming storage)
- `model/main.py:305-326`: `MultiRecorder` (composite recorder)

**Data Loading & Resampling:**
- `model/main.py:600-637`: `polars_frame_from_hdf5()`, `polars_frame_from_parquet()`
- `model/main.py:688-752`: Envelope downsampling (`_add_bucket_extrema_indices()`, `_collect_*_envelope_frame()`)
- `model/main.py:869-931`: `load_time_series_frame()`, `load_plot_frame()` (unified loaders)

**Visualization:**
- `model/main.py:957-999`: `plot_all_in_one_figure_matplotlib()` (matplotlib backend)
- `model/main.py:1002-1074`: `build_all_in_one_plotly_figure()` (plotly builder)
- `model/main.py:1077-1079`: `plot_all_in_one_plotly()` (plotly export to static HTML)
- `model/main.py:1096-1242`: `serve_zoom_adaptive_plot()` (HTTP server for live zoom)

**Trajectory Analysis:**
- `model/trajectories.py:32-75`: `plot_pairwise_trajectories()` (main entry)
- `model/trajectories.py:18-29`: Variable names and labels for state space

**Configuration:**
- `pyproject.toml`: Project metadata, Python version constraint (>=3.12), dependencies
- `.python-version`: Python version (exact version string, e.g., "3.12.1")
- `uv.lock`: Lock file for reproducible dependency versions (uv package manager)

## Naming Conventions

**Files:**
- `main.py`: Monolithic entry point and core module (not split into sub-modules)
- `trajectories.py`: Specialized analysis tool (imports from `main`)
- `README.md`: Documentation per directory

**Directories:**
- `model/`: Lowercase single word for core domain
- `paper/`: Lowercase single word for secondary materials
- `.planning/`: Dot-prefix for build/meta artifacts (hidden from standard listings)

**Functions:**
- `simulate()`: Public entry point, no leading underscore
- `_advance_state()`, `_smooth_rhs()`: Private helpers, leading underscore
- `load_time_series_frame()`: Public loader, verb-first naming
- `_pack_state()`, `_downsample_memory_rec()`: Private utilities, underscore prefix

**Classes:**
- `Params`: PascalCase dataclass for configuration
- `Recorder`: Abstract base class (PascalCase)
- `MemoryRecorder`, `HDF5Recorder`, `ParquetRecorder`: Concrete implementations (PascalCase + descriptor)

**Variables:**
- Module-level constants: UPPERCASE (e.g., `SERIES_KEYS`, `V_IDX`, `TAU_M`)
- Function arguments: snake_case (e.g., `hdf5_path`, `max_points`)
- State array indices: UPPERCASE constants (e.g., `V_IDX = 0`, `I_S_IDX = 1`)
- Loop/temporary: snake_case (e.g., `rec`, `y`, `spikes`)

**Types:**
- Dataclass fields: snake_case with type hints (e.g., `T: float = 20.0`)
- Dict keys: snake_case or descriptive (e.g., "pre_spike_times", "V", "w")

## Where to Add New Code

**New Feature (e.g., different plasticity rule):**
- Primary code: `model/main.py` (extend `Params` dataclass with new fields, update `_reward_terms()` and `_smooth_rhs()`)
- Tests: Create `model/test_main.py` (not yet present; see TESTING.md)

**New Component/Module (e.g., alternative numerical method):**
- Implementation: `model/integrators.py` (new file)
- Integration: Import into `model/main.py`, extend `--method` argument choices
- Export: Add public function to `model/__init__.py` (if created)

**New Visualization Backend (e.g., WebGL rendering):**
- Implementation: `model/visualization.py` (new file)
- Entry point: Add case to `parse_args()` for `--plot-backend webgl`
- Call site: Add elif branch in `if __name__ == "__main__"` block (line 1296)

**New Analysis Tool (like `trajectories.py`):**
- Implementation: `model/analysis_NAME.py` (separate file)
- Imports: Reuse `load_plot_frame()`, `_plotly_values()` from `main.py`
- Entry point: Standalone `if __name__ == "__main__"` block

**Utilities/Helpers:**
- Shared logic: Add to `model/main.py` as private function (leading underscore)
- Only extract to separate file if >5 functions or >200 lines

**Configuration/Constants:**
- Model parameters: Extend `Params` dataclass
- Plotting defaults: Module-level constants (e.g., `SERIES_KEYS` at line 41)
- Magic numbers in plotting: Consider moving to class attributes on recorder

## Special Directories

**`.planning/codebase/`:**
- Purpose: GSD analysis documents
- Generated: Yes (by GSD mapping tools)
- Committed: Yes (documents are checked in, not built artifacts)
- Write directly to this directory when updating architecture/structure analysis

**`__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes (automatic)
- Committed: No (.gitignore excludes)

**Output Files (root level):**
- `simulation.png`: Generated by `plot_all_in_one_figure_matplotlib()`
- `trajectories.png`: Not currently generated (trajectories.py produces .html)
- `plotly.min.js`: Static asset for server mode (checked in for offline use)

## Import Organization

**Current pattern in `model/main.py` (lines 1-38):**

1. Standard library: `argparse`, `json`, `dataclasses`, `http`, `pathlib`, `urllib`
2. Numeric/data: `numpy`, `matplotlib`, `scipy` (implicit via other imports)
3. Optional deps: `h5py`, `polars`, `pyarrow`, `plotly` (wrapped in try-except)

**Pattern in `model/trajectories.py` (lines 1-15):**

1. Standard library: `argparse`, `itertools`
2. Numeric: `numpy`
3. Optional: `plotly`
4. Internal: `from main import ...` (relative import from sibling module)

**Best practice for new code:**
- Group by: standard library, third-party, internal
- Order within group: alphabetical
- Optional deps: always wrap in try-except at module level
- Relative imports: use `from main import ...` when in same directory

---

*Structure analysis: 2026-03-13*
