# Technology Stack

**Analysis Date:** 2026-03-13

## Languages

**Primary:**
- Python 3.12+ - Core simulation and analysis code

**Secondary:**
- LaTeX - Paper and documentation (`paper/*.tex`)
- JavaScript - Plotly visualization library (minified bundle included)

## Runtime

**Environment:**
- Python 3.12 (specified in `.python-version`)

**Package Manager:**
- uv (Python package manager)
- Lockfile: `uv.lock` (present)

## Frameworks

**Core:**
- No web framework used (native Python HTTP server only)

**Scientific Computing:**
- NumPy 2.4.1+ - Numerical arrays and mathematical operations
- SciPy 1.17.0+ - Scientific computation utilities
- Polars 1.33.1+ - DataFrame operations for large-scale data
- PyArrow 23.0.1+ - Apache Arrow columnar format

**Visualization:**
- Plotly 6.0.1+ - Interactive 2D/3D plotting and web visualization
- Matplotlib 3.10.8+ - Static plotting and figure generation

**Data Storage:**
- h5py 3.14.0+ - HDF5 file format read/write

## Key Dependencies

**Critical:**
- numpy - Mathematical foundation for all simulations
- h5py - Primary data persistence format (HDF5)
- pyarrow - Parquet file format support for distributed data handling
- scipy - Specialized mathematical functions

**Visualization:**
- plotly - Interactive web-based visualization engine (6.0.1+)
- matplotlib - Fallback static visualization

**Data Processing:**
- polars - High-performance DataFrame operations for trajectory analysis
- pyarrow - Columnar format support and interoperability

## Configuration

**Environment:**
- Configuration via command-line arguments (argparse-based)
- Simulation parameters defined in `Params` dataclass (`model/main.py` lines 72-113)
- No environment variable dependencies detected

**Build:**
- No build system configured
- Project runs directly via `python model/main.py` with command-line flags

**Package Management:**
- `pyproject.toml` - Project metadata and dependencies
- `uv.lock` - Locked dependency versions

## Platform Requirements

**Development:**
- Python 3.12+ runtime
- uv package manager for dependency installation
- ~500MB disk space for simulation data (configurable)

**Production:**
- Local filesystem access for HDF5/Parquet output
- Optional: 127.0.0.1:8050 for local web server (hardcoded default)
- No external cloud services required

## Execution Modes

**Entry Points:**
- `model/main.py` - Main simulation engine with three visualization backends:
  - matplotlib: Static plot output
  - plotly: Interactive HTML file generation
  - server: Local HTTP server with zoom-adaptive rendering
- `model/trajectories.py` - Pairwise trajectory analysis script

**Simulation Options:**
```bash
python model/main.py \
  --plot-backend {matplotlib|plotly|server} \
  --hdf5 <path> \
  --parquet <path> \
  --method {euler|rk4} \
  --max-plot-points 40000 \
  --chunk-rows 100000
```

---

*Stack analysis: 2026-03-13*
