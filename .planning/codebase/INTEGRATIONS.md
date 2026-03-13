# External Integrations

**Analysis Date:** 2026-03-13

## APIs & External Services

**No External APIs Detected:**
- This is a self-contained scientific simulation project
- No third-party API integrations (Stripe, AWS, Google Cloud, etc.)
- No webhook endpoints exposed or consumed

## Data Storage

**File Formats (Local Only):**
- HDF5 - Primary scientific data format via h5py
  - Location: Configurable via `--hdf5 <path>` argument
  - Client: `h5py>=3.14.0`
  - Format metadata: "neuro-recording-v1" with JSON-encoded simulation parameters
  - Compression: gzip (configurable chunk size)
  - Implementation: `HDF5Recorder` class (`model/main.py` lines 169-226)

- Parquet - Apache Arrow columnar format
  - Location: Configurable via `--parquet <path>` argument
  - Client: `pyarrow>=23.0.1` (pq.ParquetWriter)
  - Spike data: Separate `.spikes.parquet` file alongside main recording
  - Compression: zstd (efficient columnar compression)
  - Implementation: `ParquetRecorder` class (`model/main.py` lines 229-300)

**In-Memory Storage:**
- MemoryRecorder class for simulation results (lines 148-166)
- NumPy arrays as default format when no persistence specified

**File Storage:**
- Local filesystem only - no cloud storage integration
- Simulation can write to any accessible directory
- Data directory structure controlled by user via command-line arguments

**Caching:**
- No caching layer (every frame computed on demand)
- Server backend re-samples from disk on zoom requests

## Authentication & Identity

**Not Applicable:**
- No authentication system
- Single-user, local execution only
- No user/role management

## Monitoring & Observability

**Error Tracking:**
- No external error tracking (Sentry, Rollbar, etc.)
- Python exceptions propagate to stderr

**Logs:**
- Console output only
- HTTP server logging disabled (`log_message` returns None, line 1234-1235)
- No persistent logging to files

## CI/CD & Deployment

**Hosting:**
- No cloud deployment (local execution only)

**CI Pipeline:**
- Not configured - no CI/CD system detected

**Local Server (Optional):**
- HTTP server mode: `http://127.0.0.1:8050` (hardcoded)
- Serves interactive visualization via `ThreadingHTTPServer` (stdlib)
- Three endpoints: `/` (HTML), `/figure` (JSON), `/plotly.min.js` (static JavaScript)
- No HTTPS/TLS (development-only)

## Environment Configuration

**Required Environment Vars:**
- None - All configuration via command-line arguments

**Optional Configs:**
- `.python-version` - Python 3.12 (read by version managers like pyenv/rye)
- `.envrc` - Listed in `.gitignore` but not used by application

**Secrets Location:**
- Not applicable - no API keys or credentials required
- `.env` is in `.gitignore` but project doesn't reference it

## Webhooks & Callbacks

**Incoming:**
- None - Project doesn't accept incoming webhooks

**Outgoing:**
- None - Project doesn't trigger external webhooks

## Data Exchange

**Output Formats:**
- HDF5 files: Scientific data with metadata
- Parquet files: Tabular data (row/spike data)
- HTML: Interactive Plotly visualizations
- PNG: Static matplotlib figures

**Input Formats:**
- HDF5: Can load recorded simulations for reprocessing
- Parquet: Can load recorded simulations for reprocessing
- Command-line arguments: Simulation parameter configuration

**No External Data Sources:**
- Pure computational model (no biological data fetched from databases)
- Initial conditions and parameters hardcoded or command-line specified

---

*Integration audit: 2026-03-13*
