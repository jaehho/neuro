## Data handling

`main.py` can now stream the recorded state directly into HDF5 or Parquet instead of keeping the full trace in RAM.

Example:

```bash
python model/main.py --hdf5 data/simulation.h5 --plot-backend plotly
```

```bash
python model/main.py --parquet data/simulation.parquet --plot-backend plotly
```

The HDF5 path writes compressed chunked datasets. The Parquet path writes a streamed table to `data/simulation.parquet` and a companion spike table to `data/simulation.spikes.parquet`.

## Plotting

- `main.py` uses Plotly for the main time-series dashboard when `--plot-backend plotly` is selected.
- `trajectories.py` builds an interactive pairwise trajectory view:

```bash
python model/trajectories.py --hdf5 data/simulation.h5
python model/trajectories.py --parquet data/simulation.parquet
```

The Plotly exporter now writes `plotly.min.js` as a separate local asset instead of embedding the full bundle inline.

Both plotting paths read sampled slices into Polars frames before plotting. Parquet sources use Polars' lazy scan path directly.

## Zoom-adaptive viewer

For long runs, the static HTML export is still a fixed-resolution snapshot. To resample from disk as you zoom, run the local viewer:

```bash
python model/main.py --parquet data/simulation.parquet --plot-backend server
```

That starts a local HTTP server and rebuilds the Plotly figure for the current x-range after each zoom interaction.
