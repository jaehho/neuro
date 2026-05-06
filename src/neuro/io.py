"""Loaders for HDF5/Parquet recordings, with min-max envelope downsampling.

For long runs the on-disk recording is much bigger than what plotly can
render.  ``load_time_series_frame`` performs a *min/max envelope*
downsample so peaks stay visible; ``load_plot_frame`` is the simpler
strided variant used when adaptive resolution isn't needed.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import polars as pl

from neuro.params import SERIES_KEYS, SPIKE_KEYS
from neuro.recording import _envelope_bucket_size, _sampling_stride, spike_parquet_path


def polars_frame_from_hdf5(
    path: str | Path,
    columns: list[str] | None = None,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    lazy: bool = True,
):
    selected = columns or SERIES_KEYS
    with h5py.File(path, "r") as handle:
        arrays = {key: np.asarray(handle[key][start:stop:stride]) for key in selected}  # type: ignore[index]
    frame = pl.DataFrame(arrays)
    return frame.lazy() if lazy else frame


def polars_frame_from_parquet(
    path: str | Path,
    columns: list[str] | None = None,
    stride: int = 1,
    lazy: bool = True,
):
    selected = columns or SERIES_KEYS
    frame = pl.scan_parquet(str(path)).select(selected)
    if stride > 1:
        frame = (
            frame.with_row_index("row_idx")
            .filter((pl.col("row_idx") % stride) == 0)
            .drop("row_idx")
        )
    return frame if lazy else frame.collect()


def _filter_spike_times(spikes: dict[str, np.ndarray], x0: float | None = None, x1: float | None = None) -> dict[str, np.ndarray]:
    if x0 is None and x1 is None:
        return spikes

    filtered = {}
    for key, values in spikes.items():
        mask = np.ones(len(values), dtype=bool)
        if x0 is not None:
            mask &= values >= x0
        if x1 is not None:
            mask &= values <= x1
        filtered[key] = values[mask]
    return filtered


def read_spike_times_hdf5(path: str | Path, x0: float | None = None, x1: float | None = None) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        spikes = {key: np.asarray(handle[key]) for key in SPIKE_KEYS}  # type: ignore[index]
    return _filter_spike_times(spikes, x0=x0, x1=x1)


def read_spike_times_parquet(path: str | Path, x0: float | None = None, x1: float | None = None) -> dict[str, np.ndarray]:
    spike_path = spike_parquet_path(path)
    if not spike_path.exists():
        return {k: np.array([]) for k in SPIKE_KEYS}

    spikes = pl.read_parquet(str(spike_path))
    result: dict[str, np.ndarray] = {}
    for spike_type in spikes.get_column("spike_type").unique().to_list():
        times = spikes.filter(pl.col("spike_type") == spike_type).get_column("t").to_numpy()
        result[f"{spike_type}_spike_times"] = times
    if "post_spike_times" not in result:
        result["post_spike_times"] = np.array([])
    return _filter_spike_times(result, x0=x0, x1=x1)


def _downsample_memory_rec(rec: dict[str, np.ndarray], max_points: int) -> dict[str, np.ndarray]:
    if len(rec["t"]) <= max_points:
        return rec
    stride = _sampling_stride(len(rec["t"]), max_points)
    sampled = {
        key: values[::stride] if key not in SPIKE_KEYS else values
        for key, values in rec.items()
    }
    return sampled


def _add_bucket_extrema_indices(
    indices: set[int],
    values,
    total_rows: int,
    bucket_size: int,
    offset: int = 0,
) -> None:
    if total_rows == 0:
        return

    indices.add(offset)
    indices.add(offset + total_rows - 1)

    for local_start in range(0, total_rows, bucket_size):
        local_stop = min(local_start + bucket_size, total_rows)
        start = offset + local_start
        stop = offset + local_stop
        chunk = np.asarray(values[start:stop])
        if chunk.size == 0:
            continue
        candidates = {
            start,
            stop - 1,
            start + int(np.argmin(chunk)),
            start + int(np.argmax(chunk)),
        }
        indices.update(candidates)


def _collect_memory_envelope_frame(
    rec: dict[str, np.ndarray],
    columns: list[str],
    max_points: int,
    x0: float | None = None,
    x1: float | None = None,
):
    if x0 is None and x1 is None:
        series = {key: rec[key] for key in columns}
    else:
        mask = np.ones(len(rec["t"]), dtype=bool)
        if x0 is not None:
            mask &= rec["t"] >= x0
        if x1 is not None:
            mask &= rec["t"] <= x1
        series = {key: rec[key][mask] for key in columns}

    total_rows = len(series["t"])
    if total_rows <= max_points:
        frame = pl.DataFrame(series)
        spikes = _filter_spike_times({key: rec[key] for key in SPIKE_KEYS}, x0=x0, x1=x1)
        return frame, spikes

    data_columns = [column for column in columns if column != "t"]
    bucket_size = _envelope_bucket_size(total_rows, max_points, series_count=len(data_columns))
    indices: set[int] = set()
    for column in data_columns:
        _add_bucket_extrema_indices(indices, series[column], total_rows, bucket_size)

    idx = np.asarray(sorted(indices), dtype=np.int64)
    frame = pl.DataFrame({key: series[key][idx] for key in columns})
    spikes = _filter_spike_times({key: rec[key] for key in SPIKE_KEYS}, x0=x0, x1=x1)
    return frame, spikes


def _collect_hdf5_envelope_frame(
    path: str | Path,
    columns: list[str],
    max_points: int,
    x0: float | None = None,
    x1: float | None = None,
):
    with h5py.File(path, "r") as handle:
        total_rows = int(handle.attrs["rows_written"])  # type: ignore[arg-type]
        if total_rows == 0:
            frame = pl.DataFrame({key: np.array([]) for key in columns})
            spikes = read_spike_times_hdf5(path, x0=x0, x1=x1)
            return frame, spikes

        t_ds = handle["t"]
        t0 = float(t_ds[0])  # type: ignore[arg-type]
        dt = float(t_ds[1] - t_ds[0]) if total_rows > 1 else 1.0  # type: ignore[operator]
        start_idx = 0 if x0 is None else max(0, int(np.floor((x0 - t0) / dt)))
        stop_idx = total_rows if x1 is None else min(total_rows, int(np.ceil((x1 - t0) / dt)) + 1)
        slice_rows = max(0, stop_idx - start_idx)

        if slice_rows <= max_points:
            frame = pl.DataFrame({key: np.asarray(handle[key][start_idx:stop_idx]) for key in columns})  # type: ignore[index]
            spikes = read_spike_times_hdf5(path, x0=x0, x1=x1)
            return frame, spikes

        data_columns = [column for column in columns if column != "t"]
        bucket_size = _envelope_bucket_size(slice_rows, max_points, series_count=len(data_columns))
        indices: set[int] = set()
        for column in data_columns:
            _add_bucket_extrema_indices(indices, handle[column], slice_rows, bucket_size, offset=start_idx)  # type: ignore[arg-type]

        idx = np.asarray(sorted(indices), dtype=np.int64)
        frame = pl.DataFrame({key: np.asarray(handle[key][idx]) for key in columns})  # type: ignore[index]
    spikes = read_spike_times_hdf5(path, x0=x0, x1=x1)
    return frame, spikes


def _count_parquet_rows(path: str | Path) -> int:
    count_df = pl.scan_parquet(str(path)).select(pl.len().alias("n")).collect()
    return int(count_df["n"][0])


def _collect_parquet_envelope_frame(
    path: str | Path,
    columns: list[str],
    max_points: int,
    x0: float | None = None,
    x1: float | None = None,
):
    scan = pl.scan_parquet(str(path))
    if x0 is not None:
        scan = scan.filter(pl.col("t") >= x0)
    if x1 is not None:
        scan = scan.filter(pl.col("t") <= x1)

    total_rows = int(scan.select(pl.len().alias("n")).collect()["n"][0])
    if total_rows <= max_points:
        frame = scan.select(columns).collect()
        spikes = read_spike_times_parquet(path, x0=x0, x1=x1)
        return frame, spikes

    data_columns = [column for column in columns if column != "t"]
    bucket_size = _envelope_bucket_size(total_rows, max_points, series_count=len(data_columns))
    scan = scan.select(columns).with_row_index("row_idx")
    aggs = [
        pl.col("row_idx").first().alias("first_idx"),
        pl.col("row_idx").last().alias("last_idx"),
    ]
    for column in data_columns:
        aggs.extend(
            [
                pl.col(column).arg_min().alias(f"{column}__arg_min"),
                pl.col(column).arg_max().alias(f"{column}__arg_max"),
            ]
        )

    summary = (
        scan.group_by((pl.col("row_idx") // bucket_size).alias("bucket"), maintain_order=True)
        .agg(*aggs)
        .collect()
    )

    indices: set[int] = {0, total_rows - 1}
    for row in summary.iter_rows(named=True):
        first_idx = int(row["first_idx"])
        last_idx = int(row["last_idx"])
        indices.add(first_idx)
        indices.add(last_idx)
        for column in data_columns:
            indices.add(first_idx + int(row[f"{column}__arg_min"]))
            indices.add(first_idx + int(row[f"{column}__arg_max"]))

    idx = sorted(indices)
    frame = (
        scan
        .filter(pl.col("row_idx").is_in(idx))
        .collect()
        .sort("row_idx")
        .drop("row_idx")
    )
    spikes = read_spike_times_parquet(path, x0=x0, x1=x1)
    return frame, spikes


def load_time_series_frame(
    rec_or_path,
    columns: list[str],
    max_points: int = 40_000,
    x0: float | None = None,
    x1: float | None = None,
):
    if isinstance(rec_or_path, dict) and "t" in rec_or_path:
        return _collect_memory_envelope_frame(rec_or_path, columns=columns, max_points=max_points, x0=x0, x1=x1)

    if isinstance(rec_or_path, dict):
        if "parquet_path" in rec_or_path:
            return load_time_series_frame(rec_or_path["parquet_path"], columns=columns, max_points=max_points, x0=x0, x1=x1)
        if "hdf5_path" in rec_or_path:
            return load_time_series_frame(rec_or_path["hdf5_path"], columns=columns, max_points=max_points, x0=x0, x1=x1)

    if isinstance(rec_or_path, (str, Path)):
        path = Path(rec_or_path)
        if path.suffix.lower() == ".parquet":
            return _collect_parquet_envelope_frame(path, columns=columns, max_points=max_points, x0=x0, x1=x1)
        return _collect_hdf5_envelope_frame(path, columns=columns, max_points=max_points, x0=x0, x1=x1)

    raise TypeError(f"Unsupported plot source: {type(rec_or_path)!r}")


def load_plot_frame(rec_or_path, columns: list[str], max_points: int = 40_000):
    if isinstance(rec_or_path, dict) and "t" in rec_or_path:
        sampled = _downsample_memory_rec(rec_or_path, max_points=max_points)
        frame = pl.DataFrame({key: sampled[key] for key in columns})
        spikes = {key: sampled[key] for key in SPIKE_KEYS}
        return frame, spikes

    if isinstance(rec_or_path, dict):
        if "parquet_path" in rec_or_path:
            return load_plot_frame(rec_or_path["parquet_path"], columns=columns, max_points=max_points)
        if "hdf5_path" in rec_or_path:
            return load_plot_frame(rec_or_path["hdf5_path"], columns=columns, max_points=max_points)

    if isinstance(rec_or_path, (str, Path)):
        path = Path(rec_or_path)
        suffix = path.suffix.lower()

        if suffix == ".parquet":
            total = _count_parquet_rows(path)
            stride = _sampling_stride(total, max_points)
            frame = polars_frame_from_parquet(path, columns=columns, stride=stride, lazy=False)
            spikes = read_spike_times_parquet(path)
            return frame, spikes

        with h5py.File(path, "r") as handle:
            total = int(handle.attrs["rows_written"])  # type: ignore[arg-type]
        stride = _sampling_stride(total, max_points)
        frame = polars_frame_from_hdf5(path, columns=columns, stride=stride, lazy=False)
        spikes = read_spike_times_hdf5(path)
        return frame, spikes

    raise TypeError(f"Unsupported plot source: {type(rec_or_path)!r}")
