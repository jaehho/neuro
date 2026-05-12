"""Parquet loading + envelope downsampling for the plot viewer.

For long runs the on-disk parquet is much bigger than what plotly can
render. ``load_time_series_frame`` performs a *min/max envelope*
downsample within an optional [x0, x1] window so peaks stay visible
even when we drop most of the data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from neuro.recording import spike_parquet_path


def _envelope_bucket_size(total_rows: int, max_points: int, series_count: int = 1) -> int:
    if max_points <= 0:
        raise ValueError("max_points must be positive.")
    if total_rows <= max_points:
        return max(1, total_rows)
    points_per_bucket = max(4 * max(1, series_count), 1)
    bucket_count = max(1, max_points // points_per_bucket)
    return max(1, (total_rows + bucket_count - 1) // bucket_count)


def _filter_spike_times(
    spikes: dict[str, np.ndarray], x0: float | None, x1: float | None,
) -> dict[str, np.ndarray]:
    if x0 is None and x1 is None:
        return spikes
    out: dict[str, np.ndarray] = {}
    for key, values in spikes.items():
        mask = np.ones(len(values), dtype=bool)
        if x0 is not None:
            mask &= values >= x0
        if x1 is not None:
            mask &= values <= x1
        out[key] = values[mask]
    return out


def read_spike_times(path: str | Path, x0: float | None = None, x1: float | None = None,
                     ) -> dict[str, np.ndarray]:
    """Load the sidecar spikes parquet for *path* and split by spike_type."""
    spike_path = spike_parquet_path(path)
    if not spike_path.exists():
        return {}
    spikes = pl.read_parquet(str(spike_path))
    if spikes.is_empty():
        return {}
    out: dict[str, np.ndarray] = {}
    for spike_type in spikes.get_column("spike_type").unique().to_list():
        times = spikes.filter(pl.col("spike_type") == spike_type).get_column("t").to_numpy()
        out[f"{spike_type}_spike_times"] = times
    return _filter_spike_times(out, x0, x1)


def load_time_series_frame(
    path: str | Path,
    columns: list[str],
    max_points: int = 40_000,
    x0: float | None = None,
    x1: float | None = None,
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Read a windowed slice of *path*, envelope-downsampled to ~max_points rows.

    Returns ``(frame, spikes)`` where ``frame`` has at most ~max_points rows
    that include both the per-bucket min and max of every requested data
    column (so spikes / outliers stay visible) and ``spikes`` maps
    ``{type}_spike_times`` → np.ndarray within [x0, x1].
    """
    scan = pl.scan_parquet(str(path))
    if x0 is not None:
        scan = scan.filter(pl.col("t") >= x0)
    if x1 is not None:
        scan = scan.filter(pl.col("t") <= x1)

    total_rows = int(scan.select(pl.len().alias("n")).collect()["n"][0])
    if total_rows <= max_points:
        frame = scan.select(columns).collect()
        return frame, read_spike_times(path, x0, x1)

    data_columns = [c for c in columns if c != "t"]
    bucket_size = _envelope_bucket_size(total_rows, max_points, series_count=len(data_columns))
    scan = scan.select(columns).with_row_index("row_idx")
    aggs = [
        pl.col("row_idx").first().alias("first_idx"),
        pl.col("row_idx").last().alias("last_idx"),
    ]
    for column in data_columns:
        aggs.extend([
            pl.col(column).arg_min().alias(f"{column}__arg_min"),
            pl.col(column).arg_max().alias(f"{column}__arg_max"),
        ])
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
        scan.filter(pl.col("row_idx").is_in(idx)).collect().sort("row_idx").drop("row_idx")
    )
    return frame, read_spike_times(path, x0, x1)
