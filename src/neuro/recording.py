"""Streaming Parquet recorder for the simulate() loop.

Two output files per run:

    <base>.parquet           per-step series (V, w, r_post, R, M, …)
    <base>.spikes.parquet    one row per spike  (t, spike_type)

``spike_parquet_path(<base>.parquet)`` derives the spikes path; both
recording and plotting use it so they agree on the layout.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

import pyarrow as pa
import pyarrow.parquet as pq

from neuro.params import Params, series_keys


class RecorderSummary(TypedDict):
    parquet_path: str
    spikes_path: str
    rows_written: int
    spikes_written: int


def spike_parquet_path(path: str | Path) -> Path:
    base = Path(path)
    return base.with_name(f"{base.stem}.spikes.parquet")


class ParquetRecorder:
    def __init__(self, path: str | Path, chunk_rows: int, params: Params):
        self.path = Path(path)
        self.spike_path = spike_parquet_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.ser_keys = series_keys(params.n_pre)
        metadata = {
            b"format": b"neuro-recording-v1",
            b"params_json": json.dumps(asdict(params)).encode("utf-8"),
        }
        self.schema = pa.schema(
            [(key, pa.float64()) for key in self.ser_keys], metadata=metadata,
        )
        self.spike_schema = pa.schema(
            [("t", pa.float64()), ("spike_type", pa.string())], metadata=metadata,
        )
        self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.spike_writer = pq.ParquetWriter(self.spike_path, self.spike_schema, compression="zstd")
        self.chunk_rows = chunk_rows
        self.buffer: dict[str, list[float]] = {key: [] for key in self.ser_keys}
        self.spike_buffer: dict[str, list] = {"t": [], "spike_type": []}
        self.rows_written = 0
        self.spikes_written = 0

    def _flush_rows(self) -> None:
        if not self.buffer["t"]:
            return
        table = pa.Table.from_pydict(
            {key: pa.array(values, type=pa.float64()) for key, values in self.buffer.items()},
            schema=self.schema,
        )
        self.writer.write_table(table)
        self.buffer = {key: [] for key in self.ser_keys}

    def _flush_spikes(self) -> None:
        if not self.spike_buffer["t"]:
            return
        table = pa.Table.from_pydict(
            {
                "t": pa.array(self.spike_buffer["t"], type=pa.float64()),
                "spike_type": pa.array(self.spike_buffer["spike_type"], type=pa.string()),
            },
            schema=self.spike_schema,
        )
        self.spike_writer.write_table(table)
        self.spike_buffer = {"t": [], "spike_type": []}

    def append(self, row: dict[str, float]) -> None:
        for key, value in row.items():
            self.buffer[key].append(float(value))
        self.rows_written += 1
        if len(self.buffer["t"]) >= self.chunk_rows:
            self._flush_rows()

    def append_spike(self, key: str, value: float) -> None:
        self.spike_buffer["t"].append(float(value))
        self.spike_buffer["spike_type"].append(key.removesuffix("_spike_times"))
        self.spikes_written += 1
        if len(self.spike_buffer["t"]) >= self.chunk_rows:
            self._flush_spikes()

    def finalize(self) -> RecorderSummary:
        self._flush_rows()
        self._flush_spikes()
        self.writer.close()
        self.spike_writer.close()
        return {
            "parquet_path": str(self.path),
            "spikes_path": str(self.spike_path),
            "rows_written": self.rows_written,
            "spikes_written": self.spikes_written,
        }
