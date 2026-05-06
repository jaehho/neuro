"""Recorders: in-memory, HDF5, Parquet, and Multi (fan-out).

A ``Recorder`` collects per-step rows of state and per-event spike times.
``simulate()`` calls ``append()`` and ``append_spike()``; the recorder
decides where the bytes go.  ``ParquetRecorder`` is the right choice for
long runs — it streams to disk in chunks instead of holding everything
in memory.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from neuro.params import Params, series_keys, spike_keys


def _sampling_stride(total_rows: int, max_points: int) -> int:
    if max_points <= 0:
        raise ValueError("max_points must be positive.")
    return max(1, (total_rows + max_points - 1) // max_points)


def _envelope_bucket_size(total_rows: int, max_points: int, series_count: int = 1) -> int:
    if max_points <= 0:
        raise ValueError("max_points must be positive.")
    if total_rows <= max_points:
        return max(1, total_rows)
    points_per_bucket = max(4 * max(1, series_count), 1)
    bucket_count = max(1, max_points // points_per_bucket)
    return max(1, (total_rows + bucket_count - 1) // bucket_count)


def spike_parquet_path(path: str | Path) -> Path:
    base = Path(path)
    return base.with_name(f"{base.stem}.spikes.parquet")


class Recorder:
    def append(self, row: dict[str, float]) -> None:
        raise NotImplementedError

    def append_spike(self, key: str, value: float) -> None:
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError


class MemoryRecorder(Recorder):
    def __init__(self, capacity: int, ser_keys: list[str], spk_keys: list[str]):
        self.data = {key: np.zeros(capacity, dtype=np.float64) for key in ser_keys}
        self.spikes: dict[str, list[float]] = {key: [] for key in spk_keys}
        self.spk_keys = spk_keys
        self.k = 0

    def append(self, row: dict[str, float]) -> None:
        for key, value in row.items():
            self.data[key][self.k] = value
        self.k += 1

    def append_spike(self, key: str, value: float) -> None:
        self.spikes[key].append(value)

    def finalize(self) -> dict[str, np.ndarray]:
        rec = {key: values[: self.k] for key, values in self.data.items()}
        for key, values in self.spikes.items():
            rec[key] = np.asarray(values, dtype=np.float64)
        return rec


class HDF5Recorder(Recorder):
    def __init__(self, path: str | Path, expected_rows: int, chunk_rows: int, params: Params,
                 ser_keys: list[str], spk_keys: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "w")
        self.file.attrs["format"] = "neuro-recording-v1"
        self.file.attrs["params_json"] = json.dumps(asdict(params))
        self.file.attrs["expected_rows"] = expected_rows
        self.file.attrs["rows_written"] = 0

        chunk = max(1, min(chunk_rows, expected_rows or chunk_rows))
        self.datasets = {
            key: self.file.create_dataset(
                key,
                shape=(expected_rows,),
                maxshape=(None,),
                dtype="f8",
                chunks=(chunk,),
                compression="gzip",
            )
            for key in ser_keys
        }
        self.spike_datasets = {
            key: self.file.create_dataset(
                key,
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=(chunk,),
                compression="gzip",
            )
            for key in spk_keys
        }
        self.k = 0
        self.spike_counts = {key: 0 for key in spk_keys}

    def append(self, row: dict[str, float]) -> None:
        for key, value in row.items():
            self.datasets[key][self.k] = value
        self.k += 1

    def append_spike(self, key: str, value: float) -> None:
        ds = self.spike_datasets[key]
        idx = self.spike_counts[key]
        ds.resize((idx + 1,))
        ds[idx] = value
        self.spike_counts[key] = idx + 1

    def finalize(self) -> dict[str, str | int]:
        for ds in self.datasets.values():
            ds.resize((self.k,))
        self.file.attrs["rows_written"] = self.k
        self.file.flush()
        self.file.close()
        return {"hdf5_path": str(self.path), "rows_written": self.k}


class ParquetRecorder(Recorder):
    def __init__(self, path: str | Path, chunk_rows: int, params: Params,
                 ser_keys: list[str], spk_keys: list[str]):
        self.path = Path(path)
        self.spike_path = spike_parquet_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            b"format": b"neuro-recording-v1",
            b"params_json": json.dumps(asdict(params)).encode("utf-8"),
        }
        self.ser_keys = ser_keys
        self.schema = pa.schema([(key, pa.float64()) for key in ser_keys], metadata=metadata)
        self.spike_schema = pa.schema(
            [("t", pa.float64()), ("spike_type", pa.string())],
            metadata=metadata,
        )
        self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.spike_writer = pq.ParquetWriter(self.spike_path, self.spike_schema, compression="zstd")
        self.chunk_rows = chunk_rows
        self.buffer: dict[str, list[float]] = {key: [] for key in ser_keys}
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
        spike_type = key.removesuffix("_spike_times")
        self.spike_buffer["spike_type"].append(spike_type)
        self.spikes_written += 1
        if len(self.spike_buffer["t"]) >= self.chunk_rows:
            self._flush_spikes()

    def finalize(self) -> dict[str, str | int]:
        self._flush_rows()
        self._flush_spikes()
        self.writer.close()
        self.spike_writer.close()
        return {
            "parquet_path": str(self.path),
            "parquet_spikes_path": str(self.spike_path),
            "rows_written": self.rows_written,
            "spikes_written": self.spikes_written,
        }


class MultiRecorder(Recorder):
    def __init__(self, recorders: list[Recorder]):
        self.recorders = recorders

    def append(self, row: dict[str, float]) -> None:
        for recorder in self.recorders:
            recorder.append(row)

    def append_spike(self, key: str, value: float) -> None:
        for recorder in self.recorders:
            recorder.append_spike(key, value)

    def finalize(self):
        results = [recorder.finalize() for recorder in self.recorders]
        if len(results) == 1:
            return results[0]
        merged = {}
        for result in results:
            if isinstance(result, dict):
                merged.update(result)
        return merged


def _build_recorder(
    p: Params,
    hdf5_path: str | None,
    parquet_path: str | None,
    chunk_rows: int,
) -> Recorder:
    n = int(p.T / p.dt)
    rec_step = max(1, int(p.record_every / p.dt))
    expected_rows = n // rec_step + 2
    ser = series_keys(p.n_pre)
    spk = spike_keys(p.n_pre)

    recorders: list[Recorder] = []
    if hdf5_path:
        recorders.append(HDF5Recorder(hdf5_path, expected_rows=expected_rows, chunk_rows=chunk_rows,
                                      params=p, ser_keys=ser, spk_keys=spk))
    if parquet_path:
        recorders.append(ParquetRecorder(parquet_path, chunk_rows=chunk_rows, params=p,
                                         ser_keys=ser, spk_keys=spk))

    if not recorders:
        return MemoryRecorder(expected_rows, ser_keys=ser, spk_keys=spk)
    if len(recorders) == 1:
        return recorders[0]
    return MultiRecorder(recorders)
