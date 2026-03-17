from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from typing import Annotated
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import numpy as np
import typer

try:
    import h5py
except ImportError:
    h5py = None

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    pio = None
    make_subplots = None


SERIES_KEYS = [
    "t",
    "V",
    "w",
    "I_s",
    "x_pre",
    "y_post",
    "E",
    "r_pre",
    "r_post",
    "R",
    "R_bar",
    "M",
    "pre_spike_bin",
    "post_spike_bin",
    "is_refractory",
]

SPIKE_KEYS = ["pre_spike_times", "post_spike_times"]

V_IDX = 0
I_S_IDX = 1
X_PRE_IDX = 2
Y_POST_IDX = 3
E_IDX = 4
R_PRE_IDX = 5
R_POST_IDX = 6
RBAR_IDX = 7
W_IDX = 8


@dataclass
class Params:
    T: float = 20.0
    dt: float = 1e-4
    seed: int = 1
    record_every: float = 1e-4
    method: str = "euler"

    r_pre_rate: float = 20.0

    tau_m: float = 0.02
    E_L: float = -65.0
    V_reset: float = -70.0
    theta: float = -50.0
    tau_ref: float = 0.003
    V0: float = -65.0
    ref_remaining0: float = 0.0

    tau_s: float = 0.005
    R_m: float = 50.0
    I_s0: float = 0.0

    tau_plus: float = 0.02
    tau_minus: float = 0.02
    x_pre0: float = 0.0
    y_post0: float = 0.0

    tau_r: float = 0.5
    r_pre0: float = 0.0
    r_post0: float = 0.0

    tau_e: float = 0.5
    E0: float = 0.0

    tau_Rbar: float = 5.0
    R_bar0: float = 0.0
    alpha: float = 0.5

    rate_mode: str = "exp"
    rate_window: float = 0.5

    w0: float = 2.0
    wmax: float = 10.0
    eta_plus: float = 1e-4
    eta_minus: float = 1e-4


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
    def __init__(self, capacity: int):
        self.data = {key: np.zeros(capacity, dtype=np.float64) for key in SERIES_KEYS}
        self.spikes = {key: [] for key in SPIKE_KEYS}
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
    def __init__(self, path: str | Path, expected_rows: int, chunk_rows: int, params: Params):
        if h5py is None:
            raise ImportError("h5py is required for HDF5 output. Install it with `uv add h5py`.")

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
            for key in SERIES_KEYS
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
            for key in SPIKE_KEYS
        }
        self.k = 0
        self.spike_counts = {key: 0 for key in SPIKE_KEYS}

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
    def __init__(self, path: str | Path, chunk_rows: int, params: Params):
        if pa is None or pq is None:
            raise ImportError("pyarrow is required for Parquet export. Install it with `uv add pyarrow`.")

        self.path = Path(path)
        self.spike_path = spike_parquet_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            b"format": b"neuro-recording-v1",
            b"params_json": json.dumps(asdict(params)).encode("utf-8"),
        }
        self.schema = pa.schema([(key, pa.float64()) for key in SERIES_KEYS], metadata=metadata)
        self.spike_schema = pa.schema(
            [("t", pa.float64()), ("spike_type", pa.string())],
            metadata=metadata,
        )
        self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.spike_writer = pq.ParquetWriter(self.spike_path, self.spike_schema, compression="zstd")
        self.chunk_rows = chunk_rows
        self.buffer = {key: [] for key in SERIES_KEYS}
        self.spike_buffer = {"t": [], "spike_type": []}
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
        self.buffer = {key: [] for key in SERIES_KEYS}

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
        self.spike_buffer["spike_type"].append("pre" if key == "pre_spike_times" else "post")
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

    recorders: list[Recorder] = []
    if hdf5_path:
        recorders.append(HDF5Recorder(hdf5_path, expected_rows=expected_rows, chunk_rows=chunk_rows, params=p))
    if parquet_path:
        recorders.append(ParquetRecorder(parquet_path, chunk_rows=chunk_rows, params=p))

    if not recorders:
        return MemoryRecorder(expected_rows)
    if len(recorders) == 1:
        return recorders[0]
    return MultiRecorder(recorders)


def _reward_terms(r_pre: float, r_post: float, R_bar: float, alpha: float) -> tuple[float, float]:
    R = -(r_post - alpha * r_pre) ** 2
    M = R - R_bar
    return R, M


def _pack_state(
    V: float,
    I_s: float,
    x_pre: float,
    y_post: float,
    E: float,
    r_pre: float,
    r_post: float,
    R_bar: float,
    w: float,
) -> np.ndarray:
    return np.array([V, I_s, x_pre, y_post, E, r_pre, r_post, R_bar, w], dtype=np.float64)


def _smooth_rhs(
    y: np.ndarray,
    p: Params,
    *,
    voltage_active: bool,
    rate_pre: float | None = None,
    rate_post: float | None = None,
) -> np.ndarray:
    rhs = np.zeros_like(y)

    V = float(y[V_IDX])
    I_s = float(y[I_S_IDX])
    x_pre = float(y[X_PRE_IDX])
    y_post = float(y[Y_POST_IDX])
    E = float(y[E_IDX])
    r_pre = float(y[R_PRE_IDX])
    r_post = float(y[R_POST_IDX])
    R_bar = float(y[RBAR_IDX])
    w = float(np.clip(y[W_IDX], 0.0, p.wmax))

    rr_pre = rate_pre if rate_pre is not None else r_pre
    rr_post = rate_post if rate_post is not None else r_post
    R, M = _reward_terms(rr_pre, rr_post, R_bar, p.alpha)

    if voltage_active:
        rhs[V_IDX] = (-(V - p.E_L) + p.R_m * w * I_s) / p.tau_m

    rhs[I_S_IDX] = -I_s / p.tau_s
    rhs[X_PRE_IDX] = -x_pre / p.tau_plus
    rhs[Y_POST_IDX] = -y_post / p.tau_minus
    rhs[E_IDX] = -E / p.tau_e
    rhs[R_PRE_IDX] = -r_pre / p.tau_r
    rhs[R_POST_IDX] = -r_post / p.tau_r
    rhs[RBAR_IDX] = (-R_bar + R) / p.tau_Rbar
    rhs[W_IDX] = M * E

    return rhs


def _advance_state(
    y: np.ndarray,
    dt: float,
    p: Params,
    *,
    method: str,
    voltage_active: bool,
    rate_pre: float | None = None,
    rate_post: float | None = None,
) -> np.ndarray:
    rk = dict(voltage_active=voltage_active, rate_pre=rate_pre, rate_post=rate_post)
    if dt <= 0.0:
        out = y.copy()
    elif method == "euler":
        out = y + dt * _smooth_rhs(y, p, **rk)
    elif method == "rk4":
        k1 = _smooth_rhs(y, p, **rk)
        k2 = _smooth_rhs(y + 0.5 * dt * k1, p, **rk)
        k3 = _smooth_rhs(y + 0.5 * dt * k2, p, **rk)
        k4 = _smooth_rhs(y + dt * k3, p, **rk)
        out = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        raise ValueError(f"Unknown integration method: {method!r}")

    out[W_IDX] = float(np.clip(out[W_IDX], 0.0, p.wmax))
    if not voltage_active:
        out[V_IDX] = p.V_reset
    return out


def _crossing_fraction(v0: float, v1: float, threshold: float) -> float | None:
    if v0 >= threshold or v1 < threshold:
        return None
    dv = v1 - v0
    if dv <= 0.0:
        return 1.0
    return float(np.clip((threshold - v0) / dv, 0.0, 1.0))


def _row_from_state(
    t: float,
    y: np.ndarray,
    p: Params,
    *,
    pre_spike: int,
    post_spike: int,
    is_refractory: int,
    rate_pre: float | None = None,
    rate_post: float | None = None,
) -> dict[str, float]:
    rr_pre = rate_pre if rate_pre is not None else float(y[R_PRE_IDX])
    rr_post = rate_post if rate_post is not None else float(y[R_POST_IDX])
    R, M = _reward_terms(rr_pre, rr_post, float(y[RBAR_IDX]), p.alpha)
    return {
        "t": t,
        "V": float(y[V_IDX]),
        "w": float(y[W_IDX]),
        "I_s": float(y[I_S_IDX]),
        "x_pre": float(y[X_PRE_IDX]),
        "y_post": float(y[Y_POST_IDX]),
        "E": float(y[E_IDX]),
        "r_pre": rr_pre,
        "r_post": rr_post,
        "R": float(R),
        "R_bar": float(y[RBAR_IDX]),
        "M": float(M),
        "pre_spike_bin": int(pre_spike),
        "post_spike_bin": int(post_spike),
        "is_refractory": int(is_refractory),
    }


def simulate(
    p: Params,
    hdf5_path: str | None = None,
    parquet_path: str | None = None,
    chunk_rows: int = 100_000,
):
    method = p.method.lower()
    if method not in {"euler", "rk4"}:
        raise ValueError("Params.method must be either 'euler' or 'rk4'.")
    use_window = p.rate_mode == "window"
    if p.rate_mode not in {"exp", "window"}:
        raise ValueError("Params.rate_mode must be either 'exp' or 'window'.")

    n = int(p.T / p.dt)
    period_steps = max(1, round(1.0 / (p.r_pre_rate * p.dt)))
    rec_step = max(1, int(p.record_every / p.dt))

    y = _pack_state(
        p.V0,
        p.I_s0,
        p.x_pre0,
        p.y_post0,
        p.E0,
        p.r_pre0,
        p.r_post0,
        p.R_bar0,
        p.w0,
    )
    ref_remaining = p.ref_remaining0

    pre_spike_buf: deque[float] = deque()
    post_spike_buf: deque[float] = deque()
    win_r_pre = 0.0
    win_r_post = 0.0

    recorder = _build_recorder(p, hdf5_path=hdf5_path, parquet_path=parquet_path, chunk_rows=chunk_rows)

    for step in range(n):
        t = step * p.dt

        pre_spike = 1 if (step % period_steps == 0) else 0
        if pre_spike:
            recorder.append_spike("pre_spike_times", t)
            y[I_S_IDX] += 1.0
            y[X_PRE_IDX] += 1.0
            y[R_PRE_IDX] += 1.0
            y[E_IDX] -= p.eta_minus * y[W_IDX] * y[Y_POST_IDX]
            if use_window:
                pre_spike_buf.append(t)

        post_spike = 0
        is_refractory = 1 if ref_remaining > 0.0 else 0

        if use_window:
            cutoff = t - p.rate_window
            while pre_spike_buf and pre_spike_buf[0] < cutoff:
                pre_spike_buf.popleft()
            while post_spike_buf and post_spike_buf[0] < cutoff:
                post_spike_buf.popleft()
            win_r_pre = len(pre_spike_buf) / p.rate_window
            win_r_post = len(post_spike_buf) / p.rate_window

        rp = win_r_pre if use_window else None
        ro = win_r_post if use_window else None

        if method == "euler":
            V = float(y[V_IDX])
            I_s = float(y[I_S_IDX])
            x_pre = float(y[X_PRE_IDX])
            y_post = float(y[Y_POST_IDX])
            E = float(y[E_IDX])
            r_pre = float(y[R_PRE_IDX])
            r_post = float(y[R_POST_IDX])
            R_bar = float(y[RBAR_IDX])
            w = float(y[W_IDX])

            I_s += p.dt * (-I_s / p.tau_s)
            x_pre += p.dt * (-x_pre / p.tau_plus)
            r_pre += p.dt * (-r_pre / p.tau_r)

            if ref_remaining <= 0.0:
                dV = (p.dt / p.tau_m) * (-(V - p.E_L) + p.R_m * w * I_s)
                V_new = V + dV
                if V < p.theta and V_new >= p.theta:
                    post_spike = 1
                    recorder.append_spike("post_spike_times", t)
                    V = p.V_reset
                    ref_remaining = p.tau_ref
                else:
                    V = V_new
            else:
                ref_remaining = max(0.0, ref_remaining - p.dt)
                V = p.V_reset

            if post_spike:
                y_post += 1.0
                r_post += 1.0
                E += p.eta_plus * (p.wmax - w) * x_pre
                if use_window:
                    post_spike_buf.append(t)
                    win_r_post = len(post_spike_buf) / p.rate_window
                    rp = win_r_pre
                    ro = win_r_post

            y_post += p.dt * (-y_post / p.tau_minus)
            r_post += p.dt * (-r_post / p.tau_r)
            E += p.dt * (-E / p.tau_e)

            rew_pre = rp if use_window else r_pre
            rew_post = ro if use_window else r_post
            R, _ = _reward_terms(rew_pre, rew_post, R_bar, p.alpha)
            R_bar += (p.dt / p.tau_Rbar) * (-R_bar + R)
            _, M = _reward_terms(rew_pre, rew_post, R_bar, p.alpha)
            w += p.dt * M * E
            w = min(p.wmax, max(0.0, w))

            y = _pack_state(V, I_s, x_pre, y_post, E, r_pre, r_post, R_bar, w)

        else:
            if ref_remaining <= 0.0:
                v0 = float(y[V_IDX])
                y_trial = _advance_state(y, p.dt, p, method="rk4", voltage_active=True, rate_pre=rp, rate_post=ro)
                frac = _crossing_fraction(v0, float(y_trial[V_IDX]), p.theta)

                if frac is None:
                    y = y_trial
                else:
                    dt1 = frac * p.dt
                    dt2 = p.dt - dt1

                    y_mid = _advance_state(y, dt1, p, method="rk4", voltage_active=True, rate_pre=rp, rate_post=ro)
                    spike_t = t + dt1
                    recorder.append_spike("post_spike_times", spike_t)
                    post_spike = 1

                    y_mid[V_IDX] = p.V_reset
                    y_mid[Y_POST_IDX] += 1.0
                    y_mid[R_POST_IDX] += 1.0
                    y_mid[E_IDX] += p.eta_plus * (p.wmax - y_mid[W_IDX]) * y_mid[X_PRE_IDX]

                    if use_window:
                        post_spike_buf.append(spike_t)
                        ro = len(post_spike_buf) / p.rate_window

                    y = _advance_state(y_mid, dt2, p, method="rk4", voltage_active=False, rate_pre=rp, rate_post=ro)
                    y[V_IDX] = p.V_reset
                    ref_remaining = max(0.0, p.tau_ref - dt2)
            else:
                y = _advance_state(y, p.dt, p, method="rk4", voltage_active=False, rate_pre=rp, rate_post=ro)
                y[V_IDX] = p.V_reset
                ref_remaining = max(0.0, ref_remaining - p.dt)

        if step % rec_step == 0:
            recorder.append(
                _row_from_state(
                    t,
                    y,
                    p,
                    pre_spike=pre_spike,
                    post_spike=post_spike,
                    is_refractory=is_refractory,
                    rate_pre=rp,
                    rate_post=ro,
                )
            )

    return recorder.finalize()


def polars_frame_from_hdf5(
    path: str | Path,
    columns: list[str] | None = None,
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    lazy: bool = True,
):
    if h5py is None:
        raise ImportError("h5py is required to read HDF5 recordings.")
    if pl is None:
        raise ImportError("polars is required for Polars-based analysis. Install it with `uv add polars`.")

    selected = columns or SERIES_KEYS
    with h5py.File(path, "r") as handle:
        arrays = {key: np.asarray(handle[key][start:stop:stride]) for key in selected}
    frame = pl.DataFrame(arrays)
    return frame.lazy() if lazy else frame


def polars_frame_from_parquet(
    path: str | Path,
    columns: list[str] | None = None,
    stride: int = 1,
    lazy: bool = True,
):
    if pl is None:
        raise ImportError("polars is required for Parquet analysis. Install it with `uv add polars`.")

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
    if h5py is None:
        raise ImportError("h5py is required to read HDF5 spike times.")
    with h5py.File(path, "r") as handle:
        spikes = {key: np.asarray(handle[key]) for key in SPIKE_KEYS}
    return _filter_spike_times(spikes, x0=x0, x1=x1)


def read_spike_times_parquet(path: str | Path, x0: float | None = None, x1: float | None = None) -> dict[str, np.ndarray]:
    if pl is None:
        raise ImportError("polars is required to read Parquet spike times.")

    spike_path = spike_parquet_path(path)
    if not spike_path.exists():
        return {"pre_spike_times": np.array([]), "post_spike_times": np.array([])}

    spikes = pl.read_parquet(str(spike_path))
    pre = spikes.filter(pl.col("spike_type") == "pre").get_column("t").to_numpy()
    post = spikes.filter(pl.col("spike_type") == "post").get_column("t").to_numpy()
    return _filter_spike_times({"pre_spike_times": pre, "post_spike_times": post}, x0=x0, x1=x1)


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
    if pl is None:
        raise ImportError("polars is required for plotting. Install it with `uv add polars`.")

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
    if h5py is None:
        raise ImportError("h5py is required to read HDF5 recordings.")
    if pl is None:
        raise ImportError("polars is required for plotting. Install it with `uv add polars`.")

    with h5py.File(path, "r") as handle:
        total_rows = int(handle.attrs["rows_written"])
        if total_rows == 0:
            frame = pl.DataFrame({key: np.array([]) for key in columns})
            spikes = read_spike_times_hdf5(path, x0=x0, x1=x1)
            return frame, spikes

        t0 = float(handle["t"][0])
        dt = float(handle["t"][1] - handle["t"][0]) if total_rows > 1 else 1.0
        start_idx = 0 if x0 is None else max(0, int(np.floor((x0 - t0) / dt)))
        stop_idx = total_rows if x1 is None else min(total_rows, int(np.ceil((x1 - t0) / dt)) + 1)
        slice_rows = max(0, stop_idx - start_idx)

        if slice_rows <= max_points:
            frame = pl.DataFrame({key: np.asarray(handle[key][start_idx:stop_idx]) for key in columns})
            spikes = read_spike_times_hdf5(path, x0=x0, x1=x1)
            return frame, spikes

        data_columns = [column for column in columns if column != "t"]
        bucket_size = _envelope_bucket_size(slice_rows, max_points, series_count=len(data_columns))
        indices: set[int] = set()
        for column in data_columns:
            _add_bucket_extrema_indices(indices, handle[column], slice_rows, bucket_size, offset=start_idx)

        idx = np.asarray(sorted(indices), dtype=np.int64)
        frame = pl.DataFrame({key: np.asarray(handle[key][idx]) for key in columns})
    spikes = read_spike_times_hdf5(path, x0=x0, x1=x1)
    return frame, spikes


def _count_parquet_rows(path: str | Path) -> int:
    if pl is None:
        raise ImportError("polars is required to read Parquet recordings.")
    count_df = pl.scan_parquet(str(path)).select(pl.len().alias("n")).collect()
    return int(count_df["n"][0])


def _collect_parquet_envelope_frame(
    path: str | Path,
    columns: list[str],
    max_points: int,
    x0: float | None = None,
    x1: float | None = None,
):
    if pl is None:
        raise ImportError("polars is required to read Parquet recordings.")

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
        if pl is None:
            raise ImportError("polars is required for plotting. Install it with `uv add polars`.")
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
            if pl is None:
                raise ImportError("polars is required to read Parquet recordings.")
            total = _count_parquet_rows(path)
            stride = _sampling_stride(total, max_points)
            frame = polars_frame_from_parquet(path, columns=columns, stride=stride, lazy=False)
            spikes = read_spike_times_parquet(path)
            return frame, spikes

        if h5py is None:
            raise ImportError("h5py is required to read HDF5 recordings.")
        with h5py.File(path, "r") as handle:
            total = int(handle.attrs["rows_written"])
        stride = _sampling_stride(total, max_points)
        frame = polars_frame_from_hdf5(path, columns=columns, stride=stride, lazy=False)
        spikes = read_spike_times_hdf5(path)
        return frame, spikes

    raise TypeError(f"Unsupported plot source: {type(rec_or_path)!r}")


def _plotly_values(values) -> list[float | None]:
    arr = np.asarray(values)
    if arr.size == 0:
        return []
    if arr.dtype.kind == "f":
        return [None if not np.isfinite(value) else float(value) for value in arr]
    return arr.tolist()


def write_plotly_html(fig, output_html: str) -> str:
    if pio is None:
        raise ImportError("plotly is required for HTML export. Install it with `uv add plotly`.")
    pio.write_html(
        fig,
        file=output_html,
        full_html=True,
        include_plotlyjs="directory",
        auto_open=False,
        validate=True,
    )
    return output_html


def plot_all_in_one_figure_matplotlib(rec, p: Params):
    t = rec["t"]
    fig, axs = plt.subplots(12, 1, figsize=(19, 20), sharex=True)

    axs[0].eventplot([rec["pre_spike_times"], rec["post_spike_times"]], lineoffsets=[1, 0], linelengths=0.8)
    axs[0].set_yticks([0, 1])
    axs[0].set_yticklabels(["post", "pre"])
    axs[0].set_title("Spike times")

    axs[1].plot(t, rec["V"])
    axs[1].axhline(p.theta, linestyle="--", label="theta")
    axs[1].axhline(p.V_reset, linestyle=":", label="V_reset")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Membrane potential V(t)")

    axs[2].plot(t, rec["I_s"])
    axs[2].set_title("Synaptic current I_s(t)")

    axs[3].plot(t, rec["x_pre"])
    axs[3].set_title("STDP pre-trace x_pre(t)")

    axs[4].plot(t, rec["y_post"])
    axs[4].set_title("STDP post-trace y_post(t)")

    axs[5].plot(t, rec["E"])
    axs[5].set_title("Eligibility trace E(t)")

    axs[6].plot(t, rec["r_pre"])
    axs[6].set_title("Pre-synaptic firing rate r_pre(t)")

    axs[7].plot(t, rec["r_post"])
    axs[7].plot(t, p.alpha * rec["r_pre"], linestyle="--", label="target")
    axs[7].legend(loc="upper right")
    axs[7].set_title("Post-synaptic firing rate r_post(t)")

    axs[8].plot(t, rec["R"])
    axs[8].set_title("Reward R(t)")

    axs[9].plot(t, rec["R_bar"])
    axs[9].set_title("Reward baseline R_bar(t)")

    axs[10].plot(t, rec["M"])
    axs[10].set_title("Modulation M(t)")

    axs[11].plot(t, rec["w"])
    axs[11].set_title("Synaptic weight w(t)")
    axs[11].set_xlabel("time (s)")

    plt.tight_layout()
    plt.savefig("simulation.png")
    plt.show()


ALL_PLOT_VARIABLES = ["V", "I_s", "x_pre", "y_post", "E", "r_pre", "r_post", "R", "R_bar", "M", "w"]

VARIABLE_TITLES = {
    "V": "Membrane potential V",
    "I_s": "Synaptic current I_s",
    "x_pre": "STDP pre-trace x_pre",
    "y_post": "STDP post-trace y_post",
    "E": "Eligibility trace E",
    "r_pre": "Pre-synaptic firing rate r_pre",
    "r_post": "Post-synaptic firing rate r_post",
    "R": "Reward R",
    "R_bar": "Reward baseline R_bar",
    "M": "Modulation M",
    "w": "Synaptic weight w",
}

MARKER_MODE_VARIABLES = {"E", "w"}


def build_all_in_one_plotly_figure(
    rec_or_path,
    p: Params,
    max_points: int = 40_000,
    x0: float | None = None,
    x1: float | None = None,
    variables: list[str] | None = None,
):
    if go is None or make_subplots is None:
        raise ImportError("plotly is required for interactive plotting. Install it with `uv add plotly`.")

    plot_vars = variables if variables is not None else ALL_PLOT_VARIABLES
    show_spikes = variables is None
    n_rows = len(plot_vars) + (1 if show_spikes else 0)

    needed_columns = set(plot_vars)
    if "r_post" in needed_columns:
        needed_columns.add("r_pre")
    columns = ["t"] + [v for v in ALL_PLOT_VARIABLES if v in needed_columns]

    frame, spikes = load_time_series_frame(rec_or_path, columns=columns, max_points=max_points, x0=x0, x1=x1)
    arrays = {col: _plotly_values(frame[col].to_numpy()) for col in columns}
    t = arrays["t"]

    titles = []
    if show_spikes:
        titles.append("Spike times")
    titles.extend(VARIABLE_TITLES[v] for v in plot_vars)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=titles,
    )

    row = 1
    if show_spikes:
        for label, key, y_base in [("pre", "pre_spike_times", 1.0), ("post", "post_spike_times", 0.0)]:
            times = spikes[key]
            n_spikes = len(times)
            if n_spikes > 0:
                xs = np.empty(3 * n_spikes)
                ys = np.empty(3 * n_spikes)
                xs[0::3] = times
                xs[1::3] = times
                xs[2::3] = np.nan
                ys[0::3] = y_base - 0.4
                ys[1::3] = y_base + 0.4
                ys[2::3] = np.nan
            else:
                xs = np.array([])
                ys = np.array([])
            fig.add_trace(
                go.Scattergl(
                    x=_plotly_values(xs),
                    y=_plotly_values(ys),
                    mode="lines",
                    name=label,
                    line={"width": 1},
                ),
                row=row,
                col=1,
            )
        row += 1

    for var in plot_vars:
        mode = "markers" if var in MARKER_MODE_VARIABLES else "lines"
        fig.add_trace(go.Scattergl(x=t, y=arrays[var], name=var, mode=mode), row=row, col=1)
        if var == "V":
            fig.add_hline(y=p.theta, line_dash="dash", row=row, col=1)
            fig.add_hline(y=p.V_reset, line_dash="dot", row=row, col=1)
        elif var == "r_post":
            fig.add_trace(go.Scattergl(x=t, y=_plotly_values(p.alpha * np.asarray(frame["r_pre"].to_numpy())), name="target"), row=row, col=1)
        row += 1

    fig.update_layout(height=max(400, 200 * n_rows), width=1400, title="Neuromodulated STDP simulation", showlegend=True)
    fig.update_xaxes(title_text="time (s)", row=n_rows, col=1)
    if x0 is not None and x1 is not None:
        fig.update_xaxes(range=[x0, x1], row=n_rows, col=1)
    return fig


def plot_all_in_one_plotly(rec_or_path, p: Params, output_html: str = "simulation.html", max_points: int = 40_000, variables: list[str] | None = None):
    fig = build_all_in_one_plotly_figure(rec_or_path, p, max_points=max_points, variables=variables)
    return write_plotly_html(fig, output_html)


def _plotly_package_js_path() -> Path:
    if go is None or pio is None:
        raise ImportError("plotly is required for the adaptive viewer. Install it with `uv add plotly`.")
    candidates = [
        Path(pio.__file__).resolve().parents[1] / "package_data" / "plotly.min.js",
        Path(go.__file__).resolve().parents[1] / "package_data" / "plotly.min.js",
        Path.cwd() / "plotly.min.js",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate Plotly JS asset. Checked: {', '.join(str(path) for path in candidates)}")


def serve_zoom_adaptive_plot(source_path: str | Path, p: Params, host: str = "127.0.0.1", port: int = 8050, max_points: int = 40_000, variables: list[str] | None = None):
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Plot source does not exist: {source}")

    plotly_js_path = _plotly_package_js_path()

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Neuromodulated STDP simulation</title>
  <style>
    html, body, #plot {{
      height: 100%;
      width: 100%;
      margin: 0;
      background: #ffffff;
    }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script src="/plotly.min.js"></script>
  <script>
    const BASE_MAX_POINTS = {max_points};
    const plotDiv = document.getElementById("plot");
    let isUpdating = false;
    let pendingTimer = null;
    let relayoutHandlerAttached = false;

    function relayoutRange(eventData) {{
      if ("xaxis.range[0]" in eventData && "xaxis.range[1]" in eventData) {{
        return [Number(eventData["xaxis.range[0]"]), Number(eventData["xaxis.range[1]"])];
      }}
      if ("xaxis12.range[0]" in eventData && "xaxis12.range[1]" in eventData) {{
        return [Number(eventData["xaxis12.range[0]"]), Number(eventData["xaxis12.range[1]"])];
      }}
      if ("xaxis.autorange" in eventData || "xaxis12.autorange" in eventData) {{
        return null;
      }}
      return undefined;
    }}

    function attachRelayoutHandler() {{
      if (relayoutHandlerAttached || typeof plotDiv.on !== "function") {{
        return;
      }}
      plotDiv.on("plotly_relayout", (eventData) => {{
        if (isUpdating) {{
          return;
        }}
        const range = relayoutRange(eventData);
        if (range === undefined) {{
          return;
        }}
        clearTimeout(pendingTimer);
        pendingTimer = setTimeout(() => updateFigure(range), 120);
      }});
      relayoutHandlerAttached = true;
    }}

    async function fetchFigure(range) {{
      const params = new URLSearchParams();
      params.set("max_points", String(BASE_MAX_POINTS));
      if (range) {{
        params.set("x0", String(range[0]));
        params.set("x1", String(range[1]));
      }}
      const response = await fetch("/figure?" + params.toString());
      if (!response.ok) {{
        throw new Error("Failed to fetch figure: " + response.status);
      }}
      return await response.json();
    }}

    async function updateFigure(range) {{
      if (isUpdating) {{
        return;
      }}
      isUpdating = true;
      try {{
        const fig = await fetchFigure(range);
        await Plotly.react(plotDiv, fig.data, fig.layout, {{ responsive: true }});
        attachRelayoutHandler();
      }} finally {{
        isUpdating = false;
      }}
    }}

    updateFigure(null);
  </script>
</body>
</html>
"""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                body = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/plotly.min.js":
                body = plotly_js_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return

            if parsed.path == "/figure":
                query = parse_qs(parsed.query)
                x0 = float(query["x0"][0]) if "x0" in query else None
                x1 = float(query["x1"][0]) if "x1" in query else None
                local_max_points = int(query.get("max_points", [str(max_points)])[0])
                fig = build_all_in_one_plotly_figure(source, p, max_points=local_max_points, x0=x0, x1=x1, variables=variables)
                body = json.dumps(fig.to_plotly_json()).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving zoom-adaptive plot at http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


app = typer.Typer(help="Simulate neuromodulated STDP with optional HDF5 and Parquet output.")


@app.command()
def main(
    hdf5: Annotated[str | None, typer.Option(help="Write recorded state to an HDF5 file.")] = None,
    parquet: Annotated[str | None, typer.Option(help="Write recorded state to a streamed Parquet file.")] = "sim.parquet",
    plot_backend: Annotated[str, typer.Option(help="Plot backend to use.")] = "server",
    plot_html: Annotated[str, typer.Option(help="Output HTML file for plotly backend.")] = "simulation.html",
    chunk_rows: Annotated[int, typer.Option(help="Row chunk size for streaming output.")] = 100_000,
    max_plot_points: Annotated[int, typer.Option(help="Max data points for plotting.")] = 40_000,
    host: Annotated[str, typer.Option(help="Host for server backend.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port for server backend.")] = 8050,
    method: Annotated[str, typer.Option(help="Integrator for smooth dynamics.")] = "euler",
    rate_mode: Annotated[str, typer.Option(help="Firing-rate mode: 'exp' (exponential trace) or 'window' (spike count).")] = "exp",
    rate_window: Annotated[float, typer.Option(help="Window duration in seconds for 'window' rate mode.")] = 0.5,
    reuse_existing: Annotated[bool, typer.Option(help="Reuse an existing HDF5/Parquet file instead of rerunning the simulation.")] = False,
    variables: Annotated[list[str] | None, typer.Option(help="Variables to plot (e.g. --variables E --variables w). Defaults to all.")] = None,
):
    if plot_backend not in {"matplotlib", "plotly", "server"}:
        raise typer.BadParameter(f"plot-backend must be one of: matplotlib, plotly, server (got {plot_backend!r})")
    if method not in {"euler", "rk4"}:
        raise typer.BadParameter(f"method must be one of: euler, rk4 (got {method!r})")
    if rate_mode not in {"exp", "window"}:
        raise typer.BadParameter(f"rate-mode must be one of: exp, window (got {rate_mode!r})")
    if variables is not None:
        bad = [v for v in variables if v not in ALL_PLOT_VARIABLES]
        if bad:
            raise typer.BadParameter(f"Unknown variables: {bad}. Choose from: {ALL_PLOT_VARIABLES}")

    # params = Params(T=2000, method="rk4")
    params = Params(
        T=10,
        method="rk4",
        rate_mode=rate_mode,
        rate_window=rate_window,
        V0=-62.39967779660166,
        I_s0=4.5401991625251883e-05,
        x_pre0=0.08942548983512577,
        y_post0=0.008716035584680641,
        E0=0.0030283801939102665,
        r_pre0=9.508331944774904,
        r_post0=4.562177684144224,
        R_bar0=-0.08942311829959347,
        w0=1.8925826247554693,
    )

    if plot_backend == "server" and not (parquet or hdf5):
        raise typer.BadParameter("The server backend requires --parquet or --hdf5 so zoom requests can be resampled from disk.")

    existing_server_source = None
    if plot_backend == "server" and reuse_existing:
        for candidate in (parquet, hdf5):
            if candidate and Path(candidate).exists():
                existing_server_source = candidate
                break

    rec = None
    if existing_server_source is None:
        rec = simulate(
            params,
            hdf5_path=hdf5,
            parquet_path=parquet,
            chunk_rows=chunk_rows,
        )

    if plot_backend == "matplotlib":
        if isinstance(rec, dict) and "t" in rec:
            plot_all_in_one_figure_matplotlib(rec, params)
        else:
            raise typer.BadParameter("Matplotlib plotting requires in-memory records. Use Plotly for HDF5/Parquet-backed runs.")
    elif plot_backend == "server":
        plot_source = existing_server_source or parquet or hdf5
        serve_zoom_adaptive_plot(
            plot_source,
            params,
            host=host,
            port=port,
            max_points=max_plot_points,
            variables=variables,
        )
    else:
        plot_source = parquet or hdf5 or rec
        output_html = plot_all_in_one_plotly(
            plot_source,
            params,
            output_html=plot_html,
            max_points=max_plot_points,
            variables=variables,
        )
        print(f"Wrote interactive plot to {output_html}")


if __name__ == "__main__":
    app()