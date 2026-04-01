"""
Neuromodulated Spike-Timing-Dependent Plasticity (STDP) simulation.

N presynaptic neurons project onto one postsynaptic LIF neuron.
Each synapse i has its own eligibility trace E_i and weight w_i.
The three-factor learning rule is:  dw_i/dt = M(t) · E_i(t), where

  - E_i(t) is the per-synapse eligibility trace shaped by STDP correlations
  - M(t) is a global neuromodulatory signal set by ``neuromod_type``
    (Frémaux & Gerstner 2016, Eq. 14):
        M = R − R̄      covariance rule (RPE / dopaminergic)
        M = R            gated Hebbian learning
        M = S            surprise / novelty-modulated STDP
        M = const        non-modulated STDP
  - R(t) is a reward signal set by ``reward_signal``:
        target_rate      R = -(r_post - target)²  (self-supervisory demonstration)
        biofeedback      R = delayed pulse after every post-spike (Legenstein+ 2008)
        contingent       R = delayed pulse only when pre1 fires within a
                         coincidence window before a post-spike (Izhikevich 2007;
                         Frémaux & Gerstner 2016, Eq. 10 "gated-Hebbian")

The contingent reward demonstrates spatial credit assignment: both synapses
receive the same global M, but only synapse 1 (target) has high E when the
reward arrives — because reward is triggered by pre1→post coincidences.

References
----------
Frémaux & Gerstner (2016). Neuromodulated STDP, and theory of three-factor
    learning rules. Front. Neural Circuits 9:85.
Frémaux, Sprekeler & Gerstner (2010). Functional requirements for
    reward-modulated STDP. J. Neurosci. 30(40):13326–13337.
Izhikevich (2007). Solving the distal reward problem through linkage of
    STDP and dopamine signaling. Cereb. Cortex 17(10):2443–2452.
Legenstein, Pecevski & Maass (2008). A learning theory for reward-modulated
    STDP with application to biofeedback. PLoS Comput. Biol. 4:e1000180.
Bi & Poo (1998). Synaptic modifications in cultured hippocampal neurons.
    J. Neurosci. 18(24):10464–10472.
Dayan & Abbott (2001). Theoretical Neuroscience, MIT Press, Ch. 5.
"""
from __future__ import annotations

import json
from collections import deque

from tqdm import tqdm
from dataclasses import asdict, dataclass, field, fields
from typing import Annotated
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from plotly.subplots import make_subplots


# ── Dynamic key generation ────────────────────────────────────────

def series_keys(n_pre: int) -> list[str]:
    keys = ["t", "V"]
    keys += [f"w{i+1}" for i in range(n_pre)]
    keys += [f"I_s{i+1}" for i in range(n_pre)]
    keys += [f"x_pre{i+1}" for i in range(n_pre)]
    keys.append("y_post")
    keys += [f"E{i+1}" for i in range(n_pre)]
    keys += [f"r_pre{i+1}" for i in range(n_pre)]
    keys.append("r_post")
    keys += ["R", "R_bar", "M"]
    keys += [f"pre{i+1}_spike_bin" for i in range(n_pre)]
    keys += ["post_spike_bin", "is_refractory"]
    return keys


def spike_keys(n_pre: int) -> list[str]:
    return [f"pre{i+1}_spike_times" for i in range(n_pre)] + ["post_spike_times"]


SERIES_KEYS = series_keys(2)
SPIKE_KEYS = spike_keys(2)


# ── State vector layout: 4 shared + n_pre × 5 per-synapse ────────
# Shared: [V, y_post, r_post, R_bar]
# Per synapse i: [I_s, x_pre, E, r_pre, w]

N_SHARED = 4
N_PER_SYN = 5

V_IDX = 0
Y_POST_IDX = 1
R_POST_IDX = 2
RBAR_IDX = 3


def _syn_base(i: int) -> int:
    """Start index for synapse i (0-indexed) in the state vector."""
    return N_SHARED + i * N_PER_SYN


def _I_s_idx(i: int) -> int: return _syn_base(i)
def _X_pre_idx(i: int) -> int: return _syn_base(i) + 1
def _E_idx(i: int) -> int: return _syn_base(i) + 2
def _R_pre_idx(i: int) -> int: return _syn_base(i) + 3
def _W_idx(i: int) -> int: return _syn_base(i) + 4


def n_state(n_pre: int) -> int:
    return N_SHARED + n_pre * N_PER_SYN


# Backward-compatible aliases for n_pre=2
I_S1_IDX = _I_s_idx(0)
X_PRE1_IDX = _X_pre_idx(0)
E1_IDX = _E_idx(0)
R_PRE1_IDX = _R_pre_idx(0)
W1_IDX = _W_idx(0)
I_S2_IDX = _I_s_idx(1)
X_PRE2_IDX = _X_pre_idx(1)
E2_IDX = _E_idx(1)
R_PRE2_IDX = _R_pre_idx(1)
W2_IDX = _W_idx(1)
N_STATE = n_state(2)


@dataclass
class Params:
    # ── Simulation ──────────────────────────────────────────
    T: float = 20.0           # Total duration (s)
    dt: float = 1e-4          # Integration timestep (s); 0.1 ms
    seed: int = 1
    record_every: float = 1e-4
    method: str = "euler"     # Integration method: "euler" | "rk4"

    # ── Network topology ───────────────────────────────────
    n_pre: int = 1            # Number of pre-synaptic neurons

    # ── Pre-synaptic input (per-synapse tuples) ────────────
    r_pre_rates: tuple[float, ...] | float = (20.0,)
    poisson: bool = False     # Poisson spike trains (False = deterministic)

    # ── LIF neuron (Dayan & Abbott 2001, Ch. 5) ────────────
    tau_m: float = 0.02       # Membrane time constant (s)
    E_L: float = -65.0        # Leak reversal potential (mV)
    V_reset: float = -70.0    # Post-spike reset potential (mV)
    theta: float = -50.0      # Spike threshold (mV)
    tau_ref: float = 0.003    # Absolute refractory period (s); 3 ms
    V0: float = -65.0         # Initial membrane potential (mV)
    ref_remaining0: float = 0.0

    # ── Synaptic current ───────────────────────────────────
    tau_s: float = 0.005      # Synaptic decay constant (s); 5 ms
    R_m: float = 50.0         # Membrane input resistance (MΩ)
    I_s0: tuple[float, ...] | float = (0.0,)

    # ── STDP traces (Bi & Poo 1998) ───────────────────────
    tau_plus: float = 0.02    # Pre→post LTP window (s); 20 ms
    tau_minus: float = 0.02   # Post→pre LTD window (s); 20 ms
    x_pre0: tuple[float, ...] | float = (0.0,)
    y_post0: float = 0.0

    # ── Firing-rate estimation ─────────────────────────────
    tau_r: float = 0.5        # Exponential rate-trace decay (s)
    r_pre0: tuple[float, ...] | float = (0.0,)
    r_post0: float = 0.0

    # ── Eligibility trace (Frémaux & Gerstner 2016) ───────
    tau_e: float = 0.5        # Eligibility decay (s)
    E0: tuple[float, ...] | float = (0.0,)

    # ── Reward baseline ────────────────────────────────────
    tau_Rbar: float = 5.0     # Baseline tracking time constant (s)
    R_bar0: float = 0.0

    # ── Neuromodulator role (Frémaux & Gerstner 2016, Eq. 14) ──
    neuromod_type: str = "covariance"  # covariance | gated | surprise | constant

    # ── Reward signal ──────────────────────────────────────
    reward_signal: str = "target_rate"  # target_rate | biofeedback | contingent

    # ── Target-rate parameters (reward_signal="target_rate") ──
    target_func: str = "fixed"         # fixed | linear | affine | quadratic | sqrt | log | sin | power
    target_func_params: str = ""       # JSON dict of coefficients, e.g. '{"a": 0.3, "b": 2.0}'
    r_target: float = 10.0             # Target rate-trace value for target_func="fixed"
    alpha: float = 0.5                 # Slope for target_func="linear": target = α · r_pre

    # ── Reward delivery (biofeedback / contingent) ──────────
    reward_delay: float = 1.0          # Delay from event to reward delivery (s)
    reward_amount: float = 1.0         # Reward pulse magnitude
    reward_tau: float = 0.2            # Reward pulse decay time constant (s)
    coincidence_window: float = 0.02   # Target→post window for contingent reward (s); 20 ms

    # ── Rate estimation mode ───────────────────────────────
    rate_mode: str = "exp"    # "exp" (trace) or "window" (spike count)
    rate_window: float = 0.5  # Window duration for "window" mode (s)

    # ── Synaptic weights (per-synapse tuples) ──────────────
    w0: tuple[float, ...] | float = (2.0,)
    wmax: float = 10.0        # Hard upper bound (soft bounds in STDP)
    eta_plus: float = 1e-4    # LTP eligibility step size
    eta_minus: float = 1e-4   # LTD eligibility step size

    def __post_init__(self):
        n = self.n_pre
        # Broadcast scalars (or wrong-length tuples with uniform values) to length n_pre
        for attr in ("r_pre_rates", "I_s0", "x_pre0", "r_pre0", "E0", "w0"):
            val = getattr(self, attr)
            if isinstance(val, (int, float)):
                object.__setattr__(self, attr, tuple(val for _ in range(n)))
            elif not isinstance(val, tuple):
                object.__setattr__(self, attr, tuple(val))
            current = getattr(self, attr)
            if len(current) != n:
                # If all values are the same, broadcast; otherwise error
                if len(set(current)) <= 1:
                    fill = current[0] if current else 0.0
                    object.__setattr__(self, attr, tuple(fill for _ in range(n)))
                else:
                    raise ValueError(f"{attr} has length {len(current)}, expected {n} (n_pre={n})")

    # ── Backward-compatible properties for n_pre=2 callers ─
    @property
    def r_pre_rate_1(self) -> float: return self.r_pre_rates[0]
    @property
    def r_pre_rate_2(self) -> float: return self.r_pre_rates[1]
    @property
    def w1_0(self) -> float: return self.w0[0]
    @property
    def w2_0(self) -> float: return self.w0[1]
    @property
    def I_s1_0(self) -> float: return self.I_s0[0]
    @property
    def I_s2_0(self) -> float: return self.I_s0[1]
    @property
    def x_pre1_0(self) -> float: return self.x_pre0[0]
    @property
    def x_pre2_0(self) -> float: return self.x_pre0[1]
    @property
    def r_pre1_0(self) -> float: return self.r_pre0[0]
    @property
    def r_pre2_0(self) -> float: return self.r_pre0[1]
    @property
    def E1_0(self) -> float: return self.E0[0]
    @property
    def E2_0(self) -> float: return self.E0[1]


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
        if pa is None or pq is None:
            raise ImportError("pyarrow is required for Parquet export. Install it with `uv add pyarrow`.")

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
        # pre{i}_spike_times -> pre{i}, post_spike_times -> post
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


# ── General target function helpers ──────────────────────────────────

_TARGET_PARAMS_CACHE: dict[str, dict] = {}


def _parse_target_params(s: str) -> dict:
    """Parse JSON coefficient string, caching the result."""
    if s not in _TARGET_PARAMS_CACHE:
        _TARGET_PARAMS_CACHE[s] = json.loads(s) if s else {}
    return _TARGET_PARAMS_CACHE[s]


def _compute_target_r_post(p: Params, r_pre: float) -> float:
    """Compute target r_post from r_pre using the configured target function.

    Each function type has sensible defaults so that ``target_func_params``
    can be left empty for a quick test.
    """
    c = _parse_target_params(p.target_func_params)
    func = p.target_func

    if func == "fixed":
        return c.get("target", p.r_target)
    elif func == "linear":
        a = c.get("a", p.alpha)
        return a * r_pre
    elif func == "affine":
        a = c.get("a", 0.3)
        b = c.get("b", 2.0)
        return a * r_pre + b
    elif func == "quadratic":
        a = c.get("a", -0.03)
        b = c.get("b", 0.8)
        k = c.get("c", 0.5)
        return a * r_pre ** 2 + b * r_pre + k
    elif func == "sqrt":
        a = c.get("a", 1.5)
        b = c.get("b", 0.5)
        return a * np.sqrt(max(r_pre, 0.0)) + b
    elif func == "log":
        a = c.get("a", 1.5)
        b = c.get("b", 1.5)
        return a * float(np.log(max(r_pre, 1e-6))) + b
    elif func == "sin":
        a = c.get("a", 2.0)
        b = c.get("b", 5.0)
        freq = c.get("c", 0.3)
        return a * np.sin(freq * r_pre) + b
    elif func == "power":
        a = c.get("a", 1.0)
        exp = c.get("c", 0.6)
        b = c.get("b", 0.5)
        return a * max(r_pre, 0.0) ** exp + b
    else:
        raise ValueError(f"Unknown target_func: {func!r}")


def _compute_reward(
    p: Params,
    r_pre: float,
    r_post: float,
    reward_pulse: float = 0.0,
) -> float:
    """Compute the reward signal R.

    For ``reward_signal="target_rate"``, R is a self-supervisory squared-error
    signal penalising deviation from a target firing rate.  This is useful for
    demonstrating the ODE machinery but is *not* from the three-factor STDP
    literature (where R is always an external / task-based signal).

    For ``reward_signal="biofeedback"``, R is an externally delivered pulse
    triggered by post-synaptic spikes with a configurable delay, matching
    the paradigm of Izhikevich (2007) and Legenstein et al. (2008).
    """
    sig = p.reward_signal

    if sig == "target_rate":
        target = max(_compute_target_r_post(p, r_pre), 0.0)
        return -(r_post - target) ** 2
    elif sig in ("biofeedback", "contingent"):
        return reward_pulse
    else:
        raise ValueError(f"Unknown reward_signal: {p.reward_signal!r}")


def _compute_modulation(
    p: Params,
    R: float,
    R_bar: float,
    r_post: float,
) -> tuple[float, float]:
    """Compute neuromodulator M and R̄ tracking target.

    Returns ``(M, rbar_target)`` where *rbar_target* drives dR̄/dt.

    The neuromodulator's role follows Frémaux & Gerstner (2016), Eq. 14::

        covariance   M = R − R̄       RPE / dopaminergic covariance rule
        gated        M = R             reward directly gates plasticity
        surprise     M = (r_post−R̄)²  novelty signal; R̄ tracks r_post
        constant     M = 1             non-modulated STDP baseline
    """
    mode = p.neuromod_type

    if mode == "covariance":
        return R - R_bar, R
    elif mode == "gated":
        return R, R
    elif mode == "surprise":
        # R̄ tracks expected activity (r_post), not reward.
        # Surprise = squared prediction error on firing rate.
        S = (r_post - R_bar) ** 2
        return S, r_post
    elif mode == "constant":
        # Non-modulated STDP: M = 1, R̄ is unused.
        return 1.0, R_bar
    else:
        raise ValueError(f"Unknown neuromod_type: {p.neuromod_type!r}")


def _pack_state(p: Params) -> np.ndarray:
    """Build initial state vector from Params."""
    n = p.n_pre
    y = np.zeros(n_state(n), dtype=np.float64)
    y[V_IDX] = p.V0
    y[Y_POST_IDX] = p.y_post0
    y[R_POST_IDX] = p.r_post0
    y[RBAR_IDX] = p.R_bar0
    for i in range(n):
        y[_I_s_idx(i)] = p.I_s0[i]
        y[_X_pre_idx(i)] = p.x_pre0[i]
        y[_E_idx(i)] = p.E0[i]
        y[_R_pre_idx(i)] = p.r_pre0[i]
        y[_W_idx(i)] = p.w0[i]
    return y


def _smooth_rhs(
    y: np.ndarray,
    p: Params,
    *,
    voltage_active: bool,
    rate_pre: float | None = None,
    rate_post: float | None = None,
    reward_pulse: float = 0.0,
) -> np.ndarray:
    """Continuous-time RHS of the ODE system (between spike events).

    Spike-triggered jumps (I_s += 1, x_pre += 1, eligibility updates)
    are applied in simulate(), not here.  This function handles only the
    exponential-decay and coupling terms that are smooth in t.

    The membrane voltage is driven by the sum of all weighted synaptic
    currents.  The same neuromodulator M gates all eligibility traces
    independently: dw_i/dt = M · E_i.
    """
    n = p.n_pre
    rhs = np.zeros_like(y)

    # ── Shared postsynaptic state ──
    V = float(y[V_IDX])
    yp = float(y[Y_POST_IDX])
    r_post = float(y[R_POST_IDX])
    R_bar = float(y[RBAR_IDX])

    # ── Reward and modulation (global, uses target synapse rate) ──
    rr_pre = rate_pre if rate_pre is not None else float(y[_R_pre_idx(0)])
    rr_post = rate_post if rate_post is not None else r_post
    R = _compute_reward(p, rr_pre, rr_post, reward_pulse=reward_pulse)
    M, rbar_target = _compute_modulation(p, R, R_bar, rr_post)

    # LIF membrane: τ_m · dV/dt = -(V - E_L) + R_m · Σ w_i · I_s_i
    if voltage_active:
        I_total = 0.0
        for i in range(n):
            wi = float(np.clip(y[_W_idx(i)], 0.0, p.wmax))
            I_total += wi * float(y[_I_s_idx(i)])
        rhs[V_IDX] = (-(V - p.E_L) + p.R_m * I_total) / p.tau_m

    # ── Shared traces ──
    rhs[Y_POST_IDX] = -yp / p.tau_minus
    rhs[R_POST_IDX] = -r_post / p.tau_r
    rhs[RBAR_IDX] = (-R_bar + rbar_target) / p.tau_Rbar

    # ── Per-synapse ──
    for i in range(n):
        I_s = float(y[_I_s_idx(i)])
        x = float(y[_X_pre_idx(i)])
        E = float(y[_E_idx(i)])
        r = float(y[_R_pre_idx(i)])
        rhs[_I_s_idx(i)] = -I_s / p.tau_s
        rhs[_X_pre_idx(i)] = -x / p.tau_plus
        rhs[_E_idx(i)] = -E / p.tau_e
        rhs[_R_pre_idx(i)] = -r / p.tau_r
        rhs[_W_idx(i)] = M * E

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
    reward_pulse: float = 0.0,
) -> np.ndarray:
    """Advance state vector by dt using Euler or classical RK4.

    The system is autonomous (no explicit t in the smooth RHS), so time
    is not forwarded.  Weight is projected onto [0, wmax] after each step.
    """
    kw = dict(voltage_active=voltage_active, rate_pre=rate_pre, rate_post=rate_post, reward_pulse=reward_pulse)

    def rhs(state: np.ndarray) -> np.ndarray:
        return _smooth_rhs(state, p, **kw)  # type: ignore[arg-type]

    if dt <= 0.0:
        out = y.copy()
    elif method == "euler":
        out = y + dt * rhs(y)
    elif method == "rk4":
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        out = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    else:
        raise ValueError(f"Unknown integration method: {method!r}")

    for i in range(p.n_pre):
        wi = _W_idx(i)
        out[wi] = float(np.clip(out[wi], 0.0, p.wmax))
    if not voltage_active:
        out[V_IDX] = p.V_reset
    return out


def _crossing_fraction(v0: float, v1: float, threshold: float) -> float | None:
    """Linear interpolation for upward threshold crossing.

    Returns fraction f ∈ [0,1] such that V(t + f·dt) ≈ threshold.
    None if no crossing.  Spike-time error is O(dt²).
    """
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
    pre_spikes: list[int],
    post_spike: int,
    is_refractory: int,
    rate_pre: float | None = None,
    rate_post: float | None = None,
    reward_pulse: float = 0.0,
) -> dict[str, float]:
    n = p.n_pre
    rr_pre = rate_pre if rate_pre is not None else float(y[_R_pre_idx(0)])
    rr_post = rate_post if rate_post is not None else float(y[R_POST_IDX])
    R = _compute_reward(p, rr_pre, rr_post, reward_pulse=reward_pulse)
    M, _ = _compute_modulation(p, R, float(y[RBAR_IDX]), rr_post)
    row: dict[str, float] = {"t": t, "V": float(y[V_IDX])}
    for i in range(n):
        row[f"w{i+1}"] = float(y[_W_idx(i)])
    for i in range(n):
        row[f"I_s{i+1}"] = float(y[_I_s_idx(i)])
    for i in range(n):
        row[f"x_pre{i+1}"] = float(y[_X_pre_idx(i)])
    row["y_post"] = float(y[Y_POST_IDX])
    for i in range(n):
        row[f"E{i+1}"] = float(y[_E_idx(i)])
    for i in range(n):
        row[f"r_pre{i+1}"] = float(y[_R_pre_idx(i)])
    row["r_post"] = rr_post
    row["R"] = float(R)
    row["R_bar"] = float(y[RBAR_IDX])
    row["M"] = float(M)
    for i in range(n):
        row[f"pre{i+1}_spike_bin"] = int(pre_spikes[i])
    row["post_spike_bin"] = int(post_spike)
    row["is_refractory"] = int(is_refractory)
    return row


def simulate(
    p: Params,
    hdf5_path: str | None = None,
    parquet_path: str | None = None,
    chunk_rows: int = 100_000,
):
    """Run the N-pre → 1-post neuromodulated STDP simulation.

    Hybrid event-driven / continuous scheme:
    - Smooth dynamics integrated by Euler or RK4 (_advance_state)
    - Spike events (pre/post) apply instantaneous jumps to traces
    - RK4 path uses threshold-crossing interpolation to split timestep

    Synapse 0 is the "target" (reward-paired) synapse used for contingent
    reward coincidence detection.
    """
    method = p.method.lower()
    if method not in {"euler", "rk4"}:
        raise ValueError("Params.method must be either 'euler' or 'rk4'.")
    use_window = p.rate_mode == "window"
    if p.rate_mode not in {"exp", "window"}:
        raise ValueError("Params.rate_mode must be either 'exp' or 'window'.")

    n_pre = p.n_pre
    n_steps = int(p.T / p.dt)
    rec_step = max(1, int(p.record_every / p.dt))

    # ── Spike generation ──────────────────────────────────────
    rng = np.random.default_rng(p.seed)
    probs = [r * p.dt for r in p.r_pre_rates]
    if not p.poisson:
        periods = [max(1, round(1.0 / (r * p.dt))) if r > 0 else 0
                   for r in p.r_pre_rates]

    y = _pack_state(p)
    ref_remaining = p.ref_remaining0

    pre_spike_bufs: list[deque[float]] = [deque() for _ in range(n_pre)]
    post_spike_buf: deque[float] = deque()
    win_r_pre: list[float] = [0.0] * n_pre
    win_r_post = 0.0

    # ── Reward delivery state ─────────────────────────────────
    d_reward = 0.0
    reward_schedule: deque[float] = deque()
    use_reward_pulse = p.reward_signal in ("biofeedback", "contingent")

    # ── Contingent: track recent target-synapse spike times ───
    recent_target_times: deque[float] = deque()

    recorder = _build_recorder(p, hdf5_path=hdf5_path, parquet_path=parquet_path, chunk_rows=chunk_rows)

    for step in tqdm(range(n_steps), desc="Simulating", unit="step", mininterval=0.5):
        t = step * p.dt

        # ── Deliver scheduled rewards and decay pulse ─────────
        if use_reward_pulse:
            while reward_schedule and reward_schedule[0] <= t:
                reward_schedule.popleft()
                d_reward += p.reward_amount
            d_reward += p.dt * (-d_reward / p.reward_tau)

        # ── Pre-synaptic spikes ───────────────────────────────
        pre_spikes = [0] * n_pre
        for i in range(n_pre):
            if p.poisson:
                pre_spikes[i] = 1 if rng.random() < probs[i] else 0
            else:
                pre_spikes[i] = 1 if (periods[i] > 0 and step % periods[i] == 0) else 0

        for i in range(n_pre):
            if pre_spikes[i]:
                recorder.append_spike(f"pre{i+1}_spike_times", t)
                y[_I_s_idx(i)] += 1.0
                y[_X_pre_idx(i)] += 1.0
                y[_R_pre_idx(i)] += 1.0
                y[_E_idx(i)] -= p.eta_minus * y[_W_idx(i)] * y[Y_POST_IDX]
                if use_window:
                    pre_spike_bufs[i].append(t)
                if p.reward_signal == "contingent" and i == 0:
                    recent_target_times.append(t)

        post_spike = 0
        is_refractory = 1 if ref_remaining > 0.0 else 0

        if use_window:
            cutoff = t - p.rate_window
            for i in range(n_pre):
                while pre_spike_bufs[i] and pre_spike_bufs[i][0] < cutoff:
                    pre_spike_bufs[i].popleft()
                win_r_pre[i] = len(pre_spike_bufs[i]) / p.rate_window
            while post_spike_buf and post_spike_buf[0] < cutoff:
                post_spike_buf.popleft()
            win_r_post = len(post_spike_buf) / p.rate_window

        rp = win_r_pre[0] if use_window else None
        ro = win_r_post if use_window else None

        # ── Prune old target times for contingent reward ──────
        if p.reward_signal == "contingent":
            cutoff_c = t - p.coincidence_window
            while recent_target_times and recent_target_times[0] < cutoff_c:
                recent_target_times.popleft()

        if method == "euler":
            V = float(y[V_IDX])
            yp = float(y[Y_POST_IDX])
            r_post = float(y[R_POST_IDX])
            R_bar = float(y[RBAR_IDX])

            # Unpack per-synapse into local arrays
            I_s = [float(y[_I_s_idx(i)]) for i in range(n_pre)]
            x = [float(y[_X_pre_idx(i)]) for i in range(n_pre)]
            E = [float(y[_E_idx(i)]) for i in range(n_pre)]
            r = [float(y[_R_pre_idx(i)]) for i in range(n_pre)]
            w = [float(y[_W_idx(i)]) for i in range(n_pre)]

            # Decay per-synapse variables
            for i in range(n_pre):
                I_s[i] += p.dt * (-I_s[i] / p.tau_s)
                x[i] += p.dt * (-x[i] / p.tau_plus)
                r[i] += p.dt * (-r[i] / p.tau_r)

            if ref_remaining <= 0.0:
                I_total = sum(w[i] * I_s[i] for i in range(n_pre))
                dV = (p.dt / p.tau_m) * (-(V - p.E_L) + p.R_m * I_total)
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
                yp += 1.0
                r_post += 1.0
                for i in range(n_pre):
                    E[i] += p.eta_plus * (p.wmax - w[i]) * x[i]
                if p.reward_signal == "biofeedback":
                    reward_schedule.append(t + p.reward_delay)
                elif p.reward_signal == "contingent" and recent_target_times:
                    reward_schedule.append(t + p.reward_delay)
                if use_window:
                    post_spike_buf.append(t)
                    win_r_post = len(post_spike_buf) / p.rate_window
                    rp = win_r_pre[0]
                    ro = win_r_post

            yp += p.dt * (-yp / p.tau_minus)
            r_post += p.dt * (-r_post / p.tau_r)
            for i in range(n_pre):
                E[i] += p.dt * (-E[i] / p.tau_e)

            rew_pre: float = rp if use_window else r[0]  # type: ignore[assignment]
            rew_post: float = ro if use_window else r_post  # type: ignore[assignment]
            R = _compute_reward(p, rew_pre, rew_post, reward_pulse=d_reward)
            _, rbar_target = _compute_modulation(p, R, R_bar, rew_post)
            R_bar += (p.dt / p.tau_Rbar) * (-R_bar + rbar_target)
            R = _compute_reward(p, rew_pre, rew_post, reward_pulse=d_reward)
            M, _ = _compute_modulation(p, R, R_bar, rew_post)
            for i in range(n_pre):
                w[i] += p.dt * M * E[i]
                w[i] = min(p.wmax, max(0.0, w[i]))

            # Pack back into state vector
            y[V_IDX] = V
            y[Y_POST_IDX] = yp
            y[R_POST_IDX] = r_post
            y[RBAR_IDX] = R_bar
            for i in range(n_pre):
                y[_I_s_idx(i)] = I_s[i]
                y[_X_pre_idx(i)] = x[i]
                y[_E_idx(i)] = E[i]
                y[_R_pre_idx(i)] = r[i]
                y[_W_idx(i)] = w[i]

        else:
            # RK4 with threshold-crossing detection
            if ref_remaining <= 0.0:
                v0 = float(y[V_IDX])
                y_trial = _advance_state(y, p.dt, p, method="rk4", voltage_active=True, rate_pre=rp, rate_post=ro, reward_pulse=d_reward)
                frac = _crossing_fraction(v0, float(y_trial[V_IDX]), p.theta)

                if frac is None:
                    y = y_trial
                else:
                    dt1 = frac * p.dt
                    dt2 = p.dt - dt1

                    y_mid = _advance_state(y, dt1, p, method="rk4", voltage_active=True, rate_pre=rp, rate_post=ro, reward_pulse=d_reward)
                    spike_t = t + dt1
                    recorder.append_spike("post_spike_times", spike_t)
                    post_spike = 1

                    y_mid[V_IDX] = p.V_reset
                    y_mid[Y_POST_IDX] += 1.0
                    y_mid[R_POST_IDX] += 1.0
                    for i in range(n_pre):
                        y_mid[_E_idx(i)] += p.eta_plus * (p.wmax - y_mid[_W_idx(i)]) * y_mid[_X_pre_idx(i)]

                    if p.reward_signal == "biofeedback":
                        reward_schedule.append(spike_t + p.reward_delay)
                    elif p.reward_signal == "contingent" and recent_target_times:
                        reward_schedule.append(spike_t + p.reward_delay)

                    if use_window:
                        post_spike_buf.append(spike_t)
                        ro = len(post_spike_buf) / p.rate_window

                    y = _advance_state(y_mid, dt2, p, method="rk4", voltage_active=False, rate_pre=rp, rate_post=ro, reward_pulse=d_reward)
                    y[V_IDX] = p.V_reset
                    ref_remaining = max(0.0, p.tau_ref - dt2)
            else:
                y = _advance_state(y, p.dt, p, method="rk4", voltage_active=False, rate_pre=rp, rate_post=ro, reward_pulse=d_reward)
                y[V_IDX] = p.V_reset
                ref_remaining = max(0.0, ref_remaining - p.dt)

        if step % rec_step == 0:
            recorder.append(
                _row_from_state(
                    t,
                    y,
                    p,
                    pre_spikes=pre_spikes,
                    post_spike=post_spike,
                    is_refractory=is_refractory,
                    rate_pre=rp,
                    rate_post=ro,
                    reward_pulse=d_reward,
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
        arrays = {key: np.asarray(handle[key][start:stop:stride]) for key in selected}  # type: ignore[index]
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
        spikes = {key: np.asarray(handle[key]) for key in SPIKE_KEYS}  # type: ignore[index]
    return _filter_spike_times(spikes, x0=x0, x1=x1)


def read_spike_times_parquet(path: str | Path, x0: float | None = None, x1: float | None = None) -> dict[str, np.ndarray]:
    if pl is None:
        raise ImportError("polars is required to read Parquet spike times.")

    spike_path = spike_parquet_path(path)
    if not spike_path.exists():
        return {k: np.array([]) for k in SPIKE_KEYS}

    spikes = pl.read_parquet(str(spike_path))
    result: dict[str, np.ndarray] = {}
    for spike_type in spikes.get_column("spike_type").unique().to_list():
        times = spikes.filter(pl.col("spike_type") == spike_type).get_column("t").to_numpy()
        result[f"{spike_type}_spike_times"] = times
    # Ensure post is always present
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
            total = int(handle.attrs["rows_written"])  # type: ignore[arg-type]
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
        include_plotlyjs="directory",  # type: ignore[arg-type]
        auto_open=False,
        validate=True,
    )
    return output_html


def plot_all_in_one_figure_matplotlib(rec, p: Params):
    n = p.n_pre
    t = rec["t"]
    n_panels = 7 + (1 if n > 0 else 0) + (1 if n > 0 else 0) + 1  # spikes, V, E, r_pre, r_post, R, R_bar, M, w, xlabel
    _, axs = plt.subplots(n_panels, 1, figsize=(19, 2 * n_panels), sharex=True)

    # Spike raster
    spike_data = [rec.get(f"pre{i+1}_spike_times", np.array([])) for i in range(n)] + [rec["post_spike_times"]]
    labels = [f"pre{i+1}" for i in range(n)] + ["post"]
    axs[0].eventplot(spike_data, lineoffsets=list(range(len(labels) - 1, -1, -1)), linelengths=0.8)
    axs[0].set_yticks(list(range(len(labels))))
    axs[0].set_yticklabels(labels[::-1])
    axs[0].set_title("Spike times")

    dot = dict(marker=".", markersize=1, linestyle="none")

    axs[1].plot(t, rec["V"], **dot)
    axs[1].axhline(p.theta, linestyle="--", label="theta")
    axs[1].axhline(p.V_reset, linestyle=":", label="V_reset")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Membrane potential V(t)")

    for i in range(n):
        label = f"E{i+1}" + (" (target)" if i == 0 and n > 1 else "")
        axs[2].plot(t, rec[f"E{i+1}"], label=label, **dot)
    axs[2].legend(loc="upper right")
    axs[2].set_title("Eligibility traces")

    for i in range(n):
        axs[3].plot(t, rec[f"r_pre{i+1}"], label=f"r_pre{i+1}", **dot)
    axs[3].legend(loc="upper right")
    axs[3].set_title("Pre-synaptic firing rates")

    axs[4].plot(t, rec["r_post"], **dot)
    axs[4].set_title("Post-synaptic firing rate r_post(t)")

    axs[5].plot(t, rec["R"], **dot)
    axs[5].set_title("Reward R(t)")

    axs[6].plot(t, rec["R_bar"], **dot)
    axs[6].set_title("Reward baseline R_bar(t)")

    axs[7].plot(t, rec["M"], **dot)
    axs[7].set_title("Modulation M(t)")

    for i in range(n):
        label = f"w{i+1}" + (" (target)" if i == 0 and n > 1 else "")
        axs[8].plot(t, rec[f"w{i+1}"], label=label, **dot)
    axs[8].legend(loc="upper right")
    axs[8].set_title("Synaptic weights")

    axs[-1].set_xlabel("time (s)")

    plt.tight_layout()
    plt.savefig("simulation.png")
    plt.show()


def all_plot_variables(n_pre: int) -> list[str]:
    v = ["V"]
    v += [f"I_s{i+1}" for i in range(n_pre)]
    v += [f"x_pre{i+1}" for i in range(n_pre)]
    v.append("y_post")
    v += [f"E{i+1}" for i in range(n_pre)]
    v += [f"r_pre{i+1}" for i in range(n_pre)]
    v.append("r_post")
    v += ["R", "R_bar", "M"]
    v += [f"w{i+1}" for i in range(n_pre)]
    return v


def variable_titles(n_pre: int) -> dict[str, str]:
    titles: dict[str, str] = {"V": "Membrane potential V"}
    for i in range(n_pre):
        label = "target" if i == 0 else "distractor" if n_pre > 1 else ""
        suffix = f" ({label})" if label else ""
        titles[f"I_s{i+1}"] = f"Synaptic current I_s{i+1}{suffix}"
        titles[f"x_pre{i+1}"] = f"STDP pre-trace x_pre{i+1}"
        titles[f"E{i+1}"] = f"Eligibility trace E{i+1}{suffix}"
        titles[f"r_pre{i+1}"] = f"Pre{i+1} firing rate r_pre{i+1}"
        titles[f"w{i+1}"] = f"Weight w{i+1}{suffix}"
    titles["y_post"] = "STDP post-trace y_post"
    titles["r_post"] = "Post-synaptic firing rate r_post"
    titles["R"] = "Reward R"
    titles["R_bar"] = "Reward baseline R_bar"
    titles["M"] = "Modulation M"
    return titles


ALL_PLOT_VARIABLES = all_plot_variables(2)
VARIABLE_TITLES = variable_titles(2)


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

    n_pre = p.n_pre
    apv = all_plot_variables(n_pre)
    vtitles = variable_titles(n_pre)

    plot_vars = variables if variables is not None else apv
    show_spikes = variables is None
    n_rows = len(plot_vars) + (1 if show_spikes else 0)

    needed_columns = set(plot_vars)
    if "r_post" in needed_columns:
        needed_columns.add("r_pre1")
    columns = ["t"] + [v for v in apv if v in needed_columns]

    frame, spikes = load_time_series_frame(rec_or_path, columns=columns, max_points=max_points, x0=x0, x1=x1)
    arrays = {col: _plotly_values(frame[col].to_numpy()) for col in columns}
    t = arrays["t"]

    titles = []
    if show_spikes:
        titles.append("Spike times")
    titles.extend(vtitles.get(v, v) for v in plot_vars)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=titles,
    )

    row = 1
    if show_spikes:
        # Build spike raster entries dynamically
        spike_entries = [(f"pre{i+1}", f"pre{i+1}_spike_times", float(n_pre - i)) for i in range(n_pre)]
        spike_entries.append(("post", "post_spike_times", 0.0))
        for label, key, y_base in spike_entries:
            times = spikes.get(key, np.array([]))
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
        fig.add_trace(go.Scattergl(x=t, y=arrays[var], name=var, mode="markers", marker={"size": 2}), row=row, col=1)
        if var == "V":
            fig.add_hline(y=p.theta, line_dash="dash", row=row, col=1)  # type: ignore[arg-type]
            fig.add_hline(y=p.V_reset, line_dash="dot", row=row, col=1)  # type: ignore[arg-type]
        elif var == "r_post" and "r_pre1" in needed_columns:
            fig.add_trace(go.Scattergl(x=t, y=_plotly_values(p.alpha * np.asarray(frame["r_pre1"].to_numpy())), name="target"), row=row, col=1)
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

        def log_message(self, format, *args):  # noqa: A002
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
    parquet: Annotated[str | None, typer.Option(help="Write recorded state to a Parquet file (overrides cache naming).")] = None,
    plot_backend: Annotated[str, typer.Option(help="Plot backend to use.")] = "server",
    plot_html: Annotated[str, typer.Option(help="Output HTML file for plotly backend.")] = "simulation.html",
    chunk_rows: Annotated[int, typer.Option(help="Row chunk size for streaming output.")] = 100_000,
    max_plot_points: Annotated[int, typer.Option(help="Max data points for plotting.")] = 40_000,
    host: Annotated[str, typer.Option(help="Host for server backend.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port for server backend.")] = 8050,
    method: Annotated[str, typer.Option(help="Integrator for smooth dynamics.")] = "rk4",
    rate_mode: Annotated[str, typer.Option(help="Firing-rate mode: 'exp' (exponential trace) or 'window' (spike count).")] = "exp",
    rate_window: Annotated[float, typer.Option(help="Window duration in seconds for 'window' rate mode.")] = 0.5,
    neuromod_type: Annotated[str, typer.Option(help="Neuromodulator role (Frémaux Eq.14): covariance, gated, surprise, constant.")] = "covariance",
    reward_signal: Annotated[str, typer.Option(help="Reward signal: target_rate, biofeedback, contingent.")] = "target_rate",
    n_pre: Annotated[int, typer.Option(help="Number of pre-synaptic neurons.")] = 1,
    variables: Annotated[list[str] | None, typer.Option(help="Variables to plot (e.g. --variables E1 --variables w1). Defaults to all.")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Force rerun, ignoring cached results.")] = False,
    cache_dir: Annotated[str, typer.Option(help="Directory for simulation cache.")] = "output",
):
    if plot_backend not in {"matplotlib", "plotly", "server"}:
        raise typer.BadParameter(f"plot-backend must be one of: matplotlib, plotly, server (got {plot_backend!r})")
    if method not in {"euler", "rk4"}:
        raise typer.BadParameter(f"method must be one of: euler, rk4 (got {method!r})")
    if rate_mode not in {"exp", "window"}:
        raise typer.BadParameter(f"rate-mode must be one of: exp, window (got {rate_mode!r})")
    if neuromod_type not in {"covariance", "gated", "surprise", "constant"}:
        raise typer.BadParameter(f"neuromod-type must be one of: covariance, gated, surprise, constant (got {neuromod_type!r})")
    if reward_signal not in {"target_rate", "biofeedback", "contingent"}:
        raise typer.BadParameter(f"reward-signal must be one of: target_rate, biofeedback, contingent (got {reward_signal!r})")
    if n_pre < 1:
        raise typer.BadParameter(f"n-pre must be >= 1 (got {n_pre})")

    apv = all_plot_variables(n_pre)
    if variables is not None:
        bad = [v for v in variables if v not in apv]
        if bad:
            raise typer.BadParameter(f"Unknown variables: {bad}. Choose from: {apv}")

    params = Params(
        T=100,
        n_pre=n_pre,
        r_pre_rates=20.0,
        w0=2.0,
        method=method,
        rate_mode=rate_mode,
        rate_window=rate_window,
        neuromod_type=neuromod_type,
        reward_signal=reward_signal,
    )

    # ── Cache lookup (before preview so user sees status) ─────
    from neuro.cache import params_hash, lookup_run, cached_simulate
    use_cache = not parquet and not hdf5
    cache_hit = None
    if use_cache and not no_cache:
        hash_hex = params_hash(params)
        cache_hit = lookup_run(Path(cache_dir) / "runs.db", hash_hex)
    elif use_cache:
        hash_hex = params_hash(params)

    n_steps = int(params.T / params.dt)
    rates_str = ", ".join(f"{r} Hz" for r in params.r_pre_rates)
    weights_str = ", ".join(str(w) for w in params.w0)
    typer.echo("\n── Simulation Parameters ──")
    typer.echo(f"  Duration:       {params.T} s  ({n_steps:,} steps, dt={params.dt})")
    typer.echo(f"  Integrator:     {method}")
    typer.echo(f"  Neuromod type:  {neuromod_type}")
    typer.echo(f"  Reward signal:  {reward_signal}")
    typer.echo(f"  Rate mode:      {rate_mode}" + (f" (window={rate_window} s)" if rate_mode == "window" else ""))
    typer.echo(f"  Pre neurons:    {n_pre}")
    typer.echo(f"  Pre rates:      [{rates_str}]  ({'Poisson' if params.poisson else 'deterministic'})")
    typer.echo(f"  Weights:        [{weights_str}]  (wmax={params.wmax})")
    if cache_hit:
        typer.echo(f"  Cache:          HIT — reusing run from {cache_hit['created_at']}")
        typer.echo(f"  Output:         {cache_hit['parquet_path']}")
    elif use_cache and no_cache:
        short = hash_hex[:12]
        typer.echo(f"  Cache:          FORCED RERUN — will save to {cache_dir}/{short}.parquet")
    elif use_cache:
        short = hash_hex[:12]
        typer.echo(f"  Cache:          MISS — will save to {cache_dir}/{short}.parquet")
    else:
        typer.echo(f"  Output:         {parquet or hdf5 or 'in-memory'}")
    typer.echo(f"  Plot backend:   {plot_backend}")
    typer.echo("")
    if not typer.confirm("Proceed?", default=True):
        raise typer.Abort()

    if not use_cache and plot_backend == "server" and not (parquet or hdf5):
        raise typer.BadParameter("The server backend requires --parquet or --hdf5 so zoom requests can be resampled from disk.")

    rec = None
    if use_cache:
        rec = cached_simulate(params, cache_dir=Path(cache_dir), chunk_rows=chunk_rows, force=no_cache)
        parquet = rec.get("parquet_path", parquet)
    else:
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
        plot_source = parquet or hdf5
        assert plot_source is not None, "No plot source available for server mode"
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
