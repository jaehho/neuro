"""Parameters, state-vector layout, and dynamic key generation.

The simulation state is a flat numpy vector laid out as
``[V, y_post, r_post, R_bar, (I_s, x_pre, E, w) × n_pre]``.
Use the index helpers (``_I_s_idx`` etc.) and ``n_state`` rather than
hard-coding offsets.
"""
from __future__ import annotations

from dataclasses import dataclass


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


# ── State vector layout: 4 shared + n_pre × 4 per-synapse ────────
# Shared: [V, y_post, r_post, R_bar]
# Per synapse i: [I_s, x_pre, E, w]
# (r_pre is a constant set by Params.r_pre_rates, not a state variable.)

N_SHARED = 4
N_PER_SYN = 4

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
def _W_idx(i: int) -> int: return _syn_base(i) + 3


def n_state(n_pre: int) -> int:
    return N_SHARED + n_pre * N_PER_SYN


# Backward-compatible aliases for n_pre=2
I_S1_IDX = _I_s_idx(0)
X_PRE1_IDX = _X_pre_idx(0)
E1_IDX = _E_idx(0)
W1_IDX = _W_idx(0)
I_S2_IDX = _I_s_idx(1)
X_PRE2_IDX = _X_pre_idx(1)
E2_IDX = _E_idx(1)
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
    I_ext: float = 0.0        # Constant DC bias current to post-synaptic neuron (nA)

    # ── STDP traces (Bi & Poo 1998) ───────────────────────
    tau_plus: float = 0.02    # Pre→post LTP window (s); 20 ms
    tau_minus: float = 0.02   # Post→pre LTD window (s); 20 ms
    x_pre0: tuple[float, ...] | float = (0.0,)
    y_post0: float = 0.0

    # ── Firing-rate estimation (r_post only; r_pre is constant) ──
    tau_r: float = 0.5        # Exponential rate-trace decay (s)
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
    reward_signal: str = "target_rate"  # target_rate | target_rate_linear | biofeedback | contingent | constant

    # ── Target-rate parameters (reward_signal="target_rate") ──
    target_func: str = "fixed"         # fixed | linear | affine | quadratic | sqrt | log | sin | power
    target_func_params: str = ""       # JSON dict of coefficients, e.g. '{"a": 0.3, "b": 2.0}'
    r_target: float = 10.0             # Target rate-trace value for target_func="fixed"
    alpha: float = 0.5                 # Slope for target_func="linear": target = α · r_pre

    # ── Constant-reward diagnostic (reward_signal="constant") ──
    R_const: float = 0.0               # Fixed scalar reward; bypasses any r_post / target dependence

    # ── Reward delivery (biofeedback / contingent) ──────────
    reward_delay: float = 1.0          # Delay from event to reward delivery (s)
    reward_amount: float = 1.0         # Reward pulse magnitude
    reward_tau: float = 0.2            # Reward pulse decay time constant (s)
    coincidence_window: float = 0.02   # Target→post window for contingent reward (s); 20 ms

    # ── Rate estimation mode (applies to r_post only) ──────
    rate_mode: str = "exp"    # "exp" (trace) or "window" (spike count)
    rate_window: float = 0.5  # Window duration for "window" mode (s)

    # ── Synaptic weights (per-synapse tuples) ──────────────
    w0: tuple[float, ...] | float = (2.0,)
    wmax: float = 10.0        # Hard upper bound (soft bounds in STDP)
    eta_plus: float = 1e-4    # LTP eligibility step size
    eta_minus: float = 1e-4   # LTD eligibility step size

    def __post_init__(self):
        n = self.n_pre
        for attr in ("r_pre_rates", "I_s0", "x_pre0", "E0", "w0"):
            val = getattr(self, attr)
            if isinstance(val, (int, float)):
                object.__setattr__(self, attr, tuple(val for _ in range(n)))
            elif not isinstance(val, tuple):
                object.__setattr__(self, attr, tuple(val))
            current = getattr(self, attr)
            if len(current) != n:
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
    def E1_0(self) -> float: return self.E0[0]
    @property
    def E2_0(self) -> float: return self.E0[1]
