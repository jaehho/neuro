"""Parameters and state-vector layout.

The simulation state is a flat numpy vector laid out as

    [V, y_post, r_post, R_bar,  (I_s, x_pre, E, w) × n_pre]

Use the index helpers below (``I_s_idx`` etc.) and ``n_state``; never
hard-code offsets.
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Recorded series / spike key generation ────────────────────────

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


# ── State vector layout: 4 shared + n_pre × 4 per-synapse ────────
# Shared:        [V, y_post, r_post, R_bar]
# Per synapse i: [I_s, x_pre, E, w]
# (r_pre is constant per Params.r_pre; not a state variable.)

N_SHARED = 4
N_PER_SYN = 4

V_IDX = 0
Y_POST_IDX = 1
R_POST_IDX = 2
RBAR_IDX = 3


def _syn_base(i: int) -> int:
    return N_SHARED + i * N_PER_SYN


def I_s_idx(i: int) -> int: return _syn_base(i)
def X_pre_idx(i: int) -> int: return _syn_base(i) + 1
def E_idx(i: int) -> int: return _syn_base(i) + 2
def W_idx(i: int) -> int: return _syn_base(i) + 3


def n_state(n_pre: int) -> int:
    return N_SHARED + n_pre * N_PER_SYN


_PER_SYN_FIELDS = ("r_pre", "I_s0", "x_pre0", "E0", "w0")


@dataclass
class Params:
    # ── Simulation ──
    T: float = 60.0           # Total duration (s)
    dt: float = 1e-4          # Integration timestep (s); 0.1 ms
    seed: int = 1
    record_every: float = 1e-3

    # ── Network topology ──
    n_pre: int = 1

    # ── Pre-synaptic input ──
    r_pre: tuple[float, ...] = (20.0,)   # Pre firing rate (Hz), per synapse
    poisson: bool = False      # True = Poisson; False = deterministic periodic

    # ── LIF neuron (Dayan & Abbott 2001, Ch. 5) ──
    tau_m: float = 0.02       # Membrane time constant (s)
    E_L: float = -65.0        # Leak reversal (mV)
    V_reset: float = -70.0    # Post-spike reset (mV)
    theta: float = -50.0      # Spike threshold (mV)
    tau_ref: float = 0.003    # Absolute refractory period (s); caps r_post at 1/tau_ref
    V0: float = -65.0

    # ── Synaptic current ──
    tau_s: float = 0.005      # Synaptic decay (s)
    R_m: float = 50.0         # Membrane resistance (MΩ)
    I_s0: tuple[float, ...] = (0.0,)
    I_ext: float = 0.0        # DC bias to post (nA)

    # ── STDP traces (Bi & Poo 1998) ──
    tau_plus: float = 0.02    # Pre→post LTP window (s)
    tau_minus: float = 0.02   # Post→pre LTD window (s)
    x_pre0: tuple[float, ...] = (0.0,)
    y_post0: float = 0.0

    # ── Post-rate estimation ──
    tau_r_post: float = 0.5   # Exponential rate-trace decay for r_post (s)
    r_post0: float = 0.0
    rate_mode: str = "exp"    # "exp" (trace) | "window" (spike count)
    rate_window: float = 0.5  # (s), used when rate_mode == "window"

    # ── Eligibility (Frémaux & Gerstner 2016) ──
    tau_e: float = 0.5
    E0: tuple[float, ...] = (0.0,)

    # ── Reward baseline ──
    tau_Rbar: float = 5.0
    R_bar0: float = 0.0

    # ── Three-factor learning rule (Frémaux & Gerstner 2016) ──
    # M = R - R_bar  (covariance)  |  M = R  (gated)
    M_rule: str = "gated"
    # R = -(r_post - r_target)^2  (target_rate)  |  R = r_target - r_post  (target_rate_linear)
    R_rule: str = "target_rate_linear"
    r_target: float = 10.0

    # ── Synaptic weights ──
    w0: tuple[float, ...] = (2.0,)
    wmax: float = 10.0
    eta_plus: float = 1e-4
    eta_minus: float = 1e-4

    def __post_init__(self):
        # Normalize per-synapse fields: tuple-ify (for JSON-load lists),
        # broadcast length-1 to n_pre (so defaults work for any n_pre),
        # and validate length.
        for attr in _PER_SYN_FIELDS:
            val = tuple(getattr(self, attr))
            if len(val) == 1 and self.n_pre > 1:
                val = val * self.n_pre
            if len(val) != self.n_pre:
                raise ValueError(
                    f"{attr} has length {len(val)}, expected {self.n_pre} (n_pre={self.n_pre})"
                )
            object.__setattr__(self, attr, val)
