"""Parameters and state-vector layout.

The simulation state is a flat numpy vector laid out as

    [V, y_post, r_post, R_bar,  (I_s, x_pre, E, w) × n_pre]

Use the index helpers below (``_I_s_idx`` etc.) and ``n_state``; never
hard-code offsets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


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


def _I_s_idx(i: int) -> int: return _syn_base(i)
def _X_pre_idx(i: int) -> int: return _syn_base(i) + 1
def _E_idx(i: int) -> int: return _syn_base(i) + 2
def _W_idx(i: int) -> int: return _syn_base(i) + 3


def n_state(n_pre: int) -> int:
    return N_SHARED + n_pre * N_PER_SYN


@dataclass
class Params:
    # ── Simulation ──
    T: float = 20.0           # Total duration (s)
    dt: float = 1e-4          # Integration timestep (s); 0.1 ms
    seed: int = 1
    record_every: float = 1e-4

    # ── Network topology ──
    n_pre: int = 1

    # ── Pre-synaptic input ──
    r_pre: tuple[float, ...] = (20.0,)   # Pre firing rate (Hz), per synapse
    poisson: bool = False     # True = Poisson; False = deterministic periodic

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
    M_rule: str = "covariance"
    # R = -(r_post - r_target)^2  (target_rate)  |  R = r_target - r_post  (target_rate_linear)
    R_rule: str = "target_rate"
    r_target: float = 10.0

    # ── Synaptic weights ──
    w0: tuple[float, ...] = (2.0,)
    wmax: float = 10.0
    eta_plus: float = 1e-4
    eta_minus: float = 1e-4

    if TYPE_CHECKING:
        # Constructor accepts scalars for the 5 broadcast fields; __post_init__
        # converts them to tuples. The fields are typed as tuples (above) so
        # internal reads are well-typed; this stub widens the public signature.
        def __init__(
            self,
            *,
            T: float = 20.0,
            dt: float = 1e-4,
            seed: int = 1,
            record_every: float = 1e-4,
            n_pre: int = 1,
            r_pre: tuple[float, ...] | float = (20.0,),
            poisson: bool = False,
            tau_m: float = 0.02,
            E_L: float = -65.0,
            V_reset: float = -70.0,
            theta: float = -50.0,
            tau_ref: float = 0.003,
            V0: float = -65.0,
            tau_s: float = 0.005,
            R_m: float = 50.0,
            I_s0: tuple[float, ...] | float = (0.0,),
            I_ext: float = 0.0,
            tau_plus: float = 0.02,
            tau_minus: float = 0.02,
            x_pre0: tuple[float, ...] | float = (0.0,),
            y_post0: float = 0.0,
            tau_r_post: float = 0.5,
            r_post0: float = 0.0,
            rate_mode: str = "exp",
            rate_window: float = 0.5,
            tau_e: float = 0.5,
            E0: tuple[float, ...] | float = (0.0,),
            tau_Rbar: float = 5.0,
            R_bar0: float = 0.0,
            M_rule: str = "covariance",
            R_rule: str = "target_rate",
            r_target: float = 10.0,
            w0: tuple[float, ...] | float = (2.0,),
            wmax: float = 10.0,
            eta_plus: float = 1e-4,
            eta_minus: float = 1e-4,
        ) -> None: ...

    def __post_init__(self):
        n = self.n_pre
        for attr in ("r_pre", "I_s0", "x_pre0", "E0", "w0"):
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
