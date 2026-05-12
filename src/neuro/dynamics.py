"""Continuous dynamics, integration, and the three-factor learning rule.

Weight update:
    dw_i/dt = M(t) · E_i(t)

Modulator (M_rule = "covariance" | "gated"):
    M = R - R_bar     |     M = R

Reward (R_rule = "target_rate" | "target_rate_linear"):
    R = -(r_post - r_target)^2          (quadratic)
    R = r_target - r_post                (linear)

Reward baseline (always tracks R):
    dR_bar/dt = (R - R_bar) / tau_Rbar

Spike events (pre/post) apply instantaneous jumps to traces; this module
handles only the smooth RHS between events and the Euler/RK4 step.
"""
from __future__ import annotations

import numpy as np

from neuro.params import (
    N_PER_SYN,
    N_SHARED,
    Params,
    R_POST_IDX,
    RBAR_IDX,
    V_IDX,
    Y_POST_IDX,
    n_state,
)


def _per_syn(y: np.ndarray, n_pre: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Strided views into the per-synapse block: (I_s, x_pre, E, w)."""
    block = y[N_SHARED:N_SHARED + n_pre * N_PER_SYN].reshape(n_pre, N_PER_SYN)
    return block[:, 0], block[:, 1], block[:, 2], block[:, 3]


def _reward(p: Params, r_post: float) -> float:
    if p.R_rule == "target_rate":
        return -(r_post - p.r_target) ** 2
    if p.R_rule == "target_rate_linear":
        return p.r_target - r_post
    raise ValueError(f"Unknown R_rule: {p.R_rule!r}")


def _modulation(p: Params, R: float, R_bar: float) -> float:
    if p.M_rule == "covariance":
        return R - R_bar
    if p.M_rule == "gated":
        return R
    raise ValueError(f"Unknown M_rule: {p.M_rule!r}")


def _init_state(p: Params) -> np.ndarray:
    y = np.zeros(n_state(p.n_pre), dtype=np.float64)
    y[V_IDX]      = p.V0
    y[Y_POST_IDX] = p.y_post0
    y[R_POST_IDX] = p.r_post0
    y[RBAR_IDX]   = p.R_bar0

    I_s, x_pre, E, w = _per_syn(y, p.n_pre)
    I_s[:]   = p.I_s0
    x_pre[:] = p.x_pre0
    E[:]     = p.E0
    w[:]     = p.w0
    return y


def _rhs(y: np.ndarray, p: Params, *, voltage_active: bool,
         rate_post: float | None = None) -> np.ndarray:
    """RHS of the ODE between spikes (decay + V/I_s coupling, plus dw = M·E)."""
    I_s, x_pre, E, w = _per_syn(y, p.n_pre)
    rpost = rate_post if rate_post is not None else y[R_POST_IDX]
    R = _reward(p, rpost)
    M = _modulation(p, R, y[RBAR_IDX])

    rhs = np.zeros_like(y)
    if voltage_active:
        I_total = (np.clip(w, 0.0, p.wmax) * I_s).sum() + p.I_ext
        rhs[V_IDX] = (-(y[V_IDX] - p.E_L) + p.R_m * I_total) / p.tau_m

    rhs[Y_POST_IDX] = -y[Y_POST_IDX] / p.tau_minus
    rhs[R_POST_IDX] = -y[R_POST_IDX] / p.tau_r_post
    rhs[RBAR_IDX]   = (R - y[RBAR_IDX]) / p.tau_Rbar

    rI_s, rx_pre, rE, rw = _per_syn(rhs, p.n_pre)
    rI_s[:]   = -I_s   / p.tau_s
    rx_pre[:] = -x_pre / p.tau_plus
    rE[:]     = -E     / p.tau_e
    rw[:]     = M * E
    return rhs


def _advance_state(y: np.ndarray, dt: float, p: Params, *, method: str,
                   voltage_active: bool, rate_post: float | None = None) -> np.ndarray:
    """Step y forward by dt using Euler or classical RK4."""
    def rhs(state):
        return _rhs(state, p, voltage_active=voltage_active, rate_post=rate_post)

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

    _, _, _, w = _per_syn(out, p.n_pre)
    w[:] = np.clip(w, 0.0, p.wmax)
    if not voltage_active:
        out[V_IDX] = p.V_reset
    return out


def _crossing_fraction(v0: float, v1: float, threshold: float) -> float | None:
    """Linear interp for upward threshold crossing; spike-time error is O(dt²)."""
    if v0 >= threshold or v1 < threshold:
        return None
    dv = v1 - v0
    if dv <= 0.0:
        return 1.0
    return float(np.clip((threshold - v0) / dv, 0.0, 1.0))


def _row_from_state(t: float, y: np.ndarray, p: Params, *,
                    pre_spikes: list[int], post_spike: int, is_refractory: int,
                    rate_post: float | None = None) -> dict[str, float]:
    rr = rate_post if rate_post is not None else float(y[R_POST_IDX])
    R = _reward(p, rr)
    M = _modulation(p, R, float(y[RBAR_IDX]))

    I_s, x_pre, E, w = _per_syn(y, p.n_pre)
    row: dict[str, float] = {"t": t, "V": float(y[V_IDX])}
    for i in range(p.n_pre):
        row[f"w{i+1}"]     = float(w[i])
        row[f"I_s{i+1}"]   = float(I_s[i])
        row[f"x_pre{i+1}"] = float(x_pre[i])
        row[f"E{i+1}"]     = float(E[i])
        row[f"r_pre{i+1}"] = float(p.r_pre[i])
    row["y_post"] = float(y[Y_POST_IDX])
    row["r_post"] = rr
    row["R"]      = R
    row["R_bar"]  = float(y[RBAR_IDX])
    row["M"]      = M
    for i in range(p.n_pre):
        row[f"pre{i+1}_spike_bin"] = int(pre_spikes[i])
    row["post_spike_bin"] = int(post_spike)
    row["is_refractory"]  = int(is_refractory)
    return row
