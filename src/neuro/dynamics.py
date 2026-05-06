"""Continuous dynamics, integration, and reward/modulation rules.

Three-factor STDP rule: ``dw_i/dt = M(t) · E_i(t)`` where M is a global
neuromodulatory signal and E_i is per-synapse eligibility.

This module groups: target-rate function family, reward/modulation, the
smooth RHS (between spike events), the Euler/RK4 step, threshold-crossing
interpolation, initial-state packing, and the per-step row builder.

The simulation loop ``simulate()`` lives in ``simulate.py`` and calls
into these helpers; spike-triggered jumps stay in the loop.
"""
from __future__ import annotations

import json
from typing import Callable

import numpy as np

from neuro.params import (
    Params,
    R_POST_IDX,
    RBAR_IDX,
    V_IDX,
    Y_POST_IDX,
    _E_idx,
    _I_s_idx,
    _W_idx,
    _X_pre_idx,
    n_state,
)


# ── General target function helpers ──────────────────────────────────

_TARGET_PARAMS_CACHE: dict[str, dict] = {}


def _parse_target_params(s: str) -> dict:
    """Parse JSON coefficient string, caching the result."""
    if s not in _TARGET_PARAMS_CACHE:
        _TARGET_PARAMS_CACHE[s] = json.loads(s) if s else {}
    return _TARGET_PARAMS_CACHE[s]


def _compute_target_r_post(p: Params, r_pre: float) -> float:
    """Compute target r_post from r_pre using the configured target function.

    Each function has sensible defaults so ``target_func_params`` can be
    left empty for a quick test.
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


# ── Reward and modulation ──────────────────────────────────────────

def _compute_reward(
    p: Params,
    r_pre: float,
    r_post: float,
    reward_pulse: float = 0.0,
) -> float:
    """Compute the reward signal R.

    For ``reward_signal="target_rate"``, R = -(r_post - target)² penalises
    deviation from a target firing rate.  This is a self-supervisory
    demonstration; the literature uses external/task-based R.

    For ``reward_signal="biofeedback"``, R is an externally delivered pulse
    triggered by post-synaptic spikes with a configurable delay (Izhikevich
    2007; Legenstein et al. 2008).
    """
    sig = p.reward_signal

    if sig == "target_rate":
        target = max(_compute_target_r_post(p, r_pre), 0.0)
        return -(r_post - target) ** 2
    elif sig == "target_rate_linear":
        target = max(_compute_target_r_post(p, r_pre), 0.0)
        return target - r_post
    elif sig == "constant":
        return p.R_const
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
    Follows Frémaux & Gerstner (2016), Eq. 14.
    """
    mode = p.neuromod_type

    if mode == "covariance":
        return R - R_bar, R
    elif mode == "gated":
        return R, R
    elif mode == "surprise":
        S = (r_post - R_bar) ** 2
        return S, r_post
    elif mode == "constant":
        return 1.0, R_bar
    else:
        raise ValueError(f"Unknown neuromod_type: {p.neuromod_type!r}")


# ── State packing and continuous-time dynamics ────────────────────

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
        y[_W_idx(i)] = p.w0[i]
    return y


def _smooth_rhs(
    y: np.ndarray,
    p: Params,
    *,
    voltage_active: bool,
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

    V = float(y[V_IDX])
    yp = float(y[Y_POST_IDX])
    r_post = float(y[R_POST_IDX])
    R_bar = float(y[RBAR_IDX])

    rr_pre = float(p.r_pre_rates[0])
    rr_post = rate_post if rate_post is not None else r_post
    R = _compute_reward(p, rr_pre, rr_post, reward_pulse=reward_pulse)
    M, rbar_target = _compute_modulation(p, R, R_bar, rr_post)

    if voltage_active:
        I_total = 0.0
        for i in range(n):
            wi = float(np.clip(y[_W_idx(i)], 0.0, p.wmax))
            I_total += wi * float(y[_I_s_idx(i)])
        rhs[V_IDX] = (-(V - p.E_L) + p.R_m * (I_total + p.I_ext)) / p.tau_m

    rhs[Y_POST_IDX] = -yp / p.tau_minus
    rhs[R_POST_IDX] = -r_post / p.tau_r
    rhs[RBAR_IDX] = (-R_bar + rbar_target) / p.tau_Rbar

    for i in range(n):
        I_s = float(y[_I_s_idx(i)])
        x = float(y[_X_pre_idx(i)])
        E = float(y[_E_idx(i)])
        rhs[_I_s_idx(i)] = -I_s / p.tau_s
        rhs[_X_pre_idx(i)] = -x / p.tau_plus
        rhs[_E_idx(i)] = -E / p.tau_e
        rhs[_W_idx(i)] = M * E

    return rhs


def _advance_state(
    y: np.ndarray,
    dt: float,
    p: Params,
    *,
    method: str,
    voltage_active: bool,
    rate_post: float | None = None,
    reward_pulse: float = 0.0,
) -> np.ndarray:
    """Advance state vector by dt using Euler or classical RK4.

    The system is autonomous (no explicit t in the smooth RHS), so time
    is not forwarded.  Weight is projected onto [0, wmax] after each step.
    """
    kw: dict[str, object] = dict(voltage_active=voltage_active, rate_post=rate_post, reward_pulse=reward_pulse)

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

    Returns fraction f ∈ [0,1] such that V(t + f·dt) ≈ threshold, or
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
    rate_post: float | None = None,
    reward_pulse: float = 0.0,
) -> dict[str, float]:
    n = p.n_pre
    rr_pre = float(p.r_pre_rates[0])
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
        row[f"r_pre{i+1}"] = float(p.r_pre_rates[i])
    row["r_post"] = rr_post
    row["R"] = float(R)
    row["R_bar"] = float(y[RBAR_IDX])
    row["M"] = float(M)
    for i in range(n):
        row[f"pre{i+1}_spike_bin"] = int(pre_spikes[i])
    row["post_spike_bin"] = int(post_spike)
    row["is_refractory"] = int(is_refractory)
    return row
