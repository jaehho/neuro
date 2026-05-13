"""Test RK4 integrator against systems with known analytical solutions.

The smooth ODE subsystem (between spike events) is a set of exponential
decays plus a coupled LIF membrane equation. With careful initial
conditions we can avoid spikes entirely and compare the numerical
trajectory to the exact solution.

Three test families:

1. Pure exponential decays  – each trace variable independently
2. Coupled V + I_s          – LIF driven by decaying synaptic current
3. Convergence order        – verify RK4 is O(dt⁴) and Euler is O(dt¹)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from neuro import Params
from neuro.dynamics import _advance_state, _init_state, _rhs
from neuro.params import (
    R_POST_IDX,
    RBAR_IDX,
    V_IDX,
    Y_POST_IDX,
    E_idx,
    I_s_idx,
    W_idx,
    X_pre_idx,
)


# ── Helpers ────────────────────────────────────────────────────────

def _make_y0(p: Params) -> np.ndarray:
    return _init_state(p)


def _integrate(y0: np.ndarray, p: Params, dt: float, n_steps: int,
               method: str, voltage_active: bool = True) -> np.ndarray:
    y = y0.copy()
    for _ in range(n_steps):
        y = _advance_state(y, dt, p, method=method, voltage_active=voltage_active)
    return y


# ── Analytical solutions ───────────────────────────────────────────

def _exact_decay(x0: float, tau: float, t: float) -> float:
    return x0 * np.exp(-t / tau)


def _exact_voltage(
    V0: float, E_L: float, tau_m: float,
    R_m: float, w: float, I_s0: float, tau_s: float,
    t: float,
) -> float:
    """Exact V(t) for LIF membrane driven by exponentially decaying I_s.

    ODE:  τ_m dV/dt = -(V - E_L) + R_m · w · I_s(t),  I_s(t) = I_s0 · exp(-t/τ_s)
    """
    u0 = V0 - E_L
    coeff = R_m * w * I_s0 * tau_s / (tau_s - tau_m)
    return (
        E_L
        + u0 * np.exp(-t / tau_m)
        + coeff * (np.exp(-t / tau_s) - np.exp(-t / tau_m))
    )


# ── Default Params kwargs for isolation: everything at zero/rest ──

_REST: dict[str, Any] = dict(
    n_pre=2,
    w0=(1.0, 0.0),
    E0=(0.0, 0.0),
    r_post0=0.0, R_bar0=0.0,
    V0=-65.0, I_s0=(0.0, 0.0),
    x_pre0=(0.0, 0.0), y_post0=0.0,
    r_target=0.0,
)


# ── 1. Pure exponential decays ─────────────────────────────────────

class TestExponentialDecays:
    """Each trace variable should decay as exp(-t/τ) when decoupled."""

    # (name, index, initial-value field, synapse_idx or None, tau field)
    CASES = [
        ("I_s1",   I_s_idx(0),   "I_s0",   0, "tau_s"),
        ("I_s2",   I_s_idx(1),   "I_s0",   1, "tau_s"),
        ("x_pre1", X_pre_idx(0), "x_pre0", 0, "tau_plus"),
        ("x_pre2", X_pre_idx(1), "x_pre0", 1, "tau_plus"),
        ("y_post", Y_POST_IDX,    "y_post0", None, "tau_minus"),
        ("E1",     E_idx(0),     "E0",     0, "tau_e"),
        ("E2",     E_idx(1),     "E0",     1, "tau_e"),
        ("r_post", R_POST_IDX,    "r_post0", None, "tau_r_post"),
        ("R_bar",  RBAR_IDX,      "R_bar0",  None, "tau_Rbar"),
    ]

    @pytest.mark.parametrize("name,idx,ic_field,syn_idx,tau_field", CASES,
                             ids=[c[0] for c in CASES])
    def test_rk4_decay(self, name: str, idx: int, ic_field: str,
                        syn_idx: int | None, tau_field: str) -> None:
        kwargs = dict(_REST)
        if syn_idx is not None:
            base = list(kwargs[ic_field])
            base[syn_idx] = 1.0
            kwargs[ic_field] = tuple(base)
        else:
            kwargs[ic_field] = 1.0
        p = Params(**kwargs)
        tau = getattr(p, tau_field)

        dt = 1e-4
        T = 5 * tau
        n_steps = int(T / dt)

        y_final = _integrate(_make_y0(p), p, dt, n_steps, method="rk4",
                             voltage_active=False)
        exact = _exact_decay(1.0, tau, T)
        rel_err = abs(y_final[idx] - exact) / max(abs(exact), 1e-30)
        assert rel_err < 1e-7, (
            f"{name}: relative error {rel_err:.2e} (numerical={y_final[idx]:.12e}, "
            f"exact={exact:.12e})"
        )

    @pytest.mark.parametrize("name,idx,ic_field,syn_idx,tau_field", CASES,
                             ids=[c[0] for c in CASES])
    def test_euler_decay(self, name: str, idx: int, ic_field: str,
                          syn_idx: int | None, tau_field: str) -> None:
        kwargs = dict(_REST)
        if syn_idx is not None:
            base = list(kwargs[ic_field])
            base[syn_idx] = 1.0
            kwargs[ic_field] = tuple(base)
        else:
            kwargs[ic_field] = 1.0
        p = Params(**kwargs)
        tau = getattr(p, tau_field)

        dt = 1e-4
        T = 5 * tau
        n_steps = int(T / dt)

        y_final = _integrate(_make_y0(p), p, dt, n_steps, method="euler",
                             voltage_active=False)
        exact = _exact_decay(1.0, tau, T)
        rel_err = abs(y_final[idx] - exact) / max(abs(exact), 1e-30)
        assert rel_err < 0.1, f"{name}: Euler relative error {rel_err:.2e}"


# ── 2. Coupled LIF membrane + decaying synaptic current ───────────

class TestCoupledVoltage:
    """V(t) with exponentially decaying I_s1 and fixed weight."""

    def _params(self) -> Params:
        return Params(
            n_pre=2,
            V0=-65.0, E_L=-65.0, tau_m=0.02, tau_s=0.005, R_m=50.0,
            theta=-50.0,
            I_s0=(0.1, 0.0), w0=(1.0, 0.0),
            E0=(0.0, 0.0),
            x_pre0=(0.0, 0.0), y_post0=0.0,
            r_post0=0.0, R_bar0=0.0,
            r_target=0.0,
        )

    def test_rk4_coupled_voltage(self) -> None:
        p = self._params()
        dt = 1e-4
        T = 0.1
        n_steps = int(T / dt)

        y_final = _integrate(_make_y0(p), p, dt, n_steps, method="rk4",
                             voltage_active=True)
        V_exact = _exact_voltage(p.V0, p.E_L, p.tau_m, p.R_m, p.w0[0],
                                 p.I_s0[0], p.tau_s, T)
        abs_err = abs(y_final[V_IDX] - V_exact)
        assert abs_err < 1e-8, (
            f"Coupled V: abs error {abs_err:.2e} (num={y_final[V_IDX]:.10f}, "
            f"exact={V_exact:.10f})"
        )

    def test_rk4_coupled_voltage_trajectory(self) -> None:
        p = self._params()
        dt = 1e-4
        checkpoints_ms = [1, 5, 10, 20, 50, 100]

        y = _make_y0(p)
        step = 0
        for t_ms in checkpoints_ms:
            t_target = t_ms * 1e-3
            target_step = int(t_target / dt)
            while step < target_step:
                y = _advance_state(y, dt, p, method="rk4", voltage_active=True)
                step += 1
            V_exact = _exact_voltage(p.V0, p.E_L, p.tau_m, p.R_m, p.w0[0],
                                     p.I_s0[0], p.tau_s, t_target)
            abs_err = abs(y[V_IDX] - V_exact)
            assert abs_err < 1e-7, f"t={t_ms}ms: abs error {abs_err:.2e}"

    def test_I_s_decays_correctly(self) -> None:
        p = self._params()
        dt = 1e-4
        T = 0.05
        n_steps = int(T / dt)

        y_final = _integrate(_make_y0(p), p, dt, n_steps, method="rk4",
                             voltage_active=True)
        exact = _exact_decay(p.I_s0[0], p.tau_s, T)
        rel_err = abs(y_final[I_s_idx(0)] - exact) / abs(exact)
        assert rel_err < 1e-7

    def test_weight_stays_constant(self) -> None:
        p = self._params()
        dt = 1e-4
        T = 0.1
        n_steps = int(T / dt)

        y_final = _integrate(_make_y0(p), p, dt, n_steps, method="rk4",
                             voltage_active=True)
        assert y_final[W_idx(0)] == pytest.approx(p.w0[0], abs=1e-15)
        assert y_final[W_idx(1)] == pytest.approx(p.w0[1], abs=1e-15)


# ── 3. Convergence order ───────────────────────────────────────────

class TestConvergenceOrder:
    """Verify that RK4 is 4th-order and Euler is 1st-order on smooth ODE."""

    def _run_convergence(self, method: str) -> list[float]:
        p = Params(
            n_pre=2,
            V0=-60.0, E_L=-65.0, tau_m=0.02, tau_s=0.005, R_m=50.0,
            theta=-40.0,
            I_s0=(0.5, 0.0), w0=(2.0, 0.0),
            E0=(0.0, 0.0),
            x_pre0=(0.0, 0.0), y_post0=0.0,
            r_post0=0.0, R_bar0=0.0,
            r_target=0.0,
        )
        T = 0.02
        V_exact = _exact_voltage(p.V0, p.E_L, p.tau_m, p.R_m, p.w0[0],
                                 p.I_s0[0], p.tau_s, T)

        dt_values = [2e-3, 1e-3, 5e-4, 2.5e-4]
        errors = []
        for dt in dt_values:
            n_steps = int(T / dt)
            y_final = _integrate(_make_y0(p), p, dt, n_steps, method=method,
                                 voltage_active=True)
            errors.append(abs(y_final[V_IDX] - V_exact))
        return errors

    def test_rk4_fourth_order(self) -> None:
        errors = self._run_convergence("rk4")
        ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        for i, ratio in enumerate(ratios):
            order = np.log2(ratio)
            assert order > 3.5, (
                f"RK4 refinement {i}: ratio={ratio:.2f}, order={order:.2f} (expected ~4.0)"
            )

    def test_euler_first_order(self) -> None:
        errors = self._run_convergence("euler")
        ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        for i, ratio in enumerate(ratios):
            order = np.log2(ratio)
            assert 0.8 < order < 1.5, (
                f"Euler refinement {i}: ratio={ratio:.2f}, order={order:.2f} (expected ~1.0)"
            )

    def test_rk4_much_more_accurate_than_euler(self) -> None:
        p = Params(
            n_pre=2,
            V0=-60.0, E_L=-65.0, tau_m=0.02, tau_s=0.005, R_m=50.0,
            theta=-40.0, I_s0=(0.5, 0.0), w0=(2.0, 0.0),
            E0=(0.0, 0.0),
            x_pre0=(0.0, 0.0), y_post0=0.0,
            r_post0=0.0, R_bar0=0.0,
            r_target=0.0,
        )
        dt = 1e-3
        T = 0.02
        n_steps = int(T / dt)
        y0 = _make_y0(p)
        V_exact = _exact_voltage(p.V0, p.E_L, p.tau_m, p.R_m, p.w0[0],
                                 p.I_s0[0], p.tau_s, T)

        y_rk4 = _integrate(y0, p, dt, n_steps, method="rk4", voltage_active=True)
        y_euler = _integrate(y0, p, dt, n_steps, method="euler", voltage_active=True)

        err_rk4 = abs(y_rk4[V_IDX] - V_exact)
        err_euler = abs(y_euler[V_IDX] - V_exact)
        assert err_rk4 < err_euler / 1000, (
            f"RK4 error={err_rk4:.2e}, Euler error={err_euler:.2e}, "
            f"ratio={err_euler / err_rk4:.0f}x"
        )


# ── 4. Edge cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_dt_returns_copy(self) -> None:
        p = Params(n_pre=2)
        y0 = _make_y0(p)
        y0[I_s_idx(0)] = 1.0
        result = _advance_state(y0, 0.0, p, method="rk4", voltage_active=True)
        np.testing.assert_array_equal(result, y0)
        assert result is not y0

    def test_weight_clamp_lower(self) -> None:
        p = Params(n_pre=2, w0=(0.001, 2.0), E0=(1.0, 0.0), R_bar0=100.0,
                   r_post0=0.0, r_target=0.0)
        y0 = _make_y0(p)
        result = _advance_state(y0, 0.01, p, method="rk4", voltage_active=False)
        assert result[W_idx(0)] >= 0.0

    def test_weight_clamp_upper(self) -> None:
        p = Params(n_pre=2, w0=(9.99, 2.0), wmax=10.0, E0=(1.0, 0.0), R_bar0=-100.0,
                   r_post0=0.0, r_target=0.0)
        y0 = _make_y0(p)
        result = _advance_state(y0, 0.01, p, method="rk4", voltage_active=False)
        assert result[W_idx(0)] <= p.wmax

    def test_rhs_at_equilibrium(self) -> None:
        p = Params(n_pre=2, V0=-65.0, E_L=-65.0,
                   I_s0=(0.0, 0.0),
                   x_pre0=(0.0, 0.0), y_post0=0.0,
                   E0=(0.0, 0.0),
                   r_post0=0.0,
                   R_bar0=0.0, w0=(1.0, 0.0), r_target=0.0)
        y0 = _make_y0(p)
        rhs = _rhs(y0, p, voltage_active=True)
        np.testing.assert_allclose(rhs, 0.0, atol=1e-15)
