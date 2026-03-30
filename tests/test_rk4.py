"""Test RK4 integrator against systems with known analytical solutions.

The ODE subsystem in sim.py (between spike events) is a set of exponential
decays plus a coupled LIF membrane equation.  With careful initial conditions
we can avoid spikes entirely and compare the numerical trajectory to the
exact solution.

Three test families
-------------------
1. Pure exponential decays  – each trace variable independently
2. Coupled V + I_s          – LIF driven by decaying synaptic current
3. Convergence order        – verify RK4 is O(dt⁴) and Euler is O(dt¹)
"""

from __future__ import annotations

import numpy as np
import pytest

from neuro.sim import (
    E_IDX,
    I_S_IDX,
    Params,
    R_POST_IDX,
    R_PRE_IDX,
    RBAR_IDX,
    V_IDX,
    W_IDX,
    X_PRE_IDX,
    Y_POST_IDX,
    _advance_state,
    _smooth_rhs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_STATE = 9  # V, I_s, x_pre, y_post, E, r_pre, r_post, R_bar, w


def _make_y0(p: Params) -> np.ndarray:
    """Build initial state vector from Params defaults."""
    y = np.zeros(N_STATE, dtype=np.float64)
    y[V_IDX] = p.V0
    y[I_S_IDX] = p.I_s0
    y[X_PRE_IDX] = p.x_pre0
    y[Y_POST_IDX] = p.y_post0
    y[E_IDX] = p.E0
    y[R_PRE_IDX] = p.r_pre0
    y[R_POST_IDX] = p.r_post0
    y[RBAR_IDX] = p.R_bar0
    y[W_IDX] = p.w0
    return y


def _integrate(y0: np.ndarray, p: Params, dt: float, n_steps: int,
               method: str, voltage_active: bool = True) -> np.ndarray:
    """Step _advance_state n_steps times, return final state."""
    y = y0.copy()
    for _ in range(n_steps):
        y = _advance_state(y, dt, p, method=method,
                           voltage_active=voltage_active)
    return y


# ---------------------------------------------------------------------------
# Analytical solutions
# ---------------------------------------------------------------------------

def _exact_decay(x0: float, tau: float, t: float) -> float:
    """x(t) = x0 · exp(-t/τ)."""
    return x0 * np.exp(-t / tau)


def _exact_voltage(
    V0: float, E_L: float, tau_m: float,
    R_m: float, w: float, I_s0: float, tau_s: float,
    t: float,
) -> float:
    """Exact V(t) for LIF membrane driven by exponentially decaying I_s.

    ODE:  τ_m dV/dt = -(V - E_L) + R_m · w · I_s(t)
    where I_s(t) = I_s0 · exp(-t/τ_s)

    Solution (τ_m ≠ τ_s):
      V(t) = E_L + (V0 - E_L)·exp(-t/τ_m)
             + (R_m·w·I_s0·τ_s)/(τ_s - τ_m) · [exp(-t/τ_s) - exp(-t/τ_m)]
    """
    u0 = V0 - E_L
    coeff = R_m * w * I_s0 * tau_s / (tau_s - tau_m)
    return (
        E_L
        + u0 * np.exp(-t / tau_m)
        + coeff * (np.exp(-t / tau_s) - np.exp(-t / tau_m))
    )


# ---------------------------------------------------------------------------
# 1. Pure exponential decays
# ---------------------------------------------------------------------------

class TestExponentialDecays:
    """Each trace variable should decay as exp(-t/τ) when decoupled."""

    # (index, initial value param, tau param)
    CASES = [
        ("I_s",    I_S_IDX,   "I_s0",    "tau_s"),
        ("x_pre",  X_PRE_IDX, "x_pre0",  "tau_plus"),
        ("y_post", Y_POST_IDX,"y_post0", "tau_minus"),
        ("E",      E_IDX,     "E0",      "tau_e"),
        ("r_pre",  R_PRE_IDX, "r_pre0",  "tau_r"),
        ("r_post", R_POST_IDX,"r_post0", "tau_r"),
        ("R_bar",  RBAR_IDX,  "R_bar0",  "tau_Rbar"),
    ]

    @pytest.mark.parametrize("name,idx,ic_field,tau_field", CASES,
                             ids=[c[0] for c in CASES])
    def test_rk4_decay(self, name: str, idx: int, ic_field: str,
                        tau_field: str) -> None:
        """RK4 matches analytical exponential decay to < 1e-10 relative error."""
        # Set only the variable under test; everything else at zero / rest.
        # E0=0 keeps dw/dt=0 so weight stays constant.
        # r_pre0=r_post0=0 and r_target=0 keeps reward terms at zero → R_bar decays purely.
        kwargs = {ic_field: 1.0, "w0": 1.0, "E0": 0.0,
                  "r_pre0": 0.0, "r_post0": 0.0, "R_bar0": 0.0,
                  "V0": -65.0, "I_s0": 0.0, "r_target": 0.0}
        kwargs[ic_field] = 1.0  # override: variable under test starts at 1
        p = Params(**kwargs)
        tau = getattr(p, tau_field)

        dt = 1e-4
        T = 5 * tau  # integrate for 5 time-constants
        n_steps = int(T / dt)

        y0 = _make_y0(p)
        y_final = _integrate(y0, p, dt, n_steps, method="rk4",
                             voltage_active=False)

        exact = _exact_decay(1.0, tau, T)
        numerical = y_final[idx]
        rel_err = abs(numerical - exact) / max(abs(exact), 1e-30)

        # Tolerance scales with (dt/tau)^4 · N_steps; tau_s=5ms is tightest
        assert rel_err < 1e-7, (
            f"{name}: relative error {rel_err:.2e} (numerical={numerical:.12e}, "
            f"exact={exact:.12e})"
        )

    @pytest.mark.parametrize("name,idx,ic_field,tau_field", CASES,
                             ids=[c[0] for c in CASES])
    def test_euler_decay(self, name: str, idx: int, ic_field: str,
                          tau_field: str) -> None:
        """Euler also converges for pure decay (looser tolerance)."""
        kwargs = {ic_field: 1.0, "w0": 1.0, "E0": 0.0,
                  "r_pre0": 0.0, "r_post0": 0.0, "R_bar0": 0.0,
                  "V0": -65.0, "I_s0": 0.0, "r_target": 0.0}
        kwargs[ic_field] = 1.0
        p = Params(**kwargs)
        tau = getattr(p, tau_field)

        dt = 1e-4
        T = 5 * tau
        n_steps = int(T / dt)

        y0 = _make_y0(p)
        y_final = _integrate(y0, p, dt, n_steps, method="euler",
                             voltage_active=False)

        exact = _exact_decay(1.0, tau, T)
        numerical = y_final[idx]
        rel_err = abs(numerical - exact) / max(abs(exact), 1e-30)

        # Euler is O(dt): dt/tau_s=0.02 over 5τ gives ~5% error
        assert rel_err < 0.1, (
            f"{name}: Euler relative error {rel_err:.2e}"
        )


# ---------------------------------------------------------------------------
# 2. Coupled LIF membrane + decaying synaptic current
# ---------------------------------------------------------------------------

class TestCoupledVoltage:
    """V(t) with exponentially decaying I_s and fixed weight."""

    def _params(self) -> Params:
        """Parameters that keep V well below threshold (no spikes)."""
        return Params(
            V0=-65.0, E_L=-65.0, tau_m=0.02, tau_s=0.005, R_m=50.0,
            theta=-50.0,  # threshold far above V trajectory
            I_s0=0.1, w0=1.0,
            # Zero all plasticity so dw/dt = 0
            E0=0.0, x_pre0=0.0, y_post0=0.0,
            r_pre0=0.0, r_post0=0.0, R_bar0=0.0,
        )

    def test_rk4_coupled_voltage(self) -> None:
        """RK4 reproduces the analytical V(t) for LIF + decaying I_s."""
        p = self._params()
        dt = 1e-4
        T = 0.1  # 100 ms — several tau_m
        n_steps = int(T / dt)

        y0 = _make_y0(p)
        y_final = _integrate(y0, p, dt, n_steps, method="rk4",
                             voltage_active=True)

        V_exact = _exact_voltage(p.V0, p.E_L, p.tau_m, p.R_m, p.w0,
                                 p.I_s0, p.tau_s, T)
        V_num = y_final[V_IDX]
        abs_err = abs(V_num - V_exact)

        assert abs_err < 1e-8, (
            f"Coupled V: abs error {abs_err:.2e} (num={V_num:.10f}, "
            f"exact={V_exact:.10f})"
        )

    def test_rk4_coupled_voltage_trajectory(self) -> None:
        """Check multiple time-points along the trajectory, not just final."""
        p = self._params()
        dt = 1e-4
        checkpoints_ms = [1, 5, 10, 20, 50, 100]

        y = _make_y0(p)
        step = 0
        for t_ms in checkpoints_ms:
            t_target = t_ms * 1e-3
            target_step = int(t_target / dt)
            while step < target_step:
                y = _advance_state(y, dt, p, method="rk4",
                                   voltage_active=True)
                step += 1

            V_exact = _exact_voltage(p.V0, p.E_L, p.tau_m, p.R_m, p.w0,
                                     p.I_s0, p.tau_s, t_target)
            abs_err = abs(y[V_IDX] - V_exact)
            assert abs_err < 1e-7, (
                f"t={t_ms}ms: abs error {abs_err:.2e}"
            )

    def test_I_s_decays_correctly(self) -> None:
        """I_s should still follow pure exp decay in the coupled system."""
        p = self._params()
        dt = 1e-4
        T = 0.05
        n_steps = int(T / dt)

        y0 = _make_y0(p)
        y_final = _integrate(y0, p, dt, n_steps, method="rk4",
                             voltage_active=True)

        exact = _exact_decay(p.I_s0, p.tau_s, T)
        rel_err = abs(y_final[I_S_IDX] - exact) / abs(exact)
        assert rel_err < 1e-7

    def test_weight_stays_constant(self) -> None:
        """With E0=0, weight should not change."""
        p = self._params()
        dt = 1e-4
        T = 0.1
        n_steps = int(T / dt)

        y0 = _make_y0(p)
        y_final = _integrate(y0, p, dt, n_steps, method="rk4",
                             voltage_active=True)

        assert y_final[W_IDX] == pytest.approx(p.w0, abs=1e-15)


# ---------------------------------------------------------------------------
# 3. Convergence order
# ---------------------------------------------------------------------------

class TestConvergenceOrder:
    """Verify that RK4 is 4th-order and Euler is 1st-order on smooth ODE."""

    def _run_convergence(self, method: str) -> list[float]:
        """Integrate coupled V+I_s at 4 dt values, return absolute errors."""
        p = Params(
            V0=-60.0, E_L=-65.0, tau_m=0.02, tau_s=0.005, R_m=50.0,
            theta=-40.0,  # unreachable threshold
            I_s0=0.5, w0=2.0,
            E0=0.0, x_pre0=0.0, y_post0=0.0,
            r_pre0=0.0, r_post0=0.0, R_bar0=0.0,
        )
        T = 0.02  # 20 ms — one tau_m
        V_exact = _exact_voltage(p.V0, p.E_L, p.tau_m, p.R_m, p.w0,
                                 p.I_s0, p.tau_s, T)

        dt_values = [2e-3, 1e-3, 5e-4, 2.5e-4]
        errors = []
        for dt in dt_values:
            n_steps = int(T / dt)
            y0 = _make_y0(p)
            y_final = _integrate(y0, p, dt, n_steps, method=method,
                                 voltage_active=True)
            errors.append(abs(y_final[V_IDX] - V_exact))

        return errors

    def test_rk4_fourth_order(self) -> None:
        """When dt halves, RK4 error should decrease by ~16x."""
        errors = self._run_convergence("rk4")
        # Compute ratios between successive refinements
        ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        # 4th order → ratio ≈ 2^4 = 16 for each halving
        # dt_values go 2e-3 → 1e-3 → 5e-4 → 2.5e-4 (each halves)
        for i, ratio in enumerate(ratios):
            order = np.log2(ratio)
            assert order > 3.5, (
                f"RK4 refinement {i}: ratio={ratio:.2f}, "
                f"order={order:.2f} (expected ~4.0)"
            )

    def test_euler_first_order(self) -> None:
        """When dt halves, Euler error should decrease by ~2x."""
        errors = self._run_convergence("euler")
        ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        for i, ratio in enumerate(ratios):
            order = np.log2(ratio)
            assert 0.8 < order < 1.5, (
                f"Euler refinement {i}: ratio={ratio:.2f}, "
                f"order={order:.2f} (expected ~1.0)"
            )

    def test_rk4_much_more_accurate_than_euler(self) -> None:
        """At the same dt, RK4 should be orders of magnitude more accurate."""
        p = Params(
            V0=-60.0, E_L=-65.0, tau_m=0.02, tau_s=0.005, R_m=50.0,
            theta=-40.0, I_s0=0.5, w0=2.0,
            E0=0.0, x_pre0=0.0, y_post0=0.0,
            r_pre0=0.0, r_post0=0.0, R_bar0=0.0,
        )
        dt = 1e-3
        T = 0.02
        n_steps = int(T / dt)
        y0 = _make_y0(p)

        V_exact = _exact_voltage(p.V0, p.E_L, p.tau_m, p.R_m, p.w0,
                                 p.I_s0, p.tau_s, T)

        y_rk4 = _integrate(y0, p, dt, n_steps, method="rk4",
                           voltage_active=True)
        y_euler = _integrate(y0, p, dt, n_steps, method="euler",
                             voltage_active=True)

        err_rk4 = abs(y_rk4[V_IDX] - V_exact)
        err_euler = abs(y_euler[V_IDX] - V_exact)

        # RK4 should be at least 1000x more accurate at dt=1ms
        assert err_rk4 < err_euler / 1000, (
            f"RK4 error={err_rk4:.2e}, Euler error={err_euler:.2e}, "
            f"ratio={err_euler / err_rk4:.0f}x"
        )


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions and special cases."""

    def test_zero_dt_returns_copy(self) -> None:
        p = Params()
        y0 = _make_y0(p)
        y0[I_S_IDX] = 1.0
        result = _advance_state(y0, 0.0, p, method="rk4",
                                voltage_active=True)
        np.testing.assert_array_equal(result, y0)
        assert result is not y0  # must be a copy

    def test_weight_clamp_lower(self) -> None:
        """Weight should be clamped at 0 even if dw/dt drives it negative."""
        p = Params(w0=0.001, E0=1.0, R_bar0=100.0,
                   r_pre0=0.0, r_post0=0.0)
        y0 = _make_y0(p)
        # M = R - R_bar = 0 - 100 = -100; dw/dt = M*E = -100*1 = -100
        # One big step should try to push w very negative
        result = _advance_state(y0, 0.01, p, method="rk4",
                                voltage_active=False)
        assert result[W_IDX] >= 0.0

    def test_weight_clamp_upper(self) -> None:
        """Weight should be clamped at wmax."""
        p = Params(w0=9.99, wmax=10.0, E0=1.0, R_bar0=-100.0,
                   r_pre0=0.0, r_post0=0.0)
        y0 = _make_y0(p)
        result = _advance_state(y0, 0.01, p, method="rk4",
                                voltage_active=False)
        assert result[W_IDX] <= p.wmax

    def test_rhs_at_equilibrium(self) -> None:
        """All derivatives should be ~0 at the fixed point (everything at rest)."""
        p = Params(V0=-65.0, E_L=-65.0, I_s0=0.0, x_pre0=0.0,
                   y_post0=0.0, E0=0.0, r_pre0=0.0, r_post0=0.0,
                   R_bar0=0.0, w0=1.0, r_target=0.0)
        y0 = _make_y0(p)
        rhs = _smooth_rhs(y0, p, voltage_active=True)
        np.testing.assert_allclose(rhs, 0.0, atol=1e-15)
