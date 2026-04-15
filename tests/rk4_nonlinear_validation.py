"""Validate RK4 on nonlinear ODEs with known analytical solutions.

Plot-generating validation script (NOT a pytest module — the unit-level
RK4 tests live in ``test_rk4.py``). The stepper here mirrors the RK4
branch in ``src/neuro/sim.py`` ``_advance_state``, extracted to accept
any autonomous RHS  f(y) → dy/dt.

Systems
-------
1. Logistic growth        y' = r·y·(1 − y/K)        → sigmoid
2. Cubic nonlinear decay  y' = −y³                   → algebraic tail
3. Riccati equation       y' = 1 + y²                → tan(t) blowup
4. Nonlinear pendulum     θ″ = −sin θ               → Jacobi cn elliptic

Run:  uv run python tests/rk4_nonlinear_validation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.special import ellipj, ellipk

OUTPUT = Path("output")

# ── Steppers ─────────────────────────────────────────────────────────
# rk4_step is identical to sim.py _advance_state lines 527-531.


def rk4_step(y: np.ndarray, dt: float, f) -> np.ndarray:
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_step(y: np.ndarray, dt: float, f) -> np.ndarray:
    return y + dt * f(y)


def integrate(stepper, f, y0: np.ndarray, dt: float, T: float):
    n = int(round(T / dt))
    ts = np.arange(n + 1) * dt
    ys = np.empty((n + 1,) + y0.shape)
    ys[0] = y0.copy()
    for i in range(n):
        ys[i + 1] = stepper(ys[i], dt, f)
    return ts, ys


# ── System 1: Logistic growth ───────────────────────────────────────
# dy/dt = r·y·(1 − y/K)
# Exact: y(t) = K·y₀ / [y₀ + (K − y₀)·exp(−r·t)]

R_LOG, K_LOG = 2.0, 10.0


def logistic_f(y: np.ndarray) -> np.ndarray:
    return np.array([R_LOG * y[0] * (1.0 - y[0] / K_LOG)])


def logistic_exact(t: np.ndarray, y0: float) -> np.ndarray:
    return K_LOG * y0 / (y0 + (K_LOG - y0) * np.exp(-R_LOG * t))


# ── System 2: Cubic nonlinear decay ─────────────────────────────────
# dy/dt = −y³
# Exact: y(t) = y₀ / √(1 + 2·y₀²·t)


def cubic_f(y: np.ndarray) -> np.ndarray:
    return np.array([-y[0] ** 3])


def cubic_exact(t: np.ndarray, y0: float) -> np.ndarray:
    return y0 / np.sqrt(1.0 + 2.0 * y0**2 * t)


# ── System 3: Riccati equation ──────────────────────────────────────
# dy/dt = 1 + y²
# Exact: y(t) = tan(t + arctan(y₀))
# Blows up at t* = π/2 − arctan(y₀)


def riccati_f(y: np.ndarray) -> np.ndarray:
    return np.array([1.0 + y[0] ** 2])


def riccati_exact(t: np.ndarray, y0: float) -> np.ndarray:
    return np.tan(t + np.arctan(y0))


# ── System 4: Nonlinear pendulum ────────────────────────────────────
# θ″ = −sin θ   →   state [θ, ω],  ω = dθ/dt
# Exact (starting from rest): θ(t) = 2·arcsin(k · cn(t, m))
#   where k = sin(θ₀/2), m = k²
# Period = 4·K(m)  (K = complete elliptic integral of 1st kind)

THETA0 = 2.5  # ~143°, strongly nonlinear


def pendulum_f(y: np.ndarray) -> np.ndarray:
    return np.array([y[1], -np.sin(y[0])])


def pendulum_exact(t: np.ndarray, theta0: float) -> np.ndarray:
    k = np.sin(theta0 / 2.0)
    m = k**2
    _sn, cn, _dn, _ph = ellipj(t, m)
    return 2.0 * np.arcsin(k * cn)


def pendulum_period(theta0: float) -> float:
    k = np.sin(theta0 / 2.0)
    return 4.0 * ellipk(k**2)


# ── Convergence order measurement ───────────────────────────────────


def max_error(f, y0, exact_fn, T, dt, component: int = 0):
    ts, ys = integrate(rk4_step, f, y0, dt, T)
    return float(np.max(np.abs(ys[:, component] - exact_fn(ts))))


# ── Plotting ────────────────────────────────────────────────────────

C_EXACT = "#1a1a2e"
C_RK4 = "#e63946"
C_EULER = "#457b9d"


def plot_system(
    ax_sol,
    ax_err,
    *,
    title: str,
    equation: str,
    f,
    y0: np.ndarray,
    T: float,
    dt: float,
    exact_fn,
    component: int = 0,
    ylabel: str = "y",
    extra_sol=None,
):
    """Plot one system: solutions on ax_sol, errors on ax_err."""
    # Numerical solutions
    t_rk4, y_rk4 = integrate(rk4_step, f, y0, dt, T)
    t_eu, y_eu = integrate(euler_step, f, y0, dt, T)

    # Fine analytical curve
    t_fine = np.linspace(0, T, 2000)
    y_ex_fine = exact_fn(t_fine)
    y_ex_rk4 = exact_fn(t_rk4)
    y_ex_eu = exact_fn(t_eu)

    # ── Solution panel ──
    ax_sol.plot(t_fine, y_ex_fine, color=C_EXACT, lw=2.0, label="Analytical", zorder=3)
    ax_sol.plot(
        t_rk4, y_rk4[:, component], "o",
        color=C_RK4, ms=3, label=f"RK4  (dt = {dt})", zorder=4,
    )
    ax_sol.plot(
        t_eu, y_eu[:, component], "x",
        color=C_EULER, ms=3, alpha=0.7, label=f"Euler (dt = {dt})", zorder=2,
    )
    if extra_sol is not None:
        extra_sol(ax_sol, t_fine)
    ax_sol.set_ylabel(ylabel, fontsize=11)
    ax_sol.set_title(f"{title}:  {equation}", fontsize=12)
    ax_sol.legend(fontsize=8, loc="best")

    # ── Error panel ──
    err_rk4 = np.abs(y_rk4[:, component] - y_ex_rk4)
    err_eu = np.abs(y_eu[:, component] - y_ex_eu)
    floor = 1e-16

    ax_err.semilogy(t_rk4, err_rk4 + floor, "-", color=C_RK4, lw=1.2, label="RK4")
    ax_err.semilogy(t_eu, err_eu + floor, "-", color=C_EULER, lw=1.2, alpha=0.7, label="Euler")
    ax_err.set_ylabel("|error|", fontsize=11)
    ax_err.set_title("Absolute error", fontsize=12)
    ax_err.legend(fontsize=8, loc="best")

    max_rk4 = float(np.max(err_rk4))
    max_eu = float(np.max(err_eu))
    ratio = max_eu / max_rk4 if max_rk4 > 0 else float("inf")
    ax_err.annotate(
        f"max RK4 = {max_rk4:.2e}\nmax Euler = {max_eu:.2e}\nratio = {ratio:.0f}×",
        xy=(0.98, 0.98), xycoords="axes fraction",
        ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
    )


def main(*, convergence: bool = False) -> None:
    OUTPUT.mkdir(exist_ok=True)

    n_rows = 5 if convergence else 4
    ratios = [1, 1, 1, 1, 0.9] if convergence else [1, 1, 1, 1]
    fig = plt.figure(figsize=(15, 18 if not convergence else 22), constrained_layout=True)
    gs = GridSpec(n_rows, 2, figure=fig, height_ratios=ratios)

    # ── 1. Logistic growth ──────────────────────────────────────
    y0_log = np.array([0.1])
    T_log, dt_log = 8.0, 0.1

    plot_system(
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        title="Logistic growth",
        equation=r"$y' = 2y(1 - y/10)$",
        f=logistic_f,
        y0=y0_log,
        T=T_log,
        dt=dt_log,
        exact_fn=lambda t: logistic_exact(t, y0_log[0]),
    )

    # ── 2. Cubic nonlinear decay ────────────────────────────────
    y0_cub = np.array([2.0])
    T_cub, dt_cub = 10.0, 0.1

    plot_system(
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        title="Cubic decay",
        equation=r"$y' = -y^3$",
        f=cubic_f,
        y0=y0_cub,
        T=T_cub,
        dt=dt_cub,
        exact_fn=lambda t: cubic_exact(t, y0_cub[0]),
    )

    # ── 3. Riccati equation ─────────────────────────────────────
    y0_ric = np.array([0.0])
    t_blowup = np.pi / 2.0 - np.arctan(y0_ric[0])
    T_ric = t_blowup - 0.07  # stop just before singularity
    dt_ric = 0.02

    plot_system(
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        title="Riccati",
        equation=r"$y' = 1 + y^2$  (blows up at $t = \pi/2$)",
        f=riccati_f,
        y0=y0_ric,
        T=T_ric,
        dt=dt_ric,
        exact_fn=lambda t: riccati_exact(t, y0_ric[0]),
    )

    # ── 4. Nonlinear pendulum ───────────────────────────────────
    period = pendulum_period(THETA0)
    y0_pen = np.array([THETA0, 0.0])  # start from rest at θ₀
    T_pen = 2.0 * period
    dt_pen = 0.05

    plot_system(
        fig.add_subplot(gs[3, 0]),
        fig.add_subplot(gs[3, 1]),
        title=f"Nonlinear pendulum (θ₀ = {THETA0:.1f} rad ≈ {np.degrees(THETA0):.0f}°)",
        equation=r"$\ddot\theta = -\sin\theta$",
        f=pendulum_f,
        y0=y0_pen,
        T=T_pen,
        dt=dt_pen,
        exact_fn=lambda t: pendulum_exact(t, THETA0),
        ylabel=r"$\theta$ (rad)",
    )

    # ── 5. Convergence order (optional) ─────────────────────────
    if not convergence:
        path = OUTPUT / "rk4_validation.png"
        fig.savefig(path, dpi=150)
        print(f"Saved to {path}")
        plt.show()
        return

    ax_conv = fig.add_subplot(gs[4, :])

    dt_values = np.array([0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625])

    systems_conv = [
        ("Logistic", logistic_f, y0_log,
         lambda t: logistic_exact(t, y0_log[0]), 4.0),
        ("Cubic", cubic_f, y0_cub,
         lambda t: cubic_exact(t, y0_cub[0]), 5.0),
        ("Riccati", riccati_f, y0_ric,
         lambda t: riccati_exact(t, y0_ric[0]), 1.0),
        ("Pendulum", pendulum_f, y0_pen,
         lambda t: pendulum_exact(t, THETA0), period),
    ]
    colors_conv = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]

    for (name, f, y0, exact_fn, T), color in zip(systems_conv, colors_conv):
        errors = []
        for dt in dt_values:
            ts, ys = integrate(rk4_step, f, y0, dt, T)
            err = float(np.max(np.abs(ys[:, 0] - exact_fn(ts))))
            errors.append(err)
        ax_conv.loglog(dt_values, errors, "o-", color=color, lw=1.5, ms=5, label=name)

    # Reference slopes
    dt_ref = np.array([dt_values[0], dt_values[-1]])
    scale4 = 0.3 * errors[0] / dt_values[0] ** 4
    ax_conv.loglog(dt_ref, scale4 * dt_ref**4, "k--", lw=1, alpha=0.4, label=r"$O(\Delta t^4)$")
    scale1 = 0.3 * errors[0] / dt_values[0]
    ax_conv.loglog(dt_ref, scale1 * dt_ref, "k:", lw=1, alpha=0.4, label=r"$O(\Delta t)$")

    ax_conv.set_xlabel(r"$\Delta t$", fontsize=12)
    ax_conv.set_ylabel("max |error|", fontsize=12)
    ax_conv.set_title("Convergence order — all systems follow the 4th-order slope", fontsize=12)
    ax_conv.legend(fontsize=9, ncol=3)
    ax_conv.grid(True, which="both", alpha=0.3)

    # ── Save ────────────────────────────────────────────────────
    path = OUTPUT / "rk4_validation.png"
    fig.savefig(path, dpi=150)
    print(f"Saved to {path}")
    plt.show()


if __name__ == "__main__":
    import sys
    main(convergence="--convergence" in sys.argv)
