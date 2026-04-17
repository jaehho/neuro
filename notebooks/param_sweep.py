"""Parameter sensitivity sweep: one knob at a time, baseline config.

Dropdown selects the parameter; the cell below sweeps it over a grid
and records final weight + late-half post rate. Other parameters match
`baseline.py`.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from neuro.sim import Params, simulate

    return Params, np, pl, plt, simulate


@app.cell
def _(mo):
    mo.md(r"""
    # Parameter sensitivity sweep

    Sweep one parameter at a time, hold the rest at the baseline
    defaults. Records final weight and late-half post rate.

    Edit `SWEEP_GRIDS` below to change the grids.
    """)
    return


@app.cell
def _(mo):
    sweep_param_select = mo.ui.dropdown(
        options=["alpha", "eta_plus", "tau_e", "tau_Rbar", "w0", "R_m", "r_target"],
        value="tau_e",
        label="parameter",
    )
    sweep_param_select
    return (sweep_param_select,)


@app.cell
def _(Params, np, simulate, sweep_param_select):
    SWEEP_GRIDS = {
        "alpha":    np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        "eta_plus": np.array([3e-5, 1e-4, 3e-4, 1e-3]),
        "tau_e":    np.array([0.1, 0.2, 0.5, 1.0, 2.0]),
        "tau_Rbar": np.array([1.0, 2.0, 5.0, 10.0, 20.0]),
        "w0":       np.array([0.5, 1.0, 2.0, 5.0, 8.0]),
        "R_m":      np.array([10.0, 20.0, 50.0, 80.0, 120.0]),
        "r_target": np.array([2.0, 5.0, 10.0, 20.0, 30.0]),
    }
    pname = sweep_param_select.value
    values_s = SWEEP_GRIDS[pname]
    sweep_results = []
    for v in values_s:
        kw_s = {pname: (float(v),) if pname == "w0" else float(v)}
        p_s = Params(
            T=10.0, dt=1e-4, method="rk4",
            n_pre=1, r_pre_rates=(20.0,), poisson=True,
            reward_signal="target_rate", target_func="fixed", r_target=10.0,
            neuromod_type="covariance",
            record_every=1e-3,
            **kw_s,
        )
        rec_s = simulate(p_s)
        half = p_s.T / 2
        late = rec_s["post_spike_times"][rec_s["post_spike_times"] >= half]
        rate_s = len(late) / (p_s.T - half) if p_s.T > half else 0.0
        sweep_results.append({"v": float(v), "w_final": float(rec_s["w1"][-1]), "rate": rate_s})
    return pname, sweep_results


@app.cell
def _(pl, plt, pname, sweep_results):
    df_s = pl.DataFrame(sweep_results)
    fig_s, axes_s = plt.subplots(1, 2, figsize=(11, 4))
    axes_s[0].plot(df_s["v"], df_s["w_final"], "o-", color="tab:orange")
    axes_s[0].set_xlabel(pname)
    axes_s[0].set_ylabel("w_final")
    axes_s[0].set_title(f"Final weight vs {pname}")
    axes_s[0].grid(True, alpha=0.2)

    axes_s[1].plot(df_s["v"], df_s["rate"], "o-", color="tab:red")
    axes_s[1].axhline(10.0, ls="--", color="black", alpha=0.5, label="target")
    axes_s[1].set_xlabel(pname)
    axes_s[1].set_ylabel("late post rate (Hz)")
    axes_s[1].set_title(f"Post rate vs {pname}")
    axes_s[1].legend()
    axes_s[1].grid(True, alpha=0.2)
    fig_s.tight_layout()
    fig_s
    return


if __name__ == "__main__":
    app.run()
