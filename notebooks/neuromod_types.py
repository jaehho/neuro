"""Neuromodulator types: covariance vs gated vs surprise vs constant.

Same reward signal (target_rate) on every run; the rule mapping R → M
varies.

Varies from `baseline.py` by: neuromod_type (sweeps all four values).
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    from neuro.sim import Params, simulate
    return Params, plt, simulate


@app.cell
def _(mo):
    mo.md(
        r"""
        # Neuromodulator types

        Same reward signal (`target_rate`), different rules for $M$:

        | type        | $M$                      |
        |-------------|--------------------------|
        | covariance  | $R - \bar R$ (RPE)       |
        | gated       | $R$                      |
        | surprise    | $(r_\text{post} - \bar R)^2$ |
        | constant    | $1$ (non-modulated STDP) |
        """
    )
    return


@app.cell
def _(Params, simulate):
    _NEUROMOD_TYPES = ["covariance", "gated", "surprise", "constant"]
    runs_neuromod = {}
    for _nm in _NEUROMOD_TYPES:
        _p_nm = Params(
            T=10.0,
            dt=1e-4,
            method="rk4",
            n_pre=1,
            r_pre_rates=(20.0,),
            poisson=True,
            w0=(2.0,),
            reward_signal="target_rate",
            target_func="fixed",
            r_target=10.0,
            neuromod_type=_nm,
            record_every=1e-3,
        )
        runs_neuromod[_nm] = (_p_nm, simulate(_p_nm))
    return (runs_neuromod,)


@app.cell
def _(plt, runs_neuromod):
    fig_nm, axes_nm = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for _nm, (_p_nm, _rec_nm) in runs_neuromod.items():
        axes_nm[0].plot(_rec_nm["t"], _rec_nm["w1"], label=_nm, lw=0.8, alpha=0.85)
        axes_nm[1].plot(_rec_nm["t"], _rec_nm["r_post"], label=_nm, lw=0.5, alpha=0.85)
        axes_nm[2].plot(_rec_nm["t"], _rec_nm["M"], label=_nm, lw=0.4, alpha=0.7)
        axes_nm[3].plot(_rec_nm["t"], _rec_nm["E1"], label=_nm, lw=0.4, alpha=0.7)

    _titles_nm = ["w₁", "r_post (Hz)", "M", "E₁"]
    for _ax_nm, _t_nm in zip(axes_nm, _titles_nm):
        _ax_nm.set_ylabel(_t_nm)
        _ax_nm.legend(loc="upper right", fontsize=8, ncol=2)
        _ax_nm.grid(True, alpha=0.15)
    axes_nm[1].axhline(10.0, ls="--", color="black", alpha=0.5)
    axes_nm[-1].set_xlabel("Time (s)")
    fig_nm.tight_layout()
    fig_nm
    return


if __name__ == "__main__":
    app.run()
