"""Target-function shapes: fixed vs linear vs sqrt vs log.

With reward_signal=target_rate, the target rate can be a function of
r_pre instead of a fixed constant. Useful for studying input/output
transfer functions.

Varies from `baseline.py` by: target_func (sweeps fixed/linear/sqrt/log).
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
    from neuro.sim import Params, simulate

    return Params, plt, simulate


@app.cell
def _(mo):
    mo.md(r"""
    # Target-function shapes

    With `reward_signal="target_rate"`, the target rate can be a
    function of $r_\text{pre}$ instead of a fixed constant. Useful
    for studying input/output transfer functions.
    """)
    return


@app.cell
def _(Params, simulate):
    _TARGET_FUNCS = ["fixed", "linear", "sqrt", "log"]
    runs_target = {}
    for _tf in _TARGET_FUNCS:
        _p_tf = Params(
            T=10.0, dt=1e-4, method="rk4",
            n_pre=1, r_pre_rates=(20.0,), poisson=True, w0=(2.0,),
            reward_signal="target_rate",
            target_func=_tf,
            r_target=10.0, alpha=0.5,
            neuromod_type="covariance",
            record_every=1e-3,
        )
        runs_target[_tf] = (_p_tf, simulate(_p_tf))
    return (runs_target,)


@app.cell
def _(plt, runs_target):
    fig_t, axes_t = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for _tf, (_p_tf, _rec_tf) in runs_target.items():
        axes_t[0].plot(_rec_tf["t"], _rec_tf["w1"], label=_tf, lw=0.8)
        axes_t[1].plot(_rec_tf["t"], _rec_tf["r_post"], label=_tf, lw=0.5, alpha=0.8)
    axes_t[0].set_ylabel("w₁")
    axes_t[0].legend(loc="upper right", fontsize=8)
    axes_t[1].set_ylabel("r_post (Hz)")
    axes_t[1].set_xlabel("Time (s)")
    axes_t[1].legend(loc="upper right", fontsize=8)
    for _ax_t in axes_t:
        _ax_t.grid(True, alpha=0.15)
    fig_t.tight_layout()
    fig_t
    return


if __name__ == "__main__":
    app.run()
