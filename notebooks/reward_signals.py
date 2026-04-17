"""Reward signals: target_rate vs biofeedback vs contingent.

Holds neuromod_type=covariance and sweeps the reward source. contingent
needs n_pre=2 (target + distractor); the other two use n_pre=1.

Varies from `baseline.py` by: reward_signal (and n_pre for contingent).
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
        # Reward signals

        Holding `neuromod_type="covariance"`, swap the reward source:

        - **target_rate** — $R = -(r_\text{post} - r_\text{target})^2$ (self-supervisory demo)
        - **biofeedback** — delayed pulse after every post-spike (Legenstein 2008)
        - **contingent** — delayed pulse only after pre→post coincidences (Izhikevich 2007)
        """
    )
    return


@app.cell
def _(Params, simulate):
    _REWARD_CONFIGS = {
        "target_rate": dict(reward_signal="target_rate", target_func="fixed", r_target=10.0),
        "biofeedback": dict(reward_signal="biofeedback", reward_delay=0.5, reward_amount=1.0, reward_tau=0.2),
        "contingent":  dict(reward_signal="contingent",  reward_delay=0.5, reward_amount=1.0, reward_tau=0.2,
                            coincidence_window=0.02),
    }
    runs_reward = {}
    for _name_r, _cfg_r in _REWARD_CONFIGS.items():
        # contingent needs n_pre=2 to be meaningful (target + distractor)
        _n_r = 2 if _cfg_r["reward_signal"] == "contingent" else 1
        _p_r = Params(
            T=10.0, dt=1e-4, method="rk4",
            n_pre=_n_r,
            r_pre_rates=tuple(20.0 for _ in range(_n_r)),
            poisson=True,
            w0=tuple(2.0 for _ in range(_n_r)),
            neuromod_type="covariance",
            record_every=1e-3,
            **_cfg_r,
        )
        runs_reward[_name_r] = (_p_r, simulate(_p_r))
    return (runs_reward,)


@app.cell
def _(plt, runs_reward):
    fig_r, axes_r = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for _name_r, (_p_r, _rec_r) in runs_reward.items():
        axes_r[0].plot(_rec_r["t"], _rec_r["w1"], label=f"{_name_r}: w₁", lw=0.8)
        axes_r[1].plot(_rec_r["t"], _rec_r["R"], label=_name_r, lw=0.4, alpha=0.7)
        axes_r[2].plot(_rec_r["t"], _rec_r["M"], label=_name_r, lw=0.4, alpha=0.7)
    axes_r[0].set_ylabel("w₁")
    axes_r[0].set_title("Weight trajectories")
    axes_r[0].legend(loc="upper right", fontsize=8)
    axes_r[1].set_ylabel("R")
    axes_r[1].legend(loc="upper right", fontsize=8)
    axes_r[2].set_ylabel("M")
    axes_r[2].set_xlabel("Time (s)")
    axes_r[2].legend(loc="upper right", fontsize=8)
    for _ax_r in axes_r:
        _ax_r.grid(True, alpha=0.15)
    fig_r.tight_layout()
    fig_r
    return


if __name__ == "__main__":
    app.run()
