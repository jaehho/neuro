"""Spatial credit assignment: 2 pre + contingent reward.

Two pre-synaptic neurons fire at the same rate. Reward is delivered only
when pre1 (the "target" synapse) fires within a coincidence window
before a post-spike. Both synapses see the same global modulator M, but
only E₁ is high when reward arrives, so only w₁ should grow.

Varies from `baseline.py` by: n_pre=2 and reward_signal=contingent.
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
        # Spatial credit assignment — 2 pre + contingent reward

        Two pre-synaptic neurons fire at the same rate. Reward is delivered
        only when **pre1** (the "target" synapse) fires within a coincidence
        window before a post-spike. Both synapses see the same global
        modulator $M$, but only $E_1$ is high when reward arrives, so only
        $w_1$ should grow.

        Reference: Izhikevich (2007), Frémaux & Gerstner (2016) Eq. 10.
        """
    )
    return


@app.cell
def _(Params, simulate):
    p_credit = Params(
        T=20.0,
        dt=1e-4,
        method="rk4",
        n_pre=2,
        r_pre_rates=(20.0, 20.0),
        poisson=True,
        w0=(2.0, 2.0),
        reward_signal="contingent",
        coincidence_window=0.02,
        reward_delay=0.5,
        reward_amount=1.0,
        reward_tau=0.2,
        neuromod_type="covariance",
        record_every=1e-3,
    )
    rec_credit = simulate(p_credit)
    return (rec_credit,)


@app.cell
def _(plt, rec_credit):
    fig_c, axes_c = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    axes_c[0].plot(rec_credit["t"], rec_credit["w1"], label="w₁ (target)", color="tab:orange", lw=1.0)
    axes_c[0].plot(rec_credit["t"], rec_credit["w2"], label="w₂ (distractor)", color="tab:blue", lw=1.0)
    axes_c[0].set_ylabel("weight")
    axes_c[0].set_title("Spatial credit assignment: only w₁ should grow")
    axes_c[0].legend(loc="upper right", fontsize=8)

    axes_c[1].plot(rec_credit["t"], rec_credit["E1"], label="E₁", color="tab:orange", lw=0.5, alpha=0.7)
    axes_c[1].plot(rec_credit["t"], rec_credit["E2"], label="E₂", color="tab:blue", lw=0.5, alpha=0.7)
    axes_c[1].axhline(0, ls=":", color="gray", alpha=0.4)
    axes_c[1].set_ylabel("eligibility")
    axes_c[1].legend(loc="upper right", fontsize=8)

    axes_c[2].plot(rec_credit["t"], rec_credit["M"], color="tab:red", lw=0.5)
    axes_c[2].axhline(0, ls=":", color="gray", alpha=0.4)
    axes_c[2].set_xlabel("Time (s)")
    axes_c[2].set_ylabel("M")
    axes_c[2].set_title("Global modulator (same for both synapses)")

    fig_c.tight_layout()
    fig_c
    return


if __name__ == "__main__":
    app.run()
