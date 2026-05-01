"""Spatial credit assignment: 2 pre + contingent reward.

Two pre-synaptic neurons fire at the same rate. Reward is delivered only
when pre1 (the "target" synapse) fires within a coincidence window
before a post-spike. Both synapses see the same global modulator M, but
only E₁ is high when reward arrives, so only w₁ should grow.

Varies from `baseline.py` by: n_pre=2 and reward_signal=contingent.
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
    # Spatial credit assignment — 2 pre + contingent reward

    Two pre-synaptic neurons fire at the same rate. Reward is delivered
    only when **pre1** (the "target" synapse) fires within a coincidence
    window before a post-spike. Both synapses see the same global
    modulator $M$, but only $E_1$ is high when reward arrives, so only
    $w_1$ should grow.

    Reference: Izhikevich (2007), Frémaux & Gerstner (2016) Eq. 10.
    """)
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

    axes_c[0].plot(rec_credit["t"], rec_credit["w1"], label="w₁ (target)", color="tab:orange", ls="none", marker=".", ms=1.5)
    axes_c[0].plot(rec_credit["t"], rec_credit["w2"], label="w₂ (distractor)", color="tab:blue", ls="none", marker=".", ms=1.5)
    axes_c[0].set_ylabel("weight")
    axes_c[0].set_title("Spatial credit assignment: only w₁ should grow")
    axes_c[0].legend(loc="upper right", fontsize=8)

    axes_c[1].plot(rec_credit["t"], rec_credit["E1"], label="E₁", color="tab:orange", ls="none", marker=".", ms=1.0, alpha=0.7)
    axes_c[1].plot(rec_credit["t"], rec_credit["E2"], label="E₂", color="tab:blue", ls="none", marker=".", ms=1.0, alpha=0.7)
    axes_c[1].axhline(0, ls=":", color="gray", alpha=0.4)
    axes_c[1].set_ylabel("eligibility")
    axes_c[1].legend(loc="upper right", fontsize=8)

    axes_c[2].plot(rec_credit["t"], rec_credit["M"], color="tab:red", ls="none", marker=".", ms=1.0)
    axes_c[2].axhline(0, ls=":", color="gray", alpha=0.4)
    axes_c[2].set_xlabel("Time (s)")
    axes_c[2].set_ylabel("M")
    axes_c[2].set_title("Global modulator (same for both synapses)")

    fig_c.tight_layout()
    fig_c
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Selectivity vs. control

    The success metric for spatial credit assignment is the weight
    divergence $w_1 - w_2$ over time (Izhikevich 2007, Fig. 4). To check
    that divergence comes from the modulator and not from seed noise or
    pure STDP competition, run the same params with
    `neuromod_type="constant"` (so $M=1$ and $R$ is ignored by the
    weight update) and overlay.
    """)
    return


@app.cell
def _(Params, simulate):
    p_ctrl_credit = Params(
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
        neuromod_type="constant",
        record_every=1e-3,
    )
    rec_ctrl_credit = simulate(p_ctrl_credit)
    return (rec_ctrl_credit,)


@app.cell
def _(plt, rec_credit, rec_ctrl_credit):
    fig_s, ax_s = plt.subplots(1, 1, figsize=(11, 3.5))
    ax_s.plot(rec_credit["t"], rec_credit["w1"] - rec_credit["w2"],
              label="covariance (M = R − R̄)", color="tab:orange", ls="none", marker=".", ms=1.5)
    ax_s.plot(rec_ctrl_credit["t"], rec_ctrl_credit["w1"] - rec_ctrl_credit["w2"],
              label="constant control (M = 1, R ignored)",
              color="gray", ls="none", marker=".", ms=1.5)
    ax_s.axhline(0, ls=":", color="black", alpha=0.4)
    ax_s.set_xlabel("Time (s)")
    ax_s.set_ylabel("w₁ − w₂")
    ax_s.set_title("Selectivity: weight divergence between target and distractor")
    ax_s.legend(loc="upper left", fontsize=9)
    ax_s.grid(True, alpha=0.15)
    fig_s.tight_layout()
    fig_s
    return


if __name__ == "__main__":
    app.run()
