"""Verified path: 1 pre → 1 post LIF, target_rate reward, covariance neuromodulator.

This is the focused, reviewed configuration. Anything more exploratory
(N-pre, biofeedback, contingent, gated/surprise/constant neuromod) lives
in `notebooks/exploratory.py`.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Verified: 1 pre → 1 post, target-rate reward

    Three-factor STDP with one pre-synaptic Poisson source projecting onto
    one post-synaptic LIF neuron. The reward is a self-supervisory
    squared-error against a fixed target rate

    $$R(t) \;=\; -(r_\text{post}(t) - r_\text{target})^2,$$

    the modulator is the covariance/RPE form

    $$M(t) \;=\; R(t) - \bar R(t),$$

    and weights follow the three-factor rule

    $$\dot w \;=\; M(t)\, E(t).$$

    See `docs/main.typ` for the full derivation. References in `docs/references.bib`.
    """)
    return


@app.cell
def _(mo):
    T_slider = mo.ui.slider(start=2.0, stop=60.0, step=2.0, value=10.0, label="duration T (s)")
    rate_slider = mo.ui.slider(start=5.0, stop=80.0, step=5.0, value=20.0, label="r_pre (Hz)")
    target_slider = mo.ui.slider(start=1.0, stop=30.0, step=1.0, value=10.0, label="r_target (Hz)")
    w0_slider = mo.ui.slider(start=0.5, stop=8.0, step=0.5, value=2.0, label="w0")
    seed_slider = mo.ui.slider(start=1, stop=20, step=1, value=1, label="seed")
    mo.vstack([T_slider, rate_slider, target_slider, w0_slider, seed_slider])
    return T_slider, rate_slider, seed_slider, target_slider, w0_slider


@app.cell
def _(T_slider, rate_slider, seed_slider, target_slider, w0_slider):
    from neuro.sim import Params, simulate

    p = Params(
        T=T_slider.value,
        dt=1e-4,
        method="rk4",
        seed=seed_slider.value,
        n_pre=1,
        r_pre_rates=(rate_slider.value,),
        poisson=True,
        w0=(w0_slider.value,),
        reward_signal="target_rate",
        target_func="fixed",
        r_target=target_slider.value,
        neuromod_type="covariance",
        record_every=1e-3,
    )
    rec = simulate(p)
    return Params, p, rec


@app.cell
def _(mo, p, rec):
    n_pre_spikes = len(rec["pre1_spike_times"])
    n_post_spikes = len(rec["post_spike_times"])
    half = p.T / 2
    late_post = rec["post_spike_times"][rec["post_spike_times"] >= half]
    late_rate = len(late_post) / (p.T - half) if p.T > half else 0.0
    mo.md(
        f"""
        **Run summary** — T = {p.T:.1f} s, {n_pre_spikes:,} pre spikes, {n_post_spikes:,} post spikes.

        Late-half post rate: **{late_rate:.2f} Hz** (target {p.r_target:.1f} Hz).
        Final weight: **w₁ = {rec['w1'][-1]:.3f}** (initial {p.w0[0]:.2f}, max {p.wmax:.1f}).
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return (plt,)


@app.cell
def _(rec):
    def _downsample(t, max_points=20_000):
        if len(t) <= max_points:
            return slice(None)
        stride = max(1, len(t) // max_points)
        return slice(None, None, stride)

    sl = _downsample(rec["t"])
    return (sl,)


@app.cell
def _(p, plt, rec, sl):
    fig, axes = plt.subplots(6, 1, figsize=(11, 12), sharex=True)

    # 1. Spike raster
    axes[0].eventplot(
        [rec["pre1_spike_times"], rec["post_spike_times"]],
        lineoffsets=[1, 0],
        linelengths=0.7,
        colors=["tab:blue", "tab:red"],
    )
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["post", "pre"])
    axes[0].set_title("Spike times")

    # 2. Membrane potential
    axes[1].plot(rec["t"][sl], rec["V"][sl], lw=0.5, color="tab:purple")
    axes[1].axhline(p.theta, ls="--", color="gray", alpha=0.5, label="θ")
    axes[1].axhline(p.V_reset, ls=":", color="gray", alpha=0.5, label="V_reset")
    axes[1].set_ylabel("V (mV)")
    axes[1].set_title("Membrane potential")
    axes[1].legend(loc="upper right", fontsize=8)

    # 3. Weight (the main learning curve)
    axes[2].plot(rec["t"][sl], rec["w1"][sl], lw=1.0, color="tab:orange")
    axes[2].axhline(p.w0[0], ls=":", color="gray", alpha=0.6, label=f"w0 = {p.w0[0]}")
    axes[2].set_ylabel("w₁")
    axes[2].set_title("Synaptic weight")
    axes[2].legend(loc="upper right", fontsize=8)

    # 4. Eligibility trace
    axes[3].plot(rec["t"][sl], rec["E1"][sl], lw=0.5, color="tab:green")
    axes[3].axhline(0, ls=":", color="gray", alpha=0.4)
    axes[3].set_ylabel("E₁")
    axes[3].set_title("Eligibility trace")

    # 5. Pre / post rates
    axes[4].plot(rec["t"][sl], rec["r_pre1"][sl], lw=0.6, label="r_pre", color="tab:blue")
    axes[4].plot(rec["t"][sl], rec["r_post"][sl], lw=0.6, label="r_post", color="tab:red")
    axes[4].axhline(p.r_target, ls="--", color="black", alpha=0.6, label=f"target ({p.r_target:.0f} Hz)")
    axes[4].set_ylabel("Hz")
    axes[4].set_title("Firing rates (exp trace)")
    axes[4].legend(loc="upper right", fontsize=8)

    # 6. Reward / baseline / modulator
    axes[5].plot(rec["t"][sl], rec["R"][sl], lw=0.5, label="R", color="tab:blue", alpha=0.7)
    axes[5].plot(rec["t"][sl], rec["R_bar"][sl], lw=0.8, label="R̄", color="tab:orange")
    axes[5].plot(rec["t"][sl], rec["M"][sl], lw=0.5, label="M = R − R̄", color="tab:red", alpha=0.7)
    axes[5].axhline(0, ls=":", color="gray", alpha=0.4)
    axes[5].set_xlabel("Time (s)")
    axes[5].set_ylabel("signal")
    axes[5].set_title("Reward, baseline, modulator")
    axes[5].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Long runs

    For runs longer than a few minutes of simulated time, switch to
    parquet streaming + cached lookup so we don't OOM and don't
    re-simulate identical configurations.
    """)
    return


@app.cell
def _(mo):
    long_run_button = mo.ui.run_button(label="run long simulation (60 s, parquet)")
    long_run_button
    return (long_run_button,)


@app.cell
def _(
    Params,
    long_run_button,
    mo,
    rate_slider,
    seed_slider,
    target_slider,
    w0_slider,
):
    mo.stop(not long_run_button.value, mo.md("_Press the button above to run._"))

    from neuro.cache import cached_simulate

    p_long = Params(
        T=60.0,
        dt=1e-4,
        method="rk4",
        seed=seed_slider.value,
        n_pre=1,
        r_pre_rates=(rate_slider.value,),
        poisson=True,
        w0=(w0_slider.value,),
        reward_signal="target_rate",
        target_func="fixed",
        r_target=target_slider.value,
        neuromod_type="covariance",
        record_every=1e-3,
    )
    rec_long = cached_simulate(p_long)
    mo.md(f"Long run cached → `{rec_long['parquet_path']}`")
    return p_long, rec_long


@app.cell
def _(mo, rec_long):
    import polars as pl

    df_long = pl.read_parquet(rec_long["parquet_path"], columns=["t", "w1", "r_post", "R", "R_bar", "M"])
    spikes_long = pl.read_parquet(rec_long["parquet_spikes_path"])
    mo.md(f"Loaded {df_long.height:,} rows; {spikes_long.height:,} spike events.")
    return (df_long,)


@app.cell
def _(df_long, p_long, plt):
    # Subsample for plotting; matplotlib chokes on >1M points.
    max_pts = 20_000
    stride_l = max(1, df_long.height // max_pts)
    df_plot = df_long.gather_every(stride_l)

    fig_l, axes_l = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes_l[0].plot(df_plot["t"], df_plot["w1"], lw=0.6, color="tab:orange")
    axes_l[0].set_ylabel("w₁")
    axes_l[0].set_title(f"Long run: weight trajectory (T = {p_long.T:.0f} s)")

    axes_l[1].plot(df_plot["t"], df_plot["r_post"], lw=0.5, color="tab:red")
    axes_l[1].axhline(p_long.r_target, ls="--", color="black", alpha=0.6)
    axes_l[1].set_ylabel("r_post (Hz)")
    axes_l[1].set_title("Post-synaptic firing rate")

    axes_l[2].plot(df_plot["t"], df_plot["M"], lw=0.4, color="tab:red", alpha=0.7, label="M")
    axes_l[2].plot(df_plot["t"], df_plot["R_bar"], lw=0.8, color="tab:orange", label="R̄")
    axes_l[2].axhline(0, ls=":", color="gray", alpha=0.4)
    axes_l[2].set_xlabel("Time (s)")
    axes_l[2].set_ylabel("signal")
    axes_l[2].legend(loc="upper right", fontsize=8)
    fig_l.tight_layout()
    fig_l
    return


if __name__ == "__main__":
    app.run()
