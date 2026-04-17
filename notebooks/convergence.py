"""Success criteria for the baseline target-rate path.

Same paradigm as baseline.py, plus a verdict cell checking rate
tracking, weight stability, and baseline catch-up over t > T/2.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Convergence criteria — baseline path

    For target-rate + covariance, define success as the conjunction of
    three conditions on the late half $t > T/2$:

    $$\text{(1) rate tracking}\qquad \frac{\lvert \bar r_\text{post} - r_\text{target}\rvert}{r_\text{target}} < \varepsilon_r$$

    $$\text{(2) weight stability}\qquad \frac{\sigma_{w_1}}{w_\text{max}} < \varepsilon_w$$

    $$\text{(3) baseline caught up}\qquad \frac{\lvert \overline M\rvert}{\sigma_R} < \varepsilon_m$$

    (1) says $r_\text{post}$ tracks the target. (2) says the weight has
    landed on an equilibrium, not a transient. (3) says $\bar R$ has
    caught up to $R$, so the modulator has settled near zero and
    learning has actually stopped.
    """)
    return


@app.cell
def _(mo):
    T_slider = mo.ui.slider(start=2.0, stop=60.0, step=2.0, value=20.0, label="duration T (s)", show_value=True)
    rate_slider = mo.ui.slider(start=5.0, stop=80.0, step=5.0, value=20.0, label="r_pre (Hz)", show_value=True)
    target_slider = mo.ui.slider(start=1.0, stop=30.0, step=1.0, value=10.0, label="r_target (Hz)", show_value=True)
    w0_slider = mo.ui.slider(start=0.5, stop=8.0, step=0.5, value=2.0, label="w0", show_value=True)
    seed_slider = mo.ui.slider(start=1, stop=20, step=1, value=1, label="seed", show_value=True)
    eps_r_slider = mo.ui.slider(start=0.01, stop=0.5, step=0.01, value=0.10, label="eps_r (rate error)", show_value=True)
    eps_w_slider = mo.ui.slider(start=0.001, stop=0.1, step=0.001, value=0.02, label="eps_w (sigma_w / w_max)", show_value=True)
    eps_m_slider = mo.ui.slider(start=0.01, stop=1.0, step=0.01, value=0.10, label="eps_m (|M_bar| / sigma_R)", show_value=True)
    mo.vstack([T_slider, rate_slider, target_slider, w0_slider, seed_slider, eps_r_slider, eps_w_slider, eps_m_slider])
    return (
        T_slider,
        eps_m_slider,
        eps_r_slider,
        eps_w_slider,
        rate_slider,
        seed_slider,
        target_slider,
        w0_slider,
    )


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
        poisson=False,
        w0=(w0_slider.value,),
        reward_signal="target_rate",
        target_func="fixed",
        r_target=target_slider.value,
        neuromod_type="covariance",
        rate_mode="window",
        rate_window=0.5,
        record_every=1e-3,
    )
    rec = simulate(p)
    return p, rec


@app.cell
def _(eps_m_slider, eps_r_slider, eps_w_slider, mo, p, rec):
    import numpy as np

    t = rec["t"]
    half = p.T / 2
    late = t >= half

    # (1) Rate tracking — use spike count on the late half (observable,
    # independent of rate_window smoothing bias).
    late_post = rec["post_spike_times"][rec["post_spike_times"] >= half]
    r_late = len(late_post) / (p.T - half) if p.T > half else 0.0
    rate_err = abs(r_late - p.r_target) / p.r_target

    # (2) Weight stability — late-half std normalized by the natural
    # weight scale w_max (handles the edge case where w_0 is already
    # at equilibrium and barely moves).
    w = rec["w1"]
    w_std_late = float(np.std(w[late]))
    w_stab = w_std_late / p.wmax

    # (3) Baseline catch-up — M's bias small compared to R's fluctuation
    # scale. At the learning fixed point, M is zero-mean with variance
    # matching R's noise; a nonzero M_bar means Rbar hasn't converged.
    M_arr = rec["M"]
    R_arr = rec["R"]
    M_bar_late = float(np.mean(M_arr[late]))
    R_std_late = float(np.std(R_arr[late]))
    m_ratio = abs(M_bar_late) / max(R_std_late, 1e-12)

    c1 = rate_err < eps_r_slider.value
    c2 = w_stab < eps_w_slider.value
    c3 = m_ratio < eps_m_slider.value
    converged = c1 and c2 and c3

    def _mark(ok: bool) -> str:
        return "yes" if ok else "no"

    verdict = "**CONVERGED**" if converged else "**NOT CONVERGED**"
    mo.md(
        f"""
        **Verdict:** {verdict}

        | criterion | value | threshold | pass |
        |---|---|---|---|
        | (1) rate error | {rate_err:.3f} | {eps_r_slider.value:.2f} | {_mark(c1)} |
        | (2) weight stability | {w_stab:.4f} | {eps_w_slider.value:.3f} | {_mark(c2)} |
        | (3) baseline catch-up | {m_ratio:.3f} | {eps_m_slider.value:.2f} | {_mark(c3)} |

        Late-half post rate: **{r_late:.2f} Hz** (target {p.r_target:.1f} Hz).
        Final weight: **w₁ = {w[-1]:.3f}** (initial {p.w0[0]:.2f}, max {p.wmax:.1f}).
        """
    )
    return half, np


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _(half, p, plt, rec):
    def _downsample(arr, max_points=20_000):
        if len(arr) <= max_points:
            return slice(None)
        stride = max(1, len(arr) // max_points)
        return slice(None, None, stride)

    sl = _downsample(rec["t"])
    t_ds = rec["t"][sl]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t_ds, rec["w1"][sl], lw=1.0, color="tab:orange")
    axes[0].axhline(p.w0[0], ls=":", color="gray", alpha=0.6, label=f"w0 = {p.w0[0]}")
    axes[0].axvspan(half, p.T, color="tab:gray", alpha=0.12, label="late half (scored)")
    axes[0].set_ylabel("w₁")
    axes[0].set_title("Weight — stability scored here")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(t_ds, rec["r_post"][sl], lw=0.5, color="tab:red")
    axes[1].axhline(p.r_target, ls="--", color="black", alpha=0.6, label=f"target ({p.r_target:.0f} Hz)")
    axes[1].axvspan(half, p.T, color="tab:gray", alpha=0.12)
    axes[1].set_ylabel("r_post (Hz)")
    axes[1].set_title("Post-synaptic rate — tracking scored here")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t_ds, rec["M"][sl], lw=0.4, color="tab:red", alpha=0.7, label="M")
    axes[2].axhline(0, ls=":", color="gray", alpha=0.5)
    axes[2].axvspan(half, p.T, color="tab:gray", alpha=0.12)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("M = R − R̄")
    axes[2].set_title("Modulator — should hover around 0 once R̄ catches up")
    axes[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
