"""Baseline path: 1 pre → 1 post LIF, target_rate reward, covariance neuromodulator.

The reviewed reference configuration. Other notebooks in this directory
each vary one axis from this baseline — see their docstrings.
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
    # Baseline: 1 pre → 1 post, target-rate reward

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
    T_slider = mo.ui.slider(start=2.0, stop=60.0, step=2.0, value=10.0, label="duration T (s)", include_input=True)
    rate_slider = mo.ui.slider(start=5.0, stop=80.0, step=1.0, value=20.0, label="r_pre (Hz)", include_input=True)
    target_slider = mo.ui.slider(start=1.0, stop=40.0, step=1.0, value=10.0, label="r_target (Hz)", include_input=True)
    w0_slider = mo.ui.slider(start=0.5, stop=2.0, step=0.1, value=2.0, label="w0", include_input=True)
    seed_slider = mo.ui.slider(start=1, stop=20, step=1, value=1, label="seed", include_input=True)
    plasticity_off = mo.ui.checkbox(value=False, label="disable plasticity (η₊ = η₋ = 0, weights frozen at w0)")
    mo.vstack([T_slider, rate_slider, target_slider, w0_slider, seed_slider, plasticity_off])
    return (
        T_slider,
        plasticity_off,
        rate_slider,
        seed_slider,
        target_slider,
        w0_slider,
    )


@app.cell
def _(
    T_slider,
    plasticity_off,
    rate_slider,
    seed_slider,
    target_slider,
    w0_slider,
):
    from neuro.sim import Params, simulate

    _eta = 0.0 if plasticity_off.value else 1e-4
    p = Params(
        T=T_slider.value,
        dt=1e-4,
        method="rk4",
        seed=seed_slider.value,
        n_pre=1,
        r_pre_rates=(rate_slider.value,),
        poisson=False,
        w0=(w0_slider.value,),
        eta_plus=_eta,
        eta_minus=_eta,
        reward_signal="constant", R_const=0.0,
        target_func="fixed",
        r_target=target_slider.value,
        neuromod_type="constant",
        rate_mode="window",
        rate_window=0.5,
        record_every=1e-3,
    )
        # _p = Params(
        #     T=float(T_slider.value), dt=float(dt_dropdown.value), method="rk4",
        #     seed=int(seed_slider.value),
        #     n_pre=1, r_pre_rates=(float(_rp),),
        #     poisson=bool(poisson_check.value),
        #     w0=(float(_w),),
        #     wmax=W_TOP,
        #     eta_plus=0.0, eta_minus=0.0,
        #     reward_signal="constant", R_const=0.0,
        #     neuromod_type="constant",
        #     rate_mode="window", rate_window=0.5,
        #     record_every=1e-3,
        # )
    rec = simulate(p)
    return Params, p, rec


@app.cell
def _(mo, p, rec):
    _n_pre_spikes = len(rec["pre1_spike_times"])
    _n_post_spikes = len(rec["post_spike_times"])
    _half = p.T / 2
    _late_post = rec["post_spike_times"][rec["post_spike_times"] >= _half]
    _late_rate = len(_late_post) / (p.T - _half) if p.T > _half else 0.0
    mo.md(
        f"""
        **Run summary** — T = {p.T:.1f} s, {_n_pre_spikes:,} pre spikes, {_n_post_spikes:,} post spikes.

        Late-half post rate: **{_late_rate:.2f} Hz** (target {p.r_target:.1f} Hz).
        Final weight: **w₁ = {rec['w1'][-1]:.3f}** (initial {p.w0[0]:.2f}, max {p.wmax:.1f}).
        """
    )
    return


@app.cell
def _(mo, p, rec):
    import numpy as _np

    _half = p.T / 2
    _late_mask = rec["t"] >= _half
    _late_post_spk = rec["post_spike_times"][rec["post_spike_times"] >= _half]
    _r_late = len(_late_post_spk) / (p.T - _half) if p.T > _half else 0.0

    _rate_err = abs(_r_late - p.r_target) / p.r_target
    _w_stab = float(_np.std(rec["w1"][_late_mask])) / p.wmax
    _M_bar_late = float(_np.mean(rec["M"][_late_mask]))
    _R_std_late = float(_np.std(rec["R"][_late_mask]))
    _m_ratio = abs(_M_bar_late) / max(_R_std_late, 1e-12)

    _EPS_R, _EPS_W, _EPS_M = 0.10, 0.02, 0.10
    _c1, _c2, _c3 = _rate_err < _EPS_R, _w_stab < _EPS_W, _m_ratio < _EPS_M
    _verdict = "**CONVERGED**" if (_c1 and _c2 and _c3) else "**NOT CONVERGED**"
    _mark = lambda ok: "yes" if ok else "no"

    mo.md(
        f"""
        **Convergence verdict (late half, t > T/2):** {_verdict}

        | criterion | value | threshold | pass |
        |---|---|---|---|
        | (1) rate tracking, \\|r_post − r_target\\|/r_target | {_rate_err:.3f} | {_EPS_R:.2f} | {_mark(_c1)} |
        | (2) weight stability, σ_w / w_max | {_w_stab:.4f} | {_EPS_W:.3f} | {_mark(_c2)} |
        | (3) baseline catch-up, \\|M̄\\| / σ_R | {_m_ratio:.3f} | {_EPS_M:.2f} | {_mark(_c3)} |
        """
    )
    return


@app.cell
def _():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots


@app.cell
def _(go, make_subplots, p, rec):
    fig = make_subplots(
        rows=10, cols=1, shared_xaxes=True, vertical_spacing=0.022,
        subplot_titles=(
            "Spike times",
            "Membrane potential V",
            "Synaptic current I_s1",
            "STDP traces (x_pre1, y_post)",
            "Eligibility trace E₁",
            "Synaptic weight w₁",
            f"Post firing rate r_post (sliding window, {p.rate_window:.2g} s)",
            "Tracking error |r_post − r_target|",
            "Reward R and baseline R̄",
            "Modulator M = R − R̄",
        ),
    )
    t = rec["t"]

    fig.add_trace(
        go.Scatter(x=rec["pre1_spike_times"], y=[1.0] * len(rec["pre1_spike_times"]),
                   mode="markers", marker=dict(symbol="line-ns-open", size=10, color="royalblue"),
                   name="pre", showlegend=False, hovertemplate="pre %{x:.4f}s<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=rec["post_spike_times"], y=[0.0] * len(rec["post_spike_times"]),
                   mode="markers", marker=dict(symbol="line-ns-open", size=10, color="crimson"),
                   name="post", showlegend=False, hovertemplate="post %{x:.4f}s<extra></extra>"),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scattergl(x=t, y=rec["V"], mode="markers", marker=dict(size=2, color="rebeccapurple"),
                     name="V", showlegend=False),
        row=2, col=1,
    )
    fig.add_hline(y=p.theta, line=dict(dash="dash", color="gray", width=1),
                  annotation_text="θ", annotation_position="top right", row=2, col=1)
    fig.add_hline(y=p.V_reset, line=dict(dash="dot", color="gray", width=1),
                  annotation_text="V_reset", annotation_position="bottom right", row=2, col=1)

    fig.add_trace(
        go.Scattergl(x=t, y=rec["I_s1"], mode="markers", marker=dict(size=2, color="teal"),
                     name="I_s1", showlegend=False),
        row=3, col=1,
    )

    fig.add_trace(
        go.Scattergl(x=t, y=rec["x_pre1"], mode="markers", marker=dict(size=2, color="royalblue"),
                     name="x_pre1"),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scattergl(x=t, y=rec["y_post"], mode="markers", marker=dict(size=2, color="crimson"),
                     name="y_post"),
        row=4, col=1,
    )

    fig.add_trace(
        go.Scattergl(x=t, y=rec["E1"], mode="markers", marker=dict(size=2, color="seagreen"),
                     name="E₁", showlegend=False),
        row=5, col=1,
    )
    fig.add_hline(y=0, line=dict(dash="dot", color="gray", width=1), row=5, col=1)

    fig.add_trace(
        go.Scattergl(x=t, y=rec["w1"], mode="markers", marker=dict(size=2, color="darkorange"),
                     name="w₁", showlegend=False),
        row=6, col=1,
    )
    fig.add_hline(y=p.w0[0], line=dict(dash="dot", color="gray", width=1),
                  annotation_text=f"w0 = {p.w0[0]}", annotation_position="top right", row=6, col=1)

    fig.add_trace(
        go.Scattergl(x=t, y=rec["r_post"], mode="markers", marker=dict(size=2, color="crimson"),
                     name="r_post", showlegend=False),
        row=7, col=1,
    )
    fig.add_hline(y=p.r_target, line=dict(dash="dash", color="black", width=1),
                  annotation_text=f"target ({p.r_target:.0f} Hz)", annotation_position="top right", row=7, col=1)
    fig.add_hline(y=p.r_pre_rates[0], line=dict(dash="dot", color="royalblue", width=1),
                  annotation_text=f"r_pre ({p.r_pre_rates[0]:.0f} Hz)", annotation_position="bottom right", row=7, col=1)

    fig.add_trace(
        go.Scattergl(x=t, y=abs(rec["r_post"] - p.r_target), mode="markers",
                     marker=dict(size=2, color="crimson"),
                     name="|r_post − r_target|", showlegend=False),
        row=8, col=1,
    )
    fig.add_hline(y=1 / p.rate_window, line=dict(dash="dash", color="black", width=1),
                  annotation_text="1/W", annotation_position="top right", row=8, col=1)

    fig.add_trace(
        go.Scattergl(x=t, y=rec["R"], mode="markers", marker=dict(size=2, color="crimson"),
                     opacity=0.6, name="R"),
        row=9, col=1,
    )
    fig.add_trace(
        go.Scattergl(x=t, y=rec["R_bar"], mode="markers", marker=dict(size=2, color="seagreen"),
                     name="R̄"),
        row=9, col=1,
    )

    fig.add_trace(
        go.Scattergl(x=t, y=rec["M"], mode="markers", marker=dict(size=2, color="mediumpurple"),
                     name="M", showlegend=False),
        row=10, col=1,
    )
    fig.add_hline(y=0, line=dict(dash="dot", color="gray", width=1), row=10, col=1)

    fig.update_yaxes(tickvals=[0, 1], ticktext=["post", "pre"], row=1, col=1)
    fig.update_yaxes(title_text="V (mV)", row=2, col=1)
    fig.update_yaxes(title_text="I_s1", row=3, col=1)
    fig.update_yaxes(title_text="trace", row=4, col=1)
    fig.update_yaxes(title_text="E₁", row=5, col=1)
    fig.update_yaxes(title_text="w₁", row=6, col=1)
    fig.update_yaxes(title_text="Hz", row=7, col=1)
    fig.update_yaxes(title_text="Hz", row=8, col=1)
    fig.update_yaxes(title_text="signal", row=9, col=1)
    fig.update_yaxes(title_text="M", row=10, col=1)
    fig.update_xaxes(title_text="Time (s)", row=10, col=1)
    fig.update_layout(
        height=1300, hovermode="x unified",
        margin=dict(t=40, l=70, r=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Long runs

    Notebooks are for analysis and short/medium simulations. For runs of
    many minutes or hours, **don't** drive the sim from here. Run the CLI,
    which streams to parquet and caches the result:

    ```bash
    uv run neuro --plot-backend server
    ```

    The `server` backend launches a zoom-adaptive plotly viewer at
    `http://127.0.0.1:8050` that re-decimates the parquet on every
    scroll-zoom or pan. Marimo can't do that (it captures box-select, not
    zoom, and widgets are immutable once rendered), so the standalone
    viewer is the right tool when you need to drill deep into a big run.

    The cell below is a convenience sanity-check: a 60 s cached run, one
    static plotly figure at full resolution (~60k points). Use it to
    verify the baseline path stays happy as T grows; reach for the CLI
    viewer once you want more.
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
    plasticity_off,
    rate_slider,
    seed_slider,
    target_slider,
    w0_slider,
):
    mo.stop(not long_run_button.value, mo.md("_Press the button above to run._"))

    from neuro.cache import cached_simulate

    _eta_long = 0.0 if plasticity_off.value else 1e-4
    p_long = Params(
        T=60.0,
        dt=1e-4,
        method="rk4",
        seed=seed_slider.value,
        n_pre=1,
        r_pre_rates=(rate_slider.value,),
        poisson=False,
        w0=(w0_slider.value,),
        eta_plus=_eta_long,
        eta_minus=_eta_long,
        reward_signal="target_rate",
        target_func="fixed",
        r_target=target_slider.value,
        neuromod_type="covariance",
        rate_mode="window",
        rate_window=0.5,
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
def _(df_long, go, make_subplots, p_long):
    fig_l = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=(
            f"Weight trajectory (T = {p_long.T:.0f} s, {df_long.height:,} rows)",
            "Post-synaptic firing rate",
            "Reward modulator and running mean",
        ),
    )
    fig_l.add_trace(
        go.Scattergl(x=df_long["t"], y=df_long["w1"], mode="markers",
                     marker=dict(size=2, color="darkorange"), name="w₁", showlegend=False),
        row=1, col=1,
    )
    fig_l.add_trace(
        go.Scattergl(x=df_long["t"], y=df_long["r_post"], mode="markers",
                     marker=dict(size=2, color="crimson"), name="r_post", showlegend=False),
        row=2, col=1,
    )
    fig_l.add_hline(y=p_long.r_target, line=dict(dash="dash", color="black", width=1),
                    annotation_text=f"target ({p_long.r_target:.0f} Hz)",
                    annotation_position="top right", row=2, col=1)
    fig_l.add_trace(
        go.Scattergl(x=df_long["t"], y=df_long["M"], mode="markers",
                     marker=dict(size=2, color="crimson"), opacity=0.7, name="M"),
        row=3, col=1,
    )
    fig_l.add_trace(
        go.Scattergl(x=df_long["t"], y=df_long["R_bar"], mode="markers",
                     marker=dict(size=2, color="darkorange"), name="R̄"),
        row=3, col=1,
    )
    fig_l.add_hline(y=0, line=dict(dash="dot", color="gray", width=1), row=3, col=1)
    fig_l.update_yaxes(title_text="w₁", row=1, col=1)
    fig_l.update_yaxes(title_text="Hz", row=2, col=1)
    fig_l.update_yaxes(title_text="signal", row=3, col=1)
    fig_l.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig_l.update_layout(
        height=700, hovermode="x unified",
        margin=dict(t=40, l=70, r=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_l
    return


if __name__ == "__main__":
    app.run()
