"""Constant reward: R fixed to R_const, decoupled from r_post.

Sweeps R_const for `covariance` and `gated`. Diagnoses whether the
plastic baseline's plateau is real shaping or eligibility-bias drift.

Varies from baseline.py by: reward_signal (→ constant), R_const.
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
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots
    from neuro.sim import Params, simulate

    return Params, go, make_subplots, np, pl, simulate


@app.cell
def _(mo):
    mo.md(r"""
    # Constant reward: feedback decoupled from $r_\text{post}$

    Replace the self-supervisory squared-error reward
    $$R(t) = -(r_\text{post}(t) - r_\text{target})^2$$
    with a fixed scalar $R(t) \equiv R_\text{const}$ and sweep its value.
    Reward no longer carries information about how the post neuron is
    behaving, so any drift in $w$ tells us what the rule does *without*
    feedback.

    What we expect:

    | mode         | $M$           | prediction at constant $R$                            |
    |--------------|---------------|-------------------------------------------------------|
    | `covariance` | $R - \bar R$  | $\bar R$ catches up on $\tau_{\bar R}=5$ s; $M\to 0$, $w$ flat |
    | `gated`      | $R$           | $\dot w = R_\text{const}\cdot \bar E$; monotonic drift |

    The diagnostic for the saturation worry is the **gated** panel:
    if even a tiny $R_\text{const}>0$ runs $w$ to $w_\text{max}$, the
    "convergence" we see in the baseline path is being driven by
    eligibility-trace bias, not by reward shape.
    """)
    return


@app.cell
def _(mo):
    T_slider = mo.ui.slider(start=5.0, stop=60.0, step=5.0, value=20.0, label="duration T (s)", include_input=True)
    rate_slider = mo.ui.slider(start=5.0, stop=80.0, step=1.0, value=20.0, label="r_pre (Hz)", include_input=True)
    w0_slider = mo.ui.slider(start=0.5, stop=50.0, step=0.5, value=2.0, label="w0", include_input=True)
    w_max_slider = mo.ui.slider(start=10.0, stop=500.0, step=10.0, value=100.0, label="wmax", include_input=True)
    dt_dropdown = mo.ui.dropdown(
        options={
            "1e-3 (1 ms)": 1e-3,
            "5e-4 (0.5 ms)": 5e-4,
            "1e-4 (0.1 ms)": 1e-4,
            "5e-5 (0.05 ms)": 5e-5,
            "1e-5 (0.01 ms)": 1e-5,
        },
        value="1e-4 (0.1 ms)",
        label="dt",
    )
    seed_slider = mo.ui.slider(start=1, stop=20, step=1, value=1, label="seed", include_input=True)
    poisson_check = mo.ui.checkbox(value=False, label="Poisson r_pre")
    mo.md(
        r"""
        Default `w0=2.0` matches `baseline.py`. Run **T ≥ 4 · τ_Rbar = 20 s**
        so covariance has time to neutralise. Drop `dt` if rapid
        post-spike retriggering at large `R_const` × gated looks
        under-resolved.
        """
    )
    mo.vstack([T_slider, rate_slider, w0_slider, w_max_slider, dt_dropdown, seed_slider, poisson_check])
    return (
        T_slider,
        dt_dropdown,
        poisson_check,
        rate_slider,
        seed_slider,
        w0_slider,
        w_max_slider,
    )


@app.cell
def _(
    Params,
    T_slider,
    dt_dropdown,
    mo,
    poisson_check,
    rate_slider,
    seed_slider,
    simulate,
    w0_slider,
    w_max_slider,
):
    R_CONST_GRID = (.0, 0.1, 1.0, 10.0, 100.0, 9.0, 1000.0)
    NEUROMOD_TYPES = ("covariance")
    grid_const = [(_nm, _R) for _nm in NEUROMOD_TYPES for _R in R_CONST_GRID]

    runs_const = {}
    for _nm, _R in mo.status.progress_bar(
        grid_const, title="constant-R sweep", subtitle=f"{len(grid_const)} runs"
    ):
        _p = Params(
            T=float(T_slider.value), dt=float(dt_dropdown.value), method="rk4",
            seed=int(seed_slider.value),
            n_pre=1,
            r_pre_rates=(float(rate_slider.value),),
            poisson=bool(poisson_check.value),
            w0=(float(w0_slider.value),),
            wmax=float(w_max_slider.value),
            reward_signal="constant",
            R_const=float(_R),
            neuromod_type=_nm,
            rate_mode="window", rate_window=0.5,
            record_every=1e-3,
        )
        runs_const[(_nm, _R)] = (_p, simulate(_p))
    return NEUROMOD_TYPES, R_CONST_GRID, runs_const


@app.cell
def _(NEUROMOD_TYPES, R_CONST_GRID, mo, np, pl, runs_const):
    _rows = []
    for _nm in NEUROMOD_TYPES:
        for _R in R_CONST_GRID:
            _p, _rec = runs_const[(_nm, _R)]
            _half = _p.T / 2
            _late_mask = _rec["t"] >= _half
            _late_post = _rec["post_spike_times"][_rec["post_spike_times"] >= _half]
            _r_late = len(_late_post) / (_p.T - _half)
            _rows.append({
                "neuromod": _nm,
                "R_const": float(_R),
                "w final": float(_rec["w1"][-1]),
                "Δw": float(_rec["w1"][-1] - _p.w0[0]),
                "late M̄": float(np.mean(_rec["M"][_late_mask])),
                "late R̄": float(np.mean(_rec["R_bar"][_late_mask])),
                "late r_post (Hz)": float(_r_late),
            })
    summary_const = pl.DataFrame(_rows)
    mo.vstack([
        mo.md("**Per-run summary** (late half = $t > T/2$)."),
        summary_const,
    ])
    return


@app.cell
def _(R_CONST_GRID):
    # Diverging palette: cool for negatives, warm for positives, gray for 0.
    _palette_neg = ["#08306b", "#08519c", "#3182bd", "#74a9cf", "#bdd7e7"]
    _palette_pos = ["#fdd0a2", "#fdae6b", "#fd8d3c", "#e6550d", "#7f2704"]
    _zero_color = "#888888"
    _negs = sorted([r for r in R_CONST_GRID if r < 0])  # most negative first
    _poss = sorted([r for r in R_CONST_GRID if r > 0])
    color_map = {0.0: _zero_color}
    for _i, _r in enumerate(_negs):
        color_map[_r] = _palette_neg[min(_i, len(_palette_neg) - 1)]
    for _i, _r in enumerate(_poss):
        color_map[_r] = _palette_pos[min(_i, len(_palette_pos) - 1)]
    return (color_map,)


@app.cell
def _(NEUROMOD_TYPES, R_CONST_GRID, color_map, go, make_subplots, runs_const):
    _MAX_POINTS = 3000
    fig_w = make_subplots(
        rows=len(NEUROMOD_TYPES), cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=tuple(f"w(t) — neuromod = {nm}" for nm in NEUROMOD_TYPES),
    )
    _p_any = next(iter(runs_const.values()))[0]
    for _row, _nm in enumerate(NEUROMOD_TYPES, start=1):
        for _R in R_CONST_GRID:
            _p, _rec = runs_const[(_nm, _R)]
            _t = _rec["t"]
            _stride = max(1, len(_t) // _MAX_POINTS)
            fig_w.add_trace(
                go.Scattergl(
                    x=_t[::_stride], y=_rec["w1"][::_stride],
                    mode="markers",
                    marker=dict(size=2, color=color_map[_R]),
                    name=f"R={_R:g}",
                    legendgroup=f"R={_R:g}",
                    showlegend=(_row == 1),
                ),
                row=_row, col=1,
            )
        fig_w.add_hline(y=_p_any.w0[0], line=dict(dash="dot", color="gray", width=1),
                        annotation_text=f"w0 = {_p_any.w0[0]:.2f}",
                        annotation_position="top right",
                        row=_row, col=1)
        fig_w.add_hline(y=0.0, line=dict(dash="dot", color="gray", width=1), row=_row, col=1)
        fig_w.add_hline(y=_p_any.wmax, line=dict(dash="dot", color="gray", width=1),
                        annotation_text=f"wmax = {_p_any.wmax:.0f}",
                        annotation_position="bottom right",
                        row=_row, col=1)
        fig_w.update_yaxes(title_text="w", row=_row, col=1)
    fig_w.update_xaxes(title_text="Time (s)", row=len(NEUROMOD_TYPES), col=1)
    fig_w.update_layout(
        height=620, hovermode="x unified",
        margin=dict(t=50, l=70, r=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )
    fig_w
    return


@app.cell
def _(NEUROMOD_TYPES, R_CONST_GRID, go, runs_const):
    fig_s = go.Figure()
    _palette_modes = {"covariance": "#2a9d8f", "gated": "#cc2936"}
    _idx = list(range(len(R_CONST_GRID)))
    _labels = [f"{_R:g}" for _R in R_CONST_GRID]
    for _nm in NEUROMOD_TYPES:
        _ws = [float(runs_const[(_nm, _R)][1]["w1"][-1]) for _R in R_CONST_GRID]
        fig_s.add_trace(
            go.Scatter(
                x=_idx, y=_ws,
                mode="markers+lines",
                marker=dict(size=9, color=_palette_modes[_nm]),
                line=dict(color=_palette_modes[_nm], width=2),
                name=_nm,
            )
        )
    _p_any = next(iter(runs_const.values()))[0]
    fig_s.add_hline(
        y=_p_any.w0[0], line=dict(dash="dot", color="gray", width=1),
        annotation_text=f"w0 = {_p_any.w0[0]:.2f}",
        annotation_position="top left",
    )
    fig_s.add_hline(y=0.0, line=dict(dash="dot", color="gray", width=1))
    fig_s.add_hline(
        y=_p_any.wmax, line=dict(dash="dot", color="gray", width=1),
        annotation_text=f"wmax = {_p_any.wmax:.0f}",
        annotation_position="bottom right",
    )
    fig_s.update_xaxes(
        tickvals=_idx, ticktext=_labels,
        title_text="R_const (categorical, sign-ordered)",
    )
    fig_s.update_yaxes(title_text="final w")
    fig_s.update_layout(
        title=f"Final w vs R_const after T = {_p_any.T:.0f} s, r_pre = {_p_any.r_pre_rates[0]:.0f} Hz",
        height=420,
        margin=dict(t=60, l=70, r=20, b=50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig_s
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Inspect a single run

    Pick any `(neuromod_type, R_const)` — not limited to the sweep grid —
    and re-simulate at the current global slider settings (T, r_pre, w0,
    wmax, dt, seed, poisson). The neuromod dropdown spans all four modes
    (`covariance`, `gated`, `surprise`, `constant`), so you can also poke
    at modes the sweep doesn't cover.
    """)
    return


@app.cell
def _(mo):
    inspect_nm = mo.ui.dropdown(
        options=["covariance", "gated", "surprise", "constant"],
        value="covariance",
        label="neuromod_type",
    )
    inspect_R = mo.ui.slider(
        start=-2000.0, stop=2000.0, step=0.1, value=1.0,
        label="R_const", include_input=True,
    )
    mo.vstack([inspect_nm, inspect_R])
    return inspect_R, inspect_nm


@app.cell
def _(
    Params,
    T_slider,
    dt_dropdown,
    go,
    inspect_R,
    inspect_nm,
    make_subplots,
    poisson_check,
    rate_slider,
    seed_slider,
    simulate,
    w0_slider,
    w_max_slider,
):
    _p_i = Params(
        T=float(T_slider.value), dt=float(dt_dropdown.value), method="rk4",
        seed=int(seed_slider.value),
        n_pre=1,
        r_pre_rates=(float(rate_slider.value),),
        poisson=bool(poisson_check.value),
        w0=(float(w0_slider.value),),
        wmax=float(w_max_slider.value),
        reward_signal="constant",
        R_const=float(inspect_R.value),
        neuromod_type=inspect_nm.value,
        rate_mode="window", rate_window=0.5,
        record_every=1e-3,
    )
    _rec_i = simulate(_p_i)
    _t = _rec_i["t"]
    fig_inspect = make_subplots(
        rows=9, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        subplot_titles=(
            f"Spike times (neuromod = {inspect_nm.value}, R_const = {inspect_R.value:g})",
            "Membrane potential V",
            "Synaptic current I_s1",
            "STDP traces (x_pre1, y_post)",
            "Eligibility trace E",
            "Synaptic weight w",
            f"Post firing rate r_post (window {_p_i.rate_window:.2g} s)",
            "Reward R and baseline R̄",
            "Modulator M",
        ),
    )
    fig_inspect.add_trace(
        go.Scatter(x=_rec_i["pre1_spike_times"], y=[1.0] * len(_rec_i["pre1_spike_times"]),
                   mode="markers", marker=dict(symbol="line-ns-open", size=10, color="royalblue"),
                   name="pre", showlegend=False),
        row=1, col=1,
    )
    fig_inspect.add_trace(
        go.Scatter(x=_rec_i["post_spike_times"], y=[0.0] * len(_rec_i["post_spike_times"]),
                   mode="markers", marker=dict(symbol="line-ns-open", size=10, color="crimson"),
                   name="post", showlegend=False),
        row=1, col=1,
    )
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=_rec_i["V"], mode="markers", marker=dict(size=2, color="rebeccapurple"), showlegend=False),
        row=2, col=1,
    )
    fig_inspect.add_hline(y=_p_i.theta, line=dict(dash="dash", color="gray", width=1),
                          annotation_text="θ", annotation_position="top right", row=2, col=1)
    fig_inspect.add_hline(y=_p_i.V_reset, line=dict(dash="dot", color="gray", width=1),
                          annotation_text="V_reset", annotation_position="bottom right", row=2, col=1)
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=_rec_i["I_s1"], mode="markers", marker=dict(size=2, color="teal"), showlegend=False),
        row=3, col=1,
    )
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=_rec_i["x_pre1"], mode="markers", marker=dict(size=2, color="royalblue"), name="x_pre1"),
        row=4, col=1,
    )
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=_rec_i["y_post"], mode="markers", marker=dict(size=2, color="crimson"), name="y_post"),
        row=4, col=1,
    )
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=_rec_i["E1"], mode="markers", marker=dict(size=2, color="seagreen"), showlegend=False),
        row=5, col=1,
    )
    fig_inspect.add_hline(y=0, line=dict(dash="dot", color="gray", width=1), row=5, col=1)
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=_rec_i["w1"], mode="markers", marker=dict(size=2, color="darkorange"), showlegend=False),
        row=6, col=1,
    )
    fig_inspect.add_hline(y=_p_i.w0[0], line=dict(dash="dot", color="gray", width=1),
                          annotation_text=f"w0 = {_p_i.w0[0]:.2f}", annotation_position="top right", row=6, col=1)
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=_rec_i["r_post"], mode="markers", marker=dict(size=2, color="crimson"), showlegend=False),
        row=7, col=1,
    )
    fig_inspect.add_hline(y=_p_i.r_pre_rates[0], line=dict(dash="dot", color="royalblue", width=1),
                          annotation_text=f"r_pre ({_p_i.r_pre_rates[0]:.0f} Hz)",
                          annotation_position="top right", row=7, col=1)
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=_rec_i["R"], mode="markers", marker=dict(size=2, color="crimson"), opacity=0.6, name="R"),
        row=8, col=1,
    )
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=_rec_i["R_bar"], mode="markers", marker=dict(size=2, color="seagreen"), name="R̄"),
        row=8, col=1,
    )
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=_rec_i["M"], mode="markers", marker=dict(size=2, color="mediumpurple"), showlegend=False),
        row=9, col=1,
    )
    fig_inspect.add_hline(y=0, line=dict(dash="dot", color="gray", width=1), row=9, col=1)
    fig_inspect.update_yaxes(tickvals=[0, 1], ticktext=["post", "pre"], row=1, col=1)
    fig_inspect.update_yaxes(title_text="V (mV)", row=2, col=1)
    fig_inspect.update_yaxes(title_text="I_s1", row=3, col=1)
    fig_inspect.update_yaxes(title_text="trace", row=4, col=1)
    fig_inspect.update_yaxes(title_text="E", row=5, col=1)
    fig_inspect.update_yaxes(title_text="w", row=6, col=1)
    fig_inspect.update_yaxes(title_text="Hz", row=7, col=1)
    fig_inspect.update_yaxes(title_text="signal", row=8, col=1)
    fig_inspect.update_yaxes(title_text="M", row=9, col=1)
    fig_inspect.update_xaxes(title_text="Time (s)", row=9, col=1)
    fig_inspect.update_layout(
        height=1200, hovermode="x unified",
        margin=dict(t=40, l=70, r=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_inspect
    return


if __name__ == "__main__":
    app.run()
