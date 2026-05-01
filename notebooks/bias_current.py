"""DC bias I_ext to the post LIF neuron: rescue for neuron death.

Varies from `baseline.py` by: `I_ext` (0, sub-/near-/supra-threshold).
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
    # Bias current $I_\text{ext}$: rescuing a dying neuron

    In the $r_\text{pre} \times r_\text{target}$ sweep, most non-converged
    cells fail the same way: $w_1 \to 0$, $V$ settles at $E_L$, post stops
    spiking, and learning halts. Once the neuron is silent the eligibility
    trace decays, $M(t)$ can no longer drive $w$ up, and the system is
    stuck.

    Fix: add a constant DC current to the post neuron,

    $$
    \tau_m\,\dot V \;=\; -(V - E_L) \;+\; R_m\Big(\textstyle\sum_i w_i I_{s,i} \;+\; I_\text{ext}\Big).
    $$

    Without synaptic input the membrane settles at
    $V_\infty = E_L + R_m\,I_\text{ext}$. With the defaults
    $E_L = -65$ mV, $\theta = -50$ mV, $R_m = 50$ M$\Omega$, the
    spontaneous-spike threshold is

    $$I_\text{ext}^\star \;=\; \frac{\theta - E_L}{R_m} \;=\; \frac{15\,\text{mV}}{50\,\text{M}\Omega} \;=\; 0.3\ \text{nA}.$$

    Below $I_\text{ext}^\star$ the bias is **sub-threshold** (no spontaneous
    spikes, just holds $V$ nearer to $\theta$); above it the neuron fires
    tonically on its own.
    """)
    return


@app.cell
def _(mo):
    T_slider = mo.ui.slider(start=5.0, stop=40.0, step=5.0, value=20.0, label="duration T (s)", include_input=True)
    rate_slider = mo.ui.slider(start=5.0, stop=80.0, step=5.0, value=55.0, label="r_pre (Hz)", include_input=True)
    target_slider = mo.ui.slider(start=1.0, stop=30.0, step=1.0, value=2.0, label="r_target (Hz)", include_input=True)
    w0_slider = mo.ui.slider(start=0.5, stop=8.0, step=0.5, value=2.0, label="w0", include_input=True)
    seed_slider = mo.ui.slider(start=1, stop=20, step=1, value=1, label="seed", include_input=True)
    mo.md(
        """
        **Failure case defaults:** `r_pre=55, r_target=2` is one of the
        cells from `output/sweeps/ee8ff7d48d2d` (r_pre x r_target) where
        w_1 collapses to 0 without bias.
        """
    )
    mo.vstack([T_slider, rate_slider, target_slider, w0_slider, seed_slider])
    return T_slider, rate_slider, seed_slider, target_slider, w0_slider


@app.cell
def _(
    Params,
    T_slider,
    rate_slider,
    seed_slider,
    simulate,
    target_slider,
    w0_slider,
):
    # Four bias levels spanning 0, sub-threshold, near-threshold, supra-threshold.
    # Spontaneous-spike threshold at defaults is I_ext* = 0.30 nA.
    I_EXT_VALUES = (0.20, 0.24, 0.25, 0.26)

    runs_bias = {}
    for _Iext in I_EXT_VALUES:
        _p = Params(
            T=T_slider.value,
            dt=1e-4,
            method="rk4",
            seed=seed_slider.value,
            n_pre=1,
            r_pre_rates=(rate_slider.value,),
            poisson=False,
            w0=(w0_slider.value,),
            I_ext=_Iext,
            reward_signal="target_rate",
            target_func="fixed",
            r_target=target_slider.value,
            neuromod_type="covariance",
            rate_mode="window",
            rate_window=0.5,
            record_every=1e-3,
        )
        runs_bias[_Iext] = (_p, simulate(_p))
    return (runs_bias,)


@app.cell
def _(mo, pl, runs_bias):
    _rows = []
    for _Iext, (_p, _rec) in runs_bias.items():
        _half = _p.T / 2
        _late = _rec["post_spike_times"][_rec["post_spike_times"] >= _half]
        _rlate = len(_late) / (_p.T - _half) if _p.T > _half else 0.0
        _rows.append({
            "I_ext (nA)": float(_Iext),
            "post spikes": int(len(_rec["post_spike_times"])),
            "late r_post (Hz)": float(_rlate),
            "w_1 final": float(_rec["w1"][-1]),
            "converged": bool(abs(_rlate - _p.r_target) < 1.0 / _p.rate_window),
        })
    summary_df = pl.DataFrame(_rows)
    _p0 = next(iter(runs_bias.values()))[0]
    mo.vstack([
        mo.md(
            f"**Summary.** T = {_p0.T:.0f} s, r_pre = {_p0.r_pre_rates[0]:.0f} Hz, "
            f"r_target = {_p0.r_target:.1f} Hz, w0 = {_p0.w0[0]:.2f}."
        ),
        summary_df,
    ])
    return


@app.cell
def _(go, make_subplots, runs_bias):
    _palette = ["#cc2936", "#eb8a3f", "#2a9d8f", "#1d4e89"]
    _MAX_POINTS = 4000  # per trace; 4 runs x 3 line panels x 4000 ~ under 5 MB JSON

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=(
            "Membrane potential V",
            "Synaptic weight w_1",
            "Post-synaptic firing rate (sliding window)",
            "Post-synaptic spike raster",
        ),
    )

    _p_any = next(iter(runs_bias.values()))[0]

    for _k, (_Iext, (_p, _rec)) in enumerate(runs_bias.items()):
        _color = _palette[_k % len(_palette)]
        _label = f"I_ext = {_Iext:.2f} nA"
        _t_full = _rec["t"]
        _stride = max(1, len(_t_full) // _MAX_POINTS)
        _t = _t_full[::_stride]
        fig.add_trace(
            go.Scattergl(x=_t, y=_rec["V"][::_stride], mode="markers",
                         marker=dict(size=2, color=_color),
                         name=_label, legendgroup=_label),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scattergl(x=_t, y=_rec["w1"][::_stride], mode="markers",
                         marker=dict(size=2, color=_color),
                         name=_label, legendgroup=_label, showlegend=False),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scattergl(x=_t, y=_rec["r_post"][::_stride], mode="markers",
                         marker=dict(size=2, color=_color),
                         name=_label, legendgroup=_label, showlegend=False),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=_rec["post_spike_times"],
                       y=[_k] * len(_rec["post_spike_times"]),
                       mode="markers",
                       marker=dict(symbol="line-ns-open", size=10, color=_color),
                       name=_label, legendgroup=_label, showlegend=False,
                       hovertemplate=f"{_label}  %{{x:.3f}}s<extra></extra>"),
            row=4, col=1,
        )

    fig.add_hline(y=_p_any.theta, line=dict(dash="dash", color="gray", width=1),
                  annotation_text="θ", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=_p_any.E_L, line=dict(dash="dot", color="gray", width=1),
                  annotation_text="E_L", annotation_position="bottom right", row=1, col=1)
    fig.add_hline(y=_p_any.r_target, line=dict(dash="dash", color="black", width=1),
                  annotation_text=f"target ({_p_any.r_target:.1f} Hz)",
                  annotation_position="top right", row=3, col=1)

    fig.update_yaxes(title_text="V (mV)", row=1, col=1)
    fig.update_yaxes(title_text="w_1", row=2, col=1)
    fig.update_yaxes(title_text="Hz", row=3, col=1)
    fig.update_yaxes(
        tickvals=list(range(len(runs_bias))),
        ticktext=[f"{k:.2f}" for k in runs_bias.keys()],
        title_text="I_ext (nA)", row=4, col=1,
    )
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_layout(
        height=900, hovermode="x unified",
        margin=dict(t=40, l=70, r=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Sweep: how much bias is enough?

    Fix the failure case (r_pre, r_target from above) and scan
    $I_\text{ext}$ on a denser grid. Plot late-half $r_\text{post}$
    and final $w_1$ vs $I_\text{ext}$. The dashed vertical line is
    $I_\text{ext}^\star = 0.30$ nA (tonic-firing onset).
    """)
    return


@app.cell
def _(mo):
    sweep_button = mo.ui.run_button(label="run I_ext sweep")
    n_ext_slider = mo.ui.slider(start=5, stop=25, step=1, value=13, label="grid points", include_input=True)
    I_ext_max_slider = mo.ui.slider(start=0.1, stop=0.6, step=0.05, value=0.5, label="I_ext max (nA)", include_input=True)
    mo.hstack([sweep_button, n_ext_slider, I_ext_max_slider], justify="start")
    return I_ext_max_slider, n_ext_slider, sweep_button


@app.cell
def _(
    I_ext_max_slider,
    Params,
    T_slider,
    mo,
    n_ext_slider,
    np,
    rate_slider,
    seed_slider,
    simulate,
    sweep_button,
    target_slider,
    w0_slider,
):
    mo.stop(
        not sweep_button.value,
        mo.md("_Press **run I_ext sweep** to compute the rescue curve._"),
    )

    I_grid = np.linspace(0.0, float(I_ext_max_slider.value), int(n_ext_slider.value))
    rpost_grid = np.zeros_like(I_grid)
    wfinal_grid = np.zeros_like(I_grid)
    n_post_grid = np.zeros_like(I_grid)

    _T = float(T_slider.value)
    _half = _T / 2

    def _inner(it):
        return mo.status.progress_bar(it, title="run", remove_on_exit=True, show_eta=False)

    for _idx, _Iext in enumerate(
        mo.status.progress_bar(I_grid, title="I_ext sweep", subtitle=f"{len(I_grid)} runs")
    ):
        _p = Params(
            T=_T, dt=1e-4, method="rk4",
            seed=int(seed_slider.value),
            n_pre=1, r_pre_rates=(float(rate_slider.value),), poisson=False,
            w0=(float(w0_slider.value),),
            I_ext=float(_Iext),
            reward_signal="target_rate",
            target_func="fixed", r_target=float(target_slider.value),
            neuromod_type="covariance",
            rate_mode="window", rate_window=0.5,
            record_every=1e-3,
        )
        _rec = simulate(_p, progress=_inner)
        _late = _rec["post_spike_times"][_rec["post_spike_times"] >= _half]
        rpost_grid[_idx] = len(_late) / (_T - _half)
        wfinal_grid[_idx] = float(_rec["w1"][-1])
        n_post_grid[_idx] = len(_rec["post_spike_times"])
    return I_grid, n_post_grid, rpost_grid, wfinal_grid


@app.cell
def _(
    I_grid,
    go,
    make_subplots,
    n_post_grid,
    rpost_grid,
    target_slider,
    wfinal_grid,
):
    fig_s = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(
            "Late-half r_post vs I_ext",
            "Final w_1 vs I_ext",
            "Total post spikes vs I_ext",
        ),
    )
    fig_s.add_trace(
        go.Scatter(x=I_grid, y=rpost_grid, mode="markers",
                   marker=dict(size=7, color="#2a9d8f"),
                   name="r_post (late)", showlegend=False),
        row=1, col=1,
    )
    fig_s.add_hline(y=float(target_slider.value), line=dict(dash="dash", color="black", width=1),
                    annotation_text=f"target ({target_slider.value:.1f} Hz)",
                    annotation_position="top right", row=1, col=1)
    fig_s.add_trace(
        go.Scatter(x=I_grid, y=wfinal_grid, mode="markers",
                   marker=dict(size=7, color="#eb8a3f"),
                   name="w_1", showlegend=False),
        row=2, col=1,
    )
    fig_s.add_trace(
        go.Scatter(x=I_grid, y=n_post_grid, mode="markers",
                   marker=dict(size=7, color="#1d4e89"),
                   name="post spikes", showlegend=False),
        row=3, col=1,
    )
    for _row in (1, 2, 3):
        fig_s.add_vline(x=0.30, line=dict(dash="dot", color="gray", width=1),
                        annotation_text="I_ext* = 0.30", annotation_position="top",
                        row=_row, col=1)
    fig_s.update_yaxes(title_text="Hz", row=1, col=1)
    fig_s.update_yaxes(title_text="w_1 final", row=2, col=1)
    fig_s.update_yaxes(title_text="count", row=3, col=1)
    fig_s.update_xaxes(title_text="I_ext (nA)", row=3, col=1)
    fig_s.update_layout(
        height=650,
        margin=dict(t=40, l=70, r=20, b=40),
    )
    fig_s
    return


if __name__ == "__main__":
    app.run()
