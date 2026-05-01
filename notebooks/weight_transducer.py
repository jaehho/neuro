"""Static weight transducer: r_post(w; r_pre) with plasticity disabled.

Sweeps w over [0, wmax] at several r_pre values to find w_onset (first
post spikes) and w_sat (1:1 relay). Identifies the band where the
plastic rule actually has room to operate.

Varies from baseline.py by: eta_plus, eta_minus (→ 0); w × r_pre sweep.
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
    # Weight transducer: $r_\text{post}(w \mid r_\text{pre})$ with plasticity off

    Disable plasticity ($\eta_\pm = 0$ ⇒ $\dot w = 0$). Sweep $w \in [0, w_\text{max}]$
    at several $r_\text{pre}$ levels and read off the steady-state post rate.

    The shape of this curve sets the bar for the plastic case:

    - **Below $w_\text{onset}$** — single EPSPs don't bring $V$ to $\theta$,
      so $r_\text{post} = 0$. Eligibility cannot grow, plasticity dies.
    - **Above $w_\text{sat}$** — every pre-spike triggers a post-spike,
      so $r_\text{post} = \min(r_\text{pre}, 1/\tau_\text{ref})$
      regardless of $w$. The synapse is a relay; "convergence" of the
      plastic system to $r_\text{post} = r_\text{target}$ inside this band
      is not a learned thing — it's wherever $w$ happened to plateau.
    - **In between** — the true working regime where varying $w$ actually
      changes the output.

    Defaults: $\tau_\text{ref}=3$ ms ⇒ refractory ceiling ≈ 333 Hz; LIF
    constants from `Params` set $w_\text{onset}$ implicitly.
    """)
    return


@app.cell
def _(mo):
    T_slider = mo.ui.slider(start=2.0, stop=20.0, step=1.0, value=4.0, label="duration T (s)", include_input=True)
    w_max_slider = mo.ui.slider(start=1.0, stop=5.0, step=1.0, value=4.0, label="sweep upper w", include_input=True)
    n_w_slider = mo.ui.slider(start=11, stop=51, step=2, value=31, label="# w grid points", include_input=True)
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
    poisson_check = mo.ui.checkbox(value=False, label="Poisson r_pre (else deterministic)")
    seed_slider = mo.ui.slider(start=1, stop=20, step=1, value=1, label="seed", include_input=True)
    mo.md(
        r"""
        Sweep: $w \in [0, w_\text{top}]$, $r_\text{pre} \in \{5, 10, 20, 40, 80\}$ Hz.
        `wmax` in the simulator is set to the slider value so the hard clip
        never kicks in. The absolute ceiling on $r_\text{post}$ is
        $1/\tau_\text{ref} \approx 333$ Hz (refractory cap). Drop `dt`
        if rapid post-spike retriggering at very high $w$ looks
        under-resolved.
        """
    )
    mo.vstack([T_slider, w_max_slider, n_w_slider, dt_dropdown, seed_slider, poisson_check])
    return (
        T_slider,
        dt_dropdown,
        n_w_slider,
        poisson_check,
        seed_slider,
        w_max_slider,
    )


@app.cell
def _(
    Params,
    T_slider,
    dt_dropdown,
    mo,
    n_w_slider,
    np,
    poisson_check,
    seed_slider,
    simulate,
    w_max_slider,
):
    R_PRE_VALUES = (5.0, 10.0, 20.0, 40.0, 80.0)
    W_TOP = float(w_max_slider.value)
    W_GRID = np.linspace(0.0, W_TOP, int(n_w_slider.value))

    grid_tr = [(_rp, float(_w)) for _rp in R_PRE_VALUES for _w in W_GRID]
    rpost_grid = {}
    for _rp, _w in mo.status.progress_bar(
        grid_tr, title="w transducer sweep", subtitle=f"{len(grid_tr)} runs"
    ):
        _p = Params(
            T=float(T_slider.value), dt=float(dt_dropdown.value), method="rk4",
            seed=int(seed_slider.value),
            n_pre=1, r_pre_rates=(float(_rp),),
            poisson=bool(poisson_check.value),
            w0=(float(_w),),
            wmax=W_TOP,
            eta_plus=0.0, eta_minus=0.0,
            reward_signal="constant", R_const=0.0,
            neuromod_type="constant",
            rate_mode="window", rate_window=0.5,
            record_every=1e-3,
        )
        _rec = simulate(_p)
        _half = _p.T / 2
        _late = _rec["post_spike_times"][_rec["post_spike_times"] >= _half]
        rpost_grid[(_rp, _w)] = len(_late) / (_p.T - _half)
    return R_PRE_VALUES, W_GRID, rpost_grid


@app.cell
def _(R_PRE_VALUES, W_GRID, mo, np, pl, rpost_grid):
    _SAT_FRAC = 0.95
    _rows = []
    for _rp in R_PRE_VALUES:
        _r = np.array([rpost_grid[(_rp, float(_w))] for _w in W_GRID])
        _onset_idx = np.argmax(_r > 0.0) if (_r > 0.0).any() else -1
        _w_onset = float(W_GRID[_onset_idx]) if _onset_idx >= 0 else float("nan")
        _sat_mask = _r >= _SAT_FRAC * _rp
        _sat_idx = np.argmax(_sat_mask) if _sat_mask.any() else -1
        _w_sat = float(W_GRID[_sat_idx]) if _sat_idx >= 0 else float("nan")
        _r_max = float(_r.max())
        _rows.append({
            "r_pre (Hz)": float(_rp),
            "w_onset": _w_onset,
            f"w_sat (≥{_SAT_FRAC:.2f}·r_pre)": _w_sat,
            "max r_post (Hz)": _r_max,
            "max ratio r_post/r_pre": _r_max / _rp,
        })
    transducer_summary = pl.DataFrame(_rows)
    mo.vstack([
        mo.md(
            "**Per-`r_pre` summary.** "
            "`w_onset` is the smallest swept `w` with `r_post > 0`. "
            "`w_sat` is the smallest `w` reaching 95% of `r_pre` (1:1 relay)."
        ),
        transducer_summary,
    ])
    return


@app.cell
def _(R_PRE_VALUES, W_GRID, go, np, rpost_grid):
    _palette = ["#08519c", "#3182bd", "#2a9d8f", "#eb8a3f", "#cc2936"]
    fig_t = go.Figure()
    for _i, _rp in enumerate(R_PRE_VALUES):
        _r = [rpost_grid[(_rp, float(_w))] for _w in W_GRID]
        fig_t.add_trace(
            go.Scatter(
                x=W_GRID, y=_r,
                mode="markers+lines",
                marker=dict(size=6, color=_palette[_i % len(_palette)]),
                line=dict(color=_palette[_i % len(_palette)]),
                name=f"r_pre = {_rp:g} Hz",
            )
        )
        fig_t.add_hline(
            y=_rp,
            line=dict(dash="dot", color=_palette[_i % len(_palette)], width=1),
            annotation_text=f"{_rp:g}",
            annotation_position="right",
        )
    _all_r = np.array([rpost_grid[(_rp, float(_w))] for _rp in R_PRE_VALUES for _w in W_GRID])
    _ref_ceil = 1.0 / 0.003
    _ymax = max(float(_all_r.max()) * 1.05, _ref_ceil * 1.05)
    fig_t.add_hline(
        y=_ref_ceil,
        line=dict(dash="dash", color="black", width=1),
        annotation_text=f"1/τ_ref ≈ {_ref_ceil:.0f} Hz",
        annotation_position="top right",
    )
    fig_t.update_xaxes(title_text="weight w (state at w0; plasticity off)")
    fig_t.update_yaxes(title_text="late-half r_post (Hz)", range=[0, _ymax])
    fig_t.update_layout(
        title="Static transducer r_post(w) per r_pre",
        height=520,
        margin=dict(t=60, l=70, r=40, b=50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig_t
    return


@app.cell
def _(R_PRE_VALUES, W_GRID, go, np, rpost_grid):
    _R_post_matrix = np.array([
        [rpost_grid[(_rp, float(_w))] for _w in W_GRID] for _rp in R_PRE_VALUES
    ])
    _ratio = _R_post_matrix / np.array(R_PRE_VALUES)[:, None]
    _zmax = max(float(_ratio.max()) * 1.02, 1.05)
    fig_h = go.Figure(
        data=go.Heatmap(
            x=W_GRID,
            y=list(R_PRE_VALUES),
            z=_ratio,
            colorscale="RdBu_r",
            zmin=0.0, zmid=1.0, zmax=_zmax,
            colorbar=dict(title="r_post / r_pre"),
            hovertemplate="w=%{x:.2f}, r_pre=%{y} Hz<br>r_post/r_pre=%{z:.2f}<extra></extra>",
        )
    )
    fig_h.update_xaxes(title_text="w")
    fig_h.update_yaxes(title_text="r_pre (Hz)", type="log")
    fig_h.update_layout(
        title=f"Relay ratio r_post / r_pre (white = 1:1 relay; red > 1, max {_zmax:.2f})",
        height=380,
        margin=dict(t=60, l=70, r=40, b=50),
    )
    fig_h
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Inspect a single (r_pre, w) point

    Pick any `r_pre` and `w` — not limited to the sweep grid — and
    re-simulate at the current global slider settings (T, dt, seed,
    poisson, wmax). Plasticity is still off, so $E$, $w$, $M$ are flat
    and only the spike train, $V$, $I_s$, and $r_\text{post}$ are
    interesting.
    """)
    return


@app.cell
def _(mo):
    inspect_rp = mo.ui.slider(
        start=0.5, stop=500.0, step=0.5, value=20.0,
        label="r_pre (Hz)", include_input=True,
    )
    inspect_w = mo.ui.slider(
        start=0.0, stop=1000.0, step=0.5, value=10.0,
        label="w", include_input=True,
    )
    mo.vstack([inspect_rp, inspect_w])
    return inspect_rp, inspect_w


@app.cell
def _(
    Params,
    T_slider,
    dt_dropdown,
    inspect_rp,
    inspect_w,
    poisson_check,
    seed_slider,
    simulate,
    w_max_slider,
):
    p_inspect = Params(
        T=float(T_slider.value), dt=float(dt_dropdown.value), method="rk4",
        seed=int(seed_slider.value),
        n_pre=1, r_pre_rates=(float(inspect_rp.value),),
        poisson=bool(poisson_check.value),
        w0=(float(inspect_w.value),),
        wmax=max(float(w_max_slider.value), float(inspect_w.value)),
        eta_plus=0.0, eta_minus=0.0,
        reward_signal="constant", R_const=0.0,
        neuromod_type="constant",
        rate_mode="window", rate_window=0.5,
        record_every=1e-3,
    )
    rec_inspect = simulate(p_inspect)
    return p_inspect, rec_inspect


@app.cell
def _(go, make_subplots, p_inspect, rec_inspect):
    _t = rec_inspect["t"]
    fig_inspect = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=(
            f"Spike times (r_pre = {p_inspect.r_pre_rates[0]:g} Hz, w = {p_inspect.w0[0]:.2f})",
            "Membrane potential V",
            "Synaptic current I_s1",
            "STDP traces (x_pre1, y_post)",
            f"Post firing rate r_post (window {p_inspect.rate_window:.2g} s)",
        ),
    )
    fig_inspect.add_trace(
        go.Scatter(x=rec_inspect["pre1_spike_times"], y=[1.0] * len(rec_inspect["pre1_spike_times"]),
                   mode="markers", marker=dict(symbol="line-ns-open", size=10, color="royalblue"),
                   name="pre", showlegend=False),
        row=1, col=1,
    )
    fig_inspect.add_trace(
        go.Scatter(x=rec_inspect["post_spike_times"], y=[0.0] * len(rec_inspect["post_spike_times"]),
                   mode="markers", marker=dict(symbol="line-ns-open", size=10, color="crimson"),
                   name="post", showlegend=False),
        row=1, col=1,
    )
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=rec_inspect["V"], mode="markers", marker=dict(size=2, color="rebeccapurple"), showlegend=False),
        row=2, col=1,
    )
    fig_inspect.add_hline(y=p_inspect.theta, line=dict(dash="dash", color="gray", width=1),
                          annotation_text="θ", annotation_position="top right", row=2, col=1)
    fig_inspect.add_hline(y=p_inspect.V_reset, line=dict(dash="dot", color="gray", width=1),
                          annotation_text="V_reset", annotation_position="bottom right", row=2, col=1)
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=rec_inspect["I_s1"], mode="markers", marker=dict(size=2, color="teal"), showlegend=False),
        row=3, col=1,
    )
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=rec_inspect["x_pre1"], mode="markers", marker=dict(size=2, color="royalblue"), name="x_pre1"),
        row=4, col=1,
    )
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=rec_inspect["y_post"], mode="markers", marker=dict(size=2, color="crimson"), name="y_post"),
        row=4, col=1,
    )
    fig_inspect.add_trace(
        go.Scattergl(x=_t, y=rec_inspect["r_post"], mode="markers", marker=dict(size=2, color="crimson"), showlegend=False),
        row=5, col=1,
    )
    fig_inspect.add_hline(y=p_inspect.r_pre_rates[0], line=dict(dash="dot", color="royalblue", width=1),
                          annotation_text=f"r_pre ({p_inspect.r_pre_rates[0]:.0f} Hz)",
                          annotation_position="top right", row=5, col=1)
    fig_inspect.add_hline(y=1.0 / p_inspect.tau_ref, line=dict(dash="dash", color="black", width=1),
                          annotation_text=f"1/τ_ref ≈ {1.0 / p_inspect.tau_ref:.0f} Hz",
                          annotation_position="bottom right", row=5, col=1)
    fig_inspect.update_yaxes(tickvals=[0, 1], ticktext=["post", "pre"], row=1, col=1)
    fig_inspect.update_yaxes(title_text="V (mV)", row=2, col=1)
    fig_inspect.update_yaxes(title_text="I_s1", row=3, col=1)
    fig_inspect.update_yaxes(title_text="trace", row=4, col=1)
    fig_inspect.update_yaxes(title_text="Hz", row=5, col=1)
    fig_inspect.update_xaxes(title_text="Time (s)", row=5, col=1)
    fig_inspect.update_layout(
        height=750, hovermode="x unified",
        margin=dict(t=40, l=70, r=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_inspect
    return


if __name__ == "__main__":
    app.run()
