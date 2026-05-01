"""Linear (signed) reward in place of squared error.

Varies from `baseline.py` by: reward_signal (target_rate_linear) and the
modulator pairing. Shows the square is bound to covariance; without it,
gated Hebbian is the natural pairing.
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
    # Linear reward (no square)

    Baseline uses

    $$R = -(r_\text{post} - r_\text{target})^2,\qquad M = R - \bar R.$$

    The square is what lets the covariance modulator work: $R \le 0$
    with maximum at the target makes $\bar R$ a meaningful baseline, and
    $M = R - \bar R$ is positive on lucky moments (close to target) and
    negative on unlucky ones — a policy-gradient signal that drives
    $r_\text{post}$ toward target by reducing variance.

    A signed linear reward,

    $$R = r_\text{target} - r_\text{post},$$

    has its sign tell direction directly: $R > 0$ when below target,
    $R < 0$ when above. That pairs with the gated modulator
    $M = R$, where the sign of $M$ pushes $w$ the right way.

    With covariance + linear, $\bar R$ tracks whatever equilibrium
    $r_\text{post}$ happens to settle at, so $M$ has zero mean by
    construction and there is no restoring force toward target.
    """)
    return


@app.cell
def _(mo):
    T_slider = mo.ui.slider(start=5.0, stop=500.0, step=1.0, value=20.0, label="duration T (s)", include_input=True)
    rate_slider = mo.ui.slider(start=5.0, stop=80.0, step=1.0, value=20.0, label="r_pre (Hz)", include_input=True)
    target_slider = mo.ui.slider(start=1.0, stop=40.0, step=1.0, value=10.0, label="r_target (Hz)", include_input=True)
    w0_slider = mo.ui.slider(start=0.5, stop=2.0, step=0.1, value=2.0, label="w0", include_input=True)
    eta_slider = mo.ui.slider(start=1e-5, stop=1e-3, step=1e-5, value=1e-4, label="η", include_input=True)
    seed_slider = mo.ui.slider(start=1, stop=20, step=1, value=1, label="seed", include_input=True)
    mo.vstack([T_slider, rate_slider, target_slider, w0_slider, eta_slider, seed_slider])
    return (
        T_slider,
        eta_slider,
        rate_slider,
        seed_slider,
        target_slider,
        w0_slider,
    )


@app.cell
def _(
    T_slider,
    eta_slider,
    rate_slider,
    seed_slider,
    target_slider,
    w0_slider,
):
    from neuro.sim import Params, simulate

    CONFIGS = {
        # "squared + covariance":  dict(reward_signal="target_rate",        neuromod_type="covariance"),
        # "linear + covariance":   dict(reward_signal="target_rate_linear", neuromod_type="covariance"),
        "linear + gated":        dict(reward_signal="target_rate_linear", neuromod_type="gated"),
    }
    runs = {}
    for _name, _extra in CONFIGS.items():
        _p = Params(
            T=T_slider.value, dt=1e-4, method="rk4", seed=seed_slider.value,
            n_pre=1, r_pre_rates=(rate_slider.value,), poisson=False,
            w0=(w0_slider.value,),
            eta_plus=eta_slider.value, eta_minus=eta_slider.value,
            target_func="fixed", r_target=target_slider.value,
            rate_mode="window", rate_window=0.5,
            record_every=1e-3,
            **_extra,
        )
        runs[_name] = (_p, simulate(_p))
    return (runs,)


@app.cell
def _(mo, runs):
    import numpy as _np

    rows = []
    for _name, (_p, _rec) in runs.items():
        _half = _p.T / 2
        _late_post = _rec["post_spike_times"][_rec["post_spike_times"] >= _half]
        _late_rate = len(_late_post) / (_p.T - _half) if _p.T > _half else 0.0
        _late_mask = _rec["t"] >= _half
        _w_std = float(_np.std(_rec["w1"][_late_mask])) / _p.wmax
        _err = abs(_late_rate - _p.r_target) / _p.r_target
        _ok = "✓" if (_err < 0.10 and _w_std < 0.02) else "✗"
        rows.append(
            f"| {_name} | {_late_rate:.2f} | {_err:.3f} | {_w_std:.4f} | {_rec['w1'][-1]:.3f} | {_ok} |"
        )
    _table = "\n".join(rows)
    mo.md(
        f"""
        **Late-half summary** (target = {next(iter(runs.values()))[0].r_target:.1f} Hz)

        | configuration | r_post (Hz) | rate err | σ_w / w_max | w_final | converged |
        |---|---|---|---|---|---|
        {_table}
        """
    )
    return


@app.cell
def _():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots


@app.cell
def _(go, make_subplots, runs):
    _names = list(runs.keys())
    fig = make_subplots(
        rows=4, cols=len(_names),
        shared_xaxes=True, shared_yaxes="rows",
        vertical_spacing=0.05, horizontal_spacing=0.04,
        column_titles=_names,
        row_titles=("r_post (Hz)", "w₁", "R", "M"),
    )
    _COLORS = {
        # "squared + covariance": "darkorange",
        # "linear + covariance":  "crimson",
        "linear + gated":       "seagreen",
    }
    _target = next(iter(runs.values()))[0].r_target
    _N_PTS = 3000  # per-trace point budget for plotly

    for _col, _name in enumerate(_names, start=1):
        _p, _rec = runs[_name]
        _stride = max(1, len(_rec["t"]) // _N_PTS)
        _t = _rec["t"][::_stride]
        _c = _COLORS[_name]
        for _row, _key in enumerate(("r_post", "w1", "R", "M"), start=1):
            fig.add_trace(
                go.Scattergl(x=_t, y=_rec[_key][::_stride], mode="markers",
                             marker=dict(size=2, color=_c), showlegend=False),
                row=_row, col=_col,
            )
        fig.add_hline(y=_target, line=dict(dash="dash", color="black", width=1),
                      row=1, col=_col)
        fig.add_hline(y=0, line=dict(dash="dot", color="gray", width=1), row=3, col=_col)
        fig.add_hline(y=0, line=dict(dash="dot", color="gray", width=1), row=4, col=_col)
        fig.update_xaxes(title_text="Time (s)", row=4, col=_col)

    fig.update_layout(
        height=900,
        margin=dict(t=60, l=70, r=40, b=40),
    )
    fig
    return


if __name__ == "__main__":
    app.run()
