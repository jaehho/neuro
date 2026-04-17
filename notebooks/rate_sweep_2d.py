"""2D sweep over any two of {r_pre, r_target, w0, W}.

Converged iff |r_post - r_target| < 1/W (constant-ISI bound, see
notebooks/rate_estimator.py). Non-swept variables stay at their
scalar values. Threshold varies per cell when W is swept.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import hashlib
    import json
    from datetime import datetime
    from pathlib import Path

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from neuro.sim import Params, simulate

    return (
        Params,
        Path,
        datetime,
        hashlib,
        json,
        mpatches,
        np,
        pl,
        plt,
        simulate,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2D sweep — pick any two axes

    Baseline path (1 pre $\to$ 1 post LIF, target-rate reward,
    covariance neuromod, sliding-window rate estimator of length $W$).

    Pick two of $\{r_\text{pre},\ r_\text{target},\ w_0,\ W\}$ for the
    X and Y axes; the remaining variables stay at their scalar values.
    Declare a cell **converged** when

    $$
    \bigl|\,\bar r_\text{post} - r_\text{target}\,\bigr| \;<\; 1/W,
    $$

    where $\bar r_\text{post}$ is the late-half observed post rate
    (spike count on $[T/2,\,T]$ divided by $T/2$). When $W$ is a swept
    axis the threshold varies per cell.
    """)
    return


@app.cell
def _(mo):
    _choices = [
        "r_pre", "r_target", "w0", "W",
        "eta_plus", "eta_minus", "wmax", "tau_e", "tau_Rbar", "reward_delay",
    ]
    x_var = mo.ui.dropdown(options=_choices, value="r_pre", label="X axis")
    y_var = mo.ui.dropdown(options=_choices, value="r_target", label="Y axis")
    return x_var, y_var


@app.cell
def _(mo):
    T_slider = mo.ui.slider(start=5.0, stop=40.0, step=5.0, value=20.0, label="duration T (s)", include_input=True)
    seed_slider = mo.ui.slider(start=1, stop=20, step=1, value=1, label="seed", include_input=True)
    n_x_slider = mo.ui.slider(start=2, stop=14, step=1, value=8, label="X grid points", include_input=True)
    n_y_slider = mo.ui.slider(start=2, stop=14, step=1, value=8, label="Y grid points", include_input=True)

    # Frozen (scalar) values — used when the variable is not a sweep axis.
    r_pre_val = mo.ui.slider(start=5.0, stop=120.0, step=5.0, value=20.0, label="r_pre (Hz)", include_input=True)
    r_target_val = mo.ui.slider(start=2.0, stop=60.0, step=2.0, value=10.0, label="r_target (Hz)", include_input=True)
    w0_val = mo.ui.slider(start=0.5, stop=8.0, step=0.5, value=2.0, label="w0", include_input=True)
    W_val = mo.ui.slider(start=0.1, stop=2.0, step=0.1, value=0.5, label="rate_window W (s)", include_input=True)
    eta_plus_val = mo.ui.slider(start=0.00001, stop=0.005, step=0.00001, value=0.0001, label="eta_plus", include_input=True)
    eta_minus_val = mo.ui.slider(start=0.00001, stop=0.005, step=0.00001, value=0.0001, label="eta_minus", include_input=True)
    wmax_val = mo.ui.slider(start=1.0, stop=50.0, step=1.0, value=10.0, label="wmax", include_input=True)
    tau_e_val = mo.ui.slider(start=0.05, stop=5.0, step=0.05, value=0.5, label="tau_e (s)", include_input=True)
    tau_Rbar_val = mo.ui.slider(start=0.5, stop=30.0, step=0.5, value=5.0, label="tau_Rbar (s)", include_input=True)
    reward_delay_val = mo.ui.slider(start=0.0, stop=5.0, step=0.1, value=1.0, label="reward_delay (s)", include_input=True)

    # Sweep ranges — used only when the variable IS a sweep axis.
    r_pre_range = mo.ui.range_slider(start=5.0, stop=120.0, step=5.0, value=(5.0, 80.0), label="r_pre range (Hz)", show_value=True)
    r_target_range = mo.ui.range_slider(start=2.0, stop=60.0, step=2.0, value=(2.0, 30.0), label="r_target range (Hz)", show_value=True)
    w0_range = mo.ui.range_slider(start=0.5, stop=8.0, step=0.5, value=(0.5, 8.0), label="w0 range", show_value=True)
    W_range = mo.ui.range_slider(start=0.1, stop=2.0, step=0.1, value=(0.1, 1.0), label="W range (s)", show_value=True)
    eta_plus_range = mo.ui.range_slider(start=0.00001, stop=0.005, step=0.00001, value=(0.00001, 0.001), label="eta_plus range", show_value=True)
    eta_minus_range = mo.ui.range_slider(start=0.00001, stop=0.005, step=0.00001, value=(0.00001, 0.001), label="eta_minus range", show_value=True)
    wmax_range = mo.ui.range_slider(start=1.0, stop=50.0, step=1.0, value=(1.0, 20.0), label="wmax range", show_value=True)
    tau_e_range = mo.ui.range_slider(start=0.05, stop=5.0, step=0.05, value=(0.05, 2.0), label="tau_e range (s)", show_value=True)
    tau_Rbar_range = mo.ui.range_slider(start=0.5, stop=30.0, step=0.5, value=(0.5, 20.0), label="tau_Rbar range (s)", show_value=True)
    reward_delay_range = mo.ui.range_slider(start=0.0, stop=5.0, step=0.1, value=(0.0, 3.0), label="reward_delay range (s)", show_value=True)

    return (
        T_slider,
        W_range,
        W_val,
        eta_minus_range,
        eta_minus_val,
        eta_plus_range,
        eta_plus_val,
        n_x_slider,
        n_y_slider,
        r_pre_range,
        r_pre_val,
        r_target_range,
        r_target_val,
        reward_delay_range,
        reward_delay_val,
        seed_slider,
        tau_Rbar_range,
        tau_Rbar_val,
        tau_e_range,
        tau_e_val,
        w0_range,
        w0_val,
        wmax_range,
        wmax_val,
    )


@app.cell
def _(
    T_slider,
    W_range,
    W_val,
    eta_minus_range,
    eta_minus_val,
    eta_plus_range,
    eta_plus_val,
    mo,
    n_x_slider,
    n_y_slider,
    r_pre_range,
    r_pre_val,
    r_target_range,
    r_target_val,
    reward_delay_range,
    reward_delay_val,
    seed_slider,
    tau_Rbar_range,
    tau_Rbar_val,
    tau_e_range,
    tau_e_val,
    w0_range,
    w0_val,
    wmax_range,
    wmax_val,
    x_var,
    y_var,
):
    run_button = mo.ui.run_button(label="run sweep")
    save_button = mo.ui.run_button(label="save results", kind="success")

    _vals = {
        "r_pre": r_pre_val, "r_target": r_target_val, "w0": w0_val, "W": W_val,
        "eta_plus": eta_plus_val, "eta_minus": eta_minus_val, "wmax": wmax_val,
        "tau_e": tau_e_val, "tau_Rbar": tau_Rbar_val, "reward_delay": reward_delay_val,
    }
    _ranges = {
        "r_pre": r_pre_range, "r_target": r_target_range, "w0": w0_range, "W": W_range,
        "eta_plus": eta_plus_range, "eta_minus": eta_minus_range, "wmax": wmax_range,
        "tau_e": tau_e_range, "tau_Rbar": tau_Rbar_range, "reward_delay": reward_delay_range,
    }
    _order = [
        "r_pre", "r_target", "w0", "W",
        "eta_plus", "eta_minus", "wmax", "tau_e", "tau_Rbar", "reward_delay",
    ]
    _swept = {x_var.value, y_var.value}

    _sweep_sliders = [_ranges[k] for k in _order if k in _swept]
    _frozen_sliders = [_vals[k] for k in _order if k not in _swept]

    _items = [
        mo.hstack([x_var, y_var]),
        mo.hstack([T_slider, seed_slider, n_x_slider, n_y_slider]),
    ]
    if _sweep_sliders:
        _items.append(mo.md("**Sweep ranges** (X and Y axes)"))
        _items.append(mo.hstack(_sweep_sliders))
    if _frozen_sliders:
        _items.append(mo.md("**Frozen values** (non-axis variables)"))
        _items.append(mo.hstack(_frozen_sliders))
    _items.append(mo.hstack([run_button, save_button], justify="start"))

    mo.vstack(_items)
    return run_button, save_button


@app.cell
def _(
    W_range,
    eta_minus_range,
    eta_plus_range,
    n_x_slider,
    n_y_slider,
    np,
    r_pre_range,
    r_target_range,
    reward_delay_range,
    tau_Rbar_range,
    tau_e_range,
    w0_range,
    wmax_range,
    x_var,
    y_var,
):
    _ranges = {
        "r_pre": r_pre_range,
        "r_target": r_target_range,
        "w0": w0_range,
        "W": W_range,
        "eta_plus": eta_plus_range,
        "eta_minus": eta_minus_range,
        "wmax": wmax_range,
        "tau_e": tau_e_range,
        "tau_Rbar": tau_Rbar_range,
        "reward_delay": reward_delay_range,
    }

    def _grid_for(name, n):
        lo, hi = _ranges[name].value
        return np.linspace(float(lo), float(hi), int(n))

    x_grid = _grid_for(x_var.value, n_x_slider.value)
    y_grid = _grid_for(y_var.value, n_y_slider.value)
    return x_grid, y_grid


@app.cell
def _(
    Params,
    T_slider,
    W_val,
    eta_minus_val,
    eta_plus_val,
    mo,
    np,
    r_pre_val,
    r_target_val,
    reward_delay_val,
    run_button,
    seed_slider,
    simulate,
    tau_Rbar_val,
    tau_e_val,
    w0_val,
    wmax_val,
    x_grid,
    x_var,
    y_grid,
    y_var,
):
    mo.stop(
        not run_button.value,
        mo.md("_Press **run sweep** to compute the 2D error surface._"),
    )
    mo.stop(
        x_var.value == y_var.value,
        mo.md(f"_X and Y axes must differ (both set to **{x_var.value}**)._"),
    )

    T_v = float(T_slider.value)
    half = T_v / 2

    frozen = {
        "r_pre": float(r_pre_val.value),
        "r_target": float(r_target_val.value),
        "w0": float(w0_val.value),
        "W": float(W_val.value),
        "eta_plus": float(eta_plus_val.value),
        "eta_minus": float(eta_minus_val.value),
        "wmax": float(wmax_val.value),
        "tau_e": float(tau_e_val.value),
        "tau_Rbar": float(tau_Rbar_val.value),
        "reward_delay": float(reward_delay_val.value),
    }

    n_y = len(y_grid)
    n_x = len(x_grid)
    err_grid = np.full((n_y, n_x), np.nan)
    rate_grid = np.full((n_y, n_x), np.nan)
    w_final_grid = np.full((n_y, n_x), np.nan)
    thresh_grid = np.full((n_y, n_x), np.nan)

    _pairs = [(i, j) for i in range(n_y) for j in range(n_x)]

    def _inner_progress(it):
        return mo.status.progress_bar(
            it, title="run", remove_on_exit=True, show_eta=False
        )

    for i, j in mo.status.progress_bar(
        _pairs, title="2D sweep", subtitle=f"{len(_pairs)} runs"
    ):
        vals = dict(frozen)
        vals[x_var.value] = float(x_grid[j])
        vals[y_var.value] = float(y_grid[i])

        p_s = Params(
            T=T_v,
            dt=1e-4,
            method="rk4",
            seed=int(seed_slider.value),
            n_pre=1,
            r_pre_rates=(vals["r_pre"],),
            poisson=False,
            w0=(vals["w0"],),
            reward_signal="target_rate",
            target_func="fixed",
            r_target=vals["r_target"],
            neuromod_type="covariance",
            rate_mode="window",
            rate_window=vals["W"],
            eta_plus=vals["eta_plus"],
            eta_minus=vals["eta_minus"],
            wmax=vals["wmax"],
            tau_e=vals["tau_e"],
            tau_Rbar=vals["tau_Rbar"],
            reward_delay=vals["reward_delay"],
            record_every=1e-3,
        )
        rec_s = simulate(p_s, progress=_inner_progress)
        post_times = rec_s["post_spike_times"]
        late = post_times[post_times >= half]
        r_late = float(len(late)) / (T_v - half)
        rate_grid[i, j] = r_late
        err_grid[i, j] = abs(r_late - vals["r_target"])
        w_final_grid[i, j] = float(rec_s["w1"][-1])
        thresh_grid[i, j] = 1.0 / vals["W"]

    converged = err_grid < thresh_grid

    manifest = {
        "version": 1,
        "x_var": x_var.value,
        "y_var": y_var.value,
        "x_grid": [float(v) for v in x_grid],
        "y_grid": [float(v) for v in y_grid],
        "T": T_v,
        "dt": 1e-4,
        "method": "rk4",
        "seed": int(seed_slider.value),
        "n_pre": 1,
        "poisson": False,
        "reward_signal": "target_rate",
        "target_func": "fixed",
        "neuromod_type": "covariance",
        "rate_mode": "window",
        "frozen_effective": {
            k: float(v) for k, v in frozen.items()
            if k not in {x_var.value, y_var.value}
        },
    }
    return converged, err_grid, manifest, rate_grid, thresh_grid, w_final_grid


@app.cell
def _(
    converged,
    err_grid,
    mpatches,
    np,
    plt,
    thresh_grid,
    x_grid,
    x_var,
    y_grid,
    y_var,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extend the extent by half a cell on each side so grid values land at cell centers.
    _dx = float(x_grid[1] - x_grid[0])
    _dy = float(y_grid[1] - y_grid[0])
    _extent = [
        float(x_grid[0]) - _dx / 2, float(x_grid[-1]) + _dx / 2,
        float(y_grid[0]) - _dy / 2, float(y_grid[-1]) + _dy / 2,
    ]
    _vmax = max(4.0 * float(np.nanmean(thresh_grid)), float(np.nanmax(err_grid)))

    _im = ax.imshow(err_grid, origin="lower", extent=_extent, aspect="auto",
                    cmap="viridis", vmin=0.0, vmax=_vmax)
    fig.colorbar(_im, ax=ax, label="|r_post − r_target| (Hz)")

    # Grid-aligned green outline on each converged cell.
    for _i, _j in zip(*np.where(converged)):
        ax.add_patch(mpatches.Rectangle(
            (float(x_grid[_j]) - _dx / 2, float(y_grid[_i]) - _dy / 2),
            _dx, _dy,
            fill=False, edgecolor="#00ff88", linewidth=1.8,
        ))

    ax.set_xlabel(x_var.value)
    ax.set_ylabel(y_var.value)

    _tmin = float(np.nanmin(thresh_grid))
    _tmax = float(np.nanmax(thresh_grid))
    if abs(_tmax - _tmin) < 1e-9:
        _tstr = f"1/W = {_tmin:.3g} Hz"
    else:
        _tstr = f"1/W in [{_tmin:.3g}, {_tmax:.3g}] Hz"
    ax.set_title(f"late-half error; green = converged (|err| < {_tstr})")
    fig.tight_layout()
    fig
    return


@app.cell
def _(
    converged,
    err_grid,
    mo,
    pl,
    rate_grid,
    thresh_grid,
    w_final_grid,
    x_grid,
    x_var,
    y_grid,
    y_var,
):
    _nr, _nc = err_grid.shape
    _rows = []
    for _i in range(_nr):
        for _j in range(_nc):
            _rows.append({
                x_var.value: float(x_grid[_j]),
                y_var.value: float(y_grid[_i]),
                "r_post_late": float(rate_grid[_i, _j]),
                "err": float(err_grid[_i, _j]),
                "w_final": float(w_final_grid[_i, _j]),
                "threshold": float(thresh_grid[_i, _j]),
                "converged": bool(converged[_i, _j]),
            })
    df = pl.DataFrame(_rows).sort([y_var.value, x_var.value])
    _n_conv = int(df["converged"].sum())
    mo.md(
        f"""
        **Sweep summary** — {df.height} runs, {_n_conv} converged
        ({100.0 * _n_conv / df.height:.1f}%). Axes: **{x_var.value}** × **{y_var.value}**.
        """
    )
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(Path, datetime, df, hashlib, json, manifest, mo, save_button):
    _out_dir = Path("output/sweeps")
    _canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    _h = hashlib.sha256(_canonical.encode()).hexdigest()[:12]
    _pq_path = _out_dir / f"{_h}.parquet"
    _json_path = _out_dir / f"{_h}.json"

    mo.stop(
        not save_button.value,
        mo.md(
            f"_Press **save results** to write to `output/sweeps/{_h}.*`"
            f" (current sweep hash `{_h}`)._"
        ),
    )

    _out_dir.mkdir(parents=True, exist_ok=True)

    if _pq_path.exists() and _json_path.exists():
        _msg = f"already saved — `{_pq_path}` (hash `{_h}`)"
    else:
        # Atomic write: temp file + rename. rename() on POSIX is atomic
        # for same-filesystem moves, so readers never see a half-written file.
        _pq_tmp = _pq_path.with_suffix(".parquet.tmp")
        df.write_parquet(str(_pq_tmp))
        _pq_tmp.replace(_pq_path)

        _meta = {
            "hash": _h,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "parquet": str(_pq_path),
            **manifest,
        }
        _json_tmp = _json_path.with_suffix(".json.tmp")
        _json_tmp.write_text(json.dumps(_meta, indent=2, sort_keys=True))
        _json_tmp.replace(_json_path)

        _msg = f"saved — `{_pq_path}` + `{_json_path}` (hash `{_h}`)"

    mo.md(f"**Save status** — {_msg}")
    return


if __name__ == "__main__":
    app.run()
