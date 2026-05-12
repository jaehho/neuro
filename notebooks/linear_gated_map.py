"""2D convergence map: r_pre x r_target for the linear+gated config.

Marimo notebook for interactive small grids; running as a script
(`python notebooks/linear_gated_map.py`) runs a 32x32 sweep over
[0, 16] Hz on both axes, parallelised across cores, and saves a
heatmap PNG plus a parquet to `output/sweeps/`.

Varies from `linear_reward.py` by: 2D parameter scan over (r_pre,
r_target) instead of a single operating point.
"""

from __future__ import annotations

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


def _run_one(r_pre: float, r_target: float, T: float, w0: float = 2.0, eta: float = 3e-4) -> tuple[float, float]:
    import polars as pl

    from neuro import Params, simulate

    p = Params(
        T=T, dt=1e-4, seed=1,
        n_pre=1, r_pre=(r_pre,), poisson=False,
        w0=(w0,), eta_plus=eta, eta_minus=eta,
        R_rule="target_rate_linear",
        M_rule="gated",
        r_target=r_target,
        rate_mode="window", rate_window=0.5,
        record_every=1e-3,
    )
    run = simulate(p, name=f"linear-gated-map/cell_rpre{r_pre:g}_rtarget{r_target:g}", progress=iter)
    post = pl.read_parquet(run.spikes).filter(pl.col("spike_type") == "post")["t"].to_numpy()
    half = p.T / 2
    late = post[post >= half]
    r_late = len(late) / (p.T - half) if p.T > half else 0.0
    w_final = float(pl.read_parquet(run.parquet, columns=["w1"])["w1"][-1])
    return r_late, w_final


def _worker(args: tuple[int, int, float, float, float]) -> tuple[int, int, float, float]:
    i, j, r_pre, r_target, T = args
    r_late, w_final = _run_one(r_pre, r_target, T)
    return i, j, r_late, w_final


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

    return go, make_subplots, np, pl


@app.cell
def _(mo):
    mo.md(r"""
    # 2D convergence map: r_pre x r_target (linear + gated)

    Sweeps $r_\text{pre}$ on x and $r_\text{target}$ on y for the
    linear-reward / gated-modulator config. Each cell colour is the
    late-half measured $r_\text{post}$ (left) or its absolute distance
    from target (right). A perfect tracker has $r_\text{post} = r_\text{target}$,
    so the left panel ideally shows horizontal bands and the right
    panel ideally goes dark.

    Two failure modes set the limits:

    - **r_pre too low** — the post can't reach threshold, no spikes,
      no eligibility, learning never starts (left edge stays flat at 0).
    - **r_target above the firing-rate ceiling** — the post saturates
      somewhere near $r_\text{pre}$, can't go higher (upper-right
      triangle saturates).

    Run this file as a script for the full 32x32 sweep:

    ```
    uv run python notebooks/linear_gated_map.py
    ```
    """)
    return


@app.cell
def _(mo):
    n_grid = mo.ui.slider(start=4, stop=20, step=2, value=8, label="grid size n", include_input=True)
    max_rate = mo.ui.slider(start=4.0, stop=40.0, step=2.0, value=16.0, label="max rate (Hz)", include_input=True)
    T_slider = mo.ui.slider(start=5.0, stop=30.0, step=1.0, value=15.0, label="T per cell (s)", include_input=True)
    run_button = mo.ui.run_button(label="run grid sweep")
    mo.vstack([n_grid, max_rate, T_slider, run_button])
    return T_slider, max_rate, n_grid, run_button


@app.cell
def _(T_slider, max_rate, mo, n_grid, np, run_button):
    mo.stop(not run_button.value, mo.md("_Press the button to run the grid sweep._"))

    n = int(n_grid.value)
    pre_grid = np.linspace(0.0, float(max_rate.value), n)
    target_grid = np.linspace(0.0, float(max_rate.value), n)
    rate_grid = np.full((n, n), np.nan)
    w_grid = np.full((n, n), np.nan)

    pairs = [(i, j) for i in range(n) for j in range(n)]
    for i, j in mo.status.progress_bar(pairs, title=f"{n}x{n} sweep", subtitle=f"{n*n} runs"):
        r, w = _run_one(float(pre_grid[j]), float(target_grid[i]), float(T_slider.value))
        rate_grid[i, j] = r
        w_grid[i, j] = w

    err_grid = np.abs(rate_grid - target_grid[:, None])
    return err_grid, pre_grid, rate_grid, target_grid, w_grid


@app.cell
def _(err_grid, go, make_subplots, max_rate, pre_grid, rate_grid, target_grid):
    _ext = float(max_rate.value)

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.14,
        subplot_titles=("late-half r_post (Hz)", "|r_post - r_target| (Hz)"),
    )
    fig.add_trace(
        go.Heatmap(
            x=pre_grid, y=target_grid, z=rate_grid,
            colorscale="Viridis", colorbar=dict(x=0.45, len=0.9),
            hovertemplate="r_pre=%{x:.1f}<br>r_target=%{y:.1f}<br>r_post=%{z:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=[0, _ext], y=[0, _ext], mode="lines",
                   line=dict(dash="dash", color="white", width=1),
                   showlegend=False, hoverinfo="skip"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=pre_grid, y=target_grid, z=err_grid,
            colorscale="Magma_r", colorbar=dict(x=1.0, len=0.9),
            hovertemplate="r_pre=%{x:.1f}<br>r_target=%{y:.1f}<br>err=%{z:.2f}<extra></extra>",
        ),
        row=1, col=2,
    )
    for _col in (1, 2):
        fig.update_xaxes(title_text="r_pre (Hz)", range=[0, _ext], row=1, col=_col)
        fig.update_yaxes(title_text="r_target (Hz)", range=[0, _ext], scaleanchor=f"x{_col}",
                         scaleratio=1, row=1, col=_col)
    fig.update_layout(height=520, margin=dict(t=50, l=70, r=20, b=50))
    fig
    return


def _run_full_sweep(n: int, max_rate: float, T: float) -> dict:
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    pre_grid = np.linspace(0.0, max_rate, n)
    target_grid = np.linspace(0.0, max_rate, n)
    rate_grid = np.full((n, n), np.nan)
    w_grid = np.full((n, n), np.nan)

    pairs = [
        (i, j, float(pre_grid[j]), float(target_grid[i]), T)
        for i in range(n)
        for j in range(n)
    ]
    with ProcessPoolExecutor() as ex:
        for result in tqdm(ex.map(_worker, pairs, chunksize=4), total=len(pairs), desc=f"{n}x{n}"):
            i, j, r, w = result
            rate_grid[i, j] = r
            w_grid[i, j] = w

    err_grid = np.abs(rate_grid - target_grid[:, None])
    return {
        "pre_grid": pre_grid, "target_grid": target_grid,
        "rate_grid": rate_grid, "err_grid": err_grid, "w_grid": w_grid,
    }


def _save_outputs(result: dict, *, n: int, max_rate: float, T: float, stem: str = "linear_gated_pre_target") -> None:
    from pathlib import Path

    import matplotlib.pyplot as plt
    import polars as pl

    out_dir = Path("output/linear-gated-map")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, rt in enumerate(result["target_grid"]):
        for j, rp in enumerate(result["pre_grid"]):
            rows.append({
                "r_pre": float(rp),
                "r_target": float(rt),
                "r_post_late": float(result["rate_grid"][i, j]),
                "err": float(result["err_grid"][i, j]),
                "w_final": float(result["w_grid"][i, j]),
            })
    pq_path = out_dir / f"{stem}.parquet"
    pl.DataFrame(rows).write_parquet(pq_path)
    print(f"Saved parquet -> {pq_path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    extent = [0, max_rate, 0, max_rate]

    im0 = axes[0].imshow(result["rate_grid"], origin="lower", extent=extent,
                         aspect="equal", cmap="viridis", vmin=0, vmax=max_rate)
    axes[0].plot([0, max_rate], [0, max_rate], "r--", lw=1, alpha=0.7, label="y = x")
    axes[0].set_xlabel("r_pre (Hz)")
    axes[0].set_ylabel("r_target (Hz)")
    axes[0].set_title("late-half measured r_post (Hz)")
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.7)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(result["err_grid"], origin="lower", extent=extent,
                         aspect="equal", cmap="magma_r")
    axes[1].set_xlabel("r_pre (Hz)")
    axes[1].set_ylabel("r_target (Hz)")
    axes[1].set_title("|r_post - r_target| (Hz)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"linear+gated, {n}x{n} grid, T={T:g}s", y=1.00)
    fig.tight_layout()

    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap -> {png_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="2D r_pre x r_target sweep for linear+gated.")
    parser.add_argument("--n", type=int, default=32, help="grid size (default 32)")
    parser.add_argument("--max-rate", type=float, default=16.0, help="max Hz on both axes (default 16)")
    parser.add_argument("--T", type=float, default=15.0, help="seconds per cell (default 15)")
    parser.add_argument("--stem", type=str, default="linear_gated_pre_target", help="output file stem")
    args = parser.parse_args()

    print(f"running {args.n}x{args.n} sweep on [0, {args.max_rate}] Hz, T={args.T}s per cell")
    result = _run_full_sweep(args.n, args.max_rate, args.T)
    _save_outputs(result, n=args.n, max_rate=args.max_rate, T=args.T, stem=args.stem)
