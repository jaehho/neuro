from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None

from main import Params, _plotly_values, load_plot_frame, simulate, write_plotly_html


VAR_NAMES = ["V", "I_s", "x_pre", "y_post", "E", "r_pre", "r_post", "R_bar", "w"]
LABELS = {
    "V": "V (mV)",
    "I_s": "I_s",
    "x_pre": "x_pre",
    "y_post": "y_post",
    "E": "E",
    "r_pre": "r_pre",
    "r_post": "r_post",
    "R_bar": "R_bar",
    "w": "w",
}


def plot_pairwise_trajectories(rec_or_path, output_html: str = "trajectories.html", max_points: int = 30_000):
    if go is None or make_subplots is None:
        raise ImportError("plotly is required for interactive trajectory plots. Install it with `uv add plotly`.")

    columns = ["t", *VAR_NAMES]
    frame, _ = load_plot_frame(rec_or_path, columns=columns, max_points=max_points)
    arrays = {col: frame[col].to_numpy() for col in columns}
    t = arrays["t"]
    norm_t = (t - t.min()) / max(1e-12, (t.max() - t.min()))

    pairs = list(combinations(VAR_NAMES, 2))
    ncols = 6
    nrows = -(-len(pairs) // ncols)
    fig = make_subplots(rows=nrows, cols=ncols, horizontal_spacing=0.03, vertical_spacing=0.06)

    for idx, (xvar, yvar) in enumerate(pairs):
        row = idx // ncols + 1
        col = idx % ncols + 1
        fig.add_trace(
            go.Scattergl(
                x=_plotly_values(arrays[xvar]),
                y=_plotly_values(arrays[yvar]),
                mode="markers",
                marker={
                    "size": 3,
                    "color": _plotly_values(norm_t),
                    "colorscale": "Viridis",
                    "showscale": idx == 0,
                    "colorbar": {"title": "time"},
                },
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=LABELS[xvar], row=row, col=col)
        fig.update_yaxes(title_text=LABELS[yvar], row=row, col=col)

    fig.update_layout(
        height=max(900, nrows * 240),
        width=1800,
        title="Projected trajectories: pairwise state variables",
    )
    return write_plotly_html(fig, output_html)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot pairwise state trajectories with Plotly.")
    parser.add_argument("--hdf5", type=str, default=None, help="Read from an HDF5 recording.")
    parser.add_argument("--parquet", type=str, default=None, help="Read from a Parquet recording.")
    parser.add_argument("--output-html", type=str, default="trajectories.html")
    parser.add_argument("--max-plot-points", type=int, default=30_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    params = Params(record_every=1e-3)
    rec_or_path = args.parquet or args.hdf5 or simulate(params)
    output_html = plot_pairwise_trajectories(
        rec_or_path,
        output_html=args.output_html,
        max_points=args.max_plot_points,
    )
    print(f"Wrote interactive trajectory plot to {output_html}")
