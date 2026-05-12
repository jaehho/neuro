from __future__ import annotations

import argparse
from itertools import combinations

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from neuro import Params, simulate
from neuro.io import load_time_series_frame
from neuro.plotting import _plotly_values


VAR_NAMES = ["V", "I_s1", "x_pre1", "y_post", "E1", "r_pre1", "r_post", "R_bar", "w1"]
LABELS = {
    "V": "V (mV)",
    "I_s1": "I_s1",
    "x_pre1": "x_pre1",
    "y_post": "y_post",
    "E1": "E1",
    "r_pre1": "r_pre1",
    "r_post": "r_post",
    "R_bar": "R_bar",
    "w1": "w1",
}


def plot_pairwise_trajectories(parquet_path, output_html: str = "trajectories.html", max_points: int = 30_000):
    columns = ["t", *VAR_NAMES]
    frame, _ = load_time_series_frame(parquet_path, columns=columns, max_points=max_points)
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
    # plotly accepts a string mode ("directory", "cdn", …) here, but the type
    # stubs declare `bool` — cast to silence pyright.
    pio.write_html(fig, file=output_html, full_html=True,
                   include_plotlyjs="directory",  # pyright: ignore[reportArgumentType]
                   auto_open=False, validate=True)
    return output_html


def parse_args():
    parser = argparse.ArgumentParser(description="Plot pairwise state trajectories with Plotly.")
    parser.add_argument("--parquet", type=str, default=None, help="Read from a Parquet recording.")
    parser.add_argument("--output-html", type=str, default="trajectories.html")
    parser.add_argument("--max-plot-points", type=int, default=30_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.parquet is not None:
        parquet_path = args.parquet
    else:
        run = simulate(Params(record_every=1e-3), name="trajectories-scratch")
        parquet_path = str(run.parquet)
    output_html = plot_pairwise_trajectories(
        parquet_path,
        output_html=args.output_html,
        max_points=args.max_plot_points,
    )
    print(f"Wrote interactive trajectory plot to {output_html}")
