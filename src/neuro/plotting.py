"""Plot builders: matplotlib summary, plotly summary, zoom-adaptive HTTP server.

The matplotlib path is for quick inspection of in-memory recordings.  The
plotly path produces an interactive HTML; ``serve_zoom_adaptive_plot``
serves the same plot but re-decimates from disk on each pan/zoom event,
which is how multi-million-point parquet runs stay responsive.
"""
from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from neuro.io import load_time_series_frame
from neuro.params import Params


def _plotly_values(values) -> list[float | None]:
    arr = np.asarray(values)
    if arr.size == 0:
        return []
    if arr.dtype.kind == "f":
        return [None if not np.isfinite(value) else float(value) for value in arr]
    return arr.tolist()


def write_plotly_html(fig, output_html: str) -> str:
    pio.write_html(
        fig,
        file=output_html,
        full_html=True,
        include_plotlyjs="directory",  # type: ignore[arg-type]
        auto_open=False,
        validate=True,
    )
    return output_html


def plot_all_in_one_figure_matplotlib(rec, p: Params):
    n = p.n_pre
    t = rec["t"]
    n_panels = 9
    _, axs = plt.subplots(n_panels, 1, figsize=(19, 2 * n_panels), sharex=True)

    spike_data = [rec.get(f"pre{i+1}_spike_times", np.array([])) for i in range(n)] + [rec["post_spike_times"]]
    labels = [f"pre{i+1}" for i in range(n)] + ["post"]
    axs[0].eventplot(spike_data, lineoffsets=list(range(len(labels) - 1, -1, -1)), linelengths=0.8)
    axs[0].set_yticks(list(range(len(labels))))
    axs[0].set_yticklabels(labels[::-1])
    axs[0].set_title("Spike times")

    dot = dict(marker=".", markersize=1, linestyle="none")

    axs[1].plot(t, rec["V"], **dot)
    axs[1].axhline(p.theta, linestyle="--", label="theta")
    axs[1].axhline(p.V_reset, linestyle=":", label="V_reset")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Membrane potential V(t)")

    for i in range(n):
        label = f"E{i+1}" + (" (target)" if i == 0 and n > 1 else "")
        axs[2].plot(t, rec[f"E{i+1}"], label=label, **dot)
    axs[2].legend(loc="upper right")
    axs[2].set_title("Eligibility traces")

    for i in range(n):
        axs[3].plot(t, rec[f"r_pre{i+1}"], label=f"r_pre{i+1}", **dot)
    axs[3].legend(loc="upper right")
    axs[3].set_title("Pre-synaptic firing rates")

    axs[4].plot(t, rec["r_post"], **dot)
    axs[4].set_title("Post-synaptic firing rate r_post(t)")

    axs[5].plot(t, rec["R"], **dot)
    axs[5].set_title("Reward R(t)")

    axs[6].plot(t, rec["R_bar"], **dot)
    axs[6].set_title("Reward baseline R_bar(t)")

    axs[7].plot(t, rec["M"], **dot)
    axs[7].set_title("Modulation M(t)")

    for i in range(n):
        label = f"w{i+1}" + (" (target)" if i == 0 and n > 1 else "")
        axs[8].plot(t, rec[f"w{i+1}"], label=label, **dot)
    axs[8].legend(loc="upper right")
    axs[8].set_title("Synaptic weights")

    axs[-1].set_xlabel("time (s)")

    plt.tight_layout()
    plt.savefig("simulation.png")
    plt.show()


def all_plot_variables(n_pre: int) -> list[str]:
    v = ["V"]
    v += [f"I_s{i+1}" for i in range(n_pre)]
    v += [f"x_pre{i+1}" for i in range(n_pre)]
    v.append("y_post")
    v += [f"E{i+1}" for i in range(n_pre)]
    v += [f"r_pre{i+1}" for i in range(n_pre)]
    v.append("r_post")
    v += ["R", "R_bar", "M"]
    v += [f"w{i+1}" for i in range(n_pre)]
    return v


def variable_titles(n_pre: int) -> dict[str, str]:
    titles: dict[str, str] = {"V": "Membrane potential V"}
    for i in range(n_pre):
        label = "target" if i == 0 else "distractor" if n_pre > 1 else ""
        suffix = f" ({label})" if label else ""
        titles[f"I_s{i+1}"] = f"Synaptic current I_s{i+1}{suffix}"
        titles[f"x_pre{i+1}"] = f"STDP pre-trace x_pre{i+1}"
        titles[f"E{i+1}"] = f"Eligibility trace E{i+1}{suffix}"
        titles[f"r_pre{i+1}"] = f"Pre{i+1} firing rate r_pre{i+1}"
        titles[f"w{i+1}"] = f"Weight w{i+1}{suffix}"
    titles["y_post"] = "STDP post-trace y_post"
    titles["r_post"] = "Post-synaptic firing rate r_post"
    titles["R"] = "Reward R"
    titles["R_bar"] = "Reward baseline R_bar"
    titles["M"] = "Modulation M"
    return titles


ALL_PLOT_VARIABLES = all_plot_variables(2)
VARIABLE_TITLES = variable_titles(2)


def build_all_in_one_plotly_figure(
    rec_or_path,
    p: Params,
    max_points: int = 40_000,
    x0: float | None = None,
    x1: float | None = None,
    variables: list[str] | None = None,
):
    n_pre = p.n_pre
    apv = all_plot_variables(n_pre)
    vtitles = variable_titles(n_pre)

    plot_vars = variables if variables is not None else apv
    show_spikes = variables is None
    n_rows = len(plot_vars) + (1 if show_spikes else 0)

    needed_columns = set(plot_vars)
    if "r_post" in needed_columns:
        needed_columns.add("r_pre1")
    columns = ["t"] + [v for v in apv if v in needed_columns]

    frame, spikes = load_time_series_frame(rec_or_path, columns=columns, max_points=max_points, x0=x0, x1=x1)
    arrays = {col: _plotly_values(frame[col].to_numpy()) for col in columns}
    t = arrays["t"]

    titles = []
    if show_spikes:
        titles.append("Spike times")
    titles.extend(vtitles.get(v, v) for v in plot_vars)

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=titles,
    )

    row = 1
    if show_spikes:
        spike_entries = [(f"pre{i+1}", f"pre{i+1}_spike_times", float(n_pre - i)) for i in range(n_pre)]
        spike_entries.append(("post", "post_spike_times", 0.0))
        for label, key, y_base in spike_entries:
            times = spikes.get(key, np.array([]))
            n_spikes = len(times)
            if n_spikes > 0:
                xs = np.empty(3 * n_spikes)
                ys = np.empty(3 * n_spikes)
                xs[0::3] = times
                xs[1::3] = times
                xs[2::3] = np.nan
                ys[0::3] = y_base - 0.4
                ys[1::3] = y_base + 0.4
                ys[2::3] = np.nan
            else:
                xs = np.array([])
                ys = np.array([])
            fig.add_trace(
                go.Scattergl(
                    x=_plotly_values(xs),
                    y=_plotly_values(ys),
                    mode="lines",
                    name=label,
                    line={"width": 1},
                ),
                row=row,
                col=1,
            )
        row += 1

    for var in plot_vars:
        fig.add_trace(go.Scattergl(x=t, y=arrays[var], name=var, mode="markers", marker={"size": 2}), row=row, col=1)
        if var == "V":
            fig.add_hline(y=p.theta, line_dash="dash", row=row, col=1)  # type: ignore[arg-type]
            fig.add_hline(y=p.V_reset, line_dash="dot", row=row, col=1)  # type: ignore[arg-type]
        elif var == "r_post" and "r_pre1" in needed_columns:
            fig.add_trace(go.Scattergl(x=t, y=_plotly_values(p.alpha * np.asarray(frame["r_pre1"].to_numpy())), name="target"), row=row, col=1)
        row += 1

    fig.update_layout(height=max(400, 200 * n_rows), width=1400, title="Neuromodulated STDP simulation", showlegend=True)
    fig.update_xaxes(title_text="time (s)", row=n_rows, col=1)
    if x0 is not None and x1 is not None:
        fig.update_xaxes(range=[x0, x1], row=n_rows, col=1)
    return fig


def plot_all_in_one_plotly(rec_or_path, p: Params, output_html: str = "simulation.html", max_points: int = 40_000, variables: list[str] | None = None):
    fig = build_all_in_one_plotly_figure(rec_or_path, p, max_points=max_points, variables=variables)
    return write_plotly_html(fig, output_html)


def _plotly_package_js_path() -> Path:
    candidates = [
        Path(pio.__file__).resolve().parents[1] / "package_data" / "plotly.min.js",
        Path(go.__file__).resolve().parents[1] / "package_data" / "plotly.min.js",
        Path.cwd() / "plotly.min.js",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate Plotly JS asset. Checked: {', '.join(str(path) for path in candidates)}")


def serve_zoom_adaptive_plot(source_path: str | Path, p: Params, host: str = "127.0.0.1", port: int = 8050, max_points: int = 40_000, variables: list[str] | None = None):
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Plot source does not exist: {source}")

    plotly_js_path = _plotly_package_js_path()

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Neuromodulated STDP simulation</title>
  <style>
    html, body, #plot {{
      height: 100%;
      width: 100%;
      margin: 0;
      background: #ffffff;
    }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script src="/plotly.min.js"></script>
  <script>
    const BASE_MAX_POINTS = {max_points};
    const plotDiv = document.getElementById("plot");
    let isUpdating = false;
    let pendingTimer = null;
    let relayoutHandlerAttached = false;

    function relayoutRange(eventData) {{
      if ("xaxis.range[0]" in eventData && "xaxis.range[1]" in eventData) {{
        return [Number(eventData["xaxis.range[0]"]), Number(eventData["xaxis.range[1]"])];
      }}
      if ("xaxis12.range[0]" in eventData && "xaxis12.range[1]" in eventData) {{
        return [Number(eventData["xaxis12.range[0]"]), Number(eventData["xaxis12.range[1]"])];
      }}
      if ("xaxis.autorange" in eventData || "xaxis12.autorange" in eventData) {{
        return null;
      }}
      return undefined;
    }}

    function attachRelayoutHandler() {{
      if (relayoutHandlerAttached || typeof plotDiv.on !== "function") {{
        return;
      }}
      plotDiv.on("plotly_relayout", (eventData) => {{
        if (isUpdating) {{
          return;
        }}
        const range = relayoutRange(eventData);
        if (range === undefined) {{
          return;
        }}
        clearTimeout(pendingTimer);
        pendingTimer = setTimeout(() => updateFigure(range), 120);
      }});
      relayoutHandlerAttached = true;
    }}

    async function fetchFigure(range) {{
      const params = new URLSearchParams();
      params.set("max_points", String(BASE_MAX_POINTS));
      if (range) {{
        params.set("x0", String(range[0]));
        params.set("x1", String(range[1]));
      }}
      const response = await fetch("/figure?" + params.toString());
      if (!response.ok) {{
        throw new Error("Failed to fetch figure: " + response.status);
      }}
      return await response.json();
    }}

    async function updateFigure(range) {{
      if (isUpdating) {{
        return;
      }}
      isUpdating = true;
      try {{
        const fig = await fetchFigure(range);
        await Plotly.react(plotDiv, fig.data, fig.layout, {{ responsive: true }});
        attachRelayoutHandler();
      }} finally {{
        isUpdating = false;
      }}
    }}

    updateFigure(null);
  </script>
</body>
</html>
"""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                body = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/plotly.min.js":
                body = plotly_js_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return

            if parsed.path == "/figure":
                query = parse_qs(parsed.query)
                x0 = float(query["x0"][0]) if "x0" in query else None
                x1 = float(query["x1"][0]) if "x1" in query else None
                local_max_points = int(query.get("max_points", [str(max_points)])[0])
                fig = build_all_in_one_plotly_figure(source, p, max_points=local_max_points, x0=x0, x1=x1, variables=variables)
                body = json.dumps(fig.to_plotly_json()).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving zoom-adaptive plot at http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
