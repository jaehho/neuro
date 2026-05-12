"""Zoom-adaptive plotly viewer served over HTTP.

``serve_zoom(parquet_path, params)`` opens a stacked-panel figure of the
recorded variables and re-decimates from disk on every pan/zoom event,
which is how multi-million-point parquet runs stay responsive. The
matplotlib / static-HTML paths are gone — read the parquet directly
with polars if you want to make your own figure.
"""
from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from neuro.io import load_time_series_frame
from neuro.params import Params


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


def _plotly_values(values) -> list[float | None]:
    arr = np.asarray(values)
    if arr.size == 0:
        return []
    if arr.dtype.kind == "f":
        return [None if not np.isfinite(value) else float(value) for value in arr]
    return arr.tolist()


def _build_figure(
    path: str | Path,
    p: Params,
    max_points: int,
    x0: float | None,
    x1: float | None,
    variables: list[str] | None,
):
    apv = all_plot_variables(p.n_pre)
    vtitles = variable_titles(p.n_pre)

    plot_vars = variables if variables is not None else apv
    show_spikes = variables is None
    n_rows = len(plot_vars) + (1 if show_spikes else 0)

    needed = set(plot_vars)
    columns = ["t"] + [v for v in apv if v in needed]

    frame, spikes = load_time_series_frame(
        path, columns=columns, max_points=max_points, x0=x0, x1=x1,
    )
    arrays = {col: _plotly_values(frame[col].to_numpy()) for col in columns}
    t = arrays["t"]

    titles = []
    if show_spikes:
        titles.append("Spike times")
    titles.extend(vtitles.get(v, v) for v in plot_vars)

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, subplot_titles=titles,
    )

    row = 1
    if show_spikes:
        entries = [(f"pre{i+1}", f"pre{i+1}_spike_times", float(p.n_pre - i)) for i in range(p.n_pre)]
        entries.append(("post", "post_spike_times", 0.0))
        for label, key, y_base in entries:
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
                go.Scattergl(x=_plotly_values(xs), y=_plotly_values(ys),
                             mode="lines", name=label, line={"width": 1}),
                row=row, col=1,
            )
        row += 1

    for var in plot_vars:
        fig.add_trace(go.Scattergl(x=t, y=arrays[var], name=var, mode="markers", marker={"size": 2}),
                      row=row, col=1)
        if var == "V":
            fig.add_hline(y=p.theta, line_dash="dash", row=row, col=1)  # type: ignore[arg-type]
            fig.add_hline(y=p.V_reset, line_dash="dot", row=row, col=1)  # type: ignore[arg-type]
        elif var == "r_post":
            fig.add_hline(y=p.r_target, line_dash="dash", row=row, col=1)  # type: ignore[arg-type]
        row += 1

    fig.update_layout(height=max(400, 200 * n_rows), width=1400,
                      title="Neuromodulated STDP simulation", showlegend=True)
    fig.update_xaxes(title_text="time (s)", row=n_rows, col=1)
    if x0 is not None and x1 is not None:
        fig.update_xaxes(range=[x0, x1], row=n_rows, col=1)
    return fig


def _plotly_js_path() -> Path:
    candidates = [
        Path(pio.__file__).resolve().parents[1] / "package_data" / "plotly.min.js",
        Path(go.__file__).resolve().parents[1] / "package_data" / "plotly.min.js",
        Path.cwd() / "plotly.min.js",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not locate Plotly JS asset. Checked: " + ", ".join(str(c) for c in candidates)
    )


def serve_zoom(
    parquet_path: str | Path,
    p: Params,
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    max_points: int = 40_000,
    variables: list[str] | None = None,
):
    """Serve a stacked-panel figure of the recorded run on http://host:port/.

    Re-decimates the parquet on every pan/zoom event, so figures stay
    responsive on multi-million-point recordings. Blocks until Ctrl-C.
    """
    source = Path(parquet_path)
    if not source.exists():
        raise FileNotFoundError(f"Parquet does not exist: {source}")
    js_path = _plotly_js_path()

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>neuro</title>
<style>html,body,#plot {{ height:100%; width:100%; margin:0; background:#fff; }}</style>
</head><body><div id="plot"></div>
<script src="/plotly.min.js"></script>
<script>
  const BASE_MAX_POINTS = {max_points};
  const plotDiv = document.getElementById("plot");
  let isUpdating = false, pendingTimer = null, attached = false;

  function rangeFromEvent(e) {{
    if ("xaxis.range[0]" in e && "xaxis.range[1]" in e)
      return [Number(e["xaxis.range[0]"]), Number(e["xaxis.range[1]"])];
    if ("xaxis12.range[0]" in e && "xaxis12.range[1]" in e)
      return [Number(e["xaxis12.range[0]"]), Number(e["xaxis12.range[1]"])];
    if ("xaxis.autorange" in e || "xaxis12.autorange" in e) return null;
    return undefined;
  }}
  function attach() {{
    if (attached || typeof plotDiv.on !== "function") return;
    plotDiv.on("plotly_relayout", e => {{
      if (isUpdating) return;
      const r = rangeFromEvent(e);
      if (r === undefined) return;
      clearTimeout(pendingTimer);
      pendingTimer = setTimeout(() => update(r), 120);
    }});
    attached = true;
  }}
  async function fetchFig(range) {{
    const params = new URLSearchParams();
    params.set("max_points", String(BASE_MAX_POINTS));
    if (range) {{ params.set("x0", String(range[0])); params.set("x1", String(range[1])); }}
    const r = await fetch("/figure?" + params.toString());
    if (!r.ok) throw new Error("fetch failed: " + r.status);
    return await r.json();
  }}
  async function update(range) {{
    if (isUpdating) return;
    isUpdating = true;
    try {{
      const fig = await fetchFig(range);
      await Plotly.react(plotDiv, fig.data, fig.layout, {{ responsive: true }});
      attach();
    }} finally {{ isUpdating = false; }}
  }}
  update(null);
</script></body></html>
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
                body = js_path.read_bytes()
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
                q = parse_qs(parsed.query)
                x0 = float(q["x0"][0]) if "x0" in q else None
                x1 = float(q["x1"][0]) if "x1" in q else None
                local_max = int(q.get("max_points", [str(max_points)])[0])
                fig = _build_figure(source, p, local_max, x0, x1, variables)
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
