"""Zoom-adaptive Plotly viewer served over HTTP.

``serve_zoom(parquet_path, params)`` opens a stacked-panel figure of the
recorded variables in a browser and re-decimates from disk on every
pan/zoom event, which is how multi-million-point parquet runs stay
responsive. ``serve_zoom_background`` is the same thing but runs on a
daemon thread and returns the URL — used by the TUI when you press enter
on a run row.
"""
from __future__ import annotations

import html as html_module
import json
import socket
import threading
import webbrowser
from dataclasses import asdict
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from neuro.io import load_time_series_frame
from neuro.params import Params, diff_from_defaults


def _params_block(p: Params, highlight: set[str] | None = None) -> str:
    """Collapsible parameters block; field names in `highlight` get a yellow background."""
    items = list(asdict(p).items())
    width = max(len(k) for k, _ in items)
    hi = highlight or set()
    lines = []
    for k, v in items:
        line = html_module.escape(f"{k.ljust(width)} = {v!r}")
        if k in hi:
            line = f'<span style="background:#fff3a0;">{line}</span>'
        lines.append(line)
    body = "\n".join(lines)
    summary = "Parameters" if not hi else f"Parameters ({len(hi)} differ from defaults)"
    return (
        '<details style="font-family:monospace;padding:8px 12px;border-bottom:1px solid #ddd;">'
        f'<summary style="cursor:pointer;font-weight:bold;">{summary}</summary>'
        f'<pre style="margin:8px 0 0 0;">{body}</pre>'
        '</details>'
    )


def all_plot_variables(p: Params) -> list[str]:
    n_pre = p.n_pre
    v = ["V"]
    v += [f"I_s{i+1}" for i in range(n_pre)]
    v += [f"x_pre{i+1}" for i in range(n_pre)]
    v.append("y_post")
    v += [f"E{i+1}" for i in range(n_pre)]
    v.append("r_post")
    # In gated mode M = R directly, so R_bar is unused and M duplicates R.
    v += ["R"] if p.M_rule == "gated" else ["R", "R_bar", "M"]
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
        titles[f"w{i+1}"] = f"Weight w{i+1}{suffix}"
    titles["y_post"] = "STDP post-trace y_post"
    titles["r_post"] = "Post-synaptic firing rate r_post"
    titles["R"] = "Reward R"
    titles["R_bar"] = "Reward baseline R_bar"
    titles["M"] = "Modulation M"
    return titles


def _plotly_values(values) -> list[float | None]:
    """plotly chokes on NaN/inf; convert them to None first.

    Used by static-HTML export paths (e.g. notebooks/trajectories.py). The
    HTTP viewer's hot path skips this and passes numpy arrays straight to
    ``go.Scattergl`` — plotly's orjson serializer (via ``fig.to_json``)
    handles non-finites natively.
    """
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
    apv = all_plot_variables(p)
    vtitles = variable_titles(p.n_pre)

    plot_vars = variables if variables is not None else apv
    show_spikes = variables is None
    n_rows = len(plot_vars) + (1 if show_spikes else 0)

    needed = set(plot_vars)
    columns = ["t"] + [v for v in apv if v in needed]

    frame, spikes = load_time_series_frame(
        path, columns=columns, max_points=max_points, x0=x0, x1=x1,
    )
    arrays = {col: frame[col].to_numpy() for col in columns}
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
                go.Scattergl(x=xs, y=ys, mode="lines", name=label, line={"width": 1}),
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


def _pick_free_port(host: str) -> int:
    """Bind to port 0 to let the OS hand us an unused port, then release it."""
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _build_zoom_handler(source: Path, p: Params, html: str, js_path: Path, max_points: int,
                       variables: list[str] | None):
    # Per-server LRU cache. Revisiting a window (zoom-out to full, re-zooming
    # to a prior range) skips the polars envelope downsample entirely.
    @lru_cache(maxsize=16)
    def _figure_bytes(x0: float | None, x1: float | None, local_max: int) -> bytes:
        fig = _build_figure(source, p, local_max, x0, x1, variables)
        return json.dumps(fig.to_plotly_json()).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
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
                body = _figure_bytes(x0, x1, local_max)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format, *args):  # noqa: A002, ARG002
            return

    return Handler


def _zoom_page_html(p: Params, max_points: int) -> str:
    params_block = _params_block(p, highlight=set(diff_from_defaults(p)))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>neuro</title>
<style>html,body {{ margin:0; background:#fff; }} #plot {{ width:100%; height:calc(100vh - 60px); }}</style>
</head><body>{params_block}<div id="plot"></div>
<script src="/plotly.min.js"></script>
<script>
  const BASE_MAX_POINTS = {max_points};
  const plotDiv = document.getElementById("plot");
  let isUpdating = false, pendingTimer = null, attached = false, ready = false;

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
      if (ready && range !== null) {{
        // Zoomed/panned: only data changes. restyle is much faster than react.
        const xs = fig.data.map(d => d.x);
        const ys = fig.data.map(d => d.y);
        const idx = fig.data.map((_, i) => i);
        await Plotly.restyle(plotDiv, {{ x: xs, y: ys }}, idx);
      }} else {{
        // First load or zoom-out: full rebuild so axes autorange correctly.
        await Plotly.react(plotDiv, fig.data, fig.layout, {{ responsive: true }});
        ready = true;
        attach();
      }}
    }} finally {{ isUpdating = false; }}
  }}
  update(null);
</script></body></html>
"""


def serve_zoom(
    parquet_path: str | Path,
    p: Params,
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    max_points: int = 15_000,
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
    html = _zoom_page_html(p, max_points)
    handler = _build_zoom_handler(source, p, html, js_path, max_points, variables)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving zoom-adaptive plot at http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def serve_zoom_background(
    parquet_path: str | Path,
    p: Params,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    max_points: int = 15_000,
    variables: list[str] | None = None,
    open_browser: bool = True,
) -> str:
    """Launch ``serve_zoom`` on a daemon thread; return the URL.

    ``port=None`` auto-picks a free port (so two TUI ``enter`` presses don't
    collide). Daemon thread means the server dies when the host process
    (typically the TUI) exits.
    """
    source = Path(parquet_path)
    if not source.exists():
        raise FileNotFoundError(f"Parquet does not exist: {source}")
    if port is None:
        port = _pick_free_port(host)
    js_path = _plotly_js_path()
    html = _zoom_page_html(p, max_points)
    handler = _build_zoom_handler(source, p, html, js_path, max_points, variables)
    server = ThreadingHTTPServer((host, port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://{host}:{port}"
    if open_browser:
        webbrowser.open(url)
    return url
