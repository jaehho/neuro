"""Interactive sweep viewer: clickable heatmap + per-cell zoom plots.

    uv run python scripts/serve_sweep.py

Edit ``SWEEP`` below. The script picks the most recent
``output/<SWEEP>/<YYYYMMDD_HHMMSS>/`` directory and serves:

  - ``GET /``                       the heatmap (same data as summary.png)
  - ``GET /cell/iJ_jJ/``            the cell's zoom-adaptive viewer
  - ``GET /cell/iJ_jJ/figure?x0=…`` figure JSON for pan/zoom

Clicking a cell on the heatmap opens that cell's viewer in a new tab.
Ctrl-C tears the server down.
"""
from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import plotly.graph_objects as go
import polars as pl

from neuro import Run, load_latest
from neuro.params import Params
from neuro.plotting import _build_figure, _params_block, _plotly_js_path

# ── Edit me ──────────────────────────────────────────────────────────
SWEEP = "low-rate-sweep"
HOST = "127.0.0.1"
PORT = 8050
MAX_POINTS_DEFAULT = 40_000
# ─────────────────────────────────────────────────────────────────────


def _latest_sweep_run(sweep_dir: Path) -> Path:
    if not sweep_dir.is_dir():
        raise FileNotFoundError(
            f"{sweep_dir} does not exist — run the sweep first, e.g.\n"
            f"    uv run python experiments/{SWEEP.replace('-', '_')}.py"
        )
    candidates = sorted(d for d in sweep_dir.iterdir()
                        if d.is_dir() and (d / "summary.parquet").exists())
    if not candidates:
        raise FileNotFoundError(f"No <ts>/summary.parquet under {sweep_dir}")
    return candidates[-1]


def _heatmap_figure(summary: pl.DataFrame, sweep_label: str) -> go.Figure:
    rows = summary.sort(["i", "j"])
    x_grid = sorted(set(rows["r_pre"].to_list()))
    y_grid = sorted(set(rows["r_target"].to_list()))
    nx, ny = len(x_grid), len(y_grid)
    z: list[list[float | None]] = [[None] * nx for _ in range(ny)]
    cd: list[list[str]] = [[""] * nx for _ in range(ny)]
    for r in rows.iter_rows(named=True):
        i, j = int(r["i"]), int(r["j"])
        z[i][j] = float(r["abs_error"])
        cd[i][j] = f"{i:02d}_{j:02d}"

    fig = go.Figure(go.Heatmap(
        x=x_grid, y=y_grid, z=z, customdata=cd,
        colorscale="Viridis",
        colorbar={"title": "|⟨r_post⟩ − target| (Hz)"},
        hovertemplate=("r_pre=%{x:.2f} Hz  r_target=%{y:.2f} Hz"
                       "<br>|err|=%{z:.2f} Hz<extra></extra>"),
    ))
    fig.update_layout(
        title=f"{sweep_label} ({nx}×{ny})",
        xaxis_title="r_pre (Hz)", yaxis_title="r_target (Hz)",
        width=760, height=660, margin={"l": 60, "r": 40, "t": 50, "b": 50},
    )
    return fig


def _heatmap_html(fig: go.Figure) -> str:
    fig_json = json.dumps(fig.to_plotly_json())
    return f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>{SWEEP} sweep</title>
<style>
  html,body{{margin:0;background:#fff;font-family:sans-serif;}}
  #plot{{width:100vw;height:100vh;}}
  #hint{{position:fixed;left:10px;bottom:8px;color:#666;font-size:12px;}}
</style></head><body>
<div id="plot"></div>
<div id="hint">click a cell to open its run in a new tab</div>
<script src="/plotly.min.js"></script>
<script>
  const fig = {fig_json};
  const plotDiv = document.getElementById("plot");
  Plotly.newPlot(plotDiv, fig.data, fig.layout, {{responsive: true}}).then(() => {{
    plotDiv.on("plotly_click", e => {{
      const cd = e.points && e.points[0] && e.points[0].customdata;
      if (cd) window.open(`/cell/${{cd}}/`, "_blank");
    }});
  }});
</script></body></html>
"""


def _cell_html(cell_key: str, p: Params) -> str:
    params_block = _params_block(p)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>{cell_key}</title>
<style>html,body{{margin:0;background:#fff;}}#plot{{width:100%;height:calc(100vh - 60px);}}</style>
</head><body>{params_block}<div id="plot"></div>
<script src="/plotly.min.js"></script>
<script>
  const BASE_MAX_POINTS = {MAX_POINTS_DEFAULT};
  const FIGURE_URL = "/cell/{cell_key}/figure";
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
    const r = await fetch(FIGURE_URL + "?" + params.toString());
    if (!r.ok) throw new Error("fetch failed: " + r.status);
    return await r.json();
  }}
  async function update(range) {{
    if (isUpdating) return;
    isUpdating = true;
    try {{
      const fig = await fetchFig(range);
      await Plotly.react(plotDiv, fig.data, fig.layout, {{responsive: true}});
      attach();
    }} finally {{ isUpdating = false; }}
  }}
  update(null);
</script></body></html>
"""


def main() -> None:
    sweep_run_dir = _latest_sweep_run(Path("output") / SWEEP)
    sweep_ts = sweep_run_dir.name
    summary = pl.read_parquet(sweep_run_dir / "summary.parquet")
    sweep_label = f"{SWEEP} {sweep_ts}"
    heatmap_page = _heatmap_html(_heatmap_figure(summary, sweep_label)).encode("utf-8")
    js_bytes = _plotly_js_path().read_bytes()

    cell_prefix = re.compile(r"^/cell/(\d{2})_(\d{2})")
    # Cache loaded runs so pan/zoom doesn't re-read the sidecar each event.
    cell_cache: dict[tuple[int, int], Run] = {}

    def _load_cell(i: int, j: int) -> Run:
        key = (i, j)
        if key not in cell_cache:
            cell_cache[key] = load_latest(f"{SWEEP}/{sweep_ts}/cell_{i:02d}_{j:02d}")
        return cell_cache[key]

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path in {"/", "/index.html"}:
                self._send(heatmap_page, "text/html; charset=utf-8")
                return
            if path == "/plotly.min.js":
                self._send(js_bytes, "application/javascript; charset=utf-8")
                return
            if path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            m = cell_prefix.match(path)
            if m:
                i, j = int(m.group(1)), int(m.group(2))
                tail = path[m.end():]
                try:
                    run = _load_cell(i, j)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND,
                                    f"no run for cell_{i:02d}_{j:02d}")
                    return
                if tail in ("/figure", "/figure/"):
                    q = parse_qs(parsed.query)
                    x0 = float(q["x0"][0]) if "x0" in q else None
                    x1 = float(q["x1"][0]) if "x1" in q else None
                    mx = int(q.get("max_points", [str(MAX_POINTS_DEFAULT)])[0])
                    fig = _build_figure(run.parquet, run.params, mx, x0, x1, None)
                    self._send(json.dumps(fig.to_plotly_json()).encode("utf-8"),
                               "application/json; charset=utf-8")
                    return
                if tail in ("", "/"):
                    cell_key = f"{i:02d}_{j:02d}"
                    self._send(_cell_html(cell_key, run.params).encode("utf-8"),
                               "text/html; charset=utf-8")
                    return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format, *args):  # noqa: A002, ARG002
            return

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"sweep:  {sweep_run_dir}")
    print(f"viewer: http://{HOST}:{PORT}/")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
