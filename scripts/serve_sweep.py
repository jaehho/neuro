"""Interactive sweep viewer: clickable heatmap + per-cell zoom plots.

    uv run python scripts/serve_sweep.py                            # list sweeps
    uv run python scripts/serve_sweep.py <sweep>                    # serve latest
    uv run python scripts/serve_sweep.py output/<sweep>/<ts>        # serve that ts

The script picks the most recent ``output/<sweep>/<YYYYMMDD_HHMMSS>/``
directory (or the one passed as an argument) and serves:

  - ``GET /``                          the heatmap (same data as summary.png)
  - ``GET /cell/iJ_jJ/``               the cell's zoom-adaptive viewer
  - ``GET /cell/iJ_jJ/figure?x0=…``    figure JSON for pan/zoom
  - ``GET /cell/iJ_jJ/scaffold``       JSON {code: "..."} for the
                                       "Generate experiment" modal

Clicking a cell on the heatmap opens that cell's viewer in a new tab.
Ctrl-C tears the server down.
"""
from __future__ import annotations

import json
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import plotly.graph_objects as go
import polars as pl

from neuro import Run, load_latest
from neuro.params import Params, diff_from_defaults
from neuro.plotting import _build_figure, _params_block, _plotly_js_path

HOST = "127.0.0.1"
PORT = 8050
MAX_POINTS_DEFAULT = 40_000
OUTPUT_DIR = Path("output")


def _list_sweeps(output_dir: Path) -> list[tuple[str, Path]]:
    """All ``(sweep_name, latest_ts_dir)`` under ``output_dir``, newest first."""
    found: list[tuple[str, Path]] = []
    if not output_dir.is_dir():
        return found
    for sweep_dir in sorted(output_dir.iterdir()):
        if not sweep_dir.is_dir():
            continue
        ts_dirs = sorted(d for d in sweep_dir.iterdir()
                         if d.is_dir() and (d / "summary.parquet").exists())
        if ts_dirs:
            found.append((sweep_dir.name, ts_dirs[-1]))
    found.sort(key=lambda nt: nt[1].name, reverse=True)
    return found


def _latest_sweep_run(sweep_dir: Path) -> Path:
    if not sweep_dir.is_dir():
        raise FileNotFoundError(
            f"{sweep_dir} does not exist — run the sweep first, e.g.\n"
            f"    uv run python experiments/{sweep_dir.name.replace('-', '_')}.py"
        )
    candidates = sorted(d for d in sweep_dir.iterdir()
                        if d.is_dir() and (d / "summary.parquet").exists())
    if not candidates:
        raise FileNotFoundError(f"No <ts>/summary.parquet under {sweep_dir}")
    return candidates[-1]


def _resolve_sweep_run(arg: str) -> Path:
    """Map a CLI arg to a sweep run directory.

    Accepts a sweep name (``low-rate-sweep``) or a path to a specific
    timestamped directory (``output/low-rate-sweep/20260513_110517``).
    """
    p = Path(arg)
    if p.is_dir() and (p / "summary.parquet").exists():
        return p
    return _latest_sweep_run(OUTPUT_DIR / arg)


def _scaffold_code(p: Params, cell_key: str, sweep_name: str, sweep_ts: str) -> str:
    diff = diff_from_defaults(p)
    if not diff:
        params_call = "p = Params()"
    else:
        kw_lines = "\n".join(f"    {k}={v!r}," for k, v in diff.items())
        params_call = f"p = Params(\n{kw_lines}\n)"
    return (
        f'"""Scratch experiment from {sweep_name}/{sweep_ts}/cell_{cell_key}."""\n'
        "from __future__ import annotations\n"
        "\n"
        "from neuro import Params, simulate\n"
        "\n"
        f"{params_call}\n"
        "\n"
        'if __name__ == "__main__":\n'
        '    run = simulate(p, name="scratch")\n'
        '    print(f"  parquet: {run.parquet}")\n'
        '    print(f"  duration: {run.duration_s:.1f}s, rows: {run.rows_written}")\n'
        "    run.serve()\n"
    )


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


def _heatmap_html(fig: go.Figure, sweep: str) -> str:
    fig_json = json.dumps(fig.to_plotly_json())
    return f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>{sweep} sweep</title>
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
    params_block = _params_block(p, highlight=set(diff_from_defaults(p)))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>{cell_key}</title>
<style>
  html,body{{margin:0;background:#fff;font-family:sans-serif;}}
  #plot{{width:100%;height:calc(100vh - 60px);}}
  #gen-btn{{position:fixed;top:10px;right:16px;padding:6px 12px;
           font-size:13px;cursor:pointer;background:#f3f3f3;
           border:1px solid #ccc;border-radius:3px;z-index:10;}}
  #gen-btn:hover{{background:#e6e6e6;}}
  #modal{{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.45);
         z-index:100;align-items:center;justify-content:center;}}
  #modal.open{{display:flex;}}
  #modal-content{{background:#fff;width:min(720px,90vw);max-height:80vh;
                 display:flex;flex-direction:column;border-radius:4px;
                 box-shadow:0 4px 20px rgba(0,0,0,0.25);}}
  #modal-head{{display:flex;justify-content:space-between;align-items:center;
              padding:10px 14px;border-bottom:1px solid #e6e6e6;}}
  #modal-head h3{{margin:0;font-size:14px;}}
  #modal-head button{{margin-left:6px;padding:4px 10px;font-size:12px;
                     cursor:pointer;}}
  #code{{margin:0;padding:14px;overflow:auto;font-family:monospace;
        font-size:12px;white-space:pre;background:#fafafa;}}
</style>
</head><body>{params_block}<div id="plot"></div>
<button id="gen-btn">Generate experiment</button>
<div id="modal"><div id="modal-content">
  <div id="modal-head">
    <h3>Scratch experiment for cell {cell_key}</h3>
    <span><button id="copy-btn">Copy</button><button id="close-btn">Close</button></span>
  </div>
  <pre id="code"></pre>
</div></div>
<script src="/plotly.min.js"></script>
<script>
  const BASE_MAX_POINTS = {MAX_POINTS_DEFAULT};
  const FIGURE_URL = "/cell/{cell_key}/figure";
  const SCAFFOLD_URL = "/cell/{cell_key}/scaffold";
  const plotDiv = document.getElementById("plot");
  const modal = document.getElementById("modal");
  const codeEl = document.getElementById("code");
  let isUpdating = false, pendingTimer = null, attached = false;

  document.getElementById("gen-btn").addEventListener("click", async () => {{
    const r = await fetch(SCAFFOLD_URL);
    if (!r.ok) {{ alert("scaffold failed: " + r.status); return; }}
    const data = await r.json();
    codeEl.textContent = data.code;
    modal.classList.add("open");
  }});
  document.getElementById("copy-btn").addEventListener("click", async () => {{
    try {{ await navigator.clipboard.writeText(codeEl.textContent); }}
    catch (e) {{ alert("copy failed: " + e); }}
  }});
  document.getElementById("close-btn").addEventListener("click",
    () => modal.classList.remove("open"));
  modal.addEventListener("click", e => {{
    if (e.target === modal) modal.classList.remove("open");
  }});

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
    args = sys.argv[1:]
    if not args:
        sweeps = _list_sweeps(OUTPUT_DIR)
        if not sweeps:
            print(f"No sweeps under {OUTPUT_DIR}/")
            sys.exit(1)
        print(f"{'sweep':<30} latest run")
        for name, ts_dir in sweeps:
            print(f"{name:<30} {ts_dir}")
        sys.exit(0)

    sweep_run_dir = _resolve_sweep_run(args[0])
    sweep = sweep_run_dir.parent.name
    sweep_ts = sweep_run_dir.name
    summary = pl.read_parquet(sweep_run_dir / "summary.parquet")
    sweep_label = f"{sweep} {sweep_ts}"
    heatmap_page = _heatmap_html(_heatmap_figure(summary, sweep_label), sweep).encode("utf-8")
    js_bytes = _plotly_js_path().read_bytes()

    cell_prefix = re.compile(r"^/cell/(\d{2})_(\d{2})")
    # Cache loaded runs so pan/zoom doesn't re-read the sidecar each event.
    cell_cache: dict[tuple[int, int], Run] = {}

    def _load_cell(i: int, j: int) -> Run:
        key = (i, j)
        if key not in cell_cache:
            cell_cache[key] = load_latest(f"{sweep}/{sweep_ts}/cell_{i:02d}_{j:02d}")
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
                if tail in ("/scaffold", "/scaffold/"):
                    code = _scaffold_code(run.params, f"{i:02d}_{j:02d}",
                                          sweep, sweep_ts)
                    self._send(json.dumps({"code": code}).encode("utf-8"),
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
