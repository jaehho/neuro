#!/usr/bin/env python3
"""Build output/sweeps/index.html: a matrix contact sheet of all 2D sweep PNGs.

Reads every output/sweeps/*.json and renders a single HTML page with a
(y_var, x_var) matrix of thumbnails linking through to the full PNGs.
Re-run after a new sweep to refresh. Images use loading="lazy" so a page
with many pairs stays cheap to open.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

SWEEP_DIR = Path("output/sweeps")
INDEX = SWEEP_DIR / "index.html"

VAR_ORDER = [
    "r_pre", "r_target", "w0", "W",
    "eta_plus", "eta_minus", "wmax",
    "tau_e", "tau_Rbar", "reward_delay",
]


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def main() -> None:
    manifests = [json.loads(p.read_text()) for p in sorted(SWEEP_DIR.glob("*.json"))]
    if not manifests:
        raise SystemExit(f"no sweep JSONs in {SWEEP_DIR}")

    by_pair = {(m["x_var"], m["y_var"]): m for m in manifests}
    seen = {m["x_var"] for m in manifests} | {m["y_var"] for m in manifests}
    extra = sorted(v for v in seen if v not in VAR_ORDER)
    order = [v for v in VAR_ORDER if v in seen] + extra

    first = manifests[0]
    frozen_union: dict[str, object] = {}
    for m in manifests:
        frozen_union.update(m.get("frozen_effective", {}))

    summary_items = [
        ("pairs", len(manifests)),
        ("T (s)", first.get("T")),
        ("dt", first.get("dt")),
        ("seed", first.get("seed")),
        ("method", first.get("method")),
        ("neuromod_type", first.get("neuromod_type")),
        ("reward_signal", first.get("reward_signal")),
    ]

    rows: list[str] = []
    head = "".join(f"<th class='vh top'>{html.escape(v)}</th>" for v in order)
    rows.append(f"<tr><th class='corner'></th>{head}</tr>")
    for y in order:
        cells = [f"<th class='vh side'>{html.escape(y)}</th>"]
        for x in order:
            m = by_pair.get((x, y))
            if m is None:
                cells.append("<td class='empty'></td>")
                continue
            png = f"{m['hash']}.png"
            js = f"{m['hash']}.json"
            tip = f"{x} x {y}  |  hash={m['hash']}  |  saved {m.get('saved_at', '?')}"
            cells.append(
                "<td>"
                f"<a href='{html.escape(png)}' title='{html.escape(tip)}'>"
                f"<img src='{html.escape(png)}' loading='lazy' "
                f"alt='{html.escape(x)} x {html.escape(y)}'>"
                f"</a>"
                f"<div class='cap'>"
                f"<span class='lbl'>{html.escape(x)} x {html.escape(y)}</span>"
                f"<a href='{html.escape(js)}'>json</a>"
                f"</div>"
                "</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    summary_html = "".join(
        f"<li><b>{html.escape(k)}</b> {html.escape(_fmt(v))}</li>"
        for k, v in summary_items if v is not None
    )
    frozen_html = "".join(
        f"<li><b>{html.escape(k)}</b> {html.escape(_fmt(v))}</li>"
        for k, v in sorted(frozen_union.items())
    )

    doc = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>2D sweep contact sheet</title>
<style>
  html, body {{ margin: 0; padding: 0; background: #111; color: #eee;
                font: 13px/1.4 system-ui, sans-serif; }}
  header {{ padding: 12px 16px 14px; border-bottom: 1px solid #333; }}
  header h1 {{ margin: 0 0 8px; font-size: 15px; font-weight: 600; }}
  header h2 {{ margin: 10px 0 4px; font-size: 11px; color: #888;
               text-transform: uppercase; letter-spacing: 0.06em; }}
  header ul {{ list-style: none; margin: 0; padding: 0;
               display: flex; flex-wrap: wrap; gap: 3px 18px; }}
  header li {{ color: #bbb; font-family: ui-monospace, monospace; font-size: 12px; }}
  header b {{ color: #eee; font-weight: 600; margin-right: 4px; }}
  .wrap {{ overflow: auto; padding: 8px; }}
  table {{ border-collapse: separate; border-spacing: 3px; }}
  th, td {{ padding: 0; }}
  th.corner {{ background: #111; position: sticky; top: 0; left: 0; z-index: 3; }}
  th.vh {{ background: #1c1c1c; color: #ffd66b; padding: 6px 10px;
           font-family: ui-monospace, monospace; font-size: 12px; white-space: nowrap; }}
  th.vh.top {{ position: sticky; top: 0; z-index: 2; }}
  th.vh.side {{ position: sticky; left: 0; z-index: 2;
                text-align: right; min-width: 92px; }}
  td {{ background: #1c1c1c; vertical-align: top; }}
  td.empty {{ background: #161616; }}
  td img {{ display: block; width: 240px; height: auto; }}
  td .cap {{ padding: 3px 6px; font-family: ui-monospace, monospace;
             font-size: 11px; color: #bbb; border-top: 1px solid #252525;
             display: flex; justify-content: space-between;
             align-items: center; gap: 8px; }}
  td .cap .lbl {{ color: #ddd; }}
  td .cap a {{ color: #8cf; text-decoration: none; }}
  td .cap a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<header>
  <h1>2D sweep contact sheet: output/sweeps/</h1>
  <ul>{summary_html}</ul>
  <h2>Frozen baseline</h2>
  <ul>{frozen_html}</ul>
</header>
<div class='wrap'>
<table>
{chr(10).join(rows)}
</table>
</div>
</body>
</html>
"""
    INDEX.write_text(doc, encoding="utf-8")
    print(f"wrote {INDEX}  ({len(manifests)} pairs, {len(order)} vars)")


if __name__ == "__main__":
    main()
