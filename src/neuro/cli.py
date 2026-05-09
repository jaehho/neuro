"""Click CLI: ``uv run neuro``.

Subcommands:
  run                   — single simulation; --interactive opens a wizard
  list                  — browse cached runs in output/runs.db
  show <hash>           — open the zoom-adaptive viewer on a cached run
  sweep run             — 2D sweep over arbitrary Params fields
  sweep show <hash>     — print a sweep manifest and its cell hashes
  sweep cell <h> <i> <j>— open the viewer on one cell of a sweep
  sweep index           — rebuild output/sweeps/index.html

Every ``Params`` field is exposed as a flag on ``run`` and ``sweep run``;
defaults come straight from the dataclass, so adding a field there
automatically adds a flag here.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import click

from neuro.cache import lookup_run, params_hash
from neuro.convergence import ConvergenceCriterion, StreamingConvergence
from neuro.params import Params


# ── Field metadata derived from Params ────────────────────────────

_CHOICES: dict[str, list[str]] = {
    "method": ["euler", "rk4"],
    "neuromod_type": ["covariance", "gated", "surprise", "constant"],
    "reward_signal": ["target_rate", "target_rate_linear", "biofeedback", "contingent", "constant"],
    "target_func": ["fixed", "linear", "affine", "quadratic", "sqrt", "log", "sin", "power"],
    "rate_mode": ["exp", "window"],
}

_PER_SYNAPSE_TUPLE = {"r_pre_rates", "I_s0", "x_pre0", "E0", "w0"}

# Friendly grouping for the wizard / --help (purely cosmetic)
_GROUPS: dict[str, list[str]] = {
    "Simulation": ["T", "dt", "seed", "method", "n_pre", "record_every"],
    "Pre input": ["r_pre_rates", "poisson"],
    "LIF": ["tau_m", "E_L", "V_reset", "theta", "tau_ref", "V0", "ref_remaining0"],
    "Synaptic current": ["tau_s", "R_m", "I_s0", "I_ext"],
    "STDP traces": ["tau_plus", "tau_minus", "x_pre0", "y_post0"],
    "Rate estimation": ["tau_r", "r_post0", "rate_mode", "rate_window"],
    "Eligibility": ["tau_e", "E0"],
    "Reward baseline": ["tau_Rbar", "R_bar0"],
    "Neuromod / reward": [
        "neuromod_type", "reward_signal", "target_func", "target_func_params",
        "r_target", "alpha", "R_const",
        "reward_delay", "reward_amount", "reward_tau", "coincidence_window",
    ],
    "Weights": ["w0", "wmax", "eta_plus", "eta_minus"],
}


def _scalar_default(fname: str, default: Any) -> Any:
    """For tuple fields, return the broadcast scalar default."""
    if fname in _PER_SYNAPSE_TUPLE and isinstance(default, tuple):
        return default[0] if default else 0.0
    return default


def _click_type(fname: str, default: Any) -> Any:
    if fname in _CHOICES:
        return click.Choice(_CHOICES[fname], case_sensitive=False)
    if fname in _PER_SYNAPSE_TUPLE:
        return float
    if isinstance(default, bool):
        return None  # use is_flag
    if isinstance(default, int):
        return int
    if isinstance(default, float):
        return float
    return str


def _params_options(func):
    """Decorate a click command with one --flag per Params field.

    Field defaults flow through from the Params dataclass directly, so
    adding a Params field automatically adds a CLI flag here.
    """
    for f in reversed(list(fields(Params))):
        flag = f"--{f.name.replace('_', '-')}"
        default = _scalar_default(f.name, f.default)
        ctype = _click_type(f.name, f.default)

        if isinstance(f.default, bool) or ctype is None:
            func = click.option(flag + "/" + flag.replace("--", "--no-"),
                                default=f.default, help=f.name)(func)
        else:
            func = click.option(flag, type=ctype, default=default,
                                show_default=True, help=f.name)(func)
    return func


def _kwargs_to_params(kwargs: dict[str, Any]) -> Params:
    """Build Params from CLI kwargs, broadcasting per-synapse scalars.

    Per-synapse fields accept either a scalar (broadcast to length n_pre)
    or a list/tuple (one value per synapse).
    """
    field_names = {f.name for f in fields(Params)}
    out = {k: v for k, v in kwargs.items() if k in field_names}
    n_pre = out.get("n_pre", 1)
    for fname in _PER_SYNAPSE_TUPLE:
        if fname not in out:
            continue
        v = out[fname]
        if isinstance(v, tuple):
            continue
        if isinstance(v, list):
            out[fname] = tuple(float(x) for x in v)
        else:
            out[fname] = tuple(float(v) for _ in range(n_pre))
    return Params(**out)


def _params_summary(p: Params) -> str:
    """Compact human-readable summary of a Params instance for confirmation prompts."""
    lines: list[str] = []
    for group, fnames in _GROUPS.items():
        bits: list[str] = []
        for fn in fnames:
            v = getattr(p, fn)
            if isinstance(v, tuple) and len(v) == 1:
                v = v[0]
            if isinstance(v, float):
                bits.append(f"{fn}={v:g}")
            else:
                bits.append(f"{fn}={v}")
        if bits:
            lines.append(f"  {group:<20} {' '.join(bits)}")
    return "\n".join(lines)


# ── Wizard (questionary) ─────────────────────────────────────────

def _format_field_value(fn: str, v: Any) -> str:
    if isinstance(v, tuple) and len(v) == 1:
        v = v[0]
    if isinstance(v, float):
        return f"{fn}={v:g}"
    return f"{fn}={v}"


def _group_summary(group: str, current: dict[str, Any], type_lookup: dict) -> str:
    bits: list[str] = []
    for fn in _GROUPS[group]:
        f = type_lookup[fn]
        v = current.get(fn, _scalar_default(fn, f.default))
        bits.append(_format_field_value(fn, v))
    return " ".join(bits)


def _run_wizard(initial: dict[str, Any] | None = None) -> dict[str, Any]:
    """Walk through Params field-by-field, grouped, with current values prefilled."""
    import questionary

    out: dict[str, Any] = dict(initial or {})
    type_lookup = {f.name: f for f in fields(Params)}

    # First: pick which groups to edit (default: only "essential" prompts).
    # Choice titles show each group's fields with their current/default values.
    essential = ["Simulation", "Pre input", "Neuromod / reward", "Weights"]
    name_width = max(len(g) for g in _GROUPS) + 2
    choices = [
        questionary.Choice(
            f"{g:<{name_width}} │ {_group_summary(g, out, type_lookup)}",
            value=g,
            checked=g in essential,
        )
        for g in _GROUPS
    ]
    groups = questionary.checkbox(
        "Which groups do you want to edit? (others stay at defaults shown)",
        choices=choices,
    ).ask()
    if groups is None:
        raise click.Abort()
    for group in groups:
        click.echo(click.style(f"\n[{group}]", fg="cyan", bold=True))
        for fn in _GROUPS[group]:
            f = type_lookup[fn]
            current = out.get(fn, _scalar_default(fn, f.default))
            if fn in _CHOICES:
                ans = questionary.select(fn, choices=_CHOICES[fn], default=str(current)).ask()
            elif isinstance(f.default, bool):
                ans = questionary.confirm(fn, default=bool(current)).ask()
            else:
                ans = questionary.text(fn, default=str(current)).ask()
            if ans is None:
                raise click.Abort()
            # Coerce
            if fn in _CHOICES or isinstance(f.default, bool):
                out[fn] = ans
            else:
                if isinstance(f.default, int) and fn not in _PER_SYNAPSE_TUPLE:
                    out[fn] = int(ans) if ans != "" else current
                elif isinstance(f.default, float) or fn in _PER_SYNAPSE_TUPLE:
                    out[fn] = float(ans) if ans != "" else current
                else:
                    out[fn] = ans
    return out


# ── Click app + run ───────────────────────────────────────────────

@click.group(help=__doc__)
def app() -> None:
    pass


@app.command("run", help="Run a single simulation. Use --interactive or --config FILE.")
@click.option("--interactive", is_flag=True, help="Walk through the params with prompts.")
@click.option("--config", "config_file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Load Params + runner options from a TOML config file.")
@click.option("--no-cache", is_flag=True, help="Force rerun (still writes to cache).")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("output"))
@click.option("--chunk-rows", type=int, default=100_000, help="Parquet chunk size.")
@click.option("--plot/--no-plot", default=True, help="Open the zoom-adaptive viewer after the run.")
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=int, default=8050)
@click.option("--max-plot-points", type=int, default=40_000)
@click.option("--variables", multiple=True, help="Restrict viewer to these variables (repeat).")
@_params_options
def run_cmd(
    interactive: bool,
    config_file: Path | None,
    no_cache: bool,
    cache_dir: Path,
    chunk_rows: int,
    plot: bool,
    host: str,
    port: int,
    max_plot_points: int,
    variables: tuple[str, ...],
    **kwargs: Any,
) -> None:
    if config_file is not None:
        from neuro.config import load_run
        cfg = load_run(config_file)
        # Config values override CLI defaults (but not explicit CLI flags — click can't
        # easily distinguish here, so for now: --config wins).
        kwargs = {**kwargs, **cfg["params"]}
        runner = cfg["runner"]
        if "no_cache" in runner: no_cache = bool(runner["no_cache"])
        if "plot" in runner: plot = bool(runner["plot"])
        if "host" in runner: host = str(runner["host"])
        if "port" in runner: port = int(runner["port"])
        if "max_plot_points" in runner: max_plot_points = int(runner["max_plot_points"])

    if interactive:
        kwargs = {**kwargs, **_run_wizard(initial=kwargs)}

    p = _kwargs_to_params(kwargs)
    h = params_hash(p)
    short = h[:12]

    click.echo("\n── Simulation ──")
    click.echo(_params_summary(p))
    click.echo(f"\n  hash:  {short}")

    hit = lookup_run(cache_dir / "runs.db", h) if not no_cache else None
    if hit is not None:
        click.echo(f"  cache: HIT — reusing run from {hit['created_at']}")
        click.echo(f"  file:  {hit['parquet_path']}")
    else:
        click.echo(f"  cache: {'FORCED RERUN' if no_cache else 'MISS'} — will save to {cache_dir}/{short}.parquet")

    if not click.confirm("\nProceed?", default=True):
        raise click.Abort()

    from neuro.cache import cached_simulate
    rec = cached_simulate(p, cache_dir=cache_dir, chunk_rows=chunk_rows, force=no_cache)

    if plot:
        from neuro.plotting import serve_zoom_adaptive_plot
        click.echo(f"\nServing zoom-adaptive viewer on http://{host}:{port}/")
        serve_zoom_adaptive_plot(
            rec["parquet_path"], p,
            host=host, port=port, max_points=max_plot_points,
            variables=list(variables) if variables else None,
        )


# ── list / show ───────────────────────────────────────────────────

@app.command("list", help="Show recent cached runs (output/runs.db).")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("output"))
@click.option("-n", "--limit", type=int, default=20)
@click.option("--all", "show_all", is_flag=True, help="No limit.")
def list_cmd(cache_dir: Path, limit: int, show_all: bool) -> None:
    db = cache_dir / "runs.db"
    if not db.exists():
        click.echo("No runs.db yet.")
        return
    conn = sqlite3.connect(str(db))
    try:
        cur = conn.execute(
            "SELECT hash, created_at, duration_s, parquet_path, params_json "
            "FROM runs ORDER BY created_at DESC"
        )
        rows = cur.fetchall() if show_all else cur.fetchmany(limit)
    finally:
        conn.close()

    if not rows:
        click.echo("No runs registered.")
        return

    click.echo(f"{'hash':<14} {'created':<19} {'wall(s)':>8}  summary")
    click.echo("─" * 100)
    for h, created, dur, pq, pj in rows:
        try:
            pd = json.loads(pj)
        except json.JSONDecodeError:
            pd = {}
        rate = pd.get("r_pre_rates", [None])
        rate_v = rate[0] if isinstance(rate, list) and rate else "?"
        bits = (
            f"T={pd.get('T', '?')}s "
            f"r_pre={rate_v} "
            f"r_target={pd.get('r_target', '?')} "
            f"reward={pd.get('reward_signal', '?')} "
            f"mod={pd.get('neuromod_type', '?')}"
        )
        dur_s = f"{dur:>7.1f}s" if dur is not None else "      ?s"
        click.echo(f"{h[:12]:<14} {created:<19} {dur_s:>8}  {bits}")


def _resolve_hash(cache_dir: Path, prefix: str) -> str:
    """Expand a 12-char (or longer) prefix to the full hash via runs.db."""
    db = cache_dir / "runs.db"
    if not db.exists():
        raise click.ClickException(f"No runs.db at {db}")
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT hash FROM runs WHERE hash LIKE ?", (prefix + "%",)
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        raise click.ClickException(f"No run with hash prefix {prefix!r}")
    if len(rows) > 1:
        raise click.ClickException(f"Ambiguous prefix {prefix!r}: {len(rows)} matches")
    return rows[0][0]


@app.command("show", help="Open the zoom-adaptive viewer on a cached run.")
@click.argument("hash_prefix")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("output"))
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=int, default=8050)
@click.option("--max-plot-points", type=int, default=40_000)
@click.option("--variables", multiple=True)
def show_cmd(hash_prefix: str, cache_dir: Path, host: str, port: int,
             max_plot_points: int, variables: tuple[str, ...]) -> None:
    h = _resolve_hash(cache_dir, hash_prefix)
    hit = lookup_run(cache_dir / "runs.db", h)
    assert hit is not None  # _resolve_hash already verified existence
    db = cache_dir / "runs.db"
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute("SELECT params_json FROM runs WHERE hash = ?", (h,)).fetchone()
    finally:
        conn.close()
    pd = json.loads(row[0])
    pd = {k: tuple(v) if isinstance(v, list) else v for k, v in pd.items()}
    pd.pop("__early_stop", None)
    p = Params(**pd)

    click.echo(f"hash:  {h[:12]}")
    click.echo(f"file:  {hit['parquet_path']}")
    click.echo(_params_summary(p))

    from neuro.plotting import serve_zoom_adaptive_plot
    click.echo(f"\nServing zoom-adaptive viewer on http://{host}:{port}/")
    serve_zoom_adaptive_plot(
        hit["parquet_path"], p,
        host=host, port=port, max_points=max_plot_points,
        variables=list(variables) if variables else None,
    )


# ── sweep ─────────────────────────────────────────────────────────

@app.group("sweep", help="2D parameter sweeps and drill-down.")
def sweep_group() -> None:
    pass


def _parse_frozen(ctx, param, value: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in value:
        if "=" not in s:
            raise click.BadParameter(f"--frozen expects KEY=VALUE, got {s!r}")
        k, v = s.split("=", 1)
        try:
            out[k.strip()] = json.loads(v.strip())
        except json.JSONDecodeError:
            out[k.strip()] = v.strip()
    return out


@sweep_group.command("run", help="Run a 2D sweep. Use --config FILE for config-driven workflow.")
@click.option("--interactive", is_flag=True, help="Walk through choices with prompts.")
@click.option("--config", "config_file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Load axes, base Params, and convergence from a TOML config file.")
@click.option("--x-var", help="Sweep variable on the X axis.")
@click.option("--x-min", type=float)
@click.option("--x-max", type=float)
@click.option("--n-x", type=int, default=12, show_default=True)
@click.option("--x-log", is_flag=True)
@click.option("--y-var")
@click.option("--y-min", type=float)
@click.option("--y-max", type=float)
@click.option("--n-y", type=int, default=12, show_default=True)
@click.option("--y-log", is_flag=True)
@click.option("--frozen", multiple=True, callback=_parse_frozen,
              help="Override a Params field: --frozen NAME=VALUE (repeatable).")
@click.option("--target-fixed", type=float, default=None,
              help="Force the convergence target instead of inferring from r_target axis.")
@click.option("--win", type=float, default=8.0, show_default=True, help="Rate convergence half-window (s).")
@click.option("--rel-tol", type=float, default=0.02, show_default=True)
@click.option("--abs-tol", type=float, default=0.5, show_default=True)
@click.option("--consecutive", type=int, default=5, show_default=True)
@click.option("--min-t", type=float, default=20.0, show_default=True)
@click.option("--check-interval", type=float, default=1.0, show_default=True)
@click.option("--target-abs-tol", type=float, default=2.0, show_default=True,
              help="On-target tol (Hz). Set to 0 (or omit) to skip the on-target check.")
@click.option("--target-rel-tol", type=float, default=0.05, show_default=True)
@click.option("--no-weight-check", is_flag=True, help="Disable the weight-flatness criterion.")
@click.option("--w-abs-tol", type=float, default=0.02, show_default=True,
              help="Weight half-mean drift tol over `win` (only when weight check enabled).")
@click.option("--procs", type=int, default=None,
              help="Worker processes (default: CPU count − 2).")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("output"))
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("output/sweeps"))
@click.option("--chunk-rows", type=int, default=100_000)
@_params_options
def sweep_run_cmd(
    interactive: bool,
    config_file: Path | None,
    x_var: str | None, x_min: float | None, x_max: float | None, n_x: int, x_log: bool,
    y_var: str | None, y_min: float | None, y_max: float | None, n_y: int, y_log: bool,
    frozen: dict[str, Any],
    target_fixed: float | None,
    win: float, rel_tol: float, abs_tol: float, consecutive: int, min_t: float, check_interval: float,
    target_abs_tol: float, target_rel_tol: float,
    no_weight_check: bool, w_abs_tol: float,
    procs: int | None,
    cache_dir: Path, out_dir: Path, chunk_rows: int,
    **kwargs: Any,
) -> None:
    import multiprocessing as mp

    weight_crit_kwargs: dict[str, Any] | None = None
    if not no_weight_check:
        weight_crit_kwargs = {
            "window": win, "rel_tol": 0.0, "abs_tol": w_abs_tol,
            "consecutive": consecutive, "min_t": min_t, "check_interval": check_interval,
        }

    if config_file is not None:
        from neuro.config import load_sweep
        cfg = load_sweep(config_file)
        sw = cfg["sweep"]
        x_var = sw.get("x_var", x_var)
        x_min = sw.get("x_min", x_min); x_max = sw.get("x_max", x_max)
        n_x = sw.get("n_x", n_x); x_log = sw.get("x_log", x_log)
        y_var = sw.get("y_var", y_var)
        y_min = sw.get("y_min", y_min); y_max = sw.get("y_max", y_max)
        n_y = sw.get("n_y", n_y); y_log = sw.get("y_log", y_log)
        cv = cfg["convergence"]
        win = cv.get("window", win)
        rel_tol = cv.get("rel_tol", rel_tol)
        abs_tol = cv.get("abs_tol", abs_tol)
        consecutive = cv.get("consecutive", consecutive)
        min_t = cv.get("min_t", min_t)
        check_interval = cv.get("check_interval", check_interval)
        tat = cv.get("target_abs_tol", target_abs_tol)
        target_abs_tol = float(tat) if tat else 0.0
        target_rel_tol = cv.get("target_rel_tol", target_rel_tol)
        tf = cv.get("target_fixed", 0.0)
        if tf:
            target_fixed = float(tf)
        wc = cfg.get("weight_convergence", {})
        if wc and not wc.get("enabled", True):
            weight_crit_kwargs = None
        elif wc:
            weight_crit_kwargs = {
                "window": wc.get("window", win),
                "rel_tol": wc.get("rel_tol", 0.0),
                "abs_tol": wc.get("abs_tol", w_abs_tol),
                "consecutive": wc.get("consecutive", consecutive),
                "min_t": wc.get("min_t", min_t),
                "check_interval": wc.get("check_interval", check_interval),
            }
        rn = cfg["runner"]
        if "procs" in rn and rn["procs"]:
            procs = int(rn["procs"])
        if "chunk_rows" in rn:
            chunk_rows = int(rn["chunk_rows"])
        kwargs = {**kwargs, **cfg["base"]}

    if interactive:
        x_var, x_min, x_max, n_x, x_log, y_var, y_min, y_max, n_y, y_log = _sweep_axes_wizard(
            x_var, x_min, x_max, n_x, x_log, y_var, y_min, y_max, n_y, y_log
        )
        kwargs = {**kwargs, **_run_wizard(initial=kwargs)}

    if not all([x_var, y_var]) or x_min is None or x_max is None or y_min is None or y_max is None:
        raise click.UsageError("Need --x-var/--y-var with --x-min/--x-max/--y-min/--y-max (or --interactive / --config).")

    # Apply --frozen overrides on top of Params kwargs
    for k, v in frozen.items():
        if k in {f.name for f in fields(Params)}:
            kwargs[k] = v

    base = _kwargs_to_params(kwargs)
    if procs is None:
        procs = max(1, mp.cpu_count() - 2)

    from neuro.sweep import SweepSpec, build_grid, run_sweep, write_outputs

    rate_crit = ConvergenceCriterion(
        window=win, rel_tol=rel_tol, abs_tol=abs_tol,
        consecutive=consecutive, min_t=min_t, check_interval=check_interval,
        target_abs_tol=(target_abs_tol if target_abs_tol > 0 else None),
        target_rel_tol=target_rel_tol,
    )
    weight_crit = (
        ConvergenceCriterion(**weight_crit_kwargs)
        if weight_crit_kwargs is not None else None
    )
    spec = SweepSpec(
        x_var=x_var, y_var=y_var,
        x_grid=build_grid(x_min, x_max, n_x, x_log),
        y_grid=build_grid(y_min, y_max, n_y, y_log),
        x_log=x_log, y_log=y_log,
        base=base,
        criterion=rate_crit,
        weight_criterion=weight_crit,
        target_fixed=target_fixed,
    )

    click.echo(f"Sweep: {len(spec.x_grid)}×{len(spec.y_grid)} = {spec.n_cells()} cells, T_max={base.T}s, procs={procs}")
    click.echo(f"  {spec.x_var}: {x_min}..{x_max}  (n={len(spec.x_grid)}, log={spec.x_log})")
    click.echo(f"  {spec.y_var}: {y_min}..{y_max}  (n={len(spec.y_grid)}, log={spec.y_log})")
    click.echo(f"  base: {_short_summary(base)}")
    if not click.confirm("Proceed?", default=True):
        raise click.Abort()

    df = run_sweep(spec, procs=procs, cache_dir=cache_dir, chunk_rows=chunk_rows)
    pq, js, png, h = write_outputs(spec, df, out_dir)
    n_conv = int(df["converged"].sum())
    click.echo(f"\nDone. {n_conv}/{len(df)} converged. hash={h}")
    click.echo(f"  {pq}")
    click.echo(f"  {js}")
    click.echo(f"  {png}")


def _short_summary(p: Params) -> str:
    return (
        f"reward={p.reward_signal} mod={p.neuromod_type} "
        f"rate_mode={p.rate_mode}{f' W={p.rate_window}' if p.rate_mode == 'window' else ''} "
        f"method={p.method} n_pre={p.n_pre} poisson={p.poisson} "
        f"w0={p.w0[0]} eta={p.eta_plus}"
    )


def _sweep_axes_wizard(x_var, x_min, x_max, n_x, x_log, y_var, y_min, y_max, n_y, y_log):
    import questionary
    from neuro.sweep import sweep_var_choices

    choices = sweep_var_choices()
    x_var = questionary.autocomplete("X axis variable", choices=choices, default=x_var or "r_pre").ask()
    x_min = float(questionary.text(f"{x_var} min", default=str(x_min) if x_min is not None else "10.0").ask())
    x_max = float(questionary.text(f"{x_var} max", default=str(x_max) if x_max is not None else "100.0").ask())
    n_x = int(questionary.text("n_x", default=str(n_x)).ask())
    x_log = questionary.confirm("X log-scale?", default=x_log).ask()
    y_var = questionary.autocomplete("Y axis variable", choices=choices, default=y_var or "r_target").ask()
    y_min = float(questionary.text(f"{y_var} min", default=str(y_min) if y_min is not None else "10.0").ask())
    y_max = float(questionary.text(f"{y_var} max", default=str(y_max) if y_max is not None else "100.0").ask())
    n_y = int(questionary.text("n_y", default=str(n_y)).ask())
    y_log = questionary.confirm("Y log-scale?", default=y_log).ask()
    return x_var, x_min, x_max, n_x, x_log, y_var, y_min, y_max, n_y, y_log


@sweep_group.command("show", help="Print a sweep manifest and list its cells.")
@click.argument("hash_prefix")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("output/sweeps"))
@click.option("--list-cells", is_flag=True, help="Print every cell hash + position.")
def sweep_show_cmd(hash_prefix: str, out_dir: Path, list_cells: bool) -> None:
    matches = list(out_dir.glob(f"{hash_prefix}*.json"))
    if not matches:
        raise click.ClickException(f"No sweep manifest with prefix {hash_prefix!r} in {out_dir}")
    if len(matches) > 1:
        raise click.ClickException(f"Ambiguous prefix: {[m.stem for m in matches]}")
    m = json.loads(matches[0].read_text())
    click.echo(f"hash:        {m['hash']}")
    click.echo(f"saved_at:    {m.get('saved_at', '?')}")
    click.echo(f"x_var:       {m['x_var']}  log={m.get('x_log', False)}  grid={m['x_grid'][:3]}…{m['x_grid'][-1:]}  n={len(m['x_grid'])}")
    click.echo(f"y_var:       {m['y_var']}  log={m.get('y_log', False)}  grid={m['y_grid'][:3]}…{m['y_grid'][-1:]}  n={len(m['y_grid'])}")
    click.echo(f"n_cells:     {m['n_cells']}")
    bp = m.get("base_params", {})
    click.echo(f"base:        T={bp.get('T')}s reward={bp.get('reward_signal')} mod={bp.get('neuromod_type')} "
               f"rate_mode={bp.get('rate_mode')} method={bp.get('method')}")
    crit = m.get("criterion", {})
    click.echo(f"criterion:   window={crit.get('window')} rel_tol={crit.get('rel_tol')} abs_tol={crit.get('abs_tol')} "
               f"consec={crit.get('consecutive')} min_t={crit.get('min_t')}")
    click.echo(f"png:         {out_dir}/{m['hash']}.png")
    click.echo(f"summary:     {out_dir}/{m['hash']}.parquet")
    if list_cells:
        click.echo("\ncells (i, j, hash):")
        for idx, ch in enumerate(m["cell_hashes"]):
            i = idx // len(m["x_grid"])
            j = idx % len(m["x_grid"])
            click.echo(f"  ({i:>2}, {j:>2})  {ch[:12]}")


@sweep_group.command("cell", help="Open the viewer on one cell of a sweep.")
@click.argument("hash_prefix")
@click.argument("i", type=int)
@click.argument("j", type=int)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("output/sweeps"))
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("output"))
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=int, default=8050)
@click.option("--max-plot-points", type=int, default=40_000)
def sweep_cell_cmd(hash_prefix: str, i: int, j: int, out_dir: Path, cache_dir: Path,
                   host: str, port: int, max_plot_points: int) -> None:
    matches = list(out_dir.glob(f"{hash_prefix}*.json"))
    if not matches:
        raise click.ClickException(f"No sweep manifest with prefix {hash_prefix!r}")
    m = json.loads(matches[0].read_text())
    nx = len(m["x_grid"])
    ny = len(m["y_grid"])
    if not (0 <= i < ny and 0 <= j < nx):
        raise click.UsageError(f"i must be in [0, {ny}) and j in [0, {nx})")
    cell_hash = m["cell_hashes"][i * nx + j]
    click.echo(f"cell ({i}, {j}): {m['x_var']}={m['x_grid'][j]:g}, {m['y_var']}={m['y_grid'][i]:g}")
    click.echo(f"hash: {cell_hash[:12]}")
    ctx = click.get_current_context()
    ctx.invoke(show_cmd, hash_prefix=cell_hash, cache_dir=cache_dir,
               host=host, port=port, max_plot_points=max_plot_points, variables=())


@sweep_group.command("index", help="Rebuild output/sweeps/index.html (contact sheet).")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("output/sweeps"))
def sweep_index_cmd(out_dir: Path) -> None:
    from neuro.sweep import build_index
    p = build_index(out_dir)
    click.echo(f"wrote {p}")


# ── cache merge ───────────────────────────────────────────────────

@app.group("cache", help="Cache maintenance.")
def cache_group() -> None:
    pass


@cache_group.command("merge", help="Merge rows from another runs.db (e.g. one pulled from a remote).")
@click.argument("other_db", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=Path("output"))
def cache_merge_cmd(other_db: Path, cache_dir: Path) -> None:
    from neuro.cache import merge_runs_db
    res = merge_runs_db(cache_dir / "runs.db", other_db)
    click.echo(f"merged {res['inserted']}/{res['seen']} new rows from {other_db} (total now {res['total']})")


# ── config init ───────────────────────────────────────────────────

@app.group("config", help="Generate / manage TOML config files.")
def config_group() -> None:
    pass


@config_group.command("init", help="Write a TOML config (run or sweep) with all defaults and descriptions.")
@click.argument("kind", type=click.Choice(["run", "sweep"]))
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Output path. Defaults to neuro-{kind}.toml in cwd.")
def config_init_cmd(kind: str, out_path: Path | None) -> None:
    from neuro.config import write_init
    p = write_init(kind, out_path)
    click.echo(f"wrote {p}")
    click.echo(f"edit then: uv run neuro {'sweep run' if kind == 'sweep' else 'run'} --config {p}")


if __name__ == "__main__":
    app()
