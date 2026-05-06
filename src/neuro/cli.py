"""Typer CLI: ``uv run neuro``.

Long runs that stream to parquet and serve the zoom-adaptive plot.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from neuro.params import Params
from neuro.plotting import (
    all_plot_variables,
    plot_all_in_one_figure_matplotlib,
    plot_all_in_one_plotly,
    serve_zoom_adaptive_plot,
)
from neuro.simulate import simulate

app = typer.Typer(help="Simulate neuromodulated STDP with optional HDF5 and Parquet output.")


@app.command()
def main(
    hdf5: Annotated[str | None, typer.Option(help="Write recorded state to an HDF5 file.")] = None,
    parquet: Annotated[str | None, typer.Option(help="Write recorded state to a Parquet file (overrides cache naming).")] = None,
    plot_backend: Annotated[str, typer.Option(help="Plot backend to use.")] = "server",
    plot_html: Annotated[str, typer.Option(help="Output HTML file for plotly backend.")] = "simulation.html",
    chunk_rows: Annotated[int, typer.Option(help="Row chunk size for streaming output.")] = 100_000,
    max_plot_points: Annotated[int, typer.Option(help="Max data points for plotting.")] = 40_000,
    host: Annotated[str, typer.Option(help="Host for server backend.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port for server backend.")] = 8050,
    method: Annotated[str, typer.Option(help="Integrator for smooth dynamics.")] = "rk4",
    rate_mode: Annotated[str, typer.Option(help="r_post firing-rate mode: 'exp' (exponential trace) or 'window' (spike count). r_pre is always constant.")] = "exp",
    rate_window: Annotated[float, typer.Option(help="Window duration in seconds for 'window' rate mode.")] = 0.5,
    neuromod_type: Annotated[str, typer.Option(help="Neuromodulator role (Frémaux Eq.14): covariance, gated, surprise, constant.")] = "covariance",
    reward_signal: Annotated[str, typer.Option(help="Reward signal: target_rate, biofeedback, contingent, constant.")] = "target_rate",
    n_pre: Annotated[int, typer.Option(help="Number of pre-synaptic neurons.")] = 1,
    variables: Annotated[list[str] | None, typer.Option(help="Variables to plot (e.g. --variables E1 --variables w1). Defaults to all.")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Force rerun, ignoring cached results.")] = False,
    cache_dir: Annotated[str, typer.Option(help="Directory for simulation cache.")] = "output",
):
    if plot_backend not in {"matplotlib", "plotly", "server"}:
        raise typer.BadParameter(f"plot-backend must be one of: matplotlib, plotly, server (got {plot_backend!r})")
    if method not in {"euler", "rk4"}:
        raise typer.BadParameter(f"method must be one of: euler, rk4 (got {method!r})")
    if rate_mode not in {"exp", "window"}:
        raise typer.BadParameter(f"rate-mode must be one of: exp, window (got {rate_mode!r})")
    if neuromod_type not in {"covariance", "gated", "surprise", "constant"}:
        raise typer.BadParameter(f"neuromod-type must be one of: covariance, gated, surprise, constant (got {neuromod_type!r})")
    if reward_signal not in {"target_rate", "biofeedback", "contingent", "constant"}:
        raise typer.BadParameter(f"reward-signal must be one of: target_rate, biofeedback, contingent, constant (got {reward_signal!r})")
    if n_pre < 1:
        raise typer.BadParameter(f"n-pre must be >= 1 (got {n_pre})")

    apv = all_plot_variables(n_pre)
    if variables is not None:
        bad = [v for v in variables if v not in apv]
        if bad:
            raise typer.BadParameter(f"Unknown variables: {bad}. Choose from: {apv}")

    params = Params(
        T=100,
        n_pre=n_pre,
        r_pre_rates=20.0,
        w0=2.0,
        method=method,
        rate_mode=rate_mode,
        rate_window=rate_window,
        neuromod_type=neuromod_type,
        reward_signal=reward_signal,
    )

    from neuro.cache import cached_simulate, lookup_run, params_hash
    use_cache = not parquet and not hdf5
    cache_hit = None
    if use_cache and not no_cache:
        hash_hex = params_hash(params)
        cache_hit = lookup_run(Path(cache_dir) / "runs.db", hash_hex)
    elif use_cache:
        hash_hex = params_hash(params)

    n_steps = int(params.T / params.dt)
    rates_str = ", ".join(f"{r} Hz" for r in params.r_pre_rates)
    weights_str = ", ".join(str(w) for w in params.w0)
    typer.echo("\n── Simulation Parameters ──")
    typer.echo(f"  Duration:       {params.T} s  ({n_steps:,} steps, dt={params.dt})")
    typer.echo(f"  Integrator:     {method}")
    typer.echo(f"  Neuromod type:  {neuromod_type}")
    typer.echo(f"  Reward signal:  {reward_signal}")
    typer.echo(f"  Rate mode:      {rate_mode}" + (f" (window={rate_window} s)" if rate_mode == "window" else ""))
    typer.echo(f"  Pre neurons:    {n_pre}")
    typer.echo(f"  Pre rates:      [{rates_str}]  ({'Poisson' if params.poisson else 'deterministic'})")
    typer.echo(f"  Weights:        [{weights_str}]  (wmax={params.wmax})")
    if cache_hit:
        typer.echo(f"  Cache:          HIT — reusing run from {cache_hit['created_at']}")
        typer.echo(f"  Output:         {cache_hit['parquet_path']}")
    elif use_cache and no_cache:
        short = hash_hex[:12]
        typer.echo(f"  Cache:          FORCED RERUN — will save to {cache_dir}/{short}.parquet")
    elif use_cache:
        short = hash_hex[:12]
        typer.echo(f"  Cache:          MISS — will save to {cache_dir}/{short}.parquet")
    else:
        typer.echo(f"  Output:         {parquet or hdf5 or 'in-memory'}")
    typer.echo(f"  Plot backend:   {plot_backend}")
    typer.echo("")
    if not typer.confirm("Proceed?", default=True):
        raise typer.Abort()

    if not use_cache and plot_backend == "server" and not (parquet or hdf5):
        raise typer.BadParameter("The server backend requires --parquet or --hdf5 so zoom requests can be resampled from disk.")

    rec = None
    if use_cache:
        rec = cached_simulate(params, cache_dir=Path(cache_dir), chunk_rows=chunk_rows, force=no_cache)
        parquet = rec.get("parquet_path", parquet)
    else:
        rec = simulate(
            params,
            hdf5_path=hdf5,
            parquet_path=parquet,
            chunk_rows=chunk_rows,
        )

    if plot_backend == "matplotlib":
        if isinstance(rec, dict) and "t" in rec:
            plot_all_in_one_figure_matplotlib(rec, params)
        else:
            raise typer.BadParameter("Matplotlib plotting requires in-memory records. Use Plotly for HDF5/Parquet-backed runs.")
    elif plot_backend == "server":
        plot_source = parquet or hdf5
        assert plot_source is not None, "No plot source available for server mode"
        serve_zoom_adaptive_plot(
            plot_source,
            params,
            host=host,
            port=port,
            max_points=max_plot_points,
            variables=variables,
        )
    else:
        plot_source = parquet or hdf5 or rec
        output_html = plot_all_in_one_plotly(
            plot_source,
            params,
            output_html=plot_html,
            max_points=max_plot_points,
            variables=variables,
        )
        print(f"Wrote interactive plot to {output_html}")


if __name__ == "__main__":
    app()
