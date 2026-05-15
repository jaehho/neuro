"""Main simulation loop and the run-journal it writes to.

A simulation is identified by a *name*. ``simulate(p, name)`` writes:

    output/<name>/<timestamp>.parquet           per-step series
    output/<name>/<timestamp>.spikes.parquet    spike events
    output/<name>/<timestamp>.json              sidecar (Params + metadata)

and returns a ``Run`` object that bundles those paths with the Params.

Browse past runs with ``list_runs()``; reload one with
``load_latest(name)`` or ``load_run(parquet_path)``.
"""
from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from neuro.convergence import StreamingConvergence
from neuro.dynamics import (
    _advance_state,
    _crossing_fraction,
    _init_state,
    _row_from_state,
)
from neuro.params import (
    Params,
    R_POST_IDX,
    V_IDX,
    Y_POST_IDX,
    E_idx,
    I_s_idx,
    W_idx,
    X_pre_idx,
)
from neuro.recording import ParquetRecorder, spike_parquet_path


# ── Run record + journal ──────────────────────────────────────────

@dataclass
class Run:
    name: str
    parquet: Path
    spikes: Path
    sidecar: Path
    params: Params
    saved_at: str
    duration_s: float
    rows_written: int
    spikes_written: int
    converged_at: float | None
    tags: list[str] = field(default_factory=list)
    note: str = ""

    def df(self) -> pl.DataFrame:
        """Read the per-step series into memory."""
        return pl.read_parquet(str(self.parquet))

    def serve(self, *, background: bool = False, **kwargs) -> str | None:
        """Open the Plotly zoom-adaptive viewer in a browser.

        Blocking by default (Ctrl-C exits) so ``run.serve()`` in an
        experiment script keeps the server alive. Pass ``background=True``
        to run on a daemon thread and return the URL — used by the TUI.
        """
        if background:
            from neuro.plotting import serve_zoom_background
            return serve_zoom_background(self.parquet, self.params, **kwargs)
        from neuro.plotting import serve_zoom
        serve_zoom(self.parquet, self.params, **kwargs)
        return None


def _sidecar_path(parquet: Path) -> Path:
    return parquet.with_suffix(".json")


def _write_sidecar(run: Run) -> None:
    payload = {
        "name": run.name,
        "saved_at": run.saved_at,
        "duration_s": run.duration_s,
        "rows_written": run.rows_written,
        "spikes_written": run.spikes_written,
        "converged_at": run.converged_at,
        "tags": list(run.tags),
        "note": run.note,
        "params": asdict(run.params),
    }
    run.sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_run(parquet_path: str | Path) -> Run:
    """Reconstruct a Run from a parquet file written by ``simulate``."""
    parquet = Path(parquet_path)
    sidecar = _sidecar_path(parquet)
    if not sidecar.exists():
        raise FileNotFoundError(f"No sidecar for {parquet} (expected {sidecar})")
    payload = json.loads(sidecar.read_text())
    return Run(
        name=payload["name"],
        parquet=parquet,
        spikes=spike_parquet_path(parquet),
        sidecar=sidecar,
        params=Params(**payload["params"]),
        saved_at=payload["saved_at"],
        duration_s=float(payload["duration_s"]),
        rows_written=int(payload["rows_written"]),
        spikes_written=int(payload["spikes_written"]),
        converged_at=payload.get("converged_at"),
        tags=list(payload.get("tags", [])),
        note=payload.get("note", ""),
    )


def list_runs(name: str | None = None, output_dir: str | Path = "output") -> list[Run]:
    """All runs under ``output_dir`` (optionally filtered to one name), newest first.

    A "run" is any parquet that has a matching ``.json`` sidecar; sweep
    layouts like ``ceiling-sweep/cell_03_07/<ts>.parquet`` are discovered
    via recursive search.
    """
    root = Path(output_dir)
    base = root / name if name is not None else root
    if not base.is_dir():
        return []
    runs: list[Run] = []
    for parquet in base.rglob("*.parquet"):
        if parquet.name.endswith(".spikes.parquet"):
            continue
        if not _sidecar_path(parquet).exists():
            continue
        runs.append(load_run(parquet))
    runs.sort(key=lambda r: r.saved_at, reverse=True)
    return runs


def load_latest(name: str, output_dir: str | Path = "output") -> Run:
    """Most recent run under ``output_dir/<name>/``."""
    runs = list_runs(name=name, output_dir=output_dir)
    if not runs:
        raise FileNotFoundError(f"No runs found in {Path(output_dir) / name}")
    return runs[0]


# ── Main loop ──────────────────────────────────────────────────────

def _tqdm_progress(it: Iterable[int]) -> Iterable[int]:
    return tqdm(it, desc="Simulating", unit="step", mininterval=0.5)


def simulate(
    p: Params,
    name: str,
    *,
    output_dir: str | Path = "output",
    chunk_rows: int = 100_000,
    progress: Callable[[Iterable[int]], Iterable[int]] | None = None,
    early_stop: StreamingConvergence | None = None,
    tags: list[str] | None = None,
    note: str = "",
) -> Run:
    """Run the n_pre → 1 post neuromodulated STDP simulation (RK4).

    Output is written to ``{output_dir}/{name}/{timestamp}.parquet`` along
    with a ``.spikes.parquet`` and ``.json`` sidecar; pick a stable *name*
    per experiment and runs accumulate as timestamped siblings.

    Pass ``progress`` to customize the step-loop indicator (default tqdm
    on stderr; pass ``lambda it: it`` to silence). Pass ``early_stop``
    (a ``StreamingConvergence``) to break out once r_post settles —
    everything written before the break is preserved.
    """
    if progress is None:
        progress = _tqdm_progress

    if p.rate_mode not in {"exp", "window"}:
        raise ValueError("Params.rate_mode must be 'exp' or 'window'.")
    use_window = p.rate_mode == "window"

    n_pre = p.n_pre
    n_steps = int(p.T / p.dt)
    rec_step = max(1, int(p.record_every / p.dt))

    rng = np.random.default_rng(p.seed)
    probs = [r * p.dt for r in p.r_pre]
    periods = [max(1, round(1.0 / (r * p.dt))) if r > 0 else 0 for r in p.r_pre]

    y = _init_state(p)
    ref_remaining = 0.0

    post_spike_buf: deque[float] = deque()

    saved_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / name
    parquet = run_dir / f"{saved_at}.parquet"
    recorder = ParquetRecorder(parquet, chunk_rows=chunk_rows, params=p)

    t0 = datetime.now()
    converged_at: float | None = None

    for step in progress(range(n_steps)):
        t = step * p.dt

        pre_spikes = [0] * n_pre
        for i in range(n_pre):
            if p.poisson:
                pre_spikes[i] = 1 if rng.random() < probs[i] else 0
            else:
                pre_spikes[i] = 1 if (periods[i] > 0 and step % periods[i] == 0) else 0

        for i in range(n_pre):
            if pre_spikes[i]:
                recorder.append_spike(f"pre{i+1}_spike_times", t)
                y[I_s_idx(i)] += 1.0
                y[X_pre_idx(i)] += 1.0
                y[E_idx(i)] -= p.eta_minus * y[W_idx(i)] * y[Y_POST_IDX]

        post_spike = 0
        is_refractory = 1 if ref_remaining > 0.0 else 0

        if use_window:
            cutoff = t - p.rate_window
            while post_spike_buf and post_spike_buf[0] < cutoff:
                post_spike_buf.popleft()
            ro = len(post_spike_buf) / p.rate_window
        else:
            ro = None

        if ref_remaining <= 0.0:
            v0 = float(y[V_IDX])
            y_trial = _advance_state(y, p.dt, p, method="rk4",
                                     voltage_active=True, rate_post=ro)
            frac = _crossing_fraction(v0, float(y_trial[V_IDX]), p.theta)

            if frac is None:
                y = y_trial
            else:
                dt1 = frac * p.dt
                dt2 = p.dt - dt1
                y_mid = _advance_state(y, dt1, p, method="rk4",
                                       voltage_active=True, rate_post=ro)
                spike_t = t + dt1
                recorder.append_spike("post_spike_times", spike_t)
                post_spike = 1

                y_mid[V_IDX] = p.V_reset
                y_mid[Y_POST_IDX] += 1.0
                y_mid[R_POST_IDX] += 1.0 / p.tau_r_post
                for i in range(n_pre):
                    soft_bound = (p.wmax - y_mid[W_idx(i)]) if p.bound_w else 1.0
                    y_mid[E_idx(i)] += p.eta_plus * soft_bound * y_mid[X_pre_idx(i)]

                if use_window:
                    post_spike_buf.append(spike_t)
                    ro = len(post_spike_buf) / p.rate_window

                y = _advance_state(y_mid, dt2, p, method="rk4",
                                   voltage_active=False, rate_post=ro)
                y[V_IDX] = p.V_reset
                ref_remaining = max(0.0, p.tau_ref - dt2)
        else:
            y = _advance_state(y, p.dt, p, method="rk4",
                               voltage_active=False, rate_post=ro)
            y[V_IDX] = p.V_reset
            ref_remaining = max(0.0, ref_remaining - p.dt)

        if step % rec_step == 0:
            recorder.append(
                _row_from_state(t, y, p, pre_spikes=pre_spikes, post_spike=post_spike,
                                is_refractory=is_refractory, rate_post=ro)
            )
            if early_stop is not None:
                rr_post = ro if ro is not None else float(y[R_POST_IDX])
                w_mean = float(np.mean([y[W_idx(i)] for i in range(n_pre)]))
                if early_stop.update(t, rr_post, r_post=rr_post, w=w_mean):
                    converged_at = t
                    break

    summary = recorder.finalize()
    duration_s = (datetime.now() - t0).total_seconds()
    run = Run(
        name=name,
        parquet=Path(summary["parquet_path"]),
        spikes=Path(summary["spikes_path"]),
        sidecar=_sidecar_path(parquet),
        params=p,
        saved_at=saved_at,
        duration_s=duration_s,
        rows_written=int(summary["rows_written"]),
        spikes_written=int(summary["spikes_written"]),
        converged_at=converged_at,
        tags=list(tags) if tags else [],
        note=note,
    )
    _write_sidecar(run)
    return run
