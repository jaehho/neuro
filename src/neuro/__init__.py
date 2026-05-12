"""Neuromodulated STDP simulation — public API.

The math lives in ``neuro.params`` (Params dataclass + state layout) and
``neuro.dynamics`` (RHS, integration, reward/modulation). ``neuro.simulate``
is the main loop and the run journal (``Run``, ``load_latest``,
``list_runs``). ``neuro.plotting.serve_zoom`` opens the zoom-adaptive
HTTP viewer on a parquet path.
"""
from neuro.convergence import (
    ConvergenceCriterion,
    StreamingConvergence,
    check_steady_state,
)
from neuro.params import Params, n_state, series_keys, spike_keys
from neuro.plotting import serve_zoom
from neuro.simulate import Run, list_runs, load_latest, load_run, simulate

__all__ = [
    "Params",
    "Run",
    "simulate",
    "load_run",
    "load_latest",
    "list_runs",
    "serve_zoom",
    "ConvergenceCriterion",
    "StreamingConvergence",
    "check_steady_state",
    "n_state",
    "series_keys",
    "spike_keys",
]
