"""Neuromodulated STDP simulation — public API.

The math lives in ``neuro.params`` (Params dataclass + state layout) and
``neuro.dynamics`` (RHS, integration, reward/modulation). ``neuro.simulate``
is the main loop and the run journal (``Run``, ``load_latest``,
``list_runs``). ``neuro.journal`` exposes tags + cross-run resolution
(``resolve``, ``tag``, ``load_by_tag``). ``neuro.plotting.serve_zoom``
opens the zoom-adaptive Plotly viewer in a browser, and
``neuro.sweep_viewer.serve_sweep`` does the equivalent for a sweep's
heatmap + per-cell viewers. ``neuro.tui`` is the Textual browse/tag/filter
catalog — pressing ``enter`` there launches the appropriate browser viewer
on a daemon thread.
"""
from neuro.convergence import (
    ConvergenceCriterion,
    StreamingConvergence,
    check_steady_state,
)
from neuro.journal import (
    SweepAxis,
    SweepEntry,
    list_entries,
    load_by_tag,
    resolve,
    set_note,
    tag,
    untag,
)
from neuro.params import Params, n_state, series_keys, spike_keys
from neuro.plotting import serve_zoom
from neuro.simulate import Run, list_runs, load_latest, load_run, simulate

__all__ = [
    "Params",
    "Run",
    "SweepAxis",
    "SweepEntry",
    "simulate",
    "load_run",
    "load_latest",
    "list_runs",
    "list_entries",
    "resolve",
    "load_by_tag",
    "tag",
    "untag",
    "set_note",
    "serve_zoom",
    "ConvergenceCriterion",
    "StreamingConvergence",
    "check_steady_state",
    "n_state",
    "series_keys",
    "spike_keys",
]
