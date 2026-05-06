"""Neuromodulated STDP simulation — public API.

The full implementation lives in focused submodules; this package
re-exports the names most callers want.
"""
from neuro.convergence import (
    ConvergenceCriterion,
    StreamingConvergence,
    check_steady_state,
)
from neuro.params import Params, n_state, series_keys, spike_keys
from neuro.recording import (
    HDF5Recorder,
    MemoryRecorder,
    MultiRecorder,
    ParquetRecorder,
    Recorder,
)
from neuro.simulate import simulate

__all__ = [
    "Params",
    "Recorder",
    "MemoryRecorder",
    "HDF5Recorder",
    "ParquetRecorder",
    "MultiRecorder",
    "simulate",
    "n_state",
    "series_keys",
    "spike_keys",
    "ConvergenceCriterion",
    "StreamingConvergence",
    "check_steady_state",
]
