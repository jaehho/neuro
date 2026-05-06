"""sim.py — compatibility shim.

The simulation module has been split across:
  - ``neuro.params``   – Params dataclass, state layout, key generation
  - ``neuro.recording`` – Recorder hierarchy and factory
  - ``neuro.dynamics``  – RHS, integration, reward/modulation, packing
  - ``neuro.simulate``  – main event-driven loop
  - ``neuro.io``        – Parquet/HDF5/Polars loaders, downsampling
  - ``neuro.plotting``  – matplotlib + plotly + zoom-adaptive HTTP server
  - ``neuro.cli``       – typer entry point

Prefer ``from neuro import Params, simulate`` (or import from the
submodule directly).  This file re-exports the previous flat namespace
so existing notebooks, tests, and scripts keep working.
"""
from __future__ import annotations

from neuro.cli import app, main  # noqa: F401
from neuro.dynamics import (  # noqa: F401
    _advance_state,
    _compute_modulation,
    _compute_reward,
    _compute_target_r_post,
    _crossing_fraction,
    _pack_state,
    _parse_target_params,
    _row_from_state,
    _smooth_rhs,
)
from neuro.io import (  # noqa: F401
    _add_bucket_extrema_indices,
    _collect_hdf5_envelope_frame,
    _collect_memory_envelope_frame,
    _collect_parquet_envelope_frame,
    _count_parquet_rows,
    _downsample_memory_rec,
    _filter_spike_times,
    load_plot_frame,
    load_time_series_frame,
    polars_frame_from_hdf5,
    polars_frame_from_parquet,
    read_spike_times_hdf5,
    read_spike_times_parquet,
)
from neuro.params import (  # noqa: F401
    E1_IDX,
    E2_IDX,
    I_S1_IDX,
    I_S2_IDX,
    N_PER_SYN,
    N_SHARED,
    N_STATE,
    Params,
    R_POST_IDX,
    RBAR_IDX,
    SERIES_KEYS,
    SPIKE_KEYS,
    V_IDX,
    W1_IDX,
    W2_IDX,
    X_PRE1_IDX,
    X_PRE2_IDX,
    Y_POST_IDX,
    _E_idx,
    _I_s_idx,
    _syn_base,
    _W_idx,
    _X_pre_idx,
    n_state,
    series_keys,
    spike_keys,
)
from neuro.plotting import (  # noqa: F401
    ALL_PLOT_VARIABLES,
    VARIABLE_TITLES,
    _plotly_package_js_path,
    _plotly_values,
    all_plot_variables,
    build_all_in_one_plotly_figure,
    plot_all_in_one_figure_matplotlib,
    plot_all_in_one_plotly,
    serve_zoom_adaptive_plot,
    variable_titles,
    write_plotly_html,
)
from neuro.recording import (  # noqa: F401
    HDF5Recorder,
    MemoryRecorder,
    MultiRecorder,
    ParquetRecorder,
    Recorder,
    _build_recorder,
    _envelope_bucket_size,
    _sampling_stride,
    spike_parquet_path,
)
from neuro.simulate import simulate  # noqa: F401


if __name__ == "__main__":
    app()
