"""Config-file workflow: generate, load, and run from TOML.

Two flavours:

- *run* config — every Params field flat, plus runner options (cache,
  viewer port).  Use with ``neuro run --config FILE``.
- *sweep* config — Params under [base], plus [sweep] axes and
  [convergence] criterion.  Use with ``neuro sweep run --config FILE``.

The init command writes a TOML file with comments explaining each field.
Edit it, then re-run the same command without ``--config`` to apply.

TOML was picked over YAML/JSON because it has comments natively, the
read path is in stdlib (``tomllib``), and the format matches what
``pyproject.toml`` already uses.
"""
from __future__ import annotations

import sys
import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Any

from neuro.params import Params


# ── Field descriptions (one line each) ────────────────────────────

DESCRIPTIONS: dict[str, str] = {
    # Simulation
    "T": "Total simulated duration (s).",
    "dt": "Integration timestep (s); 0.1 ms is plenty for tau_m = 20 ms.",
    "seed": "RNG seed; controls Poisson spike trains.",
    "record_every": "Output sample interval (s). Doesn't affect dynamics.",
    "method": 'Integrator: "euler" or "rk4". RK4 is O(dt^4) between spikes.',
    "n_pre": "Number of pre-synaptic neurons (synapses) onto the post.",
    # Pre input
    "r_pre_rates": "Pre-synaptic firing rate (Hz). Scalar broadcasts across n_pre.",
    "poisson": "True for Poisson spike trains; false for deterministic periodic.",
    # LIF (Dayan & Abbott 2001, Ch. 5)
    "tau_m": "Membrane time constant (s).",
    "E_L": "Leak reversal potential (mV).",
    "V_reset": "Post-spike reset potential (mV).",
    "theta": "Spike threshold (mV).",
    "tau_ref": "Absolute refractory period (s); hard-caps r_post at 1/tau_ref.",
    "V0": "Initial membrane potential (mV).",
    "ref_remaining0": "Initial refractory time remaining (s).",
    # Synaptic current
    "tau_s": "Synaptic current decay constant (s).",
    "R_m": "Membrane input resistance (MOhm); scales current → voltage.",
    "I_s0": "Initial synaptic current (per synapse). Scalar broadcasts.",
    "I_ext": "Constant DC bias current to the post-synaptic neuron (nA).",
    # STDP traces (Bi & Poo 1998)
    "tau_plus": "Pre-to-post LTP trace decay (s).",
    "tau_minus": "Post-to-pre LTD trace decay (s).",
    "x_pre0": "Initial pre-synaptic trace value (per synapse). Scalar broadcasts.",
    "y_post0": "Initial post-synaptic trace value.",
    # Rate estimation (r_post)
    "tau_r": "Exponential rate-trace decay for r_post (s). Used in 'exp' mode.",
    "r_post0": "Initial r_post value.",
    "rate_mode": "'exp' (exponential trace) or 'window' (spike count over rate_window).",
    "rate_window": "Window length for rate_mode = 'window' (s).",
    # Eligibility (Frémaux & Gerstner 2016)
    "tau_e": "Eligibility trace decay (s); the 'memory' of pre-post pairings.",
    "E0": "Initial eligibility (per synapse). Scalar broadcasts.",
    # Reward baseline
    "tau_Rbar": "Reward-baseline tracking time constant (s); the critic.",
    "R_bar0": "Initial value of R-bar.",
    # Neuromod / reward
    "neuromod_type": 'How M derives from R: "covariance", "gated", "surprise", "constant".',
    "reward_signal": 'What R measures: "target_rate", "target_rate_linear", "biofeedback", "contingent", "constant".',
    "target_func": 'Family for target_rate: "fixed", "linear", "affine", "quadratic", "sqrt", "log", "sin", "power".',
    "target_func_params": 'JSON dict of coefficients for target_func, e.g. \'{"a": 0.3, "b": 2.0}\'.',
    "r_target": "Target rate for target_func='fixed' (Hz).",
    "alpha": "Slope for target_func='linear': target = alpha * r_pre.",
    "R_const": "Fixed scalar reward for reward_signal='constant'.",
    "reward_delay": "Event-to-reward delay for biofeedback / contingent (s).",
    "reward_amount": "Reward pulse magnitude.",
    "reward_tau": "Reward pulse decay constant (s).",
    "coincidence_window": "Target-to-post coincidence window for contingent (s).",
    # Weights
    "w0": "Initial weight (per synapse). Scalar broadcasts.",
    "wmax": "Hard upper bound on weights.",
    "eta_plus": "LTP eligibility step size.",
    "eta_minus": "LTD eligibility step size.",
}

# Mirrors cli._GROUPS but lives here too so config files are self-contained.
GROUPS: dict[str, list[str]] = {
    "Simulation": ["T", "dt", "seed", "method", "n_pre", "record_every"],
    "Pre input": ["r_pre_rates", "poisson"],
    "LIF neuron": ["tau_m", "E_L", "V_reset", "theta", "tau_ref", "V0", "ref_remaining0"],
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

_PER_SYNAPSE_TUPLE = {"r_pre_rates", "I_s0", "x_pre0", "E0", "w0"}


# ── TOML formatting ───────────────────────────────────────────────

def _format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, tuple):
        # tuple defaults are per-synapse; we expose the scalar broadcast
        return _format_value(v[0]) if v else "0.0"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _scalar_default(fname: str, default: Any) -> Any:
    if fname in _PER_SYNAPSE_TUPLE and isinstance(default, tuple):
        return default[0] if default else 0.0
    return default


def _params_section(prefix: str = "") -> str:
    """Return the Params block as TOML with descriptive comments.

    *prefix* is a leading indent (used inside a [base] table for sweep configs).
    """
    by_name = {f.name: f for f in fields(Params)}
    lines: list[str] = []
    for group, fnames in GROUPS.items():
        lines.append(f"{prefix}# ── {group} ──")
        for fn in fnames:
            f = by_name[fn]
            doc = DESCRIPTIONS.get(fn, "")
            val = _format_value(_scalar_default(fn, f.default))
            lines.append(f"{prefix}{fn} = {val}  # {doc}")
        lines.append("")
    return "\n".join(lines)


def generate_run_toml() -> str:
    return f"""# neuro run config
# Generated by `neuro config init run`. Edit then run:
#   uv run neuro run --config <this-file>
# Each value below is the dataclass default. Per-synapse fields
# (r_pre_rates, w0, I_s0, x_pre0, E0) accept a scalar that broadcasts
# to length n_pre.

{_params_section()}
[runner]
no_cache = false           # If true, force rerun even on cache hit.
plot = true                # Open the zoom-adaptive viewer after the run.
host = "127.0.0.1"
port = 8050
max_plot_points = 40000
"""


def generate_sweep_toml() -> str:
    return f"""# neuro sweep config
# Generated by `neuro config init sweep`. Edit then run:
#   uv run neuro sweep run --config <this-file>
# Cells route through cached_simulate() so each cell trace lands in
# output/<hash>.parquet and is shared with single-run cache hits.

[sweep]
x_var = "r_pre"            # Variable on the X axis. Any Params field, or aliases r_pre, W, eta.
x_min = 5.0
x_max = 1500.0
n_x = 14
x_log = true               # geomspace if true, linspace if false

y_var = "r_target"
y_min = 5.0
y_max = 1000.0
n_y = 14
y_log = true

[convergence]
# Rate (r_post) flatness AND optional on-target check. A cell is
# considered "rate-converged" when (a) the half-mean drift in r_post is
# within tolerance for `consecutive` successive checks, AND (b) the
# second-half mean is within target_abs_tol + target_rel_tol*|target|
# of the target (skip on-target by leaving target_abs_tol unset).
window = 8.0               # Half-window length (s).
rel_tol = 0.02             # Flatness rel tol × |target|.
abs_tol = 0.5              # Flatness abs tol (Hz).
consecutive = 5
min_t = 20.0
check_interval = 1.0
target_abs_tol = 2.0       # On-target abs tol (Hz). null/missing → don't check.
target_rel_tol = 0.05      # On-target rel tol × |target|.
target_fixed = 0.0         # 0 → infer target from r_target axis; non-zero overrides.

[weight_convergence]
# Weight (w[0]) flatness check. The cell only "converges" when both
# this and [convergence] have fired. Set enabled=false to disable.
enabled = true
window = 8.0
rel_tol = 0.0
abs_tol = 0.02             # Half-mean weight drift over `window`.
consecutive = 5
min_t = 20.0
check_interval = 1.0

[runner]
procs = 0                  # 0 means CPU count − 2.
chunk_rows = 100000

[base]
# Frozen Params for every cell (axes above override their two fields).

{_params_section(prefix='')}"""


# ── Loaders ───────────────────────────────────────────────────────

def load_run(path: Path) -> dict[str, Any]:
    """Read a *run* config file. Returns a dict suitable for `_kwargs_to_params` plus runner kwargs."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    runner = data.pop("runner", {}) if isinstance(data.get("runner"), dict) else {}
    # Remaining keys are flat Params fields (we wrote them at the top level).
    return {"params": data, "runner": runner}


def load_sweep(path: Path) -> dict[str, Any]:
    """Read a *sweep* config file. Keys: sweep, convergence, weight_convergence, runner, base."""
    with path.open("rb") as f:
        data = tomllib.load(f)
    return {
        "sweep": data.get("sweep", {}),
        "convergence": data.get("convergence", {}),
        "weight_convergence": data.get("weight_convergence", {}),
        "runner": data.get("runner", {}),
        "base": data.get("base", {}),
    }


# ── CLI helpers ───────────────────────────────────────────────────

def write_init(kind: str, out: Path | None) -> Path:
    if kind == "run":
        text = generate_run_toml()
        default_name = "neuro-run.toml"
    elif kind == "sweep":
        text = generate_sweep_toml()
        default_name = "neuro-sweep.toml"
    else:
        raise ValueError(f"Unknown kind {kind!r} (expected 'run' or 'sweep')")

    target = out or Path(default_name)
    if target.exists():
        # Refuse to silently clobber.
        print(f"Refusing to overwrite existing {target}. Pass --out to a different path or delete first.",
              file=sys.stderr)
        raise SystemExit(2)
    target.write_text(text, encoding="utf-8")
    return target
