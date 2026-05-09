"""Tests for steady-state detection (early-stop and post-hoc check)."""
from __future__ import annotations

import numpy as np
import pytest

from neuro.convergence import (
    ConvergenceCriterion,
    MultiConvergence,
    StreamingConvergence,
    check_steady_state,
)


# ── post-hoc check ────────────────────────────────────────────────

def test_check_steady_state_flat_signal():
    t = np.linspace(0, 30, 3001)
    v = np.full_like(t, 10.0)
    out = check_steady_state(t, v, target=10.0)
    assert out["converged"] is True
    assert out["delta"] == pytest.approx(0.0)
    assert out["abs_error"] == pytest.approx(0.0)


def test_check_steady_state_drifting_signal():
    t = np.linspace(0, 30, 3001)
    v = 5.0 + 0.5 * t  # 5 → 20 over 30 s
    out = check_steady_state(t, v, target=20.0)
    assert out["converged"] is False
    assert out["delta"] > 1.0


def test_check_steady_state_window_underflow():
    t = np.linspace(0, 1, 11)
    v = np.full_like(t, 7.0)
    out = check_steady_state(t, v, target=7.0)
    assert out["converged"] is False
    assert out["reason"] in {"window_underflow", "insufficient_samples"}


def test_check_steady_state_relative_tolerance():
    t = np.linspace(0, 30, 3001)
    v = np.where(t < 15, 9.9, 10.05)
    crit = ConvergenceCriterion(window=5.0, rel_tol=0.05, abs_tol=0.0)
    out = check_steady_state(t, v, target=10.0, criterion=crit)
    assert out["converged"] is True


# ── streaming detector ────────────────────────────────────────────

def test_streaming_detects_settled_signal():
    crit = ConvergenceCriterion(
        window=2.0, rel_tol=0.0, abs_tol=0.1,
        consecutive=2, min_t=2.0, check_interval=0.5,
    )
    sc = StreamingConvergence(criterion=crit, target=10.0)
    converged = False
    converged_at = None
    for k in range(0, 1500):
        t = k * 0.01
        if sc.update(t, 10.0):
            converged = True
            converged_at = t
            break
    assert converged
    assert converged_at is not None and converged_at <= 8.0


def test_streaming_does_not_trigger_on_drifting_signal():
    crit = ConvergenceCriterion(
        window=2.0, rel_tol=0.0, abs_tol=0.1,
        consecutive=2, min_t=2.0, check_interval=0.5,
    )
    sc = StreamingConvergence(criterion=crit, target=10.0)
    fired = False
    for k in range(0, 2000):
        t = k * 0.01
        # slope 0.2/s × window 2 s → half-mean drift 0.4, well above abs_tol 0.1
        v = 5.0 + 0.2 * t
        if sc.update(t, v):
            fired = True
            break
    assert fired is False


def test_streaming_consecutive_debounces_transient_flat():
    crit = ConvergenceCriterion(
        window=1.0, rel_tol=0.0, abs_tol=0.1,
        consecutive=3, min_t=1.0, check_interval=0.5,
    )
    sc = StreamingConvergence(criterion=crit, target=10.0)
    rng = np.random.default_rng(0)
    fired_early = False
    for k in range(0, 1500):
        t = k * 0.01
        # Steady mean but high jitter; the half-means will fluctuate above abs_tol
        v = 10.0 + rng.normal(scale=2.0)
        if sc.update(t, v):
            fired_early = True
            break
    assert fired_early is False


def test_check_steady_state_flat_but_off_target_not_converged():
    """When target_abs_tol is set, a flat-but-wrong signal is NOT converged."""
    t = np.linspace(0, 30, 3001)
    v = np.full_like(t, 5.0)
    crit = ConvergenceCriterion(window=5.0, abs_tol=0.5, target_abs_tol=1.0)
    out = check_steady_state(t, v, target=10.0, criterion=crit)
    assert out["flat"] is True
    assert out["on_target"] is False
    assert out["converged"] is False


def test_streaming_target_tol_blocks_off_target_signal():
    """The streaming detector with target_abs_tol should not fire on a
    flat-but-wrong signal even after many timesteps."""
    crit = ConvergenceCriterion(
        window=2.0, rel_tol=0.0, abs_tol=0.1,
        consecutive=2, min_t=2.0, check_interval=0.5,
        target_abs_tol=1.0,
    )
    sc = StreamingConvergence(criterion=crit, target=10.0)
    fired = False
    for k in range(0, 1500):
        if sc.update(k * 0.01, 5.0):  # flat at 5, target is 10
            fired = True
            break
    assert fired is False


def test_multi_convergence_requires_all_children():
    """MultiConvergence should only fire when every child has fired."""
    rate_crit = ConvergenceCriterion(window=2.0, abs_tol=0.1, consecutive=2,
                                     min_t=2.0, check_interval=0.5)
    weight_crit = ConvergenceCriterion(window=2.0, abs_tol=0.01, consecutive=2,
                                       min_t=2.0, check_interval=0.5)
    rate_det = StreamingConvergence(criterion=rate_crit, target=10.0, signal="r_post")
    weight_det = StreamingConvergence(criterion=weight_crit, signal="w")
    multi = MultiConvergence(rate_det, weight_det)

    fired_t: float | None = None
    for k in range(0, 2000):
        t = k * 0.01
        # rate is flat from the start, but weight drifts noticeably until t=10
        r = 10.0
        w = 1.0 + (max(0.0, 10.0 - t) * 0.05)
        if multi.update(t, r_post=r, w=w):
            fired_t = t
            break
    assert fired_t is not None
    # Rate alone would fire near t=4, but weight only flattens after t=10.
    assert fired_t > 10.0


def test_streaming_reset_clears_streak():
    crit = ConvergenceCriterion(
        window=1.0, rel_tol=0.0, abs_tol=0.1,
        consecutive=2, min_t=1.0, check_interval=0.5,
    )
    sc = StreamingConvergence(criterion=crit, target=10.0)
    for k in range(0, 500):
        sc.update(k * 0.01, 10.0)
    sc.reset()
    assert sc.converged_at is None
    fired = False
    for k in range(0, 200):
        if sc.update(k * 0.01, 10.0):
            fired = True
            break
    assert fired is False  # min_t guards against immediate re-fire after reset
