"""Tests for variable-N pre-synaptic neuron support."""
from __future__ import annotations

import numpy as np
import pytest

from neuro.sim import (
    Params,
    _pack_state,
    _advance_state,
    _smooth_rhs,
    n_state,
    series_keys,
    spike_keys,
    simulate,
    _I_s_idx,
    _W_idx,
    _E_idx,
    _X_pre_idx,
    V_IDX,
    Y_POST_IDX,
    R_POST_IDX,
    RBAR_IDX,
)


# ---------------------------------------------------------------------------
# Params construction and broadcasting
# ---------------------------------------------------------------------------

class TestParamsBroadcasting:
    def test_scalar_broadcast(self) -> None:
        p = Params(n_pre=5, r_pre_rates=20.0, w0=3.0)
        assert p.r_pre_rates == (20.0, 20.0, 20.0, 20.0, 20.0)
        assert p.w0 == (3.0, 3.0, 3.0, 3.0, 3.0)

    def test_tuple_passthrough(self) -> None:
        p = Params(n_pre=3, r_pre_rates=(10.0, 20.0, 30.0), w0=(1.0, 2.0, 3.0))
        assert p.r_pre_rates == (10.0, 20.0, 30.0)
        assert p.w0 == (1.0, 2.0, 3.0)

    def test_length_mismatch_errors(self) -> None:
        with pytest.raises(ValueError, match="r_pre_rates has length 3"):
            Params(n_pre=2, r_pre_rates=(10.0, 20.0, 30.0))

    def test_n_pre_1(self) -> None:
        p = Params(n_pre=1, r_pre_rates=15.0, w0=1.5)
        assert p.r_pre_rates == (15.0,)
        assert p.w0 == (1.5,)
        assert len(p.I_s0) == 1

    def test_default_n_pre_1(self) -> None:
        p = Params()
        assert p.n_pre == 1
        assert len(p.r_pre_rates) == 1
        assert len(p.w0) == 1

    def test_backward_compat_properties(self) -> None:
        p = Params(n_pre=2, r_pre_rates=(10.0, 20.0), w0=(1.0, 2.0))
        assert p.r_pre_rate_1 == 10.0
        assert p.r_pre_rate_2 == 20.0
        assert p.w1_0 == 1.0
        assert p.w2_0 == 2.0


# ---------------------------------------------------------------------------
# State vector layout
# ---------------------------------------------------------------------------

class TestStateVectorLayout:
    @pytest.mark.parametrize("n_pre,expected_size", [(1, 8), (2, 12), (5, 24), (10, 44)])
    def test_state_vector_size(self, n_pre: int, expected_size: int) -> None:
        assert n_state(n_pre) == expected_size

    def test_pack_state_n1(self) -> None:
        p = Params(n_pre=1, V0=-60.0, w0=3.0, I_s0=0.5)
        y = _pack_state(p)
        assert len(y) == n_state(1)
        assert y[V_IDX] == -60.0
        assert y[_W_idx(0)] == 3.0
        assert y[_I_s_idx(0)] == 0.5

    def test_pack_state_n5(self) -> None:
        p = Params(n_pre=5, w0=(1.0, 2.0, 3.0, 4.0, 5.0))
        y = _pack_state(p)
        assert len(y) == n_state(5)
        for i in range(5):
            assert y[_W_idx(i)] == float(i + 1)


# ---------------------------------------------------------------------------
# Dynamic key generation
# ---------------------------------------------------------------------------

class TestKeyGeneration:
    def test_series_keys_n1(self) -> None:
        keys = series_keys(1)
        assert "w1" in keys
        assert "w2" not in keys
        assert "E1" in keys
        assert "pre1_spike_bin" in keys
        assert "pre2_spike_bin" not in keys

    def test_series_keys_n3(self) -> None:
        keys = series_keys(3)
        assert "w1" in keys
        assert "w2" in keys
        assert "w3" in keys
        assert "w4" not in keys

    def test_spike_keys_n1(self) -> None:
        keys = spike_keys(1)
        assert keys == ["pre1_spike_times", "post_spike_times"]

    def test_spike_keys_n3(self) -> None:
        keys = spike_keys(3)
        assert keys == ["pre1_spike_times", "pre2_spike_times", "pre3_spike_times", "post_spike_times"]


# ---------------------------------------------------------------------------
# ODE with N=1
# ---------------------------------------------------------------------------

class TestODEN1:
    def test_rhs_equilibrium(self) -> None:
        p = Params(n_pre=1, V0=-65.0, E_L=-65.0, w0=1.0,
                   I_s0=0.0, x_pre0=0.0, E0=0.0,
                   r_post0=0.0, R_bar0=0.0, y_post0=0.0, r_target=0.0)
        y = _pack_state(p)
        rhs = _smooth_rhs(y, p, voltage_active=True)
        np.testing.assert_allclose(rhs, 0.0, atol=1e-15)

    def test_decay_single_synapse(self) -> None:
        p = Params(n_pre=1, I_s0=1.0, w0=0.0, E0=0.0,
                   x_pre0=0.0, r_post0=0.0,
                   R_bar0=0.0, y_post0=0.0, r_target=0.0, V0=-65.0, E_L=-65.0)
        y = _pack_state(p)
        dt = 1e-4
        T = 5 * p.tau_s
        for _ in range(int(T / dt)):
            y = _advance_state(y, dt, p, method="rk4", voltage_active=False)
        exact = 1.0 * np.exp(-T / p.tau_s)
        assert abs(y[_I_s_idx(0)] - exact) / exact < 1e-7

    def test_weight_clamp(self) -> None:
        p = Params(n_pre=1, w0=0.001, E0=1.0, R_bar0=100.0,
                   r_post0=0.0, r_target=0.0)
        y = _pack_state(p)
        result = _advance_state(y, 0.01, p, method="rk4", voltage_active=False)
        assert result[_W_idx(0)] >= 0.0


# ---------------------------------------------------------------------------
# Short simulation smoke tests
# ---------------------------------------------------------------------------

class TestSimulateSmoke:
    @pytest.mark.parametrize("n_pre", [1, 2, 3, 5])
    def test_simulate_runs(self, n_pre: int) -> None:
        p = Params(n_pre=n_pre, T=0.1, method="rk4", r_pre_rates=20.0, w0=2.0)
        rec = simulate(p)
        # Check correct keys exist
        expected_ser = series_keys(n_pre)
        expected_spk = spike_keys(n_pre)
        for key in expected_ser:
            assert key in rec, f"Missing series key: {key}"
        for key in expected_spk:
            assert key in rec, f"Missing spike key: {key}"

    @pytest.mark.parametrize("n_pre", [1, 2, 5])
    def test_no_nan_inf(self, n_pre: int) -> None:
        p = Params(n_pre=n_pre, T=0.5, method="rk4", r_pre_rates=20.0, w0=2.0)
        rec = simulate(p)
        for i in range(n_pre):
            w = rec[f"w{i+1}"]
            assert not np.any(np.isnan(w)), f"NaN in w{i+1}"
            assert not np.any(np.isinf(w)), f"Inf in w{i+1}"
            assert np.all(w >= 0.0), f"w{i+1} below 0"
            assert np.all(w <= p.wmax + 1e-9), f"w{i+1} above wmax"

    def test_n1_contingent_reward(self) -> None:
        """Contingent reward should work with a single pre-neuron (synapse 0 = target)."""
        p = Params(n_pre=1, T=0.5, method="rk4", reward_signal="contingent",
                   neuromod_type="covariance", r_pre_rates=20.0, w0=2.0)
        rec = simulate(p)
        assert not np.any(np.isnan(rec["M"]))
