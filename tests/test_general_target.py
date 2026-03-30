"""Unit tests for target functions and reward/modulation computation."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from neuro.sim import Params, _compute_target_r_post, _compute_reward, _compute_modulation


# ── _compute_target_r_post tests ─────────────────────────────────────

class TestComputeTargetRPost:
    def test_fixed_defaults_to_r_target(self) -> None:
        p = Params(target_func="fixed", r_target=10.0)
        assert _compute_target_r_post(p, 99.0) == pytest.approx(10.0)

    def test_fixed_custom_target(self) -> None:
        p = Params(target_func="fixed", target_func_params='{"target": 7.5}')
        assert _compute_target_r_post(p, 99.0) == pytest.approx(7.5)

    def test_linear_defaults_to_alpha(self) -> None:
        p = Params(alpha=0.5, target_func="linear", target_func_params="")
        assert _compute_target_r_post(p, 10.0) == pytest.approx(5.0)

    def test_linear_custom_a(self) -> None:
        p = Params(target_func="linear", target_func_params='{"a": 0.8}')
        assert _compute_target_r_post(p, 10.0) == pytest.approx(8.0)

    def test_affine(self) -> None:
        p = Params(target_func="affine", target_func_params='{"a": 0.3, "b": 2.0}')
        assert _compute_target_r_post(p, 10.0) == pytest.approx(5.0)

    def test_affine_defaults(self) -> None:
        p = Params(target_func="affine", target_func_params="")
        assert _compute_target_r_post(p, 10.0) == pytest.approx(5.0)

    def test_quadratic(self) -> None:
        p = Params(target_func="quadratic", target_func_params='{"a": -0.03, "b": 0.8, "c": 0.5}')
        expected = -0.03 * 100 + 0.8 * 10 + 0.5  # = -3 + 8 + 0.5 = 5.5
        assert _compute_target_r_post(p, 10.0) == pytest.approx(expected)

    def test_sqrt(self) -> None:
        p = Params(target_func="sqrt", target_func_params='{"a": 1.5, "b": 0.5}')
        expected = 1.5 * math.sqrt(10.0) + 0.5
        assert _compute_target_r_post(p, 10.0) == pytest.approx(expected)

    def test_log(self) -> None:
        p = Params(target_func="log", target_func_params='{"a": 1.5, "b": 1.5}')
        expected = 1.5 * math.log(10.0) + 1.5
        assert _compute_target_r_post(p, 10.0) == pytest.approx(expected)

    def test_sin(self) -> None:
        p = Params(target_func="sin", target_func_params='{"a": 2.0, "b": 5.0, "c": 0.3}')
        expected = 2.0 * math.sin(0.3 * 10.0) + 5.0
        assert _compute_target_r_post(p, 10.0) == pytest.approx(expected)

    def test_power(self) -> None:
        p = Params(target_func="power", target_func_params='{"a": 1.0, "b": 0.5, "c": 0.6}')
        expected = 1.0 * 10.0 ** 0.6 + 0.5
        assert _compute_target_r_post(p, 10.0) == pytest.approx(expected)

    def test_unknown_func_raises(self) -> None:
        p = Params(target_func="banana", target_func_params="")
        with pytest.raises(ValueError, match="Unknown target_func"):
            _compute_target_r_post(p, 10.0)

    def test_sqrt_negative_r_pre_clamped(self) -> None:
        p = Params(target_func="sqrt", target_func_params='{"a": 1.0, "b": 0.0}')
        result = _compute_target_r_post(p, -1.0)
        assert result == pytest.approx(0.0)

    def test_log_near_zero(self) -> None:
        p = Params(target_func="log", target_func_params='{"a": 1.0, "b": 0.0}')
        result = _compute_target_r_post(p, 0.0)
        assert np.isfinite(result)


# ── _compute_reward / _compute_modulation tests ──────────────────────

class TestComputeRewardTargetRate:
    def test_reward_at_target(self) -> None:
        """When r_post == target, R should be 0."""
        p = Params(
            reward_signal="target_rate",
            target_func="affine",
            target_func_params='{"a": 0.3, "b": 2.0}',
        )
        r_pre = 10.0
        target = _compute_target_r_post(p, r_pre)
        R = _compute_reward(p, r_pre, target)
        assert R == pytest.approx(0.0)

    def test_reward_below_target(self) -> None:
        """When r_post < target, R should be negative."""
        p = Params(
            reward_signal="target_rate",
            target_func="affine",
            target_func_params='{"a": 0.3, "b": 2.0}',
        )
        R = _compute_reward(p, 10.0, 2.0)
        assert R < 0.0

    def test_reward_quadratic_shape(self) -> None:
        """R = -(r_post - target)^2 is a quadratic penalty."""
        p = Params(
            reward_signal="target_rate",
            target_func="sqrt",
            target_func_params='{"a": 1.5, "b": 0.5}',
        )
        r_pre = 10.0
        target = _compute_target_r_post(p, r_pre)
        delta = 1.0
        R = _compute_reward(p, r_pre, target + delta)
        assert R == pytest.approx(-(delta ** 2))

    def test_biofeedback_passthrough(self) -> None:
        """Biofeedback mode returns the reward pulse directly."""
        p = Params(reward_signal="biofeedback")
        R = _compute_reward(p, 10.0, 5.0, reward_pulse=0.75)
        assert R == pytest.approx(0.75)

    def test_fixed_target_func(self) -> None:
        """target_func='fixed' ignores r_pre."""
        p = Params(reward_signal="target_rate", target_func="fixed", r_target=8.0)
        R = _compute_reward(p, 99.0, 8.0)
        assert R == pytest.approx(0.0)


class TestComputeModulation:
    def test_covariance(self) -> None:
        """M = R - R_bar for covariance mode."""
        p = Params(neuromod_type="covariance")
        R, R_bar = -4.0, -2.0
        M, rbar_target = _compute_modulation(p, R, R_bar, r_post=5.0)
        assert M == pytest.approx(R - R_bar)
        assert rbar_target == pytest.approx(R)

    def test_gated(self) -> None:
        """M = R for gated Hebbian mode."""
        p = Params(neuromod_type="gated")
        R = -3.0
        M, rbar_target = _compute_modulation(p, R, R_bar=-1.0, r_post=5.0)
        assert M == pytest.approx(R)
        assert rbar_target == pytest.approx(R)

    def test_surprise(self) -> None:
        """M = (r_post - R_bar)^2 for surprise mode; R_bar tracks r_post."""
        p = Params(neuromod_type="surprise")
        r_post, R_bar = 7.0, 5.0
        M, rbar_target = _compute_modulation(p, R=0.0, R_bar=R_bar, r_post=r_post)
        assert M == pytest.approx((r_post - R_bar) ** 2)
        assert rbar_target == pytest.approx(r_post)

    def test_constant(self) -> None:
        """M = 1 for non-modulated STDP."""
        p = Params(neuromod_type="constant")
        M, rbar_target = _compute_modulation(p, R=-5.0, R_bar=-2.0, r_post=5.0)
        assert M == pytest.approx(1.0)
        assert rbar_target == pytest.approx(-2.0)

    def test_linear_target_matches_old_rate_ratio(self) -> None:
        """target_func='linear' + covariance should match the old rate_ratio behavior."""
        alpha = 0.5
        p = Params(
            reward_signal="target_rate",
            neuromod_type="covariance",
            target_func="linear",
            target_func_params=json.dumps({"a": alpha}),
        )
        r_pre, r_post, R_bar = 10.0, 3.0, -1.0
        R = _compute_reward(p, r_pre, r_post)
        M, _ = _compute_modulation(p, R, R_bar, r_post)
        expected_R = -(r_post - alpha * r_pre) ** 2
        assert R == pytest.approx(expected_R)
        assert M == pytest.approx(expected_R - R_bar)
