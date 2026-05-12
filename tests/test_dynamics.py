"""Unit tests for the reward and modulation rules."""
from __future__ import annotations

import pytest

from neuro import Params
from neuro.dynamics import _modulation, _reward


class TestReward:
    def test_target_rate_is_quadratic_penalty(self) -> None:
        p = Params(R_rule="target_rate", r_target=10.0)
        assert _reward(p, 10.0) == pytest.approx(0.0)
        assert _reward(p, 7.0) == pytest.approx(-9.0)
        assert _reward(p, 13.0) == pytest.approx(-9.0)

    def test_target_rate_linear_is_signed_error(self) -> None:
        p = Params(R_rule="target_rate_linear", r_target=10.0)
        assert _reward(p, 10.0) == pytest.approx(0.0)
        assert _reward(p, 7.0) == pytest.approx(3.0)     # below target → positive
        assert _reward(p, 13.0) == pytest.approx(-3.0)   # above target → negative

    def test_unknown_signal_raises(self) -> None:
        p = Params()
        object.__setattr__(p, "R_rule", "banana")
        with pytest.raises(ValueError, match="Unknown R_rule"):
            _reward(p, 10.0)



class TestModulation:
    def test_covariance(self) -> None:
        p = Params(M_rule="covariance")
        assert _modulation(p, R=-4.0, R_bar=-2.0) == pytest.approx(-2.0)

    def test_gated(self) -> None:
        p = Params(M_rule="gated")
        assert _modulation(p, R=-4.0, R_bar=-2.0) == pytest.approx(-4.0)

    def test_unknown_mode_raises(self) -> None:
        p = Params()
        object.__setattr__(p, "M_rule", "banana")
        with pytest.raises(ValueError, match="Unknown M_rule"):
            _modulation(p, R=0.0, R_bar=0.0)
