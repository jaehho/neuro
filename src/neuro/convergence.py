"""Steady-state detection for scalar trajectories.

Two use cases:

1. **Online early-stop** during ``simulate()``. Pass a
   ``StreamingConvergence`` to the simulation loop; when the monitored
   signal has been flat for *consecutive* successive checks, the loop
   breaks early and the run finishes in < T seconds of simulated time.
2. **Post-hoc verification** that a recorded run actually settled. Call
   ``check_steady_state(times, values, target=...)`` to ask "did the
   last window converge to within tolerance of target?".

Both share the same criterion: split the most recent ``2 * window`` of
samples into two halves, take their means, and call the trajectory
*flat* if the half-to-half drift is within ``abs_tol + rel_tol * scale``
where ``scale`` defaults to ``|target|`` (or the larger of the two
half-means when target is None).

Defaults are tuned for ``r_post`` in Hz on the baseline-style runs;
raise ``window`` if your rate-estimator smoothing is slower than 0.5 s
so the comparison windows still contain independent information.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class ConvergenceCriterion:
    """Tolerance settings for steady-state detection.

    *window* — half-window length in seconds. The detector compares
    ``[t − 2W, t − W]`` against ``[t − W, t]``.
    *rel_tol* — relative tolerance, multiplied by ``scale``.
    *abs_tol* — absolute tolerance in the same units as the signal.
    *consecutive* — number of successive passing checks before declaring
    convergence (debounce against transient flat patches).
    *min_t* — earliest simulated time at which to even check.
    *check_interval* — seconds between checks.
    """
    window: float = 5.0
    rel_tol: float = 0.05
    abs_tol: float = 0.5
    consecutive: int = 3
    min_t: float = 5.0
    check_interval: float = 1.0


def _half_means(times, values, t_now: float, window: float) -> tuple[float, float] | None:
    """Means of the two adjacent halves ending at *t_now*, or None if not enough data."""
    import numpy as np

    times = np.asarray(times)
    values = np.asarray(values)
    t_lo = t_now - 2 * window
    t_mid = t_now - window
    if times.size == 0 or times[0] > t_lo:
        return None
    in_first = (times >= t_lo) & (times < t_mid)
    in_second = (times >= t_mid) & (times <= t_now)
    if not in_first.any() or not in_second.any():
        return None
    return float(values[in_first].mean()), float(values[in_second].mean())


def _passes(m1: float, m2: float, target: float | None, c: ConvergenceCriterion) -> bool:
    scale = abs(target) if target is not None else max(abs(m1), abs(m2), 1.0)
    return abs(m1 - m2) <= c.abs_tol + c.rel_tol * scale


def check_steady_state(
    times,
    values,
    *,
    target: float | None = None,
    criterion: ConvergenceCriterion | None = None,
) -> dict[str, float | bool]:
    """Post-hoc check whether *values* settled at *target* by the end.

    Returns a diagnostic dict with the half-means, half-to-half delta,
    pass/fail under the criterion, and (if target given) the absolute
    error of the latter half against target.
    """
    import numpy as np

    c = criterion or ConvergenceCriterion()
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size < 2:
        return {"converged": False, "reason": "insufficient_samples"}
    t_now = float(times[-1])
    halves = _half_means(times, values, t_now, c.window)
    if halves is None:
        return {"converged": False, "reason": "window_underflow", "t_end": t_now}
    m1, m2 = halves
    out: dict[str, float | bool] = {
        "converged": _passes(m1, m2, target, c),
        "first_half_mean": m1,
        "second_half_mean": m2,
        "delta": abs(m1 - m2),
        "t_end": t_now,
    }
    if target is not None:
        out["target"] = target
        out["abs_error"] = abs(m2 - target)
    return out


class StreamingConvergence:
    """Online steady-state detector for the simulate() loop.

    Append ``(t, value)`` samples as they arrive; ``update`` returns
    True once the signal has passed the criterion ``consecutive`` times
    in a row.  The detector keeps only the last ``2 * window`` of
    samples, so its memory is bounded.
    """

    def __init__(
        self,
        criterion: ConvergenceCriterion | None = None,
        *,
        target: float | None = None,
    ) -> None:
        self.criterion = criterion or ConvergenceCriterion()
        self.target = target
        self._times: deque[float] = deque()
        self._values: deque[float] = deque()
        self._streak = 0
        self._last_check_t: float | None = None
        self.converged_at: float | None = None

    def reset(self) -> None:
        self._times.clear()
        self._values.clear()
        self._streak = 0
        self._last_check_t = None
        self.converged_at = None

    def update(self, t: float, value: float) -> bool:
        c = self.criterion
        self._times.append(t)
        self._values.append(value)
        cutoff = t - 2 * c.window
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
            self._values.popleft()

        if t < c.min_t:
            return False
        if self._last_check_t is not None and t - self._last_check_t < c.check_interval:
            return False
        self._last_check_t = t

        halves = _half_means(self._times, self._values, t, c.window)
        if halves is None:
            return False
        m1, m2 = halves
        if _passes(m1, m2, self.target, c):
            self._streak += 1
            if self._streak >= c.consecutive:
                self.converged_at = t
                return True
        else:
            self._streak = 0
        return False
