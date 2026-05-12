"""Steady-state detection for scalar trajectories.

Two use cases:

1. **Online early-stop** during ``simulate()``. Pass a
   ``StreamingConvergence`` (or a ``MultiConvergence`` composing several)
   to the simulation loop; when every detector has fired, the loop
   breaks early.
2. **Post-hoc verification** that a recorded run actually settled. Call
   ``check_steady_state(times, values, target=...)`` to ask "did the
   last window converge to within tolerance of target?".

Each detector tests two things:

- **Flatness**: split the most recent ``2 * window`` of samples into
  two halves; the half-to-half drift must be within
  ``abs_tol + rel_tol * scale`` (where ``scale = |target|`` if a target
  is supplied, else ``max(|m1|, |m2|, 1)``).
- **On target** (optional): if ``criterion.target_abs_tol`` is set and a
  target is supplied, the second-half mean must additionally satisfy
  ``|m2 - target| <= target_abs_tol + target_rel_tol * |target|``.

Defaults for *flatness* are tuned for ``r_post`` in Hz on baseline runs.
For tighter convergence — e.g. requiring the system to actually reach
the target rather than just stop wiggling — set both target tolerances.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class ConvergenceCriterion:
    """Tolerance settings for steady-state detection.

    *window* — half-window length in seconds. The detector compares
    ``[t − 2W, t − W]`` against ``[t − W, t]``.
    *rel_tol*, *abs_tol* — flatness tolerance: the half-to-half drift.
    *consecutive* — number of successive passing checks (debounce).
    *min_t* — earliest simulated time at which to even check.
    *check_interval* — seconds between checks.
    *target_abs_tol*, *target_rel_tol* — optional on-target tolerance:
    if ``target_abs_tol`` is not None and a target is supplied, the
    second-half mean must additionally be within
    ``target_abs_tol + target_rel_tol * |target|`` of the target.
    """
    window: float = 5.0
    rel_tol: float = 0.05
    abs_tol: float = 0.5
    consecutive: int = 3
    min_t: float = 5.0
    check_interval: float = 1.0
    target_abs_tol: float | None = None
    target_rel_tol: float = 0.0


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


def _passes_flatness(m1: float, m2: float, target: float | None, c: ConvergenceCriterion) -> bool:
    scale = abs(target) if target is not None else max(abs(m1), abs(m2), 1.0)
    return abs(m1 - m2) <= c.abs_tol + c.rel_tol * scale


def _passes_target(m2: float, target: float | None, c: ConvergenceCriterion) -> bool:
    if c.target_abs_tol is None or target is None:
        return True
    return abs(m2 - target) <= c.target_abs_tol + c.target_rel_tol * abs(target)


def _passes(m1: float, m2: float, target: float | None, c: ConvergenceCriterion) -> bool:
    return _passes_flatness(m1, m2, target, c) and _passes_target(m2, target, c)


def check_steady_state(
    times,
    values,
    *,
    target: float | None = None,
    criterion: ConvergenceCriterion | None = None,
) -> dict[str, float | bool | str]:
    """Post-hoc check whether *values* settled (and optionally hit *target*).

    Returns a diagnostic dict with the half-means, half-to-half delta,
    pass/fail under the criterion, and (if target given) the absolute
    error of the latter half against target. ``converged`` is True only
    if both flatness AND (if requested) on-target tests pass.
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
    flat = _passes_flatness(m1, m2, target, c)
    on_target = _passes_target(m2, target, c)
    out: dict[str, float | bool | str] = {
        "converged": flat and on_target,
        "flat": flat,
        "on_target": on_target,
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

    *signal* — optional name. When set, ``update(t, **signals)`` will
    pull the value from ``signals[signal]``. When not set, callers must
    pass the value positionally as the second argument.
    """

    def __init__(
        self,
        criterion: ConvergenceCriterion | None = None,
        *,
        target: float | None = None,
        signal: str | None = None,
    ) -> None:
        self.criterion = criterion or ConvergenceCriterion()
        self.target = target
        self.signal = signal
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

    def update(self, t: float, value: float | None = None, **signals: float) -> bool:
        if value is None:
            if self.signal is None:
                raise TypeError(
                    "StreamingConvergence.update: pass `value` positionally or "
                    "set `signal=` and pass the named kwarg."
                )
            value = signals[self.signal]
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


class MultiConvergence:
    """Composite detector: fires when every child has fired (at any past t).

    Children must each have a ``signal`` set; ``update(t, **signals)``
    routes each named value to the matching child.  Once the composite
    fires, ``converged_at`` is set to the time at which the *last*
    child finished and stays set.
    """

    def __init__(self, *children: StreamingConvergence) -> None:
        for c in children:
            if c.signal is None:
                raise ValueError("Every child of MultiConvergence must have `signal=` set.")
        self.children: list[StreamingConvergence] = list(children)
        self.converged_at: float | None = None

    def reset(self) -> None:
        for c in self.children:
            c.reset()
        self.converged_at = None

    def update(self, t: float, value: float | None = None, **signals: float) -> bool:
        # `value` is ignored — composite always uses named kwargs.
        for child in self.children:
            assert child.signal is not None
            v = signals.get(child.signal)
            if v is None:
                continue  # caller didn't supply this signal this tick
            child.update(t, v)
        if all(c.converged_at is not None for c in self.children) and self.converged_at is None:
            self.converged_at = t
        return self.converged_at is not None
