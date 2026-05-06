"""Main event-driven / continuous simulation loop."""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable

import numpy as np
from tqdm import tqdm

from neuro.dynamics import (
    _advance_state,
    _compute_modulation,
    _compute_reward,
    _crossing_fraction,
    _pack_state,
    _row_from_state,
)
from neuro.params import (
    Params,
    R_POST_IDX,
    RBAR_IDX,
    V_IDX,
    Y_POST_IDX,
    _E_idx,
    _I_s_idx,
    _W_idx,
    _X_pre_idx,
)
from neuro.recording import _build_recorder


def simulate(
    p: Params,
    hdf5_path: str | None = None,
    parquet_path: str | None = None,
    chunk_rows: int = 100_000,
    *,
    progress: Callable[[Iterable[int]], Iterable[int]] | None = None,
):
    """Run the N-pre → 1-post neuromodulated STDP simulation.

    Hybrid event-driven / continuous scheme:
    - Smooth dynamics integrated by Euler or RK4 (_advance_state)
    - Spike events (pre/post) apply instantaneous jumps to traces
    - RK4 path uses threshold-crossing interpolation to split timestep

    Synapse 0 is the "target" (reward-paired) synapse used for contingent
    reward coincidence detection.

    Pass ``progress`` to customize the step-loop progress indicator; it
    takes an iterable of step indices and returns an iterable.  Defaults
    to a tqdm bar on stderr; pass ``lambda it: it`` to silence it, or
    e.g. ``mo.status.progress_bar`` to render a marimo UI widget.
    """
    if progress is None:
        progress = lambda it: tqdm(it, desc="Simulating", unit="step", mininterval=0.5)

    method = p.method.lower()
    if method not in {"euler", "rk4"}:
        raise ValueError("Params.method must be either 'euler' or 'rk4'.")
    use_window = p.rate_mode == "window"
    if p.rate_mode not in {"exp", "window"}:
        raise ValueError("Params.rate_mode must be either 'exp' or 'window'.")

    n_pre = p.n_pre
    n_steps = int(p.T / p.dt)
    rec_step = max(1, int(p.record_every / p.dt))

    rng = np.random.default_rng(p.seed)
    probs = [r * p.dt for r in p.r_pre_rates]
    if not p.poisson:
        periods = [max(1, round(1.0 / (r * p.dt))) if r > 0 else 0
                   for r in p.r_pre_rates]

    y = _pack_state(p)
    ref_remaining = p.ref_remaining0

    post_spike_buf: deque[float] = deque()
    win_r_post = 0.0

    d_reward = 0.0
    reward_schedule: deque[float] = deque()
    use_reward_pulse = p.reward_signal in ("biofeedback", "contingent")

    recent_target_times: deque[float] = deque()

    recorder = _build_recorder(p, hdf5_path=hdf5_path, parquet_path=parquet_path, chunk_rows=chunk_rows)

    for step in progress(range(n_steps)):
        t = step * p.dt

        if use_reward_pulse:
            while reward_schedule and reward_schedule[0] <= t:
                reward_schedule.popleft()
                d_reward += p.reward_amount
            d_reward += p.dt * (-d_reward / p.reward_tau)

        pre_spikes = [0] * n_pre
        for i in range(n_pre):
            if p.poisson:
                pre_spikes[i] = 1 if rng.random() < probs[i] else 0
            else:
                pre_spikes[i] = 1 if (periods[i] > 0 and step % periods[i] == 0) else 0

        for i in range(n_pre):
            if pre_spikes[i]:
                recorder.append_spike(f"pre{i+1}_spike_times", t)
                y[_I_s_idx(i)] += 1.0
                y[_X_pre_idx(i)] += 1.0
                y[_E_idx(i)] -= p.eta_minus * y[_W_idx(i)] * y[Y_POST_IDX]
                if p.reward_signal == "contingent" and i == 0:
                    recent_target_times.append(t)

        post_spike = 0
        is_refractory = 1 if ref_remaining > 0.0 else 0

        if use_window:
            cutoff = t - p.rate_window
            while post_spike_buf and post_spike_buf[0] < cutoff:
                post_spike_buf.popleft()
            win_r_post = len(post_spike_buf) / p.rate_window

        ro = win_r_post if use_window else None

        if p.reward_signal == "contingent":
            cutoff_c = t - p.coincidence_window
            while recent_target_times and recent_target_times[0] < cutoff_c:
                recent_target_times.popleft()

        if method == "euler":
            V = float(y[V_IDX])
            yp = float(y[Y_POST_IDX])
            r_post = float(y[R_POST_IDX])
            R_bar = float(y[RBAR_IDX])

            I_s = [float(y[_I_s_idx(i)]) for i in range(n_pre)]
            x = [float(y[_X_pre_idx(i)]) for i in range(n_pre)]
            E = [float(y[_E_idx(i)]) for i in range(n_pre)]
            w = [float(y[_W_idx(i)]) for i in range(n_pre)]

            for i in range(n_pre):
                I_s[i] += p.dt * (-I_s[i] / p.tau_s)
                x[i] += p.dt * (-x[i] / p.tau_plus)

            if ref_remaining <= 0.0:
                I_total = sum(w[i] * I_s[i] for i in range(n_pre))
                dV = (p.dt / p.tau_m) * (-(V - p.E_L) + p.R_m * (I_total + p.I_ext))
                V_new = V + dV
                if V < p.theta and V_new >= p.theta:
                    post_spike = 1
                    recorder.append_spike("post_spike_times", t)
                    V = p.V_reset
                    ref_remaining = p.tau_ref
                else:
                    V = V_new
            else:
                ref_remaining = max(0.0, ref_remaining - p.dt)
                V = p.V_reset

            if post_spike:
                yp += 1.0
                r_post += 1.0
                for i in range(n_pre):
                    E[i] += p.eta_plus * (p.wmax - w[i]) * x[i]
                if p.reward_signal == "biofeedback":
                    reward_schedule.append(t + p.reward_delay)
                elif p.reward_signal == "contingent" and recent_target_times:
                    reward_schedule.append(t + p.reward_delay)
                if use_window:
                    post_spike_buf.append(t)
                    win_r_post = len(post_spike_buf) / p.rate_window
                    ro = win_r_post

            yp += p.dt * (-yp / p.tau_minus)
            r_post += p.dt * (-r_post / p.tau_r)
            for i in range(n_pre):
                E[i] += p.dt * (-E[i] / p.tau_e)

            rew_pre: float = float(p.r_pre_rates[0])
            rew_post: float = ro if use_window else r_post  # type: ignore[assignment]
            R = _compute_reward(p, rew_pre, rew_post, reward_pulse=d_reward)
            _, rbar_target = _compute_modulation(p, R, R_bar, rew_post)
            R_bar += (p.dt / p.tau_Rbar) * (-R_bar + rbar_target)
            R = _compute_reward(p, rew_pre, rew_post, reward_pulse=d_reward)
            M, _ = _compute_modulation(p, R, R_bar, rew_post)
            for i in range(n_pre):
                w[i] += p.dt * M * E[i]
                w[i] = min(p.wmax, max(0.0, w[i]))

            y[V_IDX] = V
            y[Y_POST_IDX] = yp
            y[R_POST_IDX] = r_post
            y[RBAR_IDX] = R_bar
            for i in range(n_pre):
                y[_I_s_idx(i)] = I_s[i]
                y[_X_pre_idx(i)] = x[i]
                y[_E_idx(i)] = E[i]
                y[_W_idx(i)] = w[i]

        else:
            if ref_remaining <= 0.0:
                v0 = float(y[V_IDX])
                y_trial = _advance_state(y, p.dt, p, method="rk4", voltage_active=True, rate_post=ro, reward_pulse=d_reward)
                frac = _crossing_fraction(v0, float(y_trial[V_IDX]), p.theta)

                if frac is None:
                    y = y_trial
                else:
                    dt1 = frac * p.dt
                    dt2 = p.dt - dt1

                    y_mid = _advance_state(y, dt1, p, method="rk4", voltage_active=True, rate_post=ro, reward_pulse=d_reward)
                    spike_t = t + dt1
                    recorder.append_spike("post_spike_times", spike_t)
                    post_spike = 1

                    y_mid[V_IDX] = p.V_reset
                    y_mid[Y_POST_IDX] += 1.0
                    y_mid[R_POST_IDX] += 1.0
                    for i in range(n_pre):
                        y_mid[_E_idx(i)] += p.eta_plus * (p.wmax - y_mid[_W_idx(i)]) * y_mid[_X_pre_idx(i)]

                    if p.reward_signal == "biofeedback":
                        reward_schedule.append(spike_t + p.reward_delay)
                    elif p.reward_signal == "contingent" and recent_target_times:
                        reward_schedule.append(spike_t + p.reward_delay)

                    if use_window:
                        post_spike_buf.append(spike_t)
                        ro = len(post_spike_buf) / p.rate_window

                    y = _advance_state(y_mid, dt2, p, method="rk4", voltage_active=False, rate_post=ro, reward_pulse=d_reward)
                    y[V_IDX] = p.V_reset
                    ref_remaining = max(0.0, p.tau_ref - dt2)
            else:
                y = _advance_state(y, p.dt, p, method="rk4", voltage_active=False, rate_post=ro, reward_pulse=d_reward)
                y[V_IDX] = p.V_reset
                ref_remaining = max(0.0, ref_remaining - p.dt)

        if step % rec_step == 0:
            recorder.append(
                _row_from_state(
                    t,
                    y,
                    p,
                    pre_spikes=pre_spikes,
                    post_spike=post_spike,
                    is_refractory=is_refractory,
                    rate_post=ro,
                    reward_pulse=d_reward,
                )
            )

    return recorder.finalize()
