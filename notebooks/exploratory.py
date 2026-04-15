"""Exploratory: anything beyond the verified 1-pre + target_rate + covariance path.

Sections:
  1. Spatial credit assignment — 2 pre + contingent reward
  2. Neuromodulator types — covariance vs gated vs surprise vs constant
  3. Reward signals — target_rate vs biofeedback vs contingent
  4. Target-function shapes (linear, quadratic, sqrt, …)
  5. Rate estimation — exponential trace vs sliding window
  6. Parameter sensitivity sweep template

Promote anything verified out of here into `notebooks/verified.py`.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from neuro.sim import Params, simulate
    return Params, np, pl, plt, simulate


# ─────────────────────────────────────────────────────────────────────
# Section 1: spatial credit assignment with contingent reward
# ─────────────────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        # 1. Spatial credit assignment — 2 pre + contingent reward

        Two pre-synaptic neurons fire at the same rate. Reward is delivered
        only when **pre1** (the "target" synapse) fires within a coincidence
        window before a post-spike. Both synapses see the same global
        modulator $M$, but only $E_1$ is high when reward arrives, so only
        $w_1$ should grow.

        Reference: Izhikevich (2007), Frémaux & Gerstner (2016) Eq. 10.
        """
    )
    return


@app.cell
def _(Params, simulate):
    p_credit = Params(
        T=20.0,
        dt=1e-4,
        method="rk4",
        n_pre=2,
        r_pre_rates=(20.0, 20.0),
        poisson=True,
        w0=(2.0, 2.0),
        reward_signal="contingent",
        coincidence_window=0.02,
        reward_delay=0.5,
        reward_amount=1.0,
        reward_tau=0.2,
        neuromod_type="covariance",
        record_every=1e-3,
    )
    rec_credit = simulate(p_credit)
    return p_credit, rec_credit


@app.cell
def _(plt, rec_credit):
    fig_c, axes_c = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    axes_c[0].plot(rec_credit["t"], rec_credit["w1"], label="w₁ (target)", color="tab:orange", lw=1.0)
    axes_c[0].plot(rec_credit["t"], rec_credit["w2"], label="w₂ (distractor)", color="tab:blue", lw=1.0)
    axes_c[0].set_ylabel("weight")
    axes_c[0].set_title("Spatial credit assignment: only w₁ should grow")
    axes_c[0].legend(loc="upper right", fontsize=8)

    axes_c[1].plot(rec_credit["t"], rec_credit["E1"], label="E₁", color="tab:orange", lw=0.5, alpha=0.7)
    axes_c[1].plot(rec_credit["t"], rec_credit["E2"], label="E₂", color="tab:blue", lw=0.5, alpha=0.7)
    axes_c[1].axhline(0, ls=":", color="gray", alpha=0.4)
    axes_c[1].set_ylabel("eligibility")
    axes_c[1].legend(loc="upper right", fontsize=8)

    axes_c[2].plot(rec_credit["t"], rec_credit["M"], color="tab:red", lw=0.5)
    axes_c[2].axhline(0, ls=":", color="gray", alpha=0.4)
    axes_c[2].set_xlabel("Time (s)")
    axes_c[2].set_ylabel("M")
    axes_c[2].set_title("Global modulator (same for both synapses)")

    fig_c.tight_layout()
    fig_c
    return


# ─────────────────────────────────────────────────────────────────────
# Section 2: neuromodulator type comparison
# ─────────────────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        # 2. Neuromodulator types

        Same reward signal (`target_rate`), different rules for $M$:

        | type        | $M$                      |
        |-------------|--------------------------|
        | covariance  | $R - \bar R$ (RPE)       |
        | gated       | $R$                      |
        | surprise    | $(r_\text{post} - \bar R)^2$ |
        | constant    | $1$ (non-modulated STDP) |
        """
    )
    return


@app.cell
def _(Params, simulate):
    _NEUROMOD_TYPES = ["covariance", "gated", "surprise", "constant"]
    runs_neuromod = {}
    for _nm in _NEUROMOD_TYPES:
        _p_nm = Params(
            T=10.0,
            dt=1e-4,
            method="rk4",
            n_pre=1,
            r_pre_rates=(20.0,),
            poisson=True,
            w0=(2.0,),
            reward_signal="target_rate",
            target_func="fixed",
            r_target=10.0,
            neuromod_type=_nm,
            record_every=1e-3,
        )
        runs_neuromod[_nm] = (_p_nm, simulate(_p_nm))
    return (runs_neuromod,)


@app.cell
def _(plt, runs_neuromod):
    fig_nm, axes_nm = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for _nm, (_p_nm, _rec_nm) in runs_neuromod.items():
        axes_nm[0].plot(_rec_nm["t"], _rec_nm["w1"], label=_nm, lw=0.8, alpha=0.85)
        axes_nm[1].plot(_rec_nm["t"], _rec_nm["r_post"], label=_nm, lw=0.5, alpha=0.85)
        axes_nm[2].plot(_rec_nm["t"], _rec_nm["M"], label=_nm, lw=0.4, alpha=0.7)
        axes_nm[3].plot(_rec_nm["t"], _rec_nm["E1"], label=_nm, lw=0.4, alpha=0.7)

    _titles_nm = ["w₁", "r_post (Hz)", "M", "E₁"]
    for _ax_nm, _t_nm in zip(axes_nm, _titles_nm):
        _ax_nm.set_ylabel(_t_nm)
        _ax_nm.legend(loc="upper right", fontsize=8, ncol=2)
        _ax_nm.grid(True, alpha=0.15)
    axes_nm[1].axhline(10.0, ls="--", color="black", alpha=0.5)
    axes_nm[-1].set_xlabel("Time (s)")
    fig_nm.tight_layout()
    fig_nm
    return


# ─────────────────────────────────────────────────────────────────────
# Section 3: reward signal comparison
# ─────────────────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        # 3. Reward signals

        Holding `neuromod_type="covariance"`, swap the reward source:

        - **target_rate** — $R = -(r_\text{post} - r_\text{target})^2$ (self-supervisory demo)
        - **biofeedback** — delayed pulse after every post-spike (Legenstein 2008)
        - **contingent** — delayed pulse only after pre→post coincidences (Izhikevich 2007)
        """
    )
    return


@app.cell
def _(Params, simulate):
    _REWARD_CONFIGS = {
        "target_rate": dict(reward_signal="target_rate", target_func="fixed", r_target=10.0),
        "biofeedback": dict(reward_signal="biofeedback", reward_delay=0.5, reward_amount=1.0, reward_tau=0.2),
        "contingent":  dict(reward_signal="contingent",  reward_delay=0.5, reward_amount=1.0, reward_tau=0.2,
                            coincidence_window=0.02),
    }
    runs_reward = {}
    for _name_r, _cfg_r in _REWARD_CONFIGS.items():
        # contingent needs n_pre=2 to be meaningful (target + distractor)
        _n_r = 2 if _cfg_r["reward_signal"] == "contingent" else 1
        _p_r = Params(
            T=10.0, dt=1e-4, method="rk4",
            n_pre=_n_r,
            r_pre_rates=tuple(20.0 for _ in range(_n_r)),
            poisson=True,
            w0=tuple(2.0 for _ in range(_n_r)),
            neuromod_type="covariance",
            record_every=1e-3,
            **_cfg_r,
        )
        runs_reward[_name_r] = (_p_r, simulate(_p_r))
    return (runs_reward,)


@app.cell
def _(plt, runs_reward):
    fig_r, axes_r = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for _name_r, (_p_r, _rec_r) in runs_reward.items():
        axes_r[0].plot(_rec_r["t"], _rec_r["w1"], label=f"{_name_r}: w₁", lw=0.8)
        axes_r[1].plot(_rec_r["t"], _rec_r["R"], label=_name_r, lw=0.4, alpha=0.7)
        axes_r[2].plot(_rec_r["t"], _rec_r["M"], label=_name_r, lw=0.4, alpha=0.7)
    axes_r[0].set_ylabel("w₁")
    axes_r[0].set_title("Weight trajectories")
    axes_r[0].legend(loc="upper right", fontsize=8)
    axes_r[1].set_ylabel("R")
    axes_r[1].legend(loc="upper right", fontsize=8)
    axes_r[2].set_ylabel("M")
    axes_r[2].set_xlabel("Time (s)")
    axes_r[2].legend(loc="upper right", fontsize=8)
    for _ax_r in axes_r:
        _ax_r.grid(True, alpha=0.15)
    fig_r.tight_layout()
    fig_r
    return


# ─────────────────────────────────────────────────────────────────────
# Section 4: target function shapes
# ─────────────────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        # 4. Target-function shapes

        With `reward_signal="target_rate"`, the target rate can be a
        function of $r_\text{pre}$ instead of a fixed constant. Useful
        for studying input/output transfer functions.
        """
    )
    return


@app.cell
def _(Params, simulate):
    _TARGET_FUNCS = ["fixed", "linear", "sqrt", "log"]
    runs_target = {}
    for _tf in _TARGET_FUNCS:
        _p_tf = Params(
            T=10.0, dt=1e-4, method="rk4",
            n_pre=1, r_pre_rates=(20.0,), poisson=True, w0=(2.0,),
            reward_signal="target_rate",
            target_func=_tf,
            r_target=10.0, alpha=0.5,
            neuromod_type="covariance",
            record_every=1e-3,
        )
        runs_target[_tf] = (_p_tf, simulate(_p_tf))
    return (runs_target,)


@app.cell
def _(plt, runs_target):
    fig_t, axes_t = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for _tf, (_p_tf, _rec_tf) in runs_target.items():
        axes_t[0].plot(_rec_tf["t"], _rec_tf["w1"], label=_tf, lw=0.8)
        axes_t[1].plot(_rec_tf["t"], _rec_tf["r_post"], label=_tf, lw=0.5, alpha=0.8)
    axes_t[0].set_ylabel("w₁")
    axes_t[0].legend(loc="upper right", fontsize=8)
    axes_t[1].set_ylabel("r_post (Hz)")
    axes_t[1].set_xlabel("Time (s)")
    axes_t[1].legend(loc="upper right", fontsize=8)
    for _ax_t in axes_t:
        _ax_t.grid(True, alpha=0.15)
    fig_t.tight_layout()
    fig_t
    return


# ─────────────────────────────────────────────────────────────────────
# Section 5: rate estimation — exp trace vs sliding window
# ─────────────────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        # 5. Rate estimation: exponential trace vs sliding window

        `Params.rate_mode = "exp"` runs a leaky integrator $\dot r = -r/\tau_r + \sum_k \delta(t - t_k)$
        and reports $r/\tau_r$ in Hz. `rate_mode = "window"` does a hard
        spike count over a window $W$. The two have different bias / variance
        / ripple properties on regular vs Poisson inputs.

        These plots are analytical — they don't run the full STDP simulation,
        just the rate estimators on synthesized spike trains.
        """
    )
    return


@app.cell
def _(np):
    def constant_spikes(rate, duration):
        isi = 1.0 / rate
        return np.arange(isi, duration + isi / 2, isi)

    def poisson_spikes(rate, duration, seed=42):
        rng = np.random.default_rng(seed)
        n = int(duration / 1e-4)
        return np.where(rng.random(n) < rate * 1e-4)[0] * 1e-4

    def window_hz(spikes, t, W):
        right = np.searchsorted(spikes, t, side="left")
        left = np.searchsorted(spikes, t - W, side="left")
        return (right - left).astype(float) / W

    def exp_trace_hz(spikes, t, tau_r):
        dt_step = t[1] - t[0]
        r = np.zeros(len(t))
        decay = np.exp(-dt_step / tau_r)
        si = 0
        for i in range(1, len(t)):
            r[i] = r[i - 1] * decay
            while si < len(spikes) and spikes[si] < t[i]:
                r[i] += 1.0
                si += 1
        return r / tau_r
    return constant_spikes, exp_trace_hz, poisson_spikes, window_hz


@app.cell
def _(constant_spikes, exp_trace_hz, np, plt, poisson_spikes, window_hz):
    DUR = 5.0
    DT_R = 1e-3
    RATE = 20.0
    t_r = np.arange(0, DUR, DT_R)
    spk_const = constant_spikes(RATE, DUR)
    spk_pois = poisson_spikes(RATE, DUR)

    WINS = [0.1, 0.2, 0.5, 1.0]
    TAUS = [0.1, 0.2, 0.5, 1.0]

    fig_re, axes_re = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)

    for ax_re, w in zip([axes_re[0, 0]] * len(WINS), [None]):
        pass

    for w in WINS:
        axes_re[0, 0].plot(t_r, window_hz(spk_const, t_r, w), lw=0.6, label=f"W={w}")
        axes_re[0, 1].plot(t_r, window_hz(spk_pois, t_r, w), lw=0.6, label=f"W={w}")
    for tau in TAUS:
        axes_re[1, 0].plot(t_r, exp_trace_hz(spk_const, t_r, tau), lw=0.6, label=f"τ={tau}")
        axes_re[1, 1].plot(t_r, exp_trace_hz(spk_pois, t_r, tau), lw=0.6, label=f"τ={tau}")

    titles_re = [["Window — regular", "Window — Poisson"],
                 ["Exp trace — regular", "Exp trace — Poisson"]]
    for i in range(2):
        for j in range(2):
            axes_re[i, j].axhline(RATE, ls="--", color="gray", alpha=0.5)
            axes_re[i, j].set_title(titles_re[i][j])
            axes_re[i, j].legend(fontsize=7, loc="upper right")
            axes_re[i, j].grid(True, alpha=0.15)
    axes_re[1, 0].set_xlabel("Time (s)")
    axes_re[1, 1].set_xlabel("Time (s)")
    axes_re[0, 0].set_ylabel("Rate (Hz)")
    axes_re[1, 0].set_ylabel("Rate (Hz)")
    fig_re.tight_layout()
    fig_re
    return


# ─────────────────────────────────────────────────────────────────────
# Section 6: parameter sensitivity sweep
# ─────────────────────────────────────────────────────────────────────


@app.cell
def _(mo):
    mo.md(
        r"""
        # 6. Parameter sensitivity sweep

        Sweep one parameter at a time, hold the rest at the verified
        defaults. Records final weight and late-half post rate.

        Edit `SWEEP_PARAM` and `SWEEP_VALUES` in the next cell.
        """
    )
    return


@app.cell
def _(mo):
    sweep_param_select = mo.ui.dropdown(
        options=["alpha", "eta_plus", "tau_e", "tau_Rbar", "w0", "R_m", "r_target"],
        value="tau_e",
        label="parameter",
    )
    sweep_param_select
    return (sweep_param_select,)


@app.cell
def _(Params, np, simulate, sweep_param_select):
    SWEEP_GRIDS = {
        "alpha":    np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        "eta_plus": np.array([3e-5, 1e-4, 3e-4, 1e-3]),
        "tau_e":    np.array([0.1, 0.2, 0.5, 1.0, 2.0]),
        "tau_Rbar": np.array([1.0, 2.0, 5.0, 10.0, 20.0]),
        "w0":       np.array([0.5, 1.0, 2.0, 5.0, 8.0]),
        "R_m":      np.array([10.0, 20.0, 50.0, 80.0, 120.0]),
        "r_target": np.array([2.0, 5.0, 10.0, 20.0, 30.0]),
    }
    pname = sweep_param_select.value
    values_s = SWEEP_GRIDS[pname]
    sweep_results = []
    for v in values_s:
        kw_s = {pname: (float(v),) if pname == "w0" else float(v)}
        p_s = Params(
            T=10.0, dt=1e-4, method="rk4",
            n_pre=1, r_pre_rates=(20.0,), poisson=True,
            reward_signal="target_rate", target_func="fixed", r_target=10.0,
            neuromod_type="covariance",
            record_every=1e-3,
            **kw_s,
        )
        rec_s = simulate(p_s)
        half = p_s.T / 2
        late = rec_s["post_spike_times"][rec_s["post_spike_times"] >= half]
        rate_s = len(late) / (p_s.T - half) if p_s.T > half else 0.0
        sweep_results.append({"v": float(v), "w_final": float(rec_s["w1"][-1]), "rate": rate_s})
    return pname, sweep_results, values_s


@app.cell
def _(pl, plt, pname, sweep_results):
    df_s = pl.DataFrame(sweep_results)
    fig_s, axes_s = plt.subplots(1, 2, figsize=(11, 4))
    axes_s[0].plot(df_s["v"], df_s["w_final"], "o-", color="tab:orange")
    axes_s[0].set_xlabel(pname)
    axes_s[0].set_ylabel("w_final")
    axes_s[0].set_title(f"Final weight vs {pname}")
    axes_s[0].grid(True, alpha=0.2)

    axes_s[1].plot(df_s["v"], df_s["rate"], "o-", color="tab:red")
    axes_s[1].axhline(10.0, ls="--", color="black", alpha=0.5, label="target")
    axes_s[1].set_xlabel(pname)
    axes_s[1].set_ylabel("late post rate (Hz)")
    axes_s[1].set_title(f"Post rate vs {pname}")
    axes_s[1].legend()
    axes_s[1].grid(True, alpha=0.2)
    fig_s.tight_layout()
    fig_s
    return


if __name__ == "__main__":
    app.run()
