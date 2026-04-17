"""Rate estimation: exp trace vs sliding window, constant-ISI convergence.

Poisson extension lives in `notebooks/poisson_firing.py`. No simulation
required; synthesizes spike trains directly.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Rate estimation: exponential trace vs sliding window

    `Params.rate_mode = "exp"` runs a leaky integrator $\dot r = -r/\tau_r + \sum_k \delta(t - t_k)$
    and reports $r/\tau_r$ in Hz. `rate_mode = "window"` does a hard
    spike count over a window $W$. The two have different bias / variance
    / ripple properties on regular vs Poisson inputs.

    These plots are analytical — they don't run the full STDP simulation,
    just the rate estimators on synthesized spike trains.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(constant_spikes, exp_trace_hz, np, plt, poisson_spikes, window_hz):
    DUR = 5.0
    DT_R = 1e-3
    RATE = 21.5
    t_r = np.arange(0, DUR, DT_R)
    spk_const = constant_spikes(RATE, DUR)
    spk_pois = poisson_spikes(RATE, DUR)

    WINS = [0.1, 0.2, 0.5, 1.0]
    TAUS = [0.1, 0.2, 0.5, 1.0]

    fig_re, axes_re = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sliding-window rate estimator — convergence analysis

    `rate_mode = "window"` (`sim.py:881`) computes

    $$
    \hat{r}(t;\,W) = \frac{N([t-W,\,t])}{W} \quad (\text{spikes/s}),
    $$

    where $N([a, b])$ is the count of spikes in the interval
    $[a, b]$, $t$ is the current time, $W > 0$ is the window
    length, and $r$ is the firing rate in spikes per second
    ($r = 1/\text{ISI}$ for periodic firing).

    Consider a rate change: the presynaptic neuron fires at a
    constant rate $r_0$ for times $s < 0$, then switches to a
    different constant rate $r_1$ for $s \ge 0$. Let $t \ge 0$
    be the time since the change. We want $\hat r(t;\,W)$ to
    be close to the new rate $r_1$.

    **Theorem.** *Assume constant inter-spike intervals on each
    side of the rate change: spikes at times $k/r_0$ for integers
    $k < 0$ and $k/r_1$ for integers $k \ge 0$. For all $W > 0$
    and $t \ge W$,*

    $$
    |\hat r(t;\,W) - r_1| \le 1/W.
    $$

    The proof has two parts: a single-regime Lemma giving
    $|\hat r - r| \le 1/W$ in steady state, then a case split
    on $W$ vs $t$ at the rate change. The case $W > t$ is not
    needed for the theorem but is worked out anyway, since it
    shows why the $t \ge W$ clause is essential (without it,
    a systematic error persists).

    Standard treatments: Dayan & Abbott (2001) §1.4; Gerstner et al.
    (2014) §7.2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single-regime bound

    **Lemma.** *Let the spike train have constant inter-spike
    interval $1/r$ with $r > 0$ (spikes at times $k/r$ for all
    integers $k$). For every $t \in \mathbb{R}$ and $W > 0$,*

    $$
    |\hat r(t;\,W) - r| \le 1/W.
    $$

    *Proof.* $N([t - W,\,t])$ counts the integers $k$ with
    $t - W \le k/r \le t$, equivalently $(t - W)r \le k \le tr$.
    This interval has length $rW$, so it contains either
    $\lfloor rW \rfloor$ or $\lfloor rW \rfloor + 1$ integers;
    hence $|N - rW| \le 1$. Dividing by $W$,
    $|\hat r(t;\,W) - r| = |N - rW|/W \le 1/W$. $\blacksquare$

    More precisely (non-integer $rW$, off-grid window edges):
    $N = \lceil rW \rceil$ when $\{tr\} < \{rW\}$ and
    $N = \lfloor rW \rfloor$ when $\{tr\} > \{rW\}$, where
    $\{x\}$ denotes the fractional part of $x$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rate change

    *Proof of Theorem.* The window $[t - W,\, t]$ relative to
    the rate-change point at time 0 splits into two cases.

    **Case $t \ge W$:** $t - W \ge 0$, so the window lies
    entirely in $[0,\infty)$, where the spike train is periodic
    at rate $r_1$. Applying the Lemma with $r = r_1$:

    $$
    |\hat r(t;\,W) - r_1| \le 1/W,
    $$

    completing the Theorem. $\blacksquare$

    ### The case $W > t$: why the $t \ge W$ clause is needed

    If $W > t$, the window's left edge $t - W$ is negative, so
    the window covers both regimes. A segment of length $W - t$
    is in the old regime (rate $r_0$), and a segment of length
    $t$ is in the new (rate $r_1$). Applying the same
    integer-count argument as in the Lemma to each segment
    separately: the old segment's count is
    $\lfloor r_0(W - t) \rfloor$ or $\lceil r_0(W - t) \rceil$,
    and the new's is $\lfloor r_1 t \rfloor$ or
    $\lceil r_1 t \rceil$. Summing, the total count differs
    from $r_0(W - t) + r_1 t$ by less than 2. Dividing by $W$
    and subtracting the target $r_1$,

    $$
    \hat r(t;\,W) - r_1 = (r_0 - r_1)\,\frac{W - t}{W} \pm \frac{2}{W}.
    $$

    The first term is a *systematic error* from mixing spikes
    at two different rates inside one window. It equals
    $r_0 - r_1$ at $t = 0$ (window all in the old regime), 0
    at $t = W$ (window just past the change), and falls
    linearly in between. The second term, $\pm 2/W$, is the
    floor/ceil phase contribution from the two segments,
    shrinking like $1/W$.

    Without the $t \ge W$ clause, the systematic error can
    persist as $t, W \to \infty$: along the ray $W = 2t$,
    $(W - t)/W = 1/2$, so the error stays at $(r_0 - r_1)/2$
    no matter how large $t$ and $W$ get. The theorem's
    $t \ge W$ clause rules out this regime.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    1. Lemma (single regime, rate $r$): $|\hat r(t;\,W) - r|
       \le 1/W$, from the count being within 1 of $rW$.
    2. Theorem (rate change $r_0 \to r_1$ at time 0): the
       Lemma applied to the new regime handles Case $t \ge W$
       directly, yielding $|\hat r(t;\,W) - r_1| \le 1/W$. For
       $W > t$ the window covers both regimes, producing a
       systematic error $(r_0 - r_1)(W - t)/W$ that the
       $t \ge W$ clause rules out.

    Poisson extension (probabilistic form, bias-variance tradeoff,
    exp-trace comparison): see `notebooks/poisson_firing.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
