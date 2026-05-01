"""Rate estimation: exp trace vs sliding window — constant-ISI and Poisson.

Two halves:
  1. Constant-ISI analysis. For periodic spikes at rate r_1 on t >= 0
     (possibly following a step from r_0), |r_hat(t;W) - r_1| <= 1/W
     deterministically once t >= W.
  2. Poisson extension. Same bound becomes probabilistic: Theorems 1-4
     develop bias, variance, convergence in probability (Chebyshev),
     almost-sure convergence (SLLN), plus the MSE tradeoff and
     comparison with the exponential trace.

No simulation required — synthesizes spike trains directly.
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
    # Part 1 — Constant-ISI convergence

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


@app.cell
def _(mo):
    mo.md(r"""
    # Part 2 — Poisson firing

    Real neurons do not fire with constant ISI; their spike trains
    are variable. A standard model is a homogeneous Poisson
    process: in any interval of length $L$, the spike count is
    Poisson-distributed with mean $r L$, and counts on disjoint
    intervals are independent.

    Under this model $\hat r(t;\,W)$ is a random variable. Even
    with $t \ge W$, a realization can land any distance from $r_1$
    with positive probability, so "$|\hat r - r_1| < \epsilon$"
    cannot hold deterministically. A precise version adds a second
    tolerance $\delta$ on the probability of failure:

    > *For all $\epsilon, \delta > 0$ there exists $M$ such that*
    > *$t \ge W \ge M$ implies*
    > *$P(|\hat r(t;\,W) - r_1| \ge \epsilon) < \delta.$*

    Theorems 1–4 below develop this form. The rate-change
    expectation carries over from the constant-ISI case directly
    (Campbell's theorem, now exact rather than approximate).

    Standard treatments: Dayan & Abbott (2001) §1.4; Gerstner et
    al. (2014) §7.2.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bias and variance (stationary Poisson)

    **Theorem 1 (unbiased).** *Let $N$ be a homogeneous Poisson
    process with rate $r > 0$. For all $t, W > 0$,
    $E[\hat r(t;W)] = r$.*

    *Proof.* $E[N([t{-}W,t])] = r W$ by definition; divide by
    $W$. $\blacksquare$

    **Theorem 2 (variance).** *Under the same hypothesis,
    $\operatorname{Var}[\hat r(t;W)] = r / W$.*

    *Proof.* $\operatorname{Var}[N([t{-}W,t])] = r W$ (Poisson
    mean = variance); divide by $W^2$. $\blacksquare$

    So the estimator has zero bias and $1/\sqrt{W}$ RMS error,
    same scaling as any sample-mean CLT.
    """)
    return


@app.cell
def _(np):
    def poisson_ct(rate, duration, rng):
        """Homogeneous Poisson on [0, duration] via order-statistics trick."""
        if rate <= 0:
            return np.empty(0)
        n = rng.poisson(rate * duration)
        return np.sort(rng.uniform(0.0, duration, size=n))

    return (poisson_ct,)


@app.cell
def _(np, plt):
    R_D = 20.0
    WS_DIST = [0.5, 2.0, 10.0]
    N_TRIALS_D = 6000
    rng_d = np.random.default_rng(0)

    fig_d, axes_d = plt.subplots(1, len(WS_DIST), figsize=(12, 3.3))
    for ax_d, W_d in zip(axes_d, WS_DIST):
        counts = rng_d.poisson(R_D * W_d, size=N_TRIALS_D)
        est_d = counts / W_d
        lo, hi = est_d.min(), est_d.max()
        ax_d.hist(est_d, bins=30, density=True, color="tab:blue", alpha=0.7)
        xs = np.linspace(lo, hi, 300)
        sd = np.sqrt(R_D / W_d)
        ax_d.plot(xs, np.exp(-0.5 * ((xs - R_D) / sd) ** 2)
                  / (sd * np.sqrt(2 * np.pi)),
                  color="tab:red", lw=1.5, label=r"$\mathcal{N}(r, r/W)$")
        ax_d.axvline(R_D, ls="--", color="gray", alpha=0.6)
        ax_d.set_title(f"W = {W_d} s    var = r/W = {R_D / W_d:.3f}")
        ax_d.set_xlabel(r"$\hat r$ (Hz)")
        ax_d.legend(fontsize=8)
    axes_d[0].set_ylabel("density")
    fig_d.suptitle(rf"Distribution of $\hat r(t;W)$ — r = {R_D} Hz", y=1.02)
    fig_d.tight_layout()
    fig_d
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Convergence in probability (Chebyshev)

    **Theorem 3 (convergence in probability).** *Let $N$ be
    homogeneous Poisson with rate $r > 0$. For every
    $\epsilon > 0$ and $\delta > 0$, set $M = r /
    (\delta\epsilon^2)$. Then for all $W > M$ and all $t \ge W$,*

    $$
    P\!\left(\big|\hat r(t;W) - r\big| \ge \epsilon\right)
    \;<\; \delta.
    $$

    *Proof.* By Chebyshev and Theorems 1–2,
    $$
    P(|\hat r - r| \ge \epsilon) \;\le\;
    \frac{\operatorname{Var}[\hat r]}{\epsilon^2}
    \;=\; \frac{r}{W\epsilon^2}.
    $$
    Requiring the RHS $< \delta$ gives $W > r/(\delta\epsilon^2)
    = M$. $\blacksquare$

    The correct quantifier is $\forall \epsilon, \delta > 0$: one
    tolerance ($\epsilon$) for the error and one ($\delta$) for the
    probability of failure. Chebyshev does not use the Poisson
    structure and is loose; the exact tail from the Poisson CDF is
    much tighter (sub-exponential in $W$).
    """)
    return


@app.cell
def _(np, plt):
    from scipy import stats as spstats
    R_C = 20.0
    EPS = 2.0
    WS_C = np.geomspace(0.05, 20.0, 30)
    # Exact two-sided tail: P(N >= ceil(W(r+ε))) + P(N <= floor(W(r-ε)))
    exact_tail = np.array([
        (1.0 - spstats.poisson.cdf(int(np.ceil(W * (R_C + EPS))) - 1, R_C * W))
        + spstats.poisson.cdf(int(np.floor(W * (R_C - EPS))), R_C * W)
        for W in WS_C
    ])
    cheb = np.minimum(1.0, R_C / (WS_C * EPS ** 2))
    rng_c = np.random.default_rng(1)
    N_C = 30000
    emp = np.array([
        np.mean(np.abs(rng_c.poisson(R_C * W, size=N_C) / W - R_C) >= EPS)
        for W in WS_C
    ])

    fig_chv, ax_chv = plt.subplots(figsize=(7.5, 4.0))
    ax_chv.loglog(WS_C, cheb, color="tab:red", lw=1.5,
                  label=r"Chebyshev $r/(W\epsilon^2)$")
    ax_chv.loglog(WS_C, np.maximum(exact_tail, 1e-12), color="tab:orange", lw=1.5,
                  label="Exact Poisson tail")
    ax_chv.loglog(WS_C, np.maximum(emp, 1e-12), "o", color="tab:blue", ms=4,
                  label=f"Empirical ({N_C} trials)")
    delta_mark = 0.05
    M_mark = R_C / (delta_mark * EPS ** 2)
    ax_chv.axvline(M_mark, ls=":", color="tab:red", alpha=0.7,
                   label=rf"$M=r/(\delta\epsilon^2)$, δ=0.05 → W={M_mark:.0f}")
    ax_chv.axhline(delta_mark, ls=":", color="gray", alpha=0.5)
    ax_chv.set_xlabel("W  (s)")
    ax_chv.set_ylabel(r"$P(|\hat r - r| \geq \epsilon)$")
    ax_chv.set_title(rf"Convergence in probability — r = {R_C} Hz, ε = {EPS} Hz")
    ax_chv.legend(fontsize=8)
    ax_chv.grid(True, which="both", alpha=0.2)
    fig_chv.tight_layout()
    fig_chv
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Almost-sure convergence (strong law)

    **Theorem 4 (SLLN).** *Let $N$ be homogeneous Poisson with rate
    $r > 0$. For any sequence $W_k \to \infty$ and $t_k \ge
    W_k$,*

    $$
    \hat r(t_k;\,W_k)\;\xrightarrow{\text{a.s.}}\;r
    \quad\text{as}\;k\to\infty.
    $$

    *Proof sketch.* Inter-arrival times are i.i.d.
    $\operatorname{Exp}(r)$, and counts on disjoint unit
    intervals are i.i.d. $\operatorname{Poisson}(r)$ with finite
    mean. Kolmogorov's SLLN gives $N([0,n])/n \to r$ a.s. along
    integer $n$, and a sandwiching argument (Poisson counts are
    monotone in the interval) extends this to continuous $W \to
    \infty$. $\blacksquare$

    Almost-sure convergence is stronger than convergence in
    probability: it says that for *almost every* realization
    $\omega$, and every $\epsilon > 0$, there exists $M(\omega)$ such
    that $W > M(\omega)$ implies $|\hat r(t;W;\omega) - r|
    < \epsilon$. The $M$ depends on the sample path.
    """)
    return


@app.cell
def _(np, plt, poisson_ct):
    R_S = 20.0
    DUR_S = 60.0
    WS_S = np.geomspace(0.02, 30.0, 200)

    rng_sp = np.random.default_rng(2)
    N_PATHS = 10
    fig_sp, ax_sp = plt.subplots(figsize=(8, 4.0))
    for _ in range(N_PATHS):
        spikes = poisson_ct(R_S, DUR_S, rng_sp)
        est_sp = np.array([
            np.sum((spikes > DUR_S - W) & (spikes <= DUR_S)) / W for W in WS_S
        ])
        ax_sp.semilogx(WS_S, est_sp, color="tab:blue", alpha=0.35, lw=0.8)
    sigma_sp = np.sqrt(R_S / WS_S)
    ax_sp.fill_between(WS_S, R_S - 2 * sigma_sp, R_S + 2 * sigma_sp,
                       color="tab:red", alpha=0.15,
                       label=r"$r \pm 2\sqrt{r/W}$")
    ax_sp.axhline(R_S, ls="--", color="gray", alpha=0.7, label=r"$r$")
    ax_sp.set_xlabel("W  (s)")
    ax_sp.set_ylabel(r"$\hat r(t;W)$  (Hz)")
    ax_sp.set_title(f"Sample paths concentrate on r — r = {R_S} Hz")
    ax_sp.legend(fontsize=9)
    ax_sp.grid(True, which="both", alpha=0.2)
    fig_sp.tight_layout()
    fig_sp
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Rate change under Poisson firing

    For a Poisson process, $E[N([a, b])] = \int_a^b r(s)\,ds$,
    so the expected estimate for a rate that switches from
    $r_0$ to $r_1$ at time 0 is exactly

    $$
    E[\hat r(t;\,W)] =
    \begin{cases}
    r_1 & t \ge W \\
    \big(r_0(W - t) + r_1 t\big)/W & W > t
    \end{cases}
    $$

    Same piecewise formula as the constant-ISI systematic error,
    but exact: averaging over Poisson counts eliminates the
    $\pm 2/W$ floor/ceil contribution. The bias $E[\hat r] - r_1$
    for $W > t$ is $(r_0 - r_1)(W - t)/W$. The heatmap below shows
    this bias on the $(t, W)$ plane.

    Combining the mean-level bias with the Poisson variance
    $r_1/W$ (Theorem 2) and Chebyshev's inequality, the
    probabilistic form of last week's statement for the
    rate-change case is:

    > *Fix $r_1$ on the new regime. For all*
    > *$\epsilon, \delta > 0$, set $M = r_1/(\delta\epsilon^2)$.*
    > *Then $t \ge W > M$ implies*
    > *$P(|\hat r(t;\,W) - r_1| \ge \epsilon) < \delta.$*
    """)
    return


@app.cell
def _(np, plt):
    R0, R1 = 5.0, 25.0
    ts_b = np.linspace(0.05, 3.0, 140)
    Ws_b = np.linspace(0.05, 3.0, 140)
    Tg, Wg = np.meshgrid(ts_b, Ws_b, indexing="xy")
    bias = np.where(Wg <= Tg, 0.0, (R0 - R1) * (Wg - Tg) / Wg)

    fig_hb, ax_hb = plt.subplots(figsize=(6.8, 4.5))
    vmax = abs(R0 - R1)
    im = ax_hb.imshow(
        bias, origin="lower", aspect="auto",
        extent=(ts_b.min(), ts_b.max(), Ws_b.min(), Ws_b.max()),
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    ax_hb.plot(ts_b, ts_b, "k--", lw=1.0, label="W = t")
    ax_hb.set_xlabel("t  (time since rate change, s)")
    ax_hb.set_ylabel("W  (window size, s)")
    ax_hb.set_title(rf"Bias $E[\hat r] - r_1$ — $r_0={R0}$, $r_1={R1}$ Hz")
    ax_hb.legend(loc="lower right", fontsize=8)
    fig_hb.colorbar(im, ax=ax_hb, label="bias (Hz)")
    fig_hb.tight_layout()
    fig_hb
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Response-time vs variance tradeoff

    In practice (e.g. the rate estimate feeding the reward signal at
    `sim.py:881`) we want (i) quick tracking of a changing rate and
    (ii) low noise. Variance is $r_1/W$ (small $W$ is bad for
    (ii)); step-response bias for $W > t$ is $(r_0 -
    r_1)(W - t)/W$ (large $W$ is bad for (i)). Fix a target
    time $t^\star$ after the change at which you need good tracking;
    the mean-squared error decomposes as

    $$
    \text{MSE}(W;\,t^\star) =
    \begin{cases}
    r_1/W & W \le t^\star \\
    \big((r_0-r_1)(W-t^\star)/W\big)^2 + \big(r_0(W-t^\star) + r_1 t^\star\big)/W^2 & W > t^\star
    \end{cases}
    $$

    The optimum $W^\star$ grows with $t^\star$: if you can wait
    longer after the change, you can afford a larger window for
    lower variance.
    """)
    return


@app.cell
def _(np, plt):
    _R0, _R1 = 5.0, 25.0
    _t_stars = [0.1, 0.3, 1.0, 3.0]
    _Ws = np.geomspace(0.01, 10.0, 500)

    _fig, _ax = plt.subplots(figsize=(7.5, 4.2))
    for _ts in _t_stars:
        _mean_count = np.where(
            _Ws <= _ts,
            _R1 * _Ws,
            _R0 * (_Ws - _ts) + _R1 * _ts,
        )
        _bias = _mean_count / _Ws - _R1
        _var = _mean_count / _Ws ** 2
        _mse = _bias ** 2 + _var
        _line, = _ax.loglog(_Ws, _mse, lw=1.2, label=f"t* = {_ts} s")
        _ax.plot(_Ws[np.argmin(_mse)], _mse.min(), "o", ms=6,
                 color=_line.get_color())
    _ax.set_xlabel("W  (s)")
    _ax.set_ylabel(r"MSE at $t^\star$  (Hz$^2$)")
    _ax.set_title(rf"Optimal window grows with $t^\star$ — $r_0={_R0}\,$Hz → $r_1={_R1}\,$Hz")
    _ax.legend(fontsize=9)
    _ax.grid(True, which="both", alpha=0.2)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Comparison with the exponential trace

    The other estimator, `rate_mode = "exp"`, is an EWMA over past
    spikes with kernel $k(s) = e^{-s/\tau_r}/\tau_r$ for $s \ge 0$.

    At stationary Poisson($r$), both estimators are unbiased.
    Their variances are

    $$
    \mathrm{Var}[\hat r_\mathrm{win}] = \frac{r}{W},
    \qquad
    \mathrm{Var}[\hat r_\mathrm{exp}] = \frac{r}{2\tau_r}.
    $$

    (The exp variance is $\int_0^\infty k(s)^2 r\, ds$.) So
    $W = 2\tau_r$ matches the steady-state noise of the two filters.

    Their step responses differ in shape. If the rate jumps from
    $r_0$ to $r_1$ at time 0, the mean estimates are

    $$
    E[\hat r_\mathrm{win}(t;W)] - r_1 =
    \begin{cases}
      (r_0-r_1)(W-t)/W, & t < W \\
      0, & t \ge W
    \end{cases}
    $$

    $$
    E[\hat r_\mathrm{exp}(t;\tau_r)] - r_1 =
    (r_0 - r_1)\, e^{-t/\tau_r}.
    $$

    The window has a linear ramp that reaches zero at $t = W$. The
    exp trace decays geometrically and never reaches zero. The exp
    variance at time $t$ after the step is

    $$
    \mathrm{Var}[\hat r_\mathrm{exp}(t)] = \frac{r_1}{2\tau_r}(1 - e^{-2t/\tau_r}) + \frac{r_0}{2\tau_r} e^{-2t/\tau_r},
    $$

    which interpolates between the two steady-state values.
    """)
    return


@app.cell
def _(np, plt):
    _R0, _R1 = 5.0, 25.0
    _W = 1.0
    _TAU = _W / 2.0

    _t = np.linspace(-0.3, 3.0, 400)
    _step_win = np.where(
        _t < 0,
        _R0,
        np.where(
            _t < _W,
            _R0 + (_R1 - _R0) * _t / _W,
            _R1,
        ),
    )
    _tp = np.maximum(_t, 0.0)
    _step_exp = np.where(
        _t < 0,
        _R0,
        _R1 + (_R0 - _R1) * np.exp(-_tp / _TAU),
    )

    _t_stars = [0.3, 1.0, 3.0]
    _Weff = np.geomspace(0.05, 10.0, 400)

    _fig, (_ax_s, _ax_m) = plt.subplots(1, 2, figsize=(13, 4.2))

    _ax_s.axhline(_R0, ls=":", color="gray", alpha=0.4)
    _ax_s.axhline(_R1, ls=":", color="gray", alpha=0.4)
    _ax_s.axvline(0, ls="--", color="black", alpha=0.3)
    _ax_s.plot(_t, _step_win, lw=1.8, color="tab:blue",
               label=rf"Window, $W={_W}$ s")
    _ax_s.plot(_t, _step_exp, lw=1.8, color="tab:orange",
               label=rf"Exp trace, $\tau_r={_TAU}$ s")
    _ax_s.set_xlabel("t  (s after rate change)")
    _ax_s.set_ylabel(r"$E[\hat r]$  (Hz)")
    _ax_s.set_title("Mean step response (matched steady-state noise)")
    _ax_s.legend(fontsize=9)
    _ax_s.grid(True, alpha=0.2)

    for _ts in _t_stars:
        _mean_count_w = np.where(
            _Weff <= _ts,
            _R1 * _Weff,
            _R0 * (_Weff - _ts) + _R1 * _ts,
        )
        _bias_w = _mean_count_w / _Weff - _R1
        _var_w = _mean_count_w / _Weff ** 2
        _mse_w = _bias_w ** 2 + _var_w

        _tau_r = _Weff / 2.0
        _bias_e = (_R0 - _R1) * np.exp(-_ts / _tau_r)
        _var_e = ((_R1 / (2 * _tau_r)) * (1 - np.exp(-2 * _ts / _tau_r))
                  + (_R0 / (2 * _tau_r)) * np.exp(-2 * _ts / _tau_r))
        _mse_e = _bias_e ** 2 + _var_e

        _line, = _ax_m.loglog(_Weff, _mse_w, lw=1.3,
                              label=f"Window,  t*={_ts} s")
        _ax_m.loglog(_Weff, _mse_e, lw=1.3, ls="--", color=_line.get_color(),
                     label=f"Exp trace, t*={_ts} s")
    _ax_m.set_xlabel(r"$W_\mathrm{eff}$  (s;  $W$ for window, $2\tau_r$ for exp)")
    _ax_m.set_ylabel(r"MSE at $t^\star$  (Hz$^2$)")
    _ax_m.set_title("MSE vs filter width (matched steady-state noise)")
    _ax_m.legend(fontsize=7, ncol=2)
    _ax_m.grid(True, which="both", alpha=0.2)

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    **Constant-ISI (Part 1):**

    1. Lemma (single regime, rate $r$): $|\hat r(t;\,W) - r|
       \le 1/W$, from the count being within 1 of $rW$.
    2. Theorem (rate change $r_0 \to r_1$ at time 0): the
       Lemma applied to the new regime handles Case $t \ge W$
       directly, yielding $|\hat r(t;\,W) - r_1| \le 1/W$. For
       $W > t$ the window covers both regimes, producing a
       systematic error $(r_0 - r_1)(W - t)/W$ that the
       $t \ge W$ clause rules out.

    **Poisson (Part 2):**

    1. Under Poisson firing, $\hat r$ is a random variable with
       $E[\hat r] = r$ and $\mathrm{Var}[\hat r] = r/W$ (Theorems
       1–2). Convergence is probabilistic (Theorem 3, Chebyshev)
       and almost-sure (Theorem 4, SLLN). The constant-ISI
       statement gains a second tolerance $\delta$ on the
       probability of failure.
    2. Rate change: $E[\hat r]$ matches the constant-ISI piecewise
       formula exactly (no $\pm 2/W$ floor/ceil term). Combined
       with Poisson variance $r_1/W$, Chebyshev gives the
       probabilistic $\epsilon, \delta$ form for the rate-change
       case.
    3. Window-size tradeoff: the MSE-optimal $W^\star$ trades
       Poisson variance $r_1/W$ against step-change bias. A larger
       target tracking time $t^\star$ allows a larger $W^\star$.
    4. Exponential trace at matched steady-state variance
       ($W = 2\tau_r$): the window has a linear bias ramp that
       reaches zero at $t = W$; the exp trace decays geometrically
       and never reaches zero. For $W < t^\star$, the window has
       zero bias and lower MSE.
    """)
    return


if __name__ == "__main__":
    app.run()
