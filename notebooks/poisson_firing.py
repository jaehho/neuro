"""Sliding-window rate estimator under Poisson firing.

Extends `rate_estimator.py`'s constant-ISI analysis. No simulation
required; synthesizes Poisson spike trains directly.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Rate estimator under Poisson firing

    This notebook extends the constant-ISI analysis in
    `notebooks/rate_estimator.py`. That notebook showed: for a rate
    change $r_0 \to r_1$ at time 0 with perfectly periodic firing,
    the sliding-window estimator
    $\hat r(t;\,W) = N([t - W,\,t])/W$ satisfies
    $|\hat r - r_1| \le 1/W$ once $t \ge W$, deterministically.

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
    R0_M, R1_M = 5.0, 25.0
    T_STARS = [0.1, 0.3, 1.0, 3.0]
    Ws_m = np.geomspace(0.01, 10.0, 500)

    fig_m, ax_m = plt.subplots(figsize=(7.5, 4.2))
    for t_star in T_STARS:
        mean_count = np.where(
            Ws_m <= t_star,
            R1_M * Ws_m,
            R0_M * (Ws_m - t_star) + R1_M * t_star,
        )
        exp_hat = mean_count / Ws_m
        bias_m = exp_hat - R1_M
        var_m = mean_count / Ws_m ** 2
        mse = bias_m ** 2 + var_m
        line, = ax_m.loglog(Ws_m, mse, lw=1.2, label=f"t* = {t_star} s")
        ax_m.plot(Ws_m[np.argmin(mse)], mse.min(), "o", ms=6,
                  color=line.get_color())
    ax_m.set_xlabel("W  (s)")
    ax_m.set_ylabel(r"MSE at $t^\star$  (Hz$^2$)")
    ax_m.set_title(rf"Optimal window grows with $t^\star$ — $r_0={R0_M}\,$Hz → $r_1={R1_M}\,$Hz")
    ax_m.legend(fontsize=9)
    ax_m.grid(True, which="both", alpha=0.2)
    fig_m.tight_layout()
    fig_m
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
    R0_CMP, R1_CMP = 5.0, 25.0
    W_CMP = 1.0
    TAU_CMP = W_CMP / 2.0

    t_cmp = np.linspace(-0.3, 3.0, 400)
    step_win = np.where(
        t_cmp < 0,
        R0_CMP,
        np.where(
            t_cmp < W_CMP,
            R0_CMP + (R1_CMP - R0_CMP) * t_cmp / W_CMP,
            R1_CMP,
        ),
    )
    tp = np.maximum(t_cmp, 0.0)
    step_exp = np.where(
        t_cmp < 0,
        R0_CMP,
        R1_CMP + (R0_CMP - R1_CMP) * np.exp(-tp / TAU_CMP),
    )

    T_STARS_CMP = [0.3, 1.0, 3.0]
    Weff = np.geomspace(0.05, 10.0, 400)

    fig_cmp, (ax_s, ax_m) = plt.subplots(1, 2, figsize=(13, 4.2))

    ax_s.axhline(R0_CMP, ls=":", color="gray", alpha=0.4)
    ax_s.axhline(R1_CMP, ls=":", color="gray", alpha=0.4)
    ax_s.axvline(0, ls="--", color="black", alpha=0.3)
    ax_s.plot(t_cmp, step_win, lw=1.8, color="tab:blue",
              label=rf"Window, $W={W_CMP}$ s")
    ax_s.plot(t_cmp, step_exp, lw=1.8, color="tab:orange",
              label=rf"Exp trace, $\tau_r={TAU_CMP}$ s")
    ax_s.set_xlabel("t  (s after rate change)")
    ax_s.set_ylabel(r"$E[\hat r]$  (Hz)")
    ax_s.set_title("Mean step response (matched steady-state noise)")
    ax_s.legend(fontsize=9)
    ax_s.grid(True, alpha=0.2)

    for t_star in T_STARS_CMP:
        mean_count_w = np.where(
            Weff <= t_star,
            R1_CMP * Weff,
            R0_CMP * (Weff - t_star) + R1_CMP * t_star,
        )
        bias_w = mean_count_w / Weff - R1_CMP
        var_w = mean_count_w / Weff ** 2
        mse_w = bias_w ** 2 + var_w

        tau = Weff / 2.0
        bias_e = (R0_CMP - R1_CMP) * np.exp(-t_star / tau)
        var_e = ((R1_CMP / (2 * tau)) * (1 - np.exp(-2 * t_star / tau))
                 + (R0_CMP / (2 * tau)) * np.exp(-2 * t_star / tau))
        mse_e = bias_e ** 2 + var_e

        line, = ax_m.loglog(Weff, mse_w, lw=1.3,
                            label=f"Window,  t*={t_star} s")
        ax_m.loglog(Weff, mse_e, lw=1.3, ls="--", color=line.get_color(),
                    label=f"Exp trace, t*={t_star} s")
    ax_m.set_xlabel(r"$W_\mathrm{eff}$  (s;  $W$ for window, $2\tau_r$ for exp)")
    ax_m.set_ylabel(r"MSE at $t^\star$  (Hz$^2$)")
    ax_m.set_title("MSE vs filter width (matched steady-state noise)")
    ax_m.legend(fontsize=7, ncol=2)
    ax_m.grid(True, which="both", alpha=0.2)

    fig_cmp.tight_layout()
    fig_cmp
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

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
