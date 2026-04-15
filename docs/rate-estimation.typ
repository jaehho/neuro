#set document(title: "Rate Estimation Analysis", date: auto)
#set text(font: "New Computer Modern", size: 11pt)
#set page(margin: 1in)
#set par(leading: 0.7em, first-line-indent: 1em, justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

#show heading.where(level: 1): it => {
  v(1em)
  text(size: 14pt, weight: "bold", it)
  v(0.5em)
}
#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 12pt, weight: "bold", it)
  v(0.3em)
}

#align(center)[
  #text(size: 18pt, weight: "bold")[Rate Estimation Analysis]
  #v(0.3em)
  #text(size: 14pt)[Sliding Window vs Exponential Trace]
  #v(1em)
]

Two methods for estimating a neuron's firing rate from its spike train are considered: a sliding window and an exponential trace.  We develop intuition for each on deterministic spike trains, then extend to Poisson firing.


// =====================================================================
= Sliding Window <sliding-window>

Count the number of spikes in a recent time window and divide by the window duration.  Let $W$ denote the window duration in seconds, and $N_"spikes"[t - W, t)$ the number of spikes in the half-open interval $[t - W, t)$.  The estimated rate $r(t)$ at time $t$ is:

$ r(t) = N_"spikes"[t - W, t) / W $ <eq:window>

// Signal-processing framing, commented out for now:
// The impulse response is a rectangular kernel: $h(t) = 1 slash W$ for $t in [0, W)$, zero otherwise.
The output is directly in Hz.

== Behaviour on a regular spike train

Consider a neuron firing at exactly $lambda = 20$ Hz, one spike every $"ISI" = 50$ ms.  After the window fills (the first $W$ seconds), the estimate settles.  Two quantities describe the long-run behaviour:

- *Steady-state value:* for a constant-rate neuron, $r(t) arrow lambda$.

- *Ripple:* each spike entering or leaving the window changes the count by $plus.minus 1$, causing a jump of $plus.minus 1 slash W$ Hz.  Whether this produces visible ripple depends on _alignment_: if $W dot lambda$ is an integer, a spike enters and another leaves the window at the same instant, and the count stays constant.  If $W dot lambda$ is not an integer, the count alternates between $floor(W lambda)$ and $floor(W lambda) + 1$, producing periodic jumps.  Alignment requires $W dot lambda in ZZ$, which holds only for specific combinations of $W$ and $lambda$.

#figure(
  image("../output/rate_est_ripple.png", width: 100%),
  caption: [Sliding window ripple on a regular 20 Hz spike train (ISI = 50 ms), zoomed to steady state.  Thin vertical lines mark spike times.  _Top:_ $W = 0.07$ s ($W dot lambda = 1.4$, not integer); the spike count alternates between $floor(W lambda) = 1$ and 2, producing periodic jumps.  _Bottom:_ $W = 0.25$ s ($W dot lambda = 5$, integer); a spike enters and leaves the window at the same instant, keeping the count constant.  This zero-ripple case requires $W dot lambda in ZZ$.],
) <fig:ripple>


// =====================================================================
= Exponential Trace <exp-trace>

An alternative weights recent spikes more heavily: each spike increments a trace variable that decays between spikes.  Let $tau_r$ be the decay time constant in seconds and $r$ the trace variable.  The trace evolves as:

$ tau_r (d r) / (d t) = -r, quad r arrow.l r + 1 "at each spike" $ <eq:exp-trace>

Each spike increments $r$ by 1; between spikes, $r$ decays exponentially with time constant $tau_r$.
// Signal-processing framing: the impulse response is $h(t) = e^(-t slash tau_r)$.

== Units

The raw trace is not in Hz.  Each spike adds $+1$ and the trace decays with time constant $tau_r$, producing a value in _trace units_.  Dividing by $tau_r$ converts to Hz: $r_"Hz" = r slash tau_r$.

== Behaviour on a regular spike train

For the same $lambda = 20$ Hz neuron:

- *Steady-state value:* the trace settles to $lambda tau_r$ (in trace units), or $lambda$ Hz after dividing by $tau_r$.  

- *Ripple:* unlike the window, the exponential trace _always_ has ripple.  Each spike adds 1 to the trace, which then decays until the next spike, producing a sawtooth pattern (@fig:exp-ripple).  The peak-to-trough amplitude is exactly 1 in trace units (each spike adds exactly 1).  What varies with rate is the _relative_ ripple: amplitude divided by mean $= 1 slash (lambda tau_r) = "ISI" slash tau_r$.  At low rates the sawtooth is a large fraction of the mean; at high rates it is negligible.

#figure(
  image("../output/rate_est_exp_ripple.png", width: 100%),
  caption: [Exponential trace on a regular 20 Hz spike train, zoomed to steady state ($tau_r = 0.5$ s, output in Hz).  Each spike increments the trace, which then decays until the next spike, producing a sawtooth pattern.  Unlike the sliding window, this ripple is always present regardless of $lambda$ or $tau_r$.],
) <fig:exp-ripple>


// == Response to rate changes <lag-bias>
//
// Both estimators lag behind changes in the true firing rate: the window by $W slash 2$ on average, the exponential trace by $tau_r$ (see @key-differences).
//
// The difference lies in what causes the lag.  The sliding window stores spike times from the last $W$ seconds.  When the firing rate changes (for example, because a synaptic weight was updated), spikes from before the change remain in the window until they age out.  For up to $W$ seconds, the estimate reflects a mixture of old and new rates.
//
// The exponential trace stores no spike times.  Its current value is a sufficient statistic for the future trajectory: when the rate changes, the decay trajectory changes immediately.  The lag comes from the exponential decay shape, not from stale spike history.
//
// This distinction is consequential in closed-loop systems where the rate estimate feeds back into the dynamics that generate spikes.  The window's stale history can cause the feedback loop to overshoot: the system corrects based on an outdated estimate, and by the time old spikes age out, the correction has gone too far.  This produces oscillations with period $approx 2W$.  The exponential trace, carrying no stale history, responds more smoothly to feedback-driven rate changes.
//
//
// // =====================================================================
// = Key Differences <key-differences>
//
// #table(
//   columns: 3,
//   align: (left, center, center),
//   table.header([Property], [Sliding Window], [Exponential Trace]),
//   [Output],                  [Hz (directly)],              [trace units ($div tau_r$ for Hz)],
//   [Per-spike contribution],  [$1 slash W$ for $t in [0, W)$],  [$e^(-t slash tau_r)$],
//   [Steady-state],            [$lambda$],                        [$lambda tau_r$ (raw) or $lambda$ (Hz)],
//   [Ripple (deterministic)],  [periodic; zero if aligned],      [sawtooth; always present],
//   [Step response],           [linear ramp, complete at $W$],    [exponential, 63% at $tau_r$],
//   [Mean lag],                [$W slash 2$],                      [$tau_r$],
//   [Spike exit],              [discrete jump at $t - W$],         [smooth decay],
//   [Memory],                  [exact times in last $W$ s],        [current value only],
// )
//
// At $W = tau_r$, the window has half the mean lag ($W slash 2$ vs $tau_r$) but carries stale spike history (see @lag-bias).
//
//
// // =====================================================================
= Comparison on Deterministic Neurons <comparison-deterministic>

== Constant Rate <det-constant>

#figure(
  image("../output/rate_est_constant.png", width: 100%),
  caption: [Rate estimation on a regular 20 Hz spike train.  _Second panel:_ head-to-head at matched time constant ($W = tau_r = 0.5$ s).  _Third panel:_ sliding window at varying $W$ (colorbar).  _Bottom:_ exponential trace at varying $tau_r$ (colorbar, output in Hz).  Smaller parameters produce larger ripple in both cases.],
) <fig:det-constant>

== Changing Rate <det-ramp>

#figure(
  image("../output/rate_est_ramp.png", width: 100%),
  caption: [Tracking a linearly changing firing rate (deterministic spikes).  _Left:_ 5 $arrow$ 40 Hz.  _Right:_ 40 $arrow$ 5 Hz.  _Middle row:_ sliding window at varying $W$.  _Bottom:_ exponential trace at varying $tau_r$ (converted to Hz).  Smaller parameters track faster but with more ripple; larger parameters are smoother but lag further behind the true rate (grey dashed).],
) <fig:det-ramp>


// =====================================================================
= Stochastic Fluctuations <stochastic>

// When spike timing follows a Poisson process, the deterministic ripple patterns become stochastic fluctuations.
//
// *Sliding window.*  The spike count in $[t - W, t)$ is Poisson-distributed with parameter $lambda W$.  For a Poisson random variable, mean equals variance: $chevron.l N chevron.r = "Var"(N) = lambda W$.  Dividing by $W$:
//
// $ chevron.l r chevron.r = lambda, quad "Var"(r) = lambda / W $
//
// *Exponential trace.*  The trace at time $t$ is a sum of decaying contributions from all past spikes: $r(t) = sum_i e^(-(t - t_i) slash tau_r)$.  In steady state, balancing average decay against average input gives $chevron.l r chevron.r slash tau_r = lambda$, so $chevron.l r chevron.r = lambda tau_r$.  Since Poisson arrivals are independent, the variance of the sum is the arrival rate times the integral of each contribution squared:
//
// $ "Var"(r) = lambda integral_0^infinity e^(-2 s slash tau_r) d s = (lambda tau_r) / 2 $
//
// In Hz ($r_"Hz" = r slash tau_r$): $chevron.l r_"Hz" chevron.r = lambda$, $"Var"(r_"Hz") = lambda slash (2 tau_r)$.
//
// // Signal-processing framing (Campbell's theorem), commented out for now:
// // $ chevron.l r chevron.r = lambda integral_0^infinity h(t) d t, quad "Var"(r) = lambda integral_0^infinity h(t)^2 d t $
//
// #table(
//   columns: 3,
//   align: (left, center, center),
//   table.header([Quantity], [Sliding Window], [Exponential Trace]),
//   [Steady-state value $chevron.l r chevron.r$],  [$lambda$],  [$lambda tau_r$ (raw); $lambda$ (Hz)],
//   [Fluctuation $"Var"(r)$],  [$lambda slash W$],  [$(lambda tau_r) slash 2$ (raw); $lambda slash (2 tau_r)$ (Hz)],
// )
//
// At matched time constant ($W = tau_r$), comparing in Hz: $"Var"_"window" = lambda slash tau_r$ vs $"Var"_"exp" = lambda slash (2 tau_r)$.  The window has twice the variance.  Combined with its lower lag, this is a noise-lag tradeoff: the window tracks faster but fluctuates more.

#figure(
  image("../output/rate_est_poisson.png", width: 100%),
  caption: [Rate estimation on a 20 Hz Poisson spike train.  _Second panel:_ head-to-head at $W = tau_r = 0.5$ s; both track the same mean, but the window shows larger fluctuations.  _Third panel:_ sliding window at varying $W$.  _Bottom:_ exponential trace at varying $tau_r$.  The noise-smoothing tradeoff parallels the deterministic ripple-lag tradeoff.],
) <fig:poisson>


// =====================================================================
// Rate Window Parameter Sweep — commented out for now.
// Lag-bias mechanism has been moved to §2.3 (@lag-bias).
/*
= Rate Window Parameter Sweep <rate-window-sweep>

A systematic sweep over $W in [0.05, 5.0]$ s (29 values, log-spaced with denser sampling in $[0.05, 0.2]$ s) using full neuromodulated STDP simulations with `rate_mode = "window"`.

== Summary Metrics <sweep-summary>

#figure(
  image("../output/sweep_rate_window.png", width: 100%),
  caption: [Four-panel summary.  _Top left:_ final weight.  _Top right:_ weight instability.  _Bottom left:_ achieved post-synaptic firing rate.  _Bottom right:_ relative rate error (log scale).  Grey dashed line: default $W = 0.5$ s.],
) <fig:sweep-summary>

Three regimes emerge:

+ *Silent* ($W < 80$ ms): The rate estimate fluctuates too wildly ($"Var" = lambda slash W$ is large) for the covariance neuromodulator to extract a coherent reward-prediction error.  $M$ fluctuates incoherently, driving $w arrow 0$.

+ *Convergent* ($W approx 80$--$200$ ms): Noise is tolerable.  The system converges to the target rate with $< 5%$ error.

+ *Overshoot* ($W > 200$ ms): Lag bias (@lag-bias) causes systematic overshoot to $tilde 12$ Hz; weight instability grows with $W$.

== Weight Trajectories <sweep-w-traces>

#figure(
  image("../output/sweep_rate_window_traces.png", width: 100%),
  caption: [Weight trajectories $w(t)$ coloured by $W$ (viridis, log scale).  Small $W$: collapse to zero.  Moderate $W$: smooth convergence.  Large $W$: overshoot and slow oscillation.],
) <fig:sweep-w-traces>

== Rate Traces Across Windows <sweep-rate-traces>

#figure(
  image("../output/rate_traces_by_window.png", width: 100%),
  caption: [Full-time $r_"post"(t)$ for nine window sizes plus exponential-trace reference (top, converted to Hz).  Smaller windows produce higher-amplitude oscillations; larger windows are smoother but settle higher due to lag bias.],
) <fig:rate-traces-full>

#figure(
  image("../output/rate_traces_by_window_zoomed.png", width: 100%),
  caption: [Zoomed to $t = 25$--$35$ s (steady state).  $W = 0.05$ s: near-binary.  $W = 0.08$--$0.1$ s: quantisation levels visible.  $W = 0.2$ s: periodic oscillations.  $W = 0.5$ s: smooth.  $W = 5.0$ s: slow sinusoidal modulation above the 10 Hz target.],
) <fig:rate-traces-zoomed>

== Lag-Bias Mechanism <lag-bias-detail>

The overshoot in the large-$W$ regime has a clear causal chain:

+ The window reports $r_"post" = lambda$ from spikes fired over the last $W$ seconds.
+ The learning rule adjusts $w$ to bring $r_"post"$ toward $r_"target"$.
+ But the window still contains old spikes fired when $w$ was different.  The estimate reflects the _past_ firing rate, not the _current_ one.
+ For up to $W$ seconds, the reward signal $R$ acts on stale information, overshooting the correction.
+ This creates a delayed negative-feedback oscillation with period $tilde 2 W$.

The exponential trace avoids this: its current value is a sufficient statistic, so a weight change immediately alters the trace's decay trajectory without waiting for old spikes to age out.  This is the structural advantage noted in @exp-trace, and the reason the default `rate_mode` is `"exp"`.

== Representative Traces <sweep-traces>

Six 8-panel traces spanning the three regimes.

#figure(
  image("../output/trace_00_rw0.0500.png", width: 100%),
  caption: [$W = 0.050$ s --- silent regime.  The estimate is near-binary ($0$ or $1 slash W = 20$ Hz per spike).  Reward $R$ is dominated by noise; $M$ is incoherent.  Weight collapses; the neuron goes silent.],
) <fig:trace-0050>

#figure(
  image("../output/trace_06_rw0.0816.png", width: 100%),
  caption: [$W = 0.082$ s --- onset of convergence.  Noisy rate estimate, but the covariance neuromodulator begins to extract a coherent signal.  Weight stabilises around $w approx 1.85$.],
) <fig:trace-0082>

#figure(
  image("../output/trace_09_rw0.1042.png", width: 100%),
  caption: [$W = 0.104$ s --- optimal zone.  Clean learning dynamics: $r_"post"$ settles near target, $M$ shows well-defined reward-prediction errors.],
) <fig:trace-0104>

#figure(
  image("../output/trace_17_rw0.2000.png", width: 100%),
  caption: [$W = 0.200$ s --- boundary of best convergence.  Smoother, slower adaptation.  $overline(R)$ closely tracks $R$.],
) <fig:trace-0200>

#figure(
  image("../output/trace_20_rw0.4812.png", width: 100%),
  caption: [$W = 0.481$ s --- near default.  Smooth estimate, but lag bias emerging: $r_"post"$ settles above target.],
) <fig:trace-0481>

#figure(
  image("../output/trace_28_rw5.0000.png", width: 100%),
  caption: [$W = 5.000$ s --- strongest lag bias.  Weight oscillates with period $tilde 10$ s ($approx 2 W$).  $r_"post"$ systematically overshoots.],
) <fig:trace-5000>
*/
