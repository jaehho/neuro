#set document(title: "Neuromodulated STDP: Simulation Results", date: auto)
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
  #text(size: 18pt, weight: "bold")[Neuromodulated STDP: Simulation Results]
  #v(0.3em)
  #text(size: 14pt)[Three-Factor Learning Rule Analysis and Parameter Sweeps]
  #v(1em)
]

Unless noted otherwise, all simulations use $n_"pre" = 1$, RK4 integration ($Delta t = 0.1$ ms), Poisson pre-synaptic input at 20 Hz, covariance neuromodulator, target-rate reward signal ($r_"target" = 10$), and seed 42.


// =====================================================================
= Numerical Validation <numerical-validation>

== RK4 Convergence <rk4-convergence>

The hybrid integration scheme uses classical RK4 for smooth inter-spike dynamics with linear interpolation for threshold-crossing detection.  Spike discontinuities degrade the effective convergence order below the theoretical $cal(O)(Delta t^4)$.

#figure(
  image("../output/rk4_convergence.png", width: 95%),
  caption: [RK4 convergence analysis.  The membrane voltage $V$ converges at order $tilde 1.6$ (degraded from 4 by spike discontinuities); the slow weight variable $w$ converges at order $tilde 1.3$.  Reference solution computed at $Delta t = 10^(-5)$ ms.],
) <fig:rk4-convergence>

== RK4 Validation <rk4-validation>

#figure(
  image("../output/rk4_validation.png", width: 95%),
  caption: [Validation of the RK4 integrator against a high-resolution Euler reference.  All state variables ($V$, $w$, $E$, $r_"post"$, $R$, $M$) agree to within expected truncation error.],
) <fig:rk4-validation>

== State Derivatives <state-derivatives>

#figure(
  image("../output/derivatives.png", width: 95%),
  caption: [Time derivatives of each state variable, verifying the ODE right-hand side implementation.  Discontinuous jumps correspond to spike events (pre- and post-synaptic).],
) <fig:derivatives>


// =====================================================================
= Convergence Analysis <convergence-analysis>

These figures sweep over initial weights and learning rates to characterise the basin of convergence for the target-rate self-supervisory paradigm.

== Weight Convergence <convergence-weight>

#figure(
  image("../output/convergence_w_final.png", width: 95%),
  caption: [Final weight $w_"final"$ as a function of initial weight $w_0$ and learning rate $eta$.  The target-rate reward signal drives weights toward a fixed point that achieves $r_"post" approx r_"target"$.],
) <fig:convergence-w-final>

== Weight Stability <convergence-stability>

#figure(
  image("../output/convergence_w_stability.png", width: 95%),
  caption: [Weight stability in the second half of the simulation ($t > T slash 2$), measured as standard deviation of $w(t)$.  Low $w_"std"$ indicates convergence to a stable equilibrium; high values indicate oscillation or drift.],
) <fig:convergence-w-stability>

== Rate Error <convergence-rate-error>

#figure(
  image("../output/convergence_rate_error.png", width: 95%),
  caption: [Relative rate error $|r_"post" - r_"target"| slash r_"target"$ in the second half of the simulation.  The learning rule successfully drives post-synaptic firing rate toward the target across a wide range of initial conditions.],
) <fig:convergence-rate-error>

== Convergence Map <convergence-map>

#figure(
  image("../output/convergence_map.png", width: 95%),
  caption: [Convergence map over the two-dimensional parameter space.  Green indicates successful convergence ($< 10%$ relative rate error); red indicates failure.  The basin of convergence is broad but bounded.],
) <fig:convergence-map>


// =====================================================================
= Neuromodulator Comparison <neuromod-comparison>

Four neuromodulator types are compared under identical conditions ($n_"pre" = 2$, contingent reward, $T = 100$ s).  See the methods document for the definitions of each type.

#figure(
  image("../output/neuromod_comparison.png", width: 100%),
  caption: [Side-by-side comparison of weight trajectories $w_1(t)$ (target-paired) and $w_2(t)$ (distractor) under the four neuromodulator types: covariance (RPE), gated Hebbian, surprise/novelty, and constant (two-factor STDP baseline).],
) <fig:neuromod-comparison>

#figure(
  image("../output/neuromod_details.png", width: 100%),
  caption: [Detailed state-variable traces for each neuromodulator type.  Panels show membrane voltage $V$, eligibility traces $E_i$, firing rates $r$, reward signal $R$, reward baseline $overline(R)$, and modulator $M$.],
) <fig:neuromod-details>

#figure(
  image("../output/neuromod_summary.png", width: 95%),
  caption: [Summary metrics across neuromodulator types: final weights, weight separation ($w_1 - w_2$), post-synaptic firing rate, and weight stability.  The covariance (RPE) type achieves the best spatial credit assignment (largest $w_1 - w_2$ separation).],
) <fig:neuromod-summary>


// =====================================================================
= Regime Analysis <regime-analysis>

Systematic exploration of reward signal $times$ neuromodulator type combinations.

#figure(
  image("../output/regime_comparison.png", width: 100%),
  caption: [Weight trajectories across all reward signal $times$ neuromodulator type combinations.  Rows: reward signals (target-rate, biofeedback, contingent).  Columns: neuromodulator types (covariance, gated, surprise, constant).],
) <fig:regime-comparison>

#figure(
  image("../output/regime_details.png", width: 100%),
  caption: [Detailed state-variable traces for selected regime combinations, showing the internal dynamics ($E$, $R$, $M$) that drive divergent learning outcomes.],
) <fig:regime-details>

#figure(
  image("../output/regime_summary.png", width: 95%),
  caption: [Summary heatmaps across the reward signal $times$ neuromodulator grid.  Metrics: final weight, weight separation, post-rate achieved, stability.  The contingent $times$ covariance combination is the most effective for spatial credit assignment.],
) <fig:regime-summary>

== Initial Condition Sensitivity <regime-ic-sensitivity>

#figure(
  image("../output/regime_ic_sensitivity.png", width: 95%),
  caption: [Sensitivity to initial conditions across regimes.  Each regime is simulated from multiple random seeds and initial weights.  Robust regimes (covariance, contingent) show tight convergence; fragile regimes (constant, surprise) show high variance.],
) <fig:regime-ic-sensitivity>


// =====================================================================
= Spectral Analysis <spectral-analysis>

#figure(
  image("../output/spectrograms.png", width: 100%),
  caption: [Spectrograms of key state variables ($V$, $w$, $r_"post"$) computed via short-time Fourier transform.  The weight $w$ and firing rate $r_"post"$ show low-frequency oscillations ($< 5$ Hz) driven by the reward-modulation feedback loop.  Membrane voltage $V$ exhibits broadband structure dominated by spike timing.],
) <fig:spectrograms>


// =====================================================================
= Target Function Sweeps <target-functions>

The target-rate reward signal $R = -(r_"post" - f(r_"pre"))^2$ can be parameterised by an arbitrary target function $f$.  These sweeps test the learning rule's ability to track different functional relationships.

#figure(
  image("../output/general_target_curves.png", width: 90%),
  caption: [Target functions tested: fixed, linear, affine, quadratic, square root, logarithmic, sinusoidal, and power.  Each defines a different desired mapping $r_"post" = f(r_"pre")$.],
) <fig:target-curves>

#figure(
  image("../output/general_target_timeseries.png", width: 100%),
  caption: [Weight and firing-rate time series under each target function.  The learning rule converges for all monotone targets; non-monotone functions (sinusoidal) produce oscillatory behaviour.],
) <fig:target-timeseries>

#figure(
  image("../output/general_target_scatter.png", width: 90%),
  caption: [Achieved $r_"post"$ vs.\ target $f(r_"pre")$ for each target function.  Points near the diagonal indicate successful tracking.  Scatter around the diagonal reflects stochastic variability from Poisson input.],
) <fig:target-scatter>


// =====================================================================
= Rate Estimation Methods <rate-estimation>

The simulation supports two firing-rate estimation modes:

- *Exponential trace* (`rate_mode = "exp"`): $tau_r dot.op d r slash d t = -r + rho(t)$, yielding a causal exponential filter with time constant $tau_r$.

- *Sliding window* (`rate_mode = "window"`): $r(t) = N_"spikes"[t - W, t) slash W$, counting spikes in the most recent $W$ seconds.

Both feed into the reward computation $R(t)$ and neuromodulator $M(t)$.

== Exponential Trace vs Sliding Window <rate-exp-vs-window>

To isolate the estimator difference from the learning dynamics, we generate a constant 20 Hz Poisson spike train and apply both filters with no simulation.

#figure(
  image("../output/compare_rate_estimators.png", width: 100%),
  caption: [Rate estimation on a constant 20 Hz Poisson neuron.  _Top:_ spike raster.  _Middle:_ direct comparison of exponential trace ($tau_r = 0.5$ s, divided by $tau_r$ to convert to Hz) and sliding window ($W = 0.5$ s) at matched time constant.  Both track the same mean but differ in impulse response shape --- the window trace shows staircase jumps as spikes enter and leave the window, while the exponential trace decays smoothly.  _Bottom:_ sliding window estimates across window sizes ($W = 0.05$--$2.0$ s), illustrating the noise--smoothing tradeoff.],
) <fig:rate-comparison>

== Window Size Tradeoff <rate-window-tradeoff>

#figure(
  image("../output/compare_rate_estimators_panels.png", width: 100%),
  caption: [Individual panels for each window size applied to the same constant 20 Hz spike train.  At $W = 0.05$ s, the estimate is near-binary (0 or $1 slash W = 20$ Hz per spike).  At $W = 2.0$ s, the estimate is smooth but responds slowly.  The transition from high-noise to smooth occurs around $W approx 0.2$--$0.5$ s.],
) <fig:rate-panels>


// =====================================================================
= Rate Window Parameter Sweep <rate-window-sweep>

A systematic sweep over $W in [0.05, 5.0]$ s (29 values, log-spaced, denser in $[0.05, 0.2]$ s) with `rate_mode = "window"`.  Fixed parameters: $n_"pre" = 1$, $r_"pre" = 20$ Hz, target-rate reward, covariance neuromodulator, $T = 60$ s, seed 42.

== Summary Metrics <sweep-summary>

#figure(
  image("../output/sweep_rate_window.png", width: 100%),
  caption: [Four-panel summary of the rate-window sweep.  _Top left:_ final weight $w_"final"$ vs.\ window size.  _Top right:_ weight instability (std of $w$ in second half).  _Bottom left:_ achieved post-synaptic firing rate --- small windows ($< 80$ ms) produce silence; moderate windows (80--200 ms) hit the 10 Hz target; large windows overshoot to $tilde 12$ Hz due to lag bias.  _Bottom right:_ relative rate error on log scale.  The grey dashed line marks the default $W = 0.5$ s.],
) <fig:sweep-summary>

== Weight Trajectories <sweep-w-traces>

#figure(
  image("../output/sweep_rate_window_traces.png", width: 100%),
  caption: [Weight trajectories $w(t)$ for sampled window sizes, coloured by $W$ (viridis colourbar, log scale).  Small windows ($W < 80$ ms): weights collapse to zero (neuron goes silent).  Optimal windows ($W approx 80$--$200$ ms): smooth convergence to equilibrium.  Large windows ($W > 0.5$ s): overshoot followed by slow oscillation, consistent with the lag-bias mechanism.],
) <fig:sweep-w-traces>

== Rate Traces Across Windows <sweep-rate-traces>

The key diagnostic: how does the rate estimate $r_"post"(t)$ itself look for different window sizes?

#figure(
  image("../output/rate_traces_by_window.png", width: 100%),
  caption: [Full-time $r_"post"$ traces ($0$--$60$ s) for nine window sizes plus the exponential-trace reference (top panel, converted to Hz by dividing by $tau_r$).  Smaller windows produce higher-amplitude oscillations; larger windows are smoother but settle to a higher steady-state rate due to lag bias.],
) <fig:rate-traces-full>

#figure(
  image("../output/rate_traces_by_window_zoomed.png", width: 100%),
  caption: [Zoomed view of $r_"post"$ during steady state ($t = 25$--$35$ s).  At $W = 0.05$ s, the estimate is near-binary (each spike adds $1 slash W = 20$ Hz).  At $W = 0.08$--$0.1$ s, discrete quantisation levels are visible.  By $W = 0.2$ s, regular oscillations with clear periodicity appear.  The $W = 0.5$ s trace is smooth enough for stable learning.  At $W = 5.0$ s, the estimate shows slow sinusoidal modulation with systematic overshoot above the 10 Hz target (dashed line).],
) <fig:rate-traces-zoomed>

== Representative Simulation Traces <sweep-traces>

Selected full 8-panel traces from the sweep, each showing $V$, $w$, $E$, $r_"post"$, $R$, $M$, $overline(R)$, and $r_"pre"$ for one window size.

#figure(
  image("../output/trace_00_rw0.0500.png", width: 100%),
  caption: [$W = 0.050$ s --- smallest window.  Rate estimate is too noisy; the reward signal $R$ fluctuates wildly, producing incoherent modulation.  Weight decays to zero; the neuron goes silent.],
) <fig:trace-0050>

#figure(
  image("../output/trace_06_rw0.0816.png", width: 100%),
  caption: [$W = 0.082$ s --- onset of convergence.  The rate estimate is still noisy but sufficiently stable for the covariance neuromodulator to extract a coherent reward-prediction error.  Weight stabilises and the neuron fires near the target rate.],
) <fig:trace-0082>

#figure(
  image("../output/trace_09_rw0.1042.png", width: 100%),
  caption: [$W = 0.104$ s --- within the optimal convergence zone.  Clear learning dynamics: weight rises from $w_0 = 2$ to equilibrium, $r_"post"$ settles near $r_"target"$, and the modulator $M$ shows clean reward-prediction errors.],
) <fig:trace-0104>

#figure(
  image("../output/trace_17_rw0.2000.png", width: 100%),
  caption: [$W = 0.200$ s --- boundary of the best-convergence region.  Still converges well but the rate estimate is smoother, leading to slightly slower adaptation.  The reward baseline $overline(R)$ tracks the reward signal closely.],
) <fig:trace-0200>

#figure(
  image("../output/trace_20_rw0.4812.png", width: 100%),
  caption: [$W = 0.481$ s --- near the default window.  Smooth rate estimate, but lag bias is becoming visible: the weight overshoots slightly and $r_"post"$ settles above $r_"target"$.],
) <fig:trace-0481>

#figure(
  image("../output/trace_28_rw5.0000.png", width: 100%),
  caption: [$W = 5.000$ s --- largest window.  The rate estimate is very smooth but integrates stale spike counts from before weight changes, causing systematic overshoot.  The weight shows slow oscillations driven by the delayed feedback between rate estimation and reward computation.],
) <fig:trace-5000>
