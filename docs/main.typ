#set document(title: "Neuromodulated STDP", date: auto)
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
  #text(size: 18pt, weight: "bold")[Neuromodulated Spike-Timing-Dependent Plasticity]
  #v(0.3em)
  #text(size: 14pt)[From Single-Neuron Operant Conditioning to Spatial Credit Assignment]
  #v(1em)
]

// ═══════════════════════════════════════════════════════════════════════
= Introduction <introduction>

The basic question this model addresses is the one Fetz posed in 1969 @fetz1969: can a single cortical neuron be operantly conditioned to fire more or less often when reward is contingent on its activity?  Fetz showed it could; the modern brain--machine-interface literature @moritz2011 turns this into engineering --- decode a neuron's spike rate into a reward, and the animal learns to drive it.  The natural mechanistic question is what plasticity rule the synapses onto that neuron must obey for this to work.  Three-factor learning rules @fremauxNeuromodulatedSpikeTimingDependentPlasticity2016 give an answer: a global reward-prediction-error signal $M(t)$ multiplies a per-synapse eligibility trace $E_i (t)$ that bridges the gap between a millisecond spike and a delayed reward.

Mathematically, this is REINFORCE @williams1992 with a value baseline --- the covariance neuromodulator $M = R - overline(R)$ is exactly the policy-gradient update where $overline(R)$ plays the role of the critic, as Frémaux & Gerstner derive in their §4.3.  The single-presynaptic version of the model ($n_"pre" = 1$) is a minimal in-silico Fetz experiment: one neuron, one reward contingency, learn to hit a target firing rate.  The two-presynaptic version ($n_"pre" = 2$) is the spatial-credit story this document develops in detail: when the same global reward signal reaches every synapse, only the synapse whose pre-spikes systematically precede post-spikes accumulates eligibility at reward time, so reward selectively reads out the causal input.

The theory below is written for $n_"pre" = 2$ throughout; the simulation results in §9 use $n_"pre" = 1$ unless otherwise stated.  Each axis of the parameter space ($n_"pre"$, neuromodulator type, reward signal, target shape, integrator) is studied separately in a notebook under `notebooks/`.


// ═══════════════════════════════════════════════════════════════════════
= Circuit and Neural Dynamics

Two presynaptic neurons (pre#sub[1], pre#sub[2]) project onto one postsynaptic LIF neuron through independent synapses of weight $w_1$ and $w_2$.  Pre#sub[1] is the _target_ input whose activity is paired with reward; pre#sub[2] is a _distractor_ that fires at the same rate but is not reward-paired.  Both fire independent Poisson spike trains.

== Membrane Potential (LIF)

The postsynaptic neuron is a leaky integrate-and-fire (LIF) unit @dayanTheoreticalNeuroscienceComputational2001 @gerstnerNeuronalDynamicsSingle2014.  Its membrane potential $V(t)$ obeys

$ tau_m (d V) / (d t) = -(V - E_L) + R_m (w_1 I_(s,1) + w_2 I_(s,2)) $ <eq:lif>

where $tau_m = R_m C_m$ is the membrane time constant, $E_L$ the resting potential, and $I_(s,i)$ the synaptic current from synapse $i$.  A spike fires when $V(t) = theta$ with $d V slash d t > 0$; the membrane resets to $V_"reset"$ and integration is suspended for $tau_"ref"$.

== Presynaptic Spike Trains

Each presynaptic neuron fires independent Poisson spikes:

$ rho_i (t) = sum_k delta(t - t_(i)^k), quad i in {1,2} $ <eq:spike-trains>

The postsynaptic spike train $rho_"post"(t)$ is determined by @eq:lif.  Published three-factor STDP simulations universally use Poisson (not deterministic) presynaptic input @izhikevich2007 @florian2007 @fremaux2010 @vasilaki2009.

== Synaptic Current

Each synapse has its own current, driven by its presynaptic spike train filtered through an exponential kernel:

$ (d I_(s,i)) / (d t) = -I_(s,i) / tau_s, quad I_(s,i) arrow.l I_(s,i) + 1 "at each pre"_i "spike" $ <eq:isyn>


// ═══════════════════════════════════════════════════════════════════════
= Three-Factor Plasticity

The plasticity rule belongs to the class of _three-factor learning rules_ @fremauxNeuromodulatedSpikeTimingDependentPlasticity2016 @izhikevich2007.  Standard STDP depends on pre--post spike correlations (two factors); three-factor rules gate this local signal with a global neuromodulatory factor $M$.

== STDP Traces

Each synapse has a presynaptic trace $x_i$; the postsynaptic trace $y_"post"$ is shared @biSynapticModificationsCultured1998:

$ tau_+ (d x_i) / (d t) = -x_i + rho_i (t), quad tau_- (d y_"post") / (d t) = -y_"post" + rho_"post"(t) $ <eq:traces>

== Eligibility Trace

Coincident pre- and post-spikes create a per-synapse _eligibility trace_ $E_i (t)$ @izhikevich2007 @gerstnerEligibilityTracesPlasticity2018 that bridges millisecond spike timing and delayed reward:

$ tau_e (d E_i) / (d t) = -E_i + underbrace(eta_+ (w_max - w_i) x_i rho_"post", "LTP") - underbrace(eta_- w_i y_"post" rho_i, "LTD") $ <eq:eligibility>

Soft-bound factors $(w_max - w_i)$ and $w_i$ implement _multiplicative STDP_ @Song2000, preventing unbounded growth.

== Firing-Rate Estimation

Smooth rate estimates are obtained by exponential filtering:

$ tau_r (d r_i) / (d t) = -r_i + rho_i (t), quad tau_r (d r_"post") / (d t) = -r_"post" + rho_"post"(t) $ <eq:rates>

== Weight Update

Each weight evolves under the same global modulation $M(t)$, gated by its own eligibility:

$ (d w_i) / (d t) = M(t) dot E_i (t) $ <eq:weight-update>

Because $M$ is global but $E_i$ is synapse-specific, only the synapse with high eligibility at the moment of reward undergoes a large weight change.  This is the mechanism of _spatial credit assignment_.


// ═══════════════════════════════════════════════════════════════════════
= Neuromodulator Roles <neuromod-roles>

The neuromodulator $M(t)$ determines _how_ the reward signal $R$ influences plasticity.  Frémaux & Gerstner @fremauxNeuromodulatedSpikeTimingDependentPlasticity2016 (Eq. 14) identify four roles:

== Covariance (RPE / Dopaminergic) <neuromod-covariance>

$ M = R - overline(R), quad tau_(overline(R)) (d overline(R)) / (d t) = -overline(R) + R $ <eq:covariance>

A _reward prediction error_: phasic dopamine encodes "better than expected" vs "worse than expected" @schultz1997.  The baseline $overline(R)$ adapts slowly ($tau_(overline(R)) = 5$ s), acting as a critic @Sutton1998.  This is the default in R-STDP @fremauxNeuromodulatedSpikeTimingDependentPlasticity2016 (Section 4.3, Eq. 7).

== Gated Hebbian <neuromod-gated>

$ M = R $ <eq:gated>

Reward directly gates plasticity with no baseline subtraction.

== Surprise / Novelty <neuromod-surprise>

$ M = (r_"post" - overline(R))^2, quad tau_(overline(R)) (d overline(R)) / (d t) = -overline(R) + r_"post" $ <eq:surprise>

Here $overline(R)$ tracks the _firing rate_, not reward.  The surprise signal is always non-negative; LTP vs LTD direction is determined by $E_i$ alone.  Models noradrenergic or cholinergic modulation @yu2005.

== Constant (Non-Modulated STDP) <neuromod-constant>

$ M = 1 $ <eq:constant>

Standard two-factor STDP.  Useful as a baseline for isolating the effect of neuromodulation.


// ═══════════════════════════════════════════════════════════════════════
= Reward Signals <reward-signals>

The reward signal $R(t)$ defines _what_ the system is rewarded for.  In the three-factor STDP literature, $R$ is always an external, task-based signal @fremauxNeuromodulatedSpikeTimingDependentPlasticity2016 @izhikevich2007 @florian2007 @legenstein2008.

== Target Rate (Self-Supervisory Demonstration) <reward-target-rate>

$ R = -(r_"post" - r_"target")^2 $ <eq:target-rate>

The postsynaptic rate is penalised for deviating from a target.  This is self-supervisory (not from the literature) but useful for verifying the ODE machinery and is the closest analog to the Fetz operant-conditioning experiment when $n_"pre" = 1$.

== Biofeedback <reward-biofeedback>

$ R(t) = d(t), quad d(t) = cases(R_"amount" dot e^(-(t - t_"delivery") slash tau_d) & "if reward delivered at" t_"delivery", 0 & "otherwise") $ <eq:biofeedback>

Every postsynaptic spike schedules a reward pulse at $t_"post"^k + Delta_"reward"$.  This is the Legenstein et al. @legenstein2008 biofeedback paradigm.  It demonstrates _temporal_ credit assignment but not _spatial_: with two synapses onto the same neuron, both inputs are directly upstream and both strengthen.

== Contingent Reward (Izhikevich Paradigm) <reward-contingent>

The same delayed pulse mechanism as biofeedback, but reward is scheduled *only when pre#sub[1] fires within a coincidence window $Delta_c$ before a post-spike*:

$ "schedule reward at" t + Delta_"reward" quad "iff" quad exists t_1^k in [t - Delta_c, t] $ <eq:contingent>

where $t$ is the post-spike time and $t_1^k$ is the most recent pre#sub[1] spike.  This is the "gated-Hebbian" paradigm described by Frémaux & Gerstner @fremauxNeuromodulatedSpikeTimingDependentPlasticity2016 (Eq. 10) and used by Izhikevich @izhikevich2007:

$ chevron.l dot(w)_i chevron.r = "Cov"(R, H_i) + chevron.l R chevron.r chevron.l H_i chevron.r $ <eq:izh-expected>

where $H_i = H("pre"_i, "post")$ is the Hebbian coincidence term for synapse $i$.  Because reward is correlated with pre#sub[1]$arrow$post coincidences:

- $"Cov"(R, H_1) > 0$: synapse 1 strengthens.
- $"Cov"(R, H_2) approx 0$: synapse 2 sees no systematic correlation.

The $chevron.l R chevron.r chevron.l H_i chevron.r$ bias drives both weights downward (since $chevron.l H_i chevron.r < 0$ under typical STDP windows), but the positive covariance for synapse 1 overcomes this drift.  Reward acts as a _binary gating signal_ that switches from general depression to selective potentiation of the target synapse.


// ═══════════════════════════════════════════════════════════════════════
= Summary <summary>

*State.*  Shared (4): $V$, $y_"post"$, $r_"post"$, $overline(R)$.  Per synapse $i$ (5): $I_(s,i)$, $x_i$, $E_i$, $r_i$, $w_i$.

*Initial conditions* ($t = 0$).  $V = E_L$ (refractory off);  $w_i = w_(i,0)$;  all other state variables zero.

=== Inter-spike dynamics

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: (x: 8pt, y: 6pt),
  align: (left + horizon, left + horizon),
  column-gutter: 1em,
  [Membrane],
  [$tau_m space (d V) / (d t) = -(V - E_L) + R_m sum_i w_i I_(s,i)$],

  [Synaptic current],
  [$(d I_(s,i)) / (d t) = -I_(s,i) slash tau_s$],

  [STDP traces],
  [$(d x_i) / (d t) = -x_i slash tau_+ quad quad (d y_"post") / (d t) = -y_"post" slash tau_-$],

  [Rate filters],
  [$(d r_i) / (d t) = -r_i slash tau_r quad quad (d r_"post") / (d t) = -r_"post" slash tau_r$],

  [Reward baseline],
  [$tau_(overline(R)) space (d overline(R)) / (d t) = -overline(R) + T$  ($T$ depends on modulator, see below)],

  [Eligibility],
  [$(d E_i) / (d t) = -E_i slash tau_e$],

  [Weights],
  [$(d w_i) / (d t) = M(t) dot E_i (t)$],
)

=== Spike events

#table(
  columns: (auto, 1fr),
  stroke: none,
  inset: (x: 8pt, y: 6pt),
  align: (left + horizon, left + horizon),
  column-gutter: 1em,
  [Pre#sub[$i$] spike],
  [increment $I_(s,i)$, $x_i$, $r_i$ by 1; #h(0.6em) $E_i arrow.l E_i - eta_- w_i y_"post"$ #h(0.4em) _(LTD)_],

  [Post spike],
  [$V arrow.l V_"reset"$; #h(0.6em) increment $y_"post"$, $r_"post"$ by 1; #h(0.6em) $E_i arrow.l E_i + eta_+ (w_"max" - w_i) x_i$ for all $i$ #h(0.4em) _(LTP)_],

  [Contingent reward],
  [if pre#sub[1] fired within $Delta_c$ before post-spike: schedule pulse $R_"amount"$ at $t + Delta_"reward"$],
)

=== Neuromodulator $M(t)$

#table(
  columns: 3,
  align: (left, left, left),
  stroke: none,
  inset: (x: 10pt, y: 6pt),
  table.hline(stroke: 0.6pt),
  table.header([Type], [$M$], [$overline(R)$ tracks ($T$)]),
  table.hline(stroke: 0.4pt),
  [Covariance],  [$R - overline(R)$],            [$R$],
  [Gated],       [$R$],                          [$R$],
  [Surprise],    [$(r_"post" - overline(R))^2$], [$r_"post"$],
  [Constant],    [$1$],                          [---],
  table.hline(stroke: 0.6pt),
)

=== Parameters

#table(
  columns: (auto, 1fr, auto),
  align: (left, left, right),
  stroke: none,
  inset: (x: 8pt, y: 4pt),
  table.hline(stroke: 0.6pt),
  table.header([Parameter], [Description], [Value]),
  table.hline(stroke: 0.4pt),

  table.cell(colspan: 3, fill: luma(94%), inset: (x: 8pt, y: 5pt))[*LIF neuron*],
  [$tau_m$],      [Membrane time constant], [20 ms],
  [$E_L$],        [Resting potential],      [$-65$ mV],
  [$R_m$],        [Membrane resistance],    [50 M$Omega$],
  [$theta$],      [Spike threshold],        [$-50$ mV],
  [$V_"reset"$],  [Reset potential],        [$-70$ mV],
  [$tau_"ref"$],  [Refractory period],      [3 ms],

  table.cell(colspan: 3, fill: luma(94%), inset: (x: 8pt, y: 5pt))[*Synapse*],
  [$tau_s$],      [Synaptic decay],         [5 ms],
  [$w_"max"$],    [Max weight],             [10],
  [$w_(i,0)$],    [Initial weight],         [2.0],

  table.cell(colspan: 3, fill: luma(94%), inset: (x: 8pt, y: 5pt))[*Plasticity*],
  [$tau_+, tau_-$], [STDP windows],         [20 ms],
  [$tau_e$],        [Eligibility decay],    [500 ms],
  [$eta_+, eta_-$], [Learning rates],       [$10^(-4)$],

  table.cell(colspan: 3, fill: luma(94%), inset: (x: 8pt, y: 5pt))[*Rate filters*],
  [$tau_r$],             [Rate trace decay],      [500 ms],
  [$tau_(overline(R))$], [Reward baseline decay], [5 s],

  table.cell(colspan: 3, fill: luma(94%), inset: (x: 8pt, y: 5pt))[*Reward / input*],
  [$r_"pre"$],         [Poisson firing rate (each pre)], [20 Hz],
  [$r_"target"$],      [Target post-rate (target-rate reward)], [10 Hz],
  [$Delta_"reward"$],  [Reward delay],                    [1.0 s],
  [$R_"amount"$],      [Reward pulse amplitude],          [1.0],
  [$tau_d$],           [Reward pulse decay],              [200 ms],
  [$Delta_c$],         [Coincidence window],              [20 ms],
  table.hline(stroke: 0.6pt),
)


// ═══════════════════════════════════════════════════════════════════════
= Numerical Methods <numerical-methods>

== Hybrid Integration Scheme

Smooth dynamics are integrated by Euler or classical RK4 between spike events.  Discrete events (pre#sub[1]/pre#sub[2]/post spikes) apply instantaneous jumps to the state vector.

For the RK4 path, threshold crossing detection uses linear interpolation: given $V_0$ and $V_1 = V(t + Delta t)$ from a trial step, the crossing fraction is

$ f = (theta - V_0) / (V_1 - V_0) $

The timestep is split: integrate to $t + f Delta t$, apply the spike, integrate the remainder.  Spike-time accuracy is $cal(O)(Delta t^2)$.  Spike discontinuities violate the smoothness assumption underlying classical convergence theorems, so the effective RK4 order on the full state is below 4 (see §9.1 for the empirical orders).


// ═══════════════════════════════════════════════════════════════════════
= Spatial Credit Assignment <spatial-credit>

The core result of the three-neuron model.  Both synapses receive the same global $M$, but only synapse 1 has high $E_1$ when the delayed reward arrives.

== Mechanism

+ Pre#sub[1] fires before a post-spike $arrow$ $x_1$ is high $arrow$ $E_1$ receives a large LTP jump.
+ The post-spike triggers a contingent reward check: pre#sub[1] fired within $Delta_c$ $arrow$ reward scheduled at $t + Delta_"reward"$.
+ The STDP traces ($x_i$, $y_"post"$) decay within $tilde 20$ ms, but $E_1$ persists ($tau_e = 500$ ms).
+ When the reward pulse arrives ($tilde 1$ s later), $R > 0$ $arrow$ $M > 0$ $arrow$ $d w_1 slash d t = M dot E_1 > 0$.
+ Pre#sub[2] fires independently.  Its eligibility $E_2$ is sometimes nonzero at reward time, but there is no systematic correlation $arrow$ $chevron.l M dot E_2 chevron.r approx 0$.

Over many reward events, $w_1$ grows while $w_2$ remains flat or drifts downward.

== Timescale Hierarchy

#table(
  columns: 3,
  align: (left, left, left),
  table.header([Variable], [Dominant dynamics], [Timescale]),
  [$V$],              [Spikes + subthreshold integration], [$tau_m = 20$ ms],
  [$I_(s,i)$],        [Impulse + fast decay],              [$tau_s = 5$ ms],
  [$x_i, y_"post"$],  [Impulse + moderate decay],          [$tau_(plus.minus) = 20$ ms],
  [$E_i$],            [STDP jumps + slow decay],           [$tau_e = 500$ ms],
  [$r_i, r_"post"$],  [Impulse + slow decay],              [$tau_r = 500$ ms],
  [$overline(R)$],    [Very slow tracking],                [$tau_(overline(R)) = 5$ s],
  [$w_i$],            [Slow drift via $M dot E_i$],         [Emergent; seconds],
)

The eligibility trace ($tau_e = 500$ ms) is the critical bridge.  It outlives the STDP traces ($20$ ms) but decays before the next reward ($tilde 1$ s).  This window is what allows the delayed reward to selectively read out which synapse was causal @fremaux2010.


// ═══════════════════════════════════════════════════════════════════════
= Results <results>

Unless noted otherwise, simulations use $n_"pre" = 1$, RK4 integration ($Delta t = 0.1$ ms), Poisson pre-synaptic input at 20 Hz, covariance neuromodulator, target-rate reward signal ($r_"target" = 10$ Hz), and seed 42.

== Numerical Validation <numerical-validation>

#figure(
  image("../output/rk4_convergence.png", width: 95%),
  caption: [RK4 convergence analysis.  Membrane voltage $V$ converges at order $tilde 1.6$ (degraded from 4 by spike discontinuities); the slow weight $w$ converges at order $tilde 1.3$.  Reference solution computed at $Delta t = 10^(-5)$ ms.],
) <fig:rk4-convergence>

#figure(
  image("../output/rk4_validation.png", width: 95%),
  caption: [Validation of the RK4 integrator against a high-resolution Euler reference.  All state variables ($V$, $w$, $E$, $r_"post"$, $R$, $M$) agree to within expected truncation error.],
) <fig:rk4-validation>

#figure(
  image("../output/derivatives.png", width: 95%),
  caption: [Time derivatives of each state variable, verifying the ODE right-hand side implementation.  Discontinuous jumps correspond to spike events (pre- and post-synaptic).],
) <fig:derivatives>

== Convergence over $(w_0, eta)$ <convergence-analysis>

These figures sweep over initial weights and learning rates to characterise the basin of convergence for the target-rate self-supervisory paradigm.

#figure(
  image("../output/convergence_w_final.png", width: 95%),
  caption: [Final weight $w_"final"$ as a function of initial weight $w_0$ and learning rate $eta$.  The target-rate reward signal drives weights toward a fixed point that achieves $r_"post" approx r_"target"$.],
) <fig:convergence-w-final>

#figure(
  image("../output/convergence_w_stability.png", width: 95%),
  caption: [Weight stability in the second half of the simulation ($t > T slash 2$), measured as standard deviation of $w(t)$.  Low $w_"std"$ indicates convergence to a stable equilibrium; high values indicate oscillation or drift.],
) <fig:convergence-w-stability>

#figure(
  image("../output/convergence_rate_error.png", width: 95%),
  caption: [Relative rate error $|r_"post" - r_"target"| slash r_"target"$ in the second half of the simulation.  The learning rule successfully drives post-synaptic firing rate toward the target across a wide range of initial conditions.],
) <fig:convergence-rate-error>

#figure(
  image("../output/convergence_map.png", width: 95%),
  caption: [Convergence map over the two-dimensional parameter space.  Green indicates successful convergence ($< 10%$ relative rate error); red indicates failure.  The basin of convergence is broad but bounded.],
) <fig:convergence-map>

== Neuromodulator Comparison <neuromod-comparison>

Four neuromodulator types are compared under identical conditions ($n_"pre" = 2$, contingent reward, $T = 100$ s).

#figure(
  image("../output/neuromod_comparison.png", width: 100%),
  caption: [Side-by-side weight trajectories $w_1(t)$ (target-paired) and $w_2(t)$ (distractor) under the four neuromodulator types: covariance (RPE), gated Hebbian, surprise/novelty, and constant (two-factor STDP baseline).],
) <fig:neuromod-comparison>

#figure(
  image("../output/neuromod_details.png", width: 100%),
  caption: [Detailed state-variable traces for each neuromodulator type.  Panels show membrane voltage $V$, eligibility traces $E_i$, firing rates $r$, reward signal $R$, reward baseline $overline(R)$, and modulator $M$.],
) <fig:neuromod-details>

#figure(
  image("../output/neuromod_summary.png", width: 95%),
  caption: [Summary metrics across neuromodulator types: final weights, weight separation ($w_1 - w_2$), post-synaptic firing rate, and weight stability.  The covariance (RPE) type achieves the best spatial credit assignment (largest $w_1 - w_2$ separation).],
) <fig:neuromod-summary>

== Regime Analysis <regime-analysis>

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

#figure(
  image("../output/regime_ic_sensitivity.png", width: 95%),
  caption: [Sensitivity to initial conditions across regimes.  Each regime is simulated from multiple random seeds and initial weights.  Robust regimes (covariance, contingent) show tight convergence; fragile regimes (constant, surprise) show high variance.],
) <fig:regime-ic-sensitivity>

== Spectral Analysis <spectral-analysis>

#figure(
  image("../output/spectrograms.png", width: 100%),
  caption: [Spectrograms of key state variables ($V$, $w$, $r_"post"$) computed via short-time Fourier transform.  The weight $w$ and firing rate $r_"post"$ show low-frequency oscillations ($< 5$ Hz) driven by the reward-modulation feedback loop.  Membrane voltage $V$ exhibits broadband structure dominated by spike timing.],
) <fig:spectrograms>

== Target Function Sweeps <target-functions>

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


// ═══════════════════════════════════════════════════════════════════════
#bibliography("references.bib", style: "apa")
