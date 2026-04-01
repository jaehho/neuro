#set document(title: "Neuromodulated STDP: Three-Neuron Model", date: auto)
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
  #text(size: 14pt)[A Three-Neuron Model for Spatial Credit Assignment]
  #v(1em)
]

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

The postsynaptic rate is penalised for deviating from a target.  This is self-supervisory (not from the literature) but useful for verifying the ODE machinery.

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
= Complete System of Equations <complete-system>

*State vector:* $bold(x)(t) = (V, y_"post", r_"post", overline(R), I_(s,1), x_1, E_1, r_1, w_1, I_(s,2), x_2, E_2, r_2, w_2) in RR^14$.

=== Inter-spike dynamics

$
tau_m (d V) / (d t) &= -(V - E_L) + R_m (w_1 I_(s,1) + w_2 I_(s,2)) \
(d y_"post") / (d t) &= -y_"post" slash tau_- \
(d r_"post") / (d t) &= -r_"post" slash tau_r \
tau_(overline(R)) (d overline(R)) / (d t) &= -overline(R) + T_"neuromod" \
(d I_(s,i)) / (d t) &= -I_(s,i) slash tau_s \
(d x_i) / (d t) &= -x_i slash tau_+ \
(d E_i) / (d t) &= -E_i slash tau_e \
(d r_i) / (d t) &= -r_i slash tau_r \
(d w_i) / (d t) &= M dot E_i
$ <eq:interspike>

where $T_"neuromod"$ is the baseline tracking target: $R$ for covariance and gated, $r_"post"$ for surprise, $overline(R)$ for constant.

=== Pre-synaptic spike at $t_i^k$

$ I_(s,i) arrow.l I_(s,i) + 1, quad x_i arrow.l x_i + 1, quad r_i arrow.l r_i + 1 $
$ E_i arrow.l E_i - eta_- w_i y_"post"(t_i^(k-)) quad "(LTD eligibility)" $

=== Post-synaptic spike at $t_"post"^k$

$ V arrow.l V_"reset", quad y_"post" arrow.l y_"post" + 1, quad r_"post" arrow.l r_"post" + 1 $
$ E_i arrow.l E_i + eta_+ (w_max - w_i) x_i (t_"post"^(k-)) quad "(LTP eligibility, both synapses)" $

=== Reward scheduling (contingent mode)

If pre#sub[1] fired within $Delta_c$ before the post-spike, schedule a global reward pulse at $t_"post"^k + Delta_"reward"$.

=== Summary table

#table(
  columns: 4,
  align: (left, left, left, left),
  table.header([Neuromodulator], [$M$], [$overline(R)$ tracks], [Source]),
  [Covariance],  [$R - overline(R)$],           [$R$],         [@fremauxNeuromodulatedSpikeTimingDependentPlasticity2016 Eq. 7],
  [Gated],       [$R$],                          [$R$],         [@fremauxNeuromodulatedSpikeTimingDependentPlasticity2016 Eq. 14],
  [Surprise],    [$(r_"post" - overline(R))^2$], [$r_"post"$],  [@yu2005],
  [Constant],    [$1$],                          [---],         [Two-factor baseline],
)


// ═══════════════════════════════════════════════════════════════════════
= Numerical Methods <numerical-methods>

== Hybrid Integration Scheme

Smooth dynamics are integrated by Euler or classical RK4 between spike events.  Discrete events (pre#sub[1]/pre#sub[2]/post spikes) apply instantaneous jumps to the state vector.

For the RK4 path, threshold crossing detection uses linear interpolation: given $V_0$ and $V_1 = V(t + Delta t)$ from a trial step, the crossing fraction is

$ f = (theta - V_0) / (V_1 - V_0) $

The timestep is split: integrate to $t + f Delta t$, apply the spike, integrate the remainder.  Spike-time accuracy is $cal(O)(Delta t^2)$.

== RK4 Convergence

Spike discontinuities violate the smoothness assumption underlying classical convergence theorems, reducing RK4's effective order:

#table(
  columns: 4,
  align: (left, center, center, center),
  table.header([Variable], [Euler (expect 1)], [RK4 (expect 4)], [Note]),
  [$V$ (mV)], [0.20], [1.57], [Degraded by spike discontinuities],
  [$w$],       [1.79], [1.25], [Slow variable; averaging effects],
)


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
= Parameters <parameters>

#table(
  columns: 4,
  align: (left, left, left, left),
  table.header([Parameter], [Description], [Value], [Source]),
  [$tau_m$],            [Membrane time constant],       [20 ms],       [@dayanTheoreticalNeuroscienceComputational2001],
  [$E_L$],              [Resting potential],             [$-65$ mV],    [@dayanTheoreticalNeuroscienceComputational2001],
  [$R_m$],              [Membrane resistance],           [50 M$Omega$], [@dayanTheoreticalNeuroscienceComputational2001],
  [$theta$],            [Spike threshold],               [$-50$ mV],    [@gerstnerNeuronalDynamicsSingle2014],
  [$V_"reset"$],        [Reset potential],               [$-70$ mV],    [@gerstnerNeuronalDynamicsSingle2014],
  [$tau_"ref"$],        [Refractory period],             [3 ms],        [@gerstnerNeuronalDynamicsSingle2014],
  [$tau_s$],            [Synaptic decay],                [5 ms],        [@dayanTheoreticalNeuroscienceComputational2001],
  [$tau_+, tau_-$],     [STDP windows],                  [20 ms],       [@biSynapticModificationsCultured1998],
  [$tau_r$],            [Rate trace decay],              [500 ms],      [@gerstnerNeuronalDynamicsSingle2014],
  [$tau_e$],            [Eligibility decay],             [500 ms],      [@fremauxNeuromodulatedSpikeTimingDependentPlasticity2016],
  [$tau_(overline(R))$], [Reward baseline],              [5 s],         [@fremauxNeuromodulatedSpikeTimingDependentPlasticity2016],
  [$eta_+, eta_-$],     [Learning rates],                [$10^(-4)$],   [@Song2000],
  [$w_"max"$],          [Max weight],                    [10],          [---],
  [$w_(1,0), w_(2,0)$], [Initial weights],               [2.0],         [---],
  [$r_"pre"$],          [Poisson firing rate (each)],    [20 Hz],       [@fremaux2010],
  [$Delta_"reward"$],   [Reward delay],                  [1.0 s],       [@izhikevich2007],
  [$R_"amount"$],       [Reward pulse amplitude],        [1.0],         [@izhikevich2007],
  [$tau_d$],            [Reward pulse decay],            [200 ms],      [---],
  [$Delta_c$],          [Coincidence window],            [20 ms],       [@izhikevich2007],
)


// ═══════════════════════════════════════════════════════════════════════
#bibliography("references.bib", style: "apa")
