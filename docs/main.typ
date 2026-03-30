#set document(title: "Neuromodulated STDP: Model, Reward Regimes, and Analysis", date: auto)
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
  #text(size: 14pt)[Model Specification, Reward Regimes, and Numerical Analysis]
  #v(1em)
]

// ═══════════════════════════════════════════════════════════════════════
= Microscopic Foundation: Neural Dynamics

We consider a minimal circuit: one presynaptic neuron connected to one postsynaptic neuron through a unidirectional synapse of weight $w$.

== Membrane Potential (LIF) <lif>

The postsynaptic neuron is a leaky integrate-and-fire (LIF) unit @dayanTheoreticalNeuroscienceComputational2001 @gerstnerNeuronalDynamicsSingle2014. Its membrane potential $V(t)$ obeys

$ tau_m (d V) / (d t) = -(V - E_L) + R_m I_"syn"(t) $ <eq:lif>

where $tau_m = R_m C_m$ is the membrane time constant, $E_L$ the resting potential, and $I_"syn"$ the synaptic current. A spike fires at time $t_"post"^k$ when $V(t) = theta$ with $d V slash d t > 0$; the membrane resets to $V_"reset"$ and integration is suspended for a refractory period $tau_"ref"$.

== Spike Trains

The spike trains of both neurons are sums of Dirac deltas:

$ rho_"pre"(t) = sum_k delta(t - t_"pre"^k), quad rho_"post"(t) = sum_k delta(t - t_"post"^k). $ <eq:spike-trains>

Presynaptic times $\{t_"pre"^k\}$ are given (regular at rate $r_"pre"$ Hz); postsynaptic times are determined by @eq:lif.

== Synaptic Interaction

The synaptic current is the presynaptic spike train filtered through an exponential kernel and scaled by $w$:

$ I_"syn"(t) = w integral_(-infinity)^t alpha(t - s) rho_"pre"(s) d s, quad alpha(t) = tau_s^(-1) e^(-t slash tau_s) Theta(t). $ <eq:isyn>

In ODE form: $d I_s slash d t = -I_s slash tau_s$, with $I_s arrow.l I_s + 1$ at each presynaptic spike.


// ═══════════════════════════════════════════════════════════════════════
= Three-Factor Plasticity Model

The plasticity rule belongs to the class of _three-factor learning rules_ @fremauxNeuromodulatedSpikeTimingDependentPlasticity2016 @izhikevich2007. Standard STDP depends on pre--post spike correlations; three-factor rules gate this local signal with a global neuromodulatory factor.

== STDP Traces

A presynaptic trace $x_"pre"$ and postsynaptic trace $y_"post"$ capture recent spike history @biSynapticModificationsCultured1998:

$ tau_+ (d x_"pre") / (d t) = -x_"pre" + rho_"pre"(t), quad tau_- (d y_"post") / (d t) = -y_"post" + rho_"post"(t). $ <eq:traces>

Experimental measurements place $tau_+ approx tau_- approx 20$ ms.

== Eligibility Trace

Coincident pre- and post-spikes create a temporary memory, the _eligibility trace_ $E(t)$ @izhikevich2007 @gerstnerEligibilityTracesPlasticity2018, bridging the gap between millisecond spike timing and delayed reward:

$ tau_e (d E) / (d t) = -E + underbrace(eta_+ (w_max - w) x_"pre" rho_"post", "LTP") - underbrace(eta_- w y_"post" rho_"pre", "LTD") $ <eq:eligibility>

The soft-bound factors $(w_max - w)$ and $w$ implement _multiplicative STDP_ @Song2000, preventing unbounded weight growth.

== Firing-Rate Estimation

Smooth rate estimates are obtained by exponential filtering:

$ tau_r (d r_"pre") / (d t) = -r_"pre" + rho_"pre"(t), quad tau_r (d r_"post") / (d t) = -r_"post" + rho_"post"(t). $ <eq:rates>

== Weight Update: The Three-Factor Rule

The weight evolves under the product of the eligibility trace and a global modulation signal $M(t)$:

$ (d w) / (d t) = M(t) dot E(t) $ <eq:weight-update>

The modulation signal $M$ is computed from a reward $R(t)$ and a slowly adapting baseline $overline(R)(t)$. The specific form of $R$ depends on the _reward regime_ (see @reward-regimes).


// ═══════════════════════════════════════════════════════════════════════
= Reward Regimes <reward-regimes>

All regimes share the three-factor structure @eq:weight-update. They differ only in how $R$, $M$, and the baseline target are computed.

== Rate Ratio (default) <regime-rate-ratio>

$ R = -(r_"post" - alpha r_"pre")^2, quad M = R - overline(R), quad (d overline(R)) / (d t) = (-overline(R) + R) / tau_(overline(R)) $ <eq:rate-ratio>

The postsynaptic rate is penalised for deviating from a fraction $alpha$ of the presynaptic rate. $M = R - overline(R)$ is a _reward prediction error_ (RPE), analogous to phasic dopamine @schultz1997. This is the R-STDP framework from Fremaux & Gerstner @fremauxNeuromodulatedSpikeTimingDependentPlasticity2016 (Eq. 7, Section 4.3).

== BCM Sliding Threshold <regime-bcm>

$ R = r_"post" (r_"post" - theta_M), quad M = R, quad (d theta_M) / (d t) = (r_"post"^2 slash r_"target" - theta_M) / tau_(overline(R)) $ <eq:bcm>

The sliding threshold $theta_M$ tracks $chevron.l r_"post"^2 chevron.r slash r_"target"$. At equilibrium $theta_M = r^2 slash r_"target"$, giving $R = r^2(1 - r slash r_"target")$: LTP when $r < r_"target"$, LTD when $r > r_"target"$. This provides automatic homeostatic stabilisation.

*Source:* Bienenstock, Cooper & Munro @bcm1982 --- Eq. 4 (modification function $phi$), Eq. 7 (sliding threshold), Figure 2 (the $phi$ curve).

== Fixed Target Rate <regime-fixed-target>

$ R = -(r_"post" - r_"target")^2, quad M = R - overline(R) $ <eq:fixed-target>

A simplified version of rate ratio with an absolute target (no dependence on $r_"pre"$). Models brain--computer interface paradigms where a neuron must learn a specific firing rate @florian2007 (Section 4.4, "Learning a Target Firing-Rate Pattern").

== Information Maximisation <regime-infomax>

$ R = log(r_"post" slash r_0) - beta (r_"post" - r_0), quad M = R - overline(R) $ <eq:infomax>

Derived from the gradient of mutual information $I(X; Y)$ between input and output spike trains, subject to a firing-rate constraint @toyoizumi2005. The $log$ term encourages spiking (Fisher information); the linear penalty enforces metabolic cost. The balance produces a BCM-like learning rule (Toyoizumi et al., Eqs. 12 and 16, Figure 2).

== Cross-Correlation (Temporal Precision) <regime-correlation>

$ R = x_"pre" y_"post" - C_"target", quad M = R - overline(R) $ <eq:correlation>

Rewards temporal coincidence: $R$ is large only when pre- and post-spikes occur within $tilde tau_+$ of each other. With $C_"target" = 0$, any coincidence is rewarded. Relevant to auditory localisation and sequence learning @gerstnerNeuronalDynamicsSingle2014 (Chapter 19).

== Surprise / Novelty <regime-surprise>

$ R = (r_"post" - overline(R))^2, quad M = R, quad (d overline(R)) / (d t) = (-overline(R) + r_"post") / tau_(overline(R)) $ <eq:surprise>

Here $overline(R)$ tracks the _rate_ (not the reward). Surprise $= (r_"post" - overline(R))^2$ is always non-negative; the eligibility trace $E$ alone determines LTP vs LTD direction. Models noradrenergic/cholinergic modulation: novel stimuli enhance plasticity regardless of valence @yu2005 (Introduction, p. 681; Discussion, pp. 687--688).

== ISI / Temporal Regularity <regime-isi>

$ R = -((r_"post" slash r_"target,ISI") - 1)^2, quad M = R - overline(R) $ <eq:isi>

where $r_"target,ISI" = tau_r slash "ISI"_"target"$ converts the target interspike interval to rate-trace units. Penalises fractional deviation from a target ISI, relevant to burst coding and regularity regimes @zeldenrust2018.


// ═══════════════════════════════════════════════════════════════════════
= Complete System of Equations <complete-system>

*State vector:* $bold(x)(t) = (V, y_"post", E, r_"post", overline(R), w) in RR^6$. \
*Input signals:* $(I_s(t), x_"pre"(t), r_"pre"(t))$ from the presynaptic subsystem.

=== Inter-spike dynamics (between events)

$
tau_m (d V) / (d t) &= -(V - E_L) + R_m w I_s(t) \
(d y_"post") / (d t) &= -y_"post" slash tau_- \
(d E) / (d t) &= -E slash tau_e \
(d r_"post") / (d t) &= -r_"post" slash tau_r \
tau_(overline(R)) (d overline(R)) / (d t) &= -overline(R) + T_"regime" \
(d w) / (d t) &= M dot E
$ <eq:interspike>

where $T_"regime"$ is the baseline target ($R$ for most regimes, $r_"post"^2 slash r_"target"$ for BCM, $r_"post"$ for surprise).

=== Pre-synaptic spike at $t_"pre"^k$

$ I_s arrow.l I_s + 1, quad x_"pre" arrow.l x_"pre" + 1, quad r_"pre" arrow.l r_"pre" + 1 $
$ E arrow.l E - eta_- w y_"post"(t_"pre"^(k-)) quad "(LTD eligibility)" $

=== Post-synaptic spike at $t_"post"^k$

$ V arrow.l V_"reset", quad y_"post" arrow.l y_"post" + 1, quad r_"post" arrow.l r_"post" + 1 $
$ E arrow.l E + eta_+ (w_max - w) x_"pre"(t_"post"^(k-)) quad "(LTP eligibility)" $


// ═══════════════════════════════════════════════════════════════════════
= Numerical Methods <numerical-methods>

== Hybrid Integration Scheme

The simulation uses a hybrid event-driven / continuous scheme. Smooth dynamics are integrated by Euler or classical RK4 between spike events. Discrete events (pre/post spikes) apply instantaneous jumps to the state vector.

For the RK4 path, threshold crossing detection uses linear interpolation: given $V_0$ and $V_1 = V(t + Delta t)$ from a trial step, the crossing fraction is

$ f = (theta - V_0) / (V_1 - V_0) $

The timestep is then split: integrate to $t + f Delta t$, apply the spike, integrate the remainder $t + f Delta t arrow t + Delta t$. This is first-order accurate in spike time (error $cal(O)(Delta t^2)$).

== RK4 Convergence

Convergence was measured against a fine-$Delta t$ reference solution ($Delta t = 10 mu s$):

#table(
  columns: 4,
  align: (left, center, center, center),
  table.header([Variable], [Euler (expect 1)], [RK4 (expect 4)], [Note]),
  [$V$ (mV)], [0.20], [1.57], [Degraded by spike discontinuities],
  [$w$],       [1.79], [1.25], [Slow variable; averaging effects],
)

Spike discontinuities (voltage resets, trace jumps) violate the smoothness assumption underlying classical convergence theorems. The linear interpolation for spike times limits accuracy to $cal(O)(Delta t^2)$, reducing RK4's effective order. Euler benefits from compensating-error averaging on the slow weight variable.


// ═══════════════════════════════════════════════════════════════════════
= Analysis Results <analysis-results>

All simulations use $Delta t = 0.1$ ms, RK4, $T = 10$ s unless noted.

== Timescale Hierarchy

#table(
  columns: 3,
  align: (left, left, left),
  table.header([Variable], [Dominant dynamics], [Timescale]),
  [$V$],        [Spikes + subthreshold integration], [$tau_m = 20$ ms],
  [$I_s$],      [Impulse + fast decay],              [$tau_s = 5$ ms],
  [$x_"pre", y_"post"$], [Impulse + moderate decay], [$tau_(plus.minus) = 20$ ms],
  [$E$],        [STDP jumps + slow decay],           [$tau_e = 500$ ms],
  [$r_"pre", r_"post"$], [Impulse + slow decay],    [$tau_r = 500$ ms],
  [$overline(R)$], [Very slow tracking],             [$tau_(overline(R)) = 5$ s],
  [$w$],        [Slow drift via $M dot E$],           [Emergent; seconds],
)

== Parameter Sensitivity

The most critical parameters are:

*$w_0$ (initial weight) --- bistability.* A critical weight $w_"crit" approx 1.5 dash 2.0$ separates a silent regime from an active one. Below $w_"crit"$, the neuron never fires, $E = 0$ identically, and $d w slash d t = M dot 0 = 0$ --- the weight is frozen forever. This is the _silent synapse problem_ @fremauxNeuromodulatedSpikeTimingDependentPlasticity2016 (Section 4.3): reward-modulated STDP cannot bootstrap a silent synapse.

The critical weight can be estimated from $V_"ss" approx E_L + R_m w r_"pre" tau_s$. Setting $V_"ss" = theta$ gives $w_"crit" approx (theta - E_L) / (R_m r_"pre" tau_s) approx 3$, reduced to $tilde 1.5$ by peak-to-mean EPSP ratios.

*$R_m$ (membrane resistance) --- high sensitivity.* Directly scales synaptic drive; increasing $R_m$ from 10 to 120 sweeps the post rate from 4 to 20 Hz.

*$alpha$ (target ratio) --- moderate sensitivity.* The weight adjusts to track the target direction but doesn't converge precisely within $T = 10$ s.

*$eta_+$, $tau_e$, $tau_(overline(R))$ --- low sensitivity.* Robust across order-of-magnitude changes.

== Regime Comparison <regime-comparison>

All 7 regimes were run with identical parameters ($w_0 = 2$, $r_"pre" = 20$ Hz, $T = 10$ s):

#table(
  columns: 4,
  align: (left, center, center, left),
  table.header([Regime], [Post rate (Hz)], [$w_"final"$], [Behaviour]),
  [rate_ratio],    [13.4], [1.92], [Active regulation; ongoing $M$ oscillations],
  [bcm],           [20.0], [4.50], [Strong potentiation to BCM equilibrium],
  [fixed_target],  [20.0], [2.06], [Minimal change; RPE cancellation],
  [infomax],       [20.0], [2.00], [Minimal change; RPE cancellation],
  [correlation],   [20.0], [2.00], [Zero-mean coincidence fluctuations],
  [surprise],      [20.0], [3.34], [Monotonic potentiation; $M > 0$ always],
  [isi_target],    [20.0], [2.01], [Minimal change; RPE cancellation],
)

=== Key finding: RPE baseline cancellation

With deterministic pre-spikes, the post-neuron fires at a nearly constant rate, making $R$ constant. Once $overline(R)$ converges to $R$, the modulation $M = R - overline(R) arrow 0$ and learning stops --- regardless of distance from the target. This affects all RPE-based regimes (fixed_target, infomax, correlation, isi_target).

*Rate ratio avoids this* because $R$ depends on _both_ $r_"pre"$ and $r_"post"$, creating ongoing fluctuations that keep $M eq.not 0$.

*BCM and surprise avoid this* because $M = R$ (no baseline subtraction), so $M eq.not 0$ whenever $R eq.not 0$.

In biological networks, Poisson spiking and network noise would provide the reward variability that RPE needs. The deterministic pre-spikes expose a structural limitation of RPE-based three-factor rules.


// ═══════════════════════════════════════════════════════════════════════
= Timescale Separation and Relaxation Oscillations <oscillations>

The reward baseline $overline(R)$ and weight $w$ evolve on timescales much slower than all other state variables. The baseline has $tau_(overline(R)) = 5$ s ($250 times$ slower than $tau_m$). The weight evolves through $dot(w) = M dot E$ where both $M$ and $E$ (scaled by $eta_(plus.minus) tilde 10^(-4)$) are small.

== Spike-Count Quantisation

The postsynaptic neuron fires a discrete number of spikes $N(w)$ per presynaptic inter-spike interval. At critical weights $w_c^((n))$ where $N$ increments from $n$ to $n+1$, the firing rate $r_"post"$ undergoes a step change, producing a discontinuous shift in $R$.

== Mechanism of the Limit Cycle

Near $w_c^((n))$, the system exhibits a relaxation oscillation:

+ *Slow recovery.* With $w < w_c^((n))$, the neuron fires $N = n$ spikes. $M approx 0$, $dot(w) approx 0$. Residual $M > 0$ drifts $w$ upward.

+ *Threshold crossing.* When $w$ crosses $w_c^((n))$, an extra spike fires. $r_"post"$ jumps, $R$ drops sharply. Since $overline(R)$ hasn't tracked the change ($tau_(overline(R))$ is large), $M = R - overline(R) lt.double 0$, driving $dot(w) lt.double 0$.

+ *Subcritical reset.* $w$ is pushed below $w_c^((n))$, the extra spike ceases, $R$ recovers, and the cycle repeats.

This is a _grazing bifurcation_: the smooth weight dynamics interact with a discontinuous reset map at the spike threshold, generically producing a stable limit cycle rather than a fixed point.


// ═══════════════════════════════════════════════════════════════════════
= Parameters <parameters>

#table(
  columns: 4,
  align: (left, left, left, left),
  table.header([Parameter], [Description], [Value], [Source]),
  [$tau_m$],         [Membrane time constant],     [20 ms],     [@dayanTheoreticalNeuroscienceComputational2001],
  [$E_L$],           [Resting potential],           [$-65$ mV],  [@dayanTheoreticalNeuroscienceComputational2001],
  [$R_m$],           [Membrane resistance],         [50 M$Omega$], [@dayanTheoreticalNeuroscienceComputational2001],
  [$theta$],         [Spike threshold],             [$-50$ mV],  [@gerstnerNeuronalDynamicsSingle2014],
  [$V_"reset"$],     [Reset potential],             [$-70$ mV],  [@gerstnerNeuronalDynamicsSingle2014],
  [$tau_"ref"$],     [Refractory period],           [3 ms],      [@gerstnerNeuronalDynamicsSingle2014],
  [$tau_s$],         [Synaptic decay],              [5 ms],      [@dayanTheoreticalNeuroscienceComputational2001],
  [$tau_+, tau_-$],  [STDP windows],                [20 ms],     [@biSynapticModificationsCultured1998],
  [$tau_r$],         [Rate trace decay],            [500 ms],    [@gerstnerNeuronalDynamicsSingle2014],
  [$tau_e$],         [Eligibility decay],           [500 ms],    [@fremauxNeuromodulatedSpikeTimingDependentPlasticity2016],
  [$tau_(overline(R))$], [Reward baseline],         [5 s],       [@fremauxNeuromodulatedSpikeTimingDependentPlasticity2016],
  [$alpha$],         [Target rate ratio],           [0.5],       [---],
  [$eta_+, eta_-$],  [Learning rates],              [$10^(-4)$], [@Song2000],
  [$w_"max"$],       [Max weight],                  [10],        [---],
  [$w_0$],           [Initial weight],              [2.0],       [---],
)


// ═══════════════════════════════════════════════════════════════════════
#bibliography("references.bib", style: "apa")
