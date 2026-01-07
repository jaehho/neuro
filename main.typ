#set math.equation(numbering: "(1)")

= Hodgkin-Huxley Model

== Cable Equation Derivation

#image("figures/fig_6-6.png") For a cylindrical segment with radius $a$,
intracellular resistivity $r_L$, and membrane capacitance per unit area
$c_m$:

Longitudinal resistance: $ R_L eq frac(r_L Delta x, pi a^2) $

Longitudinal voltage drop:
$ Delta V eq minus I_L R_L eq minus I_L frac(r_L Delta x, pi a^2) $

Longitudinal current:
$ I_L eq frac(minus pi a^2, r_L) frac(Delta V, Delta x) $

Membrane current: $ i_m eq sum_i g_i lr((V minus E_i)) $

The sum of the Longitudinal current coming in and out of the segment,
the membrane current, and external current ($i_e$) is equal to the total
capacitance of the neuron $2 pi a Delta x c_m$ times the rate of change
of the membrane potential $frac(diff V, diff t)$:

#quote[
The longitudinal input current is with respect to the previous
segment/neuron, and is assumed to be equal to the longitudinal output
current with respect to the next segment/neuron which have the same
radius $a$, specific resistivity $r_L$, and length $Delta x$.
]

$ 2 pi a Delta x c_m frac(diff V, diff t) eq minus lr((frac(pi a^2, r_L) frac(Delta V, Delta x))) macron(v)_(upright("in")) plus lr((frac(pi a^2, r_L) frac(Delta V, Delta x))) macron(v)_(upright("out")) minus 2 pi a Delta x lr((i_m minus i_e)) $

Dividing both sides by $2 pi a Delta x$ and taking the limit as
$Delta x arrow.r 0$ gives the cable equation:

$ c_m frac(diff V, diff t) eq frac(a, 2 r_L) frac(diff^2 V, diff x^2) minus i_m plus i_e $

---

== Multi-Compartment Approximation

#figure([#image("figures/fig_6-17.png")],
  caption: [
    Figure 6.17 Propagation of an action potential along a
    multi-compartment model axon
  ]
)

The neuron is divided into discrete compartments indexed by $mu$. Each
has membrane potential $V_mu$ and membrane area $A_mu$. For a cable
without branching:

The longitudinal resistance between compartments $mu$ and $mu prime$:
$ R_L^(lr((mu comma mu prime))) eq frac(r_L L_mu, 2 pi a_mu^2) plus frac(r_L L_(mu prime), 2 pi a_(mu prime)^2) $

The conductance between compartments $mu$ and $mu prime$ is the
reciprocal of the resistance divided by the area of compartment $mu$,
$A_mu eq 2 pi a_mu^2 L_mu$:

$ g_(mu comma mu prime) eq frac(1, R_L^(lr((mu comma mu prime))) A_mu) eq frac(a_mu a_(mu prime)^2, r_L L_mu lr((L_mu a_(mu prime)^2 plus L_(mu prime) a_mu^2))) $

Letting the length of each compartment be denoted by $L_mu$, the
equation for compartment $mu$ is:

$ c_m frac(d V_mu, d t) eq minus i_m^mu plus i_e^mu plus g_(mu comma mu plus 1) lr((V_(mu plus 1) minus V_mu)) plus g_(mu comma mu minus 1) lr((V_(mu minus 1) minus V_mu)) $

---

== Synaptic Transmission

As with a voltage-dependent conductance, a synaptic conductance can be
written as the product of a maximal conductance and an open channel
probability, $g_s eq g^(‾)_s P$, where $P$ can be expressed as the joint
probability of transmitter binding and channel opening,
$P eq P_(upright("rel")) P_s$.

$ c_m frac(d V, d t) eq minus i_m plus i_e minus g^(‾)_s P lr((V minus E_s)) $
where $E_s$ is the synaptic reversal potential, which is typically
around 0 mV for excitatory synapses and around -70 mV for inhibitory
synapses.

The synaptic open channel probability $P$ has complex dynamics, but can
be simplified to a exponentially decaying function that has a discrete
jump after a presynaptic spike time.

or in the multi-compartment model for compartment $mu$:
$ c_m frac(d V_mu, d t) eq minus i_m^mu plus i_e^mu minus g^(‾)_s P lr((V_mu minus E_s)) plus g_(mu comma mu plus 1) lr((V_(mu plus 1) minus V_mu)) plus g_(mu comma mu minus 1) lr((V_(mu minus 1) minus V_mu)) $

== synapse

== Appendix A — Conductance-based neuron models

#figure([#image("figures/tau_and_steady.png")],
  caption: [
    Gating Time Constants and Steady-State Values
  ]
)

Hodgkin–Huxley membrane current can be expressed as a sum of a leakage
current, a delayed-rectifier K⁺ current, and a transient Na⁺ current:

$ i_m eq g_L lr((V minus E_L)) plus g_K comma n^4 lr((V minus E_K)) plus g_(N a) comma m^3 h lr((V minus E_(N a))) dot.basic $

where $m comma h comma n$ are dynamic gating variables between 0 and 1.

For any gate $z in m comma h comma n$:

$ frac(d z, d t) eq alpha_z lr((V)) lr((1 minus z)) minus beta_z lr((V)) z quad arrow.r.double quad tau_z lr((V)) frac(d z, d t) eq z_oo lr((V)) minus z comma $

where

$ tau_z eq 1 slash lr((alpha_z plus beta_z)) $

and

$ z_oo eq alpha_z / lr((alpha_z plus beta_z)) $

Numerical integration:

Over small $Delta t$, integrate $V$ and then each gate with the same
stable update:

$ tau_z frac(d z, d t) eq z_oo minus z quad arrow.r.double quad z lr((t plus Delta t)) eq z_oo plus lr((z lr((t)) minus z_oo)) e^(minus Delta t slash tau_z) dot.basic $

== Appendix B - Synaptic open probability $P_s lr((t))$ and postsynaptic
conductance

#figure([#image("figures/fig_5-14.png")],
  caption: [
    Figure 5.14 A fit of the model discussed in the text to the average
    EPSC (excitatory postsynaptic current)
  ]
)

If $k$ transmitter molecules bind to open a receptor:

$ frac(d P_s, d t) eq alpha_s lr((1 minus P_s)) minus beta_s P_s comma quad alpha_s prop lr([upright("transmitter")])^k dot.basic $

#quote[
"Here, $beta_s$ determines the closing rate of the channel and is
usually assumed to be a constant. The opening rate, $alpha_s$, on the
other hand, depends on the concentration of transmitter available for
binding to the receptor. If the concentration of transmitter at the site
of the synaptic channel is \[transmitter\], the probability of finding k
transmitter molecules within binding range of the channel is
proportional to \[transmitter\]k, and $alpha_s$ is some constant of
proportionality times this factor… As a simple model of transmitter
release, we assume that the transmitter concentration in the synaptic
cleft rises extremely rapidly after vesicle release, remains at a high
value for a period of duration T, and then falls rapidly to 0. Thus, the
transmitter concentration is modeled as a square pulse."
]

While the transmitter concentration in the cleft is nonzero, $alpha_s$
is so much larger than $beta_s$ that we can ignore the term involving
$beta_s$ in the above equation. Under this assumption

$ P_s lr((t)) eq 1 plus lr((P_s lr((0)) minus 1)) exp lr((minus alpha_s t)) quad upright("for ") 0 lt.eq t lt.eq T $

The open probability takes its maximum value at time $t eq T$ and then,
for $t gt.eq T$, decays exponentially at a rate determined by the
constant $beta_s$, and $alpha_s$ is 0 after because transmitter
concentration rapidly falls after T

$ P_s lr((t)) eq P_s lr((T)) exp lr((minus beta_s lr((t minus T)))) quad upright("for ") t gt.eq T dot.basic $

If $P_s lr((0)) eq 0$, as it will if there is no synaptic release
immediately before the release at t \= 0, the first equation simplifies
to $P_s lr((t)) eq 1 minus exp lr((minus alpha_s t))$ for
$0 lt.eq t lt.eq T$, and this reaches a maximum value
$P_max eq 1 minus exp lr((minus alpha_s T))$. In terms of this parameter
the synaptic open probability at time $T$ in the general case can be
written as

$ P_s lr((T)) eq P_s lr((0)) plus P_max lr((1 minus P_s lr((0)))) $

The figure shows a fit to a recorded postsynaptic current using this
formalism. In this case, $beta_s$ was set to $0.19 m s^(minus 1)$. The
transmitter concentration was modeled as a square pulse of duration $T$
\= 1 ms during which $alpha_s eq 0.93 m s^(minus 1)$. Inverting these
values, we find that the time constant determining the rapid rise seen
in the figure is 0.9 ms, while the fall of the current is an exponential
with a time constant of 5.26 ms.

=== Spike Trains

Assuming a fast synapse, the rise of the conductance following a
presynaptic action potential is so rapid that it can be approximated as
instantaneous. Between spikes, $P_s$ decays exponentially, and after
each spike, $P_s$ jumps by an amount proportional to the distance from
its maximum value:

$ tau_s frac(d P_s, d t) eq minus P_s comma #h(2em) P_s arrow.r P_s plus P_max lr((1 minus P_s)) dot.basic $

== Appendix C — Transmitter release probability $P_(upright("rel")) lr((t))$ and short-term plasticity

$P_(upright("rel"))$ denotes the average release probability across one
or many independent sites. It is described using a simple nonmechanistic
model that has similarities to the model of $P_s$. For both facilitation
and depression, the release probability after a long period of
presynaptic silence is $P_(upright("rel")) eq P_0$.

$ tau_P frac(d P_(upright("rel")), d t) eq P_0 minus P_(upright("rel")) $

#strong[Spike-triggered updates.]

- Facilitation:
  $P_(upright("rel")) arrow.r P_(upright("rel")) plus f_F lr((1 minus P_(upright("rel")))) comma lr((0 lt.eq f_F lt.eq 1))$
- Depression:
  $P_(upright("rel")) arrow.r f_D P_(upright("rel")) comma lr((0 lt.eq f_D lt.eq 1))$.

=== Analysis for Poisson Spike Trains

Assume presynaptic spikes form a Poisson process with rate $r$. Between
spikes, $P_(upright("rel"))$ relaxes exponentially to $P_0$:
$ tau_P frac(d P_(upright("rel")), d t) eq P_0 minus P_(upright("rel")) $

The general solution is

$ P_(upright("rel")) lr((t)) eq P_0 plus lr([P_(upright("rel")) lr((t_0)) minus P_0]) e^(minus lr((t minus t_0)) slash tau_P) dot.basic $

==== Facilitation

#strong[Spike rule.] After a spike,
$ P_(upright("rel")) arrow.r P_(upright("rel")) plus f_F lr((1 minus P_(upright("rel")))) $
#strong[One ISI.] Let two spikes be separated by $tau$. If
$P_(upright("rel"))$ equals its mean
$angle.l P_(upright("rel")) angle.r$ just #emph[before] the first spike,
then just #emph[after] that spike it is
$angle.l P_(upright("rel")) angle.r plus f_F lr((1 minus angle.l P_(upright("rel")) angle.r))$.
During the interval $tau$ it decays towards it’s resting value $P_0$,
and just before the next spike it is

$ P_0 plus (angle.l P_(upright("rel")) angle.r plus f_F lr((1 minus angle.l P_(upright("rel")) angle.r)) minus P_0) e^(minus tau slash tau_P) $

#strong[Average decay factor over Poisson ISIs.]

For a Poisson process with mean firing rate $r$, the ISI distribution is
exponential:

$ p lr((tau)) eq r e^(minus r tau) comma quad tau gt.eq 0 dot.basic $
This gives the probability density for the time gap between spikes.

The expected decay factor over a random interval $tau$ is
$ angle.l e^(minus tau slash tau_P) angle.r eq integral_0^oo e^(minus tau slash tau_P) p lr((tau)) d tau eq r integral_0^oo e^(minus r tau minus tau slash tau_P) d tau eq frac(r, r plus 1 slash tau_P) eq frac(r tau_P, 1 plus r tau_P) dot.basic $

#strong[Consistency equation.] For steady state,
$ angle.l P_(upright("rel")) angle.r eq P_0 plus (angle.l P_(upright("rel")) angle.r plus f_F lr((1 minus angle.l P_(upright("rel")) angle.r)) minus P_0 ) frac(r tau_P, 1 plus r tau_P) $

#strong[Solve.]
$ angle.l P_(upright("rel")) angle.r eq frac(P_0 plus f_F r tau_P, 1 plus r f_F tau_P) $

#strong[Behavior.] Low $r$:
$angle.l P_(upright("rel")) angle.r approx P_0$. High $r$:
$angle.l P_(upright("rel")) angle.r arrow.r 1$. Transmission rate equals
$r angle.l P_(upright("rel")) angle.r$, so it grows $approx P_0 r$ at
low $r$ and $approx r$ at high $r$.

