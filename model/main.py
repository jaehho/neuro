from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Params:
    T: float = 20.0
    dt: float = 1e-4
    seed: int = 1
    record_every: float = 1e-4

    r_pre_rate: float = 20.0       # Poisson rate of presynaptic input (Hz)

    # LIF membrane  (eq:ref_V)
    tau_m: float = 0.02            # τ_m  – membrane time constant (s)
    E_L: float = -65.0             # E_L  – resting potential (mV)
    V_reset: float = -70.0         # V_reset
    theta: float = -50.0           # θ    – spike threshold (mV)
    tau_ref: float = 0.003         # τ_ref – absolute refractory period (s)
    V0: float = -65.0              # initial membrane potential (mV)
    ref_remaining0: float = 0.0    # initial refractory time remaining (s)

    # Synaptic current  (eq:ref_Is)
    tau_s: float = 0.005           # τ_s
    R_m: float = 50.0              # R_m  – membrane resistance (MΩ)
    I_s0: float = 0.0              # initial synaptic current

    # STDP eligibility traces  (eq:ref_xpre, eq:ref_ypost)
    tau_plus: float = 0.02         # τ_+  – pre-synaptic trace decay
    tau_minus: float = 0.02        # τ_-  – post-synaptic trace decay
    x_pre0: float = 0.0            # initial pre-synaptic STDP trace
    y_post0: float = 0.0           # initial post-synaptic STDP trace

    # Firing-rate filters  (eq:ref_rpre, eq:ref_rpost)
    tau_r: float = 0.5             # τ_r
    r_pre0: float = 0.0            # initial pre-synaptic firing-rate filter
    r_post0: float = 0.0           # initial post-synaptic firing-rate filter

    # Eligibility trace  (eq:ref_E)
    tau_e: float = 0.5             # τ_e
    E0: float = 0.0                # initial eligibility trace

    # Reward baseline  (eq:ref_Rbar)
    tau_Rbar: float = 5.0          # τ_R̄
    R_bar0: float = 0.0            # initial reward baseline
    alpha: float = 0.5             # α   – target r_post / r_pre ratio

    # Plasticity  (eq:ref_pre_E, eq:ref_post_E, eq:ref_w)
    w0: float = 2.0                # initial weight
    wmax: float = 10.0             # w_max
    eta_plus: float = 1e-4         # η_+  – LTP rate
    eta_minus: float = 1e-4        # η_-  – LTD rate


def simulate(p: Params):
    n = int(p.T / p.dt)
    period_steps = max(1, round(1.0 / (p.r_pre_rate * p.dt)))

    # ------------------------------------------------------------------ #
    # State variables – names and grouping follow analysis.tex Sec. 1–3  #
    # ------------------------------------------------------------------ #

    # Membrane
    V = p.V0
    ref_remaining = p.ref_remaining0

    # Input subsystem (Sec. 1 / eq:ref_Is, eq:ref_xpre, eq:ref_rpre)
    I_s   = p.I_s0
    x_pre = p.x_pre0
    r_pre = p.r_pre0

    # Postsynaptic subsystem (Sec. 1 / eq:ref_ypost..eq:ref_w)
    y_post = p.y_post0
    r_post = p.r_post0
    E      = p.E0
    R_bar  = p.R_bar0
    w      = p.w0

    rec_step = max(1, int(p.record_every / p.dt))
    m = n // rec_step + 2

    rec = {k: np.zeros(m) for k in [
        "t", "V", "w", "I_s",
        "x_pre", "y_post", "E",
        "r_pre", "r_post", "R", "R_bar", "M",
        "pre_spike_bin", "post_spike_bin", "is_refractory",
    ]}

    pre_spike_times = []
    post_spike_times = []
    k = 0

    for step in range(n):
        t = step * p.dt

        # ------------------------------------------------------------------ #
        # Sec. 2 – Pre-synaptic spike event                                  #
        # ------------------------------------------------------------------ #
        pre_spike = 1 if (step % period_steps == 0) else 0
        if pre_spike:
            pre_spike_times.append(t)
            I_s   += 1.0   # eq:ref_pre_Is
            x_pre += 1.0   # eq:ref_pre_xpre
            r_pre += 1.0   # eq:ref_pre_rpre
            E -= p.eta_minus * w * y_post  # eq:ref_pre_E  (LTD)

        # ------------------------------------------------------------------ #
        # Sec. 1 – Inter-spike dynamics: input subsystem                     #
        # ------------------------------------------------------------------ #
        I_s   += p.dt * (-I_s   / p.tau_s)      # eq:ref_Is
        x_pre += p.dt * (-x_pre / p.tau_plus)   # eq:ref_xpre
        r_pre += p.dt * (-r_pre / p.tau_r)      # eq:ref_rpre

        # ------------------------------------------------------------------ #
        # Sec. 1 – Inter-spike dynamics: membrane potential                  #
        # ------------------------------------------------------------------ #
        post_spike    = 0
        is_refractory = 1 if ref_remaining > 0.0 else 0

        if ref_remaining <= 0.0:
            dV    = (p.dt / p.tau_m) * (-(V - p.E_L) + p.R_m * w * I_s)  # eq:ref_V
            V_new = V + dV
            if V < p.theta and V_new >= p.theta:
                post_spike = 1
                post_spike_times.append(t)
                V             = p.V_reset   # eq:ref_post_V
                ref_remaining = p.tau_ref
            else:
                V = V_new
        else:
            ref_remaining -= p.dt
            V = p.V_reset

        # ------------------------------------------------------------------ #
        # Sec. 3 – Post-synaptic spike event                                 #
        # ------------------------------------------------------------------ #
        if post_spike:
            y_post += 1.0  # eq:ref_post_ypost
            r_post += 1.0  # eq:ref_post_rpost
            E += p.eta_plus * (p.wmax - w) * x_pre  # eq:ref_post_E  (LTP)

        # ------------------------------------------------------------------ #
        # Sec. 1 – Inter-spike dynamics: postsynaptic subsystem              #
        # ------------------------------------------------------------------ #
        y_post += p.dt * (-y_post / p.tau_minus)  # eq:ref_ypost
        r_post += p.dt * (-r_post / p.tau_r)      # eq:ref_rpost
        E      += p.dt * (-E      / p.tau_e)      # eq:ref_E

        R     = -(r_post - p.alpha * r_pre) ** 2                    # instantaneous reward
        R_bar += (p.dt / p.tau_Rbar) * (-R_bar + R)                # eq:ref_Rbar
        M     = R - R_bar                                           # neuromodulator: R - R̄
        w    += p.dt * M * E                                        # eq:ref_w
        w     = min(p.wmax, max(0.0, w))

        if step % rec_step == 0:
            rec["t"][k]              = t
            rec["V"][k]              = V
            rec["w"][k]              = w
            rec["I_s"][k]            = I_s
            rec["x_pre"][k]          = x_pre
            rec["y_post"][k]         = y_post
            rec["E"][k]              = E
            rec["r_pre"][k]          = r_pre
            rec["r_post"][k]         = r_post
            rec["R"][k]              = R
            rec["R_bar"][k]          = R_bar
            rec["M"][k]              = M
            rec["pre_spike_bin"][k]  = pre_spike
            rec["post_spike_bin"][k] = post_spike
            rec["is_refractory"][k]  = is_refractory
            k += 1

    for kk in rec:
        rec[kk] = rec[kk][:k]

    rec["pre_spike_times"]  = np.array(pre_spike_times)
    rec["post_spike_times"] = np.array(post_spike_times)
    return rec


def plot_all_in_one_figure(rec, p):
    PANELS = {
        "raster": True,
        "voltage": True,
        "refractory": False,
        "synapse": True,
        "stdp_traces": True,
        "eligibility": True,
        "rates": True,
        "reward": True,
        "weight": True,
        "binned_spikes": False,
    }

    t = rec["t"]
    active = [k for k, v in PANELS.items() if v]
    if not active:
        raise ValueError("No panels enabled.")

    fig, axs = plt.subplots(len(active), 1, figsize=(19, 10), sharex=True)
    if len(active) == 1:
        axs = [axs]

    i = 0

    if PANELS["raster"]:
        axs[i].eventplot([rec["pre_spike_times"], rec["post_spike_times"]], lineoffsets=[1, 0], linelengths=0.8)
        axs[i].set_yticks([0, 1])
        axs[i].set_yticklabels(["post", "pre"])
        axs[i].set_title("Spike times")
        i += 1

    if PANELS["voltage"]:
        axs[i].plot(t, rec["V"])
        axs[i].axhline(p.theta, linestyle="--", label="θ")
        axs[i].axhline(p.V_reset, linestyle=":", label="V_reset")
        axs[i].set_ylabel("mV")
        axs[i].set_title("Membrane potential V(t)")
        axs[i].legend(loc="upper right")
        i += 1

    if PANELS["refractory"]:
        axs[i].plot(t, rec["is_refractory"])
        axs[i].set_title("Refractory state")
        i += 1

    if PANELS["synapse"]:
        axs[i].plot(t, rec["I_s"])  # eq:ref_Is
        axs[i].set_title("Synaptic current I_s(t)")
        i += 1

    if PANELS["stdp_traces"]:
        axs[i].plot(t, rec["x_pre"],  label="x_pre")
        axs[i].plot(t, rec["y_post"], label="y_post")
        axs[i].legend(loc="upper right")
        axs[i].set_title("STDP traces  (eq:ref_xpre, eq:ref_ypost)")
        i += 1

    if PANELS["eligibility"]:
        axs[i].plot(t, rec["E"])
        axs[i].set_title("Eligibility trace E(t)")
        i += 1

    if PANELS["rates"]:
        axs[i].plot(t, rec["r_pre"], label="r_pre")
        axs[i].plot(t, rec["r_post"], label="r_post")
        axs[i].plot(t, p.alpha * rec["r_pre"], linestyle="--", label="target (α·r_pre)")
        axs[i].legend(loc="upper right")
        axs[i].set_ylabel("Hz")
        axs[i].set_title("Firing rates (exponential filter, τ_r)")
        i += 1

    if PANELS["reward"]:
        axs[i].plot(t, rec["R"],     label="R  (instantaneous)")
        axs[i].plot(t, rec["R_bar"], label="R̄  (eq:ref_Rbar)")
        axs[i].plot(t, rec["M"],     label="M = R − R̄  (eq:ref_w)")
        axs[i].legend(loc="upper right")
        axs[i].set_title("Reward, baseline, neuromodulator")
        i += 1

    if PANELS["weight"]:
        axs[i].plot(t, rec["w"])
        axs[i].set_title("Synaptic weight w(t)")
        i += 1

    if PANELS["binned_spikes"]:
        axs[i].plot(t, rec["pre_spike_bin"], label="pre")
        axs[i].plot(t, rec["post_spike_bin"], label="post")
        axs[i].legend(loc="upper right")
        axs[i].set_title("Binned spikes")
        i += 1

    axs[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig("simulation.png")
    plt.show()


if __name__ == "__main__":
    p = Params()

    rec = simulate(p)
    plot_all_in_one_figure(rec, p)