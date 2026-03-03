from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Params:
    T: float = 20.0
    dt: float = 1e-4
    seed: int = 0

    r_pre_rate: float = 20.0

    tau_m: float = 0.02
    E_L: float = -65.0
    V_reset: float = -70.0
    theta: float = -50.0
    tau_ref: float = 0.003

    tau_s: float = 0.005

    R_m: float = 50.0

    tau_plus: float = 0.02
    tau_minus: float = 0.02

    tau_r: float = 0.5

    tau_e: float = 0.5

    tau_Rbar: float = 5.0

    w0: float = 2.0
    wmax: float = 10.0
    eta_plus: float = 1e-4
    eta_minus: float = 1e-4

    kappa_w: float = 1.0

    record_every: float = 0.001


def simulate(p: Params):
    rng = np.random.default_rng(p.seed)
    n = int(p.T / p.dt)

    V = p.E_L
    ref_remaining = 0.0

    I_s = 0.0
    x = 0.0
    y = 0.0
    r_pre = 0.0
    r_post = 0.0
    E = 0.0
    w = p.w0
    Rbar = 0.0

    rec_step = max(1, int(p.record_every / p.dt))
    m = n // rec_step + 2

    rec = {k: np.zeros(m) for k in [
        "t", "V", "w", "I_s", "I_syn", "x", "y", "E",
        "r_pre", "r_post", "R", "Rbar", "M",
        "pre_spike_bin", "post_spike_bin", "is_refractory"
    ]}

    pre_spike_times = []
    post_spike_times = []
    k = 0

    for step in range(n):
        t = step * p.dt

        pre_spike = 1 if (rng.random() < p.r_pre_rate * p.dt) else 0
        if pre_spike:
            pre_spike_times.append(t)

        if pre_spike:
            I_s += 1.0
            x += 1.0
            r_pre += 1.0
            E -= p.eta_minus * w * y

        I_s += p.dt * (-I_s / p.tau_s)
        x += p.dt * (-x / p.tau_plus)
        r_pre += p.dt * (-r_pre / p.tau_r)

        I_syn = p.R_m * w * I_s

        post_spike = 0
        is_refractory = 1 if ref_remaining > 0.0 else 0

        if ref_remaining <= 0.0:
            dV = (p.dt / p.tau_m) * (-(V - p.E_L) + I_syn)
            V_new = V + dV

            if V < p.theta and V_new >= p.theta:
                post_spike = 1
                post_spike_times.append(t)
                V = p.V_reset
                ref_remaining = p.tau_ref
            else:
                V = V_new
        else:
            ref_remaining -= p.dt
            V = p.V_reset

        if post_spike:
            y += 1.0
            r_post += 1.0
            E += p.eta_plus * (p.wmax - w) * x

        y += p.dt * (-y / p.tau_minus)
        r_post += p.dt * (-r_post / p.tau_r)
        E += p.dt * (-E / p.tau_e)

        R = -(r_post - 0.5 * r_pre) ** 2
        Rbar += (p.dt / p.tau_Rbar) * (-Rbar + R)
        M = R - Rbar

        w += p.kappa_w * p.dt * M * E
        w = min(p.wmax, max(0.0, w))

        if step % rec_step == 0:
            rec["t"][k] = t
            rec["V"][k] = V
            rec["w"][k] = w
            rec["I_s"][k] = I_s
            rec["I_syn"][k] = I_syn
            rec["x"][k] = x
            rec["y"][k] = y
            rec["E"][k] = E
            rec["r_pre"][k] = r_pre
            rec["r_post"][k] = r_post
            rec["R"][k] = R
            rec["Rbar"][k] = Rbar
            rec["M"][k] = M
            rec["pre_spike_bin"][k] = pre_spike
            rec["post_spike_bin"][k] = post_spike
            rec["is_refractory"][k] = is_refractory
            k += 1

    for kk in rec:
        rec[kk] = rec[kk][:k]

    rec["pre_spike_times"] = np.array(pre_spike_times)
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
        axs[i].plot(t, rec["I_s"], label="I_s(t)")
        axs[i].plot(t, rec["I_syn"], label="I_syn(t) = w·I_s")
        axs[i].legend(loc="upper right")
        axs[i].set_title("Synaptic dynamics")
        i += 1

    if PANELS["stdp_traces"]:
        axs[i].plot(t, rec["x"], label="x (pre)")
        axs[i].plot(t, rec["y"], label="y (post)")
        axs[i].legend(loc="upper right")
        axs[i].set_title("STDP traces")
        i += 1

    if PANELS["eligibility"]:
        axs[i].plot(t, rec["E"])
        axs[i].set_title("Eligibility trace E(t)")
        i += 1

    if PANELS["rates"]:
        axs[i].plot(t, rec["r_pre"], label="r_pre")
        axs[i].plot(t, rec["r_post"], label="r_post")
        axs[i].plot(t, 0.5 * rec["r_pre"], linestyle="--", label="target (½ r_pre)")
        axs[i].legend(loc="upper right")
        axs[i].set_ylabel("Hz")
        axs[i].set_title("Firing rates (exponential filter, τ_r)")
        i += 1

    if PANELS["reward"]:
        axs[i].plot(t, rec["R"], label="R")
        axs[i].plot(t, rec["Rbar"], label="R̄")
        axs[i].plot(t, rec["M"], label="M = R − R̄")
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
    p = Params(
        T=20.0,
        dt=1e-4,
        record_every=1e-4,
        seed=1,

        r_pre_rate=20.0,

        tau_m=0.02,
        E_L=-65.0,
        V_reset=-70.0,
        theta=-50.0,
        tau_ref=0.003,

        tau_s=0.005,

        R_m=50.0,

        tau_plus=0.02,
        tau_minus=0.02,

        tau_r=0.5,
        tau_e=0.5,

        tau_Rbar=5.0,

        w0=2.0,
        wmax=10.0,
        eta_plus=1e-4,
        eta_minus=1e-4,
        kappa_w=1.0,
    )

    rec = simulate(p)
    plot_all_in_one_figure(rec, p)