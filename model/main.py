"""
Extended simulation + plotting of (nearly) every modeled variable:

States/derived:
- V(t), refractory flag
- pre_spike, post_spike
- synaptic filter s(t), synaptic current I_syn(t)=w*s
- external current/drive I_ext(t)=I_bias + noise term (recorded as an equivalent per-step current)
- STDP traces x(t), y(t)
- induction term S(t)
- eligibility trace E(t)
- weight w(t)
- sliding-window rates r_pre(t), r_post(t), target 0.5*r_pre(t)
- reward R(t), baseline Rbar(t), neuromodulator M(t)=R-Rbar

Requires: numpy, matplotlib
"""

from dataclasses import dataclass
import numpy as np
import math
import matplotlib.pyplot as plt


@dataclass
class Params:
    # Time
    T: float = 20.0           # seconds
    dt: float = 1e-4          # seconds (0.1 ms)
    seed: int = 0

    # Presynaptic spikes (Poisson)
    r_pre: float = 20.0       # Hz

    # LIF neuron
    tau_m: float = 0.02       # s
    E_L: float = -65.0        # mV
    V_reset: float = -70.0    # mV
    theta: float = -50.0      # mV
    tau_ref: float = 0.003    # s

    # Synapse (exponential kernel)
    tau_s: float = 0.005      # s

    # External drive/noise (Euler–Maruyama)
    I_bias: float = 15.0      # "mV-equivalent" constant drive
    sigma_V: float = 1.5      # mV / sqrt(s)

    # STDP traces
    tau_plus: float = 0.02    # s
    tau_minus: float = 0.02   # s

    # Eligibility trace
    tau_e: float = 0.5        # s

    # Reward window + baseline
    DeltaT: float = 0.5       # s (sliding window)
    tau_Rbar: float = 5.0     # s

    # Weight
    w0: float = 2.0
    wmax: float = 10.0
    eta_plus: float = 1e-4
    eta_minus: float = 1e-4

    # Optional extra scaling for dw/dt (useful for tuning)
    kappa_w: float = 1.0

    # Trace normalization:
    # True  -> traces jump by 1 on spike (common discrete implementation)
    # False -> more literal tau dz/dt=-z+rho with rho≈spike/dt; jump amplitude differs by 1/tau
    unit_jump_traces: bool = True

    # Recording
    record_every: float = 0.001  # s (downsample, e.g. 1 ms)


def simulate(p: Params):
    rng = np.random.default_rng(p.seed)
    n = int(p.T / p.dt)

    # State variables
    V = p.E_L
    ref_remaining = 0.0
    s = 0.0
    x = 0.0
    y = 0.0
    E = 0.0
    w = p.w0
    Rbar = 0.0

    # Sliding-window rates via ring buffers
    win = max(1, int(p.DeltaT / p.dt))
    pre_buf = np.zeros(win, dtype=np.int8)
    post_buf = np.zeros(win, dtype=np.int8)
    pre_sum = 0
    post_sum = 0
    buf_idx = 0

    # Downsampled recordings
    rec_step = max(1, int(p.record_every / p.dt))
    m = n // rec_step + 2

    rec = {k: np.zeros(m) for k in [
        "t", "V", "w", "s", "I_syn", "I_ext_equiv", "x", "y", "S", "E",
        "r_pre", "r_post", "R", "Rbar", "M", "Aplus", "Aminus",
        "pre_spike_bin", "post_spike_bin", "is_refractory"
    ]}

    pre_spike_times = []
    post_spike_times = []
    k = 0

    def trace_update(z, tau, spike):
        if p.unit_jump_traces:
            return z + p.dt * (-z / tau) + spike
        else:
            return z + p.dt * (-z / tau) + spike * (1.0 / tau)

    for step in range(n):
        t = step * p.dt

        # Presynaptic Poisson spike
        pre_spike = 1 if (rng.random() < p.r_pre * p.dt) else 0
        if pre_spike:
            pre_spike_times.append(t)

        # Synapse filter and syn current
        s = trace_update(s, p.tau_s, pre_spike)
        I_syn = w * s

        # External noise term as a "current-equivalent" increment:
        # dV_sto = sigma_V * sqrt(dt) * N(0,1)
        # In the voltage ODE form: dV = (dt/tau_m)*R_m*I_ext + ...
        # Here we record the per-step effective contribution converted to an equivalent current term:
        # I_noise_equiv such that (dt/tau_m)*I_noise_equiv = dV_sto  => I_noise_equiv = dV_sto * tau_m / dt
        dV_sto = p.sigma_V * math.sqrt(p.dt) * rng.standard_normal()
        I_noise_equiv = dV_sto * p.tau_m / p.dt
        I_ext_equiv = p.I_bias + I_noise_equiv  # "mV-equivalent current" total external term

        # Postsynaptic LIF update
        post_spike = 0
        is_refractory = 1 if ref_remaining > 0.0 else 0

        if ref_remaining <= 0.0:
            dV_det = (p.dt / p.tau_m) * (-(V - p.E_L) + I_syn + p.I_bias)
            V_new = V + dV_det + dV_sto

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

        # STDP traces
        x = trace_update(x, p.tau_plus, pre_spike)
        y = trace_update(y, p.tau_minus, post_spike)

        # Weight-dependent STDP amplitudes
        Aplus = p.eta_plus * (p.wmax - w)
        Aminus = p.eta_minus * w

        # Spike trains as Dirac deltas approximated by spikes per bin: rho ≈ spike/dt
        rho_pre = pre_spike / p.dt
        rho_post = post_spike / p.dt

        # Induction term
        S = Aplus * x * rho_post - Aminus * y * rho_pre

        # Eligibility
        E += (p.dt / p.tau_e) * (-E + S)

        # Sliding-window rates
        pre_sum -= int(pre_buf[buf_idx])
        post_sum -= int(post_buf[buf_idx])
        pre_buf[buf_idx] = pre_spike
        post_buf[buf_idx] = post_spike
        pre_sum += pre_spike
        post_sum += post_spike
        buf_idx = (buf_idx + 1) % win

        r_pre_t = pre_sum / (win * p.dt)
        r_post_t = post_sum / (win * p.dt)

        # Reward + baseline + neuromodulator
        R = - (r_post_t - 0.5 * r_pre_t) ** 2
        Rbar += (p.dt / p.tau_Rbar) * (-Rbar + R)
        M = R - Rbar

        # Weight update (clamped)
        w += p.kappa_w * p.dt * M * E
        w = min(p.wmax, max(0.0, w))

        # Record
        if step % rec_step == 0:
            rec["t"][k] = t
            rec["V"][k] = V
            rec["w"][k] = w
            rec["s"][k] = s
            rec["I_syn"][k] = I_syn
            rec["I_ext_equiv"][k] = I_ext_equiv
            rec["x"][k] = x
            rec["y"][k] = y
            rec["S"][k] = S
            rec["E"][k] = E
            rec["r_pre"][k] = r_pre_t
            rec["r_post"][k] = r_post_t
            rec["R"][k] = R
            rec["Rbar"][k] = Rbar
            rec["M"][k] = M
            rec["Aplus"][k] = Aplus
            rec["Aminus"][k] = Aminus
            rec["pre_spike_bin"][k] = pre_spike
            rec["post_spike_bin"][k] = post_spike
            rec["is_refractory"][k] = is_refractory
            k += 1

    # Trim arrays
    for kk in rec:
        rec[kk] = rec[kk][:k]

    rec["pre_spike_times"] = np.array(pre_spike_times)
    rec["post_spike_times"] = np.array(post_spike_times)
    return rec

# Updated single-figure plotting:
# - STDP induction S(t) and eligibility E(t) plotted in separate panels
# - Weight-dependent STDP scaling removed (A+, A− not plotted)

def plot_all_in_one_figure(rec, p):
    import matplotlib.pyplot as plt

    # ---- Panel toggles ----
    PANELS = {
        "raster": True,
        "voltage": True,
        "refractory": False,
        "synapse": True,
        "external_current": False,
        "stdp_traces": True,
        "induction": True,        # S(t)
        "eligibility": True,      # E(t)
        "rates": True,
        "reward": True,
        "weight": True,
        "binned_spikes": False,
    }

    t = rec["t"]

    active = [k for k, v in PANELS.items() if v]
    if len(active) == 0:
        raise ValueError("No panels enabled. Set at least one entry in PANELS to True.")

    fig, axs = plt.subplots(len(active), 1, figsize=(14, 2.2 * len(active)), sharex=True)
    if len(active) == 1:
        axs = [axs]

    i = 0

    # ---- 1) Raster ----
    if PANELS["raster"]:
        axs[i].eventplot([rec["pre_spike_times"], rec["post_spike_times"]],
                         lineoffsets=[1, 0], linelengths=0.8)
        axs[i].set_yticks([0, 1])
        axs[i].set_yticklabels(["post i", "pre j"])
        axs[i].set_title("Spike times")
        i += 1

    # ---- 2) Membrane voltage ----
    if PANELS["voltage"]:
        axs[i].plot(t, rec["V"])
        axs[i].axhline(p.theta, linestyle="--")
        axs[i].axhline(p.V_reset, linestyle=":")
        axs[i].set_ylabel("mV")
        axs[i].set_title("Membrane potential V(t)")
        i += 1

    # ---- 3) Refractory indicator ----
    if PANELS["refractory"]:
        axs[i].plot(t, rec["is_refractory"])
        axs[i].set_title("Refractory state")
        i += 1

    # ---- 4) Synapse filter + syn current ----
    if PANELS["synapse"]:
        axs[i].plot(t, rec["s"], label="s(t)")
        axs[i].plot(t, rec["I_syn"], label="I_syn(t)")
        axs[i].legend(loc="upper right")
        axs[i].set_title("Synaptic dynamics")
        i += 1

    # ---- 5) External current ----
    if PANELS["external_current"]:
        axs[i].plot(t, rec["I_ext_equiv"])
        axs[i].set_title("External input I_ext(t)")
        i += 1

    # ---- 6) STDP traces ----
    if PANELS["stdp_traces"]:
        axs[i].plot(t, rec["x"], label="x (pre)")
        axs[i].plot(t, rec["y"], label="y (post)")
        axs[i].legend(loc="upper right")
        axs[i].set_title("STDP traces")
        i += 1

    # ---- 7) Induction term S(t) ----
    if PANELS["induction"]:
        axs[i].plot(t, rec["S"])
        axs[i].set_title("STDP induction term S(t)")
        i += 1

    # ---- 8) Eligibility trace E(t) ----
    if PANELS["eligibility"]:
        axs[i].plot(t, rec["E"])
        axs[i].set_title("Eligibility trace E(t)")
        i += 1

    # ---- 9) Sliding-window firing rates ----
    if PANELS["rates"]:
        axs[i].plot(t, rec["r_pre"], label="r_pre")
        axs[i].plot(t, rec["r_post"], label="r_post")
        axs[i].plot(t, 0.5 * rec["r_pre"], linestyle="--", label="target")
        axs[i].legend(loc="upper right")
        axs[i].set_ylabel("Hz")
        axs[i].set_title("Firing rates")
        i += 1

    # ---- 10) Reward dynamics ----
    if PANELS["reward"]:
        axs[i].plot(t, rec["R"], label="R")
        axs[i].plot(t, rec["Rbar"], label="Rbar")
        axs[i].plot(t, rec["M"], label="M")
        axs[i].legend(loc="upper right")
        axs[i].set_title("Reward, baseline, neuromodulator")
        i += 1

    # ---- 11) Weight ----
    if PANELS["weight"]:
        axs[i].plot(t, rec["w"])
        axs[i].set_title("Synaptic weight w(t)")
        i += 1

    # ---- 12) Binned spikes ----
    if PANELS["binned_spikes"]:
        axs[i].plot(t, rec["pre_spike_bin"], label="pre")
        axs[i].plot(t, rec["post_spike_bin"], label="post")
        axs[i].legend(loc="upper right")
        axs[i].set_title("Binned spikes")
        i += 1

    axs[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    p = Params(
        T=20.0,
        dt=1e-4,
        record_every=1e-4,
        seed=1,

        r_pre=20.0,

        tau_m=0.02,
        E_L=-65.0,
        V_reset=-70.0,
        theta=-50.0,
        tau_ref=0.003,

        tau_s=0.005,

        I_bias=15.0,
        sigma_V=1.5,

        tau_plus=0.02,
        tau_minus=0.02,

        tau_e=0.5,

        DeltaT=0.5,
        tau_Rbar=5.0,

        w0=2.0,
        wmax=10.0,
        eta_plus=1e-4,
        eta_minus=1e-4,
        kappa_w=1.0,

        unit_jump_traces=True,
    )

    rec = simulate(p)
    plot_all_in_one_figure(rec, p)
