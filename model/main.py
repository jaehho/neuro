"""
LIF neuron with Goal-Directed Reward-Modulated STDP.

CORRECTION APPLIED:
The Eligibility Trace is now strictly Hebbian (positive).
- If M is positive (Reward) and trace is high -> Weight UP.
- If M is negative (Punishment) and trace is high -> Weight DOWN.
This fixes the 'double negative' instability where high firing rates
caused negative traces, inverting the punishment into reinforcement.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

save_path = "simulation_results_goal_corrected.png"

# Parameters
dt = 0.1          # ms
T_total = 10000.0  # ms
time = np.arange(0, T_total, dt)
n_steps = len(time)

# LIF parameters
tau_m = 20.0
E_L = -70.0
V_reset = -70.0
V_thresh = -54.0
R_m = 50.0
tau_ref = 5.0

# Synapse
tau_s = 5.0

# Plasticity traces (ms)
tau_plus = 20.0
tau_minus = 20.0
tau_e = 100.0

# Rate Estimation Parameters
tau_rate = 500.0

# Weight rule
w_max = 5.0
w_init = 2.0  
eta_plus = 0.05
eta_minus = 0.05 

# Inputs
np.random.seed(42)
pre_firing_rate = 0.045  # ~45 Hz
pre_spikes = np.random.rand(n_steps) < (pre_firing_rate * dt)

# Arrays
v_post = np.ones(n_steps) * E_L
i_syn = np.zeros(n_steps)
w = np.zeros(n_steps)
w[0] = w_init

x_trace = np.zeros(n_steps)
y_trace = np.zeros(n_steps) # Kept for completeness, though less critical for Hebbian-only
e_trace = np.zeros(n_steps)
M = np.zeros(n_steps)
post_spikes = np.zeros(n_steps, dtype=bool)

# Visualization arrays
r_pre_hist = np.zeros(n_steps)
r_post_hist = np.zeros(n_steps)

# State
ref_counter = 0
r_pre_smooth = 0.0
r_post_smooth = 0.0

# Simulation
for t in range(1, n_steps):

    # 1. Update Firing Rate Estimates
    r_pre_smooth += dt * (-r_pre_smooth / tau_rate)
    r_post_smooth += dt * (-r_post_smooth / tau_rate)
    
    if pre_spikes[t-1]:
        r_pre_smooth += 1.0 / tau_rate

    # 2. Calculate Goal-Directed Reward
    target_rate = 0.5 * r_pre_smooth
    rate_error = abs(r_post_smooth - target_rate)
    
    # Gaussian Reward Profile
    sigma = 0.005 
    reward_val = 2.0 * np.exp(-(rate_error**2) / (2 * sigma**2)) - 1.0
    M[t] = reward_val

    # 3. LIF Dynamics
    x_trace[t] = x_trace[t - 1] + dt * (-x_trace[t - 1] / tau_plus)
    y_trace[t] = y_trace[t - 1] + dt * (-y_trace[t - 1] / tau_minus)

    if pre_spikes[t - 1]:
        x_trace[t] += 1.0
        i_syn[t - 1] += w[t - 1]

    i_syn[t] = i_syn[t - 1] + dt * (-i_syn[t - 1] / tau_s)

    if ref_counter > 0:
        v_post[t] = V_reset
        ref_counter -= 1
    else:
        dv = (-(v_post[t - 1] - E_L) + R_m * i_syn[t]) / tau_m
        v_post[t] = v_post[t - 1] + dt * dv

        if v_post[t] >= V_thresh:
            v_post[t] = 0.0
            post_spikes[t] = True
            ref_counter = int(tau_ref / dt)
            y_trace[t] += 1.0
            r_post_smooth += 1.0 / tau_rate

    r_pre_hist[t] = r_pre_smooth
    r_post_hist[t] = r_post_smooth

    # 4. Weight Update (CORRECTED)
    # ----------------------------------------
    A_plus = eta_plus * (w_max - w[t - 1])
    
    # We remove the A_minus term from the eligibility trace. 
    # The eligibility trace should effectively be |Correlation|.
    # The Reward Sign (+/-) handles the Direction (LTP/LTD).
    
    S_ij = 0.0
    if post_spikes[t]:
        S_ij += A_plus * x_trace[t]  # Hebbian coincidence only
    
    # NOTE: The "Depression" part (Pre-after-Post) is removed from eligibility.
    # This ensures e_trace stays positive.
    # if pre_spikes[t]:
    #    S_ij -= A_minus * y_trace[t]

    de = (-e_trace[t - 1] / tau_e) + (S_ij / dt)
    e_trace[t] = e_trace[t - 1] + dt * de

    # Update weight
    w[t] = w[t - 1] + dt * (M[t] * e_trace[t])
    w[t] = np.clip(w[t], 0.0, w_max)

# Plotting
fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

axs[0].set_title("Neural Activity")
axs[0].plot(time, v_post, label="Post Vm", linewidth=0.5, color="blue")
axs[0].axhline(V_thresh, linestyle="--", color="gray", alpha=0.5)
pre_spike_times = time[pre_spikes]
axs[0].scatter(
    pre_spike_times,
    np.ones_like(pre_spike_times) * E_L - 1,
    marker="|",
    s=50,
    label=r"Pre spikes $\rho_j$",
    color="red",
)
axs[0].set_ylabel("mV")

axs[0].legend()

axs[1].set_title("Goal Tracking: Firing Rates")
axs[1].plot(time, r_post_hist, label="Actual Post Rate", color="blue", linewidth=2)
axs[1].plot(time, r_pre_hist * 0.5, label="Target (0.5 * Pre Rate)", color="green", linestyle="--", linewidth=2)
axs[1].set_ylabel("Rate")
axs[1].legend()
axs[1].grid(True, alpha=0.3)

axs[2].set_title("Reward Signal (M)")
axs[2].plot(time, M, color="purple", linewidth=1)
axs[2].fill_between(time, M, 0, where=(M>0), color='green', alpha=0.3, label="Reward")
axs[2].fill_between(time, M, 0, where=(M<0), color='red', alpha=0.3, label="Punishment")
axs[2].set_ylabel("Amplitude")
axs[2].legend()

axs[3].set_title("Eligibility Trace (Positive Only)")
axs[3].plot(time, e_trace, color="orange", linewidth=1)
axs[3].set_ylabel("Magnitude")

axs[4].set_title("Synaptic Weight Evolution")
axs[4].plot(time, w, color="black", linewidth=2)
axs[4].set_ylim(0, w_max)
axs[4].set_ylabel("Weight")
axs[4].set_xlabel("Time (ms)")

plt.tight_layout()
plt.savefig(save_path)
plt.show()