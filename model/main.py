import os
import numpy as np
import matplotlib.pyplot as plt

save_path = os.path.join(os.path.dirname(__file__), '../data/interim/simulation_results.png')

# --- 1. Simulation Parameters ---
dt = 0.1          # Time step (ms)
T_total = 2000.0  # Total duration (ms)
time = np.arange(0, T_total, dt)
n_steps = len(time)

# Neuron Physics: Parameters for Eq \ref{eq:lif_voltage}
tau_m = 20.0      # Membrane time constant $\tau_m$ (ms)
E_L = -70.0       # Resting potential $E_L$ (mV)
V_reset = -70.0   # Reset potential $V_{reset}$ (mV)
V_thresh = -54.0  # Threshold $\vartheta$ (mV)
R_m = 50.0        # Membrane resistance $R_m$ (MOhm)
tau_ref = 5.0     # Refractory period $\tau_{ref}$ (ms)

# Synaptic Dynamics: Parameters for Eq \ref{eq:synaptic_current}
tau_s = 5.0       # Synaptic time constant $\tau_s$ (ms)

# Plasticity / STDP Parameters (Local Factors)
tau_plus = 20.0   # Potentiation window $\tau_+$ (ms)
tau_minus = 20.0  # Depression window $\tau_-$ (ms)
tau_e = 100.0     # Eligibility trace decay $\tau_e$ (ms)

# Weight Constraints (Sec 2.4)
w_max = 1.0       # Max weight $w_{\text{max}}$
w_init = 0.5      # Initial weight $w_{init}$
eta_plus = 0.05   # Learning rate $\eta_+$ (LTP)
eta_minus = 0.05  # Learning rate $\eta_-$ (LTD)

# --- 2. Input Generation ---
# Generate Presynaptic Spikes (Poisson process)
np.random.seed(0)
pre_firing_rate = 0.1 # Probability of spike per ms
pre_spikes = np.random.rand(n_steps) < (pre_firing_rate * dt)

# Generate a synthetic "Reward" signal M(t)
# We simulate a reward that pulses periodically to demonstrate modulation
reward_signal = np.zeros(n_steps)
# Create a reward pulse every 500ms
for i in range(0, n_steps, int(500/dt)):
    reward_signal[i:i+int(100/dt)] = 0.1 
baseline_reward = 0.01 # Baseline expectation $\bar{R}$
M = reward_signal - baseline_reward # Neuromodulation: $M(t) = R(t) - \bar{R}(t)$

# --- 3. Initialization ---
v_post = np.ones(n_steps) * E_L
i_syn = np.zeros(n_steps)     # Synaptic current $I_{syn}$
w = np.zeros(n_steps)         # Synaptic weight $w_{ij}$
w[0] = w_init

# Plasticity Traces
x_trace = np.zeros(n_steps)   # Pre-synaptic trace $x_j(t)$
y_trace = np.zeros(n_steps)   # Post-synaptic trace $y_i(t)$
e_trace = np.zeros(n_steps)   # Eligibility trace $E_{ij}(t)$ (Eq \ref{eq:eligibility})
post_spikes = np.zeros(n_steps, dtype=bool)

# Refractory counter
ref_counter = 0

# --- 4. Simulation Loop ---
for t in range(1, n_steps):
    
    # A. Update Local Traces (Decay)
    # Euler integration: $\tau \frac{dx}{dt} = -x$
    x_trace[t] = x_trace[t-1] + dt * (-x_trace[t-1] / tau_plus)
    y_trace[t] = y_trace[t-1] + dt * (-y_trace[t-1] / tau_minus)
    
    # Check for Presynaptic Spike Arrival ($t_j^k$)
    if pre_spikes[t-1]: 
        x_trace[t] += 1.0 # Increment pre-trace: $x_j \to x_j + 1$
        # Synaptic Current Kick (Discrete approx of Eq \ref{eq:synaptic_current})
        # Weight $w_{ij}$ applied at moment of spike arrival
        i_syn[t-1] += w[t-1] 

    # B. Synaptic Current Decay
    # Dynamics: $\frac{dI_{syn}}{dt} = -\frac{I_{syn}}{\tau_s}$
    i_syn[t] = i_syn[t-1] + dt * (-i_syn[t-1] / tau_s)

    # C. LIF Voltage Dynamics (Eq \ref{eq:lif_voltage})
    if ref_counter > 0:
        v_post[t] = V_reset
        ref_counter -= 1
    else:
        # $\tau_m \frac{dV}{dt} = -(V - E_L) + R_m I_{syn}$
        dv = (-(v_post[t-1] - E_L) + R_m * i_syn[t]) / tau_m
        v_post[t] = v_post[t-1] + dt * dv
        
        # Check Spike Threshold (Eq \ref{eq:threshold})
        if v_post[t] >= V_thresh:
            v_post[t] = 0 # Visual spike marker (not part of physics)
            post_spikes[t] = True
            ref_counter = int(tau_ref / dt)
            y_trace[t] += 1.0 # Increment post-trace: $y_i \to y_i + 1$

    # D. Eligibility Trace Calculation (Eq \ref{eq:eligibility})
    # $S_{ij} = A_+ x_j \rho_i - A_- y_i \rho_j$
    
    # Scaling Functions (Sec 2.4)
    # $A_+(w) = \eta_+ (w_{max} - w)$
    # $A_-(w) = \eta_- w$
    A_plus = eta_plus * (w_max - w[t-1])
    A_minus = eta_minus * w[t-1]
    
    S_ij = 0
    # LTP: Pre-trace ($x_j$) exists + Post-spike ($\rho_i$) now
    if post_spikes[t]:
        S_ij += A_plus * x_trace[t]
        
    # LTD: Post-trace ($y_i$) exists + Pre-spike ($\rho_j$) now
    if pre_spikes[t]:
        S_ij -= A_minus * y_trace[t]
        
    # Update Eligibility Trace (Eq \ref{eq:eligibility})
    # $\tau_e \frac{dE_{ij}}{dt} = -E_{ij} + S_{ij}$
    de = (-e_trace[t-1] / tau_e) + (S_ij / dt) 
    e_trace[t] = e_trace[t-1] + dt * de

    # E. Weight Update
    # Dynamics: $\frac{dw_{ij}}{dt} = M(t) E_{ij}(t)$
    dw = M[t] * e_trace[t]
    w[t] = w[t-1] + dt * dw
    
    # Clip Weight to $[0, w_{max}]$
    w[t] = np.clip(w[t], 0, w_max)

# --- 5. Visualization ---
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Plot 1: Neural Activity
axs[0].set_title('Neural Activity (Pre & Post)')
axs[0].plot(time, v_post, label=r'Post $V_m$ (mV)', color='blue', linewidth=1)
axs[0].axhline(V_thresh, color='gray', linestyle='--', label='Threshold $\\vartheta$')
# Plot spikes as dots
pre_spike_times = time[pre_spikes]
axs[0].scatter(pre_spike_times, np.ones_like(pre_spike_times)*-70, 
               color='red', marker='|', s=50, label='Pre Spikes $\\rho_j$')
axs[0].legend(loc='upper right')
axs[0].set_ylabel('Voltage (mV)')
axs[0].set_ylim(-71, -50)

# Plot 2: Traces
axs[1].set_title('Synaptic Traces')
axs[1].plot(time, x_trace, label=r'Pre Trace $x_j(t)$', color='red', alpha=0.6)
axs[1].plot(time, y_trace, label=r'Post Trace $y_i(t)$', color='blue', alpha=0.6)
axs[1].legend(loc='upper right')
axs[1].set_ylabel('Trace Magnitude')

# Plot 3: Eligibility & Reward
axs[2].set_title('Eligibility Trace & Reward Signal')
axs[2].plot(time, e_trace, label=r'Eligibility $E_{ij}(t)$', color='purple')
axs[2].plot(time, M, label=r'Neuromodulation $M(t)$', color='green', linestyle='--', alpha=0.5)
axs[2].legend(loc='upper right')
axs[2].set_ylabel('Magnitude')

# Plot 4: Synaptic Weight
axs[3].set_title('Synaptic Weight Evolution')
axs[3].plot(time, w, label=r'Weight $w_{ij}$', color='black', linewidth=2)
axs[3].set_ylim(0, w_max)
axs[3].set_ylabel('Weight')
axs[3].set_xlabel('Time (ms)')
axs[3].legend()

plt.tight_layout()
# plt.show()
plt.savefig(save_path)