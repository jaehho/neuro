import numpy as np
import matplotlib.pyplot as plt
from fitted_rates import alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n
from HH_constants import c_m, g_Na, g_K, g_L, E_Na, E_K, E_L

# External current (uA/cm^2)
def I_ext(t):
    return 15.0 if (10.0 <= t <= 40.0) else 0.0

t_max, dt = 50.0, 0.001
t = np.arange(0.0, t_max + 1e-12, dt)
nt = len(t)

V0 = np.array(-65.0)
m0 = float(alpha_m(V0)/(alpha_m(V0)+beta_m(V0)))
h0 = float(alpha_h(V0)/(alpha_h(V0)+beta_h(V0)))
n0 = float(alpha_n(V0)/(alpha_n(V0)+beta_n(V0)))

V = np.empty(nt); m = np.empty(nt); h = np.empty(nt); n = np.empty(nt)
V[0], m[0], h[0], n[0] = V0, m0, h0, n0

for k in range(nt-1):
    Vk, mk, hk, nk = V[k], m[k], h[k], n[k]
    gNa_t = g_Na * (mk**3) * hk
    gK_t  = g_K  * (nk**4)
    gL_t  = g_L
    INa = gNa_t * (Vk - E_Na)
    IK  = gK_t  * (Vk - E_K)
    IL  = gL_t  * (Vk - E_L)
    dVdt = (I_ext(t[k]) - (INa + IK + IL)) / c_m

    am, bm = alpha_m(Vk), beta_m(Vk)
    ah, bh = alpha_h(Vk), beta_h(Vk)
    an, bn = alpha_n(Vk), beta_n(Vk)
    dmdt = am*(1.0 - mk) - bm*mk
    dhdt = ah*(1.0 - hk) - bh*hk
    dndt = an*(1.0 - nk) - bn*nk

    V[k+1] = Vk + dt*dVdt
    m[k+1] = np.clip(mk + dt*dmdt, 0.0, 1.0)
    h[k+1] = np.clip(hk + dt*dhdt, 0.0, 1.0)
    n[k+1] = np.clip(nk + dt*dndt, 0.0, 1.0)

# Calculate steady-state values and time constants over time
am_t, bm_t = alpha_m(V), beta_m(V)
ah_t, bh_t = alpha_h(V), beta_h(V)
an_t, bn_t = alpha_n(V), beta_n(V)

# Steady-state values x_inf(V(t)) and taus tau(V(t))
m_inf_t = am_t / (am_t + bm_t)
h_inf_t = ah_t / (ah_t + bh_t)
n_inf_t = an_t / (an_t + bn_t)

tau_m_t = 1.0 / (am_t + bm_t)
tau_h_t = 1.0 / (ah_t + bh_t)
tau_n_t = 1.0 / (an_t + bn_t)

# Compute time-series conductances and weights
gNa_t = g_Na * (m**3) * h
gK_t  = g_K  * (n**4)
gL_t  = g_L  * np.ones_like(t)

INa_t = gNa_t * (V - E_Na)
IK_t  = gK_t  * (V - E_K)
IL_t  = gL_t  * (V - E_L)

w_Na  = -INa_t / c_m
w_K   = -IK_t  / c_m
w_L   = -IL_t  / c_m
w_ext = np.array([I_ext(tt) for tt in t]) / c_m
w_sum = w_ext + w_Na + w_K + w_L  # should equal dV/dt


# %% Plotting Hodgekin-Huxley Voltage and Gating Variables
fig = plt.figure(figsize=(8, 6))

# 3D plot
ax = fig.add_subplot(221, projection='3d', proj_type='ortho')
ax.plot(V, t, m, label='m gate', color='b')
ax.plot(V, t, h, label='h gate', color='g')
ax.plot(V, t, n, label='n gate', color='r')
ax.set_xlabel('Voltage (mV)')
ax.set_ylabel('Time (ms)')
ax.set_zlabel('Probability')

# Orthographic projections
ax2 = fig.add_subplot(222)
ax2.plot(t, m, label='m gate', color='b')
ax2.plot(t, h, label='h gate', color='g')
ax2.plot(t, n, label='n gate', color='r')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Probability')
ax2.legend()

ax3 = fig.add_subplot(223)
ax3.plot(t, V, color='k')
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Voltage (mV)')
ax3.set_ylim(-80, 60)

ax4 = fig.add_subplot(224)
ax4.plot(V, m, label='m gate', color='b')
ax4.plot(V, h, label='h gate', color='g')
ax4.plot(V, n, label='n gate', color='r')
ax4.set_xlabel('Voltage (mV)')
ax4.set_ylabel('Probability')
ax4.legend()

plt.tight_layout()
plt.show(block=False)

# %% Voltage (top) and per-channel dV/dt weights (bottom)
fig, (axV, axW) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

# set xticks every 2 ms
xticks = np.arange(0, t_max+1, 1)

# Row 1: V(t) with reversal lines
axV.plot(t, V, color='k', linewidth=2, label='V(t)')
axV.axhline(E_Na, color='b', linewidth=1)
axV.axhline(E_K,  color='r', linewidth=1)
axV.axhline(E_L,  color='g', linewidth=1)
axV.text(t_max*0.82, E_Na+2, '$E_{Na}$', color='b')
axV.text(t_max*0.82, E_K+2,  '$E_{K}$',  color='r')
axV.text(t_max*0.82, E_L+2,  '$E_{L}$',  color='g')
axV.set_ylabel('V (mV)')
axV.set_title('Voltage and per-channel weights to dV/dt')
axV.legend(loc='best')
axV.grid(which='both', axis='x')
axV.set_xticks(xticks)

# Row 2: Weights of each channel to dV/dt
axW.axhline(0, color='gray', linewidth=0.5)
axW.plot(t, w_ext, color='k', linestyle=':', label=r'$I_{\mathrm{ext}}/c_m$')
axW.plot(t, w_Na,  color='b', linestyle='--', label=r'$-\;\overline{g}_{Na} m^{3}h(V-E_{Na})/c_m$')
axW.plot(t, w_K,   color='r', linestyle='--', label=r'$-\;\overline{g}_{K} n^{4}(V-E_{K})/c_m$')
axW.plot(t, w_L,   color='g', linestyle='--', label=r'$-\;\overline{g}_{L}(V-E_{L})/c_m$')

# show sum equals numerical dV/dt
axW.plot(t, np.gradient(V, dt), color='gray', alpha=0.6, label='dV/dt (num)')
axW.plot(t, w_sum, color='purple', linestyle=':', alpha=0.6, label='sum of weights')

axW.set_xlabel('Time (ms)')
axW.set_ylabel('Weight to dV/dt (mV/ms)')
axW.legend(loc='best')
axW.grid(which='both', axis='x')
axW.set_xticks(xticks)

plt.tight_layout()
plt.show(block=False)

# %% Comparing Activation Variables and Conductances
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left y-axis: activation values
axes[0].plot(t, m**3 * h, color='b', label=r'$m^{3}h \; (\text{Na\ activation})$')
axes[0].plot(t, n**4, color='r', label=r'$n^{4} \; (\text{K\ activation})$')
axes[0].set_ylabel('Activation (0–1)', color='k')
axes[0].set_xlabel('Time (ms)')
axes[0].legend(loc='upper right')
axes[0].set_title('Activation Variables')


# Right y-axis: activation values x max conductance
axes[1].plot(t, g_Na * (m**3) * h, color='b', label=r'$\overline{g}_{Na} m^{3}h$')
axes[1].plot(t, g_K * (n**4), color='r', label=r'$\overline{g}_{K} n^{4}$')
axes[1].plot(t, g_L * np.ones_like(t), color='g', label=r'$\overline{g}_{L}$')
axes[1].set_ylabel('Conductance (mS)', color='k')
axes[1].set_xlabel('Time (ms)')
axes[1].legend(loc='upper right')
axes[1].set_title('Conductances')

plt.suptitle('Activation Variables vs Conductances')

plt.show(block=False)

# %% gating variables vs steady state (row 1), time constants (row 2)
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

# Row 1: gating variables and steady states
axes[0, 0].plot(t, m, 'b', label='m(t)')
axes[0, 0].plot(t, m_inf_t, 'b--', label=r'$m_\infty(V)$')
axes[0, 0].set_ylabel('m')
axes[0, 0].set_title('m gate')
axes[0, 0].legend(loc='best')

axes[0, 1].plot(t, h, 'g', label='h(t)')
axes[0, 1].plot(t, h_inf_t, 'g--', label=r'$h_\infty(V)$')
axes[0, 1].set_title('h gate')
axes[0, 1].legend(loc='best')

axes[0, 2].plot(t, n, 'r', label='n(t)')
axes[0, 2].plot(t, n_inf_t, 'r--', label=r'$n_\infty(V)$')
axes[0, 2].set_title('n gate')
axes[0, 2].legend(loc='best')

# Row 2: time constants
axes[1, 0].plot(t, tau_m_t, 'b', label=r'$\tau_m$')
axes[1, 0].set_xlabel('Time (ms)')
axes[1, 0].set_ylabel('τ (ms)')
axes[1, 0].legend(loc='best')

axes[1, 1].plot(t, tau_h_t, 'g', label=r'$\tau_h$')
axes[1, 1].set_xlabel('Time (ms)')
axes[1, 1].legend(loc='best')

axes[1, 2].plot(t, tau_n_t, 'r', label=r'$\tau_n$')
axes[1, 2].set_xlabel('Time (ms)')
axes[1, 2].legend(loc='best')

plt.suptitle('Gating variables, steady states, and time constants over time')
plt.tight_layout()
plt.show()
