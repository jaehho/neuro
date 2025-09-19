import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fitted_rates import alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

# Voltage and time ranges
V_min, V_max, nV = -100.0, 50.0, 151   # mV

V = np.linspace(V_min, V_max, nV)

# Plot steady-state values and time constants for all gating variables
gates = [
    ('m', alpha_m, beta_m),
    ('h', alpha_h, beta_h),
    ('n', alpha_n, beta_n)
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colors = ['b', 'g', 'r']
for i, (name, alpha_func, beta_func) in enumerate(gates):
    tau = 1.0 / (alpha_func(V) + beta_func(V))
    x_inf = alpha_func(V) / (alpha_func(V) + beta_func(V))
    axes[0].plot(V, x_inf, color=colors[i], label=f'{name}_inf')
    axes[1].plot(V, tau, color=colors[i], label=f'tau_{name}')
axes[0].set_title('Steady-state values (x_inf)')
axes[0].set_xlabel('Voltage (mV)')
axes[0].set_ylabel('x_inf')
axes[0].legend()
axes[1].set_title('Time constants (tau)')
axes[1].set_xlabel('Voltage (mV)')
axes[1].set_ylabel('tau (ms)')
axes[1].legend()

plt.tight_layout()
plt.show()
