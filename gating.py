import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fitted_rates import alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

# Choose gating variable: 'm', 'h', or 'n'
gating = 'm'

if gating == 'm':
    alpha = alpha_m
    beta  = beta_m
elif gating == 'h':
    alpha = alpha_h
    beta  = beta_h
elif gating == 'n':
    alpha = alpha_n
    beta  = beta_n

# Voltage and time ranges
V_min, V_max, nV = -100.0, 50.0, 151   # mV
t_min, t_max, nt = 0.0, 20.0, 201      # ms

V = np.linspace(V_min, V_max, nV)
t = np.linspace(t_min, t_max, nt)

VV, TT = np.meshgrid(V, t)  # both arrays have shape (nt, nV)

# Compute steady-state and time constants at each voltage
alpha_vals = alpha(V)  # (nV,)
beta_vals  = beta(V)   # (nV,)
tau = 1.0 / (alpha_vals + beta_vals)   # ms
x_inf = alpha_vals / (alpha_vals + beta_vals)

method = 'numeric'  # 'analytic' or 'numeric'

if method == 'analytic':
    # Analytic solution with voltage clamp
    V_clamp = np.array([-65.0])
    x0 = alpha(np.array([V_clamp])) / (alpha(np.array([V_clamp])) + beta(np.array([V_clamp])))
    x0 = alpha(V_clamp) / (alpha(V_clamp) + beta(V_clamp))
    x0 = x0[0]
    x0 = 1

    X = x_inf - (x_inf - x0)*np.exp(-TT / tau)
elif method == 'numeric':
    # Numeric solution with voltage clamp from Appendix 5.11 B
    V_clamp = -65.0

    # initial condition: gate is at steady-state for the pre-clamp voltage
    alpha_c = alpha(np.array([V_clamp]))
    beta_c  = beta(np.array([V_clamp]))
    x0 = float(alpha_c / (alpha_c + beta_c))   # scalar

    # allocate the surface: rows=time, cols=voltage
    X = np.empty((nt, nV), dtype=float)
    X[0, :] = x0

    dt = (t_max - t_min) / (nt - 1)

    # time-step using z_{k+1} = z_inf + (z_k - z_inf)*exp(-dt/tau)
    for k in range(nt - 1):
        X[k+1, :] = x_inf + (X[k, :] - x_inf) * np.exp(-dt / tau)

# Plotting
fig = plt.figure(figsize=(8, 6), dpi=110)
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(VV, TT, X)

ax.set_xlabel('Voltage (mV)')
ax.set_ylabel('Time (ms)')
ax.set_zlabel(f'{gating}(V, t)')
ax.set_title(gating)
plt.tight_layout()
plt.show()