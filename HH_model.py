import numpy as np
import matplotlib.pyplot as plt
from fitted_rates import alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n
from HH_constants import C_m, g_Na, g_K, g_L, E_Na, E_K, E_L

gating = 'm'
if gating == 'm':
    alpha = alpha_m; beta = beta_m
elif gating == 'h':
    alpha = alpha_h; beta = beta_h
else:
    alpha = alpha_n; beta = beta_n

# Current-clamp numerical euler integration
def I_ext(t):
    return 10.0 if (10.0 <= t <= 40.0) else 0.0

t_max, dt = 60.0, 0.01
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
    dVdt = (I_ext(t[k]) - (INa + IK + IL)) / C_m

    am, bm = alpha_m(Vk), beta_m(Vk)
    ah, bh = alpha_h(Vk),        beta_h(Vk)
    an, bn = alpha_n(Vk), beta_n(Vk)
    dmdt = am*(1.0 - mk) - bm*mk
    dhdt = ah*(1.0 - hk) - bh*hk
    dndt = an*(1.0 - nk) - bn*nk

    V[k+1] = Vk + dt*dVdt
    m[k+1] = np.clip(mk + dt*dmdt, 0.0, 1.0)
    h[k+1] = np.clip(hk + dt*dhdt, 0.0, 1.0)
    n[k+1] = np.clip(nk + dt*dndt, 0.0, 1.0)

plt.ion()
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
ax.plot(V, t, m if gating=='m' else (h if gating=='h' else n))

ax.set_xlabel('Voltage (mV)')
ax.set_ylabel('Time (ms)')
ax.set_zlabel(f'{gating}(t)')
ax.set_title(gating)
plt.tight_layout()
plt.show(block=True)