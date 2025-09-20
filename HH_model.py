import numpy as np
import matplotlib.pyplot as plt
from fitted_rates import alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n
from HH_constants import c_m, g_Na, g_K, g_L, E_Na, E_K, E_L

# External current (uA/cm^2)
def I_ext(t):
    return 10.0 if (10.0 <= t <= 40.0) else 0.0

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

n_post = np.empty(nt)
n_post[0] = n0

for k in range(nt-1):
    Vk, nk_post = V[k], n_post[k]
    # gNa_t = g_Na * (mk**3) * hk
    # gK_t  = g_K  * (nk_post**4)
    # gL_t  = g_L
    # INa = gNa_t * (Vk - E_Na)
    # IK  = gK_t  * (Vk - E_K)
    # IL  = gL_t  * (Vk - E_L)
    # dVdt = (I_ext(t[k]) - (INa + IK + IL)) / c_m

    # am, bm = alpha_m(Vk), beta_m(Vk)
    # ah, bh = alpha_h(Vk), beta_h(Vk)
    an_post, bn_post = alpha_n(Vk), beta_n(Vk)
    # dmdt = am*(1.0 - mk) - bm*mk
    # dhdt = ah*(1.0 - hk) - bh*hk
    dndt = an_post*(1.0 - nk_post) - bn_post*nk_post

    # V[k+1] = Vk + dt*dVdt
    # m[k+1] = np.clip(mk + dt*dmdt, 0.0, 1.0)
    # h[k+1] = np.clip(hk + dt*dhdt, 0.0, 1.0)
    n_post[k+1] = np.clip(nk_post + dt*dndt, 0.0, 1.0)
    

fig = plt.figure(figsize=(8, 6))

# 3D plot
ax = fig.add_subplot(211)
ax.plot(t, V, color='k')

# Orthographic projections
ax2 = fig.add_subplot(212)
ax2.plot(t, n, label='n gate', color='b')
ax2.plot(t, n_post, label='n_post gate', color='r', linestyle='dashed', linewidth=3)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Probability')
ax2.legend()

plt.tight_layout()
plt.show()