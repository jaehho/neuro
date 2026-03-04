from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

@dataclass
class Params:
    T: float = 20.0
    dt: float = 1e-4
    seed: int = 1
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
    record_every: float = 1e-3


def simulate(p):
    n = int(p.T / p.dt)
    period_steps = max(1, round(1.0 / (p.r_pre_rate * p.dt)))
    V = p.E_L; ref_remaining = 0.0
    I_s = 0.0; x_pre = 0.0; r_pre = 0.0
    y_post = 0.0; r_post = 0.0; E = 0.0; R_bar = 0.0; w = p.w0

    rec_step = max(1, int(p.record_every / p.dt))
    m = n // rec_step + 2
    keys = ["t", "V", "I_s", "x_pre", "y_post", "E", "r_pre", "r_post", "R_bar", "w"]
    rec = {k: np.zeros(m) for k in keys}
    k = 0

    for step in range(n):
        t = step * p.dt
        pre_spike = step % period_steps == 0
        if pre_spike:
            I_s += 1.0; x_pre += 1.0; r_pre += 1.0
            E -= p.eta_minus * w * y_post
        I_s   += p.dt * (-I_s   / p.tau_s)
        x_pre += p.dt * (-x_pre / p.tau_plus)
        r_pre += p.dt * (-r_pre / p.tau_r)

        post_spike = False
        if ref_remaining <= 0.0:
            V_new = V + (p.dt / p.tau_m) * (-(V - p.E_L) + p.R_m * w * I_s)
            if V < p.theta and V_new >= p.theta:
                post_spike = True
                V = p.V_reset; ref_remaining = p.tau_ref
            else:
                V = V_new
        else:
            ref_remaining -= p.dt; V = p.V_reset

        if post_spike:
            y_post += 1.0; r_post += 1.0
            E += p.eta_plus * (p.wmax - w) * x_pre

        y_post += p.dt * (-y_post / p.tau_minus)
        r_post += p.dt * (-r_post / p.tau_r)
        E      += p.dt * (-E / p.tau_e)
        R       = -(r_post - 0.5 * r_pre) ** 2
        R_bar  += (p.dt / p.tau_Rbar) * (-R_bar + R)
        w      += p.dt * (R - R_bar) * E
        w       = np.clip(w, 0.0, p.wmax)

        if step % rec_step == 0:
            rec["t"][k] = t; rec["V"][k] = V; rec["I_s"][k] = I_s
            rec["x_pre"][k] = x_pre; rec["y_post"][k] = y_post
            rec["E"][k] = E; rec["r_pre"][k] = r_pre
            rec["r_post"][k] = r_post; rec["R_bar"][k] = R_bar
            rec["w"][k] = w
            k += 1

    return {kk: vv[:k] for kk, vv in rec.items()}


p = Params()
rec = simulate(p)

# State variables (exclude time)
VAR_NAMES = ["V", "I_s", "x_pre", "y_post", "E", "r_pre", "r_post", "R_bar", "w"]
LABELS = {
    "V":      r"$V$ (mV)",
    "I_s":    r"$I_s$",
    "x_pre":  r"$x_\mathrm{pre}$",
    "y_post": r"$y_\mathrm{post}$",
    "E":      r"$E$",
    "r_pre":  r"$r_\mathrm{pre}$",
    "r_post": r"$r_\mathrm{post}$",
    "R_bar":  r"$\bar{R}$",
    "w":      r"$w$",
}

pairs = list(combinations(VAR_NAMES, 2))  # C(9,2) = 36
n_pairs = len(pairs)
ncols = 6
nrows = -(-n_pairs // ncols)  # ceiling division

t = rec["t"]
norm_t = (t - t.min()) / (t.max() - t.min())

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
axes = axes.flatten()

for idx, (xvar, yvar) in enumerate(pairs):
    ax = axes[idx]
    ax.scatter(rec[xvar], rec[yvar], c=norm_t, cmap="viridis", s=1,
               linewidths=0, rasterized=True)
    ax.set_xlabel(LABELS[xvar], fontsize=8)
    ax.set_ylabel(LABELS[yvar], fontsize=8)
    ax.tick_params(labelsize=6)

# Hide unused subplots
for idx in range(n_pairs, len(axes)):
    axes[idx].set_visible(False)

# Single shared colorbar in the last empty cell area
sm = plt.cm.ScalarMappable(cmap="viridis",
                            norm=plt.Normalize(vmin=t.min(), vmax=t.max()))
sm.set_array([])
# Use the first invisible axis region for colorbar placement
cbar_ax = fig.add_axes([0.88, 0.02, 0.012, 0.25])
fig.colorbar(sm, cax=cbar_ax, label="time (s)")

fig.suptitle("Projected trajectories — all pairwise state variables", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("trajectories.png", dpi=130,
            bbox_inches="tight")
plt.show()
print(f"Plotted {n_pairs} pairs ({nrows}x{ncols} grid)")