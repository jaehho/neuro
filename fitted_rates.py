import numpy as np

def alpha_m(V):
    # 0.1*(V + 40.0) / (1.0 - np.exp(-(V + 40.0)/10.0))
    out = np.empty_like(V, dtype=float)
    denom = 1.0 - np.exp(-(V + 40.0)/10.0)
    mask = np.abs(denom) < 1e-7
    out[~mask] = 0.1*(V[~mask] + 40.0)/denom[~mask]
    out[mask] = 1.0
    return out

def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0)/18.0)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0)/20.0)

def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0)/10.0))

def alpha_n(V):
    # 0.01*(V + 55.0) / (1.0 - np.exp(-(V + 55.0)/10.0))
    out = np.empty_like(V, dtype=float)
    denom = 1.0 - np.exp(-(V + 55.0)/10.0)
    mask = np.abs(denom) < 1e-7
    out[~mask] = 0.01*(V[~mask] + 55.0)/denom[~mask]
    out[mask] = 0.1
    return out

def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0)/80.0)