"""Day-2 D: (1) BoN KL check, (2) ESS and KL vs beta_policy on synthetic N-sample sets. Writes outputs/synthetic_*.png."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import kl_ess as K

os.makedirs("outputs", exist_ok=True)
rng = np.random.default_rng(0)

# (1) exact BoN weights -> KL estimator vs analytic
ns = [2, 4, 8, 16, 64, 256]
N = 4096
r = rng.normal(size=N)
est = [float(K.kl_from_weights(K.bon_weights(r, n))) for n in ns]
ana = [K.bon_kl_analytic(n) for n in ns]
for n, e, a in zip(ns, est, ana):
    print(f"n={n:4d}  est={e:.4f}  analytic={a:.4f}  diff={e-a:+.4f}")

# (2) ESS / KL vs beta_policy for several N
betas = np.logspace(-2, 1, 40)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for N in (64, 256, 1024, 4096):
    r = rng.normal(size=(100, N))
    E, KL = [], []
    for b in betas:
        w = K.tilt_weights(r, b)
        E.append(K.ess(w).mean()); KL.append(K.kl_from_weights(w).mean())
    ax[0].loglog(betas, E, label=f"N={N}"); ax[1].semilogx(betas, KL, label=f"N={N}")
ax[0].set(xlabel="beta_policy", ylabel="ESS", title="G3: ESS vs beta_policy (N(0,1) rewards)"); ax[0].axhline(10, ls="--", c="k", lw=0.8)
ax[1].set(xlabel="beta_policy", ylabel="KL (nats)", title="Reachable KL (capped at log N)")
ax[0].legend(); ax[1].legend(); plt.tight_layout(); plt.savefig("outputs/synthetic_ess_kl.png", dpi=130)
print("saved outputs/synthetic_ess_kl.png")
