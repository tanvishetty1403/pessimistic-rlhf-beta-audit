"""
scripts/proxy_gold_corr.py

the proxy vs gold diagnostic 
"""

import sys, numpy as np
from scipy.stats import spearmanr

d = np.load(sys.argv[1], allow_pickle=True); m = d["R"].mean(-1); g = d["gold"]
e = d["empty"].astype(bool) if "empty" in d.files else np.zeros(g.shape, bool)

rho = [spearmanr(m[i][~e[i]], g[i][~e[i]]).correlation for i in range(len(g)) if (~e[i]).sum() > 10]

print(f"within-prompt proxy-gold rho: mean {np.mean(rho):+.3f}, min {np.min(rho):+.3f}, max {np.max(rho):+.3f}, positive in {sum(r > 0 for r in rho)}/{len(rho)} prompts")