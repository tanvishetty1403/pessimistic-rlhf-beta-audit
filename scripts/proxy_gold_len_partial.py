"""
scripts/proxy_gold_len_partial.py
"""

import sys, json, numpy as np
from scipy.stats import rankdata

d = np.load(sys.argv[1], allow_pickle=True); m = d["R"].mean(-1); g = d["gold"]
e = d["empty"].astype(bool) if "empty" in d.files else np.zeros(g.shape, bool)
recs = {r["prompt_id"]: r for r in (json.loads(l) for l in open(sys.argv[2]) if l.strip())}

L = np.array([[{x["response_id"]: x["n_tokens"] for x in recs[str(p)]["responses"]}[str(r)] for r in d["response_ids"][i]] for i, p in enumerate(d["prompt_ids"])])

def resid(y, x):
    ry, rx = rankdata(y), rankdata(x); rx = rx - rx.mean(); ry = ry - ry.mean()
    return ry - (rx * ry).sum() / (rx ** 2).sum() * rx

out = []
for i in range(len(g)):
    k = ~e[i]
    if k.sum() < 10: continue
    out.append(np.corrcoef(resid(m[i][k], L[i][k]), resid(g[i][k], L[i][k]))[0, 1])

print(f"length-partialled within-prompt proxy-gold rho: mean {np.mean(out):+.3f}, positive in {sum(o > 0 for o in out)}/{len(out)}")
