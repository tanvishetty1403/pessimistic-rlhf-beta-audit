"""
scripts/length_confound.py  (criterion 3, length confound)
Use the ALL-SAMPLES joined npz (row order must match samples.jsonl). Empties are masked out.
Reports, with member-centred ensemble:
  A) Spearman(length, proxy mean) and (length, gold): within-prompt mean + pooled
  B) within-prompt Spearman(var, |err|): raw vs length-partialled, cluster-bootstrap CI (by group)
  C) Mean-tilt gold gain at KL~target, split into length-explained vs residual part
Usage: python scripts/length_confound.py --npz X.npz Y.npz --samples data\\pilot_sft\\samples.jsonl --out figs\\length_confound.json
"""
import argparse, json, os
import numpy as np
from scipy.stats import rankdata, spearmanr

def load_len(p):
    recs = [json.loads(l) for l in open(p) if l.strip()]
    return np.array([[(r["n_tokens"] if "n_tokens" in r else len(r["clean"])) for r in rec["responses"]] for rec in recs], float)

def rk(x):  # within-prompt ranks, NaN-safe via mask handled by caller
    return rankdata(x)

def per_prompt(fn, M, *arrs):
    out = np.full(M.shape[0], np.nan)
    for i in range(M.shape[0]):
        m = M[i]
        if m.sum() < 8: continue
        out[i] = fn(*[a[i][m] for a in arrs])
    return out

def sp(a, b):
    return spearmanr(a, b).correlation

def partial_sp(a, b, z):
    ra, rb, rz = rk(a), rk(b), rk(z)
    def res(y):
        A = np.c_[np.ones_like(rz), rz]; return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return np.corrcoef(res(ra), res(rb))[0, 1]

def boot(vals, grp, B=2000, seed=0):
    ok = ~np.isnan(vals); vals, grp = vals[ok], np.asarray(grp)[ok]
    u, inv = np.unique(grp, return_inverse=True); G = len(u)
    s = np.bincount(inv, vals, G); n = np.bincount(inv, minlength=G).astype(float)
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        k = rng.integers(0, G, G); out.append(s[k].sum() / n[k].sum())
    return float(vals.mean()), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

def tilt(mu, M, beta):
    z = np.where(M, beta * mu, -np.inf); z = z - z.max(1, keepdims=True)
    w = np.exp(z); w /= w.sum(1, keepdims=True)
    n = M.sum(1)
    kl = np.where(w > 0, w * np.log(np.where(w > 0, w, 1) * n[:, None]), 0).sum(1)
    return w, kl

def run(path, L, a):
    d = np.load(path, allow_pickle=True)
    R, gold = d["R"].astype(float), d["gold"].astype(float)
    P, N, K = R.shape
    assert L.shape == (P, N), f"length shape {L.shape} != npz {(P, N)}: npz and samples.jsonl are not aligned (use the all-samples npz)"
    M = ~d["empty"].astype(bool) if "empty" in d.files else np.ones((P, N), bool)
    grp = d["group"] if "group" in d.files else np.arange(P)
    off = R[M].mean(0); Rc = R - off                      # member centring (non-empty only)
    mu, var = Rc.mean(-1), Rc.var(-1, ddof=1)
    # error as in g2_f1: within-prompt centred, std-matched
    mc = np.where(M, mu - np.nanmean(np.where(M, mu, np.nan), 1, keepdims=True), 0)
    gc = np.where(M, gold - np.nanmean(np.where(M, gold, np.nan), 1, keepdims=True), 0)
    gc = gc * np.sqrt((mc[M] ** 2).mean() / (gc[M] ** 2).mean())
    err = np.abs(mc - gc)
    o = dict(n_prompts=P, n_nonempty=int(M.sum()))
    # A
    pr = per_prompt(sp, M, L, mu); gl = per_prompt(sp, M, L, gold)
    o["A_len_vs_proxy"] = dict(within=float(np.nanmean(pr)), pooled=float(sp(L[M], mu[M])), neg_prompts=int((pr < 0).sum()))
    o["A_len_vs_gold"] = dict(within=float(np.nanmean(gl)), pooled=float(sp(L[M], gold[M])), neg_prompts=int((gl < 0).sum()))
    # B
    raw = per_prompt(sp, M, var, err)
    par = per_prompt(partial_sp, M, var, err, L)
    lv = per_prompt(sp, M, L, var); le = per_prompt(sp, M, L, err)
    o["B_rho_var_abserr_raw"] = dict(zip(("mean", "ci95"), boot(raw, grp)))
    o["B_rho_var_abserr_len_partialled"] = dict(zip(("mean", "ci95"), boot(par, grp)))
    o["B_len_vs_var_within"] = float(np.nanmean(lv)); o["B_len_vs_abserr_within"] = float(np.nanmean(le))
    # C: Mean-tilt gain split; length-explained part = within-prompt linear fit of gold on length rank
    Lr = np.where(M, np.array([rankdata(np.where(M[i], L[i], np.inf)) for i in range(P)]), 0)
    gfit = np.zeros_like(gold)
    for i in range(P):
        m = M[i]; A = np.c_[np.ones(m.sum()), Lr[i][m]]
        gfit[i][m] = A @ np.linalg.lstsq(A, gold[i][m], rcond=None)[0]
    gres = gold - gfit
    gm = lambda x: np.where(M, x, 0).sum(1) / M.sum(1)
    rows = []
    for b in np.geomspace(0.05, 8, 40):
        w, kl = tilt(mu, M, b)
        tot = (w * gold).sum(1) - gm(gold); lenp = (w * gfit).sum(1) - gm(gfit); res = (w * gres).sum(1) - gm(gres)
        sel_len = (w * Lr).sum(1) / M.sum(1) - 0.5 * (1 + 1 / M.sum(1))  # mean length-rank pct of selection minus 0.5
        rows.append((kl.mean(), tot.mean(), lenp.mean(), res.mean(), sel_len.mean()))
    rows = np.array(rows); sd = gold[M].std()
    o["C_mean_tilt"] = {}
    for tgt in (0.25, 0.5, 1.0):
        r = rows[np.argmin(np.abs(rows[:, 0] - tgt))]
        o["C_mean_tilt"][f"KL~{tgt}"] = dict(kl=float(r[0]), gold_gain_sd=float(r[1] / sd), length_part_sd=float(r[2] / sd),
                                            residual_part_sd=float(r[3] / sd), sel_len_pct_vs_median=float(r[4]))
    return o

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True); ap.add_argument("--samples", required=True)
    ap.add_argument("--out", default="figs/length_confound.json")
    a = ap.parse_args(); L = load_len(a.samples); res = {}
    for p in a.npz:
        n = os.path.splitext(os.path.basename(p))[0]; res[n] = run(p, L, a)
        print(f"\n[{n}]"); print(json.dumps(res[n], indent=1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
