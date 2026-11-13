"""scripts/ensemble_offset_diagnostic.py

Needs only R (P,N,K) from a proxy npz (no gold, no split): how much of the ensemble variance is each member's
arbitrary additive offset rather than disagreement? A Bradley-Terry reward model is identified only up to a constant,
so raw sigma^2 across members mixes the two.

python scripts/ensemble_offset_diagnostic.py --npz a.npz b.npz [--out diag.json]

Per file: variance decomposition (total / global member offsets / member x prompt offsets / within-prompt
disagreement), within-prompt CV of sigma^2 (raw vs member-centred), and the mean within-prompt Spearman of
sigma^2 against mu. rho(raw var, mu) near +1 means the raw variance is mostly a function of the reward itself.
"""
import argparse
import json
import pathlib

import numpy as np
from scipy.stats import rankdata


def spearman_rows(a, b):
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra = ra - ra.mean(1, keepdims=True)
    rb = rb - rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def diagnose(R):
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 3 or R.shape[2] < 2 or not np.isfinite(R).all():
        raise ValueError(f"need finite R (P,N,K>=2); got {R.shape}")
    P, N, K = R.shape
    var, mu = R.var(-1, ddof=1), R.mean(-1)
    gm, pm = R.mean((0, 1)), R.mean(1)
    vg, vp = (R - gm).var(-1, ddof=1), (R - pm[:, None, :]).var(-1, ddof=1)
    cv = lambda v: float((v.std(1, ddof=1) / v.mean(1)).mean())
    tot = float(var.mean())
    return dict(P=P, N=N, K=K, var_total=tot, var_member_offsets=float(gm.var(ddof=1)),
                var_member_prompt_offsets=float(pm.var(1, ddof=1).mean()), var_within_prompt=float(vp.mean()),
                frac_offset=float(gm.var(ddof=1) / tot), cv_raw=cv(var), cv_member_centered=cv(vg), cv_prompt_centered=cv(vp),
                spearman_var_mu_raw=float(spearman_rows(var, mu).mean()),
                spearman_var_mu_member_centered=float(spearman_rows(vg, mu).mean()),
                spearman_raw_var_vs_centered_var=float(spearman_rows(var, vp).mean()),
                member_means=[float(x) for x in gm], member_stds=[float(x) for x in R.reshape(-1, K).std(0)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {}
    for f in a.npz:
        if "test" in pathlib.Path(f).name.lower():
            raise SystemExit(f"refusing test-named file: {f}")
        res[pathlib.Path(f).stem] = diagnose(np.load(f, allow_pickle=False)["R"])
    for k, v in res.items():
        print(f"[{k}] var_total={v['var_total']:.3f} offsets={v['var_member_offsets']:.3f} ({v['frac_offset']:.1%}) "
              f"within-prompt={v['var_within_prompt']:.3f} | spearman(var,mu) raw={v['spearman_var_mu_raw']:+.2f} "
              f"centered={v['spearman_var_mu_member_centered']:+.2f} | spearman(raw var, centered var)={v['spearman_raw_var_vs_centered_var']:+.2f}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
