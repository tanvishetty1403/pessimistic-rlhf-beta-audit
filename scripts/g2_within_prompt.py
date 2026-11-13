"""scripts/g2_within_prompt.py

Splits the G2 calibration slope into a WITHIN-prompt and a BETWEEN-prompt part.

g2_f1_calibration.py bins sigma^2 by quantiles over all cells and regresses squared error on it, so its kappa mixes
"prompts where members disagree are prompts where the proxy errs" (between) with "samples where members disagree
are samples where the proxy errs" (within). Tilting / BoN is normalised per prompt, so only the within part can
change which sample is selected.

python scripts/g2_within_prompt.py --npz a_joined_dropempty.npz b_joined_dropempty.npz --center_members --out figs/g2_within.json

Definitions (same error and standardization as g2_f1_calibration.py):
  e = centered_error(R_std, gold_std) ; var = ensemble variance (ddof=1) of the standardized members
  kappa_within  : OLS slope (through the origin) of (e^2 - mean_p e^2) on (var - mean_p var), prompts resampled by group
  kappa_between : weighted OLS slope (with intercept) of prompt-mean e^2 on prompt-mean var
Cluster bootstrap over first-turn groups; 95% percentile intervals. Refuses test-split npz (via load_npz).
"""
import argparse
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import analysis_utils as U  # noqa: E402


def centered_error(R, gold):
    """Copied from g2_f1_calibration.py (same definition): proxy-mean minus gold, both centred per prompt,
    gold matched to the pooled within-prompt std of the proxy mean."""
    mu = R.mean(-1)
    mc = mu - mu.mean(1, keepdims=True)
    gc = gold - gold.mean(1, keepdims=True)
    gc = gc * (np.sqrt((mc ** 2).mean()) / np.sqrt((gc ** 2).mean()))
    return mc - gc, R.var(-1, ddof=1)


def slopes(e2, var, W):
    """W: (b, P) prompt weights. Returns (kappa_within (b,), kappa_between (b,))."""
    xw, yw = var - var.mean(1, keepdims=True), e2 - e2.mean(1, keepdims=True)
    sxy, sxx = (xw * yw).sum(1), (xw * xw).sum(1)
    kw = (W @ sxy) / (W @ sxx)
    xb, yb = var.mean(1), e2.mean(1)
    s = W.sum(1)
    mx, my = (W @ xb) / s, (W @ yb) / s
    cov = (W @ (xb * yb)) / s - mx * my
    vr = (W @ (xb * xb)) / s - mx ** 2
    return kw, cov / vr


def analyse(d, center_members=False, B=2000, seed=0):
    st = U.fit_std(d["R"], d["gold"], "within", center_members=center_members)
    R, gold = U.apply_std(d["R"], d["gold"], st)
    e, var = centered_error(R, gold)
    inv = U.group_index(d["group"], R.shape[0])
    e2 = e ** 2
    W1 = np.ones((1, R.shape[0]))
    kw0, kb0 = (float(x[0]) for x in slopes(e2, var, W1))
    kws, kbs = [], []
    for W in U.boot_weights(inv, B, seed):
        a, b = slopes(e2, var, W)
        kws.append(a)
        kbs.append(b)
    kws, kbs = np.concatenate(kws), np.concatenate(kbs)
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
    return dict(n_prompts=int(R.shape[0]), n_groups=int(inv.max()) + 1, center_members=center_members,
                kappa_within=dict(value=kw0, ci95=ci(kws)), kappa_between=dict(value=kb0, ci95=ci(kbs)),
                frac_var_between_prompts=float(var.mean(1).var() / var.var()),
                frac_err2_between_prompts=float(e2.mean(1).var() / e2.var()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--center_members", action="store_true")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = {}
    for f in a.npz:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = U.load_npz(f)
        res[pathlib.Path(f).stem] = analyse(d, a.center_members, a.B)
    for k, v in res.items():
        w, b = v["kappa_within"], v["kappa_between"]
        print(f"[{k}] kappa_within={w['value']:+.3f} CI [{w['ci95'][0]:+.3f}, {w['ci95'][1]:+.3f}] | "
              f"kappa_between={b['value']:+.3f} CI [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] | "
              f"share of var / err^2 variance that is between prompts: {v['frac_var_between_prompts']:.2f} / {v['frac_err2_between_prompts']:.2f}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
