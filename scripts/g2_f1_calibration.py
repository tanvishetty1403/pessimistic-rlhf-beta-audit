"""
scripts/g2_f1_calibration.py

G2: does the ensemble variance predict proxy-gold disagreement?  (rank correlation + binned calibration)
F1: calibration confound. Fit a scale kappa on sigma^2 against squared proxy-gold error on the TUNE split,
    rerun c* with the pessimism term kappa * sigma^2 / (2 c beta), and report c* before / after.

Why F1 matters: under the Gaussian form, r~ = mu - kappa*sigma^2/(2 c beta) = mu - sigma^2/(2 (c/kappa) beta),
so a fitted kappa is exactly a rescaling of c. c* != 1 only indicts the tie if kappa is ~1; if c*_uncal ~ 1/kappa,
a miscalibrated variance (not a wrong tie) explains it. The script prints c*_uncal * kappa as that analytic check.

ASSUMPTIONS:
  * "error" = proxy-mean minus gold, both centred per prompt and matched to the same within-prompt std
    (units: proxy standardized units). kappa is therefore relative to that unit match, so kappa != 1 alone is
    not proof of miscalibration. A variant that removes pure scale mismatch (kappa_lin) is also reported.
  * kappa and c* both use gold. They are separate quantities (error scale vs frontier optimum), but fit them on
    different halves of the groups (cross-fit, reported) before believing a before/after shift.
  * Primary metric is still OPEN in PREREG. Interim: M(c) = mean over a common KL grid of
    [E_gold under the tilted policy - mean gold of the reference]. c* = argmax_c M(c).
  * In-sample "best c minus c=1" is >= 0 by construction. Only the held-out (cross-fit) numbers are unbiased.

Input: one tune-split .npz (same contract as g1_g3_plots.py). 'test' is refused.
Usage:
    python scripts/g2_f1_calibration.py --npz noise_on.npz --out figs/ --config configs/base.yaml [--std_in figs/std_stats.json]
    python scripts/g2_f1_calibration.py --synthetic --out figs_smoke/
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json, os, warnings
import numpy as np
import yaml
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import pessimism as P_
from src import kl_ess as K_
from src import analysis_utils as U

BLUE, ORANGE, GRAY, DARK, GREEN = "#0072B2", "#D55E00", "#999999", "#222222", "#009E73"
DEFAULTS = dict(c=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], ess_min=10, bootstrap_reps=2000)


def load_cfg(path):
    if path and os.path.exists(path):
        cfg = yaml.safe_load(open(path)); g, s = cfg.get("grids", {}), cfg.get("stats", {})
        return dict(c=g.get("c", DEFAULTS["c"]), ess_min=g.get("ess_min", DEFAULTS["ess_min"]),
                    bootstrap_reps=s.get("bootstrap_reps", DEFAULTS["bootstrap_reps"]))
    warnings.warn(f"config '{path}' not found, using built-in defaults")
    return dict(DEFAULTS)


# ------------------------------------------------------------------------------- G2 / kappa
def centered_error(R, gold):
    """Per-sample proxy-gold disagreement in proxy units. Returns e (std-matched), e_lin (scale-mismatch removed), var."""
    mu = R.mean(-1); mc = mu - mu.mean(1, keepdims=True)
    gc = gold - gold.mean(1, keepdims=True)
    gc = gc * (np.sqrt((mc ** 2).mean()) / np.sqrt((gc ** 2).mean()))        # match pooled within-prompt std
    s = (mc * gc).sum() / (gc ** 2).sum()
    return mc - gc, mc - s * gc, R.var(-1, ddof=1), mc


def prompt_spearman(a, b):
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra = ra - ra.mean(1, keepdims=True); rb = rb - rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def bin_sums(e2, var, nb):
    """Per-prompt sums per variance-quantile bin: (P, nb) each for e^2, sigma^2 and counts."""
    Pn, N = var.shape
    edges = np.quantile(var, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.searchsorted(edges, var, side="right") - 1, 0, nb - 1)
    flat = (idx + nb * np.arange(Pn)[:, None]).ravel()
    f = lambda v: np.bincount(flat, weights=v.ravel(), minlength=Pn * nb).reshape(Pn, nb)
    return f(e2), f(var), f(np.ones_like(var))


def kappa_fit(Se2, Sv, Cn, W):
    """Count-weighted OLS of bin-mean squared error on bin-mean sigma^2, with intercept. Returns (kappa, alpha)."""
    n = W @ Cn
    m = n > 0
    x, y, wb = (W @ Sv)[m] / n[m], (W @ Se2)[m] / n[m], n[m] / n[m].sum()
    xm, ym = (wb * x).sum(), (wb * y).sum()
    k = (wb * (x - xm) * (y - ym)).sum() / (wb * (x - xm) ** 2).sum()
    return float(k), float(ym - k * xm)


# ------------------------------------------------------------------------------- F1 frontier
def frontier(r_fn, gold, betas, ess_min):
    Pn = gold.shape[0]; nb = len(betas); gm = gold.mean(1)
    kl, g, ok = np.zeros((Pn, nb)), np.zeros((Pn, nb)), np.zeros(nb, bool)
    for j, b in enumerate(betas):
        w = K_.tilt_weights(r_fn(b), b)
        kl[:, j] = K_.kl_from_weights(w)
        g[:, j] = (w * gold).sum(-1) - gm
        ok[j] = np.median(K_.ess(w)) >= ess_min
    return kl, g, ok


def M_of(W, cur, grid):
    kl, g, ok = cur
    s = W.sum(); x, y = ((W @ kl) / s)[ok], ((W @ g) / s)[ok]
    o = np.argsort(x)
    return float(np.interp(grid, x[o], y[o]).mean())


def curve_at(W, cur, grid):
    kl, g, ok = cur
    s = W.sum(); x, y = ((W @ kl) / s)[ok], ((W @ g) / s)[ok]
    o = np.argsort(x)
    return np.interp(grid, x[o], y[o])


def build(R, gold, c_grid, betas, ess_min, scale):
    return [frontier(lambda b, c=c: P_.r_gauss(R, P_.beta_rob(c, b), scale=scale), gold, betas, ess_min) for c in c_grid]


def ci(a):
    a = np.asarray(a)
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ------------------------------------------------------------------------------- run one condition
def run(name, d, a, cfg, out):
    st = json.load(open(a.std_in))[name] if a.std_in else U.fit_std(d["R"], d["gold"], a.std_mode)
    R, gold = U.apply_std(d["R"], d["gold"], st)
    Pn = R.shape[0]; inv = U.group_index(d["group"], Pn)
    G = int(inv.max()) + 1
    c_grid = list(a.cs or cfg["c"]); assert 1.0 in c_grid, "c grid must contain 1.0 (the tied default)"
    c1 = c_grid.index(1.0)
    betas = np.geomspace(a.betas[0], a.betas[1], int(a.betas[2]))
    ess_min = a.ess_min if a.ess_min is not None else cfg["ess_min"]
    B = a.B or min(cfg["bootstrap_reps"], 500)
    half = (np.random.default_rng(0).permutation(G) % 2)[inv]          # 0 / 1 per prompt, whole groups
    Wfull, WA, WB = np.ones(Pn), (half == 0).astype(float), (half == 1).astype(float)
    boots = list(U.boot_weights(inv, B, seed=1))
    Wb = np.vstack(boots)

    # ---- G2
    e, e_lin, var, mc = centered_error(R, gold)
    rho_abs = prompt_spearman(var, np.abs(e)); rho_sgn = prompt_spearman(var, e)
    g2 = {}
    for k, v in (("spearman_var_vs_abs_err", rho_abs), ("spearman_var_vs_signed_err", rho_sgn)):
        m, lo, hi = U.cluster_boot_mean(v, inv, B)
        g2[k] = dict(mean=float(m[0]), ci95=[float(lo[0]), float(hi[0])])
    Se2, Sv, Cn = bin_sums(e ** 2, var, a.bins)
    Se2l, _, _ = bin_sums(e_lin ** 2, var, a.bins)
    kap, alp = kappa_fit(Se2, Sv, Cn, Wfull)
    kap_lin, alp_lin = kappa_fit(Se2l, Sv, Cn, Wfull)
    kb = np.array([kappa_fit(Se2, Sv, Cn, w) for w in Wb])
    kA, kB = kappa_fit(Se2, Sv, Cn, WA)[0], kappa_fit(Se2, Sv, Cn, WB)[0]
    g2.update(kappa=dict(value=kap, alpha=alp, ci95=ci(kb[:, 0])), kappa_lin=dict(value=kap_lin, alpha=alp_lin),
              kappa_halves=[kA, kB], k_dof_note="K=%d -> ~%d dof per sigma^2; Hahami report ~71%% rel. error at K=5" % (R.shape[2], R.shape[2] - 1))
    g2["gate"] = ("PASS (rank corr CI > 0; threshold still OPEN)" if g2["spearman_var_vs_abs_err"]["ci95"][0] > 0
                  else "FAIL/INCONCLUSIVE: sigma^2 does not credibly predict |proxy-gold error|")
    n = Wfull @ Cn
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    xb, yb = (Wfull @ Sv) / n, (Wfull @ Se2) / n
    ax.plot(xb, yb, "o", ms=5, color=BLUE, label="binned data (20 quantile bins)")
    xs = np.array([0, xb.max()])
    ax.plot(xs, kap * xs + alp, color=ORANGE, lw=2, label=f"fit: kappa={kap:.2f}, alpha={alp:.2f}")
    ax.plot(xs, xs, color=GRAY, ls="--", lw=1.2, label="kappa=1, alpha=0 (calibrated)")
    ax.set(xlabel="ensemble variance sigma^2 (bin mean)", ylabel="mean squared proxy-gold error",
           title=f"G2: {name}  (rho(var,|err|)={g2['spearman_var_vs_abs_err']['mean']:.2f})")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, f"g2_calibration_{name}.png"), dpi=150); plt.close(fig)

    # ---- F1: frontiers under scale in {1, kappa_full, kappa_A, kappa_B}
    kpos = lambda k: max(k, 0.0)
    scales = {"uncal": 1.0, "cal": kpos(kap), "calA": kpos(kA), "calB": kpos(kB)}
    CUR = {k: build(R, gold, c_grid, betas, ess_min, s) for k, s in scales.items()}
    base = {"Mean": frontier(lambda b: P_.r_mean(R), gold, betas, ess_min),
            "WCO": frontier(lambda b: P_.r_wco(R), gold, betas, ess_min)}
    lams = {f"UWO lam={l:g}": frontier(lambda b, l=l: P_.r_uwo(R, l), gold, betas, ess_min) for l in a.lams}
    every = [x for k in CUR for x in CUR[k]] + list(base.values()) + list(lams.values())
    hi = min(float(((Wfull @ kl) / Pn)[ok].max()) if ok.any() else 0.0 for kl, g, ok in every) * 0.95
    msgs = []
    if hi < 1.0:
        msgs.append(f"common usable KL range tops out at {hi:.2f} nats: frontier comparison is weak (raise N / lower ess_min)")
    grid = np.linspace(max(0.1 * hi, 1e-3), hi, 8)
    Mfun = lambda W, key: np.array([M_of(W, cur, grid) for cur in CUR[key]])
    cstar = lambda m: int(np.argmax(m))
    Mu, Mc = Mfun(Wfull, "uncal"), Mfun(Wfull, "cal")
    iu, ic = cstar(Mu), cstar(Mc)
    # bootstrap over groups: c* distribution and in-sample gap (optimistic)
    Mu_b = np.array([Mfun(w, "uncal") for w in Wb]); Mc_b = np.array([Mfun(w, "cal") for w in Wb])
    cs_arr = np.array(c_grid)
    f1 = dict(c_grid=c_grid, kl_grid=grid, kappa_used=scales["cal"], kappa_note=(
        "kappa<=0: variance carries no calibratable scale; 'cal' = Mean aggregation" if kap <= 0 else ""),
        M_uncal=Mu, M_cal=Mc, cstar_uncal=c_grid[iu], cstar_cal=c_grid[ic],
        cstar_uncal_boot_ci95=ci(cs_arr[Mu_b.argmax(1)]), cstar_cal_boot_ci95=ci(cs_arr[Mc_b.argmax(1)]),
        analytic_cstar_cal_from_uncal=float(c_grid[iu] * kpos(kap)),
        M_baselines={k: M_of(Wfull, v, grid) for k, v in {**base, **lams}.items()},
        insample_best_minus_tied=dict(uncal=float(Mu[iu] - Mu[c1]), cal=float(Mc[ic] - Mc[c1]), note="optimistic: >=0 by construction"))
    # cross-fit: choose c* on one half, evaluate on the other (unbiased gap)
    cf = {}
    for tag, Wsel, Wev, kk in (("A->B", WA, WB, "calA"), ("B->A", WB, WA, "calB")):
        for lab, key in (("uncal", "uncal"), ("cal", kk)):
            j = cstar(Mfun(Wsel, key))
            gaps = np.array([M_of(w * Wev, CUR[key][j], grid) - M_of(w * Wev, CUR[key][c1], grid) for w in Wb])
            cf[f"{tag} {lab}"] = dict(c_selected=c_grid[j],
                                      heldout_gap_vs_tied=float(M_of(Wev, CUR[key][j], grid) - M_of(Wev, CUR[key][c1], grid)),
                                      ci95=ci(gaps))
    f1["crossfit"] = cf
    f1["tied_verdict_hint"] = ("heldout gap CI excludes 0 in at least one fold" if any(v["ci95"][0] > 0 or v["ci95"][1] < 0 for v in cf.values())
                               else "heldout gap CI includes 0 in all folds (no evidence the tie costs reward in this setup; check power/epsilon)")
    for lab, i_ in (("uncal", iu), ("cal", ic)):
        if i_ in (0, len(c_grid) - 1):
            msgs.append(f"c*_{lab}={c_grid[i_]:g} sits on the edge of the c grid: optimum not bracketed, extend the grid before reading c*")
    f1["warnings"] = msgs

    # ---- F1 plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    def line(cur, lab, col, ls="-", lw=1.6):
        ax[0].plot(grid, curve_at(Wfull, cur, grid), ls=ls, lw=lw, color=col, label=lab)
    line(base["Mean"], "Mean", DARK, "--"); line(base["WCO"], "WCO", GRAY, "--")
    bl = min(lams, key=lambda k: -M_of(Wfull, lams[k], grid)); line(lams[bl], f"best const-lambda ({bl})", GREEN, "-.")
    line(CUR["uncal"][c1], "tied c=1", BLUE, lw=2.4)
    if iu != c1: line(CUR["uncal"][iu], f"c*={c_grid[iu]:g} (uncal)", ORANGE, lw=2)
    ax[0].set(xlabel="KL budget (nats, mean over prompts)", ylabel="gold gain vs reference (standardized)", title=f"F1 frontiers: {name}")
    ax[0].legend(frameon=False, fontsize=8)
    for M_, Mb_, col, lab in ((Mu, Mu_b, BLUE, "uncalibrated (scale 1)"), (Mc, Mc_b, ORANGE, f"calibrated (kappa={kap:.2f})")):
        ax[1].plot(cs_arr, M_, marker="o", ms=4, lw=2, color=col, label=lab)
        ax[1].fill_between(cs_arr, np.percentile(Mb_, 2.5, 0), np.percentile(Mb_, 97.5, 0), color=col, alpha=0.15, lw=0)
    ax[1].axvline(1.0, color=GRAY, ls=":", lw=1)
    for k, v in f1["M_baselines"].items():
        if k in ("Mean", "WCO"): ax[1].axhline(v, color=DARK if k == "Mean" else GRAY, ls="--", lw=1, label=k)
    ax[1].set(xscale="log", xlabel="c (beta_rob = c * beta_policy)", ylabel="M(c): mean gold gain over common KL grid",
              title="Interim metric vs c (bands: cluster bootstrap)")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"f1_frontier_{name}.png"), dpi=150); plt.close(fig)
    return dict(n_prompts=Pn, n_groups=G, empty_frac=d["empty_frac"], std=st, g2=g2, f1=f1)


def synthetic(P=160, N=1024, K=5, seed=0):
    """Smoke test: heteroscedastic uncertainty s drives both the shared error and the ensemble spread."""
    rng = np.random.default_rng(seed)
    gold = rng.normal(size=(P, N)); s = np.exp(0.5 * rng.normal(size=(P, N)))
    shared = rng.standard_t(4, size=(P, N)) * 0.4 * s
    member = rng.normal(size=(P, N, K)) * 0.5 * s[..., None]
    return dict(R=gold[..., None] + shared[..., None] + member, gold=gold, split="tune",
                group=np.array([f"g{i // 4}" for i in range(P)]), empty_frac=0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", default=[]); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="figs"); ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--std_mode", choices=["within", "global"], default="within")
    ap.add_argument("--std_in", default=None, help="std_stats.json from g1_g3_plots.py (keys = npz basenames)")
    ap.add_argument("--betas", type=float, nargs=3, default=[0.03, 5.0, 30], metavar=("LO", "HI", "NUM"),
                    help="dense beta grid used only to trace frontiers (the pre-registered beta grid is too coarse for interpolation)")
    ap.add_argument("--cs", type=float, nargs="+", default=None)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0], help="const-lambda UWO baseline grid (Coste range)")
    ap.add_argument("--bins", type=int, default=20); ap.add_argument("--ess_min", type=float, default=None)
    ap.add_argument("--B", type=int, default=None, help="bootstrap reps (default min(config, 500): F1 bootstraps the whole frontier)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_cfg(a.config)
    data = {"synthetic": synthetic()} if a.synthetic else {os.path.splitext(os.path.basename(p))[0]: U.load_npz(p) for p in a.npz}
    assert data, "give --npz or --synthetic"
    res = {n: run(n, d, a, cfg, a.out) for n, d in data.items()}
    U.dump_json(res, os.path.join(a.out, "g2_f1_summary.json"))
    for n, r in res.items():
        g2, f1 = r["g2"], r["f1"]
        print(f"[{n}] G2: rho(var,|err|)={g2['spearman_var_vs_abs_err']['mean']:+.3f} CI {g2['spearman_var_vs_abs_err']['ci95']}  -> {g2['gate']}")
        print(f"   kappa={g2['kappa']['value']:.3f} CI {g2['kappa']['ci95']}  (kappa_lin={g2['kappa_lin']['value']:.3f}, halves={g2['kappa_halves']})")
        print(f"   F1: c*_uncal={f1['cstar_uncal']} (boot CI {f1['cstar_uncal_boot_ci95']})  c*_cal={f1['cstar_cal']} (boot CI {f1['cstar_cal_boot_ci95']})  "
              f"analytic kappa*c*_uncal={f1['analytic_cstar_cal_from_uncal']:.2f}")
        for k, v in f1["crossfit"].items():
            print(f"   held-out {k}: c={v['c_selected']:g}  gap vs tied={v['heldout_gap_vs_tied']:+.4f} CI {v['ci95']}")
        print("   ", f1["tied_verdict_hint"])
        for w in f1["warnings"] + ([f1["kappa_note"]] if f1["kappa_note"] else []):
            print("   WARNING:", w)