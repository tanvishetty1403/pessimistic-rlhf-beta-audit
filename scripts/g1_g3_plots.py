"""
scripts/g1_g3_plots.py

G1: BoN proxy vs gold under Mean aggregation (does overoptimization appear?)
G3: ESS and KL of the weighted-resampling policy approximation vs beta_policy and vs achieved KL

Input: one .npz per condition (e.g. noise_on.npz, noise_off.npz), see src/analysis_utils.py for the contract:
    R (P,N,K), gold (P,N), split (str), group (P,) [first-turn group], empty (P,N) [optional]
'test' npz files are refused. Standardization stats are fit on the file's own (tune) data and saved to
std_stats.json; pass --std_in to reuse saved stats (required for any later split).

Usage:
    python scripts/g1_g3_plots.py --npz noise_on.npz noise_off.npz --out figs/ --config configs/base.yaml
    python scripts/g1_g3_plots.py --synthetic --out figs_smoke/        # smoke test only, not evidence
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import argparse, json, os, warnings
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import pessimism as P_
from src import kl_ess as K_
from src import analysis_utils as U

COLORS = {"proxy": "#0072B2", "gold": "#D55E00", "analytic": "#555555", "emp": "#0072B2", "mean": "#222222"}
DEFAULTS = dict(c=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], beta_policy=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
                ess_min=10, bootstrap_reps=2000)
COSTE_REF_KL = 8.4  # Coste et al. BoN up to n=12,500 (~8.4 nats); context for the KL-range warning


def load_cfg(path):
    if path and os.path.exists(path):
        cfg = yaml.safe_load(open(path))
        g, s = cfg.get("grids", {}), cfg.get("stats", {})
        return dict(c=g.get("c", DEFAULTS["c"]), beta_policy=g.get("beta_policy", DEFAULTS["beta_policy"]),
                    ess_min=g.get("ess_min", DEFAULTS["ess_min"]),
                    bootstrap_reps=s.get("bootstrap_reps", DEFAULTS["bootstrap_reps"]))
    warnings.warn(f"config '{path}' not found, using built-in defaults (mirror of configs/base.yaml)")
    return dict(DEFAULTS)


# ----------------------------------------------------------------------------- G1
def g1_curves(R, gold, n_cand):
    """BoN (exact selection probs over the N cached samples) under Mean aggregation.
    Returns per-prompt arrays (P, n): proxy, gold, empirical KL."""
    Pn = R.shape[0]
    proxy = R.mean(axis=-1)
    pr, go, kl = (np.zeros((Pn, len(n_cand))) for _ in range(3))
    for p in range(Pn):
        for j, n in enumerate(n_cand):
            w = K_.bon_weights(proxy[p], int(n))
            pr[p, j] = (w * proxy[p]).sum()
            go[p, j] = (w * gold[p]).sum()
            kl[p, j] = K_.kl_from_weights(w)
    return pr, go, kl


def choose_n_max(kl, n_cand, tol):
    """Largest prefix of the grid where median |empirical KL - analytic KL| <= tol (pool BoN stops
    matching population BoN as n approaches N)."""
    ana = np.array([K_.bon_kl_analytic(int(n)) for n in n_cand])
    gap = np.median(np.abs(kl - ana[None]), axis=0)
    bad = np.where(gap > tol)[0]
    j = (bad[0] - 1) if len(bad) else len(n_cand) - 1
    return max(j, 1), ana, gap


def crossfit_drop(go, inv, j_full):
    """Overoptimization drop with the peak chosen on the OTHER half of the groups (no selection bias).
    d_i = gold(n_max) - gold(n at the peak chosen from the half not containing prompt i)."""
    G = int(inv.max()) + 1
    half = np.random.default_rng(0).permutation(G) % 2
    h = half[inv]                                   # per-prompt half label
    d = np.zeros(go.shape[0])
    for k in (0, 1):
        j = int(np.argmax(go[h != k].mean(0)))      # peak from the other half
        d[h == k] = go[h == k, -1] - go[h == k, j]
    return d


def g1_summarize(name, pr, go, kl, n_cand, inv, tol, n_ref, B, empty_frac):
    j_max, ana, gap = choose_n_max(kl, n_cand, tol)
    sl = slice(0, j_max + 1)
    n_grid, pr, go, kl, ana = n_cand[sl], pr[:, sl], go[:, sl], kl[:, sl], ana[sl]
    dpr, dgo = pr - pr[:, :1], go - go[:, :1]       # change vs n=1, per prompt
    x = np.median(kl, axis=0)
    j_peak = int(np.argmax(dgo.mean(0)))
    peak_at_end = j_peak == len(n_grid) - 1
    d = crossfit_drop(dgo, inv, j_peak)
    dm, dlo, dhi = U.cluster_boot_mean(d, inv, B)
    jr = int(np.argmin(np.abs(n_grid - n_ref)))
    fixed_go = U.cluster_boot_mean(dgo[:, -1] - dgo[:, jr], inv, B)
    gain_pr = U.cluster_boot_mean(dpr[:, -1] - dpr[:, jr], inv, B)
    if peak_at_end:
        verdict = "INCONCLUSIVE: gold peak not reached within the usable KL range"
    elif dhi[0] < 0:
        verdict = "PASS: gold falls after its peak (cross-fit CI excludes 0)"
    else:
        verdict = "NOT SUPPORTED: no credible gold drop (cross-fit CI includes 0)"
    msgs = []
    if x[-1] < COSTE_REF_KL:
        msgs.append(f"max reachable KL {x[-1]:.2f} nats < Coste's ~{COSTE_REF_KL} (F2 risk: larger N / label noise)")
    summ = dict(n_prompts=int(pr.shape[0]), n_groups=int(inv.max()) + 1, empty_frac=empty_frac,
                n_grid=n_grid, kl_emp_median=x, kl_analytic=ana, kl_gap_median=gap[sl],
                n_max=int(n_grid[-1]), max_kl_reached=float(x[-1]),
                gold_peak_n=int(n_grid[j_peak]), gold_peak_kl=float(x[j_peak]), peak_at_end=bool(peak_at_end),
                crossfit_drop_gold=dict(mean=float(dm[0]), ci95=[float(dlo[0]), float(dhi[0])]),
                fixed_contrast=dict(n_ref=int(n_grid[jr]), n_max=int(n_grid[-1]),
                                    gold_nmax_minus_nref=dict(mean=float(fixed_go[0][0]), ci95=[float(fixed_go[1][0]), float(fixed_go[2][0])]),
                                    proxy_nmax_minus_nref=dict(mean=float(gain_pr[0][0]), ci95=[float(gain_pr[1][0]), float(gain_pr[2][0])])),
                verdict=verdict, warnings=msgs)
    curves = dict(x=x, ana=ana, n=n_grid, dpr=dpr, dgo=dgo, j_peak=j_peak, n_cand=n_cand, j_max=j_max, kl_gap_all=gap)
    return summ, curves


def plot_g1(curves_by_cond, summ_by_cond, inv_by_cond, B, out):
    names = list(curves_by_cond)
    fig, axes = plt.subplots(2, len(names), figsize=(5.6 * len(names), 7.2), squeeze=False)
    for col, name in enumerate(names):
        c, s, inv = curves_by_cond[name], summ_by_cond[name], inv_by_cond[name]
        ax = axes[0, col]
        for lab, arr, key in (("proxy (Mean)", c["dpr"], "proxy"), ("gold", c["dgo"], "gold")):
            m, lo, hi = U.cluster_boot_mean(arr, inv, B)
            ax.plot(c["x"], m, lw=2, color=COLORS[key], label=lab)
            ax.fill_between(c["x"], lo, hi, alpha=0.18, color=COLORS[key], lw=0)
        ax.axvline(c["x"][c["j_peak"]], color=COLORS["gold"], ls=":", lw=1)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set(title=f"G1: {name}", xlabel="empirical KL(BoN || ref), nats (median over prompts)",
               ylabel="reward change vs n=1 (standardized)")
        ax.text(0.02, 0.97, s["verdict"].split(":")[0], transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(frameon=False, loc="center left")
        ax2 = axes[1, col]
        ax2.plot(c["n_cand"], [K_.bon_kl_analytic(int(n)) for n in c["n_cand"]], color=COLORS["analytic"], lw=2,
                 label="analytic log n - (n-1)/n")
        ax2.plot(c["n_cand"][: c["j_max"] + 1], c["x"], color=COLORS["emp"], lw=2, marker="o", ms=4, label="empirical (used)")
        ax2.axvline(c["n_cand"][c["j_max"]], color="#999999", ls="--", lw=1)
        ax2.set(xscale="log", xlabel="n", ylabel="KL (nats)", title="KL estimator check; dashed = chosen n_max")
        ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "g1_bon_proxy_vs_gold.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- G3
def g3(R, beta_grid, c_grid, ess_min):
    """Per condition (Mean, and tied-form r_gauss at each c): ESS/N and KL vs beta_policy."""
    N = R.shape[1]
    conds = {"Mean": None, **{f"c={c:g}": c for c in c_grid}}
    out = {}
    for name, c in conds.items():
        rec = {k: [] for k in ("ess_frac_med", "ess_frac_p10", "kl_med", "frac_ess_ok", "usable")}
        for b in beta_grid:
            r = P_.r_mean(R) if c is None else P_.r_gauss(R, P_.beta_rob(c, b))
            w = K_.tilt_weights(r, b)
            e, k = K_.ess(w), K_.kl_from_weights(w)
            rec["ess_frac_med"].append(float(np.median(e) / N))
            rec["ess_frac_p10"].append(float(np.percentile(e, 10) / N))
            rec["kl_med"].append(float(np.median(k)))
            rec["frac_ess_ok"].append(float((e >= ess_min).mean()))
            rec["usable"].append(bool(np.median(e) >= ess_min))   # criterion still OPEN: pre-register it
        out[name] = rec
    return out


def plot_g3(res, beta_grid, N, ess_min, title, out):
    names = list(res)
    cs = [n for n in names if n != "Mean"]
    shades = plt.cm.Blues(np.linspace(0.3, 0.95, max(len(cs), 1)))
    col = {"Mean": COLORS["mean"], **{n: shades[i] for i, n in enumerate(cs)}}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for name, d in res.items():
        ls = "--" if name == "Mean" else "-"
        ax[0].plot(beta_grid, d["ess_frac_med"], ls=ls, marker="o", ms=3, lw=1.6, color=col[name], label=name)
        ax[0].plot(beta_grid, d["ess_frac_p10"], ls=":", lw=1, color=col[name])
        ax[1].plot(beta_grid, d["kl_med"], ls=ls, marker="o", ms=3, lw=1.6, color=col[name], label=name)
        ax[2].plot(d["kl_med"], d["ess_frac_med"], ls=ls, marker="o", ms=3, lw=1.6, color=col[name], label=name)
    for a in (ax[0], ax[2]):
        a.axhline(ess_min / N, color="#999999", ls="--", lw=1)
        a.text(0.02, ess_min / N * 1.15, f"ess_min={ess_min:g} (config, not locked)", fontsize=8, color="#666666",
               transform=a.get_yaxis_transform())
    ax[1].axhline(np.log(N), color="#999999", ls="--", lw=1)
    ax[0].set(xscale="log", yscale="log", xlabel="beta_policy", ylabel="ESS / N (solid=median, dotted=p10)", title=f"G3: ESS, {title}")
    ax[1].set(xscale="log", xlabel="beta_policy", ylabel="median KL (nats); dashed = log N cap", title="KL vs beta_policy")
    ax[2].set(yscale="log", xlabel="achieved median KL (nats)", ylabel="ESS / N", title="ESS at matched KL")
    ax[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"g3_ess_{title}.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- misc
def synthetic(P=120, N=512, K=5, seed=0):
    """Smoke test only: proxy = gold + shared (common-to-all-members) heavy-tailed error + member noise."""
    rng = np.random.default_rng(seed)
    gold = rng.normal(size=(P, N))
    shared = rng.standard_t(3, size=(P, N)) * 0.6
    member = rng.normal(scale=0.5, size=(P, N, K))
    return dict(R=gold[..., None] + shared[..., None] + member, gold=gold, split="tune",
                group=np.array([f"g{i // 4}" for i in range(P)]), empty_frac=0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", default=[])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--std_mode", choices=["within", "global"], default="within",
                    help="PREREG decision: within = pooled within-prompt std (default), global = std over all cells")
    ap.add_argument("--std_in", default=None, help="std_stats.json from the tune run; REQUIRED to reuse stats on any later split")
    ap.add_argument("--kl_tol", type=float, default=0.15, help="max median |empirical - analytic| KL (nats) for the usable n range")
    ap.add_argument("--n_ref", type=int, default=16, help="fixed reference n for the pre-registrable contrast gold(n_max) - gold(n_ref)")
    ap.add_argument("--betas", type=float, nargs=3, default=None, metavar=("LO", "HI", "NUM"), help="override config beta grid with a geomspace")
    ap.add_argument("--cs", type=float, nargs="+", default=None, help="override config c grid")
    ap.add_argument("--ess_min", type=float, default=None)
    ap.add_argument("--B", type=int, default=None, help="bootstrap reps (default: config stats.bootstrap_reps)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cfg = load_cfg(a.config)
    c_grid = a.cs or cfg["c"]
    beta_grid = np.geomspace(a.betas[0], a.betas[1], int(a.betas[2])) if a.betas else np.array(cfg["beta_policy"], float)
    ess_min = a.ess_min if a.ess_min is not None else cfg["ess_min"]
    B = a.B or cfg["bootstrap_reps"]

    data = {"synthetic": synthetic()} if a.synthetic else {os.path.splitext(os.path.basename(p))[0]: U.load_npz(p) for p in a.npz}
    assert data, "give --npz or --synthetic"
    std_in = json.load(open(a.std_in)) if a.std_in else None
    std_out, summ, curves, invs = {}, {}, {}, {}
    for name, d in data.items():
        if std_in is not None:
            st = std_in[name]
        else:
            st = U.fit_std(d["R"], d["gold"], a.std_mode)
        std_out[name] = st
        R, gold = U.apply_std(d["R"], d["gold"], st)
        N, Pn = R.shape[1], R.shape[0]
        inv = U.group_index(d["group"], Pn)
        n_cand = np.unique(np.geomspace(1, max(2, N // 2), 20).astype(int))
        pr, go, kl = g1_curves(R, gold, n_cand)
        summ[name], curves[name] = g1_summarize(name, pr, go, kl, n_cand, inv, a.kl_tol, a.n_ref, B, d["empty_frac"])
        invs[name] = inv
        g3res = g3(R, beta_grid, c_grid, ess_min)
        plot_g3(g3res, beta_grid, N, ess_min, name, a.out)
        summ[name]["g3"] = dict(beta_grid=beta_grid, ess_min=ess_min, N=N, results=g3res)
    plot_g1(curves, summ, invs, B, a.out)
    U.dump_json(std_out, os.path.join(a.out, "std_stats.json"))
    U.dump_json(summ, os.path.join(a.out, "g1_g3_summary.json"))
    for name, s in summ.items():
        print(f"[{name}] {s['verdict']}")
        print(f"   n_max={s['n_max']} maxKL={s['max_kl_reached']:.2f}  gold peak n={s['gold_peak_n']} "
              f"cross-fit drop={s['crossfit_drop_gold']['mean']:+.3f} CI {s['crossfit_drop_gold']['ci95']}")
        fc = s["fixed_contrast"]
        print(f"   gold(n={fc['n_max']})-gold(n={fc['n_ref']}) = {fc['gold_nmax_minus_nref']['mean']:+.3f} "
              f"CI {fc['gold_nmax_minus_nref']['ci95']}")
        for w in s["warnings"]:
            print("   WARNING:", w)