"""
src/analysis_utils.py

Shared helpers for scripts/g1_g3_plots.py and scripts/g2_f1_calibration.py:
npz loading with a freeze guard, standardization stats (fit on tune, reuse on test), cluster bootstrap.

npz contract:
    R     (P, N, K)  proxy scores per ensemble member
    gold  (P, N)     gold scores
    split str        'tune' | 'rm_train' | ...   ('test' is refused unless allow_test=True)
    group (P,)       first-turn group label, saved as a STR array (not object dtype)
    empty (P, N)     optional bool mask of empty responses (clean3 -> "")
"""
import json
import warnings
import numpy as np


def load_npz(path, allow_test=False):
    d = np.load(path, allow_pickle=False)
    for k in ("R", "gold", "split"):
        if k not in d.files:
            raise KeyError(f"{path}: missing field '{k}'")
    split = str(d["split"])
    if split == "test" and not allow_test:
        raise PermissionError(f"{path}: split='test' is frozen until PREREG sign-off (final_eval only).")
    R, gold = d["R"].astype(np.float64), d["gold"].astype(np.float64)
    if R.ndim != 3 or gold.shape != R.shape[:2] or R.shape[2] < 2:
        raise ValueError(f"{path}: expected R (P,N,K>=2) and gold (P,N); got {R.shape}, {gold.shape}")
    if not (np.isfinite(R).all() and np.isfinite(gold).all()):
        raise ValueError(f"{path}: NaN/inf in R or gold (check fp16 overflow in scoring)")
    group = d["group"] if "group" in d.files else None
    if group is not None and len(group) != R.shape[0]:
        raise ValueError(f"{path}: group has {len(group)} entries, R has {R.shape[0]} prompts")
    ef = None
    if "empty" in d.files:
        ef = float(d["empty"].astype(bool).mean())
        if ef > 0:
            warnings.warn(f"{path}: {ef:.2%} empty responses are included; decide skip vs keep before go/no-go")
    else:
        warnings.warn(f"{path}: no 'empty' mask, cannot report empty-response rate")
    return dict(R=R, gold=gold, split=split, group=group, empty_frac=ef)


def group_index(group, n=None):
    """Compact 0..G-1 cluster index per prompt. group=None falls back to per-prompt clusters (optimistic)."""
    if group is None:
        warnings.warn("no 'group' field: bootstrapping over prompts, not first-turn groups (CIs too narrow)")
        return np.arange(n)
    _, inv = np.unique(np.asarray(group), return_inverse=True)
    return inv.reshape(-1)


# ---- standardization -------------------------------------------------------------------------------
def fit_std(R, gold, mode="within"):
    """mode='within': pooled within-prompt std (what BoN / tilting actually act on).
    mode='global': std over all (prompt, sample) cells (dominated by between-prompt variance).
    One scale for all K members (keeps ensemble variance structure); gold has its own scale."""
    mu = R.mean(axis=-1)
    if mode == "global":
        s_r, s_g = mu.std(), gold.std()
    elif mode == "within":
        s_r = np.sqrt(mu.var(axis=1, ddof=1).mean())
        s_g = np.sqrt(gold.var(axis=1, ddof=1).mean())
    else:
        raise ValueError(mode)
    return dict(mode=mode, r_center=float(mu.mean()), r_scale=float(s_r),
                g_center=float(gold.mean()), g_scale=float(s_g))


def apply_std(R, gold, st):
    return (R - st["r_center"]) / st["r_scale"], (gold - st["g_center"]) / st["g_scale"]


# ---- cluster bootstrap -----------------------------------------------------------------------------
def boot_weights(inv, B, seed=0, chunk=500):
    """Yield (b, P) prompt-weight matrices from resampling GROUPS with replacement."""
    G = int(inv.max()) + 1
    rng = np.random.default_rng(seed)
    done = 0
    while done < B:
        b = min(chunk, B - done)
        cnt = rng.multinomial(G, np.full(G, 1.0 / G), size=b)      # (b, G)
        yield cnt[:, inv].astype(np.float64)
        done += b


def cluster_boot_mean(x, inv, B=2000, seed=0):
    """x: (P,) or (P, m) per-prompt values. Returns (mean, lo95, hi95), each (m,). Resamples groups."""
    x = np.asarray(x, dtype=np.float64)
    x = x[:, None] if x.ndim == 1 else x
    bm = np.vstack([W @ x / W.sum(1, keepdims=True) for W in boot_weights(inv, B, seed)])
    return x.mean(0), np.percentile(bm, 2.5, axis=0), np.percentile(bm, 97.5, axis=0)


def dump_json(obj, path):
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=default)