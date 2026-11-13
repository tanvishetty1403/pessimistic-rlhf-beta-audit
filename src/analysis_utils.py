"""
src/analysis_utils.py

Shared helpers for scripts/g1_g3_plots.py and scripts/g2_f1_calibration.py:
npz loading with a freeze guard, standardization stats (fit on tune, reuse on test), cluster bootstrap.

npz contract:
    R     (P, N, K)  proxy scores per ensemble member
    gold  (P, N)     gold scores
    split str        'tune' | 'rm_train' | ...   (0-d str, or a (P,) str array as build_npz.py writes it;
                     'test' anywhere in it is refused unless allow_test=True)
    group (P,)       first-turn group label, saved as a STR array (not object dtype)
    empty (P, N)     optional bool mask of empty responses (clean3 -> "")
"""
import json
import warnings
import numpy as np


def _split_label(arr):
    """Unique split label(s) as one string ('tune', or 'rm_train,tune' if mixed). Works for a 0-d str and for the
    (P,) str array build_npz.py writes. (str(arr) on a (P,) array gives "['tune' 'tune' ...]", which never equals
    'test', so the freeze guard used to be silently bypassed.)"""
    return ",".join(np.unique(np.asarray(arr).astype(str)).tolist())


def load_npz(path, allow_test=False):
    d = np.load(path, allow_pickle=False)
    for k in ("R", "gold", "split"):
        if k not in d.files:
            raise KeyError(f"{path}: missing field '{k}'")
    split = _split_label(d["split"])
    if "test" in split.split(",") and not allow_test:
        raise PermissionError(f"{path}: split contains 'test', frozen until PREREG sign-off (final_eval only).")
    R, gold = d["R"].astype(np.float64), d["gold"].astype(np.float64)
    if R.ndim != 3 or gold.shape != R.shape[:2] or R.shape[2] < 2:
        raise ValueError(f"{path}: expected R (P,N,K>=2) and gold (P,N); got {R.shape}, {gold.shape}")
    if not (np.isfinite(R).all() and np.isfinite(gold).all()):
        raise ValueError(f"{path}: NaN/inf in R or gold (check fp16 overflow in scoring)")
    group = d["group"] if "group" in d.files else None
    if group is not None and len(group) != R.shape[0]:
        raise ValueError(f"{path}: group has {len(group)} entries, R has {R.shape[0]} prompts")
    ef = None
    emp = None
    if "empty" in d.files:
        emp = d["empty"].astype(bool)
        if emp.shape != gold.shape:
            raise ValueError(f"{path}: empty mask {emp.shape} != gold {gold.shape}")
        ef = float(emp.mean())
        if ef > 0:
            warnings.warn(f"{path}: {ef:.2%} empty responses are included; decide skip vs keep before go/no-go")
    else:
        warnings.warn(f"{path}: no 'empty' mask, cannot report empty-response rate")
    return dict(R=R, gold=gold, split=split, group=group, empty_frac=ef, empty=emp)


def exclude_prompt(d, idx):
    """Drop prompt `idx` from every per-prompt array present in d (R, gold, empty, group, per-prompt split).
    None-safe: load_npz always returns an 'empty' key, but its value is None when the npz has no mask.
    Raises on a bad index (the old inline code silently ignored an out-of-range index)."""
    P = d["R"].shape[0]
    if not -P <= idx < P:
        raise IndexError(f"--exclude_prompt {idx} out of range for {P} prompts")
    keep = np.ones(P, bool)
    keep[idx] = False
    out = dict(d)
    for k in ("R", "gold", "empty", "group"):
        if out.get(k) is not None:
            out[k] = out[k][keep]
    return out


def drop_empty(d, min_frac=0.5):
    """Remove empty responses by RESAMPLING THE CANDIDATE POOL, not by editing scores.

    Each kept prompt keeps its first N' non-empty samples (samples are iid, so this is unbiased), with
    N' = min over kept prompts of the non-empty count. R, gold and the reference (n=1) all describe the
    non-empty response distribution, KL is measured against it, and standardization is fit on it.
    Prompts with fewer than min_frac * N non-empty samples are removed (listed in drop_info) so one mostly-empty
    prompt cannot shrink N for everyone; raise min_frac to be stricter.

    (The previous implementation set R[empty] = -1e9 BEFORE fit_std: the 1e9 values went into the mean and std,
    so r_scale ~ 1e8 and every standardized proxy score collapsed toward 0; the reference mean at n=1 still
    contained the empties; and an npz with no mask made R[None] = -1e9 overwrite everything.)"""
    emp = d.get("empty")
    if emp is None:
        raise ValueError("--drop_empty needs an 'empty' mask in the npz; this file has none")
    R, gold = d["R"], d["gold"]
    P, N = gold.shape
    valid = ~emp
    cnt = valid.sum(1)
    keep = cnt >= max(2, int(np.ceil(min_frac * N)))
    if not keep.any():
        raise ValueError(f"no prompt has >= {min_frac:.0%} non-empty samples")
    rows = np.flatnonzero(keep)
    Np = int(cnt[keep].min())
    idx = np.stack([np.flatnonzero(valid[p])[:Np] for p in rows])          # (P', Np), original order kept
    out = dict(d)
    out["R"], out["gold"] = R[rows[:, None], idx], gold[rows[:, None], idx]
    out["empty"] = np.zeros(idx.shape, bool)
    out["empty_frac"] = 0.0
    if out.get("group") is not None:
        out["group"] = out["group"][rows]
    out["drop_info"] = dict(n_before=int(N), n_after=Np, prompts_before=int(P), prompts_after=int(len(rows)),
                            prompts_removed=[int(i) for i in np.flatnonzero(~keep)],
                            empty_frac_before=float(emp.mean()), min_frac=min_frac,
                            nonempty_samples_discarded=int(cnt[keep].sum() - Np * len(rows)))
    if out["drop_info"]["prompts_removed"]:
        warnings.warn(f"drop_empty removed prompts {out['drop_info']['prompts_removed']} (< {min_frac:.0%} non-empty); "
                      f"indices refer to the array BEFORE this call")
    return out


def group_index(group, n=None):
    """Compact 0..G-1 cluster index per prompt. group=None falls back to per-prompt clusters (optimistic)."""
    if group is None:
        warnings.warn("no 'group' field: bootstrapping over prompts, not first-turn groups (CIs too narrow)")
        return np.arange(n)
    _, inv = np.unique(np.asarray(group), return_inverse=True)
    return inv.reshape(-1)


# ---- standardization -------------------------------------------------------------------------------
def fit_std(R, gold, mode="within", center_members=False):
    """mode='within': pooled within-prompt std (what BoN / tilting actually act on).
    mode='global': std over all (prompt, sample) cells (dominated by between-prompt variance).
    One scale for all K members (keeps ensemble variance structure); gold has its own scale.

    center_members=True also stores each member's global mean score ('member_offsets', fit on THIS data) and
    apply_std subtracts it. A Bradley-Terry reward model is identified only up to an additive constant, so every
    member carries an arbitrary offset; on the SFT-pilot gold-relabel ensembles ~97% of the raw ensemble variance
    was that offset, not disagreement. Offsets are saved in std_stats.json so later splits reuse the tune-fit values.
    Default False = the raw-variance pipeline as originally built."""
    off = R.mean(axis=(0, 1)) if center_members else None
    Rc = R - off if center_members else R
    mu = Rc.mean(axis=-1)
    if mode == "global":
        s_r, s_g = mu.std(), gold.std()
    elif mode == "within":
        s_r = np.sqrt(mu.var(axis=1, ddof=1).mean())
        s_g = np.sqrt(gold.var(axis=1, ddof=1).mean())
    else:
        raise ValueError(mode)
    st = dict(mode=mode, r_center=float(mu.mean()), r_scale=float(s_r),
              g_center=float(gold.mean()), g_scale=float(s_g))
    if center_members:
        st["member_offsets"] = [float(x) for x in off]
    return st


def apply_std(R, gold, st):
    off = st.get("member_offsets")
    if off is not None:
        off = np.asarray(off, dtype=np.float64)
        if off.shape != (R.shape[-1],):
            raise ValueError(f"member_offsets has {off.shape[0]} entries, R has K={R.shape[-1]}")
        R = R - off
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