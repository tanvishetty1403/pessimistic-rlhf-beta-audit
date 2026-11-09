"""
scripts/build_npz.py

Join proxy scores and the gold scores BY ID (never by file order) into the (R, gold) npz
that g1_g3_plots.py reads.

  proxy npz: R (P, N, K) float, prompt_ids (P,), response_ids (P, N)
  gold  npz: gold (P, N) float, empty (P, N) bool, prompt_ids (P,), response_ids (P, N)

Output keys: R, gold, empty, prompt_ids, response_ids, split, group (in the PROXY file's order).
Usage: python scripts/build_npz.py --proxy proxy.npz --gold gold.npz --out noise_on.npz
"""
import argparse
import numpy as np


def main(proxy, gold, out):
    pz, gz = np.load(proxy, allow_pickle=False), np.load(gold, allow_pickle=False)
    R, pp, pr = pz["R"], pz["prompt_ids"], pz["response_ids"]
    g, e, gp, gr = gz["gold"], gz["empty"], gz["prompt_ids"], gz["response_ids"]
    
    assert R.ndim == 3 and R.shape[:2] == pr.shape == (len(pp), pr.shape[1]), "bad proxy shapes"
    assert g.shape == e.shape == gr.shape, "bad gold shapes"
    assert len(set(pp)) == len(pp) and len(set(gp)) == len(gp), "duplicate prompt ids"
    assert len(set(pr.ravel())) == pr.size and len(set(gr.ravel())) == gr.size, "duplicate response ids"
    assert set(pp) == set(gp), f"prompt id sets differ: {set(pp) ^ set(gp)}"
    assert set(pr.ravel()) == set(gr.ravel()), "response id sets differ"
    assert np.isfinite(g).all(), "non-finite gold scores"
    
    bad = ~np.isfinite(R).all(-1)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} responses have no/non-finite proxy score (NaN rows). They cannot be ranked in "
                         "BoN and dropping them makes N ragged. Score them too (empty text is a valid input).")

    idx = {r: (i, j) for i, row in enumerate(gr) for j, r in enumerate(row)}
    gold_a, empty_a = np.empty(pr.shape, g.dtype), np.empty(pr.shape, bool)
    
    for i, row in enumerate(pr):
        for j, r in enumerate(row):
            gi, gj = idx[r]
            assert gp[gi] == pp[i], f"{r} sits under different prompts in the two files"
            gold_a[i, j], empty_a[i, j] = g[gi, gj], e[gi, gj]

    # FIX: Add 'split' and 'group' fields required by analysis_utils
    # We assume this is the tune split, so we set it to 'tune'
    split_a = np.array(["tune"] * len(pp))
    # Use prompt_ids as the group for bootstrapping (this prevents the warning about CIs being too narrow)
    group_a = np.array(pp)
    
    np.savez(out, R=R, gold=gold_a, empty=empty_a, prompt_ids=pp, response_ids=pr, split=split_a, group=group_a)
    print(f"wrote {out}: R{R.shape} gold{gold_a.shape} empties={int(empty_a.sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.proxy, a.gold, a.out)