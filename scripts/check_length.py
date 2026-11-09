"""
scripts/check_length.py

reads the .npz and the samples.jsonl to calculate the exact Spearman correlations

"""

import argparse
import json
import numpy as np
from scipy.stats import spearmanr
import pathlib
import sys

def main(npz_path, samples_path):
    d = np.load(npz_path, allow_pickle=True)
    gold = d["gold"]
    E = d["empty"].astype(bool) if "empty" in d.files else np.zeros(gold.shape, bool)
    recs = {r["prompt_id"]: r for r in (json.loads(l) for l in open(samples_path) if l.strip())}
    tok, chs = np.zeros(gold.shape), np.zeros(gold.shape)
    for i, pid in enumerate(d["prompt_ids"]):
        by_rid = {x["response_id"]: x for x in recs[str(pid)]["responses"]}
        for j, rid in enumerate(d["response_ids"][i]):
            x = by_rid[str(rid)]; tok[i, j] = x["n_tokens"]; chs[i, j] = len(x["clean"])
    print("--- Length check (criterion 3), Spearman with gold ---")
    for nm, L in (("tokens", tok), ("chars", chs)):
        for lab, M in (("empties incl.", np.ones_like(E)), ("empties excl.", ~E)):
            pooled = spearmanr(gold[M], L[M]).correlation
            w = [spearmanr(gold[i][M[i]], L[i][M[i]]).correlation for i in range(len(gold)) if M[i].sum() > 10]
            print(f"{nm:6s} {lab}: pooled {pooled:+.4f} | within-prompt mean {np.mean(w):+.4f}, {sum(x < 0 for x in w)}/{len(w)} negative")
    print("PRE-REG gate variant = tokens, pooled, empties incl.")
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--samples", default="data/sft_pilot20/samples.jsonl")
    a = ap.parse_args()
    main(a.npz, a.samples)