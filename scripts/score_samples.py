"""Score cached policy samples with every member of a trained ensemble (one member loaded at a time).

  python scripts/score_samples.py --ensemble-dir checkpoints/human_noise25_K5 --samples data/samples_tiny.json --out scores.jsonl

Options: --npz out.npz also writes R (P, N, K), prompt_ids (P,), response_ids (P, N) in file order. Empty
responses are now scored as `prompt + ' '` (no NaNs in R). --untrained K scores with K freshly initialised
heads (no training) to check formats only.
Input: the samples format {"prompts": [{prompt_id, prompt, responses: [{response_id, clean, empty, ...}]}]}
(a bare list of those records, or a .jsonl with one record per line, also works). The `clean` text is scored.
Output (JSONL, one row per scored response): prompt_id, response_id, ensemble_id, member_scores[K], mu, var
(unbiased across members). For trained ensembles, base model and max_length come from ensemble.json unless
overridden.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch

from src import rm as R

ap = argparse.ArgumentParser()
ap.add_argument("--ensemble-dir", default=None)
ap.add_argument("--untrained", type=int, default=0, help="score with K untrained members (format check only)")
ap.add_argument("--npz", default=None)
ap.add_argument("--samples", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--base", default=None)
ap.add_argument("--max-length", type=int, default=None)
args = ap.parse_args()

if args.untrained:
    base = args.base or "EleutherAI/pythia-410m"
    max_length = args.max_length or 512
    ensemble_id = f"untrained_K{args.untrained}"
    member_dirs = None
else:
    if not args.ensemble_dir:
        sys.exit("give --ensemble-dir (trained members) or --untrained K")
    ens = pathlib.Path(args.ensemble_dir)
    info = json.load(open(ens / "ensemble.json"))
    base = args.base or info["base"]
    max_length = args.max_length or info["max_length"]
    ensemble_id = info["ensemble_id"]
    member_dirs = [ens / f"member_{k}" for k in range(info["members"])]
    missing = [str(d) for d in member_dirs if not (d / "adapter_config.json").exists()]
    if missing:
        sys.exit(f"missing members: {missing}")

if args.samples.endswith(".jsonl"):
    records = [json.loads(line) for line in open(args.samples) if line.strip()]
else:
    data = json.load(open(args.samples))
    records = data["prompts"] if isinstance(data, dict) else data
n_resp = sum(len(r["responses"]) for r in records)

# This now collects ALL responses, including empties.
texts, keys = R.collect_texts(records)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = R.load_tokenizer(base)

# --- D's over-512 token check ---
lengths = [len(tok(t, truncation=False).input_ids) for t in texts]
n_over = sum(length > 512 for length in lengths)
share_over_512 = n_over / max(len(lengths), 1)
print(f"share of texts over 512 tokens: {share_over_512:.1%} ({n_over}/{len(lengths)})")
# --------------------------------

if args.untrained:
    loaders = [(lambda s=s: R.build_rm(base, tok, s, device)) for s in range(args.untrained)]
    ms = R.score_with(texts, loaders, tok, max_length=max_length, batch_size=args.batch_size, device=device)
else:
    ms = R.score_ensemble(texts, member_dirs, tok, base, max_length=max_length, batch_size=args.batch_size,
                          device=device)

rows = R.score_rows(keys, ms, ensemble_id)
with open(args.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"scored {len(rows)} of {n_resp} responses with {ms.shape[0]} members, max_length {max_length}; wrote {args.out}")


lengths = [len(tok(t, truncation=False).input_ids) for t in texts]
n_over = sum(length > 512 for length in lengths)
share_over_512 = n_over / max(len(lengths), 1)
print(f"share of texts over 512 tokens: {share_over_512:.1%} ({n_over}/{len(lengths)})")


if args.npz:
    import numpy as np
    import pathlib
    import hashlib
    import subprocess
    
    # 1. Write NPZ FIRST
    Rr, pids, rids = R.scores_tensor(records, ms, keys)
    np.savez(args.npz, R=Rr, prompt_ids=pids, response_ids=rids)
    print(f"wrote {args.npz}: R {Rr.shape} (P, N, K), prompt_ids {pids.shape}, response_ids {rids.shape}")

    # 2. Safe git hash (with cwd and try/except)
    try:
        git_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        git_hash = ""
        
    # 3. Samples hash
    with open(args.samples, "rb") as f:
        samples_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    # 4. Meta.json (using pathlib)
    meta = {
        "ensemble_id": ensemble_id,
        "base": base,
        "max_length": max_length,
        "share_over_512": float(share_over_512),
        "n_over_512": n_over,
        "P": len(records),
        "N": len(records[0]["responses"]),
        "K": ms.shape[0],
        "git_hash": git_hash,
        "samples_sha256": samples_hash
    }
    meta_path = pathlib.Path(args.npz).with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote meta to {meta_path}")