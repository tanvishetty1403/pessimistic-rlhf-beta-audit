"""Sample the reference policy on the tune prompts and write the cache as resumable JSONL shards. Run from the repo root.

  python scripts/sample_policy.py --out samples/N256_pythia410m --n 256 --n-prompts 200 --hf-repo afml-rlhf/sample-cache

Pure sampling (temperature 1, top_k 0, top_p 1), max_new_tokens 128, seed-deterministic per prompt and chunk.
A finished shard is skipped on rerun. --restore first downloads missing shards from the Hub (use it after Colab
recycled the machine). Use a NEW --out per configuration (config.json guards against mixing settings).
Outputs: shard_XXXX.jsonl, samples.jsonl (merged), meta.json (settings, rates, skipped prompts, provenance).
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch

from src import sampling as S
from src.data import load_split

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="EleutherAI/pythia-410m")
ap.add_argument("--prompts", default="data/splits/tune_prompts.json")
ap.add_argument("--n-prompts", type=int, default=200)
ap.add_argument("--n", type=int, default=256, help="responses per prompt")
ap.add_argument("--chunk", type=int, default=128, help="draws per generate() call (halved on OOM)")
ap.add_argument("--max-new-tokens", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard-prompts", type=int, default=10)
ap.add_argument("--out", required=True)
ap.add_argument("--hf-repo", default=None, help="private DATASET repo to push finished shards to")
ap.add_argument("--restore", action="store_true", help="download shards missing locally from --hf-repo first")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
cfg = {k: v for k, v in vars(args).items() if k not in ("hf_repo", "restore")}
if (out / "config.json").exists() and json.load(open(out / "config.json")) != cfg:
    sys.exit(f"{out}/config.json differs from these arguments: use a new --out, or delete the old folder")
json.dump(cfg, open(out / "config.json", "w"), indent=2)

prompts = load_split(args.prompts, "tune")[: args.n_prompts]
run_id = out.name
api = None
if args.hf_repo:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    api.create_repo(args.hf_repo, repo_type="dataset", private=True, exist_ok=True)
    if args.restore:
        for f in api.list_repo_files(args.hf_repo, repo_type="dataset"):
            if f.startswith(f"{run_id}/shard_") and not (out / pathlib.Path(f).name).exists():
                shutil.copy(hf_hub_download(args.hf_repo, f, repo_type="dataset"), out / pathlib.Path(f).name)
                print("restored", f)

    def on_done(path):
        api.upload_file(path_or_fileobj=str(path), path_in_repo=f"{run_id}/{path.name}",
                        repo_id=args.hf_repo, repo_type="dataset")
else:
    on_done = None

print(f"{run_id}: {len(prompts)} prompts x N={args.n}, chunk {args.chunk}, max_new_tokens {args.max_new_tokens}, device {device}")
model, tok = S.load_policy(args.model, device)
info = S.run_sampling(model, tok, prompts, out, n=args.n, chunk=args.chunk, max_new_tokens=args.max_new_tokens,
                      seed=args.seed, shard_prompts=args.shard_prompts, device=device, on_shard_done=on_done)

merged = out / "samples.jsonl"
n_rec = S.merge_shards(out, merged)
records = [json.loads(line) for line in open(merged)]
r = S.rates(records)
over512 = sum((rec["prompt_tokens"] + x["n_tokens"]) > 512 for rec in records for x in rec["responses"]) / max(1, r["responses"])
try:
    git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
except Exception:
    git_hash = ""
try:
    man = json.load(open(ROOT / "data" / "splits" / "manifest.json"))
    tune_sha = man.get("sha256_tune_prompts_selected")
except Exception:
    tune_sha = None
import transformers
meta = {"run_id": run_id, "policy": args.model, "dtype": "float16" if device == "cuda" else "float32",
        "temperature": 1.0, "top_k": 0, "top_p": 1.0, "max_new_tokens": args.max_new_tokens, "n_per_prompt": args.n,
        "chunk": args.chunk, "seed": args.seed, "prompts_file": args.prompts, "n_prompts_requested": len(prompts),
        "n_prompts_sampled": n_rec, "skipped_prompt_ids": info["skipped"], "rates": r,
        "frac_over_512_tokens_prompt_plus_response": over512, "seconds_this_session": info["seconds"],
        "git_hash": git_hash, "manifest_tune_selected_sha256": tune_sha,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "clean_rule": "clean3 (src/data.py)", "response_format": "see src/sampling.py docstring"}
json.dump(meta, open(out / "meta.json", "w"), indent=2)
print(json.dumps({k: meta[k] for k in ("n_prompts_sampled", "skipped_prompt_ids", "rates",
                                         "frac_over_512_tokens_prompt_plus_response")}, indent=2))
if api:
    for name in ("samples.jsonl", "meta.json"):
        api.upload_file(path_or_fileobj=str(out / name), path_in_repo=f"{run_id}/{name}",
                        repo_id=args.hf_repo, repo_type="dataset")