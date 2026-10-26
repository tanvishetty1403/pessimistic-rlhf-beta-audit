"""Train a K-member LoRA reward-model ensemble (one member after another, resumable). Run from the repo root.

  python scripts/train_rm.py --out checkpoints/human_noise25_K5 --n-train 2000 --n-val 1000 --members 5

Training pairs are the first --n-train pairs of --train-file (default rm_train_noisy.json: labels already
swapped for the noisy fraction). Validation pairs come from --val-file (default rm_train_clean.json, human
labels, no noise) and exclude every training pair by pair_id, so rm_train_pairs in the config must be at least
n_train + n_val. Use a NEW --out per configuration: a finished member is skipped on rerun, which is what makes
a Colab disconnect cheap, and config.json stops you from mixing settings by accident.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import hashlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch

from src import rm as R
from src.data import load_split


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="EleutherAI/pythia-410m")
ap.add_argument("--train-file", default="data/splits/rm_train_noisy.json")
ap.add_argument("--val-file", default="data/splits/rm_train_clean.json")
ap.add_argument("--n-train", type=int, default=2000)
ap.add_argument("--n-val", type=int, default=1000)
ap.add_argument("--members", type=int, default=5)
ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--diversity", choices=["seeds", "bootstrap"], default="seeds")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--max-length", type=int, default=512)
ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--lora-alpha", type=int, default=32)
ap.add_argument("--ensemble-id", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--hf-repo", default=None, help="push each finished member to this private HF model repo")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
train_all = load_split(args.train_file, "rm_train")
train = train_all[: args.n_train]
if len(train) < args.n_train:
    sys.exit(f"{args.train_file} has only {len(train_all)} pairs, need {args.n_train}")
train_ids = {p["pair_id"] for p in train}
val = [p for p in load_split(args.val_file, "rm_train") if p["pair_id"] not in train_ids][: args.n_val]
if not val:
    sys.exit("no validation pairs left: raise rm_train_pairs in the config and rerun make_splits.py")
if len(val) < args.n_val:
    print(f"WARNING: only {len(val)} validation pairs available (wanted {args.n_val})")

# Hash the pair_ids to ensure noise_on and noise_off align
train_hash = hashlib.sha256(",".join([str(p["pair_id"]) for p in train]).encode()).hexdigest()[:12]
val_hash = hashlib.sha256(",".join([str(p["pair_id"]) for p in val]).encode()).hexdigest()[:12]

label_source = "gold" if "gold_model" in train[0] else "human"
noise = float(train[0].get("noise_rate", 0.0))
ensemble_id = args.ensemble_id or (
    f"{label_source}_noise{int(round(noise * 100))}_K{args.members}_{args.diversity}_s{args.seed0}")

out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
cfg = {k: v for k, v in vars(args).items() if k != "hf_repo"}
if (out / "config.json").exists() and json.load(open(out / "config.json")) != cfg:
    sys.exit(f"{out}/config.json differs from these arguments: use a new --out, or delete the old folder")
json.dump(cfg, open(out / "config.json", "w"), indent=2)

try:
    git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT).stdout.strip()
except Exception:
    git_hash = ""
meta = {"base": args.model, "max_length": args.max_length, "label_source": label_source, "noise_rate": noise,
        "train_file": args.train_file, "val_file": args.val_file, "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha, "git_hash": git_hash,
        "train_pair_ids_hash": train_hash, "val_pair_ids_hash": val_hash}

# D's review item 7: check that validation matches training label source
if ("gold_model" in val[0]) != ("gold_model" in train[0]):
    sys.exit("ERROR: train and val have different label sources. Fix your files.")

on_done = None
if args.hf_repo:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.hf_repo, repo_type="model", private=True, exist_ok=True)

    def on_done(mdir):
        api.upload_folder(folder_path=str(mdir), path_in_repo=f"{ensemble_id}/{mdir.name}",
                          repo_id=args.hf_repo, repo_type="model")

print(f"{ensemble_id}: {len(train)} train pairs ({label_source} labels, noise {noise}), {len(val)} val pairs, "
      f"{args.members} members, device {device}")
tok = R.load_tokenizer(args.model)
build = lambda seed: R.build_rm(args.model, tok, seed, device, r=args.lora_r, alpha=args.lora_alpha)
summary = R.train_ensemble(train, val, tok, build, out, members=args.members, seed0=args.seed0,
                           diversity=args.diversity, ensemble_id=ensemble_id, meta=meta, device=device,
                           on_member_done=on_done, epochs=args.epochs, bs=args.bs, accum=args.accum,
                           lr=args.lr, max_length=args.max_length)
print(json.dumps(summary, indent=2))
if args.hf_repo:
    api.upload_file(path_or_fileobj=str(out / "ensemble.json"), path_in_repo=f"{ensemble_id}/ensemble.json",
                    repo_id=args.hf_repo, repo_type="model")