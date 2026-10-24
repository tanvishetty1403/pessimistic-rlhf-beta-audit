"""Recompute the split fingerprint from the pinned HF revision and diff it against the committed manifest.
Works from any directory. Needs network (downloads the pinned dataset revision).

Usage: python scripts/verify_manifest.py [configs/base.yaml] [--local]
  --local  also hash the split files in data/splits/ (test.json, tune_prompts.json, smoke_ids.txt) against the
           manifest. Reads ids only. Meant for a working copy where make_splits.py has been run."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import json, yaml
from datasets import load_dataset
from src.data import DEFAULT_FRAC, split_pairs, filter_mismatched, split_manifest, local_checks

args = [a for a in sys.argv[1:] if not a.startswith("--")]
cfg = yaml.safe_load(open(args[0] if args else ROOT / "configs" / "base.yaml"))
d = cfg["dataset"]
old = json.load(open(ROOT / "data" / "splits" / "manifest.json"))

ds = load_dataset(d["name"], data_dir=d.get("data_dir"), revision=d.get("revision"), split="train")
pairs = [{"pair_id": f"train-{i}", "chosen": x["chosen"], "rejected": x["rejected"]} for i, x in enumerate(ds)]
pairs, _ = filter_mismatched(pairs)
new = split_manifest(split_pairs(pairs, cfg["seed"]), cfg["seed"], DEFAULT_FRAC, d.get("revision"), d.get("data_dir"))

keys = ("dataset", "data_dir", "revision", "seed", "frac", "n_pairs", "sha256_prompt_ids")
bad = [k for k in keys if new[k] != old[k]]
print("MANIFEST OK" if not bad else f"MANIFEST MISMATCH in: {bad}")

if "--local" in sys.argv:
    res = local_checks(str(ROOT / "data" / "splits"), old)
    for k, v in res.items():
        print(("LOCAL OK       " if v else "LOCAL MISMATCH ") + k)
    bad += [k for k, v in res.items() if not v]
sys.exit(1 if bad else 0)