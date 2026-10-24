"""Recompute the split fingerprint from the pinned HF revision and diff it against the committed manifest.
Needs network. Usage: python scripts/verify_manifest.py configs/base.yaml"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json, yaml
from datasets import load_dataset
from src.data import DEFAULT_FRAC, split_pairs, filter_mismatched, split_manifest

cfg = yaml.safe_load(open(sys.argv[1])); d = cfg["dataset"]
ds = load_dataset(d["name"], data_dir=d.get("data_dir"), revision=d.get("revision"), split="train")
pairs = [{"pair_id": f"train-{i}", "chosen": x["chosen"], "rejected": x["rejected"]} for i, x in enumerate(ds)]
pairs, _ = filter_mismatched(pairs)
new = split_manifest(split_pairs(pairs, cfg["seed"]), cfg["seed"], DEFAULT_FRAC, d.get("revision"), d.get("data_dir"))
old = json.load(open("data/splits/manifest.json"))
bad = [k for k in ("revision", "seed", "frac", "n_pairs", "sha256_prompt_ids") if new[k] != old[k]]
print("MANIFEST OK" if not bad else f"MANIFEST MISMATCH in: {bad}")
sys.exit(1 if bad else 0)