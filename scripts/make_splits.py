"""Download HH-RLHF, split by first turn, write JSON per split. Usage: python scripts/make_splits.py configs/base.yaml
Writes data/splits/*.json (gitignored) plus smoke_ids.txt and manifest.json (committed)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import json, os, yaml
from datasets import load_dataset
from src.data import (DEFAULT_FRAC, split_pairs, inject_label_noise, save_split, filter_mismatched,
                      order_by_hash, relabel_with_gold, split_manifest)

cfg = yaml.safe_load(open(sys.argv[1]))
d = cfg["dataset"]
ds = load_dataset(d["name"], data_dir=d.get("data_dir"), revision=d.get("revision"), split="train")
print("rows loaded:", len(ds))                     # helpful-base should be 43,835
pairs = [{"pair_id": f"train-{i}", "chosen": x["chosen"], "rejected": x["rejected"]}
         for i, x in enumerate(ds)]
pairs, n_bad = filter_mismatched(pairs)
print("dropped mismatched-prompt pairs:", n_bad)
sp = split_pairs(pairs, cfg["seed"])
os.makedirs("data/splits", exist_ok=True)

rm_train = order_by_hash(sp["rm_train"], cfg["seed"], key=lambda p: p["pair_id"])[: d["rm_train_pairs"]]
smoke_ids = [p["pair_id"] for p in rm_train[: d["smoke_n"]]]
open("data/splits/smoke_ids.txt", "w").write("\n".join(smoke_ids))

if d.get("gold_pairs"):                            # C's JSONL: pair_id, gold_chosen, gold_rejected, gold_model
    gold = {}
    for line in open(d["gold_pairs"]):
        r = json.loads(line); gold[r["pair_id"]] = r
    n_in = len(rm_train)
    rm_train = relabel_with_gold(rm_train, gold)
    save_split(rm_train, "data/splits/rm_train_gold_relabeled.json")
    agree = sum(p["gold_agrees_with_human"] for p in rm_train) / len(rm_train)
    print(f"gold agrees with the human label on {agree:.1%} of {len(rm_train)} scored, non-tied pairs (of {n_in})")

noisy, _ = inject_label_noise(rm_train, d["label_noise"], cfg["seed"])
save_split(noisy, "data/splits/rm_train_noisy.json")
save_split(rm_train, "data/splits/rm_train_clean.json")

tune_prompts = order_by_hash(sorted({p["prompt"] for p in sp["tune"]}), cfg["seed"],
                             key=lambda s: s)[: d["n_tune_prompts"]]
save_split(tune_prompts, "data/splits/tune_prompts.json")
save_split(sp["test"], "data/splits/test.json")    # written but never loaded before freeze

man = split_manifest(sp, cfg["seed"], DEFAULT_FRAC, d.get("revision"), d.get("data_dir"),
                     smoke_ids=smoke_ids, tune_prompts=tune_prompts)
json.dump(man, open("data/splits/manifest.json", "w"), indent=2)
print({k: len(v) for k, v in sp.items()}, "tune prompts:", len(tune_prompts))