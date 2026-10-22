"""Download HH-RLHF, split by prompt, write JSON per split. Usage: python scripts/make_splits.py configs/base.yaml
Run on Colab/with network. Writes data/splits/{rm_train,tune,test}.json (gitignored)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import os, sys, yaml
from datasets import load_dataset
from src.data import split_pairs, inject_label_noise, save_split

cfg = yaml.safe_load(open(sys.argv[1]))
d = cfg["dataset"]
ds = load_dataset(d["name"], split="train")      # OPEN: restrict to helpful-base via data_dir if decided
pairs = [{"chosen": x["chosen"], "rejected": x["rejected"]} for x in ds]
sp = split_pairs(pairs, cfg["seed"])
os.makedirs("data/splits", exist_ok=True)

rm_train = sp["rm_train"][: d["rm_train_pairs"]]
noisy, _ = inject_label_noise(rm_train, d["label_noise"], cfg["seed"])
save_split(noisy, "data/splits/rm_train_noisy.json")
save_split(rm_train, "data/splits/rm_train_clean.json")
# tune prompts for sampling (unique prompts only)
tune_prompts = sorted({p["prompt"] for p in sp["tune"]})[: d["n_tune_prompts"]]
save_split(tune_prompts, "data/splits/tune_prompts.json")
save_split(sp["test"], "data/splits/test.json")   # written but never loaded before freeze
print({k: len(v) for k, v in sp.items()}, "tune prompts:", len(tune_prompts))
