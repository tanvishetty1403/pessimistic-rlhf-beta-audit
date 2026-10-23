import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import collections, yaml
from datasets import load_dataset
from src.data import extract_prompt, assign_split, filter_mismatched

def first_turn(pr):
    return pr.split("\n\nAssistant:")[0].replace("\n\nHuman:", "", 1).strip()

cfg = yaml.safe_load(open(sys.argv[1])); d = cfg["dataset"]
ds = load_dataset(d["name"], data_dir=d.get("data_dir"), revision=d.get("revision"), split="train")
pairs, _ = filter_mismatched([{"chosen": x["chosen"], "rejected": x["rejected"]} for x in ds])
prompts = [extract_prompt(p["chosen"]) for p in pairs]
groups = [first_turn(pr) for pr in prompts]

for label, keys in [("OLD: hash of full prompt", prompts), ("NEW: hash of first turn", groups)]:
    where = collections.defaultdict(set)
    for g, k in zip(groups, keys):
        where[g].add(assign_split(k, cfg["seed"]))
    leaked = {g for g, v in where.items() if len(v) > 1}
    n_pairs = sum(g in leaked for g in groups)
    print(f"{label}: {len(leaked)} of {len(where)} first turns span >1 split, covering {n_pairs} of {len(pairs)} pairs")

cnt = collections.Counter(groups)
print("most common first turns:")
for g, c in cnt.most_common(5):
    print(c, repr(g[:80]))
    