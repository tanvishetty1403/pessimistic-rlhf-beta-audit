"""scripts/make_b_pool.py

python scripts/make_b_pool.py --pairs data/splits/rm_train_clean.json --val val_gold/gold_val.jsonl \
   --gold val_gold/gold_val.jsonl smoke_gold.jsonl --n 10000 --out_pool data/splits/b_pool.json --out_todo b_todo.jsonl

Pool = first n pairs (file order) of rm_train, minus val pairs AND any pair sharing a first-turn group with a val pair.

Never reads test.json: the split is recomputed with assign_split(first_turn(prompt), seed).
out_todo = pool pairs with no gold score yet (this is what Tanvi scores). Ties are NOT dropped here (raw scores kept);
relabel_with_gold drops them later (rule: tie = no preference = dropped, ~0.3%).
"""

import argparse, hashlib, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import data as D

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True); ap.add_argument("--val", required=True)
ap.add_argument("--gold", nargs="*", default=[]); ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out_pool", required=True); ap.add_argument("--out_todo", required=True)
ap.add_argument("--exclude", nargs="*", default=[], help="sft_v1_pairs.json etc: dict with 'train'/'val' pair_id lists")
a = ap.parse_args()
assert "test" not in (a.out_pool + a.out_todo).lower()

jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]

from datasets import load_dataset
ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
allp = [{"pair_id": f"train-{i}", "chosen": r["chosen"], "rejected": r["rejected"]} for i, r in enumerate(ds)]
kept, nd = D.filter_mismatched(allp)
pairs = [p for p in kept if D.assign_split(D.first_turn(D.extract_prompt(p["chosen"])), a.seed) == "rm_train"]
local = {p["pair_id"] for p in D.load_split(a.pairs, "rm_train")}      # the 2,000 already-trained pairs
pairs = ([p for p in pairs if p["pair_id"] in local] +
         D.order_by_hash([p for p in pairs if p["pair_id"] not in local], 1, lambda p: p["pair_id"]))
# assert len(local) == 2000 and local <= {p["pair_id"] for p in pairs}, "local 2k not inside rebuilt rm_train"

assert len({p["pair_id"] for p in pairs}) == len(pairs), "duplicate pair_id"

val_ids = {d["pair_id"] for d in jl(a.val)}
grp = lambda p: D.first_turn(D.extract_prompt(p["chosen"]))
val_groups = {grp(p) for p in pairs if p["pair_id"] in val_ids}
assert len(val_ids) == 1000 and len(val_groups) > 0
assert val_ids <= {p["pair_id"] for p in pairs}, "val pairs not all in rebuilt rm_train: row indexing or filter differs from Shravani's"

ex_ids = set()
for f in a.exclude:
    d = json.load(open(f))
    for k in ("train", "val"):
        ex_ids |= set(d[k])
all_ids = {p["pair_id"] for p in pairs}
assert not a.exclude or len(ex_ids) == 10500, f"expected 10,500 SFT ids, got {len(ex_ids)}"
assert ex_ids <= all_ids, f"{len(ex_ids - all_ids)} SFT ids not in the rebuilt rm_train: id indexing differs"
ex_groups = {grp(p) for p in pairs if p["pair_id"] in ex_ids}
bad_ids, bad_groups = val_ids | ex_ids, val_groups | ex_groups
cand = [p for p in pairs if p["pair_id"] not in bad_ids and grp(p) not in bad_groups]
print(f"eligible after exclusions: {len(cand)} (SFT ids {len(ex_ids)}, SFT groups {len(ex_groups)})")
pool = cand[: a.n]
assert len(pool) == a.n, f"only {len(pool)} eligible pairs"

out = []
for p in pool:
    g = grp(p)
    s = D.assign_split(g, a.seed)
    assert s == "rm_train", f"{p['pair_id']} hashes to {s}"
    out.append({**p, "group": g, "prompt_id": D.prompt_id(D.extract_prompt(p["chosen"])), "split": s})

assert not ({p["group"] for p in out} & val_groups), "val group leaked into pool"
assert not ({p["pair_id"] for p in out} & ex_ids), "SFT pair leaked into pool"
assert not ({p["group"] for p in out} & ex_groups), "SFT group leaked into pool"

scored = {d["pair_id"] for f in a.gold for d in jl(f)}
todo = [p for p in out if p["pair_id"] not in scored]
D.save_split(out, a.out_pool)

with open(a.out_todo, "w") as f:
    for p in todo:
        f.write(json.dumps({k: p[k] for k in ("pair_id", "prompt_id", "group", "split", "chosen", "rejected")}) + "\n")

meta = dict(n_pool=len(out), n_todo=len(todo), n_groups=len({p["group"] for p in out}),
            n_val_excluded=len(val_ids), n_sft_excluded=len(ex_ids), n_sft_groups=len(ex_groups), n_eligible=len(cand),
            pool_ids_sha256=hashlib.sha256("\n".join(p["pair_id"] for p in out).encode()).hexdigest(), seed=a.seed)

json.dump(meta, open(a.out_pool + ".meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))