"""scripts/build_option_b.py   

(Option B: gold-relabel THEN flip; order matters)
python scripts/build_option_b.py --pairs data/splits/b_pool_v2.json --gold data/gold_b_combined_v2.jsonl \
    --noise 0.25 --seed 0 --max_dropped 30 --out data/splits/rm_train_goldrelabel_n25.json

Use --noise 0 for the clean condition. Pairs need pair_id; gold jsonl = gold_score.py --smoke format.

--n defaults to the WHOLE --pairs file (it used to silently default to the first 2000). Give --n only to take a prefix.
--max_dropped K: abort unless every pair has a gold score (no silent drops) and at most K pairs were dropped as ties.
"""

import argparse, hashlib, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import data as D

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--gold", required=True)
ap.add_argument("--noise", type=float, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=None, help="take the first n pairs; default = all pairs in --pairs")
ap.add_argument("--max_dropped", type=int, default=None,
                help="abort if any pair is unscored or if more than this many pairs are dropped as gold ties")
ap.add_argument("--out", required=True)
a = ap.parse_args()

assert "test" not in pathlib.Path(a.out).name.lower(), "refusing to write a test-named file"

pairs = D.load_split(a.pairs, "rm_train")
assert len({p["pair_id"] for p in pairs}) == len(pairs), "duplicate pair_id in --pairs"
if a.n is not None:
    assert 0 < a.n <= len(pairs), f"--n {a.n} but --pairs has only {len(pairs)} pairs"
    pairs = pairs[: a.n]
gold = {}

for l in open(a.gold):
    if l.strip():
        d = json.loads(l)
        gold[d["pair_id"]] = d

n_unscored = sum(p["pair_id"] not in gold for p in pairs)
rel = D.relabel_with_gold(pairs, gold) # 1) relabel (drops unscored + ties)

if not rel:
    sys.exit("nothing left after relabeling")

n_ties = len(pairs) - len(rel) - n_unscored
if a.max_dropped is not None:
    if n_unscored:
        sys.exit(f"{n_unscored} pairs have no gold score (silent drops); fix the merge before building")
    if n_ties > a.max_dropped:
        sys.exit(f"{n_ties} pairs dropped as gold ties > --max_dropped {a.max_dropped}")

out, flip = D.inject_label_noise(rel, a.noise, a.seed) # 2) flip AFTER relabel
D.save_split(out, a.out)

n_rel = sum(p["relabeled"] for p in rel)
meta = dict(pairs_in=len(pairs), kept=len(rel), dropped=len(pairs) - len(rel),
            unscored=int(n_unscored), ties_dropped=int(n_ties),
            relabeled=int(n_rel), human_gold_agreement=1 - n_rel / len(rel),
            flipped=int(flip.sum()), flip_frac=float(flip.mean()), noise=a.noise, seed=a.seed,
            gold_models=sorted({p["gold_model"] for p in rel}),
            pairs_file_sha256=hashlib.sha256(pathlib.Path(a.pairs).read_bytes()).hexdigest(),
            gold_file_sha256=hashlib.sha256(pathlib.Path(a.gold).read_bytes()).hexdigest(),
            out_sha256=hashlib.sha256(pathlib.Path(a.out).read_bytes()).hexdigest())

json.dump(meta, open(a.out + ".meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))
