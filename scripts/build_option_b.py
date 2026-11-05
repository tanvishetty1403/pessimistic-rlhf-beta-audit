"""scripts/build_option_b.py   

(Option B: gold-relabel THEN flip; order matters)
python scripts/build_option_b.py --pairs data/splits/rm_train_clean.json --gold gold_pairs.jsonl \
    --noise 0.25 --seed 0 --n 2000 --out data/splits/rm_train_goldrelabel_n25.json

Use --noise 0 for the clean condition. Pairs need pair_id; gold jsonl = gold_score.py --smoke format.
"""

import argparse, hashlib, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import data as D

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--gold", required=True)
ap.add_argument("--noise", type=float, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

assert "test" not in pathlib.Path(a.out).name.lower(), "refusing to write a test-named file"

pairs = D.load_split(a.pairs, "rm_train")[: a.n]
gold = {}

for l in open(a.gold):
    if l.strip():
        d = json.loads(l)
        gold[d["pair_id"]] = d

rel = D.relabel_with_gold(pairs, gold) # 1) relabel (drops unscored + ties)

if not rel:
    sys.exit("nothing left after relabeling")

out, flip = D.inject_label_noise(rel, a.noise, a.seed) # 2) flip AFTER relabel
D.save_split(out, a.out)

n_rel = sum(p["relabeled"] for p in rel)
meta = dict(pairs_in=len(pairs), kept=len(rel), dropped=len(pairs) - len(rel),
            relabeled=int(n_rel), human_gold_agreement=1 - n_rel / len(rel),
            flipped=int(flip.sum()), flip_frac=float(flip.mean()), noise=a.noise, seed=a.seed,
            gold_models=sorted({p["gold_model"] for p in rel}),
            out_sha256=hashlib.sha256(pathlib.Path(a.out).read_bytes()).hexdigest())

json.dump(meta, open(a.out + ".meta.json", "w"), indent=2)
print(json.dumps(meta, indent=2))