"""scripts/verify_training_sets.py

Final gate before the shas go to Shravani. Re-checks the invariants on the FINISHED n0 / n25 files instead of
trusting the pool-stage asserts. Reads no test file: test/tune disjointness is checked by re-hashing each group
with assign_split (same rule that built the splits).

python scripts/verify_training_sets.py \
   --n0 data/splits/rm_train_goldrelabel_n0.json --n25 data/splits/rm_train_goldrelabel_n25.json \
   --pool data/splits/b_pool_v2.json --gold data/gold_b_combined_v2.jsonl \
   --val val_gold/gold_val.jsonl --sft sft_v1_pairs.json \
   --group_ref data/splits/rm_train_clean.json \
   --canonical "Skywork/Skywork-Reward-V2-Llama-3.2-3B@beafe5f2e6a275ce8c556f68f2d82de7fefbc3d8" \
   --noise 0.25 --flip_seed 0 --report data/splits/verify_b_v2.json

Exit code 1 if any check fails. Checks:
  A  n0/n25 same ids, same order; kept == pool minus gold ties (recomputed from the merged gold file)
  B  n0 relabel is right at text level (vs pool texts and gold scores); no flips in n0
  C  n25 == inject_label_noise(n0, noise, flip_seed) exactly (so n25 differs from n0 only by flips); flip_frac in 5 sigma
  D  ids disjoint from val and SFT ids; group/prompt_id consistent with text; every group hashes to rm_train;
     groups disjoint from val/SFT groups (needs --group_ref to map those ids to groups; skipped loudly otherwise)
  E  one gold_model, equal to --canonical if given
  F  shas, human_gold_agreement; the builders' .meta.json files agree with the files on disk
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src import data as D  # noqa: E402

FLIP_KEYS = {"chosen", "rejected", "flipped", "flip_seed", "noise_rate"}


def sha_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _no_test(path):
    if "test" in pathlib.Path(path).name.lower():
        raise PermissionError(f"refusing to read a test-named file: {path}")


def verify(n0, n25, pool, gold, val, sft=(), group_ref=(), canonical=None, noise=0.25, flip_seed=0,
           split_seed=0, flip_sigma=5.0, require_group_coverage=False):
    fails, info = [], {}

    def chk(cond, msg):
        if not cond:
            fails.append(msg)

    def bad(items, pred):
        b = [x for x in items if not pred(x)]
        return len(b), [x["pair_id"] if isinstance(x, dict) else x for x in b[:3]]

    val = [val] if isinstance(val, (str, pathlib.Path)) else list(val)
    for p in (n0, n25, pool, gold, *val, *sft, *group_ref):
        _no_test(p)
    A0, A25, P = D.load_split(n0, "rm_train"), D.load_split(n25, "rm_train"), D.load_split(pool, "rm_train")
    ids0, ids25, idsP = [p["pair_id"] for p in A0], [p["pair_id"] for p in A25], [p["pair_id"] for p in P]
    gold_d = {d["pair_id"]: d for d in jl(gold)}
    pool_d = {p["pair_id"]: p for p in P}

    # A ---------------------------------------------------------------------------------------------------
    chk(len(set(ids0)) == len(ids0) and len(set(idsP)) == len(idsP), "A: duplicate pair_id in n0 or pool")
    chk(ids0 == ids25, "A: n0 and n25 do not have identical pair_ids in identical order")
    miss_gold = [i for i in idsP if i not in gold_d]
    chk(not miss_gold, f"A: {len(miss_gold)} pool pairs have no gold score")
    tie_ids = [i for i in idsP if i in gold_d and gold_d[i]["gold_chosen"] == gold_d[i]["gold_rejected"]]
    expected = [i for i in idsP if i in gold_d and i not in set(tie_ids)]
    chk(ids0 == expected, f"A: n0 ids != pool minus gold ties ({len(ids0)} vs {len(expected)} expected)")
    info.update(n_pool=len(idsP), n_kept=len(ids0), n_ties_expected=len(tie_ids))

    # B ---------------------------------------------------------------------------------------------------
    def relabel_ok(p):
        g, pp = gold_d.get(p["pair_id"]), pool_d.get(p["pair_id"])
        if g is None or pp is None:
            return False
        agree = g["gold_chosen"] > g["gold_rejected"]
        ec, er = (pp["chosen"], pp["rejected"]) if agree else (pp["rejected"], pp["chosen"])
        return (p["chosen"] == ec and p["rejected"] == er and p["relabeled"] == (not agree)
                and p["gold_agrees_with_human"] == agree and p["gold_chosen"] == g["gold_chosen"]
                and p["gold_rejected"] == g["gold_rejected"] and p["flipped"] is False and p["noise_rate"] == 0)
    nb, ex = bad(A0, relabel_ok)
    chk(nb == 0, f"B: {nb} n0 pairs are not the correct gold-relabel of the pool (e.g. {ex})")
    nb, ex = bad(A0, lambda p: p["chosen"] != p["rejected"])
    chk(nb == 0, f"B: {nb} n0 pairs have identical chosen/rejected")
    n_rel = sum(bool(p.get("relabeled")) for p in A0)
    info["human_gold_agreement"] = 1 - n_rel / len(A0) if A0 else float("nan")
    info["n_relabeled"] = n_rel

    # C ---------------------------------------------------------------------------------------------------
    exp25, _ = D.inject_label_noise(A0, noise, flip_seed)
    if ids0 == ids25:
        nb, ex = bad([{**b, "pair_id": b["pair_id"]} for b in A25],
                     lambda b, _m={e["pair_id"]: e for e in exp25}: _m.get(b["pair_id"]) == b)
        chk(nb == 0, f"C: {nb} n25 pairs differ from inject_label_noise(n0, {noise}, {flip_seed}) (e.g. {ex})")
        d0 = {p["pair_id"]: p for p in A0}
        nb, ex = bad(A25, lambda b: all(b[k] == d0[b["pair_id"]][k] for k in d0[b["pair_id"]] if k not in FLIP_KEYS)
                     and (b["flipped"] == (b["chosen"] != d0[b["pair_id"]]["chosen"])))
        chk(nb == 0, f"C: {nb} n25 pairs differ from n0 by more than a flip (e.g. {ex})")
    else:
        fails.append("C: not compared, n0/n25 ids differ (see A)")
    n_flip = sum(bool(p.get("flipped")) for p in A25)
    ff = n_flip / len(A25) if A25 else float("nan")
    tol = flip_sigma * math.sqrt(noise * (1 - noise) / max(len(A25), 1))
    chk(abs(ff - noise) <= tol, f"C: flip_frac {ff:.4f} outside {noise} +/- {tol:.4f}")
    info.update(n_flipped=n_flip, flip_frac=ff, flip_tol=tol)

    # D ---------------------------------------------------------------------------------------------------
    val_ids = {d["pair_id"] for f in val for d in jl(f)}
    sft_ids = set()
    for f in sft:
        d = json.load(open(f, encoding="utf-8"))
        for k in ("train", "val"):
            sft_ids |= set(d[k])
    S = set(ids0) | set(ids25)
    chk(not (S & val_ids), f"D: {len(S & val_ids)} training pairs are val pairs")
    chk(not (S & sft_ids), f"D: {len(S & sft_ids)} training pairs are SFT pairs")
    info.update(n_val_ids=len(val_ids), n_sft_ids=len(sft_ids))

    def text_ok(p):
        pr = D.extract_prompt(p["chosen"])
        g = D.first_turn(pr)
        return (D.extract_prompt(p["rejected"]) == pr and p["group"] == g and p["prompt_id"] == D.prompt_id(pr)
                and p["split"] == "rm_train" and D.assign_split(g, split_seed) == "rm_train")
    for name, A in (("n0", A0), ("n25", A25)):
        nb, ex = bad(A, text_ok)
        chk(nb == 0, f"D: {nb} {name} pairs have prompt/group/prompt_id/split inconsistent with their text, "
                     f"or hash outside rm_train (e.g. {ex})")
    groups = {p["group"] for p in A0}
    info["n_groups"] = len(groups)

    id2g = {}
    for f in group_ref:
        for p in json.load(open(f, encoding="utf-8")):
            id2g[p["pair_id"]] = p.get("group") or D.first_turn(D.extract_prompt(p["chosen"]))
    if group_ref:
        cov = {"val": len(val_ids & set(id2g)) / max(len(val_ids), 1),
               "sft": len(sft_ids & set(id2g)) / max(len(sft_ids), 1) if sft_ids else 1.0}
        info["group_ref_coverage"] = cov
        if require_group_coverage:
            chk(cov["val"] == 1.0 and cov["sft"] == 1.0, f"D: --group_ref covers only {cov}; cannot prove group disjointness")
        elif min(cov.values()) < 1.0:
            info["WARNING"] = f"group-vs-val/SFT check is PARTIAL (coverage {cov})"
        leak = groups & {id2g[i] for i in (val_ids | sft_ids) if i in id2g}
        chk(not leak, f"D: {len(leak)} training groups are shared with val/SFT pairs")
        info["group_check"] = "full" if min(cov.values()) == 1.0 else "partial"
    else:
        info["group_check"] = "SKIPPED (no --group_ref): group-vs-val/SFT disjointness NOT re-verified here"
        if require_group_coverage:
            fails.append("D: --require_group_coverage needs --group_ref")

    # E ---------------------------------------------------------------------------------------------------
    models = sorted({p["gold_model"] for p in A0})
    chk(len(models) == 1, f"E: n0 has {len(models)} gold_model values: {models}")
    if canonical:
        chk(models == [canonical], f"E: gold_model {models} != --canonical")
    info["gold_models"] = models

    # F ---------------------------------------------------------------------------------------------------
    info.update(n0_sha256=sha_file(n0), n25_sha256=sha_file(n25), pool_sha256=sha_file(pool),
                gold_sha256=sha_file(gold), n0_ids_sha256=ids_sha(ids0))
    for name, path, sha, kept, ag in (("n0", n0, info["n0_sha256"], len(A0), info["human_gold_agreement"]),
                                      ("n25", n25, info["n25_sha256"], len(A25), info["human_gold_agreement"])):
        mp = pathlib.Path(str(path) + ".meta.json")
        if mp.exists():
            m = json.load(open(mp))
            chk(m.get("out_sha256") == sha, f"F: {name} meta out_sha256 != file on disk (stale meta or re-written file)")
            chk(m.get("kept") == kept, f"F: {name} meta kept {m.get('kept')} != {kept}")
            chk(abs(m.get("human_gold_agreement", -1) - ag) < 1e-9, f"F: {name} meta human_gold_agreement differs")
        else:
            info.setdefault("no_meta_for", []).append(name)
    return info, fails


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n0", required=True)
    ap.add_argument("--n25", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--gold", required=True, help="merged gold jsonl (merge_gold.py output)")
    ap.add_argument("--val", required=True, nargs="+", help="val gold jsonl(s); only pair_ids are used")
    ap.add_argument("--sft", nargs="*", default=[], help="sft_v1_pairs.json: dict with 'train'/'val' pair_id lists")
    ap.add_argument("--group_ref", nargs="*", default=[], help="json pair lists used to map val/SFT ids to groups")
    ap.add_argument("--require_group_coverage", action="store_true")
    ap.add_argument("--canonical", default=None)
    ap.add_argument("--noise", type=float, default=0.25)
    ap.add_argument("--flip_seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    info, fails = verify(a.n0, a.n25, a.pool, a.gold, a.val, a.sft, a.group_ref, a.canonical, a.noise,
                         a.flip_seed, a.split_seed, require_group_coverage=a.require_group_coverage)
    info["passed"] = not fails
    info["failures"] = fails
    print(json.dumps(info, indent=2))
    if a.report:
        pathlib.Path(a.report).write_text(json.dumps(info, indent=2))
    if fails:
        sys.exit("verify_training_sets: FAILED\n  " + "\n  ".join(fails))
    print("verify_training_sets: all checks passed")


if __name__ == "__main__":
    main()
