"""scripts/pool_provenance.py

Record the pairs removed from b_pool_v2 after the in-place edit (chosen == rejected), refresh the pool meta, and
measure Skywork's fp16 noise floor on those identical-text pairs.

The merged gold file (merge_gold.py output) was written for the ORIGINAL 10,000-pair pool and is untouched, so
removed ids = gold ids - current pool ids. Nothing here rewrites the pool or the gold file.

python scripts/pool_provenance.py --gold data/gold_b_combined_v2.jsonl --pool data/splits/b_pool_v2.json \
    --out data/splits/b_pool_v2_dropped_identical.json --meta data/splits/b_pool_v2.json.meta.json --expect 12

Identical text should score identically, so |gold_chosen - gold_rejected| on these pairs is a direct measure of the
scoring noise (batch/padding effects in fp16). Compare it with the --tol used in merge_gold.py.
Refuses to overwrite an existing --out, and refuses to apply the meta update twice.
"""
import argparse
import hashlib
import json
import pathlib
import statistics
import sys


def ids_sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def run(gold, pool, out, meta=None, expect=None, tol=0.05):
    for p in (gold, pool, out, meta):
        if p and "test" in pathlib.Path(p).name.lower():
            raise SystemExit(f"refusing test-named path: {p}")
    out = pathlib.Path(out)
    if out.exists():
        raise SystemExit(f"{out} already exists; the provenance record is not overwritten")
    grows = [json.loads(l) for l in open(gold, encoding="utf-8") if l.strip()]
    gold_d = {r["pair_id"]: r for r in grows}
    P = json.load(open(pool, encoding="utf-8"))
    pool_ids = [p["pair_id"] for p in P]
    if len(set(pool_ids)) != len(pool_ids):
        raise SystemExit("duplicate pair_id in pool")
    missing = [i for i in pool_ids if i not in gold_d]
    if missing:
        raise SystemExit(f"{len(missing)} pool pairs have no gold row (e.g. {missing[:3]}): wrong gold file for this pool")
    pool_set = set(pool_ids)
    dropped = [r for r in grows if r["pair_id"] not in pool_set]
    if expect is not None and len(dropped) != expect:
        raise SystemExit(f"found {len(dropped)} removed pairs, expected {expect}")

    pairs = [{"pair_id": r["pair_id"], "gold_chosen": r["gold_chosen"], "gold_rejected": r["gold_rejected"],
              "diff": r["gold_chosen"] - r["gold_rejected"]} for r in dropped]
    ad = [abs(p["diff"]) for p in pairs]
    stats = dict(n=len(pairs), n_exact_ties=sum(d == 0 for d in ad), n_above_tol=sum(d > tol for d in ad), tol=tol,
                 max_abs_diff=max(ad, default=None), median_abs_diff=statistics.median(ad) if ad else None,
                 n_gold_prefers_chosen=sum(p["diff"] > 0 for p in pairs),
                 n_gold_prefers_rejected=sum(p["diff"] < 0 for p in pairs))
    rec = dict(reason="chosen == rejected (identical text); removed after make_b_pool, in place",
               ids_sha256=ids_sha([p["pair_id"] for p in pairs]), noise_floor=stats, pairs=pairs,
               pool_file_sha256=hashlib.sha256(pathlib.Path(pool).read_bytes()).hexdigest(),
               gold_file_sha256=hashlib.sha256(pathlib.Path(gold).read_bytes()).hexdigest(),
               n_pool_before=len(pool_ids) + len(pairs), n_pool_after=len(pool_ids))

    if meta:
        mp = pathlib.Path(meta)
        m = json.load(open(mp, encoding="utf-8"))
        if "pre_drop" in m:
            raise SystemExit("meta already has pre_drop: the update was applied before")
        new_sha = ids_sha(pool_ids)
        m["pre_drop"] = {k: m.get(k) for k in ("n_pool", "n_groups", "pool_ids_sha256")}
        m["n_pool"] = len(pool_ids)
        if all("group" in p for p in P):
            m["n_groups"] = len({p["group"] for p in P})
        m["pool_ids_sha256"] = new_sha
        m["n_dropped_identical"] = len(pairs)
        m["dropped_identical_ids_sha256"] = rec["ids_sha256"]
        m["pool_file_sha256"] = rec["pool_file_sha256"]
        rec["meta_updated"] = str(mp)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))          # written first: a crash cannot leave meta updated w/o the record
    if meta:
        tmp = mp.with_name(mp.name + ".tmp")
        tmp.write_text(json.dumps(m, indent=2))
        tmp.replace(mp)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default=None, help="b_pool_v2.json.meta.json to update in place")
    ap.add_argument("--expect", type=int, default=None)
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args(argv)
    rec = run(a.gold, a.pool, a.out, a.meta, a.expect, a.tol)
    print(json.dumps({k: v for k, v in rec.items() if k != "pairs"}, indent=2))
    print("\nper-pair diff (gold_chosen - gold_rejected):")
    for p in rec["pairs"]:
        print(f"  {p['pair_id']:>14}  {p['diff']:+.6f}")
    s = rec["noise_floor"]
    print(f"\nPrereg line: pool = first {rec['n_pool_before']} eligible pairs minus {s['n']} chosen==rejected pairs "
          f"(ids sha256 {rec['ids_sha256'][:12]}) = {rec['n_pool_after']}.")


if __name__ == "__main__":
    main()
