"""scripts/merge_gold.py

Merge gold-score jsonl files (gold_score.py --smoke format) into ONE file covering exactly the Option-B pool.
Every path is a CLI argument; nothing is resolved relative to the script or hard-coded.

Full run (after Tanvi's file comes):
  python scripts/merge_gold.py --pool data/splits/b_pool_v2.json \
      --gold val_gold/gold_val.jsonl data/gold_smoke.jsonl data/relabel_gold/gold_relabel.jsonl \
      --new <tanvi_scores.jsonl> --todo b_todo_v2.jsonl --todo_sha256 <sha from Tanvi> \
      --canonical "<pinned gold_model string>" --alias "Skywork/Skywork-Reward-V2-Llama-3.2-3B" \
      --out data/gold_b_combined_v2.jsonl

Dry run now (before Tanvi): drop --new, add --allow_missing. The missing set must equal the todo ids exactly.

Rules
  * --gold files are priority ordered (first wins). --new is lowest priority and must contain exactly the todo ids.
  * A pair_id seen twice must agree within --tol on both scores, else abort.
  * gold_model must be --canonical, one of --alias, or the part of --canonical before '@'. Output rewrites it to --canonical.
  * Scores must be finite floats. Ties are KEPT (relabel_with_gold drops them later).
  * Output is restricted to pool ids, in pool order. Gold ids outside the pool are counted and dropped.
  * Missing pool ids abort unless --allow_missing.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import sys


class MergeError(Exception):
    pass


def sha256_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _num(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise MergeError(f"{where}: non-finite or non-numeric score {v!r}")
    return float(v)


def read_gold(path, allowed, canonical, tol):
    """-> {pair_id: record}. Validates fields, model name, finiteness, within-file duplicates."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{ln}"
            d = json.loads(line)
            for k in ("pair_id", "gold_chosen", "gold_rejected", "gold_model"):
                if k not in d:
                    raise MergeError(f"{where}: missing field {k!r}")
            if d["gold_model"] not in allowed:
                raise MergeError(f"{where}: gold_model {d['gold_model']!r} not in allowed {sorted(allowed)}")
            rec = {"pair_id": d["pair_id"], "gold_chosen": _num(d["gold_chosen"], where),
                   "gold_rejected": _num(d["gold_rejected"], where), "gold_model": canonical}
            old = out.get(rec["pair_id"])
            if old is not None:
                if _diff(old, rec) > tol:
                    raise MergeError(f"{where}: duplicate {rec['pair_id']} disagrees within file (diff {_diff(old, rec):.4g} > tol {tol})")
                continue
            out[rec["pair_id"]] = rec
    return out


def _diff(a, b):
    return max(abs(a["gold_chosen"] - b["gold_chosen"]), abs(a["gold_rejected"] - b["gold_rejected"]))


def read_ids_jsonl(path):
    return [json.loads(l)["pair_id"] for l in open(path, encoding="utf-8") if l.strip()]


def merge(pool, gold, new, todo, todo_sha256, canonical, aliases, out, tol=0.05, allow_missing=False):
    pool, out = pathlib.Path(pool), pathlib.Path(out)
    if "test" in out.name.lower():
        raise MergeError("refusing to write a test-named file")
    allowed = {canonical, canonical.split("@")[0], *aliases}

    pool_pairs = json.load(open(pool, encoding="utf-8"))
    pool_ids = [p["pair_id"] for p in pool_pairs]
    if len(set(pool_ids)) != len(pool_ids):
        raise MergeError("duplicate pair_id in pool")
    pool_set = set(pool_ids)

    todo_ids = None
    todo_sha = None
    if todo:
        todo_ids = read_ids_jsonl(todo)
        if len(set(todo_ids)) != len(todo_ids):
            raise MergeError("duplicate pair_id in todo file")
        todo_sha = sha256_file(todo)
        if todo_sha256 and todo_sha != todo_sha256.lower():
            raise MergeError(f"todo sha256 mismatch: file {todo_sha[:12]} vs given {todo_sha256[:12]} (Tanvi scored a different input)")
        if not set(todo_ids) <= pool_set:
            raise MergeError(f"{len(set(todo_ids) - pool_set)} todo ids are not in the pool: wrong pool/todo pair")
    elif todo_sha256:
        raise MergeError("--todo_sha256 given without --todo")

    sources = [(g, read_gold(g, allowed, canonical, tol)) for g in gold]
    if new:
        new_recs = read_gold(new, allowed, canonical, tol)
        if todo_ids is not None:
            miss, extra = set(todo_ids) - set(new_recs), set(new_recs) - set(todo_ids)
            if miss or extra:
                raise MergeError(f"--new does not match todo: {len(miss)} todo ids missing, {len(extra)} ids not in todo")
        sources.append((new, new_recs))

    merged, n_conflict_checked, max_diff = {}, 0, 0.0
    for _, recs in sources:
        for pid, r in recs.items():
            if pid in merged:
                n_conflict_checked += 1
                dd = _diff(merged[pid], r)
                max_diff = max(max_diff, dd)
                if dd > tol:
                    raise MergeError(f"{pid}: sources disagree (diff {dd:.4g} > tol {tol}); stale or mis-indexed score")
            else:
                merged[pid] = r

    missing = [i for i in pool_ids if i not in merged]
    n_outside = len(set(merged) - pool_set)
    if missing:
        if not allow_missing:
            raise MergeError(f"{len(missing)} pool pairs have no gold score (e.g. {missing[:3]}); use --allow_missing only for the dry run")
        if todo_ids is not None and set(missing) != set(todo_ids):
            raise MergeError(f"dry run: missing set != todo set ({len(set(missing) ^ set(todo_ids))} ids differ): pool or id indexing drifted")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for pid in pool_ids:
            if pid in merged:
                f.write(json.dumps(merged[pid]) + "\n")
    os.replace(tmp, out)

    meta = dict(n_pool=len(pool_ids), n_written=len(pool_ids) - len(missing), n_missing=len(missing),
                n_outside_pool_dropped=n_outside, n_overlap_checked=n_conflict_checked, max_overlap_diff=max_diff, tol=tol,
                n_ties_kept=sum(1 for i in pool_ids if i in merged and merged[i]["gold_chosen"] == merged[i]["gold_rejected"]),
                canonical_model=canonical, pool_sha256=sha256_file(pool), todo_sha256=todo_sha,
                input_sha256={str(p): sha256_file(p) for p, _ in sources}, out_sha256=sha256_file(out))
    pathlib.Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--gold", nargs="*", default=[], help="existing gold jsonl files, highest priority first")
    ap.add_argument("--new", default=None, help="Tanvi's new scores (must match --todo exactly)")
    ap.add_argument("--todo", default=None)
    ap.add_argument("--todo_sha256", default=None)
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--alias", nargs="*", default=[])
    ap.add_argument("--tol", type=float, default=0.05, help="max abs score diff for the same pair across sources (fp16 batch noise)")
    ap.add_argument("--allow_missing", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        meta = merge(a.pool, a.gold, a.new, a.todo, a.todo_sha256, a.canonical, a.alias, a.out, a.tol, a.allow_missing)
    except MergeError as e:
        sys.exit(f"merge_gold: {e}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
