import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import merge_gold as M  # noqa: E402

CANON = "Skywork/Skywork-Reward-V2-Llama-3.2-3B@abc123"
BARE = "Skywork/Skywork-Reward-V2-Llama-3.2-3B"


def rec(pid, c, r, model=BARE):
    return {"pair_id": pid, "gold_chosen": c, "gold_rejected": r, "gold_model": model}


def wjl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def env(tmp_path):
    ids = [f"train-{i}" for i in range(6)]
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps([{"pair_id": i} for i in ids]))
    old = wjl(tmp_path / "old.jsonl", [rec(i, 1.0 + k, 0.0) for k, i in enumerate(ids[:4])] + [rec("train-999", 1, 0)])
    todo = wjl(tmp_path / "todo.jsonl", [{"pair_id": i} for i in ids[4:]])
    new = wjl(tmp_path / "new.jsonl", [rec(ids[4], 2.0, 1.0, CANON), rec(ids[5], 3.0, 3.0, CANON)])  # one tie
    return dict(tmp=tmp_path, ids=ids, pool=pool, old=old, todo=todo, new=new,
                sha=hashlib.sha256(todo.read_bytes()).hexdigest())


def run(e, **kw):
    args = dict(pool=e["pool"], gold=[e["old"]], new=e["new"], todo=e["todo"], todo_sha256=e["sha"],
                canonical=CANON, aliases=[BARE], out=e["tmp"] / "out.jsonl")
    args.update(kw)
    return M.merge(**args)


def test_happy_path(env):
    meta = run(env)
    rows = [json.loads(l) for l in (env["tmp"] / "out.jsonl").read_text().splitlines()]
    assert [r["pair_id"] for r in rows] == env["ids"]          # pool order
    assert {r["gold_model"] for r in rows} == {CANON}          # canonicalised
    assert meta["n_missing"] == 0 and meta["n_outside_pool_dropped"] == 1 and meta["n_ties_kept"] == 1


def test_missing_aborts(env):
    with pytest.raises(M.MergeError, match="no gold score"):
        run(env, new=None)


def test_dry_run_missing_equals_todo(env):
    meta = run(env, new=None, allow_missing=True)
    assert meta["n_missing"] == 2 and meta["n_written"] == 4


def test_dry_run_detects_drift(env):
    bad_todo = wjl(env["tmp"] / "bad_todo.jsonl", [{"pair_id": "train-3"}, {"pair_id": "train-5"}])
    with pytest.raises(M.MergeError, match="drifted"):
        run(env, new=None, allow_missing=True, todo=bad_todo, todo_sha256=None)


def test_todo_sha_mismatch(env):
    with pytest.raises(M.MergeError, match="sha256 mismatch"):
        run(env, todo_sha256="0" * 64)


def test_new_must_match_todo(env):
    wjl(env["new"], [rec("train-4", 2.0, 1.0, CANON)])
    with pytest.raises(M.MergeError, match="does not match todo"):
        run(env)


def test_overlap_disagreement(env):
    dup = wjl(env["tmp"] / "dup.jsonl", [rec("train-0", 1.5, 0.0)])   # old has 1.0
    with pytest.raises(M.MergeError, match="disagree"):
        run(env, gold=[env["old"], dup])


def test_overlap_within_tol_first_wins(env):
    dup = wjl(env["tmp"] / "dup.jsonl", [rec("train-0", 1.01, 0.0)])
    meta = run(env, gold=[env["old"], dup])
    assert meta["n_overlap_checked"] == 1
    first = json.loads((env["tmp"] / "out.jsonl").read_text().splitlines()[0])
    assert first["gold_chosen"] == 1.0


def test_unknown_model(env):
    wjl(env["old"], [rec("train-0", 1, 0, "other/model")])
    with pytest.raises(M.MergeError, match="not in allowed"):
        run(env)


def test_nonfinite(env):
    env["old"].write_text(json.dumps(rec("train-0", float("nan"), 0.0)).replace("NaN", "NaN") + "\n")
    with pytest.raises(M.MergeError, match="non-finite"):
        run(env)


def test_todo_not_in_pool(env):
    wrong = wjl(env["tmp"] / "wrong.jsonl", [{"pair_id": "train-77"}])
    with pytest.raises(M.MergeError, match="not in the pool"):
        run(env, todo=wrong, todo_sha256=None)


def test_test_named_out_refused(env):
    with pytest.raises(M.MergeError, match="test-named"):
        run(env, out=env["tmp"] / "gold_test.jsonl")
