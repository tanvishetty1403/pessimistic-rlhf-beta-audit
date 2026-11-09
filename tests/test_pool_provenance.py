import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import pool_provenance as PP  # noqa: E402


def wjl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def env(tmp_path):
    ids = [f"train-{i}" for i in range(10)]
    gold = [{"pair_id": i, "gold_chosen": 1.0, "gold_rejected": 0.0, "gold_model": "m"} for i in ids]
    gold[2].update(gold_chosen=-3.0, gold_rejected=-3.0)          # identical text, exact tie
    gold[5].update(gold_chosen=-3.0, gold_rejected=-2.98)         # identical text, noise 0.02, prefers rejected
    gold[7].update(gold_chosen=-3.0, gold_rejected=-2.8)          # identical text, noise 0.2 > tol
    pool = [{"pair_id": i, "group": f"g{k % 3}"} for k, i in enumerate(ids) if i not in ("train-2", "train-5", "train-7")]
    (tmp_path / "pool.json").write_text(json.dumps(pool))
    (tmp_path / "pool.json.meta.json").write_text(json.dumps({"n_pool": 10, "n_groups": 4, "n_todo": 5, "pool_ids_sha256": "old"}))
    return dict(d=tmp_path, gold=wjl(tmp_path / "gold.jsonl", gold), pool=tmp_path / "pool.json",
                meta=tmp_path / "pool.json.meta.json", out=tmp_path / "dropped.json")


def test_happy(env):
    pool_before = env["pool"].read_bytes()
    rec = PP.run(env["gold"], env["pool"], env["out"], env["meta"], expect=3)
    s = rec["noise_floor"]
    assert s["n"] == 3 and s["n_exact_ties"] == 1 and s["n_above_tol"] == 1
    assert s["n_gold_prefers_rejected"] == 2 and s["n_gold_prefers_chosen"] == 0
    assert abs(s["max_abs_diff"] - 0.2) < 1e-9
    m = json.loads(env["meta"].read_text())
    assert m["n_pool"] == 7 and m["pre_drop"]["n_pool"] == 10 and m["pre_drop"]["pool_ids_sha256"] == "old"
    assert m["pool_ids_sha256"] == PP.ids_sha([f"train-{i}" for i in (0, 1, 3, 4, 6, 8, 9)])
    assert m["n_groups"] == 3 and m["n_todo"] == 5 and m["n_dropped_identical"] == 3
    assert env["pool"].read_bytes() == pool_before                  # pool never rewritten


def test_expect_mismatch(env):
    with pytest.raises(SystemExit, match="expected 12"):
        PP.run(env["gold"], env["pool"], env["out"], env["meta"], expect=12)
    assert not env["out"].exists()


def test_no_overwrite_and_no_double_meta(env):
    PP.run(env["gold"], env["pool"], env["out"], env["meta"])
    with pytest.raises(SystemExit, match="already exists"):
        PP.run(env["gold"], env["pool"], env["out"], env["meta"])
    with pytest.raises(SystemExit, match="applied before"):
        PP.run(env["gold"], env["pool"], env["d"] / "dropped2.json", env["meta"])


def test_pool_pair_without_gold(env):
    wjl(env["gold"], [{"pair_id": "train-0", "gold_chosen": 1, "gold_rejected": 0, "gold_model": "m"}])
    with pytest.raises(SystemExit, match="no gold row"):
        PP.run(env["gold"], env["pool"], env["out"])


def test_meta_optional(env):
    PP.run(env["gold"], env["pool"], env["out"])
    assert json.loads(env["meta"].read_text())["n_pool"] == 10
