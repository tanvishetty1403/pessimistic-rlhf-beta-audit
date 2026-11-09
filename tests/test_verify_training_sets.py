import copy
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import data as D  # noqa: E402
import verify_training_sets as V  # noqa: E402

CANON = "Skywork/Skywork-Reward-V2-Llama-3.2-3B@abc"
N = 600


def wjl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def make_pair(i, split="rm_train"):
    k = 0
    while True:
        g = f"question {i}-{k}"
        if D.assign_split(g, 0) == split:
            break
        k += 1
    pr = f"\n\nHuman: {g}\n\nAssistant:"
    return {"pair_id": f"train-{i}", "chosen": f"{pr} good {i}", "rejected": f"{pr} bad {i}",
            "group": g, "prompt_id": D.prompt_id(pr), "split": split}


@pytest.fixture
def env(tmp_path):
    import numpy as np
    rng = np.random.default_rng(1)
    pool = [make_pair(i) for i in range(N)]
    gold = []
    for p in pool:
        c, r = (float(rng.normal()), float(rng.normal()))
        if p["pair_id"] in ("train-5", "train-77"):
            r = c                                   # two ties
        gold.append({"pair_id": p["pair_id"], "gold_chosen": c, "gold_rejected": r, "gold_model": CANON})
    gd = {g["pair_id"]: g for g in gold}
    rel = D.relabel_with_gold(pool, gd)
    n0, _ = D.inject_label_noise(rel, 0.0, 0)
    n25, _ = D.inject_label_noise(rel, 0.25, 0)
    d = tmp_path
    f = dict(pool=d / "pool.json", gold=wjl(d / "gold.jsonl", gold), n0=d / "n0.json", n25=d / "n25.json",
             val=wjl(d / "val.jsonl", [{"pair_id": "train-9000"}, {"pair_id": "train-9001"}]),
             sft=d / "sft.json", ref=d / "ref.json")
    D.save_split(pool, f["pool"]); D.save_split(n0, f["n0"]); D.save_split(n25, f["n25"])
    f["sft"].write_text(json.dumps({"train": ["train-8000"], "val": ["train-8001"]}))
    D.save_split([make_pair(9000), make_pair(9001), make_pair(8000), make_pair(8001)], f["ref"])
    f.update(pool_l=pool, n0_l=n0, n25_l=n25, gold_l=gold)
    return f


def run(f, **kw):
    a = dict(n0=f["n0"], n25=f["n25"], pool=f["pool"], gold=f["gold"], val=[f["val"]], sft=[f["sft"]],
             group_ref=[f["ref"]], canonical=CANON, noise=0.25, flip_seed=0, require_group_coverage=True)
    a.update(kw)
    return V.verify(**a)


def has(fails, s):
    return any(s in x for x in fails)


def test_happy(env):
    info, fails = run(env)
    assert fails == [], fails
    assert info["n_ties_expected"] == 2 and info["n_kept"] == N - 2
    assert info["group_check"] == "full"
    assert 0.2 < info["flip_frac"] < 0.3


def test_order_mismatch(env):
    l = copy.deepcopy(env["n25_l"]); l[0], l[1] = l[1], l[0]
    D.save_split(l, env["n25"])
    assert has(run(env)[1], "identical order")


def test_n25_extra_flip(env):
    l = copy.deepcopy(env["n25_l"])
    l[3]["chosen"], l[3]["rejected"] = l[3]["rejected"], l[3]["chosen"]
    l[3]["flipped"] = not l[3]["flipped"]
    D.save_split(l, env["n25"])
    assert has(run(env)[1], "inject_label_noise")


def test_n25_silent_text_change(env):
    l = copy.deepcopy(env["n25_l"]); l[4]["chosen"] += " x"
    D.save_split(l, env["n25"])
    assert has(run(env)[1], "inject_label_noise") or has(run(env)[1], "more than a flip")


def test_wrong_relabel_in_n0(env):
    l = copy.deepcopy(env["n0_l"])
    l[0]["chosen"], l[0]["rejected"] = l[0]["rejected"], l[0]["chosen"]
    D.save_split(l, env["n0"])
    assert has(run(env)[1], "not the correct gold-relabel")


def test_tie_pair_kept(env):
    extra = [p for p in env["pool_l"] if p["pair_id"] == "train-5"][0]
    l = copy.deepcopy(env["n0_l"]) + [{**extra, "gold_chosen": 0.0, "gold_rejected": 0.0, "gold_model": CANON,
                                      "gold_agrees_with_human": False, "relabeled": True, "flipped": False,
                                      "flip_seed": 0, "noise_rate": 0.0}]
    D.save_split(l, env["n0"])
    assert has(run(env)[1], "pool minus gold ties")


def test_missing_gold(env):
    wjl(env["gold"], env["gold_l"][:-1])
    assert has(run(env)[1], "no gold score")


def test_val_overlap(env):
    wjl(env["val"], [{"pair_id": "train-10"}])
    assert has(run(env)[1], "are val pairs")


def test_sft_overlap(env):
    env["sft"].write_text(json.dumps({"train": ["train-11"], "val": []}))
    assert has(run(env)[1], "are SFT pairs")


def test_group_leak_via_ref(env):
    # a val pair (train-9000) that shares a first-turn group with a pool pair
    ref = [{**make_pair(9000), "group": env["pool_l"][20]["group"]}]
    D.save_split(ref, env["ref"])
    wjl(env["val"], [{"pair_id": "train-9000"}])
    env["sft"].write_text(json.dumps({"train": [], "val": []}))
    assert has(run(env)[1], "groups are shared")


def test_group_coverage_required(env):
    D.save_split([make_pair(9000)], env["ref"])
    assert has(run(env)[1], "cannot prove group disjointness")
    info, fails = run(env, require_group_coverage=False)
    assert info["group_check"] == "partial" and "WARNING" in info and not has(fails, "group")


def test_group_check_skipped_is_loud(env):
    info, fails = run(env, group_ref=[], require_group_coverage=False)
    assert fails == [] and "SKIPPED" in info["group_check"]


def test_pair_from_tune_split(env):
    l = copy.deepcopy(env["n0_l"]); l[0].update(make_pair(0, "tune"))   # group hashes to tune, text inconsistent too
    D.save_split(l, env["n0"])
    assert has(run(env)[1], "outside rm_train") or has(run(env)[1], "inconsistent")


def test_group_field_inconsistent(env):
    l = copy.deepcopy(env["n0_l"]); l[2]["group"] = "something else"
    D.save_split(l, env["n0"])
    assert has(run(env)[1], "inconsistent")


def test_flip_frac_off(env):
    l, _ = D.inject_label_noise(env["n0_l"], 0.6, 0)
    D.save_split(l, env["n25"])
    assert has(run(env)[1], "flip_frac")


def test_wrong_canonical(env):
    assert has(run(env, canonical="other")[1], "--canonical")


def test_stale_meta(env):
    pathlib.Path(str(env["n0"]) + ".meta.json").write_text(json.dumps({"out_sha256": "0" * 64, "kept": N - 2,
                                                                       "human_gold_agreement": 0.5}))
    assert has(run(env)[1], "stale meta")


def test_test_named_input_refused(env, tmp_path):
    bad = tmp_path / "test_pairs.json"; bad.write_text("[]")
    with pytest.raises(PermissionError):
        run(env, group_ref=[bad])
