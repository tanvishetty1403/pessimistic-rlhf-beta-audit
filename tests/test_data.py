import numpy as np, pytest
from src import data as D


def mk(i):
    pr = f"\n\nHuman: q{i}\n\nAssistant:"
    return {"chosen": pr + f" good{i}", "rejected": pr + f" bad{i}"}


def test_prompt_extraction():
    assert D.extract_prompt(mk(1)["chosen"]) == "\n\nHuman: q1\n\nAssistant:"


def test_split_deterministic_and_disjoint():
    pairs = [mk(i) for i in range(2000)] + [mk(5)]          # duplicate prompt
    a = D.split_pairs(pairs, 0); b = D.split_pairs(pairs, 0)
    assert {k: len(v) for k, v in a.items()} == {k: len(v) for k, v in b.items()}
    sets = {k: {p["prompt"] for p in v} for k, v in a.items()}
    assert not (sets["rm_train"] & sets["tune"]) and not (sets["rm_train"] & sets["test"]) and not (sets["tune"] & sets["test"])
    assert 0.7 < len(a["rm_train"]) / 2001 < 0.9


def test_label_noise_rate_and_swap():
    pairs = [mk(i) for i in range(5000)]
    new, flip = D.inject_label_noise(pairs, 0.25, 0)
    assert abs(flip.mean() - 0.25) < 0.03
    i = int(np.argmax(flip))
    assert new[i]["chosen"] == pairs[i]["rejected"]


def test_test_split_guard(tmp_path):
    p = tmp_path / "t.json"; D.save_split([1], p)
    with pytest.raises(PermissionError):
        D.load_split(p, "test")
    assert D.load_split(p, "test", allow_test=True) == [1]

def test_pair_and_prompt_ids_present():
    pairs = [{"pair_id": f"train-{i}", **mk(i)} for i in range(50)]
    allp = [p for v in D.split_pairs(pairs, 0).values() for p in v]
    assert all(p["pair_id"] and len(p["prompt_id"]) == 12 for p in allp)
    assert len({p["pair_id"] for p in allp}) == 50


def test_filter_mismatched():
    good = mk(1)
    bad = {"chosen": "\n\nHuman: a\n\nAssistant: x", "rejected": "\n\nHuman: b\n\nAssistant: y"}
    kept, n = D.filter_mismatched([good, bad])
    assert kept == [good] and n == 1


def test_order_by_hash_nested_and_deterministic():
    items = [f"p{i}" for i in range(500)]
    a = D.order_by_hash(items, 0, key=lambda s: s)
    assert a == D.order_by_hash(items, 0, key=lambda s: s)
    assert a != D.order_by_hash(items, 1, key=lambda s: s) and a != sorted(items)


def test_clean3_cases():
    assert D.clean3("Because.\n\nFood: Human: They want a quick solution.") == "Because."
    assert D.clean3("Fine for novels.\n\nThe Assistant: But how about") == "Fine for novels."
    assert D.clean3("Sure, here you go.\n\nYes\n\nHuman: next") == "Sure, here you go.\n\nYes"
    assert D.clean3("Human: How can I carry my own shopping b") == ""
    assert D.clean3("No markers here.") == "No markers here."


def test_relabel_with_gold_then_noise():
    pairs = [{"pair_id": f"train-{i}", **mk(i)} for i in range(40)]
    gold = {p["pair_id"]: {"gold_chosen": 1.0 if i % 2 == 0 else 0.0,
                           "gold_rejected": 0.0 if i % 2 == 0 else 1.0, "gold_model": "m"}
            for i, p in enumerate(pairs[:20])}
    out = D.relabel_with_gold(pairs, gold)
    assert len(out) == 20 and sum(q["relabeled"] for q in out) == 10
    noisy, _ = D.inject_label_noise(out, 0.25, 0)
    assert noisy[0]["flip_seed"] == 0 and noisy[0]["noise_rate"] == 0.25

def test_shared_first_turn_stays_in_one_split():
    pairs = []
    for i in range(300):
        first = f"\n\nHuman: opening {i % 60}\n\nAssistant: ok{i}"
        pr = first + f"\n\nHuman: follow-up {i}\n\nAssistant:"
        pairs.append({"pair_id": f"train-{i}", "chosen": pr + " good", "rejected": pr + " bad"})
    sp = D.split_pairs(pairs, 0)
    seen = {}
    for k, v in sp.items():
        for p in v:
            seen.setdefault(p["group"], set()).add(k)
    assert len(seen) == 60 and all(len(v) == 1 for v in seen.values())


def test_gold_ties_are_dropped_not_swapped():
    pairs = [{"pair_id": f"train-{i}", **mk(i)} for i in range(3)]
    gold = {"train-0": {"gold_chosen": 1.0, "gold_rejected": 1.0, "gold_model": "m"},
            "train-1": {"gold_chosen": 2.0, "gold_rejected": 1.0, "gold_model": "m"},
            "train-2": {"gold_chosen": 0.0, "gold_rejected": 1.0, "gold_model": "m"}}
    out = D.relabel_with_gold(pairs, gold)
    assert [q["pair_id"] for q in out] == ["train-1", "train-2"]
    assert out[1]["relabeled"] and out[1]["chosen"].endswith("bad2")


def test_test_guard_cannot_be_bypassed_by_split_name(tmp_path):
    p = tmp_path / "test.json"; D.save_split([1], p)
    with pytest.raises(PermissionError):
        D.load_split(p, "tune")
    assert D.load_split(p, "tune", allow_test=True) == [1]
    q = tmp_path / "tune_prompts.json"; D.save_split([2], q)
    assert D.load_split(q, "tune") == [2]


def test_manifest_deterministic_and_seed_sensitive():
    pairs = [{"pair_id": f"train-{i}", **mk(i)} for i in range(400)]
    m1 = D.split_manifest(D.split_pairs(pairs, 0), 0, D.DEFAULT_FRAC, "rev", "helpful-base")
    m2 = D.split_manifest(D.split_pairs(pairs, 0), 0, D.DEFAULT_FRAC, "rev", "helpful-base")
    m3 = D.split_manifest(D.split_pairs(pairs, 1), 1, D.DEFAULT_FRAC, "rev", "helpful-base")
    assert m1 == m2 and m1["sha256_prompt_ids"] != m3["sha256_prompt_ids"]
    assert set(m1["sha256_prompt_ids"]) == {"rm_train", "tune", "test"}

def test_flipped_flag_matches_mask_for_every_pair():
    pairs = [mk(i) for i in range(500)]
    new, flip = D.inject_label_noise(pairs, 0.25, 0)
    for i, (p, q) in enumerate(zip(pairs, new)):
        assert q["flipped"] == bool(flip[i])
        if flip[i]:
            assert (q["chosen"], q["rejected"]) == (p["rejected"], p["chosen"])
        else:
            assert (q["chosen"], q["rejected"]) == (p["chosen"], p["rejected"])
        assert q["flip_seed"] == 0 and q["noise_rate"] == 0.25
