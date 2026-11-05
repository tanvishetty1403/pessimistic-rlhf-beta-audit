"""
tests/test_option_b.py
"""

import numpy as np
from src import data as D


def mk(n=1000):
    pairs, gold = [], {}
    for i in range(n):
        q = f"\n\nHuman: topic {i // 3}\n\nAssistant: r{i}\n\nHuman: follow {i}\n\nAssistant:"
        pairs.append({"pair_id": f"train-{i}", "chosen": q + f" good{i}", "rejected": q + f" bad{i}"})
        if i % 50 == 0:
            gc = gr = 0.0                                     # tie
        elif i % 10 < 7:
            gc, gr = 1.0, 0.0                                 # gold agrees with human
        else:
            gc, gr = 0.0, 1.0                                 # gold disagrees
        gold[f"train-{i}"] = dict(gold_chosen=gc, gold_rejected=gr, gold_model="mock")
    return pairs, gold


def test_relabel_then_flip_is_relative_to_gold():
    pairs, gold = mk()
    rel = D.relabel_with_gold(pairs, gold)
    assert len(rel) == len(pairs) - 20                        # 20 ties dropped
    orig = {p["pair_id"]: p for p in pairs}
    for p in rel:
        o = orig[p["pair_id"]]
        assert p["chosen"] == (o["rejected"] if p["relabeled"] else o["chosen"])
    out, flip = D.inject_label_noise(rel, 0.25, seed=0)
    assert abs(flip.mean() - 0.25) < 0.05
    pref = {p["pair_id"]: p["chosen"] for p in rel}           # gold-preferred text after relabel
    for p in out:
        assert (p["chosen"] != pref[p["pair_id"]]) == p["flipped"]


def test_noise_deterministic_and_zero_is_identity():
    pairs, gold = mk()
    rel = D.relabel_with_gold(pairs, gold)
    a, _ = D.inject_label_noise(rel, 0.25, seed=3)
    b, _ = D.inject_label_noise(rel, 0.25, seed=3)
    assert a == b
    z, f = D.inject_label_noise(rel, 0.0, seed=3)
    assert not f.any() and all(x["chosen"] == y["chosen"] for x, y in zip(z, rel))


def test_group_safe_split():
    pairs, _ = mk()
    sp = D.split_pairs(pairs, seed=0)
    seen = {}
    for s, ps in sp.items():
        for p in ps:
            seen.setdefault(p["group"], set()).add(s)
    assert all(len(v) == 1 for v in seen.values())