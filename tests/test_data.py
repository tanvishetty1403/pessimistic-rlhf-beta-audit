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
