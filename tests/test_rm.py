import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
import json
import numpy as np
from transformers import GPTNeoXConfig, GPTNeoXForSequenceClassification

from src import rm as R


class CharTok:
    """Offline stand-in for the HF tokenizer: char ids 1..250, pad 0, LEFT truncation, right padding."""
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    truncation_side = "left"

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=64):
        ids = [[(ord(c) % 250) + 1 for c in t] for t in texts]
        if truncation:
            ids = [x[-max_length:] for x in ids]
        L = max(len(x) for x in ids)
        return {"input_ids": torch.tensor([x + [0] * (L - len(x)) for x in ids]),
                "attention_mask": torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids])}


@pytest.fixture
def base_dir(tmp_path):
    torch.manual_seed(123)
    cfg = GPTNeoXConfig(vocab_size=300, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=256, num_labels=1, pad_token_id=0)
    GPTNeoXForSequenceClassification(cfg).save_pretrained(tmp_path / "base")
    return str(tmp_path / "base")


def make_pairs(n, start=0):
    out = []
    for i in range(start, start + n):
        pr = f"\n\nHuman: question {i}\n\nAssistant:"
        out.append({"pair_id": f"train-{i}", "chosen": pr + f" good answer {i}", "rejected": pr + f" bad reply {i}"})
    return out


LORA = dict(r=4, alpha=8, dropout=0.0)
KW = dict(epochs=6, bs=4, accum=1, lr=3e-3, max_length=64, device="cpu", log=lambda *_: None)


def test_join_prompt_response_matches_hh_format():
    pr = "\n\nHuman: hi\n\nAssistant:"
    assert R.join_prompt_response(pr, "  Hello there. ") == pr + " Hello there."


def test_scoring_is_batch_independent(base_dir):
    tok = CharTok()
    model = R.build_rm(base_dir, tok, 0, "cpu", **LORA)
    texts = [p["chosen"] for p in make_pairs(6)] + ["short", "x" * 70]
    batched = R.score_texts(model, tok, texts, max_length=64, batch_size=8, device="cpu")
    single = np.array([R.score_texts(model, tok, [t], max_length=64, batch_size=1, device="cpu")[0] for t in texts])
    assert np.allclose(batched, single, atol=1e-4)


def test_training_learns_a_synthetic_preference(base_dir):
    tok = CharTok()
    model = R.build_rm(base_dir, tok, 0, "cpu", **LORA)
    res = R.train_member(model, tok, make_pairs(160), make_pairs(60, start=1000), seed=0, **KW)
    assert res["history"][-1]["train_loss"] < res["history"][0]["train_loss"]
    assert res["val_acc"] > 0.85


def test_bootstrap_path_runs(base_dir):
    tok = CharTok()
    model = R.build_rm(base_dir, tok, 0, "cpu", **LORA)
    res = R.train_member(model, tok, make_pairs(40), make_pairs(20, start=500), seed=3,
                         diversity="bootstrap", **{**KW, "epochs": 1})
    assert len(res["history"]) == 1 and "val_acc" in res


def test_ensemble_train_resume_save_load_and_score(base_dir, tmp_path):
    tok = CharTok()
    out = tmp_path / "ens"
    build = lambda seed: R.build_rm(base_dir, tok, seed, "cpu", **LORA)
    train, val = make_pairs(120), make_pairs(40, start=1000)
    kw = {**KW, "epochs": 4}

    s = R.train_ensemble(train, val, tok, build, out, members=3, seed0=10, ensemble_id="t", device="cpu",
                         meta={"max_length": 64}, **{k: v for k, v in kw.items() if k != "device"})
    assert (out / "ensemble.json").exists() and all((out / f"member_{k}" / "adapter_config.json").exists() for k in range(3))
    assert len(s["member_val_acc"]) == 3 and "ensemble_val_acc" in s and "mean_pairwise_margin_corr" in s
    assert json.load(open(out / "ensemble.json"))["seeds"] == [10, 11, 12]

    # resume: a finished member is skipped, an unfinished one is retrained
    stamp = (out / "member_0" / "val_scores.json").stat().st_mtime_ns
    (out / "member_2" / "val_scores.json").unlink()
    R.train_ensemble(train, val, tok, build, out, members=3, seed0=10, ensemble_id="t", device="cpu",
                     **{k: v for k, v in kw.items() if k != "device"})
    assert (out / "member_0" / "val_scores.json").stat().st_mtime_ns == stamp
    assert (out / "member_2" / "val_scores.json").exists()

    # save/load round trip: scores from the reloaded adapter match the stored validation scores
    saved = json.load(open(out / "member_1" / "val_scores.json"))
    model = R.load_member(base_dir, out / "member_1", tok, "cpu")
    acc, c, r = R.eval_pairs(model, tok, val, max_length=64, batch_size=16, device="cpu")
    assert np.allclose(c, saved["val_chosen"], atol=1e-4) and np.allclose(r, saved["val_rejected"], atol=1e-4)

    # members differ (different LoRA init), and ensemble scoring returns [K, N]
    texts = [p["chosen"] for p in val[:5]]
    ms = R.score_ensemble(texts, [out / f"member_{k}" for k in range(3)], tok, base_dir,
                          max_length=64, batch_size=4, device="cpu")
    assert ms.shape == (3, 5) and not np.allclose(ms[0], ms[1])


def test_ensemble_stats_use_unbiased_variance():
    ms = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    mu, var = R.ensemble_stats(ms)
    assert np.allclose(mu, [3.0, 0.0]) and np.allclose(var, [4.0, 0.0])      # ddof=1
    assert np.allclose(R.ensemble_stats(ms[:1])[1], [0.0, 0.0])


def test_collect_texts_includes_empty_and_rows_align():
    recs = [{"prompt_id": "p1", "prompt": "\n\nHuman: a\n\nAssistant:",
             "responses": [{"response_id": "p1-0", "clean": " Fine.", "empty": False},
                           {"response_id": "p1-1", "clean": "", "empty": True},
                           {"response_id": "p1-2", "clean": "Ok", "empty": False}]}]
    texts, keys = R.collect_texts(recs)
    
    assert keys == [("p1", "p1-0"), ("p1", "p1-1"), ("p1", "p1-2")]
    assert texts[0].endswith("Assistant: Fine.")
    assert texts[1].endswith("Assistant: ")  # prompt + " "
    assert texts[2].endswith("Assistant: Ok")
    
    rows = R.score_rows(keys, np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), "t")
    assert rows[1]["member_scores"] == [2.0, 4.0] and rows[1]["response_id"] == "p1-1"
    assert rows[0]["var"] == 2.0


def test_scores_tensor_layout_ids_and_dense(tmp_path):
    recs = [{"prompt_id": f"p{i}", "prompt": "\n\nHuman: q\n\nAssistant:",
             "responses": [{"response_id": f"p{i}-{j}", "clean": "" if (i, j) == (1, 2) else f"a{i}{j}",
                            "empty": (i, j) == (1, 2)} for j in range(3)]} for i in range(2)]
    texts, keys = R.collect_texts(recs)
    
    assert len(keys) == 6
    
    ms = np.arange(2 * 6, dtype=np.float32).reshape(2, 6) + 1          # K=2, M=6
    Rr, pids, rids = R.scores_tensor(recs, ms, keys)
    
    # R is dense (no NaNs)
    assert Rr.shape == (2, 3, 2) and pids.tolist() == ["p0", "p1"] and rids.shape == (2, 3)
    assert rids[1].tolist() == ["p1-0", "p1-1", "p1-2"]
    assert not np.isnan(Rr).any()  # D's review: dense!
    
    # Check values
    assert Rr[0, 0].tolist() == [1.0, 7.0]
    assert Rr[1, 0].tolist() == [4.0, 10.0]
    assert Rr[1, 1].tolist() == [5.0, 11.0]
    assert Rr[1, 2].tolist() == [6.0, 12.0]  # This was the empty one, now scored
    
    np.savez(tmp_path / "s.npz", R=Rr, prompt_ids=pids, response_ids=rids)
    z = np.load(tmp_path / "s.npz")
    assert set(z.files) == {"R", "prompt_ids", "response_ids"} and z["R"].shape == (2, 3, 2)


def test_bootstrap_idx_range_and_determinism():
    a = R.bootstrap_idx(50, 1)
    assert len(a) == 50 and all(0 <= i < 50 for i in a)
    assert a == R.bootstrap_idx(50, 1) and a != R.bootstrap_idx(50, 2)