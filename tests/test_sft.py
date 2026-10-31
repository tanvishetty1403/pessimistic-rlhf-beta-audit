import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
import numpy as np
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from src import sft as F


class CharTok:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"

    def __call__(self, text, **kw):
        return {"input_ids": [(ord(c) % 250) + 1 for c in text]}


def mkpair(i, group=None, resp=None):
    pr = f"\n\nHuman: question {i}\n\nAssistant:"
    return {"pair_id": f"train-{i}", "group": group or f"g{i}", "prompt": pr,
            "chosen": pr + " " + (resp if resp is not None else f"yes {i}"), "rejected": pr + " no"}


def test_select_excludes_rm_pairs_and_their_groups_and_empties():
    pairs = [mkpair(i) for i in range(6)]
    pairs += [mkpair(100, group="g1"), mkpair(101, resp=""), mkpair(102), mkpair(103), mkpair(104)]
    tr, va, st = F.select_sft_pairs(pairs, skip=3, n=4, n_val=2)
    ids = {p["pair_id"] for p in tr + va}
    assert not ids & {"train-0", "train-1", "train-2"}                  # the RM train/val pairs
    assert "train-100" not in ids and "train-101" not in ids            # shares group g1 with an RM pair; empty reply
    assert len(tr) == 4 and len(va) == 2 and not {p["pair_id"] for p in tr} & {p["pair_id"] for p in va}
    assert st["dropped_shared_group_while_selecting"] == 1 and st["dropped_empty_while_selecting"] == 1
    assert F.select_sft_pairs(pairs, skip=3, n=4, n_val=2)[2] == st     # deterministic


def test_build_example_masks_prompt_appends_eos_and_does_not_teach_early_stop():
    tok = CharTok()
    ex = F.build_example(tok, mkpair(1), max_length=200, max_resp_tokens=50)
    p_len = len(tok(mkpair(1)["prompt"])["input_ids"])
    assert ex["labels"][:p_len] == [-100] * p_len and ex["input_ids"][-1] == 0 and ex["labels"][-1] == 0
    assert ex["input_ids"][p_len] == tok(" ")["input_ids"][0]           # the reply starts with the space
    cut = F.build_example(tok, mkpair(2, resp="x" * 90), max_length=300, max_resp_tokens=20)
    assert len(cut["input_ids"]) == p_len + 20 and cut["input_ids"][-1] != 0   # truncated reply: NO eos
    long_prompt = mkpair(3); long_prompt["prompt"] = "\n\nHuman: " + "w" * 300 + "\n\nAssistant:"
    long_prompt["chosen"] = long_prompt["prompt"] + " ok"
    lt = F.build_example(tok, long_prompt, max_length=100)
    assert len(lt["input_ids"]) == 100 and lt["labels"][-1] == 0         # left-truncated, reply kept


def test_collate_pads_and_masks():
    b = F.collate([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [9], "labels": [9]}], 0, "cpu")
    assert b["input_ids"].tolist() == [[5, 6, 7], [9, 0, 0]] and b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert b["labels"].tolist() == [[-100, 6, 7], [9, -100, -100]]


def test_training_lowers_loss_and_teaches_eos_and_checkpoint_roundtrip(tmp_path):
    tok = CharTok()
    torch.manual_seed(2)
    cfg = GPTNeoXConfig(vocab_size=300, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=256, pad_token_id=0, eos_token_id=0)
    model = GPTNeoXForCausalLM(cfg)
    train, _ = F.build_examples(tok, [mkpair(i, resp=f"answer {i % 7}") for i in range(120)])
    val, _ = F.build_examples(tok, [mkpair(1000 + i, resp=f"answer {i % 7}") for i in range(30)])
    eos0 = F.eos_prob_at_end(model, val, 0, 0, "cpu")
    hist = F.train_sft(model, tok, train, val, epochs=6, bs=4, accum=1, lr=3e-3, device="cpu", log=lambda *_: None)
    assert hist["epochs"][-1]["val_nll"] < hist["val_nll_before"]
    assert F.eos_prob_at_end(model, val, 0, 0, "cpu") > max(0.5, 10 * eos0)
    model.save_pretrained(tmp_path / "m")
    h1 = F.dir_sha256(tmp_path / "m")
    model.save_pretrained(tmp_path / "m2")
    assert h1 and h1 == F.dir_sha256(tmp_path / "m2")
    re = GPTNeoXForCausalLM.from_pretrained(tmp_path / "m").eval()
    x = torch.tensor([train[0]["input_ids"]])
    assert torch.allclose(model.eval()(x).logits, re(x).logits, atol=1e-5)