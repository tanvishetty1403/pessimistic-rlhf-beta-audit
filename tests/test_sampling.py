import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
import json
import os

from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from src import sampling as S


class CharTok:
    """Offline stand-in tokenizer: char ids 1..250, pad/eos 0, decode maps ids back to letters."""
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"

    def __call__(self, text, return_tensors="pt", **kw):
        texts = [text] if isinstance(text, str) else text
        ids = [[(ord(c) % 250) + 1 for c in t] for t in texts]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(len(ids), len(ids[0]), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + int(i) % 26) if int(i) != 0 else "" for i in ids)


@pytest.fixture
def lm():
    torch.manual_seed(5)
    cfg = GPTNeoXConfig(vocab_size=300, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=256, pad_token_id=0, eos_token_id=0)
    return GPTNeoXForCausalLM(cfg).eval()


PROMPT = "\n\nHuman: hello\n\nAssistant:"


def test_sample_prompt_format_flags_and_determinism(lm):
    tok = CharTok()
    a = S.sample_prompt(lm, tok, PROMPT, 12, chunk=5, max_new_tokens=10, seed=3, device="cpu", log=lambda *_: None)
    b = S.sample_prompt(lm, tok, PROMPT, 12, chunk=5, max_new_tokens=10, seed=3, device="cpu", log=lambda *_: None)
    c = S.sample_prompt(lm, tok, PROMPT, 12, chunk=5, max_new_tokens=10, seed=4, device="cpu", log=lambda *_: None)
    assert a == b and a["responses"] != c["responses"]
    assert a["prompt"] == PROMPT and a["split"] == "tune" and len(a["prompt_id"]) == 12
    assert a["prompt_tokens"] == len(PROMPT) and len(a["responses"]) == 12
    ids = [r["response_id"] for r in a["responses"]]
    assert ids == [f"{a['prompt_id']}-{j:04d}" for j in range(12)]
    for r in a["responses"]:
        assert r["n_tokens"] <= 10
        if r["hit_length"]:
            assert r["n_tokens"] == 10                                  # no EOS: ran to the cap
        assert r["empty"] == (r["clean"] == "")
        assert r["hit_marker"] == (r["clean"] != r["raw"].strip())


def test_generate_chunked_halves_on_oom_and_returns_n():
    calls = []

    def gen(k, idx):
        calls.append(k)
        if k > 16:
            raise torch.cuda.OutOfMemoryError("fake")
        return [{"i": idx}] * k

    out = S.generate_chunked(gen, 100, 64, log=lambda *_: None)
    assert len(out) == 100 and max(c for c in calls[2:]) <= 16 and calls[:2] == [64, 32]


def test_generate_chunked_gives_up_at_min_chunk():
    def gen(k, idx):
        raise torch.cuda.OutOfMemoryError("always")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        S.generate_chunked(gen, 10, 16, min_chunk=8, log=lambda *_: None)


def test_run_sampling_shards_resume_skip_long_prompts_and_merge(lm, tmp_path):
    tok = CharTok()
    prompts = [f"\n\nHuman: question {i}\n\nAssistant:" for i in range(5)] + ["x" * 300]
    out = tmp_path / "run"
    pushed = []
    kw = dict(n=4, chunk=4, max_new_tokens=6, seed=1, shard_prompts=2, device="cpu", max_prompt_tokens=200,
              log=lambda *_: None)
    info = S.run_sampling(lm, tok, prompts, out, on_shard_done=pushed.append, **kw)
    shards = sorted(p.name for p in out.glob("shard_*.jsonl"))
    assert shards == ["shard_0000.jsonl", "shard_0001.jsonl", "shard_0002.jsonl"] and len(pushed) == 3
    assert info["n_usable"] == 5 and len(info["skipped"]) == 1 and not list(out.glob("*.tmp"))

    first = (out / "shard_0000.jsonl").read_text()
    stamp = (out / "shard_0000.jsonl").stat().st_mtime_ns
    (out / "shard_0001.jsonl").unlink()
    again = []
    S.run_sampling(lm, tok, prompts, out, on_shard_done=again.append, **kw)
    assert (out / "shard_0000.jsonl").stat().st_mtime_ns == stamp and len(again) == 1      # only shard 1 redone
    assert (out / "shard_0000.jsonl").read_text() == first

    n_rec = S.merge_shards(out, tmp_path / "all.jsonl")
    rows = [json.loads(l) for l in open(tmp_path / "all.jsonl")]
    assert n_rec == 5 == len(rows) and all(len(r["responses"]) == 4 for r in rows)
    rt = S.rates(rows)
    assert rt["responses"] == 20 and 0.0 <= rt["hit_length"] <= 1.0