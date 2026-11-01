import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "rloo_smoke"):
    sys.path.insert(0, str(p))

import rloo_eval as E
import run_manifest as RM

PROMPTS = ["\n\nHuman: hi\n\nAssistant:", "\n\nHuman: bye\n\nAssistant:"]


# pool_identity
def test_pool_none_smoke_ok():
    assert RM.pool_identity({}, real_run=False) is None


def test_pool_none_real_run_refused():
    with pytest.raises(SystemExit):
        RM.pool_identity({}, real_run=True)


def test_pool_hash_changes_with_file(tmp_path):
    f = tmp_path / "pool.json"
    f.write_text("a")
    h1 = RM.pool_identity({"prompt_pool_manifest": str(f)}, real_run=True)
    f.write_text("b")
    h2 = RM.pool_identity({"prompt_pool_manifest": str(f)}, real_run=True)
    assert h1 and h2 and h1 != h2


# check_meta
def test_check_meta_missing(tmp_path):
    with pytest.raises(SystemExit):
        E.check_meta(tmp_path, PROMPTS)


def test_check_meta_match_and_mismatch(tmp_path):
    (tmp_path / "eval_meta.json").write_text(json.dumps({"prompt_sha256": E.prompts_sha(PROMPTS)}))
    E.check_meta(tmp_path, PROMPTS) # no exit
    with pytest.raises(SystemExit):
        E.check_meta(tmp_path, PROMPTS[:1]) # different prompt set
    with pytest.raises(SystemExit):
        E.check_meta(tmp_path, PROMPTS[::-1]) # same prompts, different order


# gold cache
def test_gold_cache_model_check(tmp_path):
    f = tmp_path / "gold_cache.jsonl"
    f.write_text(json.dumps({"k": "x", "gold": 1.5, "gold_model": "A@abc"}) + "\n")
    assert E.load_gold_cache(f, "A@abc") == {"x": 1.5}
    with pytest.raises(SystemExit):
        E.load_gold_cache(f, "A@def")                     # different revision
    assert E.load_gold_cache(tmp_path / "nope.jsonl", "A@abc") == {}


def test_gold_id_mock_and_missing_revision():
    assert E.gold_id(SimpleNamespace(mock_gold=True, gold_revision=None)) == "MOCK"
    with pytest.raises(SystemExit):
        E.gold_id(SimpleNamespace(mock_gold=False, gold_revision=None))


# KL token boundary (n_clean_tokens)
class FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(ids)


def test_n_clean_tokens_marker_cut():
    g = ["hello", " world", " Human:", " x"]
    assert E.n_clean_tokens(FakeTok(), g, "hello world") == 2


def test_n_clean_tokens_empty():
    assert E.n_clean_tokens(FakeTok(), ["Human:", " x"], "") == 0