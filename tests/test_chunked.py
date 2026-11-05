"""
tests/test_chunked.pyx
"""

import pytest
from src.chunked import run_chunked


def test_resume_skips_done_and_guards_mismatch(tmp_path):
    ids = [f"p{i}" for i in range(10)]
    calls = []

    def fn(c):
        calls.append(list(c))
        return [{"id": i} for i in c]

    run_chunked(ids, fn, tmp_path, 4)
    assert len(calls) == 3
    (tmp_path / "shard_00001.jsonl").unlink() # simulate a dead session
    rows = run_chunked(ids, fn, tmp_path, 4)
    assert len(calls) == 4 and [r["id"] for r in rows] == ids
    with pytest.raises(SystemExit):
        run_chunked(ids[:-1], fn, tmp_path, 4)