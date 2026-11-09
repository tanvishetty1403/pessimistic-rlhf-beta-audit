import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_option_b.py"
sys.path.insert(0, str(ROOT))
from src import data as D  # noqa: E402


def pair(i):
    pr = f"\n\nHuman: q{i}\n\nAssistant:"
    return {"pair_id": f"train-{i}", "chosen": f"{pr} a{i}", "rejected": f"{pr} b{i}"}


def setup(tmp_path, n=100, ties=(), unscored=()):
    pairs = [pair(i) for i in range(n)]
    (tmp_path / "pairs.json").write_text(json.dumps(pairs))
    rows = []
    for i in range(n):
        if f"train-{i}" in unscored:
            continue
        c, r = (1.0, 1.0) if f"train-{i}" in ties else ((1.0, 0.0) if i % 3 else (0.0, 1.0))
        rows.append({"pair_id": f"train-{i}", "gold_chosen": c, "gold_rejected": r, "gold_model": "m@x"})
    (tmp_path / "gold.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def build(d, *extra, out="out.json", noise="0.25"):
    return subprocess.run([sys.executable, str(SCRIPT), "--pairs", str(d / "pairs.json"), "--gold", str(d / "gold.jsonl"),
                           "--noise", noise, "--out", str(d / out), *extra], capture_output=True, text=True)


def test_default_n_is_whole_file(tmp_path):
    d = setup(tmp_path, n=2500)
    r = build(d)
    assert r.returncode == 0, r.stderr
    m = json.loads((d / "out.json.meta.json").read_text())
    assert m["pairs_in"] == 2500 and m["kept"] == 2500 and m["unscored"] == 0 and m["ties_dropped"] == 0


def test_n_prefix_and_too_large(tmp_path):
    d = setup(tmp_path, n=50)
    assert build(d, "--n", "20").returncode == 0
    assert json.loads((d / "out.json.meta.json").read_text())["pairs_in"] == 20
    r = build(d, "--n", "51")
    assert r.returncode != 0 and "--n 51" in r.stderr


def test_ties_counted_and_budget(tmp_path):
    d = setup(tmp_path, ties={"train-1", "train-2", "train-3"})
    assert build(d, "--max_dropped", "3").returncode == 0
    m = json.loads((d / "out.json.meta.json").read_text())
    assert m["ties_dropped"] == 3 and m["kept"] == 97 and m["dropped"] == 3
    r = build(d, "--max_dropped", "2", out="o2.json")
    assert r.returncode != 0 and "max_dropped" in r.stderr and not (d / "o2.json").exists()


def test_unscored_fatal_with_budget_only(tmp_path):
    d = setup(tmp_path, unscored={"train-7"})
    r = build(d, "--max_dropped", "50", out="o2.json")
    assert r.returncode != 0 and "no gold score" in r.stderr and not (d / "o2.json").exists()
    assert build(d).returncode == 0                                   # old behaviour without the flag
    assert json.loads((d / "out.json.meta.json").read_text())["unscored"] == 1


def test_n0_n25_same_ids_flip_only(tmp_path):
    d = setup(tmp_path, ties={"train-4"})
    assert build(d, out="n0.json", noise="0").returncode == 0
    assert build(d, out="n25.json", noise="0.25").returncode == 0
    a, b = json.loads((d / "n0.json").read_text()), json.loads((d / "n25.json").read_text())
    assert [p["pair_id"] for p in a] == [p["pair_id"] for p in b]
    assert sum(p["flipped"] for p in b) > 0 and not any(p["flipped"] for p in a)


def test_test_named_out_refused(tmp_path):
    d = setup(tmp_path)
    assert build(d, out="gold_test.json").returncode != 0


def test_duplicate_pair_id(tmp_path):
    d = setup(tmp_path)
    p = json.loads((d / "pairs.json").read_text()); p.append(p[0])
    (d / "pairs.json").write_text(json.dumps(p))
    assert build(d).returncode != 0
