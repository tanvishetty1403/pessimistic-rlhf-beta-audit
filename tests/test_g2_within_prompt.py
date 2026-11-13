import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import g2_within_prompt as W  # noqa: E402

P, N, K = 40, 200, 5


def make(kind, seed=0):
    rng = np.random.default_rng(seed)
    gold = rng.normal(size=(P, N))
    if kind == "between":        # uncertainty differs by prompt only
        s = np.exp(rng.normal(size=(P, 1)) * 0.8) * np.ones((P, N))
    elif kind == "within":       # uncertainty differs by sample only
        s = np.exp(rng.normal(size=(P, N)) * 0.6)
    else:
        s = np.ones((P, N))
    shared = rng.normal(size=(P, N)) * 0.5 * s
    member = rng.normal(size=(P, N, K)) * 0.5 * s[..., None]
    offs = np.array([4.0, -3.0, 1.0, 0.0, 6.0])
    R = gold[..., None] + shared[..., None] + member + offs
    return dict(R=R, gold=gold, group=np.array([f"g{i}" for i in range(P)]))


def test_between_only_signal():
    r = W.analyse(make("between"), center_members=True, B=300)
    assert r["kappa_between"]["ci95"][0] > 0
    assert r["kappa_within"]["ci95"][0] < 0.15 and abs(r["kappa_within"]["value"]) < 0.3


def test_within_signal():
    r = W.analyse(make("within"), center_members=True, B=300)
    assert r["kappa_within"]["ci95"][0] > 0


def test_no_signal_cis_cover_zero():
    r = W.analyse(make("none"), center_members=True, B=300)
    assert r["kappa_within"]["ci95"][0] < 0 < r["kappa_within"]["ci95"][1]


def test_cli(tmp_path):
    import subprocess
    d = make("within")
    np.savez(tmp_path / "x.npz", R=d["R"], gold=d["gold"], split=np.array(["tune"] * P), group=d["group"])
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "g2_within_prompt.py"), "--npz", str(tmp_path / "x.npz"),
                        "--center_members", "--B", "50", "--out", str(tmp_path / "o.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "kappa_within" in r.stdout and (tmp_path / "o.json").exists()
    np.savez(tmp_path / "t.npz", R=d["R"], gold=d["gold"], split=np.array(["test"] * P), group=d["group"])
    bad = subprocess.run([sys.executable, str(ROOT / "scripts" / "g2_within_prompt.py"), "--npz", str(tmp_path / "t.npz"), "--B", "10"],
                         capture_output=True, text=True)
    assert bad.returncode != 0
