import json
import pathlib
import subprocess
import sys
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import analysis_utils as U  # noqa: E402


def test_roundtrip_and_cli(tmp_path):
    rng = np.random.default_rng(0)
    P, N, K = 8, 40, 5
    emp = rng.random((P, N)) < 0.05
    R, gold = rng.normal(size=(P, N, K)), rng.normal(size=(P, N))
    R[emp] = -9
    np.savez(tmp_path / "j.npz", R=R, gold=gold, empty=emp, split=np.array(["tune"] * P),
             group=np.array([f"g{i}" for i in range(P)]), prompt_ids=np.array([f"p{i}" for i in range(P)]),
             response_ids=np.array([[f"r{i}-{j}" for j in range(N)] for i in range(P)]))
    out = tmp_path / "o.npz"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_dropempty_npz.py"), "--npz", str(tmp_path / "j.npz"),
                        "--out", str(out), "--min_frac", "0.0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = U.load_npz(out)
    info = json.loads((tmp_path / "o.npz.drop_info.json").read_text())
    assert d["R"].shape == (P, info["n_after"], K) and not d["empty"].any() and (d["R"] != -9).all()
    assert d["split"] == "tune" and len(d["group"]) == P
    bad = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_dropempty_npz.py"), "--npz", str(tmp_path / "j.npz"),
                          "--out", str(tmp_path / "test_o.npz")], capture_output=True, text=True)
    assert bad.returncode != 0
