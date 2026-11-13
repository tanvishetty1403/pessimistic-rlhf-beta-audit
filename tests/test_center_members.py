import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src import analysis_utils as U  # noqa: E402
import ensemble_offset_diagnostic as D  # noqa: E402


def offset_ens(P=10, N=128, K=5, seed=0, off=(8.0, -2.0, 0.5, -5.0, 3.0), scale=(1.4, 0.6, 1.0, 0.5, 1.3), noise=0.15):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(P, N))
    R = np.stack([off[k] + scale[k] * s + noise * rng.normal(size=(P, N)) for k in range(K)], -1)
    return R, s


def test_default_is_unchanged():
    R, s = offset_ens()
    st = U.fit_std(R, s)
    assert "member_offsets" not in st
    assert np.allclose(U.apply_std(R, s, st)[0], (R - st["r_center"]) / st["r_scale"])


def test_center_members_removes_offsets_and_is_reused():
    R, s = offset_ens()
    st = U.fit_std(R, s, center_members=True)
    Rs, _ = U.apply_std(R, s, st)
    assert np.allclose(Rs.mean((0, 1)), 0.0, atol=1e-9) or np.allclose(Rs.mean((0, 1)), Rs.mean((0, 1))[0], atol=1e-9)
    R2, s2 = offset_ens(seed=5)                                   # a later split reuses the SAVED offsets
    Rb, _ = U.apply_std(R2, s2, st)
    assert np.allclose(Rb, (R2 - np.array(st["member_offsets"]) - st["r_center"]) / st["r_scale"])
    json.dumps(st)                                                 # serialisable for std_stats.json


def test_variance_after_centering_is_disagreement_not_offset():
    R, s = offset_ens()
    raw = R.var(-1, ddof=1).mean()
    st = U.fit_std(R, s, center_members=True)
    Rs, _ = U.apply_std(R, s, st)
    cen = (Rs * st["r_scale"]).var(-1, ddof=1).mean()               # undo the scalar scale
    assert raw > 15 and cen < 1.0


def test_wrong_k_offsets_rejected():
    R, s = offset_ens()
    st = U.fit_std(R, s, center_members=True)
    st["member_offsets"] = st["member_offsets"][:3]
    with pytest.raises(ValueError, match="member_offsets"):
        U.apply_std(R, s, st)


def test_diagnostic_flags_offset_dominated_ensemble():
    R, _ = offset_ens()
    d = D.diagnose(R)
    assert d["frac_offset"] > 0.9
    assert d["spearman_var_mu_raw"] > 0.5 and abs(d["spearman_var_mu_member_centered"]) < 0.3
    assert abs(d["spearman_raw_var_vs_centered_var"]) < 0.3


def test_diagnostic_clean_ensemble():
    R, _ = offset_ens(off=(0, 0, 0, 0, 0), scale=(1, 1, 1, 1, 1), noise=0.5)
    d = D.diagnose(R)
    assert d["frac_offset"] < 0.05


def test_diagnostic_cli_and_test_name_refused(tmp_path):
    R, _ = offset_ens()
    np.savez(tmp_path / "pilot.npz", R=R)
    np.savez(tmp_path / "test_pilot.npz", R=R)
    ok = subprocess.run([sys.executable, str(ROOT / "scripts" / "ensemble_offset_diagnostic.py"), "--npz",
                         str(tmp_path / "pilot.npz"), "--out", str(tmp_path / "o.json")], capture_output=True, text=True)
    assert ok.returncode == 0 and "frac" not in ok.stderr and (tmp_path / "o.json").exists()
    bad = subprocess.run([sys.executable, str(ROOT / "scripts" / "ensemble_offset_diagnostic.py"), "--npz",
                          str(tmp_path / "test_pilot.npz")], capture_output=True, text=True)
    assert bad.returncode != 0


def test_g1_center_members_end_to_end(tmp_path):
    P, N, K = 12, 64, 5
    R, s = offset_ens(P, N, K)
    np.savez(tmp_path / "x.npz", R=R, gold=s, split=np.array(["tune"] * P), group=np.array([f"g{i//2}" for i in range(P)]),
             empty=np.zeros((P, N), bool))
    out = tmp_path / "figs"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "g1_g3_plots.py"), "--npz", str(tmp_path / "x.npz"), "--out", str(out),
                        "--config", "none", "--B", "20", "--betas", "0.1", "2", "3", "--cs", "0.5", "1", "--center_members"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    st = json.loads((out / "std_stats.json").read_text())["x"]
    assert len(st["member_offsets"]) == K
    assert json.loads((out / "g1_g3_summary.json").read_text())["x"]["member_centered"] is True
