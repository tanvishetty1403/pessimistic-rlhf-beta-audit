import json
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import analysis_utils as U  # noqa: E402

P, N, K = 12, 64, 5


def make(path=None, split="tune", with_empty=True, seed=0, per_prompt_split=True):
    rng = np.random.default_rng(seed)
    gold = rng.normal(size=(P, N))
    R = gold[..., None] * 0.5 + rng.normal(size=(P, N, K))
    emp = np.zeros((P, N), bool)
    for p in range(P):
        k = 40 if p == 3 else int(rng.integers(0, 7))
        emp[p, rng.choice(N, k, replace=False)] = True
    R[emp], gold[emp] = -3.0, -3.0
    kw = dict(R=R, gold=gold, group=np.array([f"g{i // 2}" for i in range(P)]),
              split=np.array([split] * P) if per_prompt_split else np.array(split))
    if with_empty:
        kw["empty"] = emp
    if path:
        np.savez(path, **kw)
    return kw, emp


# split freeze guard 
def test_test_split_array_is_refused(tmp_path):
    make(tmp_path / "a.npz", split="test")
    with pytest.raises(PermissionError):
        U.load_npz(tmp_path / "a.npz")
    assert U.load_npz(tmp_path / "a.npz", allow_test=True)["split"] == "test"


def test_mixed_split_containing_test_refused(tmp_path):
    kw, _ = make()
    kw["split"] = np.array(["tune"] * (P - 1) + ["test"])
    np.savez(tmp_path / "m.npz", **kw)
    with pytest.raises(PermissionError):
        U.load_npz(tmp_path / "m.npz")


@pytest.mark.parametrize("per_prompt", [True, False])
def test_tune_split_ok(tmp_path, per_prompt):
    make(tmp_path / "t.npz", per_prompt_split=per_prompt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert U.load_npz(tmp_path / "t.npz")["split"] == "tune"


def test_old_str_of_array_would_have_bypassed_guard():
    assert str(np.array(["test"] * 3)) != "test" # the bug: the old check compared this to "test"


def test_empty_shape_checked(tmp_path):
    kw, emp = make()
    kw["empty"] = emp[:, :10]
    np.savez(tmp_path / "bad.npz", **kw)
    with pytest.raises(ValueError, match="empty mask"):
        U.load_npz(tmp_path / "bad.npz")


# exclude_prompt 
def test_exclude_prompt_with_and_without_mask(tmp_path):
    for with_empty in (True, False):
        make(tmp_path / "x.npz", with_empty=with_empty)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = U.load_npz(tmp_path / "x.npz")
        e = U.exclude_prompt(d, 4)
        assert e["R"].shape[0] == P - 1 and e["gold"].shape[0] == P - 1 and len(e["group"]) == P - 1
        assert (e["empty"] is None) == (not with_empty)
        assert np.array_equal(e["R"][4], d["R"][5]) and e["group"][4] == d["group"][5]
        assert d["R"].shape[0] == P                                      # input untouched


def test_exclude_prompt_bad_index():
    kw, _ = make()
    with pytest.raises(IndexError):
        U.exclude_prompt(dict(R=kw["R"], gold=kw["gold"], empty=None, group=None), P)


# drop_empty
def load(tmp_path):
    make(tmp_path / "d.npz")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return U.load_npz(tmp_path / "d.npz")


def test_drop_empty_semantics(tmp_path):
    d = load(tmp_path)
    with pytest.warns(UserWarning, match="removed prompts"):
        o = U.drop_empty(d)
    info = o["drop_info"]
    assert info["prompts_removed"] == [3] and o["R"].shape[0] == P - 1
    assert o["R"].shape == (P - 1, info["n_after"], K) and o["gold"].shape == o["R"].shape[:2]
    assert not o["empty"].any() and o["empty_frac"] == 0.0
    assert info["n_after"] == int((~d["empty"][np.arange(P) != 3]).sum(1).min())
    assert (o["R"] != -3.0).all()                                         # no empty row survived (R was -3 only for empties)
    keep = [p for p in range(P) if p != 3]
    for i, p in enumerate(keep):                                          # first N' valid samples, original order
        idx = np.flatnonzero(~d["empty"][p])[: info["n_after"]]
        assert np.array_equal(o["R"][i], d["R"][p][idx]) and np.array_equal(o["gold"][i], d["gold"][p][idx])
    assert list(o["group"]) == [d["group"][p] for p in keep]
    assert o["split"] == "tune"
    assert info["nonempty_samples_discarded"] >= 0 and d["R"].shape[0] == P


def test_drop_empty_min_frac_zero_keeps_all_prompts(tmp_path):
    o = U.drop_empty(load(tmp_path), min_frac=0.0)
    assert o["R"].shape[0] == P and o["drop_info"]["n_after"] == N - 40


def test_drop_empty_needs_mask(tmp_path):
    make(tmp_path / "n.npz", with_empty=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = U.load_npz(tmp_path / "n.npz")
    with pytest.raises(ValueError, match="no 'empty'|has none"):
        U.drop_empty(d)


def test_old_minus_1e9_trick_wrecks_standardization(tmp_path):
    d = load(tmp_path)
    R_old = d["R"].copy()
    R_old[d["empty"]] = -1e9
    old = U.fit_std(R_old, d["gold"], "within")
    new = U.fit_std(U.drop_empty(d, 0.0)["R"], U.drop_empty(d, 0.0)["gold"], "within")
    assert old["r_scale"] > 1e6 and new["r_scale"] < 10                   # standardized proxy collapses ~1e-8 vs O(1)


# end to end through g1_g3_plots.py 
def run_g1(tmp_path, npz, *extra):
    out = tmp_path / "figs"
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "g1_g3_plots.py"), "--npz", str(npz), "--out", str(out),
                           "--config", "none", "--B", "30", "--betas", "0.1", "2", "3", "--cs", "0.5", "1", *extra],
                          capture_output=True, text=True), out


def test_e2e_drop_empty_and_exclude(tmp_path):
    make(tmp_path / "noise_on.npz")
    r, out = run_g1(tmp_path, tmp_path / "noise_on.npz", "--drop_empty", "--exclude_prompt", "0")
    assert r.returncode == 0, r.stderr[-2000:]
    s = json.loads((out / "g1_g3_summary.json").read_text())["noise_on"]
    assert s["drop_empty"]["prompts_removed"] == [2] and s["excluded_prompt"] == 0   # old idx 3 -> 2 after excluding 0
    assert s["n_prompts"] == P - 2 and s["empty_frac"] == 0.0
    sd = json.loads((out / "std_stats.json").read_text())["noise_on"]
    assert sd["r_scale"] < 10 # sane standardization


def test_e2e_plain_run_unchanged(tmp_path):
    make(tmp_path / "noise_on.npz")
    r, out = run_g1(tmp_path, tmp_path / "noise_on.npz")
    assert r.returncode == 0, r.stderr[-2000:]
    s = json.loads((out / "g1_g3_summary.json").read_text())["noise_on"]
    assert s["n_prompts"] == P and s["empty_frac"] > 0 and s["drop_empty"] is None


def test_e2e_no_mask_exclude_works_and_drop_errors(tmp_path):
    make(tmp_path / "nomask.npz", with_empty=False)
    r, out = run_g1(tmp_path, tmp_path / "nomask.npz", "--exclude_prompt", "4")
    assert r.returncode == 0, r.stderr[-2000:]
    r2, _ = run_g1(tmp_path, tmp_path / "nomask.npz", "--drop_empty")
    assert r2.returncode != 0 and "has none" in r2.stderr


def test_e2e_test_npz_refused(tmp_path):
    make(tmp_path / "x.npz", split="test")
    r, _ = run_g1(tmp_path, tmp_path / "x.npz")
    assert r.returncode != 0 and "PermissionError" in r.stderr