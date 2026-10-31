import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "rloo_smoke"))
import run_manifest as RM  # noqa: E402

CFG = {"model_name": "m", "dataset_size": 16, "ensemble_mode": "mock", "hub_repo": None, "output_root": "a",
       "run_config": {"c": 1.0, "beta_policy": 0.05, "calibration_scale": 1.0, "stats_path": None},
       "lora": {"r": 8},
       "rloo_config": {"max_steps": 4, "learning_rate": 5e-4, "seed": 0, "report_to": "none", "logging_steps": 1}}
ST = SimpleNamespace(r_center=0.0, r_scale=1.0, ensemble_id="e", noise_condition="noise_off")


def test_same_config_passes(tmp_path):
    m = RM.build_manifest(CFG, ST)
    RM.write_manifest(tmp_path, m)
    RM.check_resume(tmp_path, RM.build_manifest(copy.deepcopy(CFG), ST))


@pytest.mark.parametrize("path,val", [(("rloo_config", "max_steps"), 8), (("rloo_config", "learning_rate"), 1e-4),
                                      (("run_config", "c"), 2.0), (("run_config", "beta_policy"), 0.5),
                                      (("rloo_config", "seed"), 1)])
def test_changes_refused(tmp_path, path, val):
    RM.write_manifest(tmp_path, RM.build_manifest(CFG, ST))
    c2 = copy.deepcopy(CFG)
    c2[path[0]][path[1]] = val
    with pytest.raises(SystemExit):
        RM.check_resume(tmp_path, RM.build_manifest(c2, ST))


def test_excluded_fields_ok(tmp_path):
    RM.write_manifest(tmp_path, RM.build_manifest(CFG, ST))
    c2 = copy.deepcopy(CFG)
    c2["hub_repo"] = "x/y"
    c2["rloo_config"]["logging_steps"] = 5
    RM.check_resume(tmp_path, RM.build_manifest(c2, ST))


def test_missing_manifest_refused(tmp_path):
    with pytest.raises(SystemExit):
        RM.check_resume(tmp_path, RM.build_manifest(CFG, ST))


def test_trainer_state_max_steps(tmp_path):
    RM.write_manifest(tmp_path, RM.build_manifest(CFG, ST))
    ck = tmp_path / "checkpoint-2"
    ck.mkdir()
    (ck / "trainer_state.json").write_text(json.dumps({"max_steps": 99}))
    with pytest.raises(SystemExit):
        RM.check_resume(tmp_path, RM.build_manifest(CFG, ST), ckpt=ck)


def test_allow_override(tmp_path):
    RM.write_manifest(tmp_path, RM.build_manifest(CFG, ST))
    c2 = copy.deepcopy(CFG)
    c2["lora"]["r"] = 16
    RM.check_resume(tmp_path, RM.build_manifest(c2, ST), allow=("lora.",))
