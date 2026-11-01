"""
rloo_smoke/run_grid.py

Launcher only: one train_smoke.py run per (c, beta, seed). Grid values come from the CLI.
"""

import argparse, copy, itertools, json, subprocess, sys
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent


def last_ckpt_step(run_dir):
    cks = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return cks[-1] if cks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--c", type=float, nargs="+", required=True)
    ap.add_argument("--beta", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--cfg-dir", default="runs/_grid_cfgs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    base = yaml.safe_load(open(a.config))
    cfg_dir = Path(a.cfg_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for c, b, s in itertools.product(a.c, a.beta, a.seeds):
        cfg = copy.deepcopy(base)
        cfg["run_config"].update(c=c, beta_policy=b)
        cfg["rloo_config"]["seed"] = s
        name = f"c{c:g}_b{b:g}_s{s}"
        p = cfg_dir / f"{name}.yaml"
        yaml.safe_dump(cfg, open(p, "w"))
        run_dir = Path(cfg["output_root"]) / name
        ck = last_ckpt_step(run_dir) if run_dir.is_dir() else None
        cmd = [sys.executable, str(HERE / "train_smoke.py"), "--config", str(p)]
        if ck is not None:
            st = ck / "trainer_state.json"
            if st.exists() and json.load(open(st)).get("global_step", 0) >= cfg["rloo_config"]["max_steps"]:
                print("done, skip", name)
                continue
            cmd.append("--resume")
        elif cfg.get("hub_repo"):
            cmd.append("--resume")  # restore_from_hub in train_smoke; fails if nothing there
        print(name, "->", " ".join(cmd))
        if a.dry_run:
            continue
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
    if failed:
        sys.exit(f"FAILED: {failed}")


if __name__ == "__main__":
    main()