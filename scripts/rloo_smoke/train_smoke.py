"""scripts/rloo_smoke/train_smoke.py

RLOO smoke test: generate -> K=5 reward (r~ only) -> KL (TRL) -> update -> checkpoint -> resume.

    python scripts/rloo_smoke/train_smoke.py                 # fresh run
    python scripts/rloo_smoke/train_smoke.py --resume        # continue from last local/Hub checkpoint

Notes
- TRL RLOO: reward = r~ - beta * KL_seq (sequence-summed k1). beta = run_config.beta_policy.
- The logged TRL `kl` is a per-token mean; kl_seq = kl * completions/mean_length.
- The logged TRL `reward` is PRE-KL. reward_post_kl = reward_pre_kl - beta * kl_seq.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import peft
import torch
import transformers
import trl
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import RLOOConfig, RLOOTrainer

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fake_ensemble as FE  # noqa: E402

K5_KEYS = ["mu_mean", "sigma2_mean", "penalty_mean", "r_tilde_mean", "rm_sec", "n_reward_calls"]


def git_info():
    try:
        def run(*a):
            return subprocess.check_output(["git", *a], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
        return run("rev-parse", "HEAD"), bool(run("status", "--porcelain"))
    except Exception:
        return "not_in_git_repo", None


# ----------------------------------------------------------------------------- callbacks
class JsonlLogger(TrainerCallback):
    """Appends one row per logged step. Resume-safe: rows past the restored step are dropped first."""

    def __init__(self, path, identity, beta, seqs_per_step):
        self.path, self.identity, self.beta, self.seqs_per_step = Path(path), identity, beta, seqs_per_step
        self.t0 = None
        self.step_sec = None

    def on_train_begin(self, args, state, control, **kw):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keep = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip() and json.loads(line)["step"] <= state.global_step:
                    keep.append(line)
        self.path.write_text("".join(l + "\n" for l in keep))

    def on_step_begin(self, args, state, control, **kw):
        self.t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kw):
        self.step_sec = time.perf_counter() - self.t0

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs or "kl" not in logs:
            return
        step = state.global_step
        kl_tok = logs["kl"]
        mean_len = logs.get("completions/mean_length")
        kl_seq = kl_tok * mean_len if mean_len is not None else None
        pre = logs.get("reward")
        pen = self.beta * kl_seq if kl_seq is not None else None
        k5 = {k: None for k in K5_KEYS}
        k5.update(FE.pop_step_stats(step))
        row = {
            "step": step, "seed": args.seed, **self.identity,
            "kl_tok": kl_tok, "mean_length": mean_len, "kl_seq": kl_seq,
            "reward_pre_kl": pre, "kl_penalty": pen,
            "reward_post_kl": (pre - pen) if (pre is not None and pen is not None) else None,
            "loss": logs.get("loss"), "grad_norm": logs.get("grad_norm"),
            **k5,
            "step_sec": self.step_sec,
            "tokens_per_sec": (self.seqs_per_step * mean_len / self.step_sec)
            if (mean_len is not None and self.step_sec) else None,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


class HubSync(TrainerCallback):
    """Push each checkpoint (+ log.jsonl) to the Hub. Fails loudly in the console, never kills training."""

    def __init__(self, repo, run_name, log_path):
        from huggingface_hub import HfApi
        self.api, self.repo, self.run, self.log = HfApi(), repo, run_name, Path(log_path)

    def on_save(self, args, state, control, **kw):
        ck = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        try:
            self.api.upload_folder(folder_path=str(ck), repo_id=self.repo, path_in_repo=f"{self.run}/{ck.name}")
            if self.log.exists():
                self.api.upload_file(path_or_fileobj=str(self.log), repo_id=self.repo,
                                     path_in_repo=f"{self.run}/log.jsonl")
            print(f"[hub] pushed {ck.name} -> {self.repo}/{self.run}")
        except Exception as e:  # noqa: BLE001
            print(f"!!! [hub] PUSH FAILED for {ck.name}: {e}")


def restore_from_hub(repo, run_name, out_root):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, local_dir=str(out_root), allow_patterns=[f"{run_name}/*"])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "rloo_smoke.yaml"))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    rc = {**cfg["run_config"], **FE.resolve_stats(cfg["run_config"])}   # frozen/placeholder stats resolved
    seed = cfg["rloo_config"]["seed"]
    run_name = f"c{rc['c']:g}_b{rc['beta_policy']:g}_s{seed}"
    out_root = Path(cfg["output_root"])
    out_dir = out_root / run_name
    log_path = out_dir / "log.jsonl"
    hub_repo = cfg.get("hub_repo")

    # --- checkpoint policy: never pick up a stale checkpoint silently
    ckpt = get_last_checkpoint(str(out_dir)) if out_dir.is_dir() else None
    if a.resume and ckpt is None and hub_repo:
        restore_from_hub(hub_repo, run_name, out_root)
        ckpt = get_last_checkpoint(str(out_dir)) if out_dir.is_dir() else None
    if a.resume and ckpt is None:
        raise SystemExit(f"--resume given but no checkpoint found in {out_dir}")
    if not a.resume and ckpt is not None:
        raise SystemExit(f"{out_dir} already has {Path(ckpt).name}; pass --resume or change the config")

    git_commit, git_dirty = git_info()
    identity = {
        **rc, "run_name": run_name, "ensemble_mode": cfg["ensemble_mode"], "K": FE.K,
        "versions": {"trl": trl.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__, "torch": torch.__version__},
        "git_commit": git_commit, "git_dirty": git_dirty,
    }

    tok = AutoTokenizer.from_pretrained(cfg["model_name"], padding_side="left")
    tok.pad_token = tok.eos_token
    dataset = Dataset.from_dict({"prompt": [f"Write a short sentence about topic {i}."
                                            for i in range(cfg["dataset_size"])]})

    if cfg["ensemble_mode"] == "mock":
        scorer = lambda p, c: FE.mock_scores(p, c, seed)          # noqa: E731
    elif cfg["ensemble_mode"] == "backbone":
        scorer = FE.BackboneScorer(cfg["model_name"])
    else:
        raise ValueError("ensemble_mode must be mock or backbone")
    reward_fn = FE.make_reward_fn(rc, scorer)

    args = RLOOConfig(output_dir=str(out_dir), beta=rc["beta_policy"], **cfg["rloo_config"])
    rl = cfg["rloo_config"]
    seqs_per_step = rl["per_device_train_batch_size"] * rl.get("gradient_accumulation_steps", 1)

    callbacks = [JsonlLogger(log_path, identity, rc["beta_policy"], seqs_per_step)]
    if hub_repo:
        from huggingface_hub import HfApi
        HfApi().create_repo(hub_repo, private=True, exist_ok=True)
        callbacks.append(HubSync(hub_repo, run_name, log_path))

    FE.clear_stats()
    trainer = RLOOTrainer(model=cfg["model_name"], reward_funcs=reward_fn, args=args,
                          train_dataset=dataset, processing_class=tok,
                          peft_config=LoraConfig(**cfg["lora"]), callbacks=callbacks)

    print(f"=== {'RESUMING ' + Path(ckpt).name if ckpt else 'FRESH RUN'}: {run_name} ===")
    trainer.train(resume_from_checkpoint=ckpt)
    print(f"done. log: {log_path}")


if __name__ == "__main__":
    main()