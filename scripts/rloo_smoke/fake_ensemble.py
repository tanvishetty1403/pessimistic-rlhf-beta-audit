"""K=5 ensemble reward for the RLOO smoke test (scripts/rloo_smoke/).

- Scores come from a pluggable scorer:
    mock      deterministic hash-seeded uniforms, no model (plumbing only; rm_sec is meaningless)
    backbone  5 real Pythia forwards per batch with random heads (gives the real RM time/memory)
- ALL reward math goes through src.rloo_reward.compute_ensemble_reward (Khushi's module).
  It returns r~ only. KL is added by the TRL trainer, never here.
- Per-call diagnostics are stored in STATS keyed by TRAINING STEP, not list position, so the
  JSONL join survives resume, eval calls and steps with several reward calls.

When Khushi's PR changes the API (StdStats object), edit resolve_stats() and make_reward_fn() only.
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rloo_reward import compute_ensemble_reward

K = 5
STATS: dict = {}          # step -> list of per-call dicts


def clear_stats():
    STATS.clear()


def pop_step_stats(step):
    """Mean of the per-call stats for `step` (one call per step in the smoke config)."""
    calls = STATS.pop(step, None)
    if not calls:
        return {}
    out = {k: float(np.mean([c[k] for c in calls])) for k in calls[0]}
    out["n_reward_calls"] = len(calls)
    return out


def resolve_stats(rc):
    """Frozen normalization for this run. Real runs must come from stats_path."""
    path = rc.get("stats_path")
    if path:
        if rc.get("placeholder_stats"):
            raise ValueError("stats_path is set but placeholder_stats is true")
        from src.rloo_reward import load_std_stats
        center, scale, ens, noise = load_std_stats(path)
        return dict(r_center=center, r_scale=scale, ensemble_id=ens,
                    noise_condition=noise, placeholder_stats=False)
    if not rc.get("placeholder_stats"):
        raise ValueError("No stats_path and placeholder_stats is false: refusing to run "
                         "with unfrozen normalization")
    if rc["r_center"] != 0.0 or rc["r_scale"] != 1.0:
        raise ValueError("placeholder stats must be r_center=0, r_scale=1")
    return dict(r_center=0.0, r_scale=1.0, placeholder_stats=True,
                ensemble_id=rc["ensemble_id"], noise_condition=rc["noise_condition"])


# scorers
def mock_scores(prompts, completions, seed):
    """(B, K) uniforms, a pure function of (seed, prompt, completion): reproducible and resume-safe."""
    out = np.empty((len(completions), K))
    for i, (p, c) in enumerate(zip(prompts, completions)):
        h = hashlib.blake2b(f"{seed}\x00{p}\x00{c}".encode(), digest_size=8).digest()
        out[i] = np.random.default_rng(int.from_bytes(h, "little")).uniform(0.0, 1.0, K)
    return out


class BackboneScorer:
    """Random-head 'ensemble' on a frozen Pythia backbone: K real forward passes per batch."""

    def __init__(self, model_name, device="cuda", head_seed=123, noise=0.3):
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.F, self.device = torch, F, device
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="right")
        self.tok.pad_token = self.tok.eos_token
        self.base = AutoModel.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        d = self.base.config.hidden_size
        g = torch.Generator().manual_seed(head_seed)       # fixed: ensemble identity != run seed
        w0 = torch.randn(d, generator=g) / d ** 0.5
        self.heads = torch.stack([w0 + noise * torch.randn(d, generator=g) / d ** 0.5
                                  for _ in range(K)]).to(device)

    def __call__(self, prompts, completions):
        torch, F = self.torch, self.F
        texts = [p + str(c) for p, c in zip(prompts, completions)]
        with torch.no_grad():
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=256).to(self.device)
            last = enc["attention_mask"].sum(1) - 1
            cols = []
            for k in range(K):                              # K real backbone passes
                h = self.base(**enc).last_hidden_state[torch.arange(len(texts)), last]
                cols.append(F.layer_norm(h, h.shape[-1:]) @ self.heads[k])
            out = torch.stack(cols, dim=1).float().cpu().numpy()   # .cpu() syncs the GPU
        return out


# reward fn
def make_reward_fn(rc, scorer):
    """Returns reward_fn(prompts, completions, **kw) -> list[float] (r~ only, no KL)."""

    def ensemble_reward_fn(prompts, completions, **kwargs):
        t0 = time.perf_counter()
        scores = np.asarray(scorer(prompts, completions), dtype=np.float64)   # raw (B, K)
        rm_sec = time.perf_counter() - t0

        res = compute_ensemble_reward(
            scores,
            beta_policy=rc["beta_policy"], c=rc["c"],
            r_center=rc["r_center"], r_scale=rc["r_scale"],
            calibration_scale=rc.get("calibration_scale", 1.0),
            allow_placeholder_stats=rc["placeholder_stats"],
            ensemble_id=rc["ensemble_id"], noise_condition=rc["noise_condition"],
        )

        state = kwargs.get("trainer_state")
        # TRL calls the reward fn before global_step is incremented, so this is the logged step
        step = state.global_step + 1 if state is not None else len(STATS) + 1
        STATS.setdefault(step, []).append(dict(
            mu_mean=float(res.mean_reward.mean()),
            sigma2_mean=float(res.variance.mean()),
            penalty_mean=float(res.penalty.mean()),
            r_tilde_mean=float(res.pessimistic_reward.mean()),
            rm_sec=rm_sec,
        ))
        return [float(x) for x in res.pessimistic_reward]

    return ensemble_reward_fn