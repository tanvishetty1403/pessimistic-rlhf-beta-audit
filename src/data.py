"""
src/data.py

HH-RLHF prompt extraction, leakage-free splits, label-noise injection. 
No network calls are made here.
"""
import hashlib
import json
import numpy as np

SPLITS = ("rm_train", "tune", "test")


def extract_prompt(chosen: str) -> str:
    """HH-RLHF: chosen/rejected share everything up to the last '\\n\\nAssistant:' turn."""
    k = chosen.rfind("\n\nAssistant:")
    return chosen[: k + len("\n\nAssistant:")] if k >= 0 else chosen


def assign_split(prompt: str, seed: int, frac=(0.8, 0.1, 0.1)) -> str:
    """Deterministic split by prompt hash, so the same prompt never lands in two splits."""
    h = int(hashlib.sha256(f"{seed}|{prompt}".encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < frac[0]:
        return "rm_train"
    return "tune" if h < frac[0] + frac[1] else "test"


def split_pairs(pairs, seed=0, frac=(0.8, 0.1, 0.1)):
    """pairs: list of dicts with 'chosen','rejected'. Returns {split: [pair, ...]} with 'prompt' added."""
    out = {s: [] for s in SPLITS}
    for p in pairs:
        pr = extract_prompt(p["chosen"])
        out[assign_split(pr, seed, frac)].append({**p, "prompt": pr})
    return out


def inject_label_noise(pairs, noise: float, seed=0):
    """Swap chosen/rejected for a `noise` fraction of pairs (Coste-style). Returns (new_pairs, flipped_mask)."""
    rng = np.random.default_rng(seed)
    flip = rng.random(len(pairs)) < noise
    new = []
    for p, f in zip(pairs, flip):
        q = dict(p)
        if f:
            q["chosen"], q["rejected"] = p["rejected"], p["chosen"]
        q["flipped"] = bool(f)
        new.append(q)
    return new, flip


def save_split(pairs, path):
    with open(path, "w") as f:
        json.dump(pairs, f)


def load_split(path, split, allow_test=False):
    """Guard: the test split can only be opened with an explicit flag (only scripts/final_eval.py should set it)."""
    if split == "test" and not allow_test:
        raise PermissionError("Test split is frozen until prereg is signed off. Use allow_test=True only in final_eval.")
    with open(path) as f:
        return json.load(f)
