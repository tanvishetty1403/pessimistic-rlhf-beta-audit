"""
src/data.py

HH-RLHF prompt extraction, leakage-free splits, label-noise injection. 
No network calls are made here.
"""
import hashlib
import json
import numpy as np
import re
import os

SPLITS = ("rm_train", "tune", "test")
DEFAULT_FRAC = (0.8, 0.1, 0.1)


def extract_prompt(chosen: str) -> str:
    """HH-RLHF: chosen/rejected share everything up to the last '\\n\\nAssistant:' turn."""
    k = chosen.rfind("\n\nAssistant:")
    return chosen[: k + len("\n\nAssistant:")] if k >= 0 else chosen


def assign_split(prompt: str, seed: int, frac=DEFAULT_FRAC) -> str:
    """Deterministic split by prompt hash, so the same prompt never lands in two splits."""
    h = int(hashlib.sha256(f"{seed}|{prompt}".encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < frac[0]:
        return "rm_train"
    return "tune" if h < frac[0] + frac[1] else "test"

def prompt_id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()[:12]


def hash_key(s: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}|{s}".encode()).hexdigest(), 16)


def order_by_hash(items, seed, key):
    """Reproducible pseudo-random order; the first n of a bigger n is nested inside."""
    return sorted(items, key=lambda x: hash_key(key(x), seed))


def filter_mismatched(pairs):
    """Drop pairs whose chosen/rejected prompts differ. Returns (kept, n_dropped)."""
    kept = [p for p in pairs if extract_prompt(p["chosen"]) == extract_prompt(p["rejected"])]
    return kept, len(pairs) - len(kept)


_TURN = re.compile(r"(?:^|\s)(?:Human|Assistant)\s*:")


def clean3(t):
    """Cut a sampled response at the first Human:/Assistant: marker, and drop a
    short dangling label (<= 3 words) left before a mid-line marker."""
    m = _TURN.search(t)
    if not m:
        return t.strip()
    head = t[: m.start()]
    if head and not head.endswith("\n"):
        k = head.rfind("\n\n")
        if k != -1 and len(head[k + 2:].split()) <= 3:
            head = head[:k]
    return head.strip()


def relabel_with_gold(pairs, gold):
    """gold: {pair_id: {gold_chosen, gold_rejected, gold_model}} = gold scores of the ORIGINAL
    human chosen/rejected texts. Keeps only scored pairs; swaps chosen/rejected when gold prefers
    the rejected text. Apply BEFORE inject_label_noise."""
    out = []
    for p in pairs:
        g = gold.get(p["pair_id"])
        if g is None:
            continue
        if g["gold_chosen"] == g["gold_rejected"]:
            continue                             # a tie carries no preference: drop, don't swap
        agree = g["gold_chosen"] > g["gold_rejected"]
        q = {**p, "gold_chosen": g["gold_chosen"], "gold_rejected": g["gold_rejected"],
             "gold_model": g["gold_model"], "gold_agrees_with_human": bool(agree), "relabeled": not agree}
        if not agree:
            q["chosen"], q["rejected"] = p["rejected"], p["chosen"]
        out.append(q)
    return out


def split_pairs(pairs, seed=0, frac=DEFAULT_FRAC):
    out = {s: [] for s in SPLITS}
    for p in pairs:
        pr = extract_prompt(p["chosen"])
        g = first_turn(pr)                       # split on the opening turn, not the full context
        out[assign_split(g, seed, frac)].append(
            {**p, "prompt": pr, "prompt_id": prompt_id(pr), "group": g})
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
        q["flip_seed"] = seed
        q["noise_rate"] = noise
        new.append(q)
    return new, flip


def save_split(pairs, path):
    with open(path, "w") as f:
        json.dump(pairs, f)


def load_split(path, split, allow_test=False):
    """Guard on the PATH as well as the label, so renaming the split argument doesn't bypass it."""
    if (split == "test" or "test" in os.path.basename(str(path)).lower()) and not allow_test:
        raise PermissionError("Test split is frozen until prereg is signed off. Use allow_test=True only in final_eval.")
    with open(path) as f:
        return json.load(f)

def _sha(items):
    return hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()


def split_manifest(sp, seed, frac, revision, data_dir, smoke_ids=(), tune_prompts=()):
    """Small, committable fingerprint of the frozen splits (no text)."""
    return {"dataset": "Anthropic/hh-rlhf", "data_dir": data_dir, "revision": revision,
            "seed": seed, "frac": list(frac),
            "n_pairs": {k: len(v) for k, v in sp.items()},
            "sha256_prompt_ids": {k: _sha({p["prompt_id"] for p in v}) for k, v in sp.items()},
            "sha256_smoke_ids": _sha(smoke_ids),
            "sha256_tune_prompts_selected": _sha([prompt_id(t) for t in tune_prompts])}

def first_turn(prompt: str) -> str:
    """Opening human turn of a conversation: the leakage-safe grouping key."""
    return prompt.split("\n\nAssistant:")[0].replace("\n\nHuman:", "", 1).strip()