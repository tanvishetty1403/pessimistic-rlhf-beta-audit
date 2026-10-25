"""Reward-model utilities: LoRA RM, pairwise training, K-member ensembles, scoring.

Heavy imports (torch, transformers, peft) live here only, so src.data stays light.
Conventions
- A pair's `chosen` text is the preferred one *as labeled for training*. In rm_train_noisy.json the label
  noise has already swapped chosen/rejected, so no label handling happens here.
- Texts are full conversations: prompt (ending in "\n\nAssistant:") + " " + response.
- Truncation is from the LEFT, so the end of the response is always kept. Training and scoring must use the
  same max_length; train_ensemble records it in ensemble.json and scripts/score_samples.py reads it back.
"""
import contextlib
import gc
import json
import math
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F


def bootstrap_idx(n, member_seed):
    """Indices of a bootstrap resample (with replacement), used when diversity == "bootstrap"."""
    return np.random.default_rng(member_seed).integers(0, n, size=n).tolist()


def join_prompt_response(prompt, response):
    """HH formatting: the prompt ends in '\n\nAssistant:' and the response follows after one space."""
    return prompt + " " + response.strip()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    return tok


def wrap_lora(model, r=16, alpha=32, dropout=0.05, target_modules=("query_key_value",)):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=list(target_modules), task_type="SEQ_CLS")
    return get_peft_model(model, cfg)


def build_rm(name, tok, seed, device, **lora):
    """Pretrained backbone + fresh scalar head + LoRA. The seed fixes the head and LoRA init."""
    from transformers import AutoModelForSequenceClassification
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    return wrap_lora(model, **lora).to(device)


def load_member(base, adapter_dir, tok, device):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1, dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    return PeftModel.from_pretrained(model, adapter_dir).to(device).eval()


def encode(tok, texts, max_length, device):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}


def _autocast(amp):
    return torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()


def rewards(model, enc, amp):
    with _autocast(amp):
        out = model(**enc).logits
    return out.squeeze(-1).float()


@torch.no_grad()
def score_texts(model, tok, texts, max_length=512, batch_size=32, device="cuda"):
    """One reward per text. Sorted by length for less padding; order is restored."""
    amp = str(device).startswith("cuda")
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros(len(texts), dtype=np.float32)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = encode(tok, [texts[i] for i in idx], max_length, device)
        out[idx] = rewards(model, enc, amp).cpu().numpy()
    return out


def eval_pairs(model, tok, pairs, max_length=512, batch_size=32, device="cuda"):
    """Accuracy of reward(chosen) > reward(rejected), plus the raw scores."""
    n = len(pairs)
    sc = score_texts(model, tok, [p["chosen"] for p in pairs] + [p["rejected"] for p in pairs],
                     max_length, batch_size, device)
    c, r = sc[:n], sc[n:]
    return float((c > r).mean()), c, r


def train_member(model, tok, train_pairs, val_pairs=None, *, seed=0, epochs=2, bs=4, accum=4, lr=1e-4,
                 max_length=512, warmup_frac=0.05, diversity="seeds", device="cuda", log=print):
    """Pairwise (Bradley-Terry) training of one LoRA member. Effective batch = bs * accum pairs."""
    amp = str(device).startswith("cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n = len(train_pairs)
    base_idx = bootstrap_idx(n, seed) if diversity == "bootstrap" else list(range(n))
    micro = math.ceil(n / bs)
    total = math.ceil(micro / accum) * epochs
    warm = max(1, int(warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        idx = np.random.default_rng(seed * 1000 + ep).permutation(base_idx)
        losses, accs = [], []
        opt.zero_grad(set_to_none=True)
        for m in range(micro):
            ps = [train_pairs[i] for i in idx[m * bs:(m + 1) * bs]]
            k = len(ps)
            enc = encode(tok, [p["chosen"] for p in ps] + [p["rejected"] for p in ps], max_length, device)
            r = rewards(model, enc, amp)
            rc, rr = r[:k], r[k:]
            loss = -F.logsigmoid(rc - rr).mean()
            scaler.scale(loss / accum).backward()
            losses.append(loss.item())
            accs.append((rc > rr).float().mean().item())
            if (m + 1) % accum == 0 or m == micro - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
        rec = {"epoch": ep + 1, "train_loss": float(np.mean(losses)), "train_acc": float(np.mean(accs)),
               "seconds": round(time.time() - t0, 1)}
        if val_pairs:
            rec["val_acc"], vc, vr = eval_pairs(model, tok, val_pairs, max_length, 32, device)
        history.append(rec)
        log(f"  seed {seed} epoch {ep + 1}/{epochs}: " + ", ".join(f"{k} {v}" for k, v in rec.items() if k != "epoch"))
    out = {"history": history}
    if val_pairs:
        out.update({"val_acc": history[-1]["val_acc"], "val_chosen": vc.tolist(), "val_rejected": vr.tolist()})
    return out


def summarize_ensemble(out_dir, members):
    """Per-member and ensemble validation accuracy, plus how correlated the members' margins are."""
    runs = [json.load(open(pathlib.Path(out_dir) / f"member_{k}" / "val_scores.json")) for k in range(members)]
    if "val_chosen" not in runs[0]:
        return {"members": members}
    c = np.array([r["val_chosen"] for r in runs])
    rj = np.array([r["val_rejected"] for r in runs])
    margins = c - rj
    s = {"member_val_acc": [r["val_acc"] for r in runs],
         "ensemble_val_acc": float(((c.mean(0)) > rj.mean(0)).mean())}
    if members > 1:
        cc = np.corrcoef(margins)
        s["mean_pairwise_margin_corr"] = float(cc[np.triu_indices(members, 1)].mean())
    return s


def train_ensemble(train_pairs, val_pairs, tok, build, out_dir, *, members=5, seed0=0, diversity="seeds",
                   ensemble_id="ens", meta=None, device="cuda", on_member_done=None, log=print, **train_kw):
    """Train K members one after another. A finished member is marked by its val_scores.json, so a rerun
    after a Colab disconnect skips finished members and resumes at the first unfinished one."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [seed0 + k for k in range(members)]
    for k, seed in enumerate(seeds):
        mdir = out / f"member_{k}"
        if (mdir / "val_scores.json").exists():
            log(f"member {k} (seed {seed}): already done, skipping")
            continue
        log(f"member {k} (seed {seed}): training")
        model = build(seed)
        res = train_member(model, tok, train_pairs, val_pairs, seed=seed, diversity=diversity,
                           device=device, log=log, **train_kw)
        model.save_pretrained(mdir)                                  # adapter + score head
        json.dump({"seed": seed, **res}, open(mdir / "val_scores.json", "w"))   # written last = done marker
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if on_member_done:
            on_member_done(mdir)
    summary = summarize_ensemble(out, members)
    info = {**(meta or {}), "ensemble_id": ensemble_id, "members": members, "seeds": seeds,
            "diversity": diversity, "n_train": len(train_pairs), "n_val": len(val_pairs or []),
            "train_kw": train_kw, "summary": summary}
    json.dump(info, open(out / "ensemble.json", "w"), indent=2)
    return summary


def score_with(texts, loaders, tok, *, max_length=512, batch_size=32, device="cuda"):
    """Scores of every member for every text, shape [K, N]. `loaders` are zero-argument callables that return
    a model; ONE member is alive at a time."""
    rows = []
    for load in loaders:
        model = load()
        rows.append(score_texts(model, tok, texts, max_length, batch_size, device))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.stack(rows)


def score_ensemble(texts, member_dirs, tok, base, *, max_length=512, batch_size=32, device="cuda"):
    """Same as score_with, for trained members saved by train_ensemble."""
    loaders = [(lambda d=d: load_member(base, d, tok, device)) for d in member_dirs]
    return score_with(texts, loaders, tok, max_length=max_length, batch_size=batch_size, device=device)


def ensemble_stats(member_scores):
    """mu and the unbiased variance (ddof=1, as in Hahami's estimator) across members, per text."""
    k = member_scores.shape[0]
    mu = member_scores.mean(0)
    var = member_scores.var(0, ddof=1) if k > 1 else np.zeros_like(mu)
    return mu, var


def collect_texts(records):
    """records: [{prompt_id, prompt, responses:[{response_id, clean, ...}]}]. Empty responses are INCLUDED
    and scored as `prompt + ' '`. Returns (texts, keys) with keys = (prompt_id, response_id)."""
    texts, keys = [], []
    for rec in records:
        for r in rec["responses"]:
            texts.append(join_prompt_response(rec["prompt"], r["clean"]))
            keys.append((rec["prompt_id"], r["response_id"]))
    return texts, keys


def score_rows(keys, member_scores, ensemble_id):
    mu, var = ensemble_stats(member_scores)
    return [{"prompt_id": pid, "response_id": rid, "ensemble_id": ensemble_id,
             "member_scores": member_scores[:, i].tolist(), "mu": float(mu[i]), "var": float(var[i])}
            for i, (pid, rid) in enumerate(keys)]


def scores_tensor(records, member_scores, keys):
    """Scatter the [K, M] scores back into R[P, N, K], in file order. Now rectangular and dense (no NaNs)
    because empties are scored too. Needs the same number of responses N for every prompt."""
    P, K = len(records), member_scores.shape[0]
    N = len(records[0]["responses"])
    if any(len(r["responses"]) != N for r in records):
        raise ValueError("the .npz layout needs the same number of responses for every prompt")
    pos = {(rec["prompt_id"], r["response_id"]): (i, j)
           for i, rec in enumerate(records) for j, r in enumerate(rec["responses"])}
    out = np.zeros((P, N, K), dtype=np.float32)
    for m, key in enumerate(keys):
        i, j = pos[key]
        out[i, j, :] = member_scores[:, m]
    prompt_ids = np.array([rec["prompt_id"] for rec in records])
    response_ids = np.array([[r["response_id"] for r in rec["responses"]] for rec in records])
    return out, prompt_ids, response_ids