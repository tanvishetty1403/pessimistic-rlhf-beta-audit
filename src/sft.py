import hashlib
import json
import math
import pathlib
import time

import numpy as np
import torch


def response_of(pair):
    return pair["chosen"][len(pair["prompt"]):].strip()


def ids_sha256(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def select_sft_pairs(pairs, *, skip, n, n_val=0):
    rm_groups = {p["group"] for p in pairs[:skip]}
    cand, dropped_group, dropped_empty = [], 0, 0
    for p in pairs[skip:]:
        if p["group"] in rm_groups:
            dropped_group += 1
            continue
        if not response_of(p):
            dropped_empty += 1
            continue
        cand.append(p)
        if len(cand) == n + n_val:
            break
    train, val = cand[:n], cand[n:n + n_val]
    stats = {"skip": skip, "n_train": len(train), "n_val": len(val), "dropped_shared_group_while_selecting": dropped_group,
             "dropped_empty_while_selecting": dropped_empty,
             "train_ids_sha256": ids_sha256([p["pair_id"] for p in train]),
             "val_ids_sha256": ids_sha256([p["pair_id"] for p in val])}
    return train, val, stats


def build_example(tok, pair, max_length=512, max_resp_tokens=128):
    prompt, resp = pair["prompt"], response_of(pair)
    p_ids = list(tok(prompt)["input_ids"])
    full = list(tok(prompt + " " + resp)["input_ids"])
    if full[:len(p_ids)] != p_ids:
        return None
    r_ids = full[len(p_ids):]
    ended = True
    if len(r_ids) > max_resp_tokens:
        r_ids, ended = r_ids[:max_resp_tokens], False
    if ended:
        r_ids = r_ids + [tok.eos_token_id]
    ids = p_ids + r_ids
    labels = [-100] * len(p_ids) + r_ids
    
    if len(ids) > max_length:
        drop = len(ids) - max_length
        if drop >= len(p_ids):
            return None
        ids, labels = ids[drop:], labels[drop:]
        # D's fix: Ensure the first valid label is the actual first response token
        # by finding where the prompt ends in the truncated sequence.
        first_resp_idx = len(p_ids) - drop
        labels[:first_resp_idx] = [-100] * first_resp_idx
        
    return {"input_ids": ids, "labels": labels}


def build_examples(tok, pairs, **kw):
    out = [build_example(tok, p, **kw) for p in pairs]
    return [e for e in out if e is not None], sum(e is None for e in out)


def collate(batch, pad_id, device):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n], lab[i, :n], mask[i, :n] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": lab.to(device)}


@torch.no_grad()
def mean_nll(model, examples, pad_id, device, bs=16):
    model.eval()
    tot, cnt = 0.0, 0
    for s in range(0, len(examples), bs):
        b = collate(examples[s:s + bs], pad_id, device)
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits[:, :-1].float()
        tgt = b["labels"][:, 1:]
        keep = tgt != -100
        tot += torch.nn.functional.cross_entropy(logits[keep], tgt[keep], reduction="sum").item()
        cnt += int(keep.sum())
    return tot / max(1, cnt)


@torch.no_grad()
def eos_prob_at_end(model, examples, eos_id, pad_id, device, bs=16):
    model.eval()
    ex = [e for e in examples if e["input_ids"][-1] == eos_id and e["labels"][-1] == eos_id]
    ps = []
    for s in range(0, len(ex), bs):
        chunk = ex[s:s + bs]
        b = collate(chunk, pad_id, device)
        probs = torch.softmax(model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float(), -1)
        for i, e in enumerate(chunk):
            ps.append(probs[i, len(e["input_ids"]) - 2, eos_id].item())
    return float(np.mean(ps)) if ps else float("nan")


def train_sft(model, tok, train_ex, val_ex=None, *, epochs=1, bs=4, accum=4, lr=1e-5, warmup_frac=0.05, seed=0,
              device="cuda", log=print):
    # D's fix: Use fp32 always.
    use_amp = False  
    pad_id = tok.pad_token_id
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n = len(train_ex)
    micro = math.ceil(n / bs)
    total = math.ceil(micro / accum) * epochs
    warm = max(1, int(warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    
    hist = {"val_nll_before": mean_nll(model, val_ex, pad_id, device) if val_ex else None, "epochs": [], "steps": total}
    if val_ex:
        log(f"val NLL before: {hist['val_nll_before']:.3f}")
    t0, step = time.time(), 0
    for ep in range(epochs):
        model.train()
        order = np.random.default_rng(seed * 1000 + ep).permutation(n)
        losses = []
        opt.zero_grad(set_to_none=True)
        for m in range(micro):
            b = collate([train_ex[i] for i in order[m * bs:(m + 1) * bs]], pad_id, device)
            loss = model(**b).loss
            (loss / accum).backward()
            losses.append(loss.item())
            if (m + 1) % accum == 0 or m == micro - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    log(f"  step {step}/{total}, train loss {np.mean(losses[-50:]):.3f}")
        rec = {"epoch": ep + 1, "train_loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 1)}
        if val_ex:
            rec["val_nll"] = mean_nll(model, val_ex, pad_id, device)
        hist["epochs"].append(rec)
        log(f"epoch {ep + 1}/{epochs}: " + ", ".join(f"{k} {v}" for k, v in rec.items() if k != "epoch"))
    return hist


def dir_sha256(path):
    h = hashlib.sha256()
    files = sorted(pathlib.Path(path).glob("*.safetensors"))
    for f in files:
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest() if files else None