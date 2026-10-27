"""Policy sampling for the offline cache: PURE sampling from the reference policy, chunked, resumable shards.

Why pure sampling (temperature 1, top_k 0, top_p 1): the importance-resampling math treats the cached responses as
i.i.d. draws from the reference policy, so no truncation of the distribution is allowed. Responses are cut at
max_new_tokens (the policy is then the distribution over 128-token-truncated continuations).
A record per prompt: prompt_id, prompt, split, prompt_tokens, responses[{response_id, raw, clean, n_tokens,
hit_length, hit_marker, empty}]. `raw` = decoded generation (special tokens removed); `clean` = clean3(raw).
"""
import hashlib
import json
import os
import pathlib
import time

import torch

from src.data import clean3, prompt_id

def check_prompts_path(path):
    """Refuse the frozen test split by file name (load_split has its own guard; this one fails earlier and louder)."""
    if "test" in pathlib.Path(str(path)).name.lower():
        raise ValueError(f"refusing to sample from {path}: the test split is frozen")


def seed_for(seed, pid, chunk_idx):
    return int(hashlib.sha256(f"{seed}|{pid}|{chunk_idx}".encode()).hexdigest()[:8], 16)


def load_policy(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return model, tok


def generate_chunked(gen, n, chunk, min_chunk=8, log=print):
    """gen(k, chunk_idx) -> list of k results. On a CUDA out-of-memory error the chunk is halved (never below
    min_chunk) and the same request is retried. Returns n results."""
    done, idx, cur = [], 0, chunk
    while len(done) < n:
        k = min(cur, n - len(done))
        try:
            done.extend(gen(k, idx))
            idx += 1
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if cur <= min_chunk:
                raise
            cur = max(min_chunk, cur // 2)
            log(f"    out of memory, chunk -> {cur}")
    return done


@torch.no_grad()
def sample_prompt(model, tok, prompt, n, *, chunk=128, max_new_tokens=128, seed=0, device="cuda", log=print):
    pid = prompt_id(prompt)
    enc = tok(prompt, return_tensors="pt")
    x = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    plen = int(x["input_ids"].shape[1])

    def gen(k, idx):
        torch.manual_seed(seed_for(seed, pid, idx))
        out = model.generate(**x, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                             max_new_tokens=max_new_tokens, num_return_sequences=k,
                             pad_token_id=tok.pad_token_id)
        res = []
        for o in out:
            g = o[plen:]
            is_eos = g == tok.eos_token_id
            raw = tok.decode(g, skip_special_tokens=True)
            c = clean3(raw)
            res.append({"raw": raw, "clean": c,
                        "n_tokens": int(is_eos.int().argmax().item()) if bool(is_eos.any()) else int(g.shape[0]),
                        "hit_length": not bool(is_eos.any()), "hit_marker": c != raw.strip(), "empty": c == ""})
        return res

    resp = generate_chunked(gen, n, chunk, log=log)
    for j, r in enumerate(resp):
        r["response_id"] = f"{pid}-{j:04d}"
    return {"prompt_id": pid, "prompt": prompt, "split": "tune", "prompt_tokens": plen, "responses": resp}


def rates(records):
    n = sum(len(r["responses"]) for r in records)
    if n == 0:
        return {"responses": 0}
    f = lambda key: sum(x[key] for r in records for x in r["responses"]) / n
    return {"responses": n, "hit_length": f("hit_length"), "hit_marker": f("hit_marker"), "empty": f("empty")}


def run_sampling(model, tok, prompts, out_dir, *, n=256, chunk=128, max_new_tokens=128, seed=0, shard_prompts=10,
                 device="cuda", max_prompt_tokens=1900, on_shard_done=None, log=print):
    """Sample every usable prompt in shards of `shard_prompts` prompts. A shard file is written atomically
    (tmp + rename), so an existing shard is a finished shard and is skipped on rerun."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    usable, skipped = [], []
    for p in prompts:
        if int(tok(p, return_tensors="pt")["input_ids"].shape[1]) > max_prompt_tokens:
            skipped.append(prompt_id(p))
        else:
            usable.append(p)
    t0, n_new = time.time(), 0
    for s, start in enumerate(range(0, len(usable), shard_prompts)):
        path = out / f"shard_{s:04d}.jsonl"
        if path.exists():
            log(f"shard {s}: already done, skipping")
            continue
        recs = [sample_prompt(model, tok, p, n, chunk=chunk, max_new_tokens=max_new_tokens, seed=seed,
                              device=device, log=log) for p in usable[start:start + shard_prompts]]
        tmp = out / f"shard_{s:04d}.jsonl.tmp"
        with open(tmp, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)
        n_new += len(recs)
        r_ = rates(recs)
        log(f"shard {s}: {len(recs)} prompts, {time.time() - t0:.0f}s elapsed, hit_length {r_['hit_length']:.1%}, "
            f"hit_marker {r_['hit_marker']:.1%}, empty {r_['empty']:.1%}")
        if on_shard_done:
            on_shard_done(path)
    return {"n_usable": len(usable), "skipped": skipped, "seconds": round(time.time() - t0, 1)}


def merge_shards(out_dir, dest):
    """Concatenate shard_*.jsonl in order into one JSONL. Returns the number of prompt records."""
    n = 0
    with open(dest, "w") as f:
        for p in sorted(pathlib.Path(out_dir).glob("shard_*.jsonl")):
            for line in open(p):
                if line.strip():
                    f.write(line if line.endswith("\n") else line + "\n")
                    n += 1
    return n
