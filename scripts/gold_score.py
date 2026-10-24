"""Gold reward scorer: HF sequence-classification RM on HH-RLHF text.

Default gold RM: Skywork-Reward-V2-Llama-3.2-3B (fp16, sdpa, max_len 2048).
Scores are one unbounded float per input (higher = better), in input order.

Cache mode (original interface, unchanged):
    python scripts/gold_score.py <rm_name> <samples.json> <out.npy> [batch]
    samples.json: list of {"prompt": str, "response": str}
    prompt must be the HH-formatted context ending in "\\n\\nAssistant:".

Smoke mode (JSONL for B):
    python scripts/gold_score.py --smoke <smoke_ids.txt> <out.jsonl>

Import mode:
    from gold_score import load_gold, score
    tok, rm = load_gold()
    scores = score(tok, rm, texts)
"""
import json
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

GOLD_MODEL = "Skywork/Skywork-Reward-V2-Llama-3.2-3B"
MAX_LEN = 2048


def load_gold(model_name=GOLD_MODEL, device="cuda:0", dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.truncation_side = "left"  # if anything is cut, keep the response end
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device,
        attn_implementation="sdpa", num_labels=1,
    ).eval()
    # pad token must not be a token that appears in the text
    if tok.pad_token is None or tok.pad_token_id == tok.eos_token_id:
        p = "<|finetune_right_pad_id|>"
        tok.pad_token = p if p in tok.get_vocab() else tok.eos_token
    rm.config.pad_token_id = tok.pad_token_id
    print("pad token:", tok.pad_token, tok.pad_token_id)
    return tok, rm


def hh_to_msgs(text):
    parts = re.split(r"\n\n(Human|Assistant): ", text)
    msgs = [
        {"role": "user" if r == "Human" else "assistant", "content": c.strip()}
        for r, c in zip(parts[1::2], parts[2::2])
    ]
    assert msgs and msgs[-1]["role"] == "assistant" and msgs[-1]["content"], (
        "text must be HH format and end with a non-empty Assistant turn: "
        + repr(text[-80:]))
    return msgs


def fmt(tok, msgs):
    s = tok.apply_chat_template(msgs, tokenize=False)
    if tok.bos_token and s.startswith(tok.bos_token):
        s = s[len(tok.bos_token):]  # drop duplicate BOS
    return s


@torch.no_grad()
def score(tok, rm, texts, bs=16, max_len=MAX_LEN):
    dev = next(rm.parameters()).device
    fm = [fmt(tok, hh_to_msgs(t)) for t in texts]
    order = sorted(range(len(fm)), key=lambda i: len(fm[i]))  # length-sorted batches
    out = [None] * len(fm)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([fm[j] for j in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(dev)
        s = rm(**enc).logits[:, 0].float().cpu().tolist()
        for j, v in zip(idx, s):
            out[j] = v
    return out  # original order preserved


def trunc_rate(tok, texts, max_len=MAX_LEN):
    n = np.array([len(tok(fmt(tok, hh_to_msgs(t)))["input_ids"]) for t in texts])
    return float((n > max_len).mean())


def run_cache(name, inp, out, bs=8):
    S = json.load(open(inp))
    texts = [s["prompt"] + " " + s["response"] for s in S]
    tok, rm = load_gold(name)
    scores = score(tok, rm, texts, bs=bs)
    print("truncated:", trunc_rate(tok, texts))
    print("finite:", bool(np.isfinite(scores).all()))
    np.save(out, np.array(scores))
    print("saved", out, len(scores))


def run_smoke(ids_path, out_path):
    from datasets import load_dataset

    ids = [l.strip() for l in open(ids_path) if l.strip()]  # file has CRLF endings
    assert len(ids) == len(set(ids)), "duplicate pair_ids"
    train = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
    rows = [train[int(i.split("-")[1])] for i in ids]  # assumes train-N = row N
    ch = [r["chosen"] for r in rows]
    rj = [r["rejected"] for r in rows]

    tok, rm = load_gold()
    sc = np.array(score(tok, rm, ch))
    sr = np.array(score(tok, rm, rj))

    with open(out_path, "w") as f:
        for i, a, b in zip(ids, sc, sr):
            f.write(json.dumps({"pair_id": i, "gold_chosen": float(a),
                                "gold_rejected": float(b), "gold_model": GOLD_MODEL}) + "\n")

    keep = sc != sr
    print("finite:", bool(np.isfinite(sc).all() and np.isfinite(sr).all()))
    print("ties:", int((~keep).sum()), "| agreement (ties dropped):",
          float((sc[keep] > sr[keep]).mean()))
    print("truncated chosen/rejected:", trunc_rate(tok, ch), trunc_rate(tok, rj))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--smoke":
        run_smoke(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4:
        name, inp, out = sys.argv[1:4]
        run_cache(name, inp, out, int(sys.argv[4]) if len(sys.argv) > 4 else 8)
    else:
        sys.exit(__doc__)
