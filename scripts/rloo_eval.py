"""scripts/rloo_eval.py  (run from repo root)

Three resumable stages, so gold (3B, fp16) and the policy never share GPU memory:

  gen   : sample from the SFT policy + LoRA checkpoint on fixed eval prompts, compute per-sample sequence KL
          (k1, summed over generated tokens, reference = same model with the adapter disabled), cache to JSONL.
          python scripts/rloo_eval.py gen --policy afml-rlhf/sft-policy --run-dir runs/rloo_smoke/c1_b0.05_s0 \
              --steps 0,20,40,60 --out eval/ --n-prompts 200 --n 4
  gold  : score cached clean texts with the gold RM (cache keyed by sha1(prompt+clean), so reruns are free).
          python scripts/rloo_eval.py gold --out eval/ [--mock-gold]
  plot  : per (run, step) table + per-prompt table + gold-vs-KL plot (one line per run, no CI bands).
          python scripts/rloo_eval.py plot --out eval/

Design choices (put in prereg):
  * Same prompts, same per-prompt generation seed at every checkpoint and run (common random numbers).
  * KL is counted over the tokens gold sees: the clean3-truncated prefix, plus the EOS token if the reply ended
    on EOS without a marker cut. The cut is mapped to a token index by bisecting decode(g[:k]); approximate when
    clean3 dropped a dangling label.
  * Eval/KL in fp32 (policy was trained in fp32). The offline cache was sampled in fp16, so do not compare the two
    KL numbers directly.
  * Step 0 = SFT policy itself (KL exactly 0): the shared starting point of every line.
"""
import argparse
import glob
import hashlib
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

RUN_RE = re.compile(r"c([\d.]+)_b([\d.]+)_s(\d+)")


def key_of(prompt, clean):
    return hashlib.sha1((prompt + "\x00" + clean).encode()).hexdigest()


def load_prompts(path, n):
    from src import sampling as S
    from src.data import load_split
    S.check_prompts_path(path)                     # test split stays frozen
    return load_split(path, "tune")[:n]


# gen
def n_clean_tokens(tok, g, clean):
    """Smallest k such that decode(g[:k]) covers the clean text."""
    if not clean:
        return 0
    L, lo, hi = len(clean), 0, len(g)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(g[:mid], skip_special_tokens=True).strip()) >= L:
            hi = mid
        else:
            lo = mid + 1
    return lo


def load_model(policy, adapter_dir, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(policy)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(policy, dtype=torch.float32).to(device).eval()
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir).to(device).eval()
    return model, tok


def gen_one_prompt(model, tok, prompt, n, seed, max_new, device, has_adapter):
    import torch
    from src.data import clean3, prompt_id
    from src.sampling import seed_for

    def tok_logps(ids, plen):
        logits = model(ids).logits[:, plen - 1:-1].float()
        return torch.log_softmax(logits, -1).gather(-1, ids[:, plen:, None])[..., 0]

    pid = prompt_id(prompt)
    enc = tok(prompt, return_tensors="pt").to(device)
    plen = int(enc["input_ids"].shape[1])
    with torch.no_grad():
        torch.manual_seed(seed_for(seed, pid, 0))
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=max_new,
                             num_return_sequences=n, pad_token_id=tok.pad_token_id)
        if has_adapter:
            lp = tok_logps(out, plen)
            with model.disable_adapter():
                ref = tok_logps(out, plen)
            d = (lp - ref).cpu()
        else:
            d = torch.zeros(out.shape[0], out.shape[1] - plen)
    rows = []
    for i in range(out.shape[0]):
        g = out[i, plen:]
        eos = g == tok.eos_token_id
        has_eos = bool(eos.any())
        n_gen = int(eos.int().argmax().item()) if has_eos else int(g.shape[0])
        raw = tok.decode(g, skip_special_tokens=True)
        clean = clean3(raw)
        hit_marker = clean != raw.strip()
        k = n_gen if not hit_marker else n_clean_tokens(tok, g[:n_gen], clean)
        kt = k + (1 if (has_eos and not hit_marker) else 0)
        rows.append({"prompt_id": pid, "j": i, "clean": clean, "n_gen": n_gen, "n_kl_tokens": kt,
                     "kl_seq": float(d[i, :kt].sum()), "kl_seq_full": float(d[i, :n_gen + int(has_eos)].sum()),
                     "eos": has_eos, "hit_length": not has_eos, "hit_marker": hit_marker, "empty": clean == ""})
    return rows


def stage_gen(a):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = pathlib.Path(a.run_dir).name
    prompts = load_prompts(a.prompts, a.n_prompts)
    meta = {"policy": a.policy, "prompts": a.prompts, "n_prompts": a.n_prompts, "n": a.n, "seed": a.seed,
            "max_new": a.max_new, "dtype": "float32"}
    mp = out / "eval_meta.json"
    if mp.exists() and json.load(open(mp)) != meta:
        sys.exit(f"{mp} differs from these arguments: use a new --out")
    json.dump(meta, open(mp, "w"), indent=2)
    for step in [int(s) for s in a.steps.split(",")]:
        dest = out / f"gen_{run}_s{step}.jsonl"
        if dest.exists():
            print("skip", dest.name)
            continue
        ad = None if step == 0 else str(pathlib.Path(a.run_dir) / f"checkpoint-{step}")
        if ad and not os.path.isdir(ad):
            print(f"!! {ad} missing, skipping step {step}")
            continue
        model, tok = load_model(a.policy, ad, device)
        rows = []
        for p in prompts:
            if int(tok(p, return_tensors="pt")["input_ids"].shape[1]) > 1900:
                continue                                    # same skip rule as the offline cache
            for r in gen_one_prompt(model, tok, p, a.n, a.seed, a.max_new, device, ad is not None):
                r.update(run=run, step=step)
                rows.append(r)
        tmp = dest.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, dest)
        print(f"{dest.name}: {len(rows)} samples, mean kl_seq {sum(r['kl_seq'] for r in rows) / len(rows):.3f}")
        del model
        torch.cuda.empty_cache()


# gold
def read_gen(out):
    rows = []
    for f in sorted(glob.glob(str(pathlib.Path(out) / "gen_*.jsonl"))):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def stage_gold(a):
    out = pathlib.Path(a.out)
    cache_path = out / ("gold_cache_MOCK.jsonl" if a.mock_gold else "gold_cache.jsonl")
    cache = {}
    if cache_path.exists():
        for l in open(cache_path):
            if l.strip():
                d = json.loads(l)
                cache[d["k"]] = d["gold"]
    prompts = {}
    from src.data import prompt_id
    for p in load_prompts(a.prompts, a.n_prompts):
        prompts[prompt_id(p)] = p
    todo = {}
    for r in read_gen(out):
        k = key_of(prompts[r["prompt_id"]], r["clean"])
        if k not in cache:
            todo[k] = prompts[r["prompt_id"]] + " " + r["clean"]       # same format as gold_score.run_cache
    print(f"{len(cache)} cached, {len(todo)} to score")
    if not todo:
        return
    keys = list(todo)
    if a.mock_gold:
        scores = [(int(hashlib.sha1(k.encode()).hexdigest()[:8], 16) / 2**32 - 0.5) * 4 for k in keys]
    else:
        from gold_score import load_gold, score, GOLD_MODEL
        tok, rm = load_gold()
        scores = []
        with open(cache_path, "a") as f:                                 # flush per block: Colab can die
            for i in range(0, len(keys), 512):
                blk = keys[i:i + 512]
                s = score(tok, rm, [todo[k] for k in blk], bs=a.bs)
                for k, v in zip(blk, s):
                    f.write(json.dumps({"k": k, "gold": float(v), "gold_model": GOLD_MODEL}) + "\n")
                f.flush()
                scores += s
        return
    with open(cache_path, "a") as f:
        for k, v in zip(keys, scores):
            f.write(json.dumps({"k": k, "gold": float(v), "gold_model": "MOCK"}) + "\n")


# plot
def summarize(df):
    """df columns: run, step, prompt_id, kl_seq, gold, n_gen, eos, hit_length, hit_marker, empty.
    Per-prompt means first, then mean over prompts (prompt is the unit for later bootstrap)."""
    cols = ["kl_seq", "gold", "n_gen", "eos", "hit_length", "hit_marker", "empty"]
    pp = df.groupby(["run", "step", "prompt_id"])[cols].mean().reset_index()
    sm = pp.groupby(["run", "step"])[cols].mean().reset_index()
    sm["n_prompts"] = pp.groupby(["run", "step"]).size().values
    return pp, sm


def stage_plot(a):
    import pandas as pd
    out = pathlib.Path(a.out)
    cache_path = out / ("gold_cache_MOCK.jsonl" if a.mock_gold else "gold_cache.jsonl")
    gold = {}
    for l in open(cache_path):
        if l.strip():
            d = json.loads(l)
            gold[d["k"]] = d["gold"]
    from src.data import prompt_id
    prompts = {prompt_id(p): p for p in load_prompts(a.prompts, a.n_prompts)}
    rows = read_gen(out)
    for r in rows:
        r["gold"] = gold.get(key_of(prompts[r["prompt_id"]], r["clean"]))
    df = pd.DataFrame(rows)
    miss = int(df["gold"].isna().sum())
    if miss:
        print(f"!! {miss} samples have no gold score yet (run the gold stage); dropping them")
        df = df.dropna(subset=["gold"])
    pp, sm = summarize(df)
    pp.to_csv(out / "per_prompt.csv", index=False)
    sm.to_csv(out / "summary.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for run, g in sm.groupby("run"):
        g = g.sort_values("step")
        m = RUN_RE.search(run)
        lab = f"c={m.group(1)} s{m.group(3)}" if m else run
        ax.plot(g["kl_seq"], g["gold"], marker="o", ms=3, label=lab)
    ax.set_xlabel("KL(policy || SFT ref), nats per sequence (k1, fp32)")
    ax.set_ylabel("mean gold reward (raw)")
    ax.set_title("RLOO: gold vs KL (directional only, 2 seeds, no CI)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "gold_vs_kl.png", dpi=150)
    print(sm.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "gold", "plot"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--prompts", default="data/splits/tune_prompts.json")
        p.add_argument("--n-prompts", type=int, default=200)
        if name == "gen":
            p.add_argument("--policy", required=True, help="SFT policy (HF repo or path); the LoRA base")
            p.add_argument("--run-dir", required=True)
            p.add_argument("--steps", required=True, help="comma list, 0 = SFT policy itself")
            p.add_argument("--n", type=int, default=4)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--max-new", type=int, default=128)
        else:
            p.add_argument("--mock-gold", action="store_true", help="plumbing test only; writes a separate cache")
            p.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    {"gen": stage_gen, "gold": stage_gold, "plot": stage_plot}[a.cmd](a)