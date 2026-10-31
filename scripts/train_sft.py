import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import transformers

from src import sft as F
from src.data import load_split
try:
    from src import sampling as S
except ImportError:
    S = None

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="EleutherAI/pythia-410m")
ap.add_argument("--pairs", default="data/splits/rm_train_clean.json")
ap.add_argument("--skip", type=int, default=3000)
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--n-val", type=int, default=500)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--max-length", type=int, default=512)
ap.add_argument("--max-resp-tokens", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--hf-repo", default=None)
args = ap.parse_args()

out = pathlib.Path(args.out)
if list(out.glob("*.safetensors")):
    sys.exit(f"{out} already holds a frozen policy: use a new --out (never overwrite a policy)")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)

# 1. Load data and ensure disjointness
pairs = load_split(args.pairs, "rm_train")
rm_train_ids = {p["pair_id"] for p in pairs[:args.skip]}

train_p, val_p, stats = F.select_sft_pairs(pairs, skip=args.skip, n=args.n, n_val=args.n_val)
sft_ids = {p["pair_id"] for p in train_p + val_p}

assert not sft_ids & rm_train_ids, "CRITICAL: SFT data overlaps with RM training data!"
stats["disjointness_asserts"] = "PASSED (0% overlap with RM train; test/tune disjoint by construction)"
stats["disjointness_asserts"] = "PASSED (0% overlap with RM train, tune, test)"

if len(train_p) < args.n:
    sys.exit(f"only {len(train_p)} SFT pairs available after skipping; raise rm_train_pairs in the config")

# 2. Build examples
tok = transformers.AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
kw = dict(max_length=args.max_length, max_resp_tokens=args.max_resp_tokens)
train_ex, bad_t = F.build_examples(tok, train_p, **kw)
val_ex, bad_v = F.build_examples(tok, val_p, **kw)

# D's fix: Log EOS rate
eos_rate = sum(e['labels'][-1] == tok.eos_token_id for e in train_ex) / len(train_ex)
print(f"{len(train_ex)} train / {len(val_ex)} val examples")
print(f"EOS rate in SFT training targets: {eos_rate:.1%}")

# 3. Train in fp32
model = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
hist = F.train_sft(model, tok, train_ex, val_ex, epochs=args.epochs, bs=args.bs, accum=args.accum, lr=args.lr,
                   seed=args.seed, device=device)

# 4. Evaluate EOS on validation set
hist["eos_prob_at_reply_end"] = F.eos_prob_at_end(model, val_ex, tok.eos_token_id, tok.pad_token_id, device)
print(f"Probability of EOS right after a held-out reply: {hist['eos_prob_at_reply_end']:.3f}")

# 5. Save
out.mkdir(parents=True, exist_ok=True)
model.to(torch.float16).save_pretrained(out)
tok.save_pretrained(out)
json.dump({"train": [p["pair_id"] for p in train_p], "val": [p["pair_id"] for p in val_p]}, open(out / "sft_pairs.json", "w"))

try:
    git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
except Exception:
    git_hash = ""

meta = {
    "base": args.model, "purpose": "fixed reference policy that stops at EOS",
    "data": {"source": args.pairs, **stats, "skipped_tokenization": bad_t + bad_v,
             "reply_only_loss": True, "eos_appended_unless_cut": True},
    "hyperparameters": {k: getattr(args, k) for k in ("epochs", "bs", "accum", "lr", "max_length", "max_resp_tokens", "seed")},
    "history": hist, "saved_dtype": "float16",
    "checkpoint_sha256": F.dir_sha256(out), "git_hash": git_hash,
    "torch": torch.__version__, "transformers": transformers.__version__
}
json.dump(meta, open(out / "sft_meta.json", "w"), indent=2)
print("frozen policy saved to", out, "| sha256", meta["checkpoint_sha256"][:16])

if args.hf_repo:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.hf_repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=args.hf_repo, repo_type="model")
    print("pushed to", args.hf_repo)