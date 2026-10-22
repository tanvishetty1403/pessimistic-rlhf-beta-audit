"""Score cached samples with a HF sequence-classification reward model and cache to .npy.
Usage: python scripts/gold_score.py <rm_name> <samples.json> <out.npy> [batch]
samples.json: list of {"prompt": str, "response": str}. UNTESTED against the real gold RM; check chat-template per RM card."""
import json, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name, inp, out = sys.argv[1:4]
bs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.bfloat16, num_labels=1).cuda().eval()
S = json.load(open(inp))
scores = []
with torch.no_grad():
    for i in range(0, len(S), bs):
        texts = [s["prompt"] + " " + s["response"] for s in S[i:i + bs]]   # TODO: use RM's chat template
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        scores.extend(model(**enc).logits[:, 0].float().cpu().tolist())
np.save(out, np.array(scores))
print("saved", out, len(scores))
