# Gold reward scoring

Scores HH-RLHF text with a gold reward model (proxy for human preference, not ground truth).
Gold RM: `Skywork/Skywork-Reward-V2-Llama-3.2-3B`, fp16, sdpa attention, max length 2048
(overlong inputs are cut from the start, so the response end is kept).

## Files
- `scripts/gold_score.py`: the scorer (cache mode, smoke mode, importable functions).
- `gold_scoring_colab.ipynb`: the same code as a Colab notebook (smoke scoring and fp32 check).

## Tested environment
Free-tier Colab, one T4 (15 GB). Versions printed by the last cell of `gold_scoring_colab.ipynb`:

| Package | Version |
| --- | --- |
| python | 3.13.16 |
| torch / CUDA | 2.11.0+cu130 / 13.0 |
| transformers | 5.20.2 |
| accelerate | 1.15.0 |
| datasets | 5.1.0 |
| GPU | Tesla T4 (15 GB) |
| gold RM revision (HF commit) | beafe5f2e6a275ce8c556f68f2d82de7fefbc3d8 |

The model is loaded with `device_map={"": 0}` (whole model on GPU 0).

## Commands
Cache mode (list of `{"prompt": ..., "response": ...}`; the prompt must be the HH context ending in `\n\nAssistant:`):

    python scripts/gold_score.py <rm_name> <samples.json> <out.npy> [batch]

Smoke mode (reads `smoke_ids.txt`, writes JSONL with `pair_id, gold_chosen, gold_rejected, gold_model`):

    python scripts/gold_score.py --smoke smoke_ids.txt gold_smoke.jsonl

`train-N` means row N (0-indexed) of the HF `Anthropic/hh-rlhf` helpful-base train split.
Output order always matches input order.

## Checks run (smoke set, free T4)
| Check | Result |
| --- | --- |
| fp16 vs fp32 (20 samples) | max diff 0.0109, Spearman 1.0 |
| Batched vs single scoring | max diff 0.0078 |
| Truncation at 2048 | 0% (chosen and rejected) |
| Scores finite | yes |
| Pad token | `<|finetune_right_pad_id|>` (128004), not EOS |
| Gold vs human agreement (2,000 smoke pairs, 1 tie dropped) | 69.2% |
| Peak GPU memory / throughput | 7.3 GB / about 12.5 responses per second |

## Limitations
- Gold is a proxy, not human ground truth.
- Skywork's full training data list is not public, so exposure to HH-RLHF cannot be ruled out.
- Gold vs proxy Spearman is pending the trained proxy ensemble.
