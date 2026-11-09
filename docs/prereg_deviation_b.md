## Deviation B: gold-relabelled proxy ensembles (one attempt, pre-registered before training)

Status at commit: no gold-relabelled ensemble member has been trained.

**Trigger.** On the SFT pilot, criterion 1 failed (gold peaks at n=1) and criterion 2 failed (rho ~ 0; within-prompt proxy mean anti-correlated with gold).

**What this tests.** Proxy-gold label mismatch only. The training pool is HH dataset pairs relabelled by gold, not samples from the SFT policy. A distribution shift between HH text and policy samples is an untested hypothesis, not tested here. The held-out check below uses in-distribution HH pairs, so passing it is weak evidence by itself; the pilot is the real test.

**Training data (v2, rebuilt; only these two files are valid).**
- Pool: 9,988 pairs = the first 10,000 eligible minus 12 pairs where chosen == rejected (ids in data/splits/b_pool_v2_dropped_identical.json, ids sha256 eae13815a20e...). Pool file sha256 056d4c92ca24..., merged gold file sha256 d0670e578f3030...
- Kept 9,985 (3 more dropped as gold ties). n0 ids sha256 0cee012a3bfa.... Relabelled by gold: 3,047 pairs; human/gold agreement 69.48%. n25 = the same 9,985 pairs in the same order with 2,521 flips (25.2%, seed 0).
- n0 file sha256 b27c0486253e84483064ce3f5dcd31d735482df7f74b37ea32191465b2cce7d0. n25 file sha256 51dfe12886bbb60f295f752e6bff963f3b48409105874336c5a5906b343f2130. Gold: Skywork-Reward-V2-Llama-3.2-3B pinned at beafe5f2.... Checked by scripts/verify_training_sets.py (D) and re-checked on the training side before training: [sha256 and pair counts confirmed from the check cell].
- Superseded and invalid: the earlier pool and training files (pool shas e4eaa607, 3996a590, 8eb26f4a; n0 training set c35d6ecc...). [D to confirm this list.]

**Overlap record (asserted by the pool-building script, re-checked on the training side).** Overlap with SFT v1 train and val pairs, by pair_id and by first-turn group: 0. Overlap with the 1,000 held-out pairs (rm_train_clean rows 2000-2999), tune and test: 0, by pair_id and by group. Every group hashes to rm_train. Overlap with the original 2,000 proxy-RM training pairs: [number from the check cell; recorded as context, not a leak].
History: the first pool overlapped the SFT v1 training data (3,149 pair ids, 4,212 groups). It was caught by the training-side provenance check before any training and discarded. The concern was confounding: sharing prompts between proxy training and the policy's SFT can inflate proxy scores on the policy's own training prompts. The proxies train on human-written text, so memorizing SFT outputs was not the concern.

**Training spec (unchanged except the data).** Pythia-410M, scalar head, LoRA r=16/alpha 32/dropout 0.05, K=5 seeds 0-4 (different seeds, full data), lr 3e-4, 2 epochs, effective batch 16, max length 512, n-train 9,985, validation on the 1,000 held-out pairs.

**Order and success criterion (fixed now; both on the noise-0 ensemble).** Train noise-0 first. Success = ensemble accuracy against gold on the 1,000 held-out non-tied pairs is >= 0.65 AND the paired bootstrap 95% CI of (new - old) excludes 0. Baseline: the old human-label noise-0 ensemble, 0.587 [0.556, 0.618] on the same pairs. The expected gain of about +0.04 is a design expectation, not a criterion: it is judged against the paired CI (half-width about 0.03), not the point estimate.
- Criterion met: train noise-25 on the same spec and seeds 0-4, re-score the SFT pilot with both new ensembles (raw scores, same member order), and rerun the same three pilot criteria with the same thresholds. If the held-out check passes but the pilot fails, we stop and write up option A.
- Criterion not met: noise-25 is skipped. The write-up then says only that the noise-0 retrain did not reach the criterion; there will be no noise-25 data.

**Hard cap.** One attempt. No change to data, backbone, policy or thresholds after seeing results.

**Limitations and scope.** Two things change together (2,000 to 9,985 pairs; human to gold labels), so a gain cannot be attributed to either. Gold is both the training label source and the evaluator, so accuracy against gold on pairs is partly circular: it shows the proxy learned gold's labels, not that gold is right. The held-out number is not evidence about proxy quality in general. Results from these ensembles are a different condition and are not pooled with the human-label primary.

**Compute and records.** About 33 min per member (extrapolated from the measured 6.5 min per member for 2,000 pairs x 2 epochs), about 2.7 h per ensemble. Each member is pushed to the Hub as it finishes. The training-set sha256 is printed in the training log and stored in train_provenance.json next to the checkpoints.