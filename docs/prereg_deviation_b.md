# Deviation B: gold-relabelled proxy ensembles

**Status at commit:** No gold-relabelled ensemble member has been trained.

**Trigger.** On the SFT pilot, criterion 1 failed and criterion 2 failed (rho ~ 0).

**What this tests.** Proxy-gold label mismatch. The pool is HH dataset pairs relabelled by gold (pool sha e4eaa607), not samples from the SFT policy. This tests label mismatch only; it does NOT test the distribution shift between HH text and policy samples (which is an untested hypothesis). Passing the in-distribution held-out check is weak evidence by itself. The pilot (G1, G2, length-partialled ρ on the 20-prompt SFT cache) is the real test. If B passes the held-out criterion but fails the pilot, we stop and write up A.

**Changes.** Training labels = gold labels on 9,982 pairs (val rows 2,000-2,999 and their first-turn groups excluded). Noise-25 = gold-relabelled, then 25% flipped (seed 0). Backbone, LoRA, K=5, seeds 0-4, lr, and max length are unchanged. Two things change together (2,000 to 10,000 pairs; human to gold labels), so a gain cannot be attributed to either (stated as a limitation).

**Data verification.** 
- Training set sha256: `c35d6ecc837d56c5768a4021a354a74c980ba05c4316ac145ebb824dba00cb47`
- SFT-vs-pool overlap: 0 pairs (verified against SFT training pairs).

**Order and success criterion (fixed now).** Train the noise-0 ensemble first. 
- **Baseline:** old human-label noise-0 ensemble vs gold on the 1,000 held-out pairs: **0.587 [0.556, 0.618]**.
- **Success rule:** The noise-0 ensemble reaches **≥ 0.65** absolute accuracy vs gold AND the paired 95% bootstrap CI for (new − old) excludes 0.
- **Met:** train noise-25 (same seeds 0-4), re-score SFT pilot, rerun the gates.
- **Not met:** stop and report the negative result. Noise-25 is skipped on failure, and we report that noise-0 did not reach the criterion.

**Hard cap.** One attempt. No change to data, backbone, policy, or thresholds after seeing results.

**Scope.** Gold is now both the training label source and the evaluator, so proxy-gold agreement is partly circular. The held-out number shows the proxy learned gold's labels, not that gold is right. Results are labelled as a different condition and not pooled with the human-label primary.

**Compute.** ~2.8 h per ensemble. Each member pushed to Hub as it finishes.