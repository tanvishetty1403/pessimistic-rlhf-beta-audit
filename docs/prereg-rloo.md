RLOO Experiment Preregistration

1. Research Objective
The objective is to investigate whether pessimistic ensemble reward estimates improve online policy optimization under KL regularization.

The online experiment will compare different values of the pessimism coefficient \(c\), while holding the other experimental conditions fixed.

The experiment is intended to examine whether the relationship between pessimism, independently measured policy KL, and gold reward is consistent with the offline findings.

The online experiment is a validation exercise. It is not intended to establish that a particular coefficient is universally optimal.
2. Experimental Grid
2.1 Pessimism coefficient
The planned online grid is:
\[
c \in \{0.5,\ 1.0,\ 2.0\}
\]
2.2 Seeds and number of runs
The primary experiment consists of six runs:
\[
3\ c\text{-values} \times 2\ \text{seeds} = 6\ \text{runs}
\]
Seed values: [TO BE LOCKED]

The same \(\beta_{\text{policy}}\), policy checkpoint, reward ensemble, noise condition, frozen normalization statistics, and training budget must be used across all six primary runs. Only \(c\) and the preregistered seed may vary.

A non-pessimistic Mean-reward run is not currently included in the six-run grid. Unless an additional seed-0 Mean run is approved and budgeted before training, the online experiment cannot establish whether this RLOO setup overoptimizes relative to a non-pessimistic online baseline. A flat result across the three values of \(c\) must therefore be interpreted with this limitation.
2.3 Fixed and unresolved settings
Parameter | Preregistered setting
--- | ---
Pessimism coefficient | 0.5, 1.0, 2.0
Number of primary runs | 6
Number of seeds | 2; exact values TBD
Policy model / LoRA base | SFT v1: `afml-rlhf/sft-policy`; checkpoint SHA-256 to be recorded
Reference policy | Same SFT v1 checkpoint with LoRA adapter disabled
Ensemble size | K = 5, subject to calibration/data-readiness gate
Noise condition | TBD; fixed identically across primary runs
`ensemble_id` | TBD; fixed identically across primary runs
Normalization stats-file hash | TBD; frozen file hash recorded before training
KL coefficient (\(\beta_{\text{policy}}\)) | TBD; same across all primary runs
Calibration scale (\(s\)) | TBD; determined independently of online gold evaluation
Learning rate | TBD; selected by preregistered training-only pilot
Per-device batch size | TBD; selected without gold evaluation
Maximum training steps | TBD; identical across all primary runs
Generations per prompt | TBD
Maximum completion length | TBD
Sampling temperature | TBD
LoRA configuration | TBD
Checkpoint schedule | TBD

The smoke-test configuration is not the final experiment configuration. Smoke-test values must not be treated as locked experimental settings.
3. Dataset and Prompt Protocol
3.1 Dataset
The dataset is:
- Dataset: Anthropic/hh-rlhf
- Subset: helpful-base
- Revision: 09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa
- Split proportions: 80% RM training, 10% tuning, 10% test.
- Split seed: 0 in the current base configuration.
3.2 Split methodology
Dataset pairs are filtered to remove examples where the chosen and rejected responses do not share the same prompt.

Splits are assigned deterministically using a hash of the first human turn. This keeps examples with the same first-turn grouping key in the same split.

The current manifest records:
Split	Number of pairs
RM training	35,096
Tuning	4,285
Test	4,326


These counts are from the generated split manifest currently present in the working project.
3.3 RLOO training and checkpoint-evaluation prompts
RLOO training and pilot prompts will be drawn from the unused portion of the `rm_train` pool. They will be excluded from the RM train/validation IDs, the SFT v1 training IDs, and any `rm_train` examples whose first-turn group is present in either the tuning or test split.

The RLOO training-pool construction must:
1. Start from the `rm_train` pool.
2. Remove all RM train/validation pair IDs.
3. Remove all SFT v1 training pair IDs.
4. Remove any remaining candidate whose first-turn group is shared with a tuning or test example.
5. Assert programmatically that no selected RLOO training prompt shares a first-turn group with tune or test, and that no excluded RM/SFT pair ID remains.

The checkpoint-evaluation set is the same fixed 200 tuning prompts used by the offline N=256 sampling cache. It is not split into RLOO training and evaluation subsets. Use at least 4 generated samples per evaluation prompt; the evaluation sample count \(n\) is to be locked, with 4 as the minimum/default.

The RLOO training and pilot prompts must be disjoint from the 200 checkpoint-evaluation prompts by construction. The test split remains frozen and is not used for training, pilot selection, checkpoint selection, or interim decisions.

The exact RLOO training-pool IDs must be deterministically ordered in hash order and saved as a versioned manifest. Each run uses the first M prompts from that manifest, where M is determined by the locked training steps and prompts consumed per step. The selected IDs and manifest hash must be recorded before the six-run experiment.
3.4 Test split
The test split remains frozen and must not be used for:
- Learning-rate selection.
- Batch-size selection.
- Hyperparameter tuning.
- Checkpoint selection.
- Interim experiment decisions.
It may only be accessed for the preregistered final evaluation.
3.5 Split integrity and reproducibility
The current generated split manifest records the following SHA-256 hashes:
Artifact | SHA-256
--- | ---
RM training prompt IDs | `db021fa7fbec1be0822d81227de2f648300875ffcf5429d9383b92f324baec85`
Tuning prompt IDs | `e3ff4df6834e9ae88317bcef08a49b34927f1d1d5523450278bcb123ea1350a7`
Test prompt IDs | `4e513b811ee7ed77b35be041e1a2c8bbb240f1978cd5c99f52291ba9ca9d0691`
Smoke IDs | `aa33854b69cf3c3aa489ff3d9bb4df38881d0b11bb1f35598c17066c324c7fa5`
Selected tuning prompts | `f0536e7b506ae8fe7dd88f7f9a3542c874ac525eac5ec8108af7519bc5237c44`

The additional RLOO training-pool manifest must be generated and recorded before training:
- Final eligible training-pool prompt count: [TO BE COMPUTED FROM SOURCE ID MANIFESTS]
- Ordered training-pool prompt-ID SHA-256: [TO BE COMPUTED]
- RM train/validation exclusion count (pair IDs): [TO BE COMPUTED]
- SFT v1 exclusion count (pair IDs): [TO BE COMPUTED]
- First-turn group exclusion count (additional candidate prompts/groups): [TO BE COMPUTED]
- First-turn group assertion output: [PASS/FAIL; attach or record machine-readable assertion summary]
- Evaluation prompt count: 200
- Evaluation prompt-ID SHA-256: `f0536e7b506ae8fe7dd88f7f9a3542c874ac525eac5ec8108af7519bc5237c44`
- Evaluation cache: same prompt set as the offline N=256 cache; evaluation samples per prompt \(n\): [TO BE LOCKED, minimum 4]

Do not fill the RLOO pool count or hash by estimate. They must be computed from the actual RM train/validation and SFT v1 ID manifests, with first-turn group assertions run on the resulting pool.

These hashes identify the generated data artifacts and provide a reproducibility check. The test split remains frozen and must not be used for training, tuning, or model-selection decisions.
4. Reward and Pessimism Definition
The online experiment is planned to use a K=5 reward-model ensemble. The final ensemble identity, trained checkpoint, and readiness for online use must be recorded before the six-run experiment.
4.1 Online reward normalization
Before online training, normalization statistics are fitted using designated reference samples.

For each ensemble member score:
\[
R_k^{std}=\frac{R_k-r_{\text{center}}}{r_{\text{scale}}}
\]
The center and scale are computed from the ensemble-mean rewards across reference samples. The scale uses population standard deviation (ddof=0).

The fitted statistics are saved with the ensemble identity, noise condition, and reference sample count. The same frozen statistics are used for all subsequent online reward calculations and Mean baseline comparisons.

Normalization statistics must never be refitted on online training batches.

The ensemble variance used in the pessimistic penalty is calculated across the K standardized ensemble-member scores using sample variance (ddof=1).
4.2 Gaussian pessimistic reward
The robust coefficient is defined as:
\[
\beta_{\text{rob}}=c\beta_{\text{policy}}
\]
The Gaussian pessimistic reward is:
\[
\widetilde r
=
\mu-\frac{s\sigma^2}{2c\beta_{\text{policy}}}
\]
where:
- \(\mu\) is the mean of the standardized ensemble-member rewards.
- \(\sigma^2\) is the sample variance across ensemble members.
- \(c\) is the experimental pessimism coefficient.
- \(\beta_{\text{policy}}\) is the policy KL coefficient.
- \(s\) is the calibration scale.
The effective variance penalty coefficient is:
\[
\lambda=\frac{s}{2c\beta_{\text{policy}}}
\]
This is the Gaussian variance-penalty form implemented by src/pessimism.py and called by src/rloo_reward.py.

At fixed \(\beta_{\text{policy}}\) and calibration scale, smaller values of \(c\) impose a stronger variance penalty. In particular, the penalty at \(c=0.5\) is four times the penalty at \(c=2.0\).

The calibration scale is the G2/F1 variance-penalty calibration factor. Its exact value and the mapping from the offline calibration output to the online calibration_scale must be documented and frozen before online gold evaluation. It must not be selected using online gold evaluation results.

Until the calibration scale is established and its mapping is specified, the online experiment must be interpreted as testing an uncalibrated or not-yet-fully-specified absolute variance penalty, rather than a fully calibrated pessimism comparison.
4.3 RLOO reward and KL integration
The online reward function returns the Gaussian pessimistic reward, not the ensemble mean. The current RLOO smoke reward wrapper calls compute_ensemble_reward() and returns its pessimistic_reward field.

KL regularization is applied separately by the TRL RLOO trainer. The intended optimization reward is:
\[
R_{\text{post-KL}} =
\widetilde r -
\beta_{\text{policy}}\, KL_{\text{seq}}
\]
Here, \(KL_{\text{seq}}\) denotes the sequence-level KL quantity used by the training objective. The installed TRL implementation and exact KL convention must be pinned and verified for the final experiment.

The smoke implementation documents the training KL calculation as sequence-summed k1 KL. Its logged TRL kl metric is treated as a per-token mean; it must not be confused with the sequence-summed quantity. The exact conversion and aggregation must be checked against the pinned TRL version before interpreting logged values.

The implementation logs pre-KL reward, KL penalty, post-KL reward, ensemble mean, ensemble variance, and pessimism penalty separately.
4.4 Smoke-test limitation
The current RLOO smoke implementation supports two scorer modes:
- mock: deterministic hash-seeded synthetic scores for testing reward plumbing and reproducibility.
- backbone: a frozen Pythia backbone with fixed random heads, intended to estimate approximate reward-scoring compute and memory cost.
Neither mode is the final trained K=5 reward-model ensemble. Smoke-test results validate integration and feasibility only; they are not scientific evidence for the primary experiment and must not be reported as final reward-model results.
5. RLOO Training Protocol
The online policy optimization uses TRL's RLOOTrainer.

The intended training objective applies a KL penalty to the pessimistic reward:
\[
R_{\text{post-KL}} =
\widetilde r -
\beta_{\text{policy}}\, KL_{\text{seq}}
\]
The exact implementation and KL convention must be verified against the pinned TRL version used for the six-run experiment.
5.1 Settings to lock
The following must be fixed before the six-run experiment:
- TRL version.
- Transformers version.
- PEFT version.
- PyTorch version.
- Policy model and revision.
- Reward ensemble checkpoint(s), identity, and revision/hash.
- Frozen normalization-statistics file and hash.
- Calibration scale and its provenance.
- LoRA configuration.
- Number of generations.
- Completion-length limit.
- Sampling temperature.
- Advantage normalization.
- Reward clipping.
- Learning rate.
- Batch size and gradient accumulation.
- KL coefficient.
- Maximum training steps.
- Checkpoint schedule.
- Prompt allocation and prompt-manifest hashes.
The exact versions and configuration must be recorded in the run metadata.
5.2 Learning-rate and batch-size selection
Learning rate and batch size must not be selected using gold evaluation results.

The learning-rate pilot is 100 optimizer steps using the real SFT v1 policy and the real K=5 reward-model ensemble, with frozen normalization statistics, preregistered \(\beta_{\text{policy}}\), and \(c=1.0\). The pilot must not use mock or random-head backbone rewards. It can begin only after the real reward ensemble, its stats file, and \(\beta_{\text{policy}}\) are locked.

Proposed learning-rate rule: Select the largest candidate learning rate for which (a) the median sequence-summed training k1 KL over steps 81–100 is at least 0.05 nats per sequence, (b) that median exceeds the median over steps 1–20, (c) no NaN/Inf loss, reward, KL, or gradient is observed, and (d) skipped updates are no more than 1% of pilot updates. Record the candidate LR grid and all pilot metrics. These are proposed numerical thresholds for team approval before the pilot; they must not be changed after seeing gold evaluation results.

Before the pilot begins, the team must also lock:
- Maximum acceptable median sequence-summed training k1 KL over steps 81–100: [TO BE APPROVED].
- Maximum acceptable gradient norm: [TO BE APPROVED], with the norm definition and measurement convention specified.
- Any additional loss-stability limit: [TO BE APPROVED], if used.

A candidate learning rate is eligible only if it satisfies both the minimum KL-growth criteria above and these upper stability limits. These limits must be fixed before the pilot and must not be adjusted after observing pilot results.

Batch-size selection: [TO BE LOCKED BEFORE PILOT; use training-side throughput, memory, KL stability, and skipped-update rate only.]

Pilot prompts must be drawn from the RLOO training-pool manifest defined in §3.3, never from the 200 checkpoint-evaluation prompts. The six-run training prompts use the same eligible pool and remain disjoint from checkpoint evaluation.

The pilot must not use the frozen test split or gold scores to select hyperparameters. The real gold scorer must not be queried for pilot-based LR or batch-size selection.

The candidate learning-rate values and selection grid must be specified and frozen before the 100-step pilot begins. They must not be added, removed, or adjusted based on the observed pilot results.
### 5.3 Resume Integrity

The repository currently includes a dataset split manifest and `scripts/verify_manifest.py` for checking split integrity. These do not, by themselves, establish run-level checkpoint resume integrity.

Before the six-run experiment begins, the training implementation must provide a run manifest that records identity-critical experiment settings, including:

- Policy/model revision.
- Pessimism coefficient \(c\).
- KL coefficient \(\beta_{\text{policy}}\).
- Random seed.
- Reward ensemble identity and noise condition.
- Normalization statistics identity and hash.
- Calibration scale.
- Training configuration, including `max_steps`.
- Training-pool prompt manifest and hash.

A resumed run must refuse to continue if its run manifest is missing or if any identity-critical setting differs from the saved manifest. In particular, a mismatch in `max_steps` must prevent resumption.

The run manifest must be saved with the checkpoint and made available through the experiment's Hub artifacts.

The Git commit should be recorded as provenance metadata but excluded from the identity hash, so that unrelated documentation or non-identity code commits do not prevent a valid resume.

**Implementation status:** Run-level manifest generation, identity validation, mismatch rejection, and Hub persistence must be verified before the six-run experiment. These capabilities are not established by the currently verified repository files.
6. KL Measurement and Checkpoint Evaluation
Training-batch KL and independently evaluated KL are different measurements.

Training-batch kl_seq is a training diagnostic. It must not be used as the x-axis for the primary checkpoint evaluation.
6.1 Held-out evaluation
At every evaluation checkpoint:
1. Use the same fixed 200 tuning prompts as the offline N=256 sampling cache for all runs and checkpoints.
2. Generate \(n\) responses per prompt, with \(n\) locked before evaluation (minimum/default 4). Use a fixed per-prompt generation seed shared across all runs and checkpoints (common random numbers), with the sampling procedure and generation settings fixed in advance.
3. Use the SFT v1 policy (`afml-rlhf/sft-policy`) as the reference: the same model checkpoint with the LoRA adapter disabled.
4. Estimate policy KL in FP32 using the k1 estimator, summed over exactly the response tokens visible to the gold scorer: the clean3 prefix, plus the EOS token if the reply ended on EOS without a marker cut. Apply the same token mask and completion-boundary convention to policy and reference log-probabilities.
5. Score those same checkpoint-generated samples using the pinned gold reward model.
6. Record checkpoint, seed, \(c\), prompt IDs, per-prompt generation seed, sample index, generation settings, token-mask/boundary metadata, summed k1 KL, and gold score.

The reported checkpoint KL is the mean of per-sample sequence-summed k1 values over the fixed prompt/sample evaluation set; retain per-prompt values for aggregation and uncertainty analysis. The reference policy remains the same SFT v1 checkpoint with the adapter disabled across all runs. The evaluation sample count \(n\) and exact scorer/tokenization revisions must be frozen before the first checkpoint evaluation.
6.2 Gold scorer
The planned gold reward model is:

Skywork/Skywork-Reward-V2-Llama-3.2-3B

The existing gold-scoring documentation specifies:
- FP16 inference.
- SDPA attention.
- Maximum sequence length of 2048.
- Truncation from the start while retaining the response end.
The gold scorer implementation and model revision must be pinned for the final experiment.
6.3 Evaluation schedule
Checkpoint evaluation steps: [TO BE LOCKED]

The schedule must be identical across all six runs.

Evaluation must use the same prompts and generation settings at every checkpoint.
7. Directional Agreement with Offline Results
The offline and online analyses use different KL estimators. Offline resampling KL and online held-out k1 KL are not comparable in absolute value and must not be placed on a shared numeric KL axis. Compare orderings within each setting only. The online comparison uses independently measured online k1 KL to match online gold reward across the three \(c\) conditions; the offline ordering is taken from the finalized offline tuning-split analysis.

Before online gold results are inspected, record the offline tuning-split ordering for the preregistered gold-reward contrast(s) across \(c=0.5\), \(c=1.0\), and \(c=2.0\). Define the contrast and its direction from the finalized offline output; do not infer or alter it after online results are available.

For the online comparison, define the common usable KL interval as the intersection of the observed evaluation-KL ranges for all three \(c\) values, using only the preregistered checkpoint schedule. Use a fixed interpolation rule on a common grid within that intersection. The grid spacing, endpoint-trimming rule, and minimum number of common grid points must be locked before results are inspected. If the intersection is empty or contains fewer than the preregistered minimum number of grid points, report the matched-KL comparison as not estimable; do not extrapolate and do not substitute training-batch KL.

The directional agreement criterion requires both seeds, evaluated separately, to reproduce the same preregistered ordering of the gold-reward contrasts across the three \(c\) values as the offline tuning-split ordering. If the two seeds disagree with each other, or either seed fails to reproduce the offline ordering, classify the result as directional agreement not demonstrated. Do not use a pooled two-seed ordering to override a seed-level disagreement.

The offline analysis's cross-fitting and group-bootstrap procedures must not be assumed to be part of the RLOO evaluation procedure unless explicitly implemented there. Any online uncertainty intervals must state their resampling unit and procedure. With only two seeds, do not interpret seed-level variation as support for population-level significance.

Offline comparison metric and ordering: [TO BE RECORDED FROM FINALIZED OFFLINE TUNING-SPLIT RESULTS BEFORE ONLINE RESULTS]

Online matched-KL grid, interpolation, endpoint trimming, and minimum grid-point rule: [TO BE LOCKED]

Aggregation across prompts and checkpoints within each seed: [TO BE LOCKED]

Agreement criterion: both seeds must independently show the same preregistered ordering as offline; otherwise agreement is not demonstrated.

The comparison must not be redefined after seeing online results. The frozen test split must not be used to choose the offline metric, online KL range, interpolation method, or agreement criterion.
8. November 5 Go/No-Go Criteria
The following are proposed numerical thresholds for team approval before the November 5 feasibility test. They are based on the reported T4 smoke timing of approximately 5–6 seconds per training step plus approximately 1 second of reward-model scoring per batch; the real-model pilot must confirm the estimate.

Criterion | Proposed threshold
--- | ---
Pilot horizon | N = 100 optimizer steps
Minimum KL growth | Median sequence-summed training k1 KL over steps 81–100 must be at least 0.05 nats/sequence and exceed the median over steps 1–20
Maximum step time | Mean wall time no more than 8 seconds/optimizer step over steps 21–100, including reward scoring but excluding one-time model download/setup
Six-run compute budget | At most 8 seconds × locked max_steps × 6 runs of training time, plus separately budgeted checkpoint evaluation, setup, and recovery; this total must fit the team's declared Colab session allocation before launch
NaN/Inf count | 0 across loss, reward, KL, and gradients
Skipped-update rate | No more than 1% of optimizer updates in the 100-step pilot

The 0.05-nat KL threshold and 8-second step-time threshold are proposed feasibility gates, not scientific effect-size claims. If the team changes them, the approved values must be recorded before the pilot starts. If the pilot's training KL diagnostic is dominated by completion length, the KL growth gate must be evaluated with the independently logged sequence/token convention documented in §4.3, not by treating raw per-token means as sequence KL.

The step-time budget must use the locked maximum training steps and explicitly state the available Colab session hours and the time reserved for setup, checkpointing, evaluation, and recovery. If six runs cannot fit the declared allocation, reduce the locked training budget or obtain additional compute before starting; do not change the budget after inspecting gold scores.

The go/no-go decision must be based on feasibility and stability criteria, not on gold reward outcomes.

If a criterion fails, document the failure and decision before proceeding.
9. Checkpoint, Logging, and Reproducibility
Each run must record:
- Run name.
- \(c\) value.
- \(\beta_{\text{policy}}\).
- Calibration scale and provenance.
- Seed.
- Maximum training steps.
- Policy model and revision.
- Reward ensemble identifiers and revision/hash.
- Normalization-statistics file/hash and reference sample count.
- Prompt-set identifiers and hashes.
- Package versions.
- Git commit and dirty state.
- Full training configuration and run identity.
- Training metrics.
- Training KL metrics, with units and aggregation convention.
- Independent evaluation KL, with estimator details.
- Gold reward scores.
- Checkpoint identifiers.
- Timing and memory metrics.
- NaN, skipped-update, and failure counts.
The six runs must use consistent checkpoint schedules and run identities.

Checkpoint recovery must not silently mix incompatible configurations. A resumed run must validate its saved identity against the requested configuration before training continues.
10. Claims-Out List
The experiment will not claim:
1. That the gold reward model represents ground-truth human preference.
2. That agreement with the gold scorer proves alignment with human preferences.
3. That the online experiment establishes a universally optimal pessimism coefficient.
4. That arbitrary values of \(c\) at fixed \(\beta_{\text{policy}}\) constitute a calibrated comparison before the calibration scale and its mapping are established.
5. That training-batch KL is equivalent to independently evaluated policy KL.
6. That results from the tuning split establish generalization to the frozen test split.
7. That six runs establish broad statistical generality beyond the tested seeds and configurations; with only two seeds, no p-values, significance claims, or population-level statistical conclusions will be reported.
8. That smoke-test results demonstrate successful full-scale online training.
9. That the random-head backbone smoke scorer is equivalent to the trained K=5 reward-model ensemble.
10. That the RLOO experiment establishes anything about PPO dynamics; this is an RLOO study, not a PPO experiment.
11. That online k1 KL values are numerically comparable to offline resampling-KL values. Only within-setting orderings are compared; no cross-setting equality or magnitude claim is made.
12. That the experiment demonstrates online overoptimization relative to a non-pessimistic baseline, because a Mean-reward online run is not included in the primary six-run grid unless separately approved and completed.
The possibility of overlap between the gold reward model's training data and HH-RLHF must also be acknowledged. The full Skywork training-data list is not publicly established in the current project documentation.
11. Literature Follow-ups
The following checks remain outstanding:
- Google Scholar and Semantic Scholar cited-by checks for Hahami.
- Google Scholar and Semantic Scholar cited-by checks for AdvPO.
- Review relevant 2025 literature on pessimism.
- Review Xiong et al. ablations.
- Record any literature-derived changes to the hypotheses before online results are examined.
Preregistration freeze date: [TO BE ENTERED]
Approved by: [TO BE ENTERED]