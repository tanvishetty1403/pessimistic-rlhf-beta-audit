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
The experiment consists of six runs:
\[
3\ c\text{-values} \times 2\ \text{seeds} = 6\ \text{runs}
\]
Seed values: [TO BE LOCKED]

The same \(\beta_{\text{policy}}\) must be used across all six runs. Only the pessimism coefficient \(c\) and the preregistered seed may vary between runs.

All six runs must use the same training budget and comparable configurations.
2.3 Fixed and unresolved settings
Parameter	Preregistered setting
Pessimism coefficient	0.5, 1.0, 2.0
Number of runs	6
Number of seeds	2
Seed values	TBD
Policy model	TBD
Ensemble size	K = 5, subject to calibration/data-readiness gate
KL coefficient (\(\beta_{\text{policy}}\))	TBD
Calibration scale (\(s\))	TBD; must be determined independently of online gold evaluation
Learning rate	TBD
Per-device batch size	TBD
Maximum training steps	TBD
Generations per prompt	TBD
Maximum completion length	TBD
Sampling temperature	TBD
LoRA configuration	TBD
Checkpoint schedule	TBD


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
3.3 Selected tuning prompts
The current split-generation script:
1. Collects unique prompts from the tuning split.
2. Orders them deterministically using the configured seed.
3. Selects the first 200 prompts.
4. Saves them to data/splits/tune_prompts.json.
5. Records their hash in the split manifest.
The current repository does not yet implement a separate RLOO training/evaluation allocation within these 200 prompts.

RLOO training prompt allocation: [TO BE LOCKED]
Checkpoint evaluation prompt allocation: [TO BE LOCKED]

The training and checkpoint-evaluation prompt sets must be disjoint.

The exact selected prompt IDs and their hashes must be recorded before the experiment begins. The allocation must also be reconciled with the offline analysis protocol before it is frozen; the 200 prompts must not be divided arbitrarily if that would break comparability with the offline analysis.
3.4 Test split
The test split remains frozen and must not be used for:
- Learning-rate selection.
- Batch-size selection.
- Hyperparameter tuning.
- Checkpoint selection.
- Interim experiment decisions.
It may only be accessed for the preregistered final evaluation.
3.5 Split integrity and reproducibility
The generated split manifest records the following SHA-256 hashes:
Artifact	SHA-256
RM training prompt IDs	db021fa7fbec1be0822d81227de2f648300875ffcf5429d9383b92f324baec85
Tuning prompt IDs	e3ff4df6834e9ae88317bcef08a49b34927f1d1d5523450278bcb123ea1350a7
Test prompt IDs	4e513b811ee7ed77b35be041e1a2c8bbb240f1978cd5c99f52291ba9ca9d0691
Smoke IDs	aa33854b69cf3c3aa489ff3d9bb4df38881d0b11bb1f35598c17066c324c7fa5
Selected tuning prompts	f0536e7b506ae8fe7dd88f7f9a3542c874ac525eac5ec8108af7519bc5237c44


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

The selection rule must be locked before examining online gold results.

Proposed learning-rate rule: Select the largest learning rate that demonstrates stable KL growth and no NaNs during a 100-step pilot on the designated tuning prompts.

Batch-size selection: [TO BE LOCKED]

Pilot prompts must be drawn from the designated RLOO training/pilot prompt allocation. They must be disjoint from the fixed checkpoint-evaluation prompt set. The same separation must be maintained during the six-run training experiment.

The pilot must not use the frozen test split or gold scores to select hyperparameters.

The pilot's KL-growth and stability criteria must be made numerical before the pilot is run. Final values must be recorded here before the six-run experiment starts.
5.3 Resume integrity
max_steps must be part of the run identity.

A resumed run must refuse to continue if the checkpoint's recorded max_steps differs from the requested configuration.

Other identity-critical settings must also be checked to prevent accidental mixing of runs. At minimum, the identity should cover the policy/model revision, \(c\), \(\beta_{\text{policy}}\), seed, reward ensemble identity, normalization-statistics identity, calibration scale, training configuration, and code/config revision.
6. KL Measurement and Checkpoint Evaluation
Training-batch KL and independently evaluated KL are different measurements.

Training-batch kl_seq is a training diagnostic. It must not be used as the x-axis for the primary checkpoint evaluation.
6.1 Held-out evaluation
At every evaluation checkpoint:
1. Use the same fixed held-out prompt set for all runs and checkpoints.
2. Generate responses from the checkpoint policy using fixed generation settings.
   - The evaluation generation seed(s), sampling procedure, and response-generation protocol must be fixed in advance and kept consistent across all runs and checkpoints.
3. Evaluate the reference policy with the LoRA adapter disabled, keeping the base reference policy fixed across all runs.
4. Independently estimate policy KL between the checkpoint policy and the reference policy on the generated responses.
5. Score those same checkpoint-generated responses using the pinned gold reward model.
6. Record the checkpoint, seed, \(c\), prompt IDs, generation settings, KL estimate, and gold scores.
The reference policy must remain fixed across all runs. The exact estimator, token masking, sequence aggregation, and treatment of variable-length completions must be specified before the first checkpoint evaluation.
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
The online experiment will compare gold reward at matched independently measured KL values. Training-batch KL must not be substituted for independently measured evaluation KL.

The offline G1/G2/F1 analysis uses tuning-split data and first-turn groups. Its analysis includes group-aware resampling/cross-fitting and a common usable KL range for relevant comparisons. The exact offline summary, metric, and ordering to be carried into the online comparison must be extracted from the finalized offline result artifacts before online gold results are inspected.

The preregistered directional comparison will examine whether the ordering of gold reward differences across \(c=0.5\), \(c=1.0\), and \(c=2.0\) agrees with the ordering observed in the offline tuning-split analysis.

The offline analysis's cross-fitting and group-bootstrap procedures must not be assumed to be part of the RLOO evaluation procedure unless they are explicitly implemented there. If uncertainty intervals are reported online, their resampling unit and procedure must be specified in advance.

Offline comparison metric and ordering: [TO BE RECORDED FROM FINALIZED OFFLINE TUNING-SPLIT RESULTS BEFORE ONLINE RESULTS]

Matched-KL grid/range and interpolation rule: [TO BE LOCKED]

Aggregation across prompts, checkpoints, and seeds: [TO BE LOCKED]

Agreement criterion: [TO BE LOCKED]

The comparison must not be redefined after seeing the online results. The frozen test split must not be used to choose the offline metric, matched-KL range, interpolation method, or agreement criterion.
8. November 5 Go/No-Go Criteria
Before the November 5 test, numerical thresholds must be entered for all three criteria below.
Criterion	Required threshold
Minimum independently specified training KL growth by step N	[NUMERIC THRESHOLD]
Maximum step time compatible with six runs in available Colab sessions	[NUMERIC THRESHOLD]
Maximum NaN or skipped-update count	[NUMERIC THRESHOLD]


The value of N must also be specified. The step-time budget must state the assumed available session duration and the time reserved for setup, checkpointing, evaluation, and recovery.

The go/no-go decision must be based on these feasibility and stability criteria, not on gold reward outcomes.

If a criterion fails, the failure and resulting decision must be documented before proceeding.
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
7. That six runs establish broad statistical generality beyond the tested seeds and configurations.
8. That smoke-test results demonstrate successful full-scale online training.
9. That the random-head backbone smoke scorer is equivalent to the trained K=5 reward-model ensemble.
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