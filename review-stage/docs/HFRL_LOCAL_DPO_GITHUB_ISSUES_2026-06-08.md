# HFRL Local DPO GitHub Issue Plan

Date: `2026-06-08`

Scope: solve every HFRL/DPO task that can be completed locally without requiring the user to operate an Android phone.

## Issue 1: HFRL: define local A/B feedback JSONL contract

### Goal

Create a local feedback format that matches the future Android A/B choice flow while staying directly consumable by native DPO.

### Acceptance Criteria

- Use append-only JSONL records with `prompt`, `chosen`, and `rejected`.
- Include metadata such as `source`, `cluster`, `user_id`, and `goal` without breaking native `PreferenceDataset`.
- Document that current rows are simulated feedback, not real user feedback.
- Validate that generated rows survive tokenization/truncation for the chosen `seq_len`.

### Local Status

Done. Implemented locally through `review-stage/scripts/build_health_ab_preference_dataset.py` and validated by the 1-step smoke plus 220-step run.

## Issue 2: HFRL: generate richer synthetic health A/B preference data

### Goal

Move beyond MCQ-letter preference pairs and generate a more product-shaped A/B dataset for health coaching.

### Acceptance Criteria

- Prompts include health cluster, steps, heart rate, active minutes, sleep, constraints, and user goal.
- Chosen responses are specific, safe, and context-aware.
- Rejected responses cover realistic bad-answer patterns: generic advice, unsafe intensity, ignored context, over-medical claims, and vague advice.
- Dataset generation is deterministic from a seed.
- Produce train/eval splits large enough for a 220-step DPO run.

### Local Status

Done. Implemented locally through `review-stage/scripts/build_health_ab_preference_dataset.py`.

## Issue 3: HFRL: run native DPO 220-step local baseline

### Goal

Establish a 220-step local DPO baseline using simulated A/B health feedback, matching the project’s previous 220-step reporting convention.

### Acceptance Criteria

- Run native GPT-2 LoRA DPO for `STEPS=220`.
- Emit `train.log`, `metrics.ndjson`, `eval_metrics.jsonl`, `SUMMARY.json`, and `SUMMARY.md`.
- Keep `metrics.ndjson` Android-readable.
- Keep `LOG_INTERVAL=1` by default so short smoke runs also produce at least one metrics row.
- Report `loss`, `reward_accuracy`, `reward_margin`, `grad_norm`, RSS, reference-cache time, and train-loop time.
- Do not require Android device feedback.

### Local Status

Done. Runner added: `review-stage/scripts/run_hfrl_health_ab_220.sh`.

Validated run:

- `runs/hfrl_health_ab_dpo_s220`
- train/eval split: `512/128`
- steps logged: `220`
- final train loss: `1.06563e-05`
- final train reward accuracy: `1.0`
- final train reward margin: `11.4494`
- final eval loss: `0.00234474`
- final eval reward accuracy: `1.0`
- final eval reward margin: `8.96158`
- held-out length-normalized pair accuracy: base `0.4531` -> DPO adapter `1.0`
- held-out mean length-normalized margin: base `-0.1750` -> DPO adapter `5.6488`

## Issue 4: HFRL: DPO performance policy for paired and cached-reference modes

### Goal

Use paired chosen/rejected forwards and cached reference logps where they improve runtime without changing DPO semantics.

### Acceptance Criteria

- Keep paired 2-forward mode enabled for local HFRL runs.
- Use cached reference logps for longer static-feedback runs such as the 220-step benchmark.
- Include loss/reward/gradient/RSS metrics in performance reports, not only wall time.
- Preserve non-cached smoke mode for quick correctness checks.

### Local Status

Existing measurements show cache can win by 50 steps on short static-feedback runs, but precomputing train+eval reference logps can dominate larger generated datasets. The 220-step runner defaults to `DPO_PAIR_FORWARD=1` and `CACHE_REF_LOGPS=0`; cache remains opt-in for reused static datasets after a short amortization check.

220-step health A/B result:

- `ref_cache_seconds=1036.218`
- `train_loop_seconds=801.1588`
- decision: do not use cached-reference as the default on this generated 512/128 split.

No-cache 220-step confirmation:

- `runs/hfrl_health_ab_dpo_s220_nocache`
- `ref_cache_seconds=0.0`
- `train_loop_seconds=1349.5051`
- final eval loss: `0.00234474`
- final eval reward accuracy: `1.0`
- final eval reward margin: `8.96158`
- decision: no-cache is the simpler default for repeated local runs, while cached reference stays opt-in only when amortization is proven on the exact dataset.

## Issue 5: HFRL: sparse-feedback amplification for "few taps, many samples"

### Goal

Simulate the Android UX where the user only gives a few A/B taps, but training consumes a much larger effective dataset.

### Acceptance Criteria

- Start from a tiny set of simulated click events.
- Expand them into a larger preference dataset with controlled amplification metadata.
- Keep the user-facing interaction short.
- Preserve direct native DPO compatibility.
- Add length-normalized evaluation so the larger synthetic set is not misread through raw pair accuracy.

### Local Status

Done. Implemented through `review-stage/scripts/build_sparse_feedback_amplified_dataset.py` and `review-stage/scripts/run_sparse_feedback_hfrl_220.sh`.

Validated run:

- `runs/hfrl_sparse_feedback_dpo_s220`
- steps logged: `220`
- final train loss: `3.73706`
- final train reward accuracy: `0`
- final train reward margin: `-3.71295`
- final eval loss: `1.09399`
- final eval reward accuracy: `0.5`
- final eval reward margin: `1.84861`
- base length-normalized pair accuracy: `1.0`
- DPO length-normalized pair accuracy: `1.0`
- base mean length-normalized margin: `1.0881688594818115`
- DPO mean length-normalized margin: `5.193685531616211`

## Issue 6: HFRL: Android real-device feedback and trainer launch

### Goal

Move from local simulated feedback to real interactive Android feedback.

### Acceptance Criteria

- User selects A/B responses repeatedly on Android.
- App appends DPO-compatible preference JSONL.
- Native trainer is launched on device or through an adb-shell binary path.
- Visualizer reads the resulting HFRL metrics.

### Local Status

Partially done locally after the initial local-only batch:

- Android `Test DPO` now collects repeated A/B selections before training starts.
- Each tap appends a DPO-compatible JSONL row with `prompt`, `chosen`, `rejected`, `source`, `user_id`, `session_id`, labels, choice index, and timestamp.
- The feedback file is written to the app-specific external files area when available, with app-private storage as fallback.
- The simulated DPO telemetry now records how many feedback pairs were saved and where the JSONL was written.

Still deferred:

- native trainer launch on device against the generated JSONL
- visualizer scan of the resulting on-device HFRL run folder

Emulator validation on `2026-06-09`:

- AVD: `Pixel_7a`, Android 15 / SDK 35
- installed debug APK on `emulator-5554`
- pushed `runs/hfrl_offline_android_smoke_batched_dpo` to `/sdcard/Download/mft_runs/hfrl_offline_android_smoke_batched_dpo`
- completed `Test DPO` by tapping option A for 5 prompts
- verified generated feedback JSONL at `/sdcard/Android/data/com.mobilefinetuner.visualizer/files/hfrl_feedback/android_ab_feedback.jsonl`
- pulled local evidence to `verification/emulator_hfrl_20260608/android_ab_feedback.jsonl`
- installed Android NDK `27.3.13750724`
- cross-compiled `gpt2_small_lora_finetune/train` to Android arm64
- pushed the native trainer and minimal GPT-2 pretrained assets to the emulator
- ran a constrained 1-step native DPO smoke against the Android-generated feedback JSONL
- output evidence: `verification/emulator_native_hfrl_small/REPORT.md`

## Issue 7: HFRL: binary-feedback KTO-style training path

### Goal

Support the real product case where user feedback is sparse and binary: a user may tap like/dislike or choose one answer a few times, without providing a strong pairwise judgment that the other answer is objectively bad.

### Acceptance Criteria

- Add a binary feedback JSONL format with `prompt`, `response`, and `label`.
- Keep `chosen/rejected` DPO for pairwise A/B feedback.
- Add `--train_mode kto` to the GPT-2 PyTorch LoRA trainer.
- Use frozen-reference response-only logprob rewards.
- Emit `loss`, `reward_acc`, `reward_margin`, desirable reward, and undesirable reward.
- Add a converter from existing pairwise preference JSONL to binary feedback JSONL.
- Run a local smoke and a 40-step compare on the same synthetic health split.

### Local Status

Done and train-validated for the GPT-2 PyTorch trainer. CLI parity has also been added to the Qwen and Gemma PyTorch trainers, with syntax validation only so far.

Implemented:

- `pytorch_alignment/dpo_utils.py`
- `pytorch_alignment/gpt2_lora_finetune.py`
- `pytorch_alignment/qwen_lora_finetune.py`
- `pytorch_alignment/gemma_lora_finetune.py`
- `review-stage/scripts/preference_to_binary_feedback_jsonl.py`
- `review-stage/scripts/run_gpt2_kto_binary_feedback_compare.sh`
- `review-stage/docs/github_issues/hfrl_method_06_binary_feedback_kto.md`

Validation:

- syntax check passed for the modified Python files
- 2-step script smoke passed:
  - `OUT_ROOT=/private/tmp/gpt2_kto_binary_script_smoke STEPS=2 TRAIN_SIZE=16 EVAL_SIZE=8 review-stage/scripts/run_gpt2_kto_binary_feedback_compare.sh`
- 40-step KTO-style compare passed:
  - `OUT_ROOT=/private/tmp/gpt2_kto_binary_feedback_compare STEPS=40 TRAIN_SIZE=128 EVAL_SIZE=64 review-stage/scripts/run_gpt2_kto_binary_feedback_compare.sh`

40-step result:

- binary train rows: `256`
- binary eval rows: `128`
- eval loss: `0.005764880922697557`
- eval reward accuracy: `1.0`
- eval reward margin: `6.452292062342167`
- eval desirable reward: `4.115100279450417`
- eval undesirable reward: `-2.3371917828917503`

Remaining:

- run real Qwen/Gemma KTO smoke tests after model paths and runtime budget are available
- port KTO-style binary feedback to native C++ only after the data contract is settled
