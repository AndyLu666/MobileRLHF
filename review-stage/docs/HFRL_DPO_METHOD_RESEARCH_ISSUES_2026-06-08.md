# HFRL DPO Method Research Issues

Date: `2026-06-08`

Scope: migrate MobileFineTuner from hard chosen/rejected thinking to a softer preference modeling workflow that matches how users actually behave.

## Issue 1: Reframe preference pairs as soft preferences, not absolute good/bad labels

### Goal

Treat `chosen` and `rejected` as `preferred` and `less-preferred` responses, not as perfect vs terrible answers.

### Why

In real use, a user may slightly prefer one response without strongly disliking the other. The current hard pair semantics are useful for code compatibility, but the UX and data contract should carry confidence/strength.

### Acceptance Criteria

- Add metadata for preference strength or confidence.
- Keep native DPO compatibility while clarifying semantics in docs and UI.
- Support partial preference labels and equal-preference cases.
- Add explicit guidance that a rejected answer may still be acceptable, just less preferred.

### Local Notes

The current 220-step health A/B run reaches `length_normalized_pair_accuracy=1.0` on held-out eval, while raw pair accuracy stays `0.0` because response length differs. This is a sign that our current evaluation and data semantics should be treated as preference strength, not absolute good/bad labels.

## Issue 2: Benchmark DPO variants that better match our feedback shape

### Goal

Compare the main TRL-style preference losses on the project data and choose defaults based on the actual feedback type.

### Candidate Methods

- `DPO` with `loss_type="sigmoid"` for the baseline pairwise objective.
- `DPO` with `loss_type="robust"` for noisy or weak preferences.
- `DPO` with `loss_type="ipo"` for a softer identity-style objective.
- `DPO` with `loss_type="sigmoid_norm"` or a length-normalized eval path when answer lengths differ.
- `ORPO` for a reference-free monolithic SFT+preference blend.
- `KTO` for binary desirable / undesirable feedback.
- `CPO` for direct pairwise contrastive preference optimization.

### Acceptance Criteria

- Run all candidates on the same small health A/B split.
- Report length-normalized preference accuracy, not just raw pairwise logprob margin.
- Select one default for paired feedback and one default for binary feedback.
- Separate methods for pairwise preference data from methods that only need positive/negative signals.

### Local Status

Executed for DPO-family variants and a first KTO-style binary-feedback path on the synthetic health A/B split.

Research baseline:

- Hugging Face TRL documents DPO `loss_type` options including `sigmoid`, `robust`, `ipo`, and `sigmoid_norm`; `sigmoid_norm` is specifically exposed for length-normalized preference optimization.
- Hugging Face TRL documents KTO as an alignment method for binary feedback such as upvote/downvote, with separate `desirable_weight` and `undesirable_weight` controls.
- This matches the product assumption: a user click is often a partial preference signal, not proof that the unselected answer is objectively bad.

Run:

- `OUT_ROOT=/private/tmp/gpt2_dpo_loss_variant_compare STEPS=40 TRAIN_SIZE=128 EVAL_SIZE=64 review-stage/scripts/run_gpt2_dpo_loss_variant_compare.sh`
- `OUT_ROOT=/private/tmp/gpt2_kto_binary_feedback_compare STEPS=40 TRAIN_SIZE=128 EVAL_SIZE=64 review-stage/scripts/run_gpt2_kto_binary_feedback_compare.sh`

Results:

- `sigmoid`: eval loss `1.742e-05`, eval reward margin `11.5581`
- `robust`: eval loss `1.831e-05`, eval reward margin `11.6035`
- `ipo`: eval loss `0.3293`, eval reward margin `1.7793`
- `sigmoid_norm`: eval loss `0.000615`, eval reward margin `15.0537`
- `kto_binary_feedback`: train rows `256`, eval rows `128`, eval loss `0.005765`, eval reward accuracy `1.0`, eval reward margin `6.4523`

Current working conclusion:

- keep `sigmoid` DPO as the paired-feedback baseline
- use `sigmoid_norm` as the next candidate for length-biased health A/B data
- keep `robust` available for noisy/weak preferences, though it is close to baseline on this clean split
- treat `ipo` as a softer regularized option; it deliberately keeps margins smaller
- use the new KTO-style path for sparse binary feedback where the user taps like/dislike or picks one answer without giving a reliable pairwise strength label
- do not claim full TRL KTO parity yet; the current implementation is a lightweight project KTO-style objective with frozen-reference rewards and desirable/undesirable weights

Local artifacts:

- `pytorch_alignment/dpo_utils.py`: `BinaryFeedbackDataset`, binary collator, KTO-style batch/eval metrics
- `pytorch_alignment/gpt2_lora_finetune.py`: `--train_mode kto`
- `review-stage/scripts/preference_to_binary_feedback_jsonl.py`: pairwise preference to binary feedback converter
- `review-stage/scripts/run_gpt2_kto_binary_feedback_compare.sh`: reproducible KTO-style 40-step compare runner
- `/private/tmp/gpt2_kto_binary_feedback_compare/SUMMARY.json`: latest 40-step result

References:

- Hugging Face TRL DPO Trainer: https://huggingface.co/docs/trl/dpo_trainer
- Hugging Face TRL KTO Trainer: https://huggingface.co/docs/trl/v0.11.1/en/kto_trainer

## Issue 3: Add sparse-feedback expansion so a few user clicks can train on many examples

### Goal

Simulate the project’s federated-scale effect locally: the user only gives a few feedback clicks, while training sees a larger synthetic/augmented preference set.

### Acceptance Criteria

- Start from a tiny set of human-like feedback events.
- Expand them into a larger training set through controlled synthetic generation and judge-based relabeling.
- Keep the user-visible interaction short, but train on many examples.
- Do not implement federation in this issue.
- Document that the apparent large training set is simulated amplification from sparse user feedback, not repeated burden on the user.

## Issue 4: Align Android UX with sparse feedback plus large training batches

### Goal

Make the Android flow show a few feedback taps, then background training on a much larger effective dataset.

### Acceptance Criteria

- User only needs to click feedback a few times.
- UI should not expose raw loss as the main story.
- Training can run later on expanded data and reflect the larger effective dataset.
- Keep the current Android visualizer path, but change the wording and flow to match sparse feedback.
- Use language like "a few taps unlock a large background training set" instead of "hard chosen/rejected pairs".

### Local Status

Started on Android:

- `Test DPO` collects several A/B choices before showing simulated HFRL training telemetry.
- Each choice is persisted as append-only DPO-compatible JSONL, so the Android feedback flow now has a concrete file contract for native DPO.
- The remaining product step is to replace simulated telemetry with a native trainer launch that consumes the generated JSONL, then optionally expand sparse taps into a larger training set.

## Issue 5: Use length-normalized metrics as the default evaluation view

### Goal

Avoid misreading raw pair logprob when chosen and rejected answers have different lengths.

### Acceptance Criteria

- Keep raw pair metrics for debugging.
- Add length-normalized pair accuracy and margin to eval outputs.
- Make the reporting path prefer length-normalized metrics for comparison across methods.

### Local Status

Done in practice for current health and sparse-feedback runs:

- raw pair accuracy remains available for debugging
- length-normalized pair accuracy is now the main comparison signal
- the current sparse-feedback 220-step run reports base `1.0` vs DPO `1.0` on length-normalized accuracy, while raw pair accuracy remains misleading
