# HFRL: add binary-feedback KTO-style training path

## Goal

Support the product case where a user gives only a few binary feedback taps, and the trainer consumes binary desirable/undesirable rows instead of always requiring hard chosen/rejected preference pairs.

## Context

User feedback is often partial: choosing one answer does not necessarily mean the other answer is terrible. Pairwise DPO remains useful for A/B comparisons, but the training stack also needs a binary-feedback path for upvote/downvote-like events and sparse feedback amplification.

TRL uses DPO for pairwise preference data and KTO for binary desirable/undesirable feedback. This issue ports a lightweight KTO-style path into the project first for GPT-2 PyTorch LoRA training.

## Acceptance Criteria

- Add a binary feedback JSONL dataset format with `prompt`, `response`, and `label`.
- Support labels such as `desirable`, `undesirable`, booleans, and simple positive/negative strings.
- Add `--train_mode kto` to the GPT-2 LoRA trainer.
- Use a frozen reference model and response-only logprobs.
- Log and evaluate `loss`, `reward_acc`, `reward_margin`, desirable reward, and undesirable reward.
- Add a converter from existing pairwise preference JSONL to binary feedback JSONL.
- Run a local smoke test and a 40-step compare on the synthetic health A/B split.

## Local Result

Implemented locally on `2026-06-08`.

Files:

- `pytorch_alignment/dpo_utils.py`
- `pytorch_alignment/gpt2_lora_finetune.py`
- `pytorch_alignment/qwen_lora_finetune.py`
- `pytorch_alignment/gemma_lora_finetune.py`
- `review-stage/scripts/preference_to_binary_feedback_jsonl.py`
- `review-stage/scripts/run_gpt2_kto_binary_feedback_compare.sh`

Validation:

- Syntax check passed:
  - `PYTHONPYCACHEPREFIX=.pycache_tmp pt_env/bin/python -m py_compile pytorch_alignment/dpo_utils.py pytorch_alignment/gpt2_lora_finetune.py pytorch_alignment/qwen_lora_finetune.py pytorch_alignment/gemma_lora_finetune.py review-stage/scripts/preference_to_binary_feedback_jsonl.py`
- 2-step script smoke passed:
  - `OUT_ROOT=/private/tmp/gpt2_kto_binary_script_smoke STEPS=2 TRAIN_SIZE=16 EVAL_SIZE=8 review-stage/scripts/run_gpt2_kto_binary_feedback_compare.sh`
- 40-step compare passed:
  - `OUT_ROOT=/private/tmp/gpt2_kto_binary_feedback_compare STEPS=40 TRAIN_SIZE=128 EVAL_SIZE=64 review-stage/scripts/run_gpt2_kto_binary_feedback_compare.sh`

40-step result:

- binary train rows: `256`
- binary eval rows: `128`
- eval loss: `0.005764880922697557`
- eval reward accuracy: `1.0`
- eval reward margin: `6.452292062342167`
- eval desirable reward: `4.115100279450417`
- eval undesirable reward: `-2.3371917828917503`

## Follow-up

- Run Qwen and Gemma KTO smoke tests when model paths and runtime budget are available.
- Add native C++ support only after the PyTorch behavior and data contract settle.
- Add explicit sparse-click binary feedback generation from Android-collected events.
