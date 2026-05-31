# MMLU-Style Dataset Checkpoint

## Done
- Added `review-stage/scripts/generate_mmlu_style_sft_dataset.py` to generate a larger synthetic multiple-choice chosen-only dataset.
- Added `review-stage/scripts/mcq_chosen_to_preference.py` to convert single-letter correct answers into matched DPO preference pairs.
- Generated:
  - `review-stage/data/synthetic_mmlu_style_sft_train.jsonl`
  - `review-stage/data/synthetic_mmlu_style_sft_eval.jsonl`
  - `review-stage/data/synthetic_mmlu_style_pref_train.jsonl`
  - `review-stage/data/synthetic_mmlu_style_pref_eval.jsonl`
- Switched `review-stage/scripts/run_gpt2_dpo_vs_sft.sh` defaults to the new MMLU-style preference split.

## Current Shape
- Train size: `480`
- Eval size: `120`
- Format: `Question + A/B/C/D + Answer:`
- Chosen answer: single correct letter
- Rejected answer: single incorrect letter from the same option set

## Why This Exists
- The Android demo data is synthetic UI playback, not a real training dataset.
- The repo's mobile-aligned GPT-2 path expects masked JSONL built from multiple-choice prompts.
- This dataset is meant to be a closer proxy for the on-device GPT-2 SFT path before real MMLU data is restored.

## Next
- Run GPT-2 DPO vs SFT on this `480/120` split with `200+` steps.
- Compare trainer loss curves and held-out preference metrics against the earlier tiny synthetic sets.

## 220-Step Result
- Command: `PYTHON=pt_env/bin/python STEPS=220 OUT_DIR=runs/gpt2_dpo_vs_sft_mmlu_style bash review-stage/scripts/run_gpt2_dpo_vs_sft.sh`
- DPO trainer eval at step `220`:
  - `loss = 0.4062`
  - `reward_acc = 0.8400`
  - `reward_margin = 2.0862`
- SFT trainer eval at step `220`:
  - `valid_ppl = 2.92`
- Unified reference-relative held-out metrics on `120` eval pairs:
  - `DPO`: `loss = 0.3569`, `reward_accuracy = 0.8583`, `reward_margin = 2.2502`
  - `SFT`: `loss = 0.6388`, `reward_accuracy = 0.7750`, `reward_margin = 0.1160`
- Raw policy-only held-out comparison:
  - `DPO`: `pair_accuracy = 0.8583`, `mean_pair_margin = 23.8520`
  - `SFT`: `pair_accuracy = 0.8417`, `mean_pair_margin = 2.5100`

## Takeaway
- On this larger MMLU-style proxy split, current DPO is not worse than chosen-only SFT.
- Under the DPO objective, DPO is clearly stronger on held-out preference discrimination.
