# Multi-User DPO Checkpoint

## Status
- Shared DPO utilities are integrated into GPT-2, Gemma, and Qwen PyTorch alignment trainers.
- Tiny sanity checks and CI coverage are in place.
- GPT-2 trainer-level DPO vs chosen-only SFT comparison has run on a synthetic held-out preference set.
- Review-stage artifacts are organized into `docs/`, `scripts/`, and `data/`.

## What Is Integrated
- `pytorch_alignment/dpo_utils.py`
- `--train_mode dpo`
- `--preference_jsonl`
- `--preference_valid_jsonl`
- frozen reference-model path
- reward-accuracy and reward-margin logging

## What We Still Lack For Many Users Doing DPO
- A stable chosen-only SFT dataset with real user-facing prompts and answers
- A scalable rejected-answer generation pipeline with quality control
- User identity and annotator metadata for weighting or filtering preferences
- Disagreement handling across multiple users on the same prompt
- Privacy policy for storing user prompts, chosen answers, and rejected answers
- A deduplication and decontamination pass before training
- Trainer-level comparisons on larger held-out preference sets
- A plan for aggregating many users into one model update schedule

## Near-Term Plan
1. Expand the longer-form chosen-only SFT set
2. Generate preference pairs from that set with a rejected-answer agent
3. Re-run GPT-2 DPO vs SFT on the expanded held-out split
4. Move from synthetic data to real user-like prompts

## Long-Term Plan
1. Preference collection UI or logging interface
2. Annotator/user metadata schema
3. Multi-user aggregation and weighting strategy
4. Safety filters before preference ingestion
5. Online or periodic offline retraining policy
