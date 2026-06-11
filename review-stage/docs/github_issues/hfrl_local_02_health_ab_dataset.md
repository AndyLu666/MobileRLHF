## Goal

Move beyond MCQ-letter preference pairs and generate a more product-shaped A/B dataset for health coaching.

## Acceptance Criteria

- Prompts include health cluster, steps, heart rate, active minutes, sleep, constraints, and user goal.
- Chosen responses are specific, safe, and context-aware.
- Rejected responses cover realistic bad-answer patterns: generic advice, unsafe intensity, ignored context, over-medical claims, and vague advice.
- Dataset generation is deterministic from a seed.
- Produce train/eval splits large enough for a 220-step DPO run.

## Local Implementation Plan

- Add `review-stage/scripts/build_health_ab_preference_dataset.py`.
- Generate at least 512 train and 128 eval preference pairs by default.
- Run a short native DPO smoke to verify the generated data is consumable.

## Device Dependency

No Android phone required.
