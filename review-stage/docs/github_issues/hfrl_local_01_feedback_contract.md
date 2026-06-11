## Goal

Create a local feedback format that matches the future Android A/B choice flow while staying directly consumable by native DPO.

## Acceptance Criteria

- Use append-only JSONL records with `prompt`, `chosen`, and `rejected`.
- Include metadata such as `source`, `cluster`, `user_id`, and `goal` without breaking native `PreferenceDataset`.
- Document that current rows are simulated feedback, not real user feedback.
- Validate that generated rows survive tokenization/truncation for the chosen `seq_len`.

## Local Implementation Plan

- Add a deterministic health A/B preference generator.
- Keep generated records directly compatible with native `--preference_train` and `--preference_valid`.
- Use this contract as the local stand-in for later Android A/B selections.

## Device Dependency

No Android phone required.
