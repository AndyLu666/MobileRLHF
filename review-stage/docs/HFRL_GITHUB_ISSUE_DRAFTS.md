# HFRL GitHub Issue Drafts

Date: `2026-06-06`

GitHub issue creation was attempted with `gh issue create`, but network/escalation approval timed out. These drafts are ready to submit once GitHub access is available.

## Issue 1: HFRL: Android real-device offline feedback smoke

### Body

## Goal

Run the current offline HFRL vertical slice on a real Android device, using preference JSONL as simulated human feedback first.

## Context

Current local artifacts:

- native DPO supports Android-readable metrics via `--metrics_out`
- Android visualizer parses `reward_accuracy`, `reward_margin`, `grad_norm`, `rss_mb`
- offline run folders already work locally: `runs/hfrl_offline_android_smoke_batched_dpo`

Related existing issues:

- #11 native mobile-side DPO trainer
- #16 C++ DPO validation

## Acceptance Criteria

- Build or select an Android-runnable native train binary path.
- Push/generate a run folder on device containing `train.log` and `metrics.ndjson`.
- Android visualizer scans the folder and displays HFRL reward metrics.
- Capture a short QA note with device, command, runtime, and screenshots/log tail.

## Notes

Start with adb shell + pushed binary/run folder. Do not block on JNI/app-integrated launch yet.

## Issue 2: HFRL: DPO reference-logp cache policy and amortization benchmark

### Body

## Goal

Turn the measured DPO forward optimization into a clear runtime policy: when to use paired 2-forward and when to enable cached reference logps.

## Context

Measured results:

- 4-forward baseline, 4 steps: `real=13.08s`, `loss=0.696726`, `reward_margin=-0.00714569`, `nonzero_grad=24`
- paired 2-forward, 4 steps: `real=10.76s`, `loss=0.696727`, `reward_margin=-0.00714724`, `nonzero_grad=24`
- cached reference, 20 steps: paired `real=54.04s`, cached `real=56.68s`
- cached reference, 50 steps: paired `real=131.45s`, cached `real=105.21s`

Artifacts:

- `review-stage/scripts/compare_hfrl_dpo_forward_modes.sh`
- `runs/hfrl_dpo_forward_mode_compare/SUMMARY.md`
- `runs/hfrl_dpo_forward_mode_compare_s20/SUMMARY.md`
- `runs/hfrl_dpo_forward_mode_compare_s50_pair_cache/SUMMARY.md`

## Acceptance Criteria

- Keep `--dpo_pair_forward 1` as the default path.
- Keep `--cache_ref_logps 0` for quick smoke runs.
- Document and enforce a policy for enabling `--cache_ref_logps 1`, initially around longer static-feedback runs.
- Add at least one longer amortization run or a calculated break-even estimate from measured timings.
- Ensure any speed optimization report includes loss, reward margin, grad norm, nonzero grad tensors, and RSS, not only wall time.

## Issue 3: HFRL: Android reward-metric UI polish

### Body

## Goal

Make HFRL runs visibly distinct and useful in the Android visualizer, beyond basic parsing.

## Context

Android already parses reward metrics:

- `reward_accuracy`
- `reward_margin`
- `chosen_reward`
- `rejected_reward`
- `grad_norm`
- `rss_mb`

Current UI conditionally shows reward accuracy and reward margin when present, but the experience is still mostly inherited from SFT telemetry.

## Acceptance Criteria

- Add an HFRL/RL run mode indicator when reward metrics are present.
- Show reward accuracy and reward margin charts prominently.
- Keep SFT/Gemma views unchanged when reward metrics are absent.
- Add UI screenshots or QA notes for a run folder containing `metrics.ndjson`.
- Avoid claiming human feedback quality from the current fake-feedback dataset; label it as simulated feedback.

## Issue 4: HFRL: interactive feedback JSONL collection on Android

### Body

## Goal

Move from dataset-simulated feedback to actual interactive Android feedback collection.

## Context

Current HFRL uses static preference JSONL:

```json
{"prompt":"...","chosen":"...","rejected":"..."}
```

This is good for the offline vertical slice, but the final target is interactive HFRL: user chooses between two candidate responses on Android, and the app appends preference pairs that native DPO can consume.

Related existing issue:

- #10 DPO multi-user preference collection and aggregation flow

## Acceptance Criteria

- Define an append-only Android feedback JSONL format compatible with native `PreferenceDataset`.
- Add a minimal UI flow for prompt + two responses + user preference.
- Save chosen/rejected pairs on-device.
- Confirm native DPO can consume the generated JSONL without conversion.
- Include a fake/local generation path first if full model inference UI is not ready.

## Suggested Commands When Network Works

```bash
gh issue create --title "HFRL: Android real-device offline feedback smoke" --body-file /tmp/hfrl_issue_1.md
gh issue create --title "HFRL: DPO reference-logp cache policy and amortization benchmark" --body-file /tmp/hfrl_issue_2.md
gh issue create --title "HFRL: Android reward-metric UI polish" --body-file /tmp/hfrl_issue_3.md
gh issue create --title "HFRL: interactive feedback JSONL collection on Android" --body-file /tmp/hfrl_issue_4.md
```
