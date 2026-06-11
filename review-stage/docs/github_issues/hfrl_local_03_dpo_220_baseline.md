## Goal

Establish a 220-step local DPO baseline using simulated A/B health feedback, matching the project’s previous 220-step reporting convention.

## Acceptance Criteria

- Run native GPT-2 LoRA DPO for `STEPS=220`.
- Emit `train.log`, `metrics.ndjson`, `eval_metrics.jsonl`, `SUMMARY.json`, and `SUMMARY.md`.
- Keep `metrics.ndjson` Android-readable.
- Keep `LOG_INTERVAL=1` by default so short smoke runs also produce at least one metrics row.
- Report `loss`, `reward_accuracy`, `reward_margin`, `grad_norm`, RSS, reference-cache time, and train-loop time.
- Do not require Android device feedback.

## Local Implementation Plan

- Add `review-stage/scripts/run_hfrl_health_ab_220.sh`.
- Default to paired DPO forward for the 220-step run; keep cached reference logps opt-in after a short amortization check.
- Run a 1-step smoke first, then the full 220-step baseline.

## Local Result

- Run directory: `runs/hfrl_health_ab_dpo_s220`
- Steps logged: `220`
- Final train loss: `1.06563e-05`
- Final train reward accuracy: `1.0`
- Final train reward margin: `11.4494`
- Final eval loss: `0.00234474`
- Final eval reward accuracy: `1.0`
- Final eval reward margin: `8.96158`
- Held-out length-normalized pair accuracy: base `0.4531` -> DPO adapter `1.0`
- Held-out mean length-normalized margin: base `-0.1750` -> DPO adapter `5.6488`

## Device Dependency

No Android phone required.
