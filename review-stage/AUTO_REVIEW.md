# Auto Review Loop: MobileFineTuner + RL / HFRL

## Round 1 (2026-06-06 14:45)

### Assessment (Summary)

- Score: 4/10
- Verdict: not ready
- Key criticisms:
  - Native DPO exists, but Android does not yet parse or display reward metrics.
  - The current human feedback source is a static preference dataset, not interactive feedback.
  - There is no named HFRL run folder contract that a phone can scan end to end.
  - The real Android execution mode is still undecided.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

External LLM review was requested via `auto-review-loop-llm`, but this session has no exposed `llm-chat` MCP tool and no `LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL` environment variables for the curl fallback. Local strict review was used as fallback.

Score: 4/10.

Verdict: not ready for an interactive HFRL claim. The project is ready for an offline HFRL vertical slice using preference JSONL as simulated human feedback.

Minimum fixes:
1. Make native DPO emit Android-readable structured metrics.
2. Extend Android parsing/data model for reward accuracy, reward margin, chosen reward, and rejected reward.
3. Create a repeatable HFRL smoke run folder containing `train.log` and `metrics.ndjson`.
4. Keep the claim narrow: dataset-simulated human feedback first, interactive Android feedback later.

</details>

### Actions Taken

- Created `review-stage/HFRL_ISSUES.md` and timestamped copy.
- Initialized `review-stage/REVIEW_STATE.json`.
- Added native `--metrics_out` support for per-step Android-readable metrics.
- Added DPO/HFRL reward fields to Android `StepMetric`, log parser, NDJSON parser, demo data, and overview/training UI.
- Added `review-stage/scripts/run_hfrl_offline_android_smoke.sh` for a fake-human-feedback native DPO run folder.
- Ran a 2-step offline HFRL smoke. It produced `runs/hfrl_offline_android_smoke/train.log` and two `metrics.ndjson` rows with reward metrics.
- Optimized native DPO forward path by batching chosen/rejected together for policy and reference scoring. Per micro-step forward calls drop from 4 to 2.
- Ran a 1-step BLAS smoke at `runs/hfrl_offline_android_smoke_batched_dpo`; it completed in `real 3.47s` and produced `nonzero_grad_tensors=24`.
- Added optional cached-reference logps. Training-time DPO can run with one policy forward when feedback data is static.
- Added `review-stage/scripts/compare_hfrl_dpo_forward_modes.sh` and ran 4-step performance comparison across 4-forward, 2-forward, and cached-reference modes.
- Added an Android `HFRL` run-mode indicator plus reward-specific overview/training telemetry when `metrics.ndjson` contains reward metrics.
- Extended `android-visualizer/scripts/qa/mobile_device_qa.sh` with `--push-run` for pushing offline HFRL run folders to `/sdcard/Download/mft_runs`.

### Results

- Native build check passed: `cmake --build gpt2_small_lora_finetune/build --target train -j2`.
- BLAS build check passed: `cmake --build gpt2_small_lora_finetune/build_release_blas --target train -j2`.
- Forward-mode performance comparison:
  - 4-forward: `real=13.08s`, `train_loop=12.3239s`, `loss=0.696726`, `reward_margin=-0.00714569`, `nonzero_grad=24`.
  - 2-forward: `real=10.76s`, `train_loop=10.4642s`, `loss=0.696727`, `reward_margin=-0.00714724`, `nonzero_grad=24`.
  - cached-reference: `real=30.46s`, `train_loop=6.5674s`, `ref_cache=23.6079s`, `loss=0.696727`, `reward_margin=-0.00714724`, `nonzero_grad=24`.
  - Decision: keep paired 2-forward as default; use cached-reference only for longer/reused static-feedback runs.
- Amortization checks:
  - 20-step: paired `real=54.04s`, cached-reference `real=56.68s`; cache not yet worth it in total wall time.
  - 50-step: paired `real=131.45s`, cached-reference `real=105.21s`; cache wins total wall time with unchanged loss/reward/grad.
  - Updated decision: keep paired 2-forward as the smoke/default path, enable cached-reference for longer static-feedback runs.
- Android Gradle unit tests could not run in sandbox: default Java was missing, and the Android Studio JBR retry was blocked on `~/.gradle` wrapper lock permissions.
- ADB real-device validation could not run in sandbox because the adb daemon could not bind its smart socket. The QA script now has the device-side push path ready for a real terminal.
- The first smoke used `EVAL_INTERVAL=STEPS` and reached the DPO eval stage slowly. The script now defaults to `EVAL_INTERVAL=0` for quick Android run-folder generation.
- Residual process note: the first smoke eval process was still running as PID `6285` during review, but termination approval was denied/timeout. The generated training artifacts are already present.

### Status

- HFRL-2/HFRL-3/HFRL-4/HFRL-5 complete. HFRL-6 is in progress with an adb-pushed run-folder path; remaining work is native trainer launch/package on Android and real interactive feedback collection.
