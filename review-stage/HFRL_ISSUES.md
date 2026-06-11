# MobileFineTuner + RL / HFRL Issue Tracker

Date: `2026-06-06`
Scope: use existing preference datasets as simulated human feedback first, then make the flow observable and usable on Android.

## Goal

Build an interactive HFRL path for MobileFineTuner:

1. Start with offline preference JSONL as fake human feedback.
2. Run native C++ DPO/RL-style LoRA training on-device-compatible code paths.
3. Surface training, reward, and feedback metrics in the Android app.
4. Leave a clean upgrade path from dataset-simulated feedback to real interactive human feedback.

## Issues

### HFRL-1: Define the runnable feedback contract

- Status: in progress
- Need: one data contract that both fake feedback and later interactive feedback can use.
- Current base: `review-stage/data/*pref*.jsonl` with `prompt`, `chosen`, `rejected`.
- Minimum fix:
  - Treat preference JSONL as the first feedback source.
  - Document fields and the eventual Android feedback extension.
  - Keep DPO as the first RL objective because it is already implemented in native C++.

### HFRL-2: Make native DPO produce Android-readable structured metrics

- Status: done
- Problem: native DPO logs reward metrics in text, but Android only consumes loss/ppl/lr/tokens from logs or `metrics.ndjson`.
- Minimum fix:
  - Emit per-step DPO metrics to `metrics.ndjson` or `--eval_out`.
  - Include `loss`, `lr`, `tokens`, `reward_accuracy`, `reward_margin`, `chosen_reward`, `rejected_reward`, `grad_norm`, and `rss`.
  - Preserve existing text logs.
- 2026-06-06 update:
  - Native DPO now emits per-step `metrics.ndjson`.
  - DPO chosen/rejected are batched together for policy and reference forward, reducing DPO forward calls from 4 to 2 per micro-step.

### HFRL-3: Extend Android parser/data model for RL metrics

- Status: done
- Problem: `StepMetric` cannot represent reward metrics, so HFRL runs are not first-class in the app.
- Minimum fix:
  - Add nullable reward fields to `StepMetric`.
  - Parse DPO train/eval text logs.
  - Parse reward fields from `metrics.ndjson`.
  - Add focused parser tests.

### HFRL-4: Add Android UI affordances for HFRL

- Status: done
- Problem: even after parsing, charts and KPI cards are still SFT-oriented.
- Minimum fix:
  - Show reward accuracy/margin cards when present.
  - Add reward charts to the training tab or replace PPL when unavailable.
  - Keep SFT/Gemma views unchanged.
- 2026-06-06 update:
  - Android overview shows an `HFRL` run-mode pill when reward metrics are present.
  - Overview and training cards switch from PPL/RSS to reward accuracy/reward margin for DPO/HFRL runs.
  - SFT/Gemma runs keep the existing PPL/RSS layout.

### HFRL-5: One-command local/offline HFRL smoke

- Status: done
- Problem: previous scripts validate native DPO, but not a named HFRL demo run contract for Android.
- Minimum fix:
  - Add a script that runs native DPO on a small preference subset.
  - Produce a run folder with `train.log`, `metrics.ndjson`, optional `rss.csv`, and LoRA output.
  - Verify Android can scan the folder.
- 2026-06-06 validation:
  - `runs/hfrl_offline_android_smoke_batched_dpo` completed 1 native DPO step with batched chosen/rejected forward.
  - Runtime: `real 3.47s` with BLAS build.
  - Gradient check from log: `nonzero_grad_tensors=24`.
  - Android metrics file: `metrics.ndjson`.
- 2026-06-06 performance compare:
  - Output: `runs/hfrl_dpo_forward_mode_compare/SUMMARY.md`.
  - 4-forward baseline, 4 steps: `real=13.08s`, `train_loop=12.3239s`, `loss=0.696726`, `reward_margin=-0.00714569`, `nonzero_grad=24`.
  - 2-forward paired chosen/rejected, 4 steps: `real=10.76s`, `train_loop=10.4642s`, `loss=0.696727`, `reward_margin=-0.00714724`, `nonzero_grad=24`.
  - cached-reference 1-policy-forward training, 4 steps: `real=30.46s`, `train_loop=6.5674s`, `ref_cache=23.6079s`, `loss=0.696727`, `reward_margin=-0.00714724`, `nonzero_grad=24`.
  - Interpretation: paired forward is an immediate win with no meaningful metric regression; reference cache only pays off when the cache cost is amortized over longer runs or reused across epochs.
  - 20-step compare: paired `real=54.04s`, cached-reference `real=56.68s`; cached train loop was faster (`32.6341s` vs `53.761s`) but total time was still slightly worse because `ref_cache=23.7691s`.
  - 50-step compare: paired `real=131.45s`, cached-reference `real=105.21s`; metrics were identical (`loss=0.606498`, `reward_margin=0.181525`, `nonzero_grad=24`), so cache is beneficial for longer static-feedback runs.

### HFRL-6: Real Android execution path

- Status: in progress
- Problem: Android visualizer can read runs, but the actual native trainer is not packaged/launched as an Android task.
- Minimum fix:
  - Decide first integration mode: push native binary + run via adb shell, or package JNI/NDK runner.
  - Start with adb-shell binary mode for fastest real-device validation.
  - Later upgrade to in-app interactive feedback collection and training launch.
- 2026-06-06 update:
  - Added QA script support for pushing a local HFRL run folder to `/sdcard/Download/mft_runs`.
  - This validates the Android-readable run-folder path on a real device once adb is available outside the sandbox.
  - Remaining gap: package or launch the native trainer on Android, instead of only pushing completed run artifacts.

### HFRL-7: Interactive feedback upgrade

- Status: in progress
- Problem: current "human feedback" is simulated by dataset labels.
- Minimum fix:
  - Add an append-only feedback JSONL format generated from Android choices.
  - Allow the user to compare two responses and mark preferred/rejected.
  - Feed the accumulated JSONL into native DPO.
- 2026-06-08 update:
  - `Test DPO` now appends Android A/B choices to DPO-compatible JSONL.
  - Feedback rows include `prompt`, `chosen`, `rejected`, `source`, `user_id`, `session_id`, labels, choice index, and timestamp.
  - Rows are written under the app-specific external files area when available, falling back to app-private files.
  - The remaining gap is launching native DPO on-device against the generated JSONL instead of the current simulated DPO telemetry.
- 2026-06-09 emulator validation:
  - AVD `Pixel_7a` installed the debug APK and ran `Test DPO`.
  - Five A/B taps produced `/sdcard/Android/data/com.mobilefinetuner.visualizer/files/hfrl_feedback/android_ab_feedback.jsonl`.
  - The generated file has 5 DPO-compatible rows and was copied to `verification/emulator_hfrl_20260608/android_ab_feedback.jsonl`.
  - Android NDK `27.3.13750724` was installed locally and the native GPT-2 trainer was cross-compiled for Android arm64.
  - The Android arm64 native trainer ran on `emulator-5554` against the generated feedback JSONL.
  - A constrained 1-step native DPO smoke completed with `seq_len=32`, `rank=1`, peak RSS about `1104.84 MB`, and saved a LoRA adapter.
  - Evidence: `verification/emulator_native_hfrl_small/REPORT.md`.
  - The `Pixel_7a` AVD was cold-booted with `hw.ramSize=8192` and verified inside Android with `MemTotal: 8129460 kB`.
  - A longer 20-step native DPO run completed on the 8GB AVD using the Android-generated feedback JSONL.
  - Metrics moved away from the DPO `log(2)` baseline: initial train loss `0.693147`, minimum train loss `0.461023`, final eval loss `0.4395`, final eval reward margin `0.5943`.
  - Peak RSS stabilized around `1954.24 MB`, and the LoRA adapter was saved successfully.
  - Evidence: `verification/emulator_native_hfrl_s20_ram8g/REPORT.md`.

## Current Review Assessment

External LLM review was requested through `auto-review-loop-llm`, but no `llm-chat` MCP tool or `LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL` environment configuration is available in this session. This round therefore uses local strict review as fallback and records the missing external review as a process blocker, not an implementation blocker.

Local score: `4/10`

Verdict: not ready as interactive HFRL, but ready to start an offline HFRL vertical slice.

Critical weaknesses:

1. The native DPO core exists, but Android cannot yet understand reward metrics.
2. There is no named "HFRL run folder" contract for real-device demos.
3. Human feedback is still simulated by static preference datasets.
4. The final Android execution path is not decided.

## Execution Order

1. HFRL-2 and HFRL-3: make reward metrics observable on Android. Done.
2. HFRL-5: create a repeatable offline HFRL run folder. Done.
3. Use reference-logp cache only for longer static-feedback runs or repeated epochs. Based on current measurements, keep it off for quick smoke and enable around 50+ steps or when reusing the same feedback dataset.
4. HFRL-4: improve UI for reward-specific metrics. Done.
5. HFRL-7: Android A/B feedback JSONL collection started locally. Done for append-only local JSONL.
6. HFRL-6: launch native DPO on-device against generated feedback JSONL. Done via adb-shell emulator path; remaining work is in-app launcher packaging.
