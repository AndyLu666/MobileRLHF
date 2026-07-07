# MobileFineTuner Android KTO + Paper UI Progress Report

Date: 2026-07-06  
Workspace: `/Users/anakinju/DKU/MobileFineTuner`  
Branch: `fedcampus-inference-integration`  
Primary run verified on Android emulator: `phone_qwen_feedback_kto_10s_0706_200251`

This document is written for two readers:

- Human reader: what the demo can show now, what the numbers mean, and what is still missing.
- Codex / future agent: exact files, commands, run paths, and implementation status needed to continue without rediscovering the project.

## 1. Executive Summary

Current status: the first major update is now demo-ready for the narrow goal of showing a clean Android UI plus real on-device KTO training from DPO-compatible feedback data.

What is working now:

- The Android app can launch into a simplified paper/demo-facing KTO console.
- The first screen is no longer dominated by debug controls.
- The demo supports DPO-style `prompt/chosen/rejected` data and converts/uses it as KTO binary feedback internally.
- The app supports both preference styles at the UX/data level:
  - A/B pair preference: user chooses between answer A and answer B.
  - Single answer rating: user gives thumbs up/down on one output.
- The Android native C++ KTO trainer runs with Qwen3-0.6B assets on the emulator.
- A 10-step KTO run completed successfully on Android.
- The app shows live line charts for training loss and reward margin.
- The completed run saved `metrics.ndjson`, `report.json`, `reference_logps.jsonl`, a LoRA adapter, and before/after evidence files.

What this proves:

- Real Android native C++ training executes end to end.
- The KTO loss is finite across 10 steps.
- LoRA gradients are nonzero for trainable policy parameters.
- Frozen policy parameters did not receive gradients.
- The reference side is frozen / cached for the Android KTO run.
- Metrics are emitted in a format the UI can visualize.
- The UI can show a clear improving trend: loss down, reward margin up.

What this does not prove yet:

- It does not prove general long-answer quality improvement.
- It does not prove clinical correctness or health-coaching quality.
- It does not prove stable behavior across many seeds/devices.
- It does not prove full production readiness for arbitrary large models or long sequences.
- The "before/after" generation evidence is still limited and should not be oversold as a robust evaluation.

For the current MobileECom WIP demo goal, the current state is enough to show: "user feedback can be collected, converted into KTO-compatible supervision, trained on-device, and visualized with a clear training trend."

## 2. Human-Facing Screenshots

Screenshots were captured from the Android app after the completed KTO run.

| File | What it shows |
| --- | --- |
| `briefing/2026-07-06/screenshots/01_completed_live_training_top.png` | Completed demo screen, model name, status, and first live loss chart. |
| `briefing/2026-07-06/screenshots/02_completed_live_training_charts.png` | Loss and reward-margin line charts after the 10-step KTO run. |
| `briefing/2026-07-06/screenshots/03_completed_workflow_advanced_toggle.png` | Workflow summary: data saved, KTO done, evidence done, peak RSS, LoRA adapter saved. |
| `briefing/2026-07-06/screenshots/04_advanced_debug_controls_expanded.png` | Same completed area with advanced/debug section expanded; useful mainly for showing debug controls are no longer the first-screen experience. |

Suggested paper/demo use:

- Use screenshot 1 to show the product surface.
- Use screenshot 2 to show training trend visualization.
- Use screenshot 3 to show the end-to-end workflow status.

## 3. Main Android Run Evidence

Run name:

```text
phone_qwen_feedback_kto_10s_0706_200251
```

Device-side run directory:

```text
/data/user/0/com.mobilefinetuner.visualizer/files/mft_runs/phone_qwen_feedback_kto_10s_0706_200251
```

Saved files in the run directory:

```text
before_after_eval.json
before_after_generation.json
inference_feedback.jsonl
metrics.ndjson
qwen_kto_lora_adapter.jsonl
reference_logps.jsonl
report.json
train.log
```

Run configuration from `report.json`:

```json
{
  "status": "ok",
  "backend": "android_native_cpp",
  "objective": "kto",
  "model_dir": "/data/local/tmp/mft_qwen3",
  "train_samples": 48,
  "feedback_pairs": 24,
  "max_steps": 10,
  "batch_size": 2,
  "beta": 0.1,
  "lora_rank": 2,
  "lora_alpha": 4,
  "cached_reference_pairs": 24,
  "cached_reference_valid_response_tokens": 1091,
  "eval_pairs": 10,
  "max_train_rss_mb": 5279.34765625,
  "total_elapsed_ms": 830278
}
```

Training interpretation:

- Model path: `/data/local/tmp/mft_qwen3`
- Model family used by the demo: Qwen3-0.6B
- Feedback pairs: 24
- KTO training rows: 48, because each DPO pair becomes two binary KTO rows
- Steps: 10
- Batch size: 2 preference pairs per step, expanded to 4 KTO rows per step
- Sequence length: 64
- LoRA rank: 2
- LoRA alpha: 4
- KTO beta: 0.1
- Peak train RSS: about 5.28 GB
- Wall time: about 13.8 minutes

The UI also verified the completed state:

```text
COMPLETED
Live Training Charts
Loss: 0.395
Reward Margin: 0.885
```

## 4. Training Metrics

Full per-step metrics from `metrics.ndjson`:

| Step | Loss | Reward Accuracy | Reward Margin | RSS MB | Step Time MS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.50000000 | 0.00 | 0.00000000 | 5257.03 | 62342 |
| 2 | 0.49613664 | 0.75 | 0.03090973 | 5270.07 | 56582 |
| 3 | 0.49846030 | 0.50 | 0.01232071 | 5272.66 | 55553 |
| 4 | 0.49468353 | 0.75 | 0.04254418 | 5271.00 | 58421 |
| 5 | 0.47364780 | 1.00 | 0.21127625 | 5275.20 | 58788 |
| 6 | 0.46243763 | 0.75 | 0.30251083 | 5273.93 | 57029 |
| 7 | 0.46477840 | 1.00 | 0.28422853 | 5272.82 | 56814 |
| 8 | 0.44194610 | 1.00 | 0.47067186 | 5273.48 | 57044 |
| 9 | 0.40030822 | 1.00 | 0.82242393 | 5279.35 | 55593 |
| 10 | 0.39494130 | 1.00 | 0.88533133 | 5278.43 | 56846 |

Key trend:

- Loss decreased from `0.5000` to `0.3949`.
- Reward accuracy increased from `0.0` to `1.0`.
- Reward margin increased from `0.0` to `0.8853`.
- RSS stayed around 5.26-5.28 GB during the train steps.

Gradient / training integrity fields emitted by each step:

```text
policy_trainable_params = 224
policy_nonzero_grad_params = 224
policy_frozen_params_with_grad = 0
reference_grad_params = 0
reference_logps = base_model_cached
```

Why those fields matter:

- `policy_nonzero_grad_params = 224` means the trainable LoRA policy tensors received gradients.
- `policy_frozen_params_with_grad = 0` means frozen policy tensors were not accidentally updated.
- `reference_grad_params = 0` means the reference side was not trained.
- `reference_logps = base_model_cached` means the Android run cached base/reference log probabilities instead of training the reference.

## 5. Before/After Evidence Status

The run generated:

```text
before_after_generation.json
before_after_eval.json
```

The current before/after eval method is:

```text
reference_before_plus_training_reward_trend
```

Current `before_after_eval.json` summary:

```json
{
  "eval_pairs": 10,
  "method": "reference_before_plus_training_reward_trend",
  "reference_before": {
    "accuracy": 0,
    "mean_margin": -39.75934829711914
  },
  "train_reward_trend": {
    "first_step": {
      "accuracy": 0,
      "mean_margin": 0
    },
    "final_step": {
      "accuracy": 1,
      "mean_margin": 0.88533133
    },
    "delta": {
      "accuracy": 1,
      "mean_margin": 0.8853313326835632
    }
  }
}
```

Important caveat:

This is useful demo evidence, but it is not a full rigorous before/after generation-quality study. It mainly combines the cached reference baseline with the observed training reward trend. For paper claims, phrase this carefully as "training telemetry and reward-margin trend improved during the on-device run", not "the model is generally better at health advice."

## 6. What Was Implemented

### 6.1 KTO Data Path

Implemented DPO-pair to KTO-binary conversion:

- Source file: `scripts/convert_dpo_to_kto_jsonl.py`
- Function: `convert_dpo_pairs_to_kto_records`
- Rule:
  - `prompt + chosen` becomes `label: true`
  - `prompt + rejected` becomes `label: false`

Curated data files now exist:

```text
review-stage/data/health_dpo_curated_30.jsonl
review-stage/data/health_dpo_curated_train.jsonl
review-stage/data/health_dpo_curated_eval.jsonl
review-stage/data/health_kto_train.jsonl
review-stage/data/health_kto_eval.jsonl
```

The Android app also ships demo feedback assets:

```text
android-visualizer/app/src/main/assets/mft_demo_data/hf_fitness_pref_train.jsonl
android-visualizer/app/src/main/assets/mft_demo_data/hf_fitness_pref_eval.jsonl
```

Current Android run used feedback written as:

```text
inference_feedback.jsonl
```

### 6.2 PyTorch KTO Path

Implemented / updated Python-side KTO helpers:

```text
pytorch_alignment/dpo_utils.py
pytorch_alignment/kto_trainer.py
pytorch_alignment/kto_cli.py
```

Key Python pieces:

- `BinaryFeedbackDataset`
- `binary_feedback_collate_batch`
- `compute_kto_batch_metrics`
- `evaluate_kto`
- `KtoTrainingConfig`
- `train_kto`

The Python loss is a KL-centered KTO-style binary objective:

- Compute policy log probability on response tokens.
- Compute frozen reference log probability.
- Compute log ratio.
- Use nonnegative detached batch KL center.
- Reward: `beta * (logratio - kl_center)`.
- Desirable labels optimize `1 - sigmoid(reward)`.
- Undesirable labels optimize `1 - sigmoid(-reward)`.
- Supports `desirable_weight` and `undesirable_weight`.

This is useful for fast CPU/GPU sanity checks before moving into Android.

### 6.3 Native C++ KTO Trainer

Implemented native KTO trainer:

```text
operator/finetune_ops/optim/kto_trainer.h
operator/finetune_ops/optim/kto_trainer.cpp
```

Key class:

```cpp
ops::KTOTrainer
```

Key behavior:

- Accepts `PreferenceBatch`.
- Expands DPO-style chosen/rejected pairs into KTO binary samples.
- Assigns labels:
  - chosen rows: `+1`
  - rejected rows: `-1`
- Uses cached reference logps if available.
- Falls back to a reference model if cached logps are not available.
- Supports streaming KTO loss to reduce memory pressure.
- Runs backward pass.
- Applies Adam optimizer to trainable parameters.
- Returns KTO metrics per step.

Important config:

```cpp
struct KTOTrainerConfig {
    float learning_rate = 2e-4f;
    float weight_decay = 0.0f;
    float max_grad_norm = 1.0f;
    float beta = 0.1f;
    float desirable_weight = 1.0f;
    float undesirable_weight = 1.0f;
    bool use_streaming_kto_loss = true;
    int gradient_accumulation_steps = 1;
};
```

### 6.4 Android JNI / SDK Surface

Updated Android native bridge and SDK surface:

```text
android-visualizer/mft-sdk/src/main/cpp/mobile_finetuner_jni.cpp
android-visualizer/mft-sdk/src/main/java/com/mobilefinetuner/sdk/MobileFineTuner.java
```

The Android app calls native C++ training through the SDK and records output into per-run directories.

### 6.5 Android ViewModel Workflow

Main implementation file:

```text
android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/viewmodel/ExperimentViewModel.kt
```

Important behavior:

- Default demo config is 10 steps.
- `nativeKtoBatchSize(config)` returns:
  - batch 4 when `steps <= 5`
  - batch 2 when `steps > 5`
- Therefore the default paper demo is 10 steps / batch 2.
- The app writes:
  - `train.log`
  - `metrics.ndjson`
  - `report.json`
  - `reference_logps.jsonl`
  - `before_after_generation.json`
  - `before_after_eval.json`
  - `qwen_kto_lora_adapter.jsonl`
- The ViewModel polls run files and updates the UI state.

### 6.6 UI Redesign for Paper Demo

Main UI files:

```text
android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/DashboardScreen.kt
android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/PaperDemoConsoleState.kt
android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/components/InteractiveSeriesChart.kt
android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/components/MetricLineChart.kt
android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/viewmodel/PreferenceEvalSummary.kt
```

UI changes made:

- Added a simplified paper/demo console as the first experience.
- Moved messy test/debug controls behind `Show Advanced / Debug Controls`.
- Added a one-tap `Start KTO Training` / completed review flow.
- Added live training chart area.
- Added line charts for:
  - loss
  - reward margin
- Added workflow strip:
  - Data
  - Train KTO
  - Evidence
- Added result cards:
  - data ready
  - KTO completed
  - peak RSS
  - LoRA adapter saved
- Added clearer labels for DPO/KTO data, KTO rows, and current selected run.

Current visual state after completed run:

- Status: `COMPLETED`
- Chart title: `Live Training Charts`
- Loss label: `Loss: 0.395`
- Reward margin label: `Reward Margin: 0.885`
- Workflow cards: `Data Saved`, `Train KTO Done`, `Evidence Done`
- Summary cards: `DPO/KTO ready`, `KTO completed`, `5279.3 MB`, `Adapter saved`

## 7. Supported Feedback Modes

The app/data model now has support for two feedback modes:

```kotlin
enum class NativeFeedbackMode {
    PAIR,
    SINGLE
}
```

PAIR mode:

- Generate or show two answers.
- User chooses A or B.
- Saves DPO-style preference pair.
- Can become KTO binary data by chosen/rejected expansion.

SINGLE mode:

- Generate or show one answer.
- User gives thumbs up/down.
- Saves binary feedback rows directly.
- This is the more token/memory efficient path because the app does not need to generate two answers.

Current implementation status:

- Data model exists.
- UI/state supports the modes.
- KTO path can consume binary labels.
- DPO-style pair data can be converted to binary KTO rows.
- More polish is still needed if we want the single-output thumbs flow to be the dominant user-facing interaction in a production app.

## 8. Verification Commands Already Run

Earlier build/test verification after the UI changes:

```bash
cd android-visualizer
./gradlew :app:testDebugUnitTest
./gradlew :app:assembleDebug
```

Both completed successfully in the previous verification cycle before the Android run.

Android install/launch/run verification:

```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"
"$ADB" install -r android-visualizer/app/build/outputs/apk/debug/app-debug.apk
"$ADB" shell monkey -p com.mobilefinetuner.visualizer -c android.intent.category.LAUNCHER 1
```

Android evidence commands used for the completed run:

```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"
PKG=com.mobilefinetuner.visualizer
RUN=/data/user/0/$PKG/files/mft_runs/phone_qwen_feedback_kto_10s_0706_200251

"$ADB" shell run-as "$PKG" cat "$RUN/report.json"
"$ADB" shell run-as "$PKG" tail -10 "$RUN/metrics.ndjson"
"$ADB" shell run-as "$PKG" cat "$RUN/train.log"
```

UI state verification:

```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"
"$ADB" shell uiautomator dump /sdcard/window.xml >/dev/null
"$ADB" shell cat /sdcard/window.xml \
  | rg -o 'text="[^"]*"' \
  | rg 'COMPLETED|Live Training Charts|Loss:|Reward Margin:'
```

Observed UI text:

```text
text="COMPLETED"
text="Live Training Charts"
text="Loss: 0.395"
text="Reward Margin: 0.885"
```

Screenshot capture commands:

```bash
ADB="$HOME/Library/Android/sdk/platform-tools/adb"
OUT_DIR="briefing/2026-07-06/screenshots"
mkdir -p "$OUT_DIR"
"$ADB" exec-out screencap -p > "$OUT_DIR/01_completed_live_training_top.png"
```

## 9. Current Git / Workspace State

Recent commits on this branch:

```text
2e8c672 feat: split paper demo from debug controls
813ce04 feat: show training preview charts in demo
663c74c feat: add first-screen kto training action
05deb5d feat: clarify demo data run labels
f8726d7 fix: make demo app background opaque
7f97684 feat: tighten kto demo readiness ui
59b8ab4 feat: polish kto demo console
0c62d47 feat: record native kto rss metrics
```

Important: the worktree is not clean. There are many modified and untracked files from the KTO/native/UI work. Do not reset or discard them.

Major changed or new areas:

```text
pytorch_alignment/
scripts/
review-stage/data/
review-stage/scripts/
operator/finetune_ops/
operator/include/mobile_finetuner/
android-visualizer/app/
android-visualizer/mft-sdk/
docs/superpowers/plans/
PROJECT_STRUCTURE.md
```

This report and screenshots were added under:

```text
briefing/2026-07-06/
```

## 10. What Is Done

Done for the current demo target:

- KTO path exists in Python.
- KTO path exists in native C++.
- Android app can run native C++ KTO.
- Android run completed with Qwen3-0.6B path.
- DPO-compatible data can feed KTO training.
- Default demo uses 10 steps and batch 2.
- 5-step mode maps to batch 4 by the current Android batch rule.
- 10-step mode maps to batch 2 by the current Android batch rule.
- UI has real training line charts.
- UI hides advanced controls by default.
- UI shows run status, training evidence, peak memory, and saved adapter status.
- Screenshots exist for paper/demo review.

## 11. What Is Not Done Yet

High-priority but not finished:

- A polished, final paper-story walkthrough is not yet written.
- Before/after generation comparison is not rigorous enough for strong quality claims.
- Single-output thumbs up/down flow needs more product polish if it becomes the main user path.
- The app does not yet have a fully guided "new user" flow with minimal terminology.
- The UI is better, but some text still wraps awkwardly on narrow mobile widths, for example `KTO completed` splitting across lines in the summary card.
- The charts are visually useful, but not yet publication-perfect; axis labels and small instructional text can be simplified for a paper figure.
- The Android run is from emulator; we still need physical-device validation if the paper/demo expects phone hardware claims.
- Multi-seed or repeated-run stability has not been tested.
- Memory is high for mobile: peak train RSS around 5.28 GB during training, higher during post-train inference/save/load stages.
- No robust energy/thermal measurement was captured in this run.
- No cloud/off-device baseline comparison is included in this report.
- No formal ablation exists yet for DPO vs KTO, A/B vs thumbs, 5-step vs 10-step, or batch 4 vs batch 2.

Lower-priority cleanup still needed:

- Decide whether to keep old visualizer tabs or make a separate paper-only app surface.
- Clean old naming where files/classes still mention DPO even when the path is now KTO.
- Make docs consistent across `README.md`, `scripts/README.md`, `pytorch_alignment/README.md`, and `PROJECT_STRUCTURE.md`.
- Consolidate runbook commands for Android install, model push, run, and evidence pull.
- Consider exporting the completed run report to a repo-side artifact automatically.

## 12. Honest Claim Boundaries

Safe claims for a meeting or WIP paper:

- "We implemented a KTO training path that can consume DPO-style feedback by converting preference pairs into binary desirable/undesirable examples."
- "The Android app now runs a native C++ KTO training loop on Qwen3-0.6B assets."
- "A 10-step Android run completed with finite loss, nonzero LoRA gradients, frozen reference behavior, saved adapter output, and live UI charts."
- "The demo shows loss decreasing and reward margin increasing over the short training run."
- "The UI now presents a cleaner paper-facing workflow and hides debug controls by default."

Do not claim yet:

- "The model is clinically better."
- "The trained model gives better health advice in general."
- "This is production-ready."
- "This has been validated on all Android devices."
- "This is memory efficient enough for all phones."
- "The before/after generation quality is proven."

## 13. Next Recommended Steps

Recommended order:

1. Make one final UI pass for paper screenshots.
   - Fix awkward line wraps in summary cards.
   - Make chart axis text cleaner.
   - Possibly add a screenshot-friendly "Paper Mode" lock.

2. Add a clearer single-output thumbs flow.
   - One generated answer.
   - Thumbs up/down.
   - Saved directly as KTO binary record.
   - Avoid forcing users to understand DPO pairs.

3. Improve before/after evidence.
   - Run the same prompts before and after adapter load.
   - Save outputs side by side.
   - Add a cautious preference-score or rubric evaluation.

4. Add physical-device validation.
   - Same 5-step and 10-step configs.
   - Capture RSS, time, and thermal if available.

5. Prepare paper assets.
   - One workflow figure.
   - One app screenshot panel.
   - One line-chart panel.
   - One concise table of training params and run result.

## 14. Codex Handoff Checklist

If another Codex session continues from here:

1. Start in:

   ```bash
   cd /Users/anakinju/DKU/MobileFineTuner
   git status --short
   ```

2. Read this report:

   ```text
   briefing/2026-07-06/ANDROID_KTO_UI_PROGRESS_2026-07-06.md
   ```

3. Check the current Android run:

   ```bash
   ADB="$HOME/Library/Android/sdk/platform-tools/adb"
   PKG=com.mobilefinetuner.visualizer
   RUN=/data/user/0/$PKG/files/mft_runs/phone_qwen_feedback_kto_10s_0706_200251
   "$ADB" shell run-as "$PKG" cat "$RUN/report.json"
   ```

4. Inspect these implementation files first:

   ```text
   android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/viewmodel/ExperimentViewModel.kt
   android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/DashboardScreen.kt
   android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/PaperDemoConsoleState.kt
   android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/model/TrainingModels.kt
   operator/finetune_ops/optim/kto_trainer.cpp
   operator/finetune_ops/optim/kto_trainer.h
   pytorch_alignment/dpo_utils.py
   pytorch_alignment/kto_trainer.py
   scripts/convert_dpo_to_kto_jsonl.py
   ```

5. Before claiming anything is complete, rerun at least:

   ```bash
   cd android-visualizer
   ./gradlew :app:testDebugUnitTest
   ./gradlew :app:assembleDebug
   ```

6. For a fresh Android proof run, use the app's default KTO demo start button and then pull:

   ```bash
   ADB="$HOME/Library/Android/sdk/platform-tools/adb"
   PKG=com.mobilefinetuner.visualizer
   "$ADB" shell run-as "$PKG" ls -lt /data/user/0/$PKG/files/mft_runs | head
   ```

7. Treat 5-10 step runs as smoke/demo evidence only. They prove mechanics and trend telemetry, not broad model quality.

