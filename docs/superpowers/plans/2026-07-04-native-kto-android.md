# Native KTO Android Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the preference-training path from bridge-backed PyTorch KTO to native C++ KTO that can be launched inside the Android app.

**Architecture:** Reuse the existing native DPO tokenizer, model, LoRA, optimizer, and Android JNI plumbing. Add KTO loss and KTO trainer beside the existing DPO trainer, expose it through the SDK, then make the Android visualizer call the SDK for saved A/B feedback.

**Tech Stack:** C++17 operators library, JNI, Java SDK wrapper, Kotlin Android visualizer, Gradle/JDK 17.

## Global Constraints

- Keep Qwen3-0.6B as the default model family.
- Preserve A/B user feedback as the UI collection format, but train with KTO objective internally.
- Do not depend on the FedCampus bridge for the main Android KTO training path.
- Keep the old DPO runner as a fallback/debug path only.
- Verify with native unit tests, Android build, APK install, and emulator run-folder output.

---

### Task 1: Native KTO Loss

**Files:**
- Modify: `operator/finetune_ops/core/dpo_loss.h`
- Modify: `operator/finetune_ops/core/dpo_loss.cpp`
- Modify: `operator/finetune_ops/core/test_dpo_loss.cpp`

**Interfaces:**
- Produces: `kto_loss_with_ref_logps(...)`, `streaming_kto_loss_with_ref_logps(...)`, and `compute_kto_metrics_from_logps(...)`.

- [ ] Write failing C++ tests for KTO metrics and streaming backward.
- [ ] Run `cmake --build ... --target test_dpo_loss` and verify the KTO symbols are missing.
- [ ] Implement dense/streaming KTO loss by reusing masked sequence logprob and logp gradient accumulation.
- [ ] Run `test_dpo_loss` and verify DPO regression tests plus KTO tests pass.

### Task 2: Native KTO Trainer

**Files:**
- Create: `operator/finetune_ops/optim/kto_trainer.h`
- Create: `operator/finetune_ops/optim/kto_trainer.cpp`
- Create: `operator/finetune_ops/optim/test_kto_trainer.cpp`
- Modify: `operator/CMakeLists.txt`
- Modify: `operator/include/mobile_finetuner/mobile_finetuner.h`

**Interfaces:**
- Produces: `KTOTrainer`, `KTOTrainerConfig`, `KTOTrainStepResult`.

- [ ] Write failing C++ trainer test proving LoRA parameters update from one chosen/rejected pair.
- [ ] Add KTO trainer implementation and CMake target.
- [ ] Run `test_kto_trainer`.

### Task 3: Android SDK KTO API

**Files:**
- Modify: `android-visualizer/mft-sdk/src/main/cpp/mobile_finetuner_jni.cpp`
- Modify: `android-visualizer/mft-sdk/src/main/java/com/mobilefinetuner/sdk/MobileFineTuner.java`

**Interfaces:**
- Produces: `createKtoTrainer(...)` and `trainKtoPreferenceBatch(...)`.

- [ ] Add Java wrapper and native declarations.
- [ ] Add JNI KTO trainer/session wiring.
- [ ] Run Android `:app:assembleDebug`.

### Task 4: Android App Native KTO Workflow

**Files:**
- Modify: `android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/viewmodel/ExperimentViewModel.kt`
- Modify: `android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/model/TrainingModels.kt`

**Interfaces:**
- Consumes: SDK `createKtoTrainer` and `trainKtoPreferenceBatch`.
- Produces: Android run folder with `metrics.ndjson`, `report.json`, and `train.log`.

- [ ] Route saved A/B feedback to SDK-native KTO when bridge mode is disabled.
- [ ] Emit KTO metrics in the same UI-readable NDJSON schema.
- [ ] Install APK, inject feedback JSONL, tap `Train KTO`, and verify Android files show native KTO results.
