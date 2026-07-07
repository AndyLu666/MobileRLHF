# KTO Before/After Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight before/after preference-eval artifact so the Android KTO demo can show whether reward margin improves after training.

**Architecture:** Reuse the existing Android native `scorePreferenceBatch` API to score the same eval subset before LoRA initialization and after KTO training. Keep the UI parser stable by writing the new evidence to `before_after_eval.json` and embedding a compact summary in `report.json`.

**Tech Stack:** Kotlin Android ViewModel, MobileFineTuner Java SDK, native C++ score path already exposed through JNI, Gradle Android build.

## Global Constraints

- Keep the change demo-scoped: no long-running new training configuration.
- Do not alter existing `metrics.ndjson` row schema.
- Use the existing 30-pair Qwen A/B JSONL as the default data source.
- Report honest trend metrics only: accuracy before/after, mean margin before/after, margin delta.

---

### Task 1: Eval Metric Helper

**Files:**
- Modify: `android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/viewmodel/ExperimentViewModel.kt`

**Interfaces:**
- Consumes: `FloatArray chosenLogps`, `FloatArray rejectedLogps`.
- Produces: `PreferenceEvalSummary(pairCount: Int, accuracy: Double, meanMargin: Double)` and `summarizePreferenceScores(...)`.

- [ ] Add a private data class and helper inside `ExperimentViewModel`.
- [ ] Validate by compiling the Android app.

### Task 2: Runner Before/After Eval

**Files:**
- Modify: `android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/viewmodel/ExperimentViewModel.kt`

**Interfaces:**
- Consumes: `MobileFineTuner.scorePreferenceBatch(...)`.
- Produces: `before_after_eval.json` with `before`, `after`, and `delta` objects.

- [ ] Score an eval subset before `initLora`.
- [ ] Score the same subset after KTO training and adapter reload.
- [ ] Write `before_after_eval.json`.
- [ ] Add eval summary fields to `report.json`.

### Task 3: Verification

**Files:**
- Read: Android build output and emulator run artifacts.

- [ ] Run `JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :app:assembleDebug`.
- [ ] Install the APK and run `Train KTO` on `emulator-5554`.
- [ ] Verify `before_after_eval.json`, `report.json`, `metrics.ndjson`, and `reference_logps.jsonl`.
