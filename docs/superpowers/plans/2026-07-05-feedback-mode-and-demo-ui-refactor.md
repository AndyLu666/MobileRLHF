# Feedback Mode and Demo UI Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add A/B plus single thumbs up/down feedback and refactor the Android demo UI into a guided Generate -> Feedback -> Train -> Track workflow with live KTO training evidence.

**Architecture:** Feedback JSONL becomes a small typed boundary that can emit legacy-compatible pair rows and KTO-native single rows. `ExperimentViewModel` owns orchestration and file writes, while `PaperDemoConsoleState.kt` derives presentation state for the Compose screen. `DashboardScreen.kt` uses existing chart components but moves reward accuracy and reward margin into the primary demo path.

**Tech Stack:** Kotlin, Android Jetpack Compose, Android ViewModel, JSONL via `org.json`, existing native Qwen/KTO runner, JUnit + Truth, Gradle debug build.

## Global Constraints

- Keep legacy `prompt` / `chosen` / `rejected` pair records working.
- Single thumbs feedback must save one KTO-style labeled response and must not fabricate a rejected answer.
- Reward accuracy and reward margin are the primary training signals; loss is secondary.
- Smoke runs must be labeled as mechanics/trend validation, not final model-quality proof.
- Do not redesign native C++ trainer internals in this pass.
- Only commit files touched by this refactor; keep unrelated dirty KTO/native files untouched.

---

### Task 1: Feedback JSONL Schema

**Files:**
- Modify: `android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/data/FeedbackJsonl.kt`
- Modify: `android-visualizer/app/src/test/java/com/mobilefinetuner/visualizer/data/FeedbackJsonlTest.kt`

**Interfaces:**
- Produces: `FeedbackJsonl.RatingRecord(prompt: String, response: String, desirable: Boolean, sessionId: String, ratingIndex: Int, timestampMs: Long, source: String, userId: String)`
- Produces: `FeedbackJsonl.encodeRating(record: RatingRecord): String`
- Produces: `FeedbackJsonl.countKtoRows(lines: List<String>): Int`
- Produces: `FeedbackJsonl.hasTypedSingle(line: String): Boolean`

- [ ] **Step 1: Write failing tests**

Add tests that verify typed pair records include `"type":"pair"` while preserving `chosen` and `rejected`, and typed single records include `"type":"single"`, `"response"`, and boolean `"label"`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :app:testDebugUnitTest --tests com.mobilefinetuner.visualizer.data.FeedbackJsonlTest`

Expected: fails because `RatingRecord`, `encodeRating`, and typed pair fields are missing.

- [ ] **Step 3: Implement minimal schema helpers**

Add `type=pair` to `encodeChoice`, add `RatingRecord`, add `encodeRating`, and keep escaping through the existing JSON writer.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same Gradle test command. Expected: PASS.

### Task 2: ViewModel Single Thumbs Flow

**Files:**
- Modify: `android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/model/TrainingModels.kt`
- Modify: `android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/viewmodel/ExperimentViewModel.kt`
- Add or modify: focused JVM tests if existing ViewModel seams allow direct testing; otherwise cover this through `FeedbackJsonlTest` and `PaperDemoConsoleStateTest`.

**Interfaces:**
- Produces: `enum class NativeFeedbackMode { PAIR, SINGLE }`
- Produces: `NativeInferenceState.feedbackMode: NativeFeedbackMode`
- Produces: `NativeInferenceState.savedFeedbackRows: Int`
- Produces: `NativeInferenceState.lastRatingLabel: String?`
- Produces: `ExperimentViewModel.chooseNativeInferenceRating(desirable: Boolean)`

- [ ] **Step 1: Write failing state tests**

Extend `PaperDemoConsoleStateTest` so workflow treats `savedFeedbackRows > 0` as feedback saved even when `savedPairs == 0`.

- [ ] **Step 2: Run and verify RED**

Run: `JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :app:testDebugUnitTest --tests com.mobilefinetuner.visualizer.ui.PaperDemoConsoleStateTest`

Expected: fails because `savedFeedbackRows` does not exist.

- [ ] **Step 3: Add model fields and ViewModel rating write path**

Add the enum and state fields. In `runNativeInference`, clear stale pair candidates and rating status. Add `chooseNativeInferenceRating(desirable)` that requires a generated single output, writes `FeedbackJsonl.encodeRating`, increments `savedFeedbackRows`, sets `lastRatingLabel` to `UP` or `DOWN`, and refreshes runner training count.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same state test command. Expected: PASS.

### Task 3: Guided Demo UI

**Files:**
- Modify: `android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/DashboardScreen.kt`
- Modify: `android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/PaperDemoConsoleState.kt`
- Modify: `android-visualizer/app/src/test/java/com/mobilefinetuner/visualizer/ui/PaperDemoConsoleStateTest.kt`

**Interfaces:**
- Consumes: `NativeInferenceState.feedbackMode`
- Consumes: `ExperimentViewModel.runNativeInference()`
- Consumes: `ExperimentViewModel.runNativeInferencePair()`
- Consumes: `ExperimentViewModel.chooseNativeInferenceRating(Boolean)`
- Produces: primary screen sections `Feedback Lab`, `KTO Training`, and `Live Evidence`.

- [ ] **Step 1: Write failing UI-state tests**

Add tests for labels and progress state: single mode says `Single Rating`, pair mode says `A/B Compare`, and live evidence reports `3/10` from metrics.

- [ ] **Step 2: Run and verify RED**

Run: `JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :app:testDebugUnitTest --tests com.mobilefinetuner.visualizer.ui.PaperDemoConsoleStateTest`

Expected: fails because the mode labels and progress state do not exist.

- [ ] **Step 3: Refactor composables**

Replace the current generate/prefer panel with a mode switch. In pair mode, show A/B generation and Prefer buttons. In single mode, show single generation plus thumbs up/down buttons. Move model path, bridge URL, temp, top-p, and token controls into an Advanced expandable area.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same state test command. Expected: PASS.

### Task 4: Live Training Charts

**Files:**
- Modify: `android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/DashboardScreen.kt`
- Modify: `android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/PaperDemoConsoleState.kt`
- Modify: `android-visualizer/app/src/test/java/com/mobilefinetuner/visualizer/ui/PaperDemoConsoleStateTest.kt`

**Interfaces:**
- Consumes: `RunSnapshot.metrics`
- Produces: a live progress summary containing `stepLabel`, `rewardAccuracy`, `rewardMargin`, `loss`, and `peakMemory`.

- [ ] **Step 1: Write failing tests**

Add tests that a snapshot with step 3 of total 10 formats progress as `3 / 10`, reward margin as a primary signal, and missing metrics as `Waiting for metrics`.

- [ ] **Step 2: Run and verify RED**

Run: `JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :app:testDebugUnitTest --tests com.mobilefinetuner.visualizer.ui.PaperDemoConsoleStateTest`

Expected: fails because live progress state is missing.

- [ ] **Step 3: Implement live evidence panel**

Use existing `InteractiveMetricLineChart` for reward accuracy, reward margin, loss, and RSS when enough points exist. For one or zero points, show metric chips and the waiting message.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same state test command. Expected: PASS.

### Task 5: Verification

**Files:**
- No new production files unless tests reveal a focused fix.

**Commands:**
- `JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :app:testDebugUnitTest`
- `JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :app:assembleDebug`
- If adb is available: install and inspect the UI with a short demo flow.

- [ ] **Step 1: Run full JVM tests**

Expected: all tests pass.

- [ ] **Step 2: Build debug APK**

Expected: `:app:assembleDebug` succeeds.

- [ ] **Step 3: adb smoke check**

Install the debug APK, launch it, and verify the screen contains `Feedback Lab`, `A/B Compare`, `Single Rating`, `KTO Training`, and `Live Evidence`.

