# Paper Demo Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Android visualizer's primary Demo screen with a focused paper-ready KTO workflow console.

**Architecture:** Add a small formatting/status helper for demo workflow state, then refactor the existing Compose Demo tab around four visible sections: workflow, generate/prefer, train KTO, and evidence. Keep the existing detailed tabs available through the existing tab row.

**Tech Stack:** Kotlin, Jetpack Compose Material 3, Android Gradle, Truth/JUnit unit tests.

## Global Constraints

- Do not change native C++ training behavior or PyTorch KTO behavior.
- Do not claim smoke-run quality improvement; label result fields as `training trend`.
- Main screen must keep `reward_accuracy` and `reward_margin` more prominent than loss.
- Use Qwen3-0.6B wording in the paper-facing UI.
- Keep text compact enough for phone screenshots.
- Avoid large decorative gradients, orbs, nested cards, and oversized hero typography.
- Use TDD for behavior helpers before production code.

---

### Task 1: Demo Console State Helpers

**Files:**
- Create: `android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/PaperDemoConsoleState.kt`
- Test: `android-visualizer/app/src/test/java/com/mobilefinetuner/visualizer/ui/PaperDemoConsoleStateTest.kt`

**Interfaces:**
- Consumes: `NativeInferenceState`, `NativeDemoRunnerState`, `NativeDemoConfig`, `RunSnapshot`.
- Produces:
  - `enum class DemoWorkflowStatus { READY, NEEDS_INPUT, SAVED, RUNNING, DONE }`
  - `data class DemoWorkflowStepState(val label: String, val status: DemoWorkflowStatus)`
  - `data class DemoEvidenceState(...)`
  - `fun buildDemoWorkflowSteps(inference: NativeInferenceState, runner: NativeDemoRunnerState, snapshot: RunSnapshot?): List<DemoWorkflowStepState>`
  - `fun buildDemoEvidenceState(config: NativeDemoConfig, runner: NativeDemoRunnerState, snapshot: RunSnapshot?): DemoEvidenceState`
  - `fun formatDemoMetric(value: Double?, digits: Int = 3): String`

- [ ] **Step 1: Write failing tests**

```kotlin
package com.mobilefinetuner.visualizer.ui

import com.google.common.truth.Truth.assertThat
import com.mobilefinetuner.visualizer.model.NativeDemoRunStatus
import com.mobilefinetuner.visualizer.model.NativeDemoRunnerState
import com.mobilefinetuner.visualizer.model.NativeInferenceState
import org.junit.Test

class PaperDemoConsoleStateTest {
    @Test
    fun workflowMarksPreferSavedWhenFeedbackPairsExist() {
        val steps = buildDemoWorkflowSteps(
            inference = NativeInferenceState(savedPairs = 3),
            runner = NativeDemoRunnerState(status = NativeDemoRunStatus.READY),
            snapshot = null
        )

        assertThat(steps.map { it.label }).containsExactly("Generate", "Prefer", "Train", "Evidence").inOrder()
        assertThat(steps[0].status).isEqualTo(DemoWorkflowStatus.DONE)
        assertThat(steps[1].status).isEqualTo(DemoWorkflowStatus.SAVED)
        assertThat(steps[2].status).isEqualTo(DemoWorkflowStatus.READY)
        assertThat(steps[3].status).isEqualTo(DemoWorkflowStatus.NEEDS_INPUT)
    }

    @Test
    fun workflowMarksTrainingAndEvidenceDoneFromRunnerAndSnapshot() {
        val steps = buildDemoWorkflowSteps(
            inference = NativeInferenceState(savedPairs = 30),
            runner = NativeDemoRunnerState(status = NativeDemoRunStatus.COMPLETED),
            snapshot = demoSnapshot(
                loss = 0.4939,
                rewardAccuracy = 0.5,
                rewardMargin = 0.052
            )
        )

        assertThat(steps[2].status).isEqualTo(DemoWorkflowStatus.DONE)
        assertThat(steps[3].status).isEqualTo(DemoWorkflowStatus.DONE)
    }

    @Test
    fun evidenceHighlightsRewardMetricsAndFormatsMissingValues() {
        val evidence = buildDemoEvidenceState(
            config = com.mobilefinetuner.visualizer.model.NativeDemoConfig(steps = 10),
            runner = NativeDemoRunnerState(trainingPairCount = 30),
            snapshot = demoSnapshot(
                loss = 0.4939921,
                rewardAccuracy = 0.5,
                rewardMargin = 0.05205345
            )
        )

        assertThat(evidence.objective).isEqualTo("KTO")
        assertThat(evidence.model).isEqualTo("Qwen3-0.6B")
        assertThat(evidence.pairs).isEqualTo("30")
        assertThat(evidence.steps).isEqualTo("10")
        assertThat(evidence.loss).isEqualTo("0.494")
        assertThat(evidence.rewardAccuracy).isEqualTo("0.500")
        assertThat(evidence.rewardMargin).isEqualTo("0.052")
        assertThat(formatDemoMetric(null)).isEqualTo("N/A")
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :app:testDebugUnitTest --tests 'com.mobilefinetuner.visualizer.ui.PaperDemoConsoleStateTest'
```

Expected: fails with unresolved references for new helper types/functions.

- [ ] **Step 3: Implement helper**

Create `PaperDemoConsoleState.kt` with focused formatting and status mapping. Use the latest metric from `snapshot.metrics.lastOrNull()` and avoid reading report files directly in this task.

- [ ] **Step 4: Run helper tests**

Run the same Gradle test command.

Expected: pass.

---

### Task 2: Paper Demo Console Compose Layout

**Files:**
- Modify: `android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/DashboardScreen.kt`

**Interfaces:**
- Consumes Task 1 helpers.
- Produces a refactored `ExperimentsTab` that renders:
  - workflow strip
  - A/B generation and preference section
  - KTO training section
  - evidence section

- [ ] **Step 1: Update `ExperimentsTab` structure**

Replace the old sequence of `DemoCommandCenter`, optional feedback panel, `NativeInferenceCard`, and `NativeDemoRunnerCard` with:

```kotlin
PaperDemoConsole(
    runs = runs,
    selectedRunId = selectedRunId,
    compareRunId = compareRunId,
    config = config,
    runner = runner,
    inference = inference,
    snapshot = snapshot,
    feedbackSession = feedbackSession,
    onPickDirectory = onPickDirectory,
    onEnableDemo = onEnableDemo,
    onEnableDpoTest = onEnableDpoTest,
    onChooseFeedback = onChooseFeedback,
    onInferenceChange = onInferenceChange,
    onGenerate = onGenerate,
    onGeneratePair = onGeneratePair,
    onChooseCandidate = onChooseCandidate,
    onConfigChange = onConfigChange,
    onPreset = onPreset,
    onStart = onStart,
    onStop = onStop
)
```

- [ ] **Step 2: Add new composables**

Add focused composables in `DashboardScreen.kt`:

```kotlin
@Composable private fun PaperDemoConsole(...)
@Composable private fun WorkflowStrip(steps: List<DemoWorkflowStepState>)
@Composable private fun PaperGeneratePreferPanel(...)
@Composable private fun PaperKtoTrainingPanel(...)
@Composable private fun PaperEvidencePanel(...)
@Composable private fun PaperEvidenceMetric(label: String, value: String, accent: Color)
```

Reuse existing low-level controls where possible: `IntField`, `DecimalField`, `ToggleRow`, `CandidatePreferenceBlock`, `MiniMetricChip`, and `FeedbackCollectionPanel`.

- [ ] **Step 3: Visual constraints**

Use white or near-white surfaces, 8dp card corners, thin dividers, and compact titles. Remove decorative background blobs from the primary screen by simplifying `DashboardScreen` background to a flat light background.

- [ ] **Step 4: Keep developer details available**

Do not remove existing `OverviewTab`, `TrainingTab`, `ComparisonTab`, and `LogsTab`. Keep the tab row.

---

### Task 3: Build and Emulator Verification

**Files:**
- No new source files expected.

**Interfaces:**
- Consumes Task 1 and Task 2.
- Produces verified Android APK and screenshot evidence.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :app:testDebugUnitTest --tests 'com.mobilefinetuner.visualizer.ui.PaperDemoConsoleStateTest'
```

Expected: pass.

- [ ] **Step 2: Run existing Android unit tests**

Run:

```bash
JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :app:testDebugUnitTest
```

Expected: pass.

- [ ] **Step 3: Build debug APK**

Run:

```bash
JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home' ./gradlew :app:assembleDebug
```

Expected: build success.

- [ ] **Step 4: Install and inspect on emulator**

Run:

```bash
ADB=~/Library/Android/sdk/platform-tools/adb
$ADB install -r android-visualizer/app/build/outputs/apk/debug/app-debug.apk
$ADB shell am start -n com.mobilefinetuner.visualizer/.MainActivity
$ADB shell uiautomator dump /sdcard/window.xml
$ADB exec-out cat /sdcard/window.xml | rg 'Generate|Prefer|Train|Evidence|Qwen3-0.6B|Reward Accuracy|Reward Margin|Train KTO'
```

Expected: all main demo labels are visible in the UI tree.

- [ ] **Step 5: Capture screenshot**

Run:

```bash
mkdir -p verification/ui
ADB=~/Library/Android/sdk/platform-tools/adb
$ADB exec-out screencap -p > verification/ui/paper_demo_console.png
```

Expected: screenshot file exists and is non-empty.

---

