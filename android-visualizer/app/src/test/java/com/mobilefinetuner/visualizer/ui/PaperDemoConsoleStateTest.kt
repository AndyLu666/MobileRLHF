package com.mobilefinetuner.visualizer.ui

import com.google.common.truth.Truth.assertThat
import com.mobilefinetuner.visualizer.model.NativeDemoConfig
import com.mobilefinetuner.visualizer.model.NativeDemoRunStatus
import com.mobilefinetuner.visualizer.model.NativeDemoRunnerState
import com.mobilefinetuner.visualizer.model.NativeFeedbackMode
import com.mobilefinetuner.visualizer.model.NativeInferenceState
import com.mobilefinetuner.visualizer.model.RunSnapshot
import com.mobilefinetuner.visualizer.model.RunStatus
import com.mobilefinetuner.visualizer.model.RunSummary
import com.mobilefinetuner.visualizer.model.StepMetric
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
    fun workflowMarksFeedbackSavedWhenSingleRatingsExist() {
        val steps = buildDemoWorkflowSteps(
            inference = NativeInferenceState(savedFeedbackRows = 2),
            runner = NativeDemoRunnerState(status = NativeDemoRunStatus.READY),
            snapshot = null
        )

        assertThat(steps[0].status).isEqualTo(DemoWorkflowStatus.DONE)
        assertThat(steps[1].status).isEqualTo(DemoWorkflowStatus.SAVED)
        assertThat(steps[2].status).isEqualTo(DemoWorkflowStatus.READY)
    }

    @Test
    fun feedbackModeStateNamesPairAndSingleModes() {
        val pair = buildFeedbackModeState(NativeInferenceState(feedbackMode = NativeFeedbackMode.PAIR))
        val single = buildFeedbackModeState(NativeInferenceState(feedbackMode = NativeFeedbackMode.SINGLE))

        assertThat(pair.selectedLabel).isEqualTo("A/B Compare")
        assertThat(pair.actionLabel).isEqualTo("Generate A/B")
        assertThat(pair.helpText).contains("choose the better answer")
        assertThat(single.selectedLabel).isEqualTo("Single Rating")
        assertThat(single.actionLabel).isEqualTo("Generate One")
        assertThat(single.helpText).contains("thumbs up or down")
    }

    @Test
    fun liveProgressFormatsCurrentStepFromSnapshot() {
        val progress = buildDemoLiveProgressState(
            snapshot = demoSnapshot(
                loss = 0.49,
                rewardAccuracy = 0.6,
                rewardMargin = 0.08
            )
        )

        assertThat(progress.stepLabel).isEqualTo("10 / 10")
        assertThat(progress.rewardAccuracy).isEqualTo("0.600")
        assertThat(progress.rewardMargin).isEqualTo("0.080")
        assertThat(progress.message).isEqualTo("Reward trend is visible.")
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
            config = NativeDemoConfig(steps = 10),
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

    private fun demoSnapshot(
        loss: Double,
        rewardAccuracy: Double,
        rewardMargin: Double
    ): RunSnapshot {
        return RunSnapshot(
            runId = "demo",
            runName = "phone_qwen_ab_kto_10s",
            status = RunStatus.COMPLETED,
            metrics = listOf(
                StepMetric(
                    step = 10,
                    totalSteps = 10,
                    loss = loss,
                    rewardAccuracy = rewardAccuracy,
                    rewardMargin = rewardMargin,
                    source = "android_native_kto"
                )
            ),
            rssPoints = emptyList(),
            events = emptyList(),
            logTail = emptyList(),
            summary = RunSummary(totalSteps = 10),
            updatedAtMs = 1234L
        )
    }
}
