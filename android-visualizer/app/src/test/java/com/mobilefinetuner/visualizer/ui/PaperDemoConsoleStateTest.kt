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

        assertThat(steps.map { it.label }).containsExactly("Data", "Train KTO", "Evidence").inOrder()
        assertThat(steps[0].status).isEqualTo(DemoWorkflowStatus.SAVED)
        assertThat(steps[1].status).isEqualTo(DemoWorkflowStatus.READY)
        assertThat(steps[2].status).isEqualTo(DemoWorkflowStatus.NEEDS_INPUT)
    }

    @Test
    fun workflowMarksFeedbackSavedWhenSingleRatingsExist() {
        val steps = buildDemoWorkflowSteps(
            inference = NativeInferenceState(savedFeedbackRows = 2),
            runner = NativeDemoRunnerState(status = NativeDemoRunStatus.READY),
            snapshot = null
        )

        assertThat(steps[0].status).isEqualTo(DemoWorkflowStatus.SAVED)
        assertThat(steps[1].status).isEqualTo(DemoWorkflowStatus.READY)
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
    fun displayFeedbackRowsUsesTrainingFileCountWhenPreloadedByAdb() {
        val rows = displayFeedbackRows(
            inference = NativeInferenceState(savedFeedbackRows = 0, savedPairs = 0),
            runner = NativeDemoRunnerState(trainingPairCount = 2)
        )

        assertThat(rows).isEqualTo(2)
    }

    @Test
    fun inputStateLabelsDpoPairsAsKtoReady() {
        val input = buildDemoInputState(
            inference = NativeInferenceState(savedPairs = 12),
            runner = NativeDemoRunnerState(trainingPairCount = 24)
        )

        assertThat(input.title).isEqualTo("1. Data")
        assertThat(input.formatLabel).isEqualTo("DPO pairs -> KTO")
        assertThat(input.formatLabel.length).isAtMost(16)
        assertThat(input.rowLabel).isEqualTo("24 KTO rows")
        assertThat(input.statusLabel).isEqualTo("Ready")
    }

    @Test
    fun inputStateUsesShortLabelForPreloadedKtoRows() {
        val input = buildDemoInputState(
            inference = NativeInferenceState(),
            runner = NativeDemoRunnerState(trainingPairCount = 24)
        )

        assertThat(input.formatLabel).isEqualTo("KTO/DPO rows")
        assertThat(input.formatLabel.length).isAtMost(14)
    }

    @Test
    fun paperSummaryHighlightsKtoTrainingAndPeakMemory() {
        val summary = buildPaperDemoSummaryState(
            config = NativeDemoConfig(steps = 10),
            runner = NativeDemoRunnerState(
                status = NativeDemoRunStatus.COMPLETED,
                trainingPairCount = 24,
                trainingSourceLabel = "Qwen feedback native KTO"
            ),
            snapshot = demoSnapshot(
                loss = 0.3949413,
                rewardAccuracy = 1.0,
                rewardMargin = 0.88533133,
                rssMb = 5289.6328125
            )
        )

        assertThat(summary.title).isEqualTo("Qwen3-0.6B On-device KTO")
        assertThat(summary.dataStatus).isEqualTo("DPO/KTO ready")
        assertThat(summary.dataStatus.length).isAtMost(14)
        assertThat(summary.trainingStatus).isEqualTo("KTO completed")
        assertThat(summary.trainingStatus.length).isAtMost(16)
        assertThat(summary.peakMemory).isEqualTo("5289.6 MB")
        assertThat(summary.readinessLabel).isEqualTo("Demo ready")
        assertThat(summary.readinessDetail).isEqualTo("DPO/KTO data, Android KTO training, and live evidence are all available.")
    }

    @Test
    fun paperSummaryExplainsMissingDataBeforeTraining() {
        val summary = buildPaperDemoSummaryState(
            config = NativeDemoConfig(steps = 10),
            runner = NativeDemoRunnerState(
                status = NativeDemoRunStatus.READY,
                trainingPairCount = 0
            ),
            snapshot = null
        )

        assertThat(summary.readinessLabel).isEqualTo("Need data")
        assertThat(summary.readinessDetail).isEqualTo("Load DPO pairs, KTO binary rows, or collect thumbs feedback first.")
    }

    @Test
    fun runLabelsDoNotClaimNoRunWhenDemoDataIsLoaded() {
        val labels = buildPaperRunLabelState(
            selectedRunName = null,
            compareRunName = null,
            runner = NativeDemoRunnerState(trainingPairCount = 24)
        )

        assertThat(labels.selectedRun).isEqualTo("Demo data loaded")
        assertThat(labels.compareRun).isEqualTo("Optional")
    }

    @Test
    fun primaryActionStartsKtoWhenDataIsReady() {
        val action = buildDemoPrimaryActionState(
            runner = NativeDemoRunnerState(
                status = NativeDemoRunStatus.READY,
                trainingPairCount = 24
            )
        )

        assertThat(action.label).isEqualTo("Start KTO Training")
        assertThat(action.kind).isEqualTo(DemoPrimaryActionKind.START)
        assertThat(action.enabled).isTrue()
    }

    @Test
    fun primaryActionStopsActiveTraining() {
        val action = buildDemoPrimaryActionState(
            runner = NativeDemoRunnerState(
                status = NativeDemoRunStatus.RUNNING,
                trainingPairCount = 24
            )
        )

        assertThat(action.label).isEqualTo("Stop Training")
        assertThat(action.kind).isEqualTo(DemoPrimaryActionKind.STOP)
        assertThat(action.enabled).isTrue()
    }

    @Test
    fun chartPreviewProvidesSampleCurvesBeforeTraining() {
        val preview = buildDemoChartPreviewState(snapshot = null)

        assertThat(preview.title).isEqualTo("Training Preview")
        assertThat(preview.isLive).isFalse()
        assertThat(preview.lossValues).hasSize(5)
        assertThat(preview.rewardMarginValues).hasSize(5)
        assertThat(preview.lossValues.first()).isGreaterThan(preview.lossValues.last())
        assertThat(preview.rewardMarginValues.last()).isGreaterThan(preview.rewardMarginValues.first())
    }

    @Test
    fun chartPreviewUsesLiveMetricsWhenAvailable() {
        val preview = buildDemoChartPreviewState(
            snapshot = RunSnapshot(
                runId = "demo",
                runName = "phone_qwen_kto",
                status = RunStatus.RUNNING,
                metrics = listOf(
                    StepMetric(step = 1, loss = 0.62, rewardMargin = 0.1, source = "android_native_kto"),
                    StepMetric(step = 2, loss = 0.51, rewardMargin = 0.3, source = "android_native_kto")
                ),
                rssPoints = emptyList(),
                events = emptyList(),
                logTail = emptyList(),
                summary = RunSummary(totalSteps = 10),
                updatedAtMs = 1L
            )
        )

        assertThat(preview.title).isEqualTo("Live Training Charts")
        assertThat(preview.isLive).isTrue()
        assertThat(preview.lossValues).containsExactly(0.62, 0.51).inOrder()
        assertThat(preview.rewardMarginValues).containsExactly(0.1, 0.3).inOrder()
    }

    @Test
    fun paperModeKeepsDebugSectionsOutOfDefaultView() {
        val sections = buildPaperModeSections(showAdvanced = false)

        assertThat(sections).containsExactly(
            PaperModeSection.HERO,
            PaperModeSection.WORKFLOW,
            PaperModeSection.ADVANCED_TOGGLE
        ).inOrder()
    }

    @Test
    fun paperModeShowsDebugSectionsOnlyWhenExpanded() {
        val sections = buildPaperModeSections(showAdvanced = true)

        assertThat(sections).containsExactly(
            PaperModeSection.HERO,
            PaperModeSection.WORKFLOW,
            PaperModeSection.ADVANCED_TOGGLE,
            PaperModeSection.DATA_CONTROLS,
            PaperModeSection.TRAINING_CONTROLS,
            PaperModeSection.LIVE_EVIDENCE,
            PaperModeSection.PAPER_EVIDENCE
        ).inOrder()
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
    fun liveProgressFormatsMissingMetricsAsWaiting() {
        val progress = buildDemoLiveProgressState(snapshot = null)

        assertThat(progress.stepLabel).isEqualTo("Waiting")
        assertThat(progress.rewardAccuracy).isEqualTo("N/A")
        assertThat(progress.rewardMargin).isEqualTo("N/A")
        assertThat(progress.loss).isEqualTo("N/A")
        assertThat(progress.peakMemory).isEqualTo("N/A")
        assertThat(progress.message).isEqualTo("Waiting for metrics.")
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

        assertThat(steps[1].status).isEqualTo(DemoWorkflowStatus.DONE)
        assertThat(steps[2].status).isEqualTo(DemoWorkflowStatus.DONE)
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
        assertThat(evidence.artifacts).isEqualTo("report.json, metrics.ndjson, adapter.safetensors")
        assertThat(evidence.artifacts).doesNotContain("before_after")
        assertThat(formatDemoMetric(null)).isEqualTo("N/A")
    }

    private fun demoSnapshot(
        loss: Double,
        rewardAccuracy: Double,
        rewardMargin: Double,
        rssMb: Double? = null
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
                    rssMb = rssMb,
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
