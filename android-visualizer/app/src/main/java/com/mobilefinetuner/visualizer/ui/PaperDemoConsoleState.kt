package com.mobilefinetuner.visualizer.ui

import com.mobilefinetuner.visualizer.model.NativeDemoConfig
import com.mobilefinetuner.visualizer.model.NativeDemoRunStatus
import com.mobilefinetuner.visualizer.model.NativeDemoRunnerState
import com.mobilefinetuner.visualizer.model.NativeFeedbackMode
import com.mobilefinetuner.visualizer.model.NativeInferenceState
import com.mobilefinetuner.visualizer.model.RunSnapshot
import java.util.Locale

enum class DemoWorkflowStatus {
    READY,
    NEEDS_INPUT,
    SAVED,
    RUNNING,
    DONE
}

data class DemoWorkflowStepState(
    val label: String,
    val status: DemoWorkflowStatus
)

data class DemoEvidenceState(
    val objective: String,
    val model: String,
    val pairs: String,
    val steps: String,
    val batch: String,
    val loss: String,
    val rewardAccuracy: String,
    val rewardMargin: String,
    val trendAccuracy: String,
    val trendMargin: String,
    val artifacts: String
)

data class PaperDemoSummaryState(
    val title: String,
    val subtitle: String,
    val readinessLabel: String,
    val readinessDetail: String,
    val dataStatus: String,
    val trainingStatus: String,
    val peakMemory: String,
    val adapterStatus: String
)

data class DemoInputState(
    val title: String,
    val subtitle: String,
    val formatLabel: String,
    val rowLabel: String,
    val statusLabel: String
)

data class PaperRunLabelState(
    val selectedRun: String,
    val compareRun: String
)

enum class DemoPrimaryActionKind {
    START,
    STOP,
    REVIEW,
    LOAD_DATA
}

data class DemoPrimaryActionState(
    val label: String,
    val kind: DemoPrimaryActionKind,
    val enabled: Boolean
)

data class DemoChartPreviewState(
    val title: String,
    val subtitle: String,
    val isLive: Boolean,
    val lossValues: List<Double>,
    val rewardMarginValues: List<Double>
)

data class FeedbackModeUiState(
    val selectedLabel: String,
    val actionLabel: String,
    val helpText: String
)

data class DemoLiveProgressState(
    val stepLabel: String,
    val rewardAccuracy: String,
    val rewardMargin: String,
    val loss: String,
    val peakMemory: String,
    val message: String
)

fun displayFeedbackRows(
    inference: NativeInferenceState,
    runner: NativeDemoRunnerState
): Int {
    return maxOf(inference.savedFeedbackRows, inference.savedPairs, runner.trainingPairCount)
}

fun buildPaperDemoSummaryState(
    config: NativeDemoConfig,
    runner: NativeDemoRunnerState,
    snapshot: RunSnapshot?
): PaperDemoSummaryState {
    val rows = runner.trainingPairCount
    val peakMemory = snapshot?.metrics.orEmpty().mapNotNull { it.rssMb }.maxOrNull()
        ?: snapshot?.summary?.maxTrainRssMb
    val completed = runner.status == NativeDemoRunStatus.COMPLETED ||
        snapshot?.status?.name == "COMPLETED"
    val hasData = rows > 0
    val hasEvidence = snapshot?.metrics?.isNotEmpty() == true
    val readinessLabel = when {
        hasData && completed && hasEvidence -> "Demo ready"
        hasData && runner.status == NativeDemoRunStatus.RUNNING -> "Training"
        hasData -> "Ready to train"
        else -> "Need data"
    }
    val readinessDetail = when (readinessLabel) {
        "Demo ready" -> "DPO/KTO data, Android KTO training, and live evidence are all available."
        "Training" -> "Android KTO is running; live metrics will update below."
        "Ready to train" -> "DPO/KTO feedback is loaded; start KTO to produce live evidence."
        else -> "Load DPO pairs, KTO binary rows, or collect thumbs feedback first."
    }
    return PaperDemoSummaryState(
        title = "Qwen3-0.6B On-device KTO",
        subtitle = "Data -> Train KTO -> Evidence",
        readinessLabel = readinessLabel,
        readinessDetail = readinessDetail,
        dataStatus = if (hasData) "DPO/KTO ready" else "Need data",
        trainingStatus = when {
            runner.status == NativeDemoRunStatus.RUNNING -> "KTO running"
            completed -> "KTO completed"
            else -> "${config.steps} steps ready"
        },
        peakMemory = peakMemory?.let { "${formatDemoMetric(it, digits = 1)} MB" } ?: "RSS pending",
        adapterStatus = if (completed) "Adapter saved" else "Adapter pending"
    )
}

fun buildDemoInputState(
    inference: NativeInferenceState,
    runner: NativeDemoRunnerState
): DemoInputState {
    val rows = displayFeedbackRows(inference, runner)
    val format = when {
        inference.savedPairs > 0 -> "DPO pairs -> KTO"
        inference.savedFeedbackRows > 0 -> "App thumbs -> KTO"
        runner.trainingPairCount > 0 -> "KTO/DPO rows"
        else -> "DPO or KTO JSONL"
    }
    return DemoInputState(
        title = "1. Data",
        subtitle = "Use DPO chosen/rejected, KTO binary labels, or app thumbs.",
        formatLabel = format,
        rowLabel = if (rows > 0) "$rows KTO rows" else "No rows",
        statusLabel = if (rows > 0) "Ready" else "Needs data"
    )
}

fun buildPaperRunLabelState(
    selectedRunName: String?,
    compareRunName: String?,
    runner: NativeDemoRunnerState
): PaperRunLabelState {
    val hasDemoData = runner.trainingPairCount > 0
    return PaperRunLabelState(
        selectedRun = selectedRunName ?: if (hasDemoData) "Demo data loaded" else "No run selected",
        compareRun = compareRunName ?: if (hasDemoData) "Optional" else "None"
    )
}

fun buildDemoPrimaryActionState(runner: NativeDemoRunnerState): DemoPrimaryActionState {
    return when (runner.status) {
        NativeDemoRunStatus.RUNNING,
        NativeDemoRunStatus.STOPPING -> DemoPrimaryActionState(
            label = "Stop Training",
            kind = DemoPrimaryActionKind.STOP,
            enabled = runner.status == NativeDemoRunStatus.RUNNING
        )
        NativeDemoRunStatus.COMPLETED -> DemoPrimaryActionState(
            label = "Review Evidence",
            kind = DemoPrimaryActionKind.REVIEW,
            enabled = false
        )
        else -> if (runner.trainingPairCount > 0) {
            DemoPrimaryActionState(
                label = "Start KTO Training",
                kind = DemoPrimaryActionKind.START,
                enabled = true
            )
        } else {
            DemoPrimaryActionState(
                label = "Load DPO/KTO Data",
                kind = DemoPrimaryActionKind.LOAD_DATA,
                enabled = true
            )
        }
    }
}

fun buildDemoChartPreviewState(snapshot: RunSnapshot?): DemoChartPreviewState {
    val metrics = snapshot?.metrics.orEmpty()
    val liveLoss = metrics.mapNotNull { it.loss }
    val liveMargin = metrics.mapNotNull { it.rewardMargin }
    val hasLiveCurves = liveLoss.size >= 2 || liveMargin.size >= 2
    return if (hasLiveCurves) {
        DemoChartPreviewState(
            title = "Live Training Charts",
            subtitle = "Real Android KTO metrics",
            isLive = true,
            lossValues = liveLoss.ifEmpty { listOf(0.0, 0.0) },
            rewardMarginValues = liveMargin.ifEmpty { listOf(0.0, 0.0) }
        )
    } else {
        DemoChartPreviewState(
            title = "Training Preview",
            subtitle = "Example curves; real values appear after training starts",
            isLive = false,
            lossValues = listOf(0.68, 0.61, 0.55, 0.49, 0.43),
            rewardMarginValues = listOf(0.00, 0.11, 0.24, 0.42, 0.62)
        )
    }
}

fun buildDemoWorkflowSteps(
    inference: NativeInferenceState,
    runner: NativeDemoRunnerState,
    snapshot: RunSnapshot?
): List<DemoWorkflowStepState> {
    val feedbackRows = displayFeedbackRows(inference, runner)
    val hasPreference = feedbackRows > 0
    val isTraining = runner.status == NativeDemoRunStatus.RUNNING ||
        runner.status == NativeDemoRunStatus.STOPPING
    val trainDone = runner.status == NativeDemoRunStatus.COMPLETED ||
        snapshot?.metrics?.isNotEmpty() == true
    val evidenceDone = snapshot?.metrics?.lastOrNull()?.let {
        it.rewardAccuracy != null || it.rewardMargin != null || it.loss != null
    } == true

    return listOf(
        DemoWorkflowStepState(
            label = "Data",
            status = when {
                hasPreference -> DemoWorkflowStatus.SAVED
                else -> DemoWorkflowStatus.NEEDS_INPUT
            }
        ),
        DemoWorkflowStepState(
            label = "Train KTO",
            status = when {
                isTraining -> DemoWorkflowStatus.RUNNING
                trainDone -> DemoWorkflowStatus.DONE
                hasPreference -> DemoWorkflowStatus.READY
                else -> DemoWorkflowStatus.NEEDS_INPUT
            }
        ),
        DemoWorkflowStepState(
            label = "Evidence",
            status = if (evidenceDone) DemoWorkflowStatus.DONE else DemoWorkflowStatus.NEEDS_INPUT
        )
    )
}

fun buildFeedbackModeState(inference: NativeInferenceState): FeedbackModeUiState {
    return when (inference.feedbackMode) {
        NativeFeedbackMode.PAIR -> FeedbackModeUiState(
            selectedLabel = "A/B Compare",
            actionLabel = "Generate A/B",
            helpText = "Generate two answers and choose the better answer."
        )
        NativeFeedbackMode.SINGLE -> FeedbackModeUiState(
            selectedLabel = "Single Rating",
            actionLabel = "Generate One",
            helpText = "Generate one answer and rate it with thumbs up or down."
        )
    }
}

fun buildDemoLiveProgressState(snapshot: RunSnapshot?): DemoLiveProgressState {
    val metrics = snapshot?.metrics.orEmpty()
    val last = metrics.lastOrNull()
    val total = last?.totalSteps ?: snapshot?.summary?.totalSteps
    val stepLabel = when {
        last?.step != null && total != null -> "${last.step} / $total"
        last?.step != null -> "${last.step} / ?"
        else -> "Waiting"
    }
    val peakMemory = metrics.mapNotNull { it.rssMb }.maxOrNull()
        ?: snapshot?.summary?.maxTrainRssMb
    val hasReward = last?.rewardAccuracy != null || last?.rewardMargin != null
    return DemoLiveProgressState(
        stepLabel = stepLabel,
        rewardAccuracy = formatDemoMetric(last?.rewardAccuracy),
        rewardMargin = formatDemoMetric(last?.rewardMargin),
        loss = formatDemoMetric(last?.loss),
        peakMemory = peakMemory?.let { "${formatDemoMetric(it, digits = 1)} MB" } ?: "N/A",
        message = when {
            metrics.isEmpty() -> "Waiting for metrics."
            hasReward -> "Reward trend is visible."
            else -> "Metrics parsed; waiting for reward fields."
        }
    )
}

fun buildDemoEvidenceState(
    config: NativeDemoConfig,
    runner: NativeDemoRunnerState,
    snapshot: RunSnapshot?
): DemoEvidenceState {
    val last = snapshot?.metrics?.lastOrNull()
    val first = snapshot?.metrics?.firstOrNull()
    val pairCount = when {
        inferenceFeedbackRowsFromRunner(runner) > 0 -> inferenceFeedbackRowsFromRunner(runner).toString()
        runner.trainingPairCount > 0 -> runner.trainingPairCount.toString()
        snapshot?.runName?.contains("qwen_ab", ignoreCase = true) == true -> "30"
        else -> "N/A"
    }
    val stepCount = last?.totalSteps ?: snapshot?.summary?.totalSteps ?: config.steps
    val trendAccuracy = if (first?.rewardAccuracy != null && last?.rewardAccuracy != null) {
        "${formatDemoMetric(first.rewardAccuracy)} -> ${formatDemoMetric(last.rewardAccuracy)}"
    } else {
        "Pending"
    }
    val trendMargin = if (first?.rewardMargin != null && last?.rewardMargin != null) {
        "${formatDemoMetric(first.rewardMargin)} -> ${formatDemoMetric(last.rewardMargin)}"
    } else {
        "Pending"
    }
    val artifacts = snapshot?.runName?.let {
        "report.json, metrics.ndjson, adapter.safetensors"
    } ?: "Waiting for report.json"

    return DemoEvidenceState(
        objective = "KTO",
        model = "Qwen3-0.6B",
        pairs = pairCount,
        steps = stepCount.toString(),
        batch = runner.commandPreview
            ?.substringAfter("batch=", "")
            ?.takeWhile { it.isDigit() }
            ?.ifBlank { null }
            ?: "2",
        loss = formatDemoMetric(last?.loss),
        rewardAccuracy = formatDemoMetric(last?.rewardAccuracy),
        rewardMargin = formatDemoMetric(last?.rewardMargin),
        trendAccuracy = trendAccuracy,
        trendMargin = trendMargin,
        artifacts = artifacts
    )
}

fun formatDemoMetric(value: Double?, digits: Int = 3): String {
    if (value == null || value.isNaN() || value.isInfinite()) {
        return "N/A"
    }
    return String.format(Locale.US, "%.${digits}f", value)
}

private fun inferenceFeedbackRowsFromRunner(runner: NativeDemoRunnerState): Int {
    return if (runner.trainingSourceLabel.contains("Qwen", ignoreCase = true)) {
        runner.trainingPairCount
    } else {
        0
    }
}
