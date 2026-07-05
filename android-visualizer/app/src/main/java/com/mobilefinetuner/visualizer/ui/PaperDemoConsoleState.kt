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

fun buildDemoWorkflowSteps(
    inference: NativeInferenceState,
    runner: NativeDemoRunnerState,
    snapshot: RunSnapshot?
): List<DemoWorkflowStepState> {
    val hasGenerated = inference.candidateA != null ||
        inference.candidateB != null ||
        inference.output != null ||
        inference.savedPairs > 0 ||
        inference.savedFeedbackRows > 0
    val hasPreference = inference.savedFeedbackRows > 0 ||
        inference.savedPairs > 0 ||
        runner.trainingPairCount > 0
    val isTraining = runner.status == NativeDemoRunStatus.RUNNING ||
        runner.status == NativeDemoRunStatus.STOPPING
    val trainDone = runner.status == NativeDemoRunStatus.COMPLETED ||
        snapshot?.metrics?.isNotEmpty() == true
    val evidenceDone = snapshot?.metrics?.lastOrNull()?.let {
        it.rewardAccuracy != null || it.rewardMargin != null || it.loss != null
    } == true

    return listOf(
        DemoWorkflowStepState(
            label = "Generate",
            status = when {
                inference.isRunning -> DemoWorkflowStatus.RUNNING
                hasGenerated -> DemoWorkflowStatus.DONE
                else -> DemoWorkflowStatus.READY
            }
        ),
        DemoWorkflowStepState(
            label = "Prefer",
            status = when {
                hasPreference -> DemoWorkflowStatus.SAVED
                hasGenerated -> DemoWorkflowStatus.READY
                else -> DemoWorkflowStatus.NEEDS_INPUT
            }
        ),
        DemoWorkflowStepState(
            label = "Train",
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
        "report.json, metrics.ndjson, before_after_eval.json"
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
