package com.mobilefinetuner.visualizer.model

import android.net.Uri

enum class RunStatus {
    IDLE,
    RUNNING,
    STOPPED,
    COMPLETED,
    FAILED
}

enum class EventType {
    INFO,
    EVAL,
    CHECKPOINT,
    CLEANUP,
    RSS,
    ENERGY,
    WARNING,
    ERROR
}

enum class DashboardTab(val title: String) {
    OVERVIEW("Overview"),
    TRAINING("Training"),
    COMPARISON("Comparison"),
    LOGS("Logs"),
    EXPERIMENTS("Experiments")
}

data class StepMetric(
    val step: Int,
    val totalSteps: Int? = null,
    val loss: Double? = null,
    val ppl: Double? = null,
    val lr: Double? = null,
    val tokens: Int? = null,
    val rewardAccuracy: Double? = null,
    val rewardMargin: Double? = null,
    val chosenReward: Double? = null,
    val rejectedReward: Double? = null,
    val gradNorm: Double? = null,
    val rssMb: Double? = null,
    val source: String = "log"
)

data class RssPoint(
    val index: Int,
    val rssMb: Double,
    val timeLabel: String? = null
)

data class RunEvent(
    val step: Int? = null,
    val type: EventType,
    val message: String,
    val raw: String
)

data class RunSummary(
    val maxInitRssMb: Double? = null,
    val maxTrainRssMb: Double? = null,
    val maxStepEndRssMb: Double? = null,
    val finalEmaLoss: Double? = null,
    val totalTokens: Long? = null,
    val totalSteps: Int? = null
)

data class InferenceRecord(
    val promptId: String? = null,
    val prompt: String,
    val completion: String? = null,
    val chosen: String? = null,
    val rejected: String? = null,
    val chosenSource: String? = null,
    val elapsedMs: Double? = null,
    val source: String? = null
)

data class RunSnapshot(
    val runId: String,
    val runName: String,
    val status: RunStatus,
    val metrics: List<StepMetric>,
    val rssPoints: List<RssPoint>,
    val events: List<RunEvent>,
    val inferenceRecords: List<InferenceRecord> = emptyList(),
    val logTail: List<String>,
    val summary: RunSummary,
    val updatedAtMs: Long
)

data class RunHandle(
    val id: String,
    val name: String,
    val dirUri: Uri,
    val trainLogUri: Uri,
    val rssCsvUri: Uri? = null,
    val metricsNdjsonUri: Uri? = null,
    val generatedResponsesUri: Uri? = null,
    val inferenceFeedbackUri: Uri? = null,
    val lastModifiedMs: Long
)

data class FeedbackPrompt(
    val prompt: String,
    val answerA: String,
    val answerB: String
)

data class FeedbackSessionState(
    val prompts: List<FeedbackPrompt>,
    val currentIndex: Int = 0,
    val chosenLabels: List<String> = emptyList(),
    val sessionId: String = "hfrl_${System.currentTimeMillis()}",
    val feedbackJsonlPath: String? = null
) {
    val isComplete: Boolean
        get() = currentIndex >= prompts.size

    val currentPrompt: FeedbackPrompt?
        get() = prompts.getOrNull(currentIndex)
}

enum class NativeDemoRunStatus {
    READY,
    RUNNING,
    STOPPING,
    STOPPED,
    COMPLETED,
    FAILED
}

data class NativeDemoConfig(
    val presetName: String = "Smoke",
    val steps: Int = 8,
    val seqLen: Int = 128,
    val rank: Int = 1,
    val alpha: Int = 2,
    val learningRate: Double = 0.001,
    val evalInterval: Int = 4,
    val evalBatches: Int = 1,
    val cacheRefLogps: Boolean = false,
    val dpoPairForward: Boolean = false
)

data class NativeDemoRunnerState(
    val status: NativeDemoRunStatus = NativeDemoRunStatus.READY,
    val currentRunName: String? = null,
    val currentRunPath: String? = null,
    val commandPreview: String? = null,
    val trainingSourceLabel: String = "HF fitness preference dataset",
    val trainingJsonlPath: String? = null,
    val trainingPairCount: Int = 0,
    val lastExitCode: Int? = null,
    val message: String? = null
)

data class NativeInferenceState(
    val modelDir: String = "/data/local/tmp/mft_qwen3",
    val prompt: String = "How can I maintain my fitness routine while traveling for work?",
    val maxNewTokens: Int = 16,
    val temperature: Double = 0.7,
    val topP: Double = 0.9,
    val modelLoaded: Boolean = false,
    val isRunning: Boolean = false,
    val output: String? = null,
    val candidateA: String? = null,
    val candidateB: String? = null,
    val elapsedMs: Double? = null,
    val loadElapsedMs: Double? = null,
    val generateElapsedMs: Double? = null,
    val savedPairs: Int = 0,
    val feedbackJsonlPath: String? = null,
    val lastChosenLabel: String? = null,
    val error: String? = null
)

data class DashboardUiState(
    val selectedRootUri: Uri? = null,
    val activeTab: DashboardTab = DashboardTab.OVERVIEW,
    val runHandles: List<RunHandle> = emptyList(),
    val selectedRunId: String? = null,
    val compareRunId: String? = null,
    val snapshot: RunSnapshot? = null,
    val compareSnapshot: RunSnapshot? = null,
    val feedbackSession: FeedbackSessionState? = null,
    val nativeDemoConfig: NativeDemoConfig = NativeDemoConfig(),
    val nativeDemoRunner: NativeDemoRunnerState = NativeDemoRunnerState(),
    val nativeInference: NativeInferenceState = NativeInferenceState(),
    val isLoading: Boolean = false,
    val error: String? = null
)
