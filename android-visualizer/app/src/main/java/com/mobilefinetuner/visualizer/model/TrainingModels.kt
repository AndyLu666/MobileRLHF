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
    RUNNER("Runner"),
    OVERVIEW("Overview"),
    TRAINING("Training"),
    COMPARISON("Comparison"),
    LOGS("Logs"),
    EXPERIMENTS("Experiments")
}

enum class RunnerStatus {
    IDLE,
    IMPORTING,
    READY,
    RUNNING,
    COMPLETED,
    FAILED,
    STOPPING
}

data class StepMetric(
    val step: Int,
    val totalSteps: Int? = null,
    val trainSamples: Int? = null,
    val feedbackPairs: Int? = null,
    val sampleCount: Int? = null,
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
    val steps: Int = 10,
    val seqLen: Int = 64,
    val rank: Int = 2,
    val alpha: Int = 4,
    val learningRate: Double = 0.001,
    val evalInterval: Int = 4,
    val evalBatches: Int = 1,
    val cacheRefLogps: Boolean = false,
    val dpoPairForward: Boolean = false,
    val globalAdapterPath: String? = null
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

enum class NativeFeedbackMode {
    PAIR,
    SINGLE
}

data class NativeInferenceState(
    val modelDir: String = "/data/local/tmp/mft_qwen3",
    val useMobileRLHFBridge: Boolean = false,
    val bridgeUrl: String = "http://10.0.2.2:8765",
    val prompt: String = "Wearable summary: 6,200 steps, 6h 10m sleep, normal resting heart rate, and slightly low energy. What is one safe activity recommendation for today?",
    val referenceDate: String = "2026-06-25",
    val maxNewTokens: Int = 72,
    val temperature: Double = 0.2,
    val topP: Double = 0.9,
    val modelLoaded: Boolean = false,
    val isRunning: Boolean = false,
    val output: String? = null,
    val candidateA: String? = null,
    val candidateB: String? = null,
    val elapsedMs: Double? = null,
    val loadElapsedMs: Double? = null,
    val generateElapsedMs: Double? = null,
    val feedbackMode: NativeFeedbackMode = NativeFeedbackMode.PAIR,
    val savedPairs: Int = 0,
    val savedFeedbackRows: Int = 0,
    val feedbackJsonlPath: String? = null,
    val lastChosenLabel: String? = null,
    val lastRatingLabel: String? = null,
    val feedbackStatus: String? = null,
    val debugLines: List<String> = listOf("Ready. Native KTO trains on the Android device."),
    val error: String? = null
)

data class LocalModelAsset(
    val id: String,
    val name: String,
    val path: String,
    val family: String,
    val hasWeights: Boolean,
    val sizeBytes: Long,
    val updatedAtMs: Long
)

data class RunnerStepResult(
    val step: Int,
    val loss: Float,
    val trainableTensorCount: Int,
    val validLabelCount: Int,
    val accumulationStep: Int,
    val gradientAccumulationSteps: Int,
    val optimizerStep: Boolean,
    val elapsedMs: Long
)

data class RunnerUiState(
    val status: RunnerStatus = RunnerStatus.IDLE,
    val models: List<LocalModelAsset> = emptyList(),
    val selectedModelId: String? = null,
    val loadWeights: Boolean = true,
    val sequenceLength: Int = 2,
    val batchSize: Int = 1,
    val gradientAccumulationSteps: Int = 1,
    val steps: Int = 1,
    val trainingText: String = "MobileFineTuner foreground training smoke.",
    val importingModelName: String? = null,
    val currentMessage: String = "Ready",
    val logs: List<String> = emptyList(),
    val results: List<RunnerStepResult> = emptyList(),
    val error: String? = null
)

data class DashboardUiState(
    val selectedRootUri: Uri? = null,
    val activeTab: DashboardTab = DashboardTab.EXPERIMENTS,
    val runHandles: List<RunHandle> = emptyList(),
    val selectedRunId: String? = null,
    val compareRunId: String? = null,
    val snapshot: RunSnapshot? = null,
    val compareSnapshot: RunSnapshot? = null,
    val feedbackSession: FeedbackSessionState? = null,
    val nativeDemoConfig: NativeDemoConfig = NativeDemoConfig(),
    val nativeDemoRunner: NativeDemoRunnerState = NativeDemoRunnerState(),
    val nativeInference: NativeInferenceState = NativeInferenceState(),
    val runner: RunnerUiState = RunnerUiState(),
    val isLoading: Boolean = false,
    val error: String? = null
)
