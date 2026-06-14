package com.mobilefinetuner.visualizer.viewmodel

import android.app.Application
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.mobilefinetuner.sdk.MobileFineTuner
import com.mobilefinetuner.visualizer.data.FeedbackJsonl
import com.mobilefinetuner.visualizer.data.NativeDemoCommandBuilder
import com.mobilefinetuner.visualizer.data.TrainingLogParser
import com.mobilefinetuner.visualizer.model.DashboardTab
import com.mobilefinetuner.visualizer.model.DashboardUiState
import com.mobilefinetuner.visualizer.model.EventType
import com.mobilefinetuner.visualizer.model.FeedbackPrompt
import com.mobilefinetuner.visualizer.model.FeedbackSessionState
import com.mobilefinetuner.visualizer.model.InferenceRecord
import com.mobilefinetuner.visualizer.model.NativeDemoConfig
import com.mobilefinetuner.visualizer.model.NativeDemoRunStatus
import com.mobilefinetuner.visualizer.model.NativeDemoRunnerState
import com.mobilefinetuner.visualizer.model.NativeInferenceState
import com.mobilefinetuner.visualizer.model.RssPoint
import com.mobilefinetuner.visualizer.model.RunEvent
import com.mobilefinetuner.visualizer.model.RunHandle
import com.mobilefinetuner.visualizer.model.RunSnapshot
import com.mobilefinetuner.visualizer.model.RunStatus
import com.mobilefinetuner.visualizer.model.RunSummary
import com.mobilefinetuner.visualizer.model.StepMetric
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

class ExperimentViewModel(application: Application) : AndroidViewModel(application) {

    private val resolver = application.contentResolver
    private val prefs = application.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val _uiState = MutableStateFlow(DashboardUiState())
    val uiState: StateFlow<DashboardUiState> = _uiState.asStateFlow()

    private var pollingJob: Job? = null
    private var nativeDemoJob: Job? = null
    private var nativeInferenceJob: Job? = null
    private var nativeInferenceSession: MobileFineTuner? = null
    private var nativeInferenceModelDir: String? = null
    private var nativeDemoProcess: Process? = null
    private var nativeDemoRunDir: File? = null
    private var nativeDemoStopRequested: Boolean = false
    private var demoCursor: Int = DEMO_START_CURSOR
    private var dpoCursor: Int = DPO_START_CURSOR

    init {
        val rawUri = prefs.getString(KEY_LAST_ROOT_URI, null)
        if (!rawUri.isNullOrBlank()) {
            val uri = Uri.parse(rawUri)
            _uiState.value = _uiState.value.copy(selectedRootUri = uri)
            scanRuns(uri)
        }
        refreshNativeDemoCommandPreview()
    }

    fun selectTab(tab: DashboardTab) {
        _uiState.value = _uiState.value.copy(activeTab = tab)
    }

    fun onDirectoryPicked(uri: Uri) {
        try {
            resolver.takePersistableUriPermission(
                uri,
                Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        } catch (_: SecurityException) {
            // Some providers cannot persist permissions; still try to read immediately.
        }

        prefs.edit().putString(KEY_LAST_ROOT_URI, uri.toString()).apply()
        _uiState.value = _uiState.value.copy(
            selectedRootUri = uri,
            error = null,
            isLoading = true,
            runHandles = emptyList(),
            selectedRunId = null,
            compareRunId = null,
            snapshot = null,
            compareSnapshot = null,
            feedbackSession = null
        )
        scanRuns(uri)
    }

    fun refreshNow() {
        val root = _uiState.value.selectedRootUri ?: return
        scanRuns(root)
    }

    fun enableDemoMode() {
        val demoRuns = listOf(
            RunHandle(
                id = DEMO_PRIMARY_ID,
                name = "demo/primary_gpt2",
                dirUri = Uri.EMPTY,
                trainLogUri = Uri.EMPTY,
                rssCsvUri = null,
                metricsNdjsonUri = null,
                lastModifiedMs = System.currentTimeMillis()
            ),
            RunHandle(
                id = DEMO_COMPARE_ID,
                name = "demo/compare_gemma",
                dirUri = Uri.EMPTY,
                trainLogUri = Uri.EMPTY,
                rssCsvUri = null,
                metricsNdjsonUri = null,
                lastModifiedMs = System.currentTimeMillis()
            )
        )
        demoCursor = DEMO_START_CURSOR
        _uiState.value = _uiState.value.copy(
            activeTab = DashboardTab.OVERVIEW,
            runHandles = demoRuns,
            selectedRunId = DEMO_PRIMARY_ID,
            compareRunId = DEMO_COMPARE_ID,
            snapshot = null,
            compareSnapshot = null,
            feedbackSession = null,
            isLoading = false,
            error = null
        )
        startPolling()
    }

    fun enableDpoTestMode() {
        pollingJob?.cancel()
        dpoCursor = DPO_START_CURSOR
        _uiState.value = _uiState.value.copy(
            activeTab = DashboardTab.OVERVIEW,
            runHandles = emptyList(),
            selectedRunId = null,
            compareRunId = null,
            snapshot = null,
            compareSnapshot = null,
            feedbackSession = FeedbackSessionState(
                prompts = DPO_FEEDBACK_PROMPTS,
                feedbackJsonlPath = feedbackJsonlFile().absolutePath
            ),
            isLoading = false,
            error = null
        )
    }

    fun applyNativeDemoPreset(presetName: String) {
        val preset = when (presetName) {
            "Phone 80" -> NativeDemoConfig(
                presetName = "Phone 80",
                steps = 80,
                seqLen = 128,
                rank = 1,
                alpha = 2,
                learningRate = 0.001,
                evalInterval = 20,
                evalBatches = 1,
                cacheRefLogps = false,
                dpoPairForward = false
            )
            "Long 220" -> NativeDemoConfig(
                presetName = "Long 220",
                steps = 220,
                seqLen = 128,
                rank = 1,
                alpha = 2,
                learningRate = 0.001,
                evalInterval = 55,
                evalBatches = 1,
                cacheRefLogps = false,
                dpoPairForward = false
            )
            else -> NativeDemoConfig()
        }
        _uiState.value = _uiState.value.copy(
            nativeDemoConfig = preset,
            nativeDemoRunner = _uiState.value.nativeDemoRunner.copy(
                commandPreview = buildNativeDemoPreview(preset),
                message = "Preset loaded: ${preset.presetName}"
            )
        )
    }

    fun updateNativeDemoConfig(config: NativeDemoConfig) {
        val sanitized = config.copy(
            steps = config.steps.coerceIn(1, 1000),
            seqLen = config.seqLen.coerceIn(32, 512),
            rank = config.rank.coerceIn(1, 32),
            alpha = config.alpha.coerceIn(1, 64),
            evalInterval = config.evalInterval.coerceIn(0, 1000),
            evalBatches = config.evalBatches.coerceIn(0, 64)
        )
        _uiState.value = _uiState.value.copy(
            nativeDemoConfig = sanitized,
            nativeDemoRunner = _uiState.value.nativeDemoRunner.copy(
                commandPreview = buildNativeDemoPreview(sanitized),
                trainingSourceLabel = currentNativeDemoTrainingSourceLabel(),
                trainingJsonlPath = currentNativeDemoTrainingJsonlPath(),
                trainingPairCount = countJsonlLines(currentNativeDemoTrainingJsonlFile())
            )
        )
    }

    fun updateNativeInference(state: NativeInferenceState) {
        val modelDir = state.modelDir.trim()
        if (modelDir != nativeInferenceModelDir && modelDir != _uiState.value.nativeInference.modelDir.trim()) {
            closeNativeInferenceSession()
        }
        val sanitized = state.copy(
            modelDir = modelDir,
            prompt = state.prompt,
            maxNewTokens = state.maxNewTokens.coerceIn(1, 256),
            temperature = state.temperature.coerceIn(0.0, 2.0),
            topP = state.topP.coerceIn(0.05, 1.0),
            modelLoaded = modelDir == nativeInferenceModelDir && nativeInferenceSession != null,
            error = null
        )
        _uiState.value = _uiState.value.copy(nativeInference = sanitized)
    }

    fun runNativeInference() {
        if (nativeInferenceJob?.isActive == true) return
        val request = _uiState.value.nativeInference
        if (request.modelDir.isBlank()) {
            _uiState.value = _uiState.value.copy(
                nativeInference = request.copy(error = "Model directory is required.")
            )
            return
        }
        if (request.prompt.isBlank()) {
            _uiState.value = _uiState.value.copy(
                nativeInference = request.copy(error = "Prompt is required.")
            )
            return
        }

        _uiState.value = _uiState.value.copy(
            nativeInference = request.copy(
                isRunning = true,
                output = null,
                elapsedMs = null,
                loadElapsedMs = null,
                generateElapsedMs = null,
                error = null
            )
        )

        nativeInferenceJob = viewModelScope.launch(Dispatchers.IO) {
            val startNs = System.nanoTime()
            var loadMs = 0.0
            var generateMs = 0.0
            val result = runCatching {
                val session = if (nativeInferenceSession != null && nativeInferenceModelDir == request.modelDir) {
                    nativeInferenceSession!!
                } else {
                    closeNativeInferenceSession()
                    val loadStartNs = System.nanoTime()
                    MobileFineTuner.open(request.modelDir).also { tuner ->
                        loadMs = (System.nanoTime() - loadStartNs) / 1_000_000.0
                        nativeInferenceSession = tuner
                        nativeInferenceModelDir = request.modelDir
                    }
                }
                val generateStartNs = System.nanoTime()
                session.generate(
                    request.prompt,
                    request.maxNewTokens,
                    request.temperature.toFloat(),
                    request.topP.toFloat(),
                    42L
                ).also {
                    generateMs = (System.nanoTime() - generateStartNs) / 1_000_000.0
                }
            }
            val elapsedMs = (System.nanoTime() - startNs) / 1_000_000.0
            val current = _uiState.value.nativeInference
            val failure = result.exceptionOrNull()
            if (failure != null) {
                closeNativeInferenceSession()
            }
            _uiState.value = _uiState.value.copy(
                nativeInference = current.copy(
                    isRunning = false,
                    modelLoaded = nativeInferenceSession != null && nativeInferenceModelDir == request.modelDir,
                    output = result.getOrNull(),
                    elapsedMs = elapsedMs.takeIf { result.isSuccess },
                    loadElapsedMs = loadMs.takeIf { result.isSuccess },
                    generateElapsedMs = generateMs.takeIf { result.isSuccess },
                    error = failure?.message ?: failure?.javaClass?.simpleName
                )
            )
        }
    }

    fun runNativeInferencePair() {
        if (nativeInferenceJob?.isActive == true) return
        val request = _uiState.value.nativeInference
        if (request.modelDir.isBlank()) {
            _uiState.value = _uiState.value.copy(
                nativeInference = request.copy(error = "Model directory is required.")
            )
            return
        }
        if (request.prompt.isBlank()) {
            _uiState.value = _uiState.value.copy(
                nativeInference = request.copy(error = "Prompt is required.")
            )
            return
        }

        _uiState.value = _uiState.value.copy(
            nativeInference = request.copy(
                isRunning = true,
                output = null,
                candidateA = null,
                candidateB = null,
                elapsedMs = null,
                loadElapsedMs = null,
                generateElapsedMs = null,
                lastChosenLabel = null,
                error = null
            )
        )

        nativeInferenceJob = viewModelScope.launch(Dispatchers.IO) {
            val startNs = System.nanoTime()
            var loadMs = 0.0
            var generateMs = 0.0
            val result = runCatching {
                val session = if (nativeInferenceSession != null && nativeInferenceModelDir == request.modelDir) {
                    nativeInferenceSession!!
                } else {
                    closeNativeInferenceSession()
                    val loadStartNs = System.nanoTime()
                    MobileFineTuner.open(request.modelDir).also { tuner ->
                        loadMs = (System.nanoTime() - loadStartNs) / 1_000_000.0
                        nativeInferenceSession = tuner
                        nativeInferenceModelDir = request.modelDir
                    }
                }
                val generateStartNs = System.nanoTime()
                val a = session.generate(
                    request.prompt,
                    request.maxNewTokens,
                    request.temperature.toFloat(),
                    request.topP.toFloat(),
                    System.currentTimeMillis()
                )
                val b = session.generate(
                    request.prompt,
                    request.maxNewTokens,
                    request.temperature.toFloat(),
                    request.topP.toFloat(),
                    System.currentTimeMillis() + 104729L
                )
                generateMs = (System.nanoTime() - generateStartNs) / 1_000_000.0
                a to b
            }
            val elapsedMs = (System.nanoTime() - startNs) / 1_000_000.0
            val current = _uiState.value.nativeInference
            val failure = result.exceptionOrNull()
            if (failure != null) {
                closeNativeInferenceSession()
            }
            val pair = result.getOrNull()
            _uiState.value = _uiState.value.copy(
                nativeInference = current.copy(
                    isRunning = false,
                    modelLoaded = nativeInferenceSession != null && nativeInferenceModelDir == request.modelDir,
                    output = null,
                    candidateA = pair?.first,
                    candidateB = pair?.second,
                    elapsedMs = elapsedMs.takeIf { result.isSuccess },
                    loadElapsedMs = loadMs.takeIf { result.isSuccess },
                    generateElapsedMs = generateMs.takeIf { result.isSuccess },
                    feedbackJsonlPath = inferenceFeedbackJsonlFile().absolutePath,
                    error = failure?.message ?: failure?.javaClass?.simpleName
                )
            )
        }
    }

    fun chooseNativeInferenceCandidate(label: String) {
        val state = _uiState.value.nativeInference
        val a = state.candidateA?.takeIf { it.isNotBlank() } ?: return
        val b = state.candidateB?.takeIf { it.isNotBlank() } ?: return
        val normalized = label.trim().uppercase(Locale.US)
        if (normalized != "A" && normalized != "B") return

        val prompt = FeedbackPrompt(
            prompt = state.prompt,
            answerA = a,
            answerB = b
        )
        val file = inferenceFeedbackJsonlFile()
        val appendError = runCatching {
            file.parentFile?.mkdirs()
            val line = FeedbackJsonl.encodeChoice(
                FeedbackJsonl.ChoiceRecord(
                    prompt = prompt,
                    chosenLabel = normalized,
                    sessionId = "qwen_infer_${System.currentTimeMillis()}",
                    choiceIndex = state.savedPairs,
                    timestampMs = System.currentTimeMillis(),
                    source = "android_qwen_inference_feedback"
                )
            )
            file.appendText(line + "\n")
        }.exceptionOrNull()

        _uiState.value = _uiState.value.copy(
            nativeInference = state.copy(
                savedPairs = if (appendError == null) state.savedPairs + 1 else state.savedPairs,
                feedbackJsonlPath = file.absolutePath,
                lastChosenLabel = normalized.takeIf { appendError == null },
                error = appendError?.message ?: appendError?.javaClass?.simpleName
            ),
            nativeDemoRunner = _uiState.value.nativeDemoRunner.copy(
                commandPreview = buildNativeDemoPreview(_uiState.value.nativeDemoConfig),
                trainingSourceLabel = currentNativeDemoTrainingSourceLabel(),
                trainingJsonlPath = currentNativeDemoTrainingJsonlPath(),
                trainingPairCount = countJsonlLines(currentNativeDemoTrainingJsonlFile())
            )
        )
    }

    private fun closeNativeInferenceSession() {
        runCatching { nativeInferenceSession?.close() }
        nativeInferenceSession = null
        nativeInferenceModelDir = null
    }

    override fun onCleared() {
        closeNativeInferenceSession()
        super.onCleared()
    }

    fun startNativeDemoRun() {
        if (nativeDemoJob?.isActive == true) return
        val config = _uiState.value.nativeDemoConfig

        nativeDemoJob = viewModelScope.launch(Dispatchers.IO) {
            val app = getApplication<Application>()
            val root = app.filesDir
            val inferenceFeedbackFile = inferenceFeedbackJsonlFile()
            val useInferenceFeedback = inferenceFeedbackFile.exists() && inferenceFeedbackFile.length() > 0L
            val runPrefix = if (useInferenceFeedback) "phone_qwen_ab_dpo" else "phone_hf_fitness"
            val runName = "${runPrefix}_${config.steps}s_${timestampSuffix()}"
            val runDir = File(root, "mft_runs/$runName")
            val trainFile = if (useInferenceFeedback) {
                File(runDir, "inference_feedback.jsonl")
            } else {
                File(root, "mft_demo_data/hf_fitness_pref_train.jsonl")
            }
            val evalFile = File(root, "mft_demo_data/hf_fitness_pref_eval.jsonl")

            try {
                nativeDemoStopRequested = false
                nativeDemoRunDir = runDir
                runDir.mkdirs()
                if (useInferenceFeedback) {
                    writeFitnessPreferenceData(File(root, "mft_demo_data/hf_fitness_pref_train.jsonl"), evalFile)
                    inferenceFeedbackFile.copyTo(trainFile, overwrite = true)
                } else {
                    writeFitnessPreferenceData(trainFile, evalFile)
                }
                val pairCount = countJsonlLines(trainFile)

                val paths = NativeDemoCommandBuilder.Paths(
                    binaryPath = nativeDemoBinaryPath(app),
                    trainJsonl = trainFile.absolutePath,
                    evalJsonl = evalFile.absolutePath,
                    runDir = runDir.absolutePath,
                    loraOut = "hf_fitness_adapter_${config.steps}s.safetensors"
                )
                val command = NativeDemoCommandBuilder.buildShellCommand(config, paths)
                val handle = runHandleForFileRun(runName, runDir)
                val runs = listOf(handle) + _uiState.value.runHandles.filterNot { it.id == handle.id }

                _uiState.value = _uiState.value.copy(
                    activeTab = DashboardTab.OVERVIEW,
                    runHandles = runs,
                    selectedRunId = handle.id,
                    compareRunId = _uiState.value.compareRunId?.takeIf { it != handle.id },
                    nativeDemoRunner = NativeDemoRunnerState(
                        status = NativeDemoRunStatus.RUNNING,
                        currentRunName = runName,
                        currentRunPath = runDir.absolutePath,
                        commandPreview = command,
                        trainingSourceLabel = if (useInferenceFeedback) {
                            "Qwen A/B feedback"
                        } else {
                            "HF fitness preference dataset"
                        },
                        trainingJsonlPath = trainFile.absolutePath,
                        trainingPairCount = pairCount,
                        message = if (useInferenceFeedback) {
                            "Training native DPO on saved Qwen A/B feedback."
                        } else {
                            "Training native DPO on bundled HF fitness preferences."
                        }
                    ),
                    error = null
                )
                startPolling()

                nativeDemoProcess = ProcessBuilder("/system/bin/sh", "-c", command)
                    .directory(runDir)
                    .redirectErrorStream(true)
                    .start()
                val exit = nativeDemoProcess?.waitFor() ?: -1
                nativeDemoProcess = null

                val status = when {
                    nativeDemoStopRequested -> NativeDemoRunStatus.STOPPED
                    exit == 0 -> NativeDemoRunStatus.COMPLETED
                    else -> NativeDemoRunStatus.FAILED
                }
                _uiState.value = _uiState.value.copy(
                    nativeDemoRunner = _uiState.value.nativeDemoRunner.copy(
                        status = status,
                        lastExitCode = exit,
                        message = when (status) {
                            NativeDemoRunStatus.STOPPED -> "Native demo stopped."
                            NativeDemoRunStatus.COMPLETED -> "Native demo completed."
                            else -> "Native demo exited with code $exit."
                        }
                    )
                )
                startPolling()
            } catch (t: Throwable) {
                nativeDemoProcess = null
                _uiState.value = _uiState.value.copy(
                    nativeDemoRunner = _uiState.value.nativeDemoRunner.copy(
                        status = NativeDemoRunStatus.FAILED,
                        message = "Launch failed: ${t.message ?: t.javaClass.simpleName}"
                    ),
                    error = "Native demo launch failed. Check that the bundled trainer and /data/local/tmp/mft_gpt2 model files are available."
                )
            }
        }
    }

    fun stopNativeDemoRun() {
        nativeDemoStopRequested = true
        _uiState.value = _uiState.value.copy(
            nativeDemoRunner = _uiState.value.nativeDemoRunner.copy(
                status = NativeDemoRunStatus.STOPPING,
                message = "Stopping native demo..."
            )
        )
        nativeDemoJob?.let {
            viewModelScope.launch(Dispatchers.IO) {
                killNativeDemoChildProcess()
            }
        }
        nativeDemoProcess?.destroy()
    }

    private fun killNativeDemoChildProcess() {
        val runDir = nativeDemoRunDir ?: return
        val pidFile = File(runDir, "native.pid")
        val pid = (0 until 10).firstNotNullOfOrNull {
            runCatching {
                pidFile.readText().trim().toIntOrNull()
            }.getOrNull().also { found ->
                if (found == null) Thread.sleep(200)
            }
        } ?: return

        runCatching {
            File(runDir, "train.log").appendText("\n[Stopped] User requested stop. Sending TERM to native pid $pid.\n")
        }

        runCatching {
            ProcessBuilder(
                "/system/bin/sh",
                "-c",
                "kill -TERM $pid 2>/dev/null; sleep 1; kill -KILL $pid 2>/dev/null"
            ).start().waitFor()
        }
    }

    fun chooseFeedback(label: String) {
        val session = _uiState.value.feedbackSession ?: return
        if (session.isComplete) return
        val prompt = session.currentPrompt ?: return

        val appendError = runCatching {
            appendFeedbackChoice(
                session = session,
                prompt = prompt,
                label = label
            )
        }.exceptionOrNull()
        if (appendError != null) {
            _uiState.value = _uiState.value.copy(
                error = "Could not save feedback JSONL: ${appendError.message ?: "unknown error"}"
            )
            return
        }

        val nextSession = session.copy(
            currentIndex = session.currentIndex + 1,
            chosenLabels = session.chosenLabels + label
        )
        if (nextSession.isComplete) {
            startDpoTestTraining(nextSession)
        } else {
            _uiState.value = _uiState.value.copy(feedbackSession = nextSession)
        }
    }

    private fun startDpoTestTraining(session: FeedbackSessionState) {
        val dpoRuns = listOf(
            RunHandle(
                id = DPO_TEST_ID,
                name = "test/dpo_hfrl_android",
                dirUri = Uri.EMPTY,
                trainLogUri = Uri.EMPTY,
                rssCsvUri = null,
                metricsNdjsonUri = null,
                lastModifiedMs = System.currentTimeMillis()
            )
        )
        dpoCursor = DPO_START_CURSOR
        _uiState.value = _uiState.value.copy(
            activeTab = DashboardTab.OVERVIEW,
            runHandles = dpoRuns,
            selectedRunId = DPO_TEST_ID,
            compareRunId = null,
            snapshot = null,
            compareSnapshot = null,
            feedbackSession = session,
            isLoading = false,
            error = null
        )
        startPolling()
    }

    private fun appendFeedbackChoice(
        session: FeedbackSessionState,
        prompt: FeedbackPrompt,
        label: String
    ) {
        val file = feedbackJsonlFile()
        file.parentFile?.mkdirs()
        val line = FeedbackJsonl.encodeChoice(
            FeedbackJsonl.ChoiceRecord(
                prompt = prompt,
                chosenLabel = label,
                sessionId = session.sessionId,
                choiceIndex = session.chosenLabels.size,
                timestampMs = System.currentTimeMillis()
            )
        )
        file.appendText(line + "\n")
    }

    private fun feedbackJsonlFile(): File {
        val app = getApplication<Application>()
        val root = app.getExternalFilesDir(null) ?: app.filesDir
        return File(root, "hfrl_feedback/android_ab_feedback.jsonl")
    }

    private fun inferenceFeedbackJsonlFile(): File {
        val app = getApplication<Application>()
        val root = app.filesDir
        return File(root, "mft_demo_data/inference_feedback.jsonl")
    }

    fun selectRun(runId: String) {
        _uiState.value = _uiState.value.copy(selectedRunId = runId)
        startPolling()
    }

    fun selectComparisonRun(runId: String?) {
        _uiState.value = _uiState.value.copy(compareRunId = runId)
        startPolling()
    }

    private fun scanRuns(rootUri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val root = DocumentFile.fromTreeUri(getApplication(), rootUri)
            if (root == null || !root.exists() || !root.isDirectory) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = "Cannot open selected directory."
                )
                return@launch
            }

            val runs = discoverRuns(root)
            val selectedId = _uiState.value.selectedRunId
            val activeId = when {
                runs.isEmpty() -> null
                selectedId != null && runs.any { it.id == selectedId } -> selectedId
                else -> runs.first().id
            }
            val existingCompareId = _uiState.value.compareRunId
            val compareId = when {
                runs.isEmpty() -> null
                existingCompareId != null && runs.any { it.id == existingCompareId } &&
                    existingCompareId != activeId -> existingCompareId
                else -> runs.firstOrNull { it.id != activeId }?.id
            }

            _uiState.value = _uiState.value.copy(
                isLoading = false,
                runHandles = runs,
                selectedRunId = activeId,
                compareRunId = compareId,
                compareSnapshot = null,
                error = if (runs.isEmpty()) "No train.log found under this folder." else null
            )

            startPolling()
        }
    }

    private fun startPolling() {
        pollingJob?.cancel()
        val runId = _uiState.value.selectedRunId ?: return

        pollingJob = viewModelScope.launch(Dispatchers.IO) {
            if (runId == DPO_TEST_ID) {
                while (true) {
                    val feedbackSession = _uiState.value.feedbackSession
                    _uiState.value = _uiState.value.copy(
                        snapshot = buildDpoTestSnapshot(dpoCursor, feedbackSession),
                        compareSnapshot = null
                    )
                    dpoCursor = if (dpoCursor >= DPO_TOTAL_STEPS) DPO_START_CURSOR else dpoCursor + 1
                    delay(DEMO_TICK_MS)
                }
            }

            if (runId.startsWith(DEMO_PREFIX)) {
                while (true) {
                    val primary = buildDemoSnapshot(DEMO_PRIMARY_ID, "demo/primary_gpt2", isCompare = false, cursor = demoCursor)
                    val compare = buildDemoSnapshot(DEMO_COMPARE_ID, "demo/compare_gemma", isCompare = true, cursor = demoCursor)
                    _uiState.value = _uiState.value.copy(
                        snapshot = primary,
                        compareSnapshot = compare
                    )
                    demoCursor = if (demoCursor >= DEMO_TOTAL_STEPS) DEMO_START_CURSOR else demoCursor + 1
                    delay(DEMO_TICK_MS)
                }
            }

            while (true) {
                val state = _uiState.value
                val run = state.runHandles.firstOrNull { it.id == runId } ?: break
                val snapshot = parseRunSnapshot(run)

                val compareSnapshot = state.compareRunId
                    ?.takeIf { it != runId }
                    ?.let { compareId ->
                        state.runHandles.firstOrNull { it.id == compareId }
                    }
                    ?.let { parseRunSnapshot(it) }

                _uiState.value = _uiState.value.copy(
                    snapshot = snapshot,
                    compareSnapshot = compareSnapshot
                )

                val delayMs = if (snapshot.status == RunStatus.RUNNING) POLL_FAST_MS else POLL_SLOW_MS
                delay(delayMs)
            }
        }
    }

    private fun discoverRuns(root: DocumentFile): List<RunHandle> {
        val runs = mutableListOf<RunHandle>()

        fun walk(dir: DocumentFile, relPath: String, depth: Int) {
            if (depth > MAX_SCAN_DEPTH || !dir.isDirectory) return

            val children = runCatching { dir.listFiles().toList() }.getOrDefault(emptyList())
            if (children.isEmpty()) return

            val train = children.firstOrNull { it.isFile && it.name == "train.log" }
            if (train != null) {
                val rss = children.firstOrNull { it.isFile && it.name == "rss.csv" }
                val ndjson = children.firstOrNull {
                    it.isFile && (it.name == "metrics.ndjson" || it.name == "metrics.jsonl")
                }
                val generated = children.firstOrNull {
                    it.isFile && it.name == "generated_responses.jsonl"
                }
                val feedback = children.firstOrNull {
                    it.isFile && it.name == "inference_feedback.jsonl"
                }
                val runName = if (relPath.isBlank()) {
                    dir.name ?: "run"
                } else {
                    relPath
                }
                val lastModified = listOfNotNull(
                    train.lastModified(),
                    rss?.lastModified(),
                    ndjson?.lastModified(),
                    generated?.lastModified(),
                    feedback?.lastModified()
                ).maxOrNull() ?: 0L

                runs += RunHandle(
                    id = dir.uri.toString(),
                    name = runName,
                    dirUri = dir.uri,
                    trainLogUri = train.uri,
                    rssCsvUri = rss?.uri,
                    metricsNdjsonUri = ndjson?.uri,
                    generatedResponsesUri = generated?.uri,
                    inferenceFeedbackUri = feedback?.uri,
                    lastModifiedMs = lastModified
                )
            }

            children
                .filter { it.isDirectory }
                .forEach { child ->
                    val childName = child.name ?: "folder"
                    val nextPath = if (relPath.isBlank()) childName else "$relPath/$childName"
                    walk(child, nextPath, depth + 1)
                }
        }

        walk(root, "", 0)
        return runs.sortedByDescending { it.lastModifiedMs }
    }

    private suspend fun parseRunSnapshot(run: RunHandle): RunSnapshot = withContext(Dispatchers.IO) {
        val logLines = readLastLines(run.trainLogUri, MAX_LOG_LINES)
        val parsedLog = TrainingLogParser.parseTrainLog(logLines)

        val rssPoints = run.rssCsvUri
            ?.let { readLastLines(it, MAX_RSS_LINES) }
            ?.let { TrainingLogParser.parseRssCsv(it) }
            ?: emptyList()

        val ndjsonMetrics = run.metricsNdjsonUri
            ?.let { readLastLines(it, MAX_NDJSON_LINES) }
            ?.let { TrainingLogParser.parseMetricsNdjson(it) }
            ?: emptyList()

        val mergedMetrics = mergeMetrics(parsedLog.metrics, ndjsonMetrics)
        val generatedRecords = run.generatedResponsesUri
            ?.let { readLastLines(it, MAX_INFERENCE_LINES) }
            ?.let { TrainingLogParser.parseGeneratedResponsesJsonl(it) }
            ?: emptyList()
        val feedbackRecords = run.inferenceFeedbackUri
            ?.let { readLastLines(it, MAX_INFERENCE_LINES) }
            ?.let { TrainingLogParser.parseInferenceFeedbackJsonl(it) }
            ?: emptyList()
        val inferenceRecords = mergeInferenceRecords(generatedRecords, feedbackRecords)

        RunSnapshot(
            runId = run.id,
            runName = run.name,
            status = if (ndjsonMetrics.isNotEmpty() && parsedLog.status == RunStatus.IDLE) {
                RunStatus.RUNNING
            } else {
                parsedLog.status
            },
            metrics = mergedMetrics,
            rssPoints = rssPoints,
            events = parsedLog.events.takeLast(MAX_EVENT_LINES),
            inferenceRecords = inferenceRecords,
            logTail = logLines.takeLast(MAX_LOG_TAIL_LINES),
            summary = parsedLog.summary,
            updatedAtMs = System.currentTimeMillis()
        )
    }

    private fun mergeInferenceRecords(
        generated: List<InferenceRecord>,
        feedback: List<InferenceRecord>
    ): List<InferenceRecord> {
        if (feedback.isEmpty()) return generated
        if (generated.isEmpty()) return feedback

        val generatedByKey = generated.associateBy { inferenceKey(it) }
        return feedback.map { item ->
            val source = generatedByKey[inferenceKey(item)]
            item.copy(
                completion = source?.completion ?: item.completion,
                chosenSource = item.chosenSource ?: source?.chosenSource,
                elapsedMs = source?.elapsedMs ?: item.elapsedMs,
                source = item.source ?: source?.source
            )
        }
    }

    private fun inferenceKey(record: InferenceRecord): String {
        return record.promptId?.takeIf { it.isNotBlank() } ?: record.prompt
    }

    private fun mergeMetrics(logMetrics: List<StepMetric>, ndjsonMetrics: List<StepMetric>): List<StepMetric> {
        if (ndjsonMetrics.isEmpty()) return logMetrics
        if (logMetrics.isEmpty()) return ndjsonMetrics

        val map = linkedMapOf<Int, StepMetric>()
        for (m in logMetrics) {
            map[m.step] = m
        }
        for (m in ndjsonMetrics) {
            val cur = map[m.step]
            map[m.step] = if (cur == null) {
                m
            } else {
                val keepEvalValues = cur.source.contains("eval", ignoreCase = true)
                StepMetric(
                    step = m.step,
                    totalSteps = m.totalSteps ?: cur.totalSteps,
                    loss = if (keepEvalValues) cur.loss ?: m.loss else m.loss ?: cur.loss,
                    ppl = m.ppl ?: cur.ppl,
                    lr = m.lr ?: cur.lr,
                    tokens = if (keepEvalValues) cur.tokens ?: m.tokens else m.tokens ?: cur.tokens,
                    rewardAccuracy = if (keepEvalValues) cur.rewardAccuracy ?: m.rewardAccuracy else m.rewardAccuracy ?: cur.rewardAccuracy,
                    rewardMargin = if (keepEvalValues) cur.rewardMargin ?: m.rewardMargin else m.rewardMargin ?: cur.rewardMargin,
                    chosenReward = if (keepEvalValues) cur.chosenReward ?: m.chosenReward else m.chosenReward ?: cur.chosenReward,
                    rejectedReward = if (keepEvalValues) cur.rejectedReward ?: m.rejectedReward else m.rejectedReward ?: cur.rejectedReward,
                    gradNorm = m.gradNorm ?: cur.gradNorm,
                    rssMb = m.rssMb ?: cur.rssMb,
                    source = if (keepEvalValues) {
                        "dpo_eval_merge"
                    } else {
                        "merge"
                    }
                )
            }
        }
        return map.values.sortedBy { it.step }
    }

    private fun buildDemoSnapshot(
        runId: String,
        runName: String,
        isCompare: Boolean,
        cursor: Int
    ): RunSnapshot {
        val maxStep = cursor.coerceIn(DEMO_START_CURSOR, DEMO_TOTAL_STEPS)
        val metrics = (1..maxStep).map { step ->
            val baseLoss = if (isCompare) 5.2 else 5.0
            val speed = if (isCompare) 0.021 else 0.025
            val oscillation = if (isCompare) 0.24 else 0.18
            val loss = (baseLoss - speed * step + oscillation * kotlin.math.sin(step / 7.2)).coerceAtLeast(1.2)
            val ppl = kotlin.math.exp(loss / 2.15).coerceAtMost(2200.0)
            val lr = (2e-4 * (1.0 - (step / DEMO_TOTAL_STEPS.toDouble()) * 0.82)).coerceAtLeast(2.0e-5)
            val tokens = 1024 + (if (isCompare) 128 else 0)
            val rewardAccuracySlope = if (isCompare) 0.0012 else 0.0018
            val rewardMarginSlope = if (isCompare) 0.004 else 0.006
            val rewardAccuracy = (0.54 + step * rewardAccuracySlope).coerceAtMost(if (isCompare) 0.78 else 0.86)
            val rewardMargin = (-0.08 + step * rewardMarginSlope).coerceAtMost(if (isCompare) 0.82 else 1.18)
            StepMetric(
                step = step,
                totalSteps = DEMO_TOTAL_STEPS,
                loss = loss,
                ppl = ppl,
                lr = lr,
                tokens = tokens,
                rewardAccuracy = rewardAccuracy,
                rewardMargin = rewardMargin,
                source = "demo"
            )
        }

        val rss = (0 until maxStep).map { i ->
            val trend = if (isCompare) 1940.0 else 1820.0
            val ramp = i * if (isCompare) 0.9 else 0.7
            val wobble = kotlin.math.sin(i / 4.0) * 21.0
            RssPoint(
                index = i,
                rssMb = trend + ramp + wobble,
                timeLabel = null
            )
        }

        val evalPpl = formatDemoValue("%.2f", metrics.lastOrNull()?.ppl)
        val events = listOf(
            RunEvent(step = maxStep, type = EventType.INFO, message = "[Train] step=$maxStep running", raw = "demo"),
            RunEvent(step = maxStep, type = EventType.RSS, message = "[RSS] ${(rss.lastOrNull()?.rssMb ?: 0.0).toInt()} MB", raw = "demo"),
            RunEvent(step = maxStep, type = EventType.EVAL, message = "[Eval] ppl=$evalPpl", raw = "demo")
        )

        val status = if (maxStep >= DEMO_TOTAL_STEPS) RunStatus.COMPLETED else RunStatus.RUNNING
        val summary = RunSummary(
            maxTrainRssMb = rss.maxOfOrNull { it.rssMb },
            finalEmaLoss = metrics.lastOrNull()?.loss,
            totalSteps = DEMO_TOTAL_STEPS,
            totalTokens = metrics.sumOf { it.tokens?.toLong() ?: 0L }
        )

        return RunSnapshot(
            runId = runId,
            runName = runName,
            status = status,
            metrics = metrics,
            rssPoints = rss,
            events = events,
            logTail = buildList {
                val tail = metrics.takeLast(90)
                tail.forEachIndexed { idx, m ->
                    add("[Train] step=${m.step}/${m.totalSteps} | loss=${formatDemoValue("%.3f", m.loss)} | ppl=${formatDemoValue("%.2f", m.ppl)} | lr=${formatDemoValue("%.6f", m.lr)}")
                    // Interleave RSS every 10 steps
                    if (idx % 10 == 9) {
                        val rssMb = (if (isCompare) 1940.0 + idx * 0.9 else 1820.0 + idx * 0.7)
                            .coerceAtMost(2200.0)
                        add("[RSS] step=${m.step} rss=${rssMb.toInt()} MB")
                    }
                    // Interleave Eval every 20 steps
                    if (idx % 20 == 19) {
                        val logEvalPpl = formatDemoValue("%.2f", m.ppl)
                        add("[Eval] step=${m.step} ppl=$logEvalPpl")
                    }
                    // Interleave Checkpoint every 30 steps
                    if (idx % 30 == 29) {
                        add("[Checkpoint] step=${m.step} saved checkpoint to /data/ckpt/step_${m.step}.bin")
                    }
                }
            },
            summary = summary,
            updatedAtMs = System.currentTimeMillis()
        )
    }

    private fun buildDpoTestSnapshot(
        cursor: Int,
        feedbackSession: FeedbackSessionState?
    ): RunSnapshot {
        val maxStep = cursor.coerceIn(DPO_START_CURSOR, DPO_TOTAL_STEPS)
        val metrics = (1..maxStep).map { step ->
            val progress = step / DPO_TOTAL_STEPS.toDouble()
            val loss = (0.72 - progress * 0.22 + 0.018 * kotlin.math.sin(step / 4.8)).coerceAtLeast(0.43)
            val rewardAccuracy = (0.50 + progress * 0.36 + 0.018 * kotlin.math.sin(step / 8.0)).coerceIn(0.0, 0.92)
            val rewardMargin = (-0.05 + progress * 0.78 + 0.045 * kotlin.math.sin(step / 6.2))
            val chosenReward = rewardMargin / 2.0 + 0.08 + 0.01 * kotlin.math.sin(step / 5.0)
            val rejectedReward = chosenReward - rewardMargin
            val lr = (1.5e-4 * (1.0 - progress * 0.65)).coerceAtLeast(2.5e-5)
            StepMetric(
                step = step,
                totalSteps = DPO_TOTAL_STEPS,
                loss = loss,
                ppl = null,
                lr = lr,
                tokens = 512,
                rewardAccuracy = rewardAccuracy,
                rewardMargin = rewardMargin,
                chosenReward = chosenReward,
                rejectedReward = rejectedReward,
                gradNorm = 0.82 + 0.07 * kotlin.math.sin(step / 7.0),
                rssMb = 1460.0 + step * 0.9 + 18.0 * kotlin.math.sin(step / 5.5),
                source = "dpo-test"
            )
        }

        val rss = metrics.mapIndexed { index, metric ->
            RssPoint(index = index, rssMb = metric.rssMb ?: 0.0)
        }
        val last = metrics.last()
        val feedbackPath = feedbackSession?.feedbackJsonlPath ?: feedbackJsonlFile().absolutePath
        val feedbackPairs = feedbackSession?.chosenLabels?.size ?: 0
        val events = listOf(
            RunEvent(
                step = maxStep,
                type = EventType.INFO,
                message = "[DPO] feedback JSONL saved: $feedbackPairs pairs at $feedbackPath",
                raw = "dpo-test"
            ),
            RunEvent(step = maxStep, type = EventType.EVAL, message = "[DPO] reward_acc=${formatDemoValue("%.3f", last.rewardAccuracy)} margin=${formatDemoValue("%.3f", last.rewardMargin)}", raw = "dpo-test"),
            RunEvent(step = maxStep, type = EventType.RSS, message = "[RSS] ${formatDemoValue("%.1f", last.rssMb)} MB", raw = "dpo-test")
        )
        val status = if (maxStep >= DPO_TOTAL_STEPS) RunStatus.COMPLETED else RunStatus.RUNNING
        return RunSnapshot(
            runId = DPO_TEST_ID,
            runName = "test/dpo_hfrl_android",
            status = status,
            metrics = metrics,
            rssPoints = rss,
            events = events,
            logTail = metrics.takeLast(80).map { m ->
                "[DPO] step=${m.step}/${m.totalSteps} | loss=${formatDemoValue("%.4f", m.loss)} | reward_acc=${formatDemoValue("%.3f", m.rewardAccuracy)} | reward_margin=${formatDemoValue("%.3f", m.rewardMargin)} | grad_norm=${formatDemoValue("%.3f", m.gradNorm)}"
            },
            summary = RunSummary(
                maxTrainRssMb = rss.maxOfOrNull { it.rssMb },
                finalEmaLoss = last.loss,
                totalSteps = DPO_TOTAL_STEPS,
                totalTokens = metrics.sumOf { it.tokens?.toLong() ?: 0L }
            ),
            updatedAtMs = System.currentTimeMillis()
        )
    }

    private fun readLastLines(uri: Uri, maxLines: Int): List<String> {
        val deque = ArrayDeque<String>(maxLines)
        val input = runCatching {
            if (uri.scheme == "file") {
                uri.path?.let { path ->
                    File(path).takeIf { it.exists() }?.inputStream()
                }
            } else {
                resolver.openInputStream(uri)
            }
        }.getOrNull()
        input?.bufferedReader()?.useLines { sequence ->
            sequence.forEach { line ->
                if (deque.size == maxLines) {
                    deque.removeFirst()
                }
                deque.addLast(line)
            }
        }
        return deque.toList()
    }

    private fun runHandleForFileRun(runName: String, runDir: File): RunHandle {
        val trainLog = File(runDir, "train.log")
        val metrics = File(runDir, "metrics.ndjson")
        val eval = File(runDir, "eval_metrics.jsonl")
        val generated = File(runDir, "generated_responses.jsonl")
        val feedback = File(runDir, "inference_feedback.jsonl")
        return RunHandle(
            id = Uri.fromFile(runDir).toString(),
            name = "native/$runName",
            dirUri = Uri.fromFile(runDir),
            trainLogUri = Uri.fromFile(trainLog),
            rssCsvUri = null,
            metricsNdjsonUri = Uri.fromFile(metrics.takeIf { it.exists() } ?: metrics),
            generatedResponsesUri = generated.takeIf { it.exists() }?.let { Uri.fromFile(it) },
            inferenceFeedbackUri = feedback.takeIf { it.exists() }?.let { Uri.fromFile(it) },
            lastModifiedMs = listOf(trainLog, metrics, eval, generated, feedback)
                .filter { it.exists() }
                .maxOfOrNull { it.lastModified() }
                ?: System.currentTimeMillis()
        )
    }

    private fun refreshNativeDemoCommandPreview() {
        val config = _uiState.value.nativeDemoConfig
        _uiState.value = _uiState.value.copy(
            nativeDemoRunner = _uiState.value.nativeDemoRunner.copy(
                commandPreview = buildNativeDemoPreview(config),
                trainingSourceLabel = currentNativeDemoTrainingSourceLabel(),
                trainingJsonlPath = currentNativeDemoTrainingJsonlPath(),
                trainingPairCount = countJsonlLines(currentNativeDemoTrainingJsonlFile())
            )
        )
    }

    private fun buildNativeDemoPreview(config: NativeDemoConfig): String {
        val app = getApplication<Application>()
        val root = app.filesDir.absolutePath
        val trainJsonl = currentNativeDemoTrainingJsonlPath()
        return NativeDemoCommandBuilder.buildShellCommand(
            config,
            NativeDemoCommandBuilder.Paths(
                binaryPath = nativeDemoBinaryPath(app),
                trainJsonl = trainJsonl,
                evalJsonl = "$root/mft_demo_data/hf_fitness_pref_eval.jsonl",
                runDir = "$root/mft_runs/<new-run>",
                loraOut = "hf_fitness_adapter_${config.steps}s.safetensors"
            )
        )
    }

    private fun currentNativeDemoTrainingJsonlFile(): File {
        val app = getApplication<Application>()
        val feedback = inferenceFeedbackJsonlFile()
        return if (feedback.exists() && feedback.length() > 0L) {
            feedback
        } else {
            File(app.filesDir, "mft_demo_data/hf_fitness_pref_train.jsonl")
        }
    }

    private fun currentNativeDemoTrainingJsonlPath(): String {
        return currentNativeDemoTrainingJsonlFile().absolutePath
    }

    private fun currentNativeDemoTrainingSourceLabel(): String {
        val feedback = inferenceFeedbackJsonlFile()
        return if (feedback.exists() && feedback.length() > 0L) {
            "Qwen A/B feedback"
        } else {
            "HF fitness preference dataset"
        }
    }

    private fun countJsonlLines(file: File): Int {
        if (!file.exists() || file.length() == 0L) return 0
        return runCatching {
            file.useLines { lines -> lines.count { it.isNotBlank() } }
        }.getOrDefault(0)
    }

    private fun nativeDemoBinaryPath(app: Application): String {
        return File(app.applicationInfo.nativeLibraryDir, "libmft_train.so").absolutePath
    }

    private fun writeFitnessPreferenceData(trainFile: File, evalFile: File) {
        trainFile.parentFile?.mkdirs()
        evalFile.parentFile?.mkdirs()
        val app = getApplication<Application>()
        runCatching {
            copyAssetToFile(app, FITNESS_PREF_TRAIN_ASSET, trainFile)
            copyAssetToFile(app, FITNESS_PREF_EVAL_ASSET, evalFile)
        }.onFailure {
            trainFile.writeText(BUILT_IN_FITNESS_PREF_TRAIN.joinToString("\n") + "\n")
            evalFile.writeText(BUILT_IN_FITNESS_PREF_EVAL.joinToString("\n") + "\n")
        }
    }

    private fun copyAssetToFile(app: Application, assetPath: String, outFile: File) {
        app.assets.open(assetPath).use { input ->
            outFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
    }

    private fun timestampSuffix(): String {
        return LocalDateTime.now().format(DateTimeFormatter.ofPattern("MMdd_HHmmss"))
    }

    private fun formatDemoValue(pattern: String, value: Double?): String {
        return value?.let { String.format(Locale.US, pattern, it) } ?: "N/A"
    }

    companion object {
        private const val PREFS_NAME = "mft_visualizer"
        private const val KEY_LAST_ROOT_URI = "last_root_uri"

        private const val MAX_SCAN_DEPTH = 10
        private const val MAX_LOG_LINES = 20000
        private const val MAX_LOG_TAIL_LINES = 1000
        private const val MAX_RSS_LINES = 12000
        private const val MAX_NDJSON_LINES = 20000
        private const val MAX_INFERENCE_LINES = 200
        private const val MAX_EVENT_LINES = 500

        private const val POLL_FAST_MS = 1500L
        private const val POLL_SLOW_MS = 3500L

        private const val DEMO_PREFIX = "demo:"
        private const val DEMO_PRIMARY_ID = "${DEMO_PREFIX}primary"
        private const val DEMO_COMPARE_ID = "${DEMO_PREFIX}compare"
        private const val DEMO_TOTAL_STEPS = 220
        private const val DEMO_START_CURSOR = 24
        private const val DEMO_TICK_MS = 1200L

        private const val DPO_TEST_ID = "demo:dpo_hfrl_test"
        private const val DPO_TOTAL_STEPS = 80
        private const val DPO_START_CURSOR = 8
        private const val FITNESS_PREF_TRAIN_ASSET = "mft_demo_data/hf_fitness_pref_train.jsonl"
        private const val FITNESS_PREF_EVAL_ASSET = "mft_demo_data/hf_fitness_pref_eval.jsonl"

        private val DPO_FEEDBACK_PROMPTS = listOf(
            FeedbackPrompt(
                prompt = "How should MobileFineTuner reduce memory during LoRA training?",
                answerA = "Use smaller batches, cache static reference logprobs, and keep reward metrics visible during training.",
                answerB = "Increase batch size and disable monitoring so the run has fewer logs."
            ),
            FeedbackPrompt(
                prompt = "Which response is better for explaining DPO loss?",
                answerA = "DPO loss measures preference separation against a reference policy; reward accuracy and margin are easier to read for progress.",
                answerB = "DPO loss is the same as normal accuracy, so lower than 1.0 always means the model is good."
            ),
            FeedbackPrompt(
                prompt = "What should happen after a user chooses a preferred answer?",
                answerA = "Append a preference pair with prompt, chosen response, rejected response, and timestamp.",
                answerB = "Only update the chart locally and discard the choice."
            ),
            FeedbackPrompt(
                prompt = "Which Android integration path is safer first?",
                answerA = "Start with adb-shell native trainer launch and shared run folders, then move to an in-app launcher.",
                answerB = "Start with a full JNI rewrite before validating native training on the phone."
            ),
            FeedbackPrompt(
                prompt = "What metric should the app show for HFRL progress?",
                answerA = "Reward accuracy, reward margin, loss, grad norm, and RSS together.",
                answerB = "Only DPO loss, with no reward or memory context."
            )
        )

        private val BUILT_IN_FITNESS_PREF_TRAIN = listOf(
            """{"prompt":"What equipment should I prioritize for a home gym on a budget?","chosen":"For a budget-friendly home gym, start with adjustable dumbbells or resistance bands, an exercise mat, and a doorway pull-up bar. These cover strength, mobility, and upper-body pulling without taking much space. Add a jump rope or kettlebell later if you want conditioning. Buy one quality piece at a time and use second-hand listings to keep cost down.","rejected":"Buy every machine you can find immediately. A home gym is only useful if it looks like a commercial gym, and budget does not matter.","source":"android_builtin_fitness_demo"}""",
            """{"prompt":"What are the signs of overtraining and how can I prevent it?","chosen":"Watch for persistent fatigue, declining performance, poor sleep, elevated resting heart rate, frequent soreness, irritability, and loss of motivation. Prevent it with planned rest days, gradual load increases, deload weeks, enough sleep, hydration, and adequate food. If several symptoms persist, reduce training and consider professional guidance.","rejected":"The main sign of overtraining is laziness. Push harder every day and ignore soreness so your body adapts faster.","source":"android_builtin_fitness_demo"}""",
            """{"prompt":"How can I modify push-ups if I cannot do a full push-up yet?","chosen":"Use wall push-ups, incline push-ups on a bench, or knee push-ups. Keep your body line stable, brace your core, and lower under control. Start with sets of 5 to 10 clean reps, then lower the incline over time. Slow negative reps can also build strength for the full movement.","rejected":"Skip all modifications and only attempt full push-ups from the floor. If you fail, keep repeating them with poor form until you improve.","source":"android_builtin_fitness_demo"}""",
            """{"prompt":"Is Power Zone a good Peloton workout?","chosen":"Power Zone can be a strong Peloton option because it uses your FTP to personalize intensity. It is especially useful for structured endurance and performance goals. Start with Power Zone Endurance rides to learn the format, then take an FTP test when you are ready for calibrated zones.","rejected":"Power Zone is always the best workout for everyone. Take the hardest FTP test first and ignore whether you enjoy structured cycling.","source":"android_builtin_fitness_demo"}""",
            """{"prompt":"How should I return to exercise after a week of poor sleep?","chosen":"Keep the first session easy and short. Choose walking, light cycling, mobility, or a reduced strength session, and stop if effort feels unusually high. Use the week to rebuild consistency, prioritize sleep, and avoid max-effort intervals until energy and recovery normalize.","rejected":"Make up for the missed week by doubling your workout time and adding maximum-intensity intervals even if you still feel tired.","source":"android_builtin_fitness_demo"}"""
        )

        private val BUILT_IN_FITNESS_PREF_EVAL = listOf(
            """{"prompt":"How should I start running if I am completely new?","chosen":"Begin with run-walk intervals two or three times per week, such as one minute jogging and two minutes walking for 20 minutes. Keep the pace conversational, rest between sessions, and increase total running time gradually. Comfortable shoes and consistency matter more than speed.","rejected":"Start with a hard 5K every day so your body gets used to running quickly. Ignore discomfort and keep your pace as fast as possible.","source":"android_builtin_fitness_demo_eval"}""",
            """{"prompt":"What should I do if my knees hurt after squats?","chosen":"Reduce load and range of motion, check your form, warm up, and try pain-free variations such as box squats. Stop movements that create sharp pain, and seek a qualified clinician or coach if symptoms persist or worsen.","rejected":"Keep squatting heavy through knee pain because pain means the joint is getting stronger.","source":"android_builtin_fitness_demo_eval"}"""
        )
    }
}
