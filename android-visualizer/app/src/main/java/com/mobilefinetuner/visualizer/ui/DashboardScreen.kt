package com.mobilefinetuner.visualizer.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.MutableTransitionState
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ExpandMore
import androidx.compose.material.icons.rounded.FolderOpen
import androidx.compose.material.icons.rounded.PlayArrow
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.ThumbDown
import androidx.compose.material.icons.rounded.ThumbUp
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.mobilefinetuner.visualizer.model.DashboardTab
import com.mobilefinetuner.visualizer.model.EventType
import com.mobilefinetuner.visualizer.model.FeedbackSessionState
import com.mobilefinetuner.visualizer.model.InferenceRecord
import com.mobilefinetuner.visualizer.model.NativeDemoConfig
import com.mobilefinetuner.visualizer.model.NativeDemoRunStatus
import com.mobilefinetuner.visualizer.model.NativeDemoRunnerState
import com.mobilefinetuner.visualizer.model.NativeFeedbackMode
import com.mobilefinetuner.visualizer.model.NativeInferenceState
import com.mobilefinetuner.visualizer.model.RunHandle
import com.mobilefinetuner.visualizer.model.RunSnapshot
import com.mobilefinetuner.visualizer.model.RunStatus
import com.mobilefinetuner.visualizer.model.StepMetric
import com.mobilefinetuner.visualizer.ui.components.InteractiveComparisonChart
import com.mobilefinetuner.visualizer.ui.components.InteractiveMetricLineChart
import com.mobilefinetuner.visualizer.ui.components.MemoryWaterfallChart
import com.mobilefinetuner.visualizer.ui.components.MetricCard
import com.mobilefinetuner.visualizer.ui.components.StatusPill
import com.mobilefinetuner.visualizer.ui.components.TerminalLogView
import com.mobilefinetuner.visualizer.ui.theme.LossColor
import com.mobilefinetuner.visualizer.ui.theme.LrColor
import com.mobilefinetuner.visualizer.ui.theme.NeonAmber
import com.mobilefinetuner.visualizer.ui.theme.NeonBlue
import com.mobilefinetuner.visualizer.ui.theme.NeonGreen
import com.mobilefinetuner.visualizer.ui.theme.PplColor
import com.mobilefinetuner.visualizer.ui.theme.RssColor
import com.mobilefinetuner.visualizer.viewmodel.ExperimentViewModel
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DashboardScreen(
    viewModel: ExperimentViewModel,
    modifier: Modifier = Modifier
) {
    val ui by viewModel.uiState.collectAsState()
    val tabOrder = remember {
        listOf(
            DashboardTab.EXPERIMENTS,
            DashboardTab.OVERVIEW,
            DashboardTab.TRAINING,
            DashboardTab.COMPARISON,
            DashboardTab.LOGS
        )
    }

    val picker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { uri: Uri? ->
        if (uri != null) viewModel.onDirectoryPicked(uri)
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(Color(0xFFF6F7F9))
    ) {
        Scaffold(
            modifier = Modifier.fillMaxSize(),
            containerColor = Color.Transparent,
            topBar = {
                TopAppBar(
                    colors = TopAppBarDefaults.topAppBarColors(containerColor = Color.Transparent),
                    title = {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            // Brand accent gradient square
                            Box(
                                modifier = Modifier
                                    .size(9.dp)
                                    .background(
                                        brush = Brush.linearGradient(listOf(NeonBlue, NeonGreen)),
                                        shape = androidx.compose.foundation.shape.RoundedCornerShape(2.dp)
                                    )
                            )
                            Column {
                                Text(
                                    text  = "MobileFineTuner",
                                    style = MaterialTheme.typography.headlineSmall
                                )
                                Text(
                                    text     = ui.snapshot?.runName ?: "On-device preference tuning with Qwen3-0.6B",
                                    style    = MaterialTheme.typography.bodyMedium,
                                    color    = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.88f),
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                            }
                        }
                    },
                    actions = {
                        IconButton(onClick = { viewModel.refreshNow() }) {
                            Icon(Icons.Rounded.Refresh, contentDescription = "Refresh")
                        }
                    }
                )
            }
        ) { innerPadding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
            ) {
                if (ui.activeTab != DashboardTab.EXPERIMENTS && ui.activeTab != DashboardTab.RUNNER) {
                    ControlBar(
                        runs = ui.runHandles,
                        selectedRunId = ui.selectedRunId,
                        compareRunId = ui.compareRunId,
                        onPickDirectory = { picker.launch(null) },
                        onEnableDemo = viewModel::enableDemoMode,
                        onEnableDpoTest = viewModel::enableDpoTestMode,
                        onSelectRun = viewModel::selectRun,
                        onSelectCompareRun = viewModel::selectComparisonRun
                    )
                }

                TabRow(
                    selectedTabIndex = tabOrder.indexOf(ui.activeTab).coerceAtLeast(0),
                    modifier = Modifier
                        .padding(horizontal = 12.dp)
                        .clip(RoundedCornerShape(16.dp)),
                    containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.84f)
                ) {
                    tabOrder.forEach { tab ->
                        Tab(
                            selected = tab == ui.activeTab,
                            onClick = { viewModel.selectTab(tab) },
                            text = {
                                Text(
                                    text = shortTabTitle(tab),
                                    maxLines = 1,
                                    overflow = TextOverflow.Clip,
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }
                        )
                    }
                }

                if (ui.error != null) {
                    Surface(
                        color = MaterialTheme.colorScheme.error.copy(alpha = 0.12f),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(12.dp)
                    ) {
                        Text(
                            text = ui.error ?: "",
                            modifier = Modifier.padding(12.dp),
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                }

                AnimatedContent(
                    targetState = ui.activeTab,
                    label = "tabSwitch",
                    transitionSpec = {
                        (fadeIn() + slideInVertically { it / 10 }) togetherWith
                            (fadeOut() + slideOutVertically { -it / 12 })
                    }
                ) { tab ->
                    val feedbackSession = ui.feedbackSession
                    if (feedbackSession != null &&
                        !feedbackSession.isComplete &&
                        tab != DashboardTab.EXPERIMENTS &&
                        tab != DashboardTab.RUNNER
                    ) {
                        FeedbackCollectionPanel(
                            session = feedbackSession,
                            onChoose = viewModel::chooseFeedback
                        )
                    } else when (tab) {
                        DashboardTab.RUNNER -> RunnerScreen(runner = ui.runner, viewModel = viewModel)
                        DashboardTab.OVERVIEW -> OverviewTab(snapshot = ui.snapshot)
                        DashboardTab.TRAINING -> TrainingTab(snapshot = ui.snapshot)
                        DashboardTab.COMPARISON -> ComparisonTab(
                            primary = ui.snapshot,
                            compare = ui.compareSnapshot,
                            compareName = ui.runHandles.firstOrNull { it.id == ui.compareRunId }?.name
                        )
                        DashboardTab.LOGS -> LogsTab(snapshot = ui.snapshot)
                        DashboardTab.EXPERIMENTS -> ExperimentsTab(
                            runs = ui.runHandles,
                            selectedRunId = ui.selectedRunId,
                            compareRunId = ui.compareRunId,
                            config = ui.nativeDemoConfig,
                            runner = ui.nativeDemoRunner,
                            inference = ui.nativeInference,
                            snapshot = ui.snapshot,
                            feedbackSession = feedbackSession,
                            onPickDirectory = { picker.launch(null) },
                            onEnableDemo = viewModel::enableDemoMode,
                            onEnableDpoTest = viewModel::enableDpoTestMode,
                            onChooseFeedback = viewModel::chooseFeedback,
                            onInferenceChange = viewModel::updateNativeInference,
                            onGenerateSingle = viewModel::runNativeInference,
                            onGeneratePair = viewModel::runNativeInferencePair,
                            onChooseCandidate = viewModel::chooseNativeInferenceCandidate,
                            onChooseRating = viewModel::chooseNativeInferenceRating,
                            onConfigChange = viewModel::updateNativeDemoConfig,
                            onStart = viewModel::startNativeDemoRun,
                            onStop = viewModel::stopNativeDemoRun
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ControlBar(
    runs: List<RunHandle>,
    selectedRunId: String?,
    compareRunId: String?,
    onPickDirectory: () -> Unit,
    onEnableDemo: () -> Unit,
    onEnableDpoTest: () -> Unit,
    onSelectRun: (String) -> Unit,
    onSelectCompareRun: (String?) -> Unit
) {
    var primaryExpanded by remember { mutableStateOf(false) }
    var compareExpanded by remember { mutableStateOf(false) }

    val selected = runs.firstOrNull { it.id == selectedRunId }
    val compare = runs.firstOrNull { it.id == compareRunId }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 6.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Action buttons row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Button(onClick = onPickDirectory, modifier = Modifier.weight(1f)) {
                Icon(Icons.Rounded.FolderOpen, contentDescription = null)
                Text("Select Runs", modifier = Modifier.padding(start = 8.dp))
            }
            OutlinedButton(onClick = onEnableDemo) {
                Text("Load Demo")
            }
            OutlinedButton(onClick = onEnableDpoTest) {
                Text("Test DPO")
            }
        }

        // Unified run selector — single card with Primary + Compare rows
        Box(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .border(1.dp, Color(0xFFDDE2EC), RoundedCornerShape(16.dp))
                    .background(Color(0xFFFAFBFF))
            ) {
                // Primary row
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable(enabled = runs.isNotEmpty()) { primaryExpanded = true }
                        .padding(horizontal = 14.dp, vertical = 11.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text(
                        text  = "Primary",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color(0xFF007AFF),
                        modifier = Modifier
                            .background(Color(0xFF007AFF).copy(alpha = 0.10f), RoundedCornerShape(6.dp))
                            .padding(horizontal = 7.dp, vertical = 3.dp)
                    )
                    Text(
                        text     = selected?.name ?: "Select a run",
                        style    = MaterialTheme.typography.bodyMedium,
                        color    = if (selected != null) MaterialTheme.colorScheme.onSurface
                                   else MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f)
                    )
                    Icon(
                        Icons.Rounded.ExpandMore,
                        contentDescription = null,
                        tint     = Color(0xFFB0B8C8),
                        modifier = Modifier.size(18.dp)
                    )
                }

                HorizontalDivider(color = Color(0xFFEAEDF4), thickness = 0.5.dp)

                // Compare row
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable(enabled = runs.size >= 2) { compareExpanded = true }
                        .padding(horizontal = 14.dp, vertical = 11.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text(
                        text  = "Compare",
                        style = MaterialTheme.typography.labelSmall,
                        color = Color(0xFFAF52DE),
                        modifier = Modifier
                            .background(Color(0xFFAF52DE).copy(alpha = 0.10f), RoundedCornerShape(6.dp))
                            .padding(horizontal = 7.dp, vertical = 3.dp)
                    )
                    Text(
                        text     = compare?.name ?: "None selected",
                        style    = MaterialTheme.typography.bodyMedium,
                        color    = if (compare != null) MaterialTheme.colorScheme.onSurface
                                   else MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.weight(1f)
                    )
                    Icon(
                        Icons.Rounded.ExpandMore,
                        contentDescription = null,
                        tint     = Color(0xFFB0B8C8),
                        modifier = Modifier.size(18.dp)
                    )
                }
            }

            // Primary dropdown
            DropdownMenu(expanded = primaryExpanded, onDismissRequest = { primaryExpanded = false }) {
                runs.take(60).forEach { run ->
                    DropdownMenuItem(
                        text = {
                            Text(run.name, maxLines = 1, overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.width(280.dp))
                        },
                        onClick = { primaryExpanded = false; onSelectRun(run.id) }
                    )
                }
            }

            // Compare dropdown
            DropdownMenu(expanded = compareExpanded, onDismissRequest = { compareExpanded = false }) {
                DropdownMenuItem(
                    text = { Text("None") },
                    onClick = { compareExpanded = false; onSelectCompareRun(null) }
                )
                runs.take(60).forEach { run ->
                    DropdownMenuItem(
                        text = {
                            Text(run.name, maxLines = 1, overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.width(280.dp))
                        },
                        onClick = { compareExpanded = false; onSelectCompareRun(run.id) }
                    )
                }
            }
        }
    }
}

@Composable
private fun FeedbackCollectionPanel(
    session: FeedbackSessionState,
    onChoose: (String) -> Unit
) {
    val current = session.currentPrompt
    if (current == null) {
        EmptyState("Feedback collection finished.")
        return
    }

    val total = session.prompts.size
    val currentNumber = session.currentIndex + 1

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.92f)),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
            shape = RoundedCornerShape(18.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Human Feedback",
                        style = MaterialTheme.typography.titleLarge,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                    Text(
                        text = "$currentNumber/$total",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier
                            .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.10f), RoundedCornerShape(8.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
                Text(
                    text = current.prompt,
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Text(
                    text = "Choose the better response. Training starts after ${total - session.chosenLabels.size} choices.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
        FeedbackChoiceCard(
            label = "A",
            text = current.answerA,
            accent = NeonBlue,
            onClick = { onChoose("A") }
        )
        FeedbackChoiceCard(
            label = "B",
            text = current.answerB,
            accent = NeonGreen,
            onClick = { onChoose("B") }
        )
    }
}

@Composable
private fun FeedbackChoiceCard(
    label: String,
    text: String,
    accent: Color,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.94f)),
        elevation = CardDefaults.cardElevation(defaultElevation = 6.dp),
        shape = RoundedCornerShape(14.dp)
    ) {
        Row(
            modifier = Modifier.padding(14.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.Top
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.titleMedium,
                color = accent,
                modifier = Modifier
                    .background(accent.copy(alpha = 0.12f), RoundedCornerShape(10.dp))
                    .padding(horizontal = 11.dp, vertical = 7.dp)
            )
            Text(
                text = text,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.weight(1f)
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun OverviewTab(snapshot: RunSnapshot?) {
    if (snapshot == null) {
        EmptyState("Select a run folder and a run to inspect training progress.")
        return
    }

    val last = snapshot.metrics.lastOrNull()
    val prev = snapshot.metrics.dropLast(1).lastOrNull()
    val evalMetrics = snapshot.metrics.filter { it.source.contains("eval", ignoreCase = true) }
    val latestEval = evalMetrics.lastOrNull()
    val prevEval = evalMetrics.dropLast(1).lastOrNull()
    val displayMetric = latestEval ?: last
    val displayPrev = if (latestEval != null) prevEval else prev
    val displayLoss = latestEval?.loss ?: snapshot.summary.finalEmaLoss ?: last?.loss

    var pulse by remember(snapshot.runId) { mutableStateOf(false) }
    var seenStep by remember(snapshot.runId) { mutableIntStateOf(-1) }

    LaunchedEffect(last?.step) {
        val s = last?.step ?: return@LaunchedEffect
        if (s != seenStep) {
            seenStep = s
            pulse = true
            delay(1400)
            pulse = false
        }
    }

    val entryState = remember { MutableTransitionState(false).apply { targetState = true } }
    val hasRewardMetrics = snapshot.metrics.any {
        it.rewardAccuracy != null || it.rewardMargin != null
    }

    Box(modifier = Modifier.fillMaxSize().padding(top = 8.dp)) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            AnimatedVisibility(
                visibleState = entryState,
                enter = fadeIn() + slideInVertically { it / 8 }
            ) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f)
                    ),
                    elevation = CardDefaults.cardElevation(defaultElevation = 9.dp),
                    shape = RoundedCornerShape(22.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier
                            .background(
                                brush = Brush.linearGradient(
                                    listOf(
                                        Color(0xFFEFF6FF),
                                        Color(0xFFFFFFFF),
                                        Color(0xFFF2F2F7)
                                    )
                                )
                            )
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = snapshot.runName,
                                style = MaterialTheme.typography.headlineSmall,
                                color = MaterialTheme.colorScheme.onSurface,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                if (hasRewardMetrics) {
                                    FeedbackModePill()
                                }
                                StatusPill(snapshot.status)
                            }
                        }

                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = if (snapshot.status == RunStatus.RUNNING) "Realtime stream" else "Snapshot mode",
                                style = MaterialTheme.typography.labelLarge,
                                color = MaterialTheme.colorScheme.primary
                            )
                            if (hasRewardMetrics) {
                                Text(
                                    text = "•",
                                    style = MaterialTheme.typography.labelLarge,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Text(
                                    text = "DPO reward metrics",
                                    style = MaterialTheme.typography.labelLarge,
                                    color = RssColor
                                )
                            }
                            Text(
                                text = "•",
                                style = MaterialTheme.typography.labelLarge,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                            Text(
                                text = if (snapshot.metrics.isEmpty() && snapshot.status == RunStatus.RUNNING) {
                                    inferLiveStage(snapshot)
                                } else {
                                    "${snapshot.metrics.size} steps parsed"
                                },
                                style = MaterialTheme.typography.labelLarge,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        Text(
                            text = "Last update ${formatTime(snapshot.updatedAtMs)}",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )

                        val total = snapshot.summary.totalSteps ?: last?.totalSteps
                        val stepText = if (total != null && last != null) {
                            "Step ${last.step} / $total"
                        } else if (last != null) {
                            "Step ${last.step}"
                        } else if (snapshot.status == RunStatus.RUNNING) {
                            val totalText = total?.let { " / $it" } ?: ""
                            "Waiting for first step$totalText"
                        } else {
                            "No training step parsed"
                        }

                        val progress = if (total != null && total > 0 && last != null) {
                            (last.step.toFloat() / total.toFloat()).coerceIn(0f, 1f)
                        } else {
                            0f
                        }

                        // Circular progress ring
                        val primaryColor = MaterialTheme.colorScheme.primary
                        val surfaceColor = MaterialTheme.colorScheme.onSurface
                        Box(
                            modifier = Modifier.fillMaxWidth(),
                            contentAlignment = Alignment.Center
                        ) {
                            Canvas(modifier = Modifier.size(118.dp)) {
                                val sw = 13.dp.toPx()
                                // Track circle
                                drawArc(
                                    color      = surfaceColor.copy(alpha = 0.12f),
                                    startAngle = -90f,
                                    sweepAngle = 360f,
                                    useCenter  = false,
                                    style      = Stroke(width = sw, cap = StrokeCap.Round)
                                )
                                // Progress arc
                                if (progress > 0f) {
                                    drawArc(
                                        brush      = Brush.sweepGradient(
                                            listOf(
                                                primaryColor.copy(alpha = 0.5f),
                                                primaryColor
                                            )
                                        ),
                                        startAngle = -90f,
                                        sweepAngle = progress * 360f,
                                        useCenter  = false,
                                        style      = Stroke(width = sw, cap = StrokeCap.Round)
                                    )
                                }
                            }
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text(
                                    text  = "${(progress * 100).toInt()}%",
                                    style = MaterialTheme.typography.headlineSmall,
                                    color = MaterialTheme.colorScheme.primary,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(Modifier.height(2.dp))
                                Text(
                                    text  = stepText,
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                }
            }
        }

        if (snapshot.status == RunStatus.RUNNING || snapshot.metrics.size < 2) {
            item {
                LiveRunStatusPanel(snapshot = snapshot, title = "Live Run State")
            }
        }

        item {
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 2
            ) {
                val metricsRss = snapshot.metrics.mapNotNull { it.rssMb }
                MetricCard(
                    label = if (latestEval != null) "Eval Loss" else "Loss",
                    value = displayLoss?.let { format3(it) } ?: "N/A",
                    deltaText = if (latestEval != null) {
                        deltaText(latestEval.loss, prevEval?.loss, lowerIsBetter = true)
                    } else {
                        deltaText(last?.loss, prev?.loss, lowerIsBetter = true)
                    },
                    accent = LossColor,
                    modifier = Modifier.weight(1f),
                    isLiveHighlight = pulse,
                    sparklineValues = if (latestEval != null) {
                        evalMetrics.takeLast(30).mapNotNull { it.loss }
                    } else {
                        snapshot.metrics.takeLast(30).mapNotNull { it.loss }
                    }
                )
                MetricCard(
                    label = if (hasRewardMetrics && latestEval != null) {
                        "Eval Reward Acc"
                    } else if (hasRewardMetrics) {
                        "Reward Acc"
                    } else {
                        "PPL"
                    },
                    value = if (hasRewardMetrics) {
                        displayMetric?.rewardAccuracy?.let { formatPct(it) } ?: "N/A"
                    } else {
                        last?.ppl?.let { format2(it) } ?: "N/A"
                    },
                    deltaText = if (hasRewardMetrics) {
                        deltaText(displayMetric?.rewardAccuracy, displayPrev?.rewardAccuracy, lowerIsBetter = false)
                    } else {
                        deltaText(last?.ppl, prev?.ppl, lowerIsBetter = true)
                    },
                    accent = PplColor,
                    modifier = Modifier.weight(1f),
                    isLiveHighlight = pulse,
                    sparklineValues = if (hasRewardMetrics) {
                        (evalMetrics.takeIf { it.isNotEmpty() } ?: snapshot.metrics).takeLast(30).mapNotNull { it.rewardAccuracy }
                    } else {
                        snapshot.metrics.takeLast(30).mapNotNull { it.ppl }
                    }
                )
                MetricCard(
                    label = "LR",
                    value = last?.lr?.let { format6(it) } ?: "N/A",
                    deltaText = deltaText(last?.lr, prev?.lr, lowerIsBetter = false),
                    accent = LrColor,
                    modifier = Modifier.weight(1f),
                    isLiveHighlight = pulse,
                    sparklineValues = snapshot.metrics.takeLast(30).mapNotNull { it.lr }
                )
                MetricCard(
                    label = if (hasRewardMetrics && latestEval != null) {
                        "Eval Margin"
                    } else if (hasRewardMetrics) {
                        "Reward Margin"
                    } else {
                        "Peak RSS"
                    },
                    value = if (hasRewardMetrics) {
                        displayMetric?.rewardMargin?.let { format3(it) } ?: "N/A"
                    } else {
                        snapshot.summary.maxTrainRssMb?.let { "${format1(it)} MB" }
                            ?: snapshot.rssPoints.maxByOrNull { it.rssMb }?.rssMb?.let { "${format1(it)} MB" }
                            ?: metricsRss.maxOrNull()?.let { "${format1(it)} MB" }
                            ?: "N/A"
                    },
                    deltaText = if (hasRewardMetrics) {
                        deltaText(displayMetric?.rewardMargin, displayPrev?.rewardMargin, lowerIsBetter = false)
                    } else {
                        null
                    },
                    accent = RssColor,
                    modifier = Modifier.weight(1f),
                    isLiveHighlight = pulse,
                    sparklineValues = if (hasRewardMetrics) {
                        (evalMetrics.takeIf { it.isNotEmpty() } ?: snapshot.metrics).takeLast(30).mapNotNull { it.rewardMargin }
                    } else {
                        snapshot.rssPoints.takeLast(30).map { it.rssMb }.ifEmpty {
                            snapshot.metrics.takeLast(30).mapNotNull { it.rssMb }
                        }
                    }
                )
            }
        }

        if (snapshot.inferenceRecords.isNotEmpty()) {
            item {
                InferenceFeedbackPanel(records = snapshot.inferenceRecords)
            }
        }

        item {
            MemoryWaterfallChart(
                modifier = Modifier.fillMaxWidth(),
                metrics = snapshot.metrics,
                rssPoints = snapshot.rssPoints
            )
        }

        item {
            EventTimeline(snapshot = snapshot)
        }
    }
    } // end Box
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun InferenceFeedbackPanel(records: List<InferenceRecord>) {
    val avgLatencyMs = records.mapNotNull { it.elapsedMs }.takeIf { it.isNotEmpty() }?.average()
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.95f), RoundedCornerShape(18.dp))
            .border(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.16f), RoundedCornerShape(18.dp))
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text("Inference Feedback", style = MaterialTheme.typography.titleLarge)
                Text(
                    text = "Qwen3 generation converted into DPO preference pairs.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
            Text(
                text = "${records.size} pairs",
                style = MaterialTheme.typography.labelLarge,
                color = RssColor,
                modifier = Modifier
                    .background(RssColor.copy(alpha = 0.12f), RoundedCornerShape(8.dp))
                    .padding(horizontal = 8.dp, vertical = 5.dp)
            )
        }

        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
            maxItemsInEachRow = 3
        ) {
            MiniMetricChip(
                label = "Pairs",
                value = records.size.toString(),
                accent = NeonBlue,
                modifier = Modifier.weight(1f)
            )
            MiniMetricChip(
                label = "Avg Infer",
                value = avgLatencyMs?.let { "${format1(it / 1000.0)}s" } ?: "N/A",
                accent = NeonAmber,
                modifier = Modifier.weight(1f)
            )
            MiniMetricChip(
                label = "Source",
                value = records.firstOrNull()?.source?.take(18) ?: "N/A",
                accent = NeonGreen,
                modifier = Modifier.weight(1f)
            )
        }

        records.take(4).forEachIndexed { index, record ->
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.36f), RoundedCornerShape(14.dp))
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Prompt ${index + 1}: ${record.prompt}",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurface,
                    maxLines = 3,
                    overflow = TextOverflow.Ellipsis
                )
                record.completion?.takeIf { it.isNotBlank() }?.let {
                    LabeledTextBlock(label = "Generated", value = it, accent = NeonBlue)
                }
                record.chosen?.takeIf { it.isNotBlank() }?.let {
                    LabeledTextBlock(label = "Chosen", value = it, accent = NeonGreen)
                }
                record.chosenSource?.takeIf { it.isNotBlank() }?.let {
                    Text(
                        text = "Chosen source: $it",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
                record.rejected?.takeIf { it.isNotBlank() }?.let {
                    LabeledTextBlock(label = "Rejected", value = it, accent = LossColor)
                }
            }
        }
    }
}

@Composable
private fun LabeledTextBlock(label: String, value: String, accent: Color) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelLarge,
            color = accent
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
            maxLines = 4,
            overflow = TextOverflow.Ellipsis,
            modifier = Modifier
                .fillMaxWidth()
                .background(Color.White.copy(alpha = 0.52f), RoundedCornerShape(10.dp))
                .padding(10.dp)
        )
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun LiveRunStatusPanel(
    snapshot: RunSnapshot,
    title: String
) {
    val last = snapshot.metrics.lastOrNull()
    val stage = inferLiveStage(snapshot)
    val recentLogs = snapshot.logTail.filter { it.isNotBlank() }.takeLast(10)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.94f), RoundedCornerShape(18.dp))
            .border(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.18f), RoundedCornerShape(18.dp))
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(title, style = MaterialTheme.typography.titleLarge)
                Text(
                    text = stage,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
            StatusPill(snapshot.status)
        }

        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
            maxItemsInEachRow = 3
        ) {
            MiniMetricChip(
                label = "Step",
                value = last?.let { "${it.step}/${it.totalSteps ?: "?"}" } ?: "loading",
                accent = LrColor,
                modifier = Modifier.weight(1f)
            )
            MiniMetricChip(
                label = "Loss",
                value = last?.loss?.let { format3(it) } ?: "pending",
                accent = LossColor,
                modifier = Modifier.weight(1f)
            )
            MiniMetricChip(
                label = "RSS",
                value = last?.rssMb?.let { "${format1(it)} MB" } ?: "pending",
                accent = RssColor,
                modifier = Modifier.weight(1f)
            )
            MiniMetricChip(
                label = "Reward Acc",
                value = last?.rewardAccuracy?.let { formatPct(it) } ?: "pending",
                accent = PplColor,
                modifier = Modifier.weight(1f)
            )
            MiniMetricChip(
                label = "Margin",
                value = last?.rewardMargin?.let { format3(it) } ?: "pending",
                accent = NeonBlue,
                modifier = Modifier.weight(1f)
            )
            MiniMetricChip(
                label = "Log Lines",
                value = snapshot.logTail.size.toString(),
                accent = NeonAmber,
                modifier = Modifier.weight(1f)
            )
        }

        if (recentLogs.isNotEmpty()) {
            TerminalLogView(
                logs = recentLogs,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 120.dp, max = 240.dp)
            )
        }
    }
}

private fun inferLiveStage(snapshot: RunSnapshot): String {
    val lines = snapshot.logTail.asReversed()
    val stageLine = lines.firstOrNull { line ->
        val lower = line.lowercase()
        Regex("""\[[1-6]/6\]""").containsMatchIn(line) ||
            "[train]" in lower ||
            "[eval]" in lower ||
            "[stopped]" in lower ||
            "training complete" in lower ||
            "exception" in lower ||
            "error" in lower
    }
    return stageLine ?: when (snapshot.status) {
        RunStatus.RUNNING -> "Native process is starting; waiting for the first log line."
        RunStatus.STOPPED -> "Run stopped by user."
        RunStatus.COMPLETED -> "Run completed."
        RunStatus.FAILED -> "Run failed."
        RunStatus.IDLE -> "No active training stage parsed yet."
    }
}

@Composable
private fun FeedbackModePill(modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .clip(RoundedCornerShape(50))
            .background(RssColor.copy(alpha = 0.14f))
            .padding(horizontal = 10.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .clip(CircleShape)
                .background(RssColor)
                .padding(4.dp)
        )
        Text(
            text = "HFRL",
            style = MaterialTheme.typography.labelLarge,
            color = RssColor
        )
    }
}

@Composable
private fun EventTimeline(snapshot: RunSnapshot) {
    val events = snapshot.events.takeLast(15).reversed()

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                Brush.verticalGradient(
                    listOf(
                        MaterialTheme.colorScheme.surface.copy(alpha = 0.95f),
                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.88f)
                    )
                ),
                RoundedCornerShape(18.dp)
            )
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(0.dp)
    ) {
        Text(
            "Event Timeline",
            style = MaterialTheme.typography.titleLarge,
            color = MaterialTheme.colorScheme.onSurface
        )
        Spacer(Modifier.height(12.dp))

        if (events.isEmpty()) {
            Text(
                "No events parsed yet.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        } else {
            events.forEachIndexed { idx, event ->
                val color   = eventTypeColor(event.type)
                val typeTag = event.type.name
                val isLast  = idx == events.lastIndex

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    // Left: dot + vertical stem
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.width(12.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(10.dp)
                                .clip(CircleShape)
                                .background(color)
                        )
                        if (!isLast) {
                            Box(
                                modifier = Modifier
                                    .width(2.dp)
                                    .height(42.dp)
                                    .background(
                                        Brush.verticalGradient(
                                            listOf(
                                                color.copy(alpha = 0.45f),
                                                color.copy(alpha = 0.04f)
                                            )
                                        )
                                    )
                            )
                        }
                    }

                    // Right: type badge + step + message
                    Column(
                        verticalArrangement = Arrangement.spacedBy(3.dp),
                        modifier = Modifier.padding(bottom = if (isLast) 0.dp else 20.dp)
                    ) {
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text     = typeTag,
                                style    = MaterialTheme.typography.labelLarge,
                                color    = color,
                                modifier = Modifier
                                    .background(color.copy(alpha = 0.15f), RoundedCornerShape(6.dp))
                                    .padding(horizontal = 6.dp, vertical = 2.dp)
                            )
                            if (event.step != null) {
                                Text(
                                    text  = "step ${event.step}",
                                    style = MaterialTheme.typography.labelLarge,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                        Text(
                            text     = event.message,
                            style    = MaterialTheme.typography.bodyMedium,
                            color    = MaterialTheme.colorScheme.onSurface,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun TrainingTab(snapshot: RunSnapshot?) {
    if (snapshot == null) {
        EmptyState("No run snapshot to draw chart.")
        return
    }

    val metrics = snapshot.metrics
    val last = metrics.lastOrNull()
    val lossSeries = metrics.mapNotNull { it.loss }
    val pplSeries = metrics.mapNotNull { it.ppl }
    val lrSeries = metrics.mapNotNull { it.lr }
    val rewardAccSeries = metrics.mapNotNull { it.rewardAccuracy }
    val rewardMarginSeries = metrics.mapNotNull { it.rewardMargin }
    val rssSeries = snapshot.rssPoints.map { it.rssMb }.ifEmpty {
        metrics.mapNotNull { it.rssMb }
    }
    val lossVarianceSeries = rollingLossDeltaVariance(lossSeries, window = 10)
    val hasRewardMetrics = rewardAccSeries.isNotEmpty() || rewardMarginSeries.isNotEmpty()

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.88f)),
                elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
                shape = RoundedCornerShape(20.dp)
            ) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text("Training Telemetry", style = MaterialTheme.typography.titleLarge)
                    Text(
                        "Live training visualization with zoom, pan, and step focus.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    FlowRow(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                        maxItemsInEachRow = 2
                    ) {
                        MiniMetricChip(label = "Loss", value = last?.loss?.let { format3(it) } ?: "N/A", accent = LossColor, modifier = Modifier.weight(1f))
                        MiniMetricChip(
                            label = if (hasRewardMetrics) "Reward Acc" else "PPL",
                            value = if (hasRewardMetrics) {
                                last?.rewardAccuracy?.let { formatPct(it) } ?: "N/A"
                            } else {
                                last?.ppl?.let { format2(it) } ?: "N/A"
                            },
                            accent = PplColor,
                            modifier = Modifier.weight(1f)
                        )
                        MiniMetricChip(label = "LR",   value = last?.lr?.let { format6(it) } ?: "N/A",   accent = LrColor,   modifier = Modifier.weight(1f))
                        MiniMetricChip(
                            label    = if (hasRewardMetrics) "Margin" else "RSS",
                            value    = if (hasRewardMetrics) {
                                last?.rewardMargin?.let { format3(it) } ?: "N/A"
                            } else {
                                rssSeries.lastOrNull()?.let { "${format1(it)} MB" } ?: "N/A"
                            },
                            accent   = RssColor,
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
        }

        if (snapshot.status == RunStatus.RUNNING || metrics.size < 2) {
            item {
                LiveRunStatusPanel(snapshot = snapshot, title = "Current Native Run")
            }
        }

        item {
            InteractiveMetricLineChart(
                title          = "Loss",
                subtitle       = "Drag horizontally to pan, tap for step focus, double-tap zoom/reset",
                values         = lossSeries,
                color          = LossColor,
                valueFormatter = { format3(it) }
            )
        }
        item {
            InteractiveMetricLineChart(
                title          = "Loss Delta Variance (Proxy)",
                subtitle       = "Lower usually means more stable convergence",
                values         = lossVarianceSeries,
                color          = NeonAmber,
                seriesLabel    = "Variance",
                valueFormatter = { format4(it) }
            )
        }
        if (hasRewardMetrics) {
            item {
                InteractiveMetricLineChart(
                    title          = "Reward Accuracy",
                    subtitle       = "Preference win rate from simulated human feedback",
                    values         = rewardAccSeries,
                    color          = PplColor,
                    valueFormatter = { formatPct(it) }
                )
            }
            item {
                InteractiveMetricLineChart(
                    title          = "Reward Margin",
                    subtitle       = "Chosen reward minus rejected reward",
                    values         = rewardMarginSeries,
                    color          = RssColor,
                    valueFormatter = { format3(it) }
                )
            }
        }
        item {
            InteractiveMetricLineChart(
                title          = "Perplexity",
                subtitle       = "Lower is better — measures model uncertainty",
                values         = pplSeries,
                color          = PplColor,
                valueFormatter = { format2(it) }
            )
        }
        item {
            InteractiveMetricLineChart(
                title          = "Learning Rate",
                subtitle       = "Warmup + decay schedule",
                values         = lrSeries,
                color          = LrColor,
                valueFormatter = { format6(it) }
            )
        }
        item {
            InteractiveMetricLineChart(
                title          = "RSS Memory (MB)",
                subtitle       = "Peak resident set size over training",
                values         = rssSeries,
                color          = RssColor,
                valueFormatter = { format1(it) }
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ComparisonTab(
    primary: RunSnapshot?,
    compare: RunSnapshot?,
    compareName: String?
) {
    if (primary == null) {
        EmptyState("Select both Primary and Compare runs to see side-by-side analytics.")
        return
    }

    if (compare == null) {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            item {
                LiveRunStatusPanel(snapshot = primary, title = "Primary Run")
            }
            item {
                Text(
                    text = "Set a Compare run in Runs when you want side-by-side charts.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(horizontal = 4.dp)
                )
            }
        }
        return
    }

    val (lossA, lossB) = alignSeries(primary.metrics, compare.metrics) { it.loss }
    val (pplA, pplB) = alignSeries(primary.metrics, compare.metrics) { it.ppl }
    val rssA = primary.rssPoints.map { it.rssMb }
    val rssB = compare.rssPoints.map { it.rssMb }

    val finalLossA = lossA.lastOrNull()
    val finalLossB = lossB.lastOrNull()
    val finalPplA = pplA.lastOrNull()
    val finalPplB = pplB.lastOrNull()
    val peakRssA = primary.summary.maxTrainRssMb ?: rssA.maxOrNull()
    val peakRssB = compare.summary.maxTrainRssMb ?: rssB.maxOrNull()
    val primaryLeadCount = listOfNotNull(
        finalLossA?.let { a -> finalLossB?.let { b -> a < b } },
        finalPplA?.let { a -> finalPplB?.let { b -> a < b } },
        peakRssA?.let { a -> peakRssB?.let { b -> a < b } }
    ).count { it }
    val overallHint = if (primaryLeadCount >= 2) "Primary is leading on most core metrics" else "Compare is currently stronger"

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f)),
                elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
                shape = RoundedCornerShape(20.dp)
            ) {
                Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Experiment Comparison", style = MaterialTheme.typography.headlineSmall)
                    Text("Primary: ${primary.runName}", style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text("Compare: ${compareName ?: compare.runName}", style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                    // Winner banner
                    val winnerColor = if (primaryLeadCount >= 2) NeonGreen else NeonAmber
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(winnerColor.copy(alpha = 0.12f), RoundedCornerShape(10.dp))
                            .padding(horizontal = 10.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text("●", color = winnerColor, style = MaterialTheme.typography.labelLarge)
                        Text(
                            text  = overallHint,
                            style = MaterialTheme.typography.labelLarge,
                            color = winnerColor
                        )
                    }
                }
            }
        }

        item {
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 2
            ) {
                MetricCard(
                    label = "Final Loss Δ",
                    value = diffText(finalLossA, finalLossB, true),
                    accent = LossColor,
                    modifier = Modifier.weight(1f)
                )
                MetricCard(
                    label = "Final PPL Δ",
                    value = diffText(finalPplA, finalPplB, true),
                    accent = PplColor,
                    modifier = Modifier.weight(1f)
                )
                MetricCard(
                    label = "Peak RSS Δ",
                    value = diffText(peakRssA, peakRssB, true, suffix = " MB"),
                    accent = RssColor,
                    modifier = Modifier.weight(1f)
                )
                MetricCard(
                    label = "Common Steps",
                    value = minOf(lossA.size, lossB.size).toString(),
                    accent = LrColor,
                    modifier = Modifier.weight(1f)
                )
            }
        }

        item {
            InteractiveComparisonChart(
                title          = "Loss Overlay",
                subtitle       = "Primary vs Compare (aligned by step)",
                baselineName   = "Primary",
                baselineValues = lossA,
                candidateName  = "Compare",
                candidateValues = lossB,
                baselineColor  = LrColor,
                candidateColor = LossColor,
                valueFormatter = { format3(it) }
            )
        }

        item {
            InteractiveComparisonChart(
                title          = "PPL Overlay",
                subtitle       = "Primary vs Compare",
                baselineName   = "Primary",
                baselineValues = pplA,
                candidateName  = "Compare",
                candidateValues = pplB,
                baselineColor  = LrColor,
                candidateColor = PplColor,
                valueFormatter = { format2(it) }
            )
        }

        item {
            InteractiveComparisonChart(
                title          = "RSS Memory Overlay",
                subtitle       = "Memory pressure profile — lower is better",
                baselineName   = "Primary",
                baselineValues = rssA,
                candidateName  = "Compare",
                candidateValues = rssB,
                baselineColor  = LrColor,
                candidateColor = RssColor,
                valueFormatter = { format1(it) }
            )
        }
    }
}

@Composable
private fun LogsTab(snapshot: RunSnapshot?) {
    if (snapshot == null) {
        EmptyState("No logs loaded.")
        return
    }

    Column(modifier = Modifier.fillMaxSize().padding(12.dp)) {
        TerminalLogView(logs = snapshot.logTail, modifier = Modifier.fillMaxSize())
    }
}

@Composable
private fun MiniMetricChip(
    modifier: Modifier = Modifier,
    label: String,
    value: String,
    accent: Color = MaterialTheme.colorScheme.onSurfaceVariant
) {
    Column(
        modifier = modifier
            .heightIn(min = 78.dp)
            .background(accent.copy(alpha = 0.12f), RoundedCornerShape(14.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp)
    ) {
        Text(
            text  = label,
            style = MaterialTheme.typography.labelLarge,
            color = accent.copy(alpha = 0.80f)
        )
        Text(
            text  = value,
            style = MaterialTheme.typography.titleMedium,
            color = accent,
            maxLines = 3,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
private fun ExperimentsTab(
    runs: List<RunHandle>,
    selectedRunId: String?,
    compareRunId: String?,
    config: NativeDemoConfig,
    runner: NativeDemoRunnerState,
    inference: NativeInferenceState,
    snapshot: RunSnapshot?,
    feedbackSession: FeedbackSessionState?,
    onPickDirectory: () -> Unit,
    onEnableDemo: () -> Unit,
    onEnableDpoTest: () -> Unit,
    onChooseFeedback: (String) -> Unit,
    onInferenceChange: (NativeInferenceState) -> Unit,
    onGenerateSingle: () -> Unit,
    onGeneratePair: () -> Unit,
    onChooseCandidate: (String) -> Unit,
    onChooseRating: (Boolean) -> Unit,
    onConfigChange: (NativeDemoConfig) -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit
) {
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
        onGenerateSingle = onGenerateSingle,
        onGeneratePair = onGeneratePair,
        onChooseCandidate = onChooseCandidate,
        onChooseRating = onChooseRating,
        onConfigChange = onConfigChange,
        onStart = onStart,
        onStop = onStop
    )
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PaperDemoConsole(
    runs: List<RunHandle>,
    selectedRunId: String?,
    compareRunId: String?,
    config: NativeDemoConfig,
    runner: NativeDemoRunnerState,
    inference: NativeInferenceState,
    snapshot: RunSnapshot?,
    feedbackSession: FeedbackSessionState?,
    onPickDirectory: () -> Unit,
    onEnableDemo: () -> Unit,
    onEnableDpoTest: () -> Unit,
    onChooseFeedback: (String) -> Unit,
    onInferenceChange: (NativeInferenceState) -> Unit,
    onGenerateSingle: () -> Unit,
    onGeneratePair: () -> Unit,
    onChooseCandidate: (String) -> Unit,
    onChooseRating: (Boolean) -> Unit,
    onConfigChange: (NativeDemoConfig) -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit
) {
    val workflowSteps = buildDemoWorkflowSteps(inference, runner, snapshot)
    val summary = buildPaperDemoSummaryState(config, runner, snapshot)
    val input = buildDemoInputState(inference, runner)
    val evidence = buildDemoEvidenceState(config, runner, snapshot)
    val selectedRun = runs.firstOrNull { it.id == selectedRunId }?.name ?: "No run selected"
    val compareRun = runs.firstOrNull { it.id == compareRunId }?.name ?: "None"

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        item {
            PaperConsoleHeader(
                summary = summary,
                selectedRun = selectedRun,
                compareRun = compareRun,
                runner = runner,
                onPickDirectory = onPickDirectory,
                onEnableDemo = onEnableDemo,
                onEnableDpoTest = onEnableDpoTest
            )
        }
        item {
            WorkflowStrip(steps = workflowSteps)
        }
        if (feedbackSession != null && !feedbackSession.isComplete) {
            item {
                FeedbackCollectionPanel(
                    session = feedbackSession,
                    onChoose = onChooseFeedback
                )
            }
        }
        item {
            PaperGeneratePreferPanel(
                state = inference,
                runner = runner,
                input = input,
                onStateChange = onInferenceChange,
                onGenerateSingle = onGenerateSingle,
                onGeneratePair = onGeneratePair,
                onChooseCandidate = onChooseCandidate,
                onChooseRating = onChooseRating
            )
        }
        item {
            PaperKtoTrainingPanel(
                config = config,
                runner = runner,
                snapshot = snapshot,
                onConfigChange = onConfigChange,
                onStart = onStart,
                onStop = onStop
            )
        }
        item {
            PaperLiveProgressPanel(snapshot = snapshot)
        }
        item {
            PaperEvidencePanel(evidence = evidence)
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PaperConsoleHeader(
    summary: PaperDemoSummaryState,
    selectedRun: String,
    compareRun: String,
    runner: NativeDemoRunnerState,
    onPickDirectory: () -> Unit,
    onEnableDemo: () -> Unit,
    onEnableDpoTest: () -> Unit
) {
    val running = runner.status == NativeDemoRunStatus.RUNNING || runner.status == NativeDemoRunStatus.STOPPING
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        color = Color.White,
        tonalElevation = 1.dp
    ) {
        Column(
            modifier = Modifier
                .border(1.dp, Color(0xFFE1E5EC), RoundedCornerShape(8.dp))
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(summary.title, style = MaterialTheme.typography.titleLarge)
                    Text(
                        summary.subtitle,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Text(
                    text = runner.status.name,
                    style = MaterialTheme.typography.labelLarge,
                    color = nativeRunnerColor(runner.status),
                    modifier = Modifier
                        .background(nativeRunnerColor(runner.status).copy(alpha = 0.10f), RoundedCornerShape(6.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                )
            }
            InlineStatus(
                message = "${summary.readinessLabel}: ${summary.readinessDetail}",
                accent = when (summary.readinessLabel) {
                    "Demo ready" -> NeonGreen
                    "Training" -> NeonBlue
                    "Ready to train" -> NeonAmber
                    else -> RssColor
                }
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 4
            ) {
                MiniMetricChip(
                    label = "Data",
                    value = summary.dataStatus,
                    accent = NeonGreen,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "Training",
                    value = summary.trainingStatus,
                    accent = NeonBlue,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "Peak RSS",
                    value = summary.peakMemory,
                    accent = RssColor,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "LoRA",
                    value = summary.adapterStatus,
                    accent = LrColor,
                    modifier = Modifier.weight(1f)
                )
            }
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3
            ) {
                Button(
                    onClick = onPickDirectory,
                    enabled = !running,
                    modifier = Modifier.weight(1f).heightIn(min = 44.dp)
                ) {
                    Icon(Icons.Rounded.FolderOpen, contentDescription = null)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Runs", maxLines = 1)
                }
                OutlinedButton(
                    onClick = onEnableDemo,
                    enabled = !running,
                    modifier = Modifier.weight(1f).heightIn(min = 44.dp)
                ) {
                    Text("Load Demo", maxLines = 1)
                }
                OutlinedButton(
                    onClick = onEnableDpoTest,
                    enabled = !running,
                    modifier = Modifier.weight(1f).heightIn(min = 44.dp)
                ) {
                    Text("DPO Data", maxLines = 1)
                }
            }
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 2
            ) {
                MiniMetricChip(
                    label = "Selected Run",
                    value = selectedRun,
                    accent = NeonBlue,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "Compare",
                    value = compareRun,
                    accent = RssColor,
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

@Composable
private fun WorkflowStrip(steps: List<DemoWorkflowStepState>) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        color = Color.White
    ) {
        Row(
            modifier = Modifier
                .border(1.dp, Color(0xFFE1E5EC), RoundedCornerShape(8.dp))
                .padding(10.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            steps.forEachIndexed { index, step ->
                WorkflowStepPill(
                    index = index + 1,
                    step = step,
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

@Composable
private fun WorkflowStepPill(
    index: Int,
    step: DemoWorkflowStepState,
    modifier: Modifier = Modifier
) {
    val accent = workflowStatusColor(step.status)
    Column(
        modifier = modifier
            .background(accent.copy(alpha = 0.08f), RoundedCornerShape(8.dp))
            .border(1.dp, accent.copy(alpha = 0.24f), RoundedCornerShape(8.dp))
            .padding(horizontal = 8.dp, vertical = 7.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp)
    ) {
        Text(
            text = "$index. ${step.label}",
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurface,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
        Text(
            text = workflowStatusLabel(step.status),
            style = MaterialTheme.typography.labelSmall,
            color = accent,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PaperGeneratePreferPanel(
    state: NativeInferenceState,
    runner: NativeDemoRunnerState,
    input: DemoInputState,
    onStateChange: (NativeInferenceState) -> Unit,
    onGenerateSingle: () -> Unit,
    onGeneratePair: () -> Unit,
    onChooseCandidate: (String) -> Unit,
    onChooseRating: (Boolean) -> Unit
) {
    val mode = buildFeedbackModeState(state)
    val isPairMode = state.feedbackMode == NativeFeedbackMode.PAIR
    val feedbackRows = displayFeedbackRows(state, runner)
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        color = Color.White
    ) {
        Column(
            modifier = Modifier
                .border(1.dp, Color(0xFFE1E5EC), RoundedCornerShape(8.dp))
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            SectionHeader(
                title = input.title,
                subtitle = input.subtitle
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3
            ) {
                MiniMetricChip(
                    label = "Input",
                    value = input.formatLabel,
                    accent = NeonBlue,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "KTO Rows",
                    value = input.rowLabel,
                    accent = if (feedbackRows > 0) NeonGreen else NeonAmber,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "Status",
                    value = input.statusLabel,
                    accent = if (feedbackRows > 0) NeonGreen else NeonAmber,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "Model",
                    value = if (state.useFedCampusBridge) "FedCampus bridge" else "Qwen3-0.6B",
                    accent = RssColor,
                    modifier = Modifier.weight(1f)
                )
            }
            InlineStatus(
                message = "Accepted formats: DPO chosen/rejected, KTO binary response/label, or app thumbs.",
                accent = NeonBlue
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 2
            ) {
                FeedbackModeButton(
                    label = "A/B Compare",
                    selected = state.feedbackMode == NativeFeedbackMode.PAIR,
                    enabled = !state.isRunning,
                    modifier = Modifier.weight(1f)
                ) {
                    onStateChange(
                        state.copy(
                            feedbackMode = NativeFeedbackMode.PAIR,
                            output = null,
                            lastRatingLabel = null,
                            feedbackStatus = null
                        )
                    )
                }
                FeedbackModeButton(
                    label = "Single Rating",
                    selected = state.feedbackMode == NativeFeedbackMode.SINGLE,
                    enabled = !state.isRunning,
                    modifier = Modifier.weight(1f)
                ) {
                    onStateChange(
                        state.copy(
                            feedbackMode = NativeFeedbackMode.SINGLE,
                            candidateA = null,
                            candidateB = null,
                            lastChosenLabel = null,
                            feedbackStatus = null
                        )
                    )
                }
            }
            ToggleRow(
                label = "Use FedCampus bridge",
                checked = state.useFedCampusBridge,
                enabled = !state.isRunning,
                modifier = Modifier.fillMaxWidth()
            ) {
                onStateChange(state.copy(useFedCampusBridge = it))
            }
            if (state.useFedCampusBridge) {
                OutlinedTextField(
                    value = state.bridgeUrl,
                    onValueChange = { onStateChange(state.copy(bridgeUrl = it)) },
                    enabled = !state.isRunning,
                    label = { Text("Bridge URL") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
            } else {
                OutlinedTextField(
                    value = state.modelDir,
                    onValueChange = { onStateChange(state.copy(modelDir = it)) },
                    enabled = !state.isRunning,
                    label = { Text("Model Dir") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
            }
            OutlinedTextField(
                value = state.prompt,
                onValueChange = { onStateChange(state.copy(prompt = it)) },
                enabled = !state.isRunning,
                label = { Text("Health prompt") },
                minLines = 3,
                maxLines = 5,
                modifier = Modifier.fillMaxWidth()
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3
            ) {
                IntField("Max Tokens", state.maxNewTokens, !state.isRunning, Modifier.weight(1f)) {
                    onStateChange(state.copy(maxNewTokens = it))
                }
                DecimalField("Temp", state.temperature, !state.isRunning, Modifier.weight(1f)) {
                    onStateChange(state.copy(temperature = it))
                }
                Button(
                    onClick = if (isPairMode) onGeneratePair else onGenerateSingle,
                    enabled = !state.isRunning &&
                        state.prompt.isNotBlank() &&
                        (state.useFedCampusBridge || state.modelDir.isNotBlank()),
                    modifier = Modifier.weight(1f).heightIn(min = 56.dp)
                ) {
                    Icon(Icons.Rounded.PlayArrow, contentDescription = null)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(mode.actionLabel, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }
            if (state.isRunning) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp), strokeWidth = 3.dp)
                    Text("Generating candidates...", style = MaterialTheme.typography.bodyMedium)
                }
            }
            state.feedbackStatus?.takeIf { it.isNotBlank() }?.let {
                InlineStatus(message = it, accent = NeonGreen)
            }
            state.error?.takeIf { it.isNotBlank() }?.let {
                InlineStatus(message = it, accent = MaterialTheme.colorScheme.error)
            }
            if (!isPairMode) {
                state.output?.takeIf { it.isNotBlank() }?.let { output ->
                    SingleRatingBlock(
                        text = output,
                        upSelected = state.lastRatingLabel == "UP",
                        downSelected = state.lastRatingLabel == "DOWN",
                        enabled = state.lastRatingLabel == null && !state.isRunning,
                        onThumbUp = { onChooseRating(true) },
                        onThumbDown = { onChooseRating(false) }
                    )
                }
            }
            val a = state.candidateA
            val b = state.candidateB
            if (isPairMode && (!a.isNullOrBlank() || !b.isNullOrBlank())) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                    maxItemsInEachRow = 2
                ) {
                    CandidatePreferenceBlock(
                        label = "A",
                        text = a ?: "",
                        enabled = state.lastChosenLabel == null && !state.isRunning && !a.isNullOrBlank() && !b.isNullOrBlank(),
                        selected = state.lastChosenLabel == "A",
                        modifier = Modifier.weight(1f),
                        onChoose = { onChooseCandidate("A") }
                    )
                    CandidatePreferenceBlock(
                        label = "B",
                        text = b ?: "",
                        enabled = state.lastChosenLabel == null && !state.isRunning && !a.isNullOrBlank() && !b.isNullOrBlank(),
                        selected = state.lastChosenLabel == "B",
                        modifier = Modifier.weight(1f),
                        onChoose = { onChooseCandidate("B") }
                    )
                }
            }
            state.feedbackJsonlPath?.takeIf { it.isNotBlank() }?.let {
                Text(
                    text = "Feedback JSONL: $it",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
private fun FeedbackModeButton(
    label: String,
    selected: Boolean,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    if (selected) {
        Button(
            onClick = onClick,
            enabled = enabled,
            modifier = modifier.heightIn(min = 44.dp)
        ) {
            Text(label, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
    } else {
        OutlinedButton(
            onClick = onClick,
            enabled = enabled,
            modifier = modifier.heightIn(min = 44.dp)
        ) {
            Text(label, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
    }
}

@Composable
private fun SingleRatingBlock(
    text: String,
    upSelected: Boolean,
    downSelected: Boolean,
    enabled: Boolean,
    onThumbUp: () -> Unit,
    onThumbDown: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color.White, RoundedCornerShape(8.dp))
            .border(1.dp, Color(0xFFDDE2EC), RoundedCornerShape(8.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "Model response",
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
            maxLines = 8,
            overflow = TextOverflow.Ellipsis
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = onThumbUp,
                enabled = enabled,
                modifier = Modifier.weight(1f).heightIn(min = 44.dp)
            ) {
                Icon(Icons.Rounded.ThumbUp, contentDescription = null)
                Spacer(modifier = Modifier.width(6.dp))
                Text(if (upSelected) "Saved Up" else "Good")
            }
            OutlinedButton(
                onClick = onThumbDown,
                enabled = enabled,
                modifier = Modifier.weight(1f).heightIn(min = 44.dp)
            ) {
                Icon(Icons.Rounded.ThumbDown, contentDescription = null)
                Spacer(modifier = Modifier.width(6.dp))
                Text(if (downSelected) "Saved Down" else "Bad")
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PaperKtoTrainingPanel(
    config: NativeDemoConfig,
    runner: NativeDemoRunnerState,
    snapshot: RunSnapshot?,
    onConfigChange: (NativeDemoConfig) -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit
) {
    val running = runner.status == NativeDemoRunStatus.RUNNING || runner.status == NativeDemoRunStatus.STOPPING
    val last = snapshot?.metrics?.lastOrNull()
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        color = Color.White
    ) {
        Column(
            modifier = Modifier
                .border(1.dp, Color(0xFFE1E5EC), RoundedCornerShape(8.dp))
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            SectionHeader(
                title = "2. Train KTO on Android",
                subtitle = runner.message ?: "Train Qwen3-0.6B LoRA from DPO/KTO feedback on the Android device."
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3
            ) {
                MiniMetricChip(label = "Dataset", value = runner.trainingSourceLabel, accent = NeonGreen, modifier = Modifier.weight(1f))
                MiniMetricChip(label = "KTO Rows", value = if (runner.trainingPairCount > 0) runner.trainingPairCount.toString() else "Bundled", accent = NeonBlue, modifier = Modifier.weight(1f))
                MiniMetricChip(label = "Run", value = runner.currentRunName ?: "Ready", accent = RssColor, modifier = Modifier.weight(1f))
                MiniMetricChip(label = "Steps", value = "${config.steps}", accent = LrColor, modifier = Modifier.weight(1f))
                MiniMetricChip(label = "Batch", value = "2", accent = NeonAmber, modifier = Modifier.weight(1f))
                MiniMetricChip(label = "Seq Len", value = "${config.seqLen}", accent = RssColor, modifier = Modifier.weight(1f))
            }
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3
            ) {
                IntField("Steps", config.steps, !running, Modifier.weight(1f)) {
                    onConfigChange(config.copy(steps = it, presetName = "Custom"))
                }
                IntField("Rank", config.rank, !running, Modifier.weight(1f)) {
                    onConfigChange(config.copy(rank = it, presetName = "Custom"))
                }
                DecimalField("LR", config.learningRate, !running, Modifier.weight(1f)) {
                    onConfigChange(config.copy(learningRate = it, presetName = "Custom"))
                }
            }
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3
            ) {
                MiniMetricChip(label = "Loss", value = last?.loss?.let { formatDemoMetric(it) } ?: "N/A", accent = LossColor, modifier = Modifier.weight(1f))
                MiniMetricChip(label = "Reward Accuracy", value = last?.rewardAccuracy?.let { formatDemoMetric(it) } ?: "N/A", accent = NeonGreen, modifier = Modifier.weight(1f))
                MiniMetricChip(label = "Reward Margin", value = last?.rewardMargin?.let { formatDemoMetric(it) } ?: "N/A", accent = NeonBlue, modifier = Modifier.weight(1f))
            }
            runner.trainingJsonlPath?.takeIf { it.isNotBlank() }?.let {
                Text(
                    text = "Training JSONL: $it",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                Button(
                    onClick = onStart,
                    enabled = !running,
                    modifier = Modifier.weight(1f).heightIn(min = 54.dp)
                ) {
                    Icon(Icons.Rounded.PlayArrow, contentDescription = null)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Train KTO")
                }
                OutlinedButton(
                    onClick = onStop,
                    enabled = running,
                    modifier = Modifier.weight(1f).heightIn(min = 48.dp)
                ) {
                    Text("Stop")
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PaperLiveProgressPanel(snapshot: RunSnapshot?) {
    val progress = buildDemoLiveProgressState(snapshot)
    val metrics = snapshot?.metrics.orEmpty()
    val rewardMarginSeries = metrics.mapNotNull { it.rewardMargin }
    val lossSeries = metrics.mapNotNull { it.loss }
    val rssSeries = metrics.mapNotNull { it.rssMb }

    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        color = Color.White
    ) {
        Column(
            modifier = Modifier
                .border(1.dp, Color(0xFFE1E5EC), RoundedCornerShape(8.dp))
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            SectionHeader(
                title = "3. Live Evidence",
                subtitle = "Loss, reward margin, and RSS memory from Android native KTO."
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3
            ) {
                MiniMetricChip(label = "Step", value = progress.stepLabel, accent = LrColor, modifier = Modifier.weight(1f))
                MiniMetricChip(label = "Reward Margin", value = progress.rewardMargin, accent = NeonBlue, modifier = Modifier.weight(1f))
                MiniMetricChip(label = "Loss", value = progress.loss, accent = LossColor, modifier = Modifier.weight(1f))
                MiniMetricChip(label = "Peak Memory", value = progress.peakMemory, accent = RssColor, modifier = Modifier.weight(1f))
            }
            if (metrics.size >= 2) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                    maxItemsInEachRow = 2
                ) {
                    if (rewardMarginSeries.isNotEmpty()) {
                        InteractiveMetricLineChart(
                            title = "Reward Margin",
                            subtitle = "Higher means feedback alignment is improving",
                            values = rewardMarginSeries,
                            color = NeonBlue,
                            modifier = Modifier.weight(1f).heightIn(min = 190.dp)
                        )
                    }
                    if (lossSeries.isNotEmpty()) {
                        InteractiveMetricLineChart(
                            title = "Loss",
                            subtitle = "Secondary training signal",
                            values = lossSeries,
                            color = LossColor,
                            modifier = Modifier.weight(1f).heightIn(min = 190.dp)
                        )
                    }
                    if (rssSeries.isNotEmpty()) {
                        InteractiveMetricLineChart(
                            title = "Memory",
                            subtitle = "RSS MB",
                            values = rssSeries,
                            color = RssColor,
                            modifier = Modifier.weight(1f).heightIn(min = 190.dp),
                            valueFormatter = { "${format1(it)} MB" }
                        )
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PaperEvidencePanel(evidence: DemoEvidenceState) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(8.dp),
        color = Color.White
    ) {
        Column(
            modifier = Modifier
                .border(1.dp, Color(0xFFE1E5EC), RoundedCornerShape(8.dp))
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            SectionHeader(
                title = "4. Demo Evidence",
                subtitle = "Use the latest Android KTO metrics and saved LoRA adapter as the demo proof."
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3
            ) {
                PaperEvidenceMetric("Objective", evidence.objective, NeonBlue, Modifier.weight(1f))
                PaperEvidenceMetric("Model", evidence.model, RssColor, Modifier.weight(1f))
                PaperEvidenceMetric("KTO Rows", evidence.pairs, NeonGreen, Modifier.weight(1f))
                PaperEvidenceMetric("Steps", evidence.steps, LrColor, Modifier.weight(1f))
                PaperEvidenceMetric("Batch", evidence.batch, NeonAmber, Modifier.weight(1f))
                PaperEvidenceMetric("Loss", evidence.loss, LossColor, Modifier.weight(1f))
                PaperEvidenceMetric("Reward Acc.", evidence.rewardAccuracy, NeonGreen, Modifier.weight(1f))
                PaperEvidenceMetric("Reward Margin", evidence.rewardMargin, NeonBlue, Modifier.weight(1f))
                PaperEvidenceMetric("Training Trend", evidence.trendMargin, RssColor, Modifier.weight(1f))
            }
            Text(
                text = "Artifacts: ${evidence.artifacts}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
private fun SectionHeader(title: String, subtitle: String) {
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(title, style = MaterialTheme.typography.titleLarge)
        Text(
            subtitle,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun InlineStatus(message: String, accent: Color) {
    Surface(
        color = accent.copy(alpha = 0.10f),
        shape = RoundedCornerShape(8.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Text(
            text = message,
            modifier = Modifier.padding(10.dp),
            color = accent,
            style = MaterialTheme.typography.bodyMedium
        )
    }
}

@Composable
private fun PaperEvidenceMetric(
    label: String,
    value: String,
    accent: Color,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .background(Color(0xFFF8FAFD), RoundedCornerShape(8.dp))
            .border(1.dp, accent.copy(alpha = 0.20f), RoundedCornerShape(8.dp))
            .padding(horizontal = 10.dp, vertical = 9.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp)
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
        Text(
            text = value,
            style = MaterialTheme.typography.titleMedium,
            color = accent,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

private fun workflowStatusLabel(status: DemoWorkflowStatus): String = when (status) {
    DemoWorkflowStatus.READY -> "Ready"
    DemoWorkflowStatus.NEEDS_INPUT -> "Needs input"
    DemoWorkflowStatus.SAVED -> "Saved"
    DemoWorkflowStatus.RUNNING -> "Running"
    DemoWorkflowStatus.DONE -> "Done"
}

private fun workflowStatusColor(status: DemoWorkflowStatus): Color = when (status) {
    DemoWorkflowStatus.READY -> NeonBlue
    DemoWorkflowStatus.NEEDS_INPUT -> Color(0xFF8E8E93)
    DemoWorkflowStatus.SAVED -> NeonGreen
    DemoWorkflowStatus.RUNNING -> NeonAmber
    DemoWorkflowStatus.DONE -> NeonGreen
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DemoCommandCenter(
    runs: List<RunHandle>,
    selectedRunId: String?,
    compareRunId: String?,
    runner: NativeDemoRunnerState,
    inference: NativeInferenceState,
    snapshot: RunSnapshot?,
    onPickDirectory: () -> Unit,
    onEnableDemo: () -> Unit,
    onEnableDpoTest: () -> Unit
) {
    val selectedRun = runs.firstOrNull { it.id == selectedRunId }?.name ?: "No run selected"
    val compareRun = runs.firstOrNull { it.id == compareRunId }?.name ?: "None"
    val lastMetric = snapshot?.metrics?.lastOrNull()
    val isTraining = runner.status == NativeDemoRunStatus.RUNNING || runner.status == NativeDemoRunStatus.STOPPING
    val inferenceStatus = when {
        inference.isRunning -> "Generating"
        inference.savedPairs > 0 -> "${inference.savedPairs} pairs"
        inference.modelLoaded -> "Model loaded"
        else -> "Ready"
    }

    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        color = Color(0xFFF7FAFF),
        tonalElevation = 1.dp
    ) {
        Column(
            modifier = Modifier
                .border(1.dp, Color(0xFFDDE2EC), RoundedCornerShape(18.dp))
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Demo Workbench", style = MaterialTheme.typography.titleLarge)
                    Text(
                        text = selectedRun,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
                Text(
                    text = runner.status.name,
                    style = MaterialTheme.typography.labelLarge,
                    color = nativeRunnerColor(runner.status),
                    modifier = Modifier
                        .background(nativeRunnerColor(runner.status).copy(alpha = 0.12f), RoundedCornerShape(8.dp))
                        .padding(horizontal = 8.dp, vertical = 5.dp)
                )
            }

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3
            ) {
                Button(
                    onClick = onPickDirectory,
                    enabled = !isTraining,
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 48.dp)
                ) {
                    Icon(Icons.Rounded.FolderOpen, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Select Runs", maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
                OutlinedButton(
                    onClick = onEnableDemo,
                    enabled = !isTraining,
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 48.dp)
                ) {
                    Text("Load Demo", maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
                OutlinedButton(
                    onClick = onEnableDpoTest,
                    enabled = !isTraining,
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 48.dp)
                ) {
                    Text("DPO Test", maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3
            ) {
                MiniMetricChip(
                    label = "Inference",
                    value = inferenceStatus,
                    accent = if (inference.savedPairs > 0) NeonGreen else NeonBlue,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "Train Data",
                    value = runner.trainingSourceLabel,
                    accent = if (runner.trainingSourceLabel.contains("Qwen", ignoreCase = true)) NeonGreen else RssColor,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "Step",
                    value = lastMetric?.let { "${it.step}/${it.totalSteps ?: "?"}" } ?: "N/A",
                    accent = LrColor,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "Loss",
                    value = lastMetric?.loss?.let { format3(it) } ?: "N/A",
                    accent = LossColor,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "Reward",
                    value = lastMetric?.rewardAccuracy?.let { format3(it) } ?: "N/A",
                    accent = NeonGreen,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "Compare",
                    value = compareRun,
                    accent = RssColor,
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun NativeInferenceCard(
    state: NativeInferenceState,
    onStateChange: (NativeInferenceState) -> Unit,
    onGenerate: () -> Unit,
    onGeneratePair: () -> Unit,
    onChooseCandidate: (String) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFFF8FCFF)),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        shape = RoundedCornerShape(18.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Qwen Inference", style = MaterialTheme.typography.titleLarge)
                    Text(
                        text = if (state.isRunning) {
                            "Running native Qwen generate..."
                        } else {
                            "Generate answers, choose A/B, then train Qwen KTO from the saved feedback."
                        },
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                    )
                }
                if (state.isRunning) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(28.dp),
                        strokeWidth = 3.dp
                    )
                } else {
                    Text(
                        text = when {
                            state.modelLoaded -> "LOADED"
                            state.elapsedMs != null -> "${format1(state.elapsedMs)} ms"
                            else -> "READY"
                        },
                        style = MaterialTheme.typography.labelLarge,
                        color = NeonBlue,
                        modifier = Modifier
                            .background(NeonBlue.copy(alpha = 0.12f), RoundedCornerShape(8.dp))
                            .padding(horizontal = 8.dp, vertical = 5.dp)
                    )
                }
            }

            if (state.elapsedMs != null || state.modelLoaded) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                    maxItemsInEachRow = 3
                ) {
                    MiniMetricChip(
                        label = "Total",
                        value = state.elapsedMs?.let { "${format1(it)} ms" } ?: "Loaded",
                        accent = NeonBlue,
                        modifier = Modifier.weight(1f)
                    )
                    MiniMetricChip(
                        label = "Load",
                        value = state.loadElapsedMs?.let { "${format1(it)} ms" }
                            ?: if (state.modelLoaded) "Cached" else "N/A",
                        accent = RssColor,
                        modifier = Modifier.weight(1f)
                    )
                    MiniMetricChip(
                        label = "Generate",
                        value = state.generateElapsedMs?.let { "${format1(it)} ms" } ?: "N/A",
                        accent = NeonGreen,
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            OutlinedTextField(
                value = state.modelDir,
                onValueChange = { onStateChange(state.copy(modelDir = it)) },
                enabled = !state.isRunning,
                label = { Text("Model Dir") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth()
            )

            OutlinedTextField(
                value = state.prompt,
                onValueChange = { onStateChange(state.copy(prompt = it)) },
                enabled = !state.isRunning,
                label = { Text("Prompt") },
                minLines = 3,
                maxLines = 6,
                modifier = Modifier.fillMaxWidth()
            )

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 2
            ) {
                IntField("Max Tokens", state.maxNewTokens, !state.isRunning, Modifier.weight(1f)) {
                    onStateChange(state.copy(maxNewTokens = it))
                }
                DecimalField("Temp", state.temperature, !state.isRunning, Modifier.weight(1f)) {
                    onStateChange(state.copy(temperature = it))
                }
                DecimalField("Top P", state.topP, !state.isRunning, Modifier.weight(1f)) {
                    onStateChange(state.copy(topP = it))
                }
                Button(
                    onClick = onGenerate,
                    enabled = !state.isRunning && state.modelDir.isNotBlank() && state.prompt.isNotBlank(),
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 56.dp)
                ) {
                    Icon(Icons.Rounded.PlayArrow, contentDescription = null)
                    Text("Generate", modifier = Modifier.padding(start = 8.dp))
                }
                OutlinedButton(
                    onClick = onGeneratePair,
                    enabled = !state.isRunning && state.modelDir.isNotBlank() && state.prompt.isNotBlank(),
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 56.dp)
                ) {
                    Icon(Icons.Rounded.PlayArrow, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Generate A/B")
                }
            }

            if (state.savedPairs > 0 || !state.feedbackJsonlPath.isNullOrBlank()) {
                Surface(
                    color = NeonGreen.copy(alpha = 0.10f),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Text(
                            text = "Saved feedback pairs: ${state.savedPairs}",
                            style = MaterialTheme.typography.labelLarge,
                            color = NeonGreen
                        )
                        Text(
                            text = state.feedbackJsonlPath ?: "",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
            }
            state.feedbackStatus?.takeIf { it.isNotBlank() }?.let { message ->
                Surface(
                    color = NeonGreen.copy(alpha = 0.12f),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = message,
                        modifier = Modifier.padding(12.dp),
                        color = NeonGreen,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
            state.lastChosenLabel?.let {
                Text(
                    text = "Last preference: $it",
                    style = MaterialTheme.typography.labelLarge,
                    color = NeonGreen
                )
            }

            state.error?.takeIf { it.isNotBlank() }?.let { message ->
                Surface(
                    color = MaterialTheme.colorScheme.error.copy(alpha = 0.10f),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = message,
                        modifier = Modifier.padding(12.dp),
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            state.output?.takeIf { it.isNotBlank() }?.let { output ->
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color.White, RoundedCornerShape(14.dp))
                        .border(1.dp, Color(0xFFDDE2EC), RoundedCornerShape(14.dp))
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        text = "Response",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Text(
                        text = output,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            }

            if (state.debugLines.isNotEmpty()) {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(
                        text = "Inference Logs",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    InferenceDebugLog(
                        lines = state.debugLines,
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 96.dp, max = 180.dp)
                    )
                }
            }

            val a = state.candidateA
            val b = state.candidateB
            val preferenceSaved = state.lastChosenLabel != null
            if (!a.isNullOrBlank() || !b.isNullOrBlank()) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                    maxItemsInEachRow = 2
                ) {
                    CandidatePreferenceBlock(
                        label = "A",
                        text = a ?: "",
                        enabled = !preferenceSaved && !state.isRunning && !a.isNullOrBlank() && !b.isNullOrBlank(),
                        selected = state.lastChosenLabel == "A",
                        modifier = Modifier.weight(1f),
                        onChoose = { onChooseCandidate("A") }
                    )
                    CandidatePreferenceBlock(
                        label = "B",
                        text = b ?: "",
                        enabled = !preferenceSaved && !state.isRunning && !a.isNullOrBlank() && !b.isNullOrBlank(),
                        selected = state.lastChosenLabel == "B",
                        modifier = Modifier.weight(1f),
                        onChoose = { onChooseCandidate("B") }
                    )
                }
            }
        }
    }
}

@Composable
private fun InferenceDebugLog(
    lines: List<String>,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .background(Color(0xFF101820), RoundedCornerShape(12.dp))
            .border(1.dp, Color(0xFF243447), RoundedCornerShape(12.dp))
            .padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(5.dp)
    ) {
        lines.takeLast(6).forEachIndexed { index, line ->
            Text(
                text = "${index + 1}. $line",
                style = MaterialTheme.typography.bodySmall,
                color = Color(0xFFE6F1FF),
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

@Composable
private fun CandidatePreferenceBlock(
    label: String,
    text: String,
    enabled: Boolean,
    selected: Boolean,
    modifier: Modifier = Modifier,
    onChoose: () -> Unit
) {
    val borderColor = if (selected) NeonGreen else Color(0xFFDDE2EC)
    Column(
        modifier = modifier
            .background(Color.White, RoundedCornerShape(8.dp))
            .border(if (selected) 2.dp else 1.dp, borderColor, RoundedCornerShape(8.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "Answer $label",
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = text.ifBlank { "No candidate generated." },
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
            maxLines = 8,
            overflow = TextOverflow.Ellipsis
        )
        Button(
            onClick = onChoose,
            enabled = enabled,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(if (selected) "Saved $label" else "Prefer $label")
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun NativeDemoRunnerCard(
    config: NativeDemoConfig,
    runner: NativeDemoRunnerState,
    snapshot: RunSnapshot?,
    onConfigChange: (NativeDemoConfig) -> Unit,
    onPreset: (String) -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit
) {
    val running = runner.status == NativeDemoRunStatus.RUNNING || runner.status == NativeDemoRunStatus.STOPPING
    val last = snapshot?.metrics?.lastOrNull()
    var commandExpanded by remember { mutableStateOf(false) }
    val usingFeedback = runner.trainingSourceLabel.contains("Qwen", ignoreCase = true)

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFFFAFBFF)),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        shape = RoundedCornerShape(18.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Native Demo Runner", style = MaterialTheme.typography.titleLarge)
                    Text(
                        text = runner.message ?: "Configure and launch Qwen KTO training.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                    )
                }
                Text(
                    text = runner.status.name,
                    style = MaterialTheme.typography.labelLarge,
                    color = nativeRunnerColor(runner.status),
                    modifier = Modifier
                        .background(nativeRunnerColor(runner.status).copy(alpha = 0.12f), RoundedCornerShape(8.dp))
                        .padding(horizontal = 8.dp, vertical = 5.dp)
                )
            }

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                listOf("Smoke", "Phone 80", "Long 220").forEach { preset ->
                    OutlinedButton(
                        onClick = { onPreset(preset) },
                        enabled = !running
                    ) {
                        Text(preset)
                    }
                }
            }

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 2
            ) {
                IntField("Steps", config.steps, !running, Modifier.weight(1f)) {
                    onConfigChange(config.copy(steps = it, presetName = "Custom"))
                }
                IntField("Seq Len", config.seqLen, !running, Modifier.weight(1f)) {
                    onConfigChange(config.copy(seqLen = it, presetName = "Custom"))
                }
                IntField("Rank", config.rank, !running, Modifier.weight(1f)) {
                    onConfigChange(config.copy(rank = it, presetName = "Custom"))
                }
                IntField("Alpha", config.alpha, !running, Modifier.weight(1f)) {
                    onConfigChange(config.copy(alpha = it, presetName = "Custom"))
                }
                DecimalField("LR", config.learningRate, !running, Modifier.weight(1f)) {
                    onConfigChange(config.copy(learningRate = it, presetName = "Custom"))
                }
                IntField("Eval Every", config.evalInterval, !running, Modifier.weight(1f)) {
                    onConfigChange(config.copy(evalInterval = it, presetName = "Custom"))
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                ToggleRow(
                    label = "Pair forward",
                    checked = config.dpoPairForward,
                    enabled = !running,
                    modifier = Modifier.weight(1f)
                ) {
                    onConfigChange(config.copy(dpoPairForward = it, presetName = "Custom"))
                }
                ToggleRow(
                    label = "Cache ref",
                    checked = config.cacheRefLogps,
                    enabled = !running,
                    modifier = Modifier.weight(1f)
                ) {
                    onConfigChange(config.copy(cacheRefLogps = it, presetName = "Custom"))
                }
            }

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3
            ) {
                MiniMetricChip(
                    label = "Train Data",
                    value = runner.trainingSourceLabel,
                    accent = if (usingFeedback) NeonGreen else NeonBlue,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "KTO Rows",
                    value = if (runner.trainingPairCount > 0) runner.trainingPairCount.toString() else "Bundled",
                    accent = if (usingFeedback) NeonGreen else RssColor,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "Preset",
                    value = config.presetName,
                    accent = NeonBlue,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "Step",
                    value = last?.let { "${it.step}/${it.totalSteps ?: config.steps}" } ?: "N/A",
                    accent = LrColor,
                    modifier = Modifier.weight(1f)
                )
                MiniMetricChip(
                    label = "Loss",
                    value = last?.loss?.let { format3(it) } ?: "N/A",
                    accent = LossColor,
                    modifier = Modifier.weight(1f)
                )
            }

            runner.trainingJsonlPath?.takeIf { it.isNotBlank() }?.let { path ->
                Text(
                    text = "Training JSONL: $path",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Button(
                    onClick = onStart,
                    enabled = !running,
                    modifier = Modifier.weight(1f)
                ) {
                    Text(if (usingFeedback) "Train KTO" else "Run Demo")
                }
                OutlinedButton(
                    onClick = onStop,
                    enabled = running,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Stop")
                }
            }

            TextButton(onClick = { commandExpanded = !commandExpanded }) {
                Text(if (commandExpanded) "Hide Command" else "Show Command")
            }
            if (commandExpanded) {
                TerminalLogView(
                    logs = listOf(runner.commandPreview ?: "Command preview unavailable."),
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 120.dp, max = 220.dp)
                )
            }
        }
    }
}

@Composable
private fun IntField(
    label: String,
    value: Int,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    onValue: (Int) -> Unit
) {
    OutlinedTextField(
        value = value.toString(),
        onValueChange = { raw -> raw.toIntOrNull()?.let(onValue) },
        enabled = enabled,
        label = { Text(label) },
        singleLine = true,
        modifier = modifier
    )
}

@Composable
private fun DecimalField(
    label: String,
    value: Double,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    onValue: (Double) -> Unit
) {
    OutlinedTextField(
        value = value.toString(),
        onValueChange = { raw -> raw.toDoubleOrNull()?.let(onValue) },
        enabled = enabled,
        label = { Text(label) },
        singleLine = true,
        modifier = modifier
    )
}

@Composable
private fun ToggleRow(
    label: String,
    checked: Boolean,
    enabled: Boolean,
    modifier: Modifier = Modifier,
    onChecked: (Boolean) -> Unit
) {
    Row(
        modifier = modifier
            .heightIn(min = 56.dp)
            .border(1.dp, Color(0xFFDDE2EC), RoundedCornerShape(12.dp))
            .padding(horizontal = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Checkbox(
            checked = checked,
            enabled = enabled,
            onCheckedChange = onChecked
        )
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
private fun EmptyState(message: String) {
    Box(
        modifier = Modifier.fillMaxSize().padding(20.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = message,
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

private fun eventTypeColor(type: EventType): Color {
    return when (type) {
        EventType.INFO       -> Color(0xFF34C759)   // systemGreen
        EventType.EVAL       -> Color(0xFF007AFF)   // systemBlue
        EventType.CHECKPOINT -> Color(0xFFFF9500)   // systemOrange
        EventType.CLEANUP    -> Color(0xFFFF6B00)   // darker orange
        EventType.RSS        -> Color(0xFFAF52DE)   // systemPurple
        EventType.ENERGY     -> Color(0xFF5AC8FA)   // systemTeal
        EventType.WARNING    -> Color(0xFFFF9500)   // systemOrange
        EventType.ERROR      -> Color(0xFFFF3B30)   // systemRed
    }
}

private fun nativeRunnerColor(status: NativeDemoRunStatus): Color {
    return when (status) {
        NativeDemoRunStatus.READY -> Color(0xFF8E8E93)
        NativeDemoRunStatus.RUNNING -> Color(0xFF34C759)
        NativeDemoRunStatus.STOPPING -> Color(0xFFFF9500)
        NativeDemoRunStatus.STOPPED -> Color(0xFFFF9500)
        NativeDemoRunStatus.COMPLETED -> Color(0xFF007AFF)
        NativeDemoRunStatus.FAILED -> Color(0xFFFF3B30)
    }
}

private fun shortTabTitle(tab: DashboardTab): String {
    return when (tab) {
        DashboardTab.RUNNER -> "Run"
        DashboardTab.OVERVIEW -> "Prog"
        DashboardTab.TRAINING -> "Charts"
        DashboardTab.COMPARISON -> "Diff"
        DashboardTab.LOGS -> "Logs"
        DashboardTab.EXPERIMENTS -> "Demo"
    }
}

private fun formatTime(ms: Long): String {
    if (ms <= 0L) return "N/A"
    val fmt = SimpleDateFormat("MM-dd HH:mm:ss", Locale.getDefault())
    return fmt.format(Date(ms))
}

private fun format1(v: Double): String = String.format(Locale.US, "%.1f", v)
private fun format2(v: Double): String = String.format(Locale.US, "%.2f", v)
private fun format3(v: Double): String = String.format(Locale.US, "%.3f", v)
private fun format4(v: Double): String = String.format(Locale.US, "%.4f", v)
private fun format6(v: Double): String = String.format(Locale.US, "%.6f", v)
private fun formatPct(v: Double): String = String.format(Locale.US, "%.1f%%", v * 100.0)

private fun deltaText(current: Double?, previous: Double?, lowerIsBetter: Boolean): String? {
    if (current == null || previous == null) return null
    val delta = current - previous
    val improved = if (lowerIsBetter) delta < 0 else delta > 0
    val arrow = if (improved) "▲" else "▼"
    val sign = if (delta >= 0) "+" else ""
    return "$arrow $sign${format4(delta)} vs prev"
}

private fun diffText(
    primary: Double?,
    compare: Double?,
    lowerIsBetter: Boolean,
    suffix: String = ""
): String {
    if (primary == null || compare == null) return "N/A"
    val delta = primary - compare
    val primaryBetter = if (lowerIsBetter) primary < compare else primary > compare
    val winner = if (primaryBetter) "Primary better" else "Compare better"
    val sign = if (delta >= 0) "+" else ""
    return "$winner ($sign${format3(delta)}$suffix)"
}

private fun alignSeries(
    first: List<StepMetric>,
    second: List<StepMetric>,
    selector: (StepMetric) -> Double?
): Pair<List<Double>, List<Double>> {
    val firstMap = first.mapNotNull { m -> selector(m)?.let { m.step to it } }.toMap()
    val secondMap = second.mapNotNull { m -> selector(m)?.let { m.step to it } }.toMap()

    val commonSteps = firstMap.keys.intersect(secondMap.keys).toList().sorted()
    val a = commonSteps.mapNotNull { firstMap[it] }
    val b = commonSteps.mapNotNull { secondMap[it] }
    return a to b
}

private fun rollingLossDeltaVariance(lossSeries: List<Double>, window: Int): List<Double> {
    if (lossSeries.size < 3) return emptyList()
    val deltas = lossSeries.zipWithNext { a, b -> b - a }
    if (deltas.size < 2) return emptyList()
    val safeWindow = window.coerceAtLeast(2)
    return deltas.indices.map { idx ->
        val start = (idx - safeWindow + 1).coerceAtLeast(0)
        val slice = deltas.subList(start, idx + 1)
        val mean = slice.average()
        val variance = if (slice.size > 1) {
            slice.sumOf { d ->
                val diff = d - mean
                diff * diff
            } / (slice.size - 1).toDouble()
        } else {
            0.0
        }
        variance
    }
}

// ── Model type badge helper ────────────────────────────────────────────────────
private fun inferModelBadge(name: String): Pair<String, Color> {
    val lower = name.lowercase()
    return when {
        "gemma" in lower && "1b"     in lower -> "Gemma 1B"   to Color(0xFF34C759)   // systemGreen
        "gemma" in lower && "270"    in lower -> "Gemma 270M" to Color(0xFF34C759)
        "gemma" in lower                      -> "Gemma"       to Color(0xFF34C759)
        "gpt2"  in lower && "medium" in lower -> "GPT-2 M"    to Color(0xFF007AFF)   // systemBlue
        "gpt2"  in lower && "large"  in lower -> "GPT-2 L"    to Color(0xFF007AFF)
        "gpt2"  in lower                      -> "GPT-2"       to Color(0xFF007AFF)
        "lora"  in lower                      -> "LoRA"         to Color(0xFFAF52DE)   // systemPurple
        else                                  -> "Run"          to Color(0xFF8E8E93)   // systemGray
    }
}
