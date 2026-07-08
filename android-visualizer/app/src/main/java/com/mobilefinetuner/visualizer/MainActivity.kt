package com.mobilefinetuner.visualizer

import android.content.Intent
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.SystemBarStyle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.lifecycle.lifecycleScope
import com.mobilefinetuner.visualizer.model.NativeDemoConfig
import com.mobilefinetuner.visualizer.ui.DashboardScreen
import com.mobilefinetuner.visualizer.ui.theme.MobileFineTunerVisualizerTheme
import com.mobilefinetuner.visualizer.viewmodel.ExperimentViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private val viewModel by viewModels<ExperimentViewModel>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val backgroundColor = Color.rgb(246, 247, 249)
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.light(backgroundColor, backgroundColor),
            navigationBarStyle = SystemBarStyle.light(backgroundColor, backgroundColor)
        )
        window.setBackgroundDrawable(ColorDrawable(backgroundColor))
        setContent {
            MobileFineTunerVisualizerTheme(darkTheme = false) {
                DashboardScreen(viewModel = viewModel)
            }
        }
        handleFedRlIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleFedRlIntent(intent)
    }

    private fun handleFedRlIntent(intent: Intent?) {
        if (intent == null) return
        val hasFedRlExtra = intent.hasExtra("fedrl_objective") ||
            intent.hasExtra("fedrl_start_native_demo")
        if (!hasFedRlExtra) return

        val objective = intent.getStringExtra("fedrl_objective")
            ?.trim()
            ?.lowercase()
            ?: "dpo"
        val config = NativeDemoConfig(
            presetName = intent.getStringExtra("fedrl_preset_name")
                ?: "FedRL-${objective.uppercase()}",
            steps = intent.getIntExtra("fedrl_steps", 10),
            seqLen = intent.getIntExtra("fedrl_seq_len", 64),
            rank = intent.getIntExtra("fedrl_rank", 2),
            alpha = intent.getIntExtra("fedrl_alpha", 4),
            learningRate = intent.getFloatExtra("fedrl_lr", 0.001f).toDouble(),
            evalInterval = intent.getIntExtra("fedrl_eval_interval", 4),
            evalBatches = intent.getIntExtra("fedrl_eval_batches", 1),
            cacheRefLogps = intent.getBooleanExtra("fedrl_cache_ref_logps", true),
            dpoPairForward = objective == "dpo",
            globalAdapterPath = intent.getStringExtra("fedrl_global_adapter_path")
                ?.trim()
                ?.takeIf { it.isNotEmpty() }
        )
        viewModel.updateNativeDemoConfig(config)

        if (intent.getBooleanExtra("fedrl_start_native_demo", false)) {
            lifecycleScope.launch {
                delay(300)
                viewModel.startNativeDemoRun()
            }
        }
    }
}
