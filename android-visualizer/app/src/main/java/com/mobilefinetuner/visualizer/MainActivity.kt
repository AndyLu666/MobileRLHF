package com.mobilefinetuner.visualizer

import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.SystemBarStyle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import com.mobilefinetuner.visualizer.ui.DashboardScreen
import com.mobilefinetuner.visualizer.ui.theme.MobileFineTunerVisualizerTheme
import com.mobilefinetuner.visualizer.viewmodel.ExperimentViewModel

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
    }
}
