package com.mobilefinetuner.visualizer.viewmodel

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class PreferenceEvalSummaryTest {
    @Test
    fun summarizePreferenceScoresComputesAccuracyAndMeanMargin() {
        val summary = summarizePreferenceScores(
            chosenLogps = floatArrayOf(-1.0f, -3.0f, -2.0f),
            rejectedLogps = floatArrayOf(-2.0f, -2.5f, -4.0f)
        )

        assertThat(summary.pairCount).isEqualTo(3)
        assertThat(summary.accuracy).isWithin(1e-6).of(2.0 / 3.0)
        assertThat(summary.meanMargin).isWithin(1e-6).of((1.0 - 0.5 + 2.0) / 3.0)
    }

    @Test
    fun summarizePreferenceScoresRejectsMismatchedArrays() {
        val error = runCatching {
            summarizePreferenceScores(floatArrayOf(-1.0f), floatArrayOf(-1.0f, -2.0f))
        }.exceptionOrNull()

        assertThat(error).isInstanceOf(IllegalArgumentException::class.java)
    }
}
