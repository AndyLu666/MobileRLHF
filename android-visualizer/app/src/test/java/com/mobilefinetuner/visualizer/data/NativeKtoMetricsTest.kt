package com.mobilefinetuner.visualizer.data

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class NativeKtoMetricsTest {
    @Test
    fun stepFieldsIncludeRssMbWhenSampled() {
        val fields = NativeKtoMetrics.stepFields(
            step = 3,
            batchSize = 2,
            trainSamples = 4,
            feedbackPairs = 2,
            trainableParams = 224,
            loss = 0.25f,
            rewardAccuracy = 1.0f,
            rewardMargin = 2.5f,
            desirableReward = 1.2f,
            undesirableReward = -1.3f,
            sampleCount = 4,
            pairCount = 2,
            validResponseTokens = 64,
            elapsedMs = 50123,
            rssMb = 5665.74
        )

        assertThat(fields["rss_mb"] as Double).isWithin(1e-6).of(5665.74)
        assertThat(fields["step"]).isEqualTo(3)
        assertThat(fields["source"]).isEqualTo("android_native_kto")
    }

    @Test
    fun peakRssKeepsLargestSample() {
        val peak = NativeKtoMetrics.updatePeakRssMb(
            currentPeakMb = 5120.0,
            sampleMb = 5665.74
        )

        assertThat(peak).isWithin(1e-6).of(5665.74)
        assertThat(NativeKtoMetrics.updatePeakRssMb(peak, 5400.0)).isWithin(1e-6).of(5665.74)
    }
}
