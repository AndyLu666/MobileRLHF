package com.mobilefinetuner.visualizer.data

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class NativeKtoMetricsTest {
    @Test
    fun stepFieldsIncludeRssMbWhenSampled() {
        val fields = NativeKtoMetrics.stepFields(
            step = 3,
            batchSize = 2,
            binaryFeedbackRows = 4,
            trainableParams = 224,
            loss = 0.25f,
            accumulatedLoss = 0.5f,
            rewardAccuracy = 1.0f,
            rewardMargin = 2.5f,
            desirableReward = 1.2f,
            undesirableReward = -1.3f,
            klEstimate = 0.15f,
            sampleCount = 2,
            desirableCount = 1,
            undesirableCount = 1,
            validResponseTokens = 64,
            validKlResponseTokens = 60,
            accumulationStep = 1,
            gradientAccumulationSteps = 2,
            optimizerStep = false,
            elapsedMs = 50123,
            rssMb = 5665.74
        )

        assertThat(fields["rss_mb"] as Double).isWithin(1e-6).of(5665.74)
        assertThat(fields["step"]).isEqualTo(3)
        assertThat(fields["source"]).isEqualTo("android_native_kto_binary")
        assertThat(fields["train_samples"]).isEqualTo(4)
        assertThat(fields["binary_feedback_rows"]).isEqualTo(4)
        assertThat(fields["trainable_tensor_count"]).isEqualTo(224)
        assertThat(fields["kl_estimate"]).isEqualTo(0.15f)
        assertThat(fields["valid_response_tokens"]).isEqualTo(64)
        assertThat(fields["valid_kl_response_tokens"]).isEqualTo(60)
        assertThat(fields["desirable_count"]).isEqualTo(1)
        assertThat(fields["undesirable_count"]).isEqualTo(1)
        assertThat(fields["accumulated_loss"]).isEqualTo(0.5f)
        assertThat(fields["accumulation_step"]).isEqualTo(1)
        assertThat(fields["gradient_accumulation_steps"]).isEqualTo(2)
        assertThat(fields["optimizer_step"]).isEqualTo(false)
        assertThat(fields).doesNotContainKey("feedback_pairs")
        assertThat(fields).doesNotContainKey("pair_count")
        assertThat(fields).doesNotContainKey("kl_center")
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
