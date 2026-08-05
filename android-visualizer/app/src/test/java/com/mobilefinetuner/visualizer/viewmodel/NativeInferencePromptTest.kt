package com.mobilefinetuner.visualizer.viewmodel

import com.google.common.truth.Truth.assertThat
import com.mobilefinetuner.visualizer.data.FeedbackJsonl
import org.junit.Test

class NativeInferencePromptTest {
    @Test
    fun sanitizeNativeInferenceTextRemovesChatTemplateTokens() {
        val cleaned = sanitizeNativeInferenceText(
            "My activity level today is positive and encouraging.<|im_end|>\n<|assistant|>"
        )

        assertThat(cleaned).isEqualTo("My activity level today is positive and encouraging.")
        assertThat(cleaned).doesNotContain("<|")
    }

    @Test
    fun buildNativeHealthPromptAsksForConcreteShortAdvice() {
        val prompt = buildNativeHealthPrompt(
            userPrompt = "How is my activity level today?",
            variant = NativePromptVariant.SINGLE
        )

        assertThat(prompt).contains("1-2 sentences")
        assertThat(prompt).contains("specific action")
        assertThat(prompt).contains("Do not include chat template tokens")
        assertThat(prompt).contains("How is my activity level today?")
    }

    @Test
    fun canonicalKtoBatchIndicesBalanceLabelsAndAvoidSamePromptKlEdges() {
        val rows = listOf(
            FeedbackJsonl.TrainingKtoRow("P1", "good-1", true),
            FeedbackJsonl.TrainingKtoRow("P2", "good-2", true),
            FeedbackJsonl.TrainingKtoRow("P1", "bad-1", false),
            FeedbackJsonl.TrainingKtoRow("P2", "bad-2", false)
        )
        val first = canonicalKtoBatchIndices(
            rows = rows,
            configuredBatchSize = 8,
            step = 1
        )
        val second = canonicalKtoBatchIndices(
            rows = rows,
            configuredBatchSize = 2,
            step = 2
        )

        assertThat(first).hasSize(4)
        assertThat(first.count { rows[it].desirable }).isEqualTo(2)
        assertThat(second).hasSize(2)
        assertThat(second.count { rows[it].desirable }).isEqualTo(1)
        assertThat(first.toSet()).hasSize(first.size)
        assertThat(second.toSet()).hasSize(second.size)
        for (indices in listOf(first, second)) {
            indices.indices.forEach { position ->
                val previous = indices[(position - 1 + indices.size) % indices.size]
                assertThat(rows[previous].prompt).isNotEqualTo(rows[indices[position]].prompt)
            }
        }
    }

    @Test
    fun canonicalKtoBatchIndicesRequireAtLeastTwoRows() {
        val failure = runCatching {
            canonicalKtoBatchIndices(
                rows = listOf(FeedbackJsonl.TrainingKtoRow("P", "R", true)),
                configuredBatchSize = 4,
                step = 1
            )
        }.exceptionOrNull()

        assertThat(failure).isInstanceOf(IllegalArgumentException::class.java)
        assertThat(failure).hasMessageThat().contains("at least two")
    }
}
