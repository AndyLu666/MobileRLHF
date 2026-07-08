package com.mobilefinetuner.visualizer.viewmodel

import com.google.common.truth.Truth.assertThat
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
}
