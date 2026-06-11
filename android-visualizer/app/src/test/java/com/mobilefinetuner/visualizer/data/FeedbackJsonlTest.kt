package com.mobilefinetuner.visualizer.data

import com.google.common.truth.Truth.assertThat
import com.mobilefinetuner.visualizer.model.FeedbackPrompt
import org.junit.Test

class FeedbackJsonlTest {

    @Test
    fun encodeChoice_labelAProducesDpoCompatibleChosenRejectedFields() {
        val line = FeedbackJsonl.encodeChoice(
            FeedbackJsonl.ChoiceRecord(
                prompt = FeedbackPrompt(
                    prompt = "Pick a plan",
                    answerA = "Walk for 20 minutes after lunch.",
                    answerB = "Ignore sleep and sprint now."
                ),
                chosenLabel = "A",
                sessionId = "session-1",
                choiceIndex = 2,
                timestampMs = 1234L
            )
        )

        assertThat(line).contains("\"prompt\":\"Pick a plan\"")
        assertThat(line).contains("\"chosen\":\"Walk for 20 minutes after lunch.\"")
        assertThat(line).contains("\"rejected\":\"Ignore sleep and sprint now.\"")
        assertThat(line).contains("\"source\":\"android_ab_feedback\"")
        assertThat(line).contains("\"chosen_label\":\"A\"")
        assertThat(line).contains("\"choice_index\":2")
        assertThat(line).contains("\"timestamp_ms\":1234")
    }

    @Test
    fun encodeChoice_labelBReversesChosenRejectedAndEscapesJson() {
        val line = FeedbackJsonl.encodeChoice(
            FeedbackJsonl.ChoiceRecord(
                prompt = FeedbackPrompt(
                    prompt = "Prompt with \"quotes\"",
                    answerA = "First\nanswer",
                    answerB = "Second \\ answer"
                ),
                chosenLabel = "b",
                sessionId = "session-2",
                choiceIndex = 0,
                timestampMs = 55L
            )
        )

        assertThat(line).contains("\"prompt\":\"Prompt with \\\"quotes\\\"\"")
        assertThat(line).contains("\"chosen\":\"Second \\\\ answer\"")
        assertThat(line).contains("\"rejected\":\"First\\nanswer\"")
        assertThat(line).contains("\"chosen_label\":\"B\"")
        assertThat(line).contains("\"rejected_label\":\"A\"")
    }
}
