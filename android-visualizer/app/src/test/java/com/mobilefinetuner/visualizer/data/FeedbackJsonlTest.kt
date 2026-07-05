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

        assertThat(line).contains("\"type\":\"pair\"")
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

    @Test
    fun encodeRating_thumbsUpProducesSingleKtoRecord() {
        val line = FeedbackJsonl.encodeRating(
            FeedbackJsonl.RatingRecord(
                prompt = "How should I recover after poor sleep?",
                response = "Keep exercise light and take a short walk after lunch.",
                desirable = true,
                sessionId = "single-session",
                ratingIndex = 3,
                timestampMs = 999L
            )
        )

        assertThat(line).contains("\"type\":\"single\"")
        assertThat(line).contains("\"prompt\":\"How should I recover after poor sleep?\"")
        assertThat(line).contains("\"response\":\"Keep exercise light and take a short walk after lunch.\"")
        assertThat(line).contains("\"label\":true")
        assertThat(line).contains("\"rating_label\":\"up\"")
        assertThat(line).contains("\"rating_index\":3")
        assertThat(line).contains("\"timestamp_ms\":999")
    }

    @Test
    fun encodeRating_thumbsDownProducesUndesirableSingleKtoRecord() {
        val line = FeedbackJsonl.encodeRating(
            FeedbackJsonl.RatingRecord(
                prompt = "How should I train today?",
                response = "Ignore fatigue and do maximum intervals now.",
                desirable = false,
                sessionId = "single-session",
                ratingIndex = 0,
                timestampMs = 1000L,
                source = "android_qwen_inference_feedback"
            )
        )

        assertThat(line).contains("\"type\":\"single\"")
        assertThat(line).contains("\"label\":false")
        assertThat(line).contains("\"rating_label\":\"down\"")
        assertThat(line).contains("\"source\":\"android_qwen_inference_feedback\"")
    }

    @Test
    fun countKtoRowsCountsPairAsTwoRowsAndSingleAsOneRow() {
        val pair = FeedbackJsonl.encodeChoice(
            FeedbackJsonl.ChoiceRecord(
                prompt = FeedbackPrompt("Prompt", "Good", "Bad"),
                chosenLabel = "A",
                sessionId = "pair-session",
                choiceIndex = 0,
                timestampMs = 1L
            )
        )
        val single = FeedbackJsonl.encodeRating(
            FeedbackJsonl.RatingRecord(
                prompt = "Prompt",
                response = "Good",
                desirable = true,
                sessionId = "single-session",
                ratingIndex = 1,
                timestampMs = 2L
            )
        )

        assertThat(FeedbackJsonl.countKtoRows(listOf(pair, single))).isEqualTo(3)
    }

    @Test
    fun hasTypedSingleDetectsSingleRecordsOnly() {
        val single = "{\"type\":\"single\",\"prompt\":\"P\",\"response\":\"R\",\"label\":true}"
        val pair = "{\"type\":\"pair\",\"prompt\":\"P\",\"chosen\":\"A\",\"rejected\":\"B\"}"

        assertThat(FeedbackJsonl.hasTypedSingle(single)).isTrue()
        assertThat(FeedbackJsonl.hasTypedSingle(pair)).isFalse()
    }
}
