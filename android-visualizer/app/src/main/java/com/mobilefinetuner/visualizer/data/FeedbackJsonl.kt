package com.mobilefinetuner.visualizer.data

import com.mobilefinetuner.visualizer.model.FeedbackPrompt

object FeedbackJsonl {
    data class ChoiceRecord(
        val prompt: FeedbackPrompt,
        val chosenLabel: String,
        val sessionId: String,
        val choiceIndex: Int,
        val timestampMs: Long,
        val source: String = "android_ab_feedback",
        val userId: String = "local_device_user"
    )

    fun encodeChoice(record: ChoiceRecord): String {
        val normalizedLabel = record.chosenLabel.trim().uppercase()
        require(normalizedLabel == "A" || normalizedLabel == "B") {
            "chosenLabel must be A or B"
        }

        val chosen = if (normalizedLabel == "A") record.prompt.answerA else record.prompt.answerB
        val rejected = if (normalizedLabel == "A") record.prompt.answerB else record.prompt.answerA
        val rejectedLabel = if (normalizedLabel == "A") "B" else "A"

        val fields = listOf(
            "prompt" to record.prompt.prompt,
            "chosen" to chosen,
            "rejected" to rejected,
            "source" to record.source,
            "user_id" to record.userId,
            "session_id" to record.sessionId,
            "choice_index" to record.choiceIndex,
            "chosen_label" to normalizedLabel,
            "rejected_label" to rejectedLabel,
            "timestamp_ms" to record.timestampMs
        )
        return buildString {
            append('{')
            fields.forEachIndexed { index, (key, value) ->
                if (index > 0) append(',')
                appendJsonString(key)
                append(':')
                when (value) {
                    is Number -> append(value)
                    else -> appendJsonString(value.toString())
                }
            }
            append('}')
        }
    }

    private fun StringBuilder.appendJsonString(value: String) {
        append('"')
        value.forEach { ch ->
            when (ch) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\b' -> append("\\b")
                '\u000C' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> {
                    if (ch.code < 0x20) {
                        append("\\u")
                        append(ch.code.toString(16).padStart(4, '0'))
                    } else {
                        append(ch)
                    }
                }
            }
        }
        append('"')
    }
}
