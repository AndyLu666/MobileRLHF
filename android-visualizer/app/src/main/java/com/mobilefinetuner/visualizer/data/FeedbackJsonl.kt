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

    data class RatingRecord(
        val prompt: String,
        val response: String,
        val desirable: Boolean,
        val sessionId: String,
        val ratingIndex: Int,
        val timestampMs: Long,
        val source: String = "android_single_feedback",
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
            "type" to "pair",
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
        return encodeFields(fields)
    }

    fun encodeRating(record: RatingRecord): String {
        require(record.prompt.isNotBlank()) { "prompt must not be blank" }
        require(record.response.isNotBlank()) { "response must not be blank" }

        val fields = listOf(
            "type" to "single",
            "prompt" to record.prompt,
            "response" to record.response,
            "label" to record.desirable,
            "rating_label" to if (record.desirable) "up" else "down",
            "source" to record.source,
            "user_id" to record.userId,
            "session_id" to record.sessionId,
            "rating_index" to record.ratingIndex,
            "timestamp_ms" to record.timestampMs
        )
        return encodeFields(fields)
    }

    fun countKtoRows(lines: List<String>): Int {
        var count = 0
        lines.forEach { line ->
            val trimmed = line.trim()
            count += if (trimmed.isBlank()) {
                0
            } else if (hasTypedSingle(trimmed)) {
                1
            } else {
                if (hasJsonField(trimmed, "chosen") && hasJsonField(trimmed, "rejected")) 2 else 0
            }
        }
        return count
    }

    fun hasTypedSingle(line: String): Boolean {
        return hasJsonStringValue(line, "type", "single") &&
            hasJsonField(line, "response") &&
            hasJsonField(line, "label")
    }

    private fun hasJsonField(line: String, key: String): Boolean {
        return line.contains("\"$key\":")
    }

    private fun hasJsonStringValue(line: String, key: String, value: String): Boolean {
        return line.contains("\"$key\":\"$value\"")
    }

    private fun encodeFields(fields: List<Pair<String, Any>>): String {
        return buildString {
            append('{')
            fields.forEachIndexed { index, (key, value) ->
                if (index > 0) append(',')
                appendJsonString(key)
                append(':')
                appendJsonValue(value)
            }
            append('}')
        }
    }

    private fun StringBuilder.appendJsonValue(value: Any) {
        when (value) {
            is Boolean -> append(value)
            is Number -> append(value)
            else -> appendJsonString(value.toString())
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
