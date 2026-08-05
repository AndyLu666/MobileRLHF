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

    data class TrainingPair(
        val prompt: String,
        val chosen: String,
        val rejected: String
    )

    data class TrainingKtoRow(
        val prompt: String,
        val response: String,
        val desirable: Boolean
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
        return (
            hasJsonStringValue(line, "type", "single") ||
                hasJsonStringValue(line, "feedback_format", "kto_binary")
            ) &&
            hasJsonField(line, "response") &&
            hasJsonField(line, "label")
    }

    fun toTrainingPairs(
        lines: List<String>,
        desirableBaseline: String,
        undesirableBaseline: String
    ): List<TrainingPair> {
        return lines.mapNotNull { line ->
            val trimmed = line.trim()
            if (trimmed.isBlank()) return@mapNotNull null
            val prompt = extractJsonString(trimmed, "prompt")?.trim().orEmpty()
            if (prompt.isBlank()) return@mapNotNull null

            if (hasTypedSingle(trimmed)) {
                val response = extractJsonString(trimmed, "response")?.trim().orEmpty()
                if (response.isBlank()) return@mapNotNull null
                if (hasJsonBooleanValue(trimmed, "label", true)) {
                    TrainingPair(prompt, response, undesirableBaseline)
                } else if (hasJsonBooleanValue(trimmed, "label", false)) {
                    TrainingPair(prompt, desirableBaseline, response)
                } else {
                    null
                }
            } else {
                val chosen = extractJsonString(trimmed, "chosen")?.trim().orEmpty()
                val rejected = extractJsonString(trimmed, "rejected")?.trim().orEmpty()
                if (chosen.isBlank() || rejected.isBlank()) null else TrainingPair(prompt, chosen, rejected)
            }
        }
    }

    /**
     * Preserve native KTO's unpaired prompt/response/label contract.
     *
     * Pair records are explicitly unpaired without synthetic baselines. The
     * returned rows place desirable feedback first and undesirable feedback
     * second, matching the standard KTO preference-unpairing layout. Single
     * thumbs rows keep their label and payload.
     */
    fun toKtoRows(lines: List<String>): List<TrainingKtoRow> {
        val desirableRows = mutableListOf<TrainingKtoRow>()
        val undesirableRows = mutableListOf<TrainingKtoRow>()
        lines.forEach { line ->
            val trimmed = line.trim()
            if (trimmed.isBlank()) return@forEach
            val prompt = extractJsonString(trimmed, "prompt")?.trim().orEmpty()
            if (prompt.isBlank()) return@forEach

            if (hasTypedSingle(trimmed)) {
                val response = extractJsonString(trimmed, "response")?.trim().orEmpty()
                if (response.isBlank()) return@forEach
                when {
                    hasJsonBooleanValue(trimmed, "label", true) ->
                        desirableRows += TrainingKtoRow(prompt, response, true)
                    hasJsonBooleanValue(trimmed, "label", false) ->
                        undesirableRows += TrainingKtoRow(prompt, response, false)
                }
            } else {
                val chosen = extractJsonString(trimmed, "chosen")?.trim().orEmpty()
                val rejected = extractJsonString(trimmed, "rejected")?.trim().orEmpty()
                if (chosen.isNotBlank() && rejected.isNotBlank()) {
                    desirableRows += TrainingKtoRow(prompt, chosen, true)
                    undesirableRows += TrainingKtoRow(prompt, rejected, false)
                }
            }
        }
        return desirableRows + undesirableRows
    }

    private fun hasJsonField(line: String, key: String): Boolean {
        return Regex("\"${Regex.escape(key)}\"\\s*:").containsMatchIn(line)
    }

    private fun hasJsonStringValue(line: String, key: String, value: String): Boolean {
        return extractJsonString(line, key) == value
    }

    private fun hasJsonBooleanValue(line: String, key: String, value: Boolean): Boolean {
        return Regex("\"${Regex.escape(key)}\"\\s*:\\s*$value\\b").containsMatchIn(line)
    }

    private fun extractJsonString(line: String, key: String): String? {
        val match = Regex("\"${Regex.escape(key)}\"\\s*:\\s*\"").find(line) ?: return null
        val valueStart = match.range.last + 1
        val out = StringBuilder()
        var escaped = false
        for (i in valueStart until line.length) {
            val ch = line[i]
            if (escaped) {
                when (ch) {
                    '\\' -> out.append('\\')
                    '"' -> out.append('"')
                    'b' -> out.append('\b')
                    'f' -> out.append('\u000C')
                    'n' -> out.append('\n')
                    'r' -> out.append('\r')
                    't' -> out.append('\t')
                    'u' -> {
                        val hex = line.substring(i + 1, (i + 5).coerceAtMost(line.length))
                        val code = hex.toIntOrNull(16)
                        if (code != null) out.append(code.toChar())
                    }
                    else -> out.append(ch)
                }
                escaped = false
            } else if (ch == '\\') {
                escaped = true
            } else if (ch == '"') {
                return out.toString()
            } else {
                out.append(ch)
            }
        }
        return null
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
