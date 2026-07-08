package com.mobilefinetuner.visualizer.data

import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

data class MobileRLHFSingleResponse(
    val suggestion: String,
    val backend: String,
    val contextChars: Int
)

data class MobileRLHFPairResponse(
    val answerA: String,
    val answerB: String,
    val backend: String,
    val contextChars: Int
)

data class MobileRLHFKtoResponse(
    val backend: String,
    val reportJson: String,
    val metricsNdjson: String,
    val stdout: String,
    val stderr: String,
    val trainSamples: Int,
    val evalSamples: Int,
    val status: String
)

object MobileRLHFBridgeClient {
    fun buildRequestJson(query: String, referenceDate: String): String {
        return buildString {
            append("{\"query\":")
            append(jsonString(query))
            if (referenceDate.isNotBlank()) {
                append(",\"reference_date\":")
                append(jsonString(referenceDate))
            }
            append("}")
        }
    }

    fun ask(baseUrl: String, query: String, referenceDate: String): MobileRLHFSingleResponse {
        val json = postJson(baseUrl, "/ask", buildRequestJson(query, referenceDate))
        if (!json.optBoolean("ok", false)) {
            throw IOException(json.optString("error", "MobileRLHF bridge returned ok=false"))
        }
        return MobileRLHFSingleResponse(
            suggestion = json.optString("suggestion"),
            backend = json.optString("backend"),
            contextChars = json.optJSONObject("debug")?.optInt("context_chars") ?: 0
        )
    }

    fun askPair(baseUrl: String, query: String, referenceDate: String): MobileRLHFPairResponse {
        val json = postJson(baseUrl, "/ask_ab", buildRequestJson(query, referenceDate))
        if (!json.optBoolean("ok", false)) {
            val errorA = json.optJSONObject("answer_a")?.optString("error").orEmpty()
            val errorB = json.optJSONObject("answer_b")?.optString("error").orEmpty()
            throw IOException(listOf(errorA, errorB).filter { it.isNotBlank() }.joinToString("; ").ifBlank {
                "MobileRLHF bridge returned ok=false"
            })
        }
        val answerA = json.getJSONObject("answer_a")
        val answerB = json.getJSONObject("answer_b")
        return MobileRLHFPairResponse(
            answerA = answerA.optString("suggestion"),
            answerB = answerB.optString("suggestion"),
            backend = answerA.optString("backend", answerB.optString("backend")),
            contextChars = maxOf(
                answerA.optJSONObject("debug")?.optInt("context_chars") ?: 0,
                answerB.optJSONObject("debug")?.optInt("context_chars") ?: 0
            )
        )
    }

    fun trainKto(
        baseUrl: String,
        preferenceJsonl: String,
        maxSteps: Int,
        batchSize: Int,
        seqLen: Int,
        beta: Double = 0.1,
        loraRank: Int = 2,
        loraAlpha: Int = 4
    ): MobileRLHFKtoResponse {
        val body = JSONObject()
            .put("preference_jsonl", preferenceJsonl)
            .put("max_steps", maxSteps)
            .put("batch_size", batchSize)
            .put("seq_len", seqLen)
            .put("beta", beta)
            .put("lora_rank", loraRank)
            .put("lora_alpha", loraAlpha)
            .toString()
        val json = postJson(baseUrl, "/train_kto", body, readTimeoutMs = 900_000)
        if (!json.optBoolean("ok", false)) {
            throw IOException(json.optString("error", "MobileRLHF bridge KTO returned ok=false"))
        }
        val report = json.optJSONObject("report") ?: JSONObject()
        return MobileRLHFKtoResponse(
            backend = json.optString("backend", "qwen"),
            reportJson = report.toString(2),
            metricsNdjson = json.optString("metrics_ndjson"),
            stdout = json.optString("stdout"),
            stderr = json.optString("stderr"),
            trainSamples = report.optInt("train_samples", 0),
            evalSamples = report.optInt("eval_samples", 0),
            status = report.optString("status", "unknown")
        )
    }

    private fun postJson(baseUrl: String, path: String, body: String, readTimeoutMs: Int = 180_000): JSONObject {
        val cleanBase = baseUrl.trim().trimEnd('/')
        val connection = (URL(cleanBase + path).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 5_000
            readTimeout = readTimeoutMs
            doOutput = true
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
        }
        return try {
            connection.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }
            val status = connection.responseCode
            val stream = if (status in 200..299) connection.inputStream else connection.errorStream
            val text = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() }.orEmpty()
            if (text.isBlank()) {
                throw IOException("MobileRLHF bridge returned HTTP $status with empty body")
            }
            val json = JSONObject(text)
            if (status !in 200..299) {
                throw IOException(json.optString("error", "MobileRLHF bridge HTTP $status"))
            }
            json
        } finally {
            connection.disconnect()
        }
    }

    private fun jsonString(value: String): String {
        return buildString {
            append('"')
            value.forEach { char ->
                when (char) {
                    '\\' -> append("\\\\")
                    '"' -> append("\\\"")
                    '\n' -> append("\\n")
                    '\r' -> append("\\r")
                    '\t' -> append("\\t")
                    else -> if (char.code < 0x20) {
                        append("\\u")
                        append(char.code.toString(16).padStart(4, '0'))
                    } else {
                        append(char)
                    }
                }
            }
            append('"')
        }
    }
}
