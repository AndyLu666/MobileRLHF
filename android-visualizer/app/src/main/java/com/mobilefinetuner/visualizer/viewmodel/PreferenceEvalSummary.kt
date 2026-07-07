package com.mobilefinetuner.visualizer.viewmodel

data class PreferenceEvalSummary(
    val pairCount: Int,
    val accuracy: Double,
    val meanMargin: Double
)

fun summarizePreferenceScores(
    chosenLogps: FloatArray,
    rejectedLogps: FloatArray
): PreferenceEvalSummary {
    require(chosenLogps.isNotEmpty()) { "chosenLogps must not be empty" }
    require(chosenLogps.size == rejectedLogps.size) {
        "chosenLogps and rejectedLogps must have the same length"
    }

    var correct = 0
    var marginSum = 0.0
    for (i in chosenLogps.indices) {
        val margin = chosenLogps[i].toDouble() - rejectedLogps[i].toDouble()
        if (margin > 0.0) {
            correct += 1
        }
        marginSum += margin
    }
    return PreferenceEvalSummary(
        pairCount = chosenLogps.size,
        accuracy = correct.toDouble() / chosenLogps.size.toDouble(),
        meanMargin = marginSum / chosenLogps.size.toDouble()
    )
}
