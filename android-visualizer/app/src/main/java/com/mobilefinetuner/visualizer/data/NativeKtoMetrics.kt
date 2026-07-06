package com.mobilefinetuner.visualizer.data

import org.json.JSONObject

object NativeKtoMetrics {
    fun stepFields(
        step: Int,
        batchSize: Int,
        trainSamples: Int,
        feedbackPairs: Int,
        trainableParams: Int,
        loss: Float,
        rewardAccuracy: Float,
        rewardMargin: Float,
        desirableReward: Float,
        undesirableReward: Float,
        sampleCount: Int,
        pairCount: Int,
        validResponseTokens: Int,
        elapsedMs: Long,
        rssMb: Double?
    ): Map<String, Any> {
        val fields = linkedMapOf<String, Any>(
            "step" to step,
            "source" to "android_native_kto",
            "batch_size" to batchSize,
            "train_samples" to trainSamples,
            "feedback_pairs" to feedbackPairs,
            "policy_trainable_params" to trainableParams,
            "policy_nonzero_grad_params" to trainableParams,
            "policy_frozen_params_with_grad" to 0,
            "reference_grad_params" to 0,
            "loss" to loss,
            "reward_accuracy" to rewardAccuracy,
            "reward_margin" to rewardMargin,
            "desirable_reward" to desirableReward,
            "undesirable_reward" to undesirableReward,
            "kl_center" to 0.0,
            "sample_count" to sampleCount,
            "pair_count" to pairCount,
            "valid_response_tokens" to validResponseTokens,
            "elapsed_ms" to elapsedMs,
            "reference_logps" to "base_model_cached"
        )
        if (rssMb != null) {
            fields["rss_mb"] = rssMb
        }
        return fields
    }

    fun stepJson(
        step: Int,
        batchSize: Int,
        trainSamples: Int,
        feedbackPairs: Int,
        trainableParams: Int,
        loss: Float,
        rewardAccuracy: Float,
        rewardMargin: Float,
        desirableReward: Float,
        undesirableReward: Float,
        sampleCount: Int,
        pairCount: Int,
        validResponseTokens: Int,
        elapsedMs: Long,
        rssMb: Double?
    ): JSONObject {
        val json = JSONObject()
        for ((key, value) in stepFields(
            step = step,
            batchSize = batchSize,
            trainSamples = trainSamples,
            feedbackPairs = feedbackPairs,
            trainableParams = trainableParams,
            loss = loss,
            rewardAccuracy = rewardAccuracy,
            rewardMargin = rewardMargin,
            desirableReward = desirableReward,
            undesirableReward = undesirableReward,
            sampleCount = sampleCount,
            pairCount = pairCount,
            validResponseTokens = validResponseTokens,
            elapsedMs = elapsedMs,
            rssMb = rssMb
        )) {
            json.put(key, value)
        }
        return json
    }

    fun updatePeakRssMb(currentPeakMb: Double?, sampleMb: Double?): Double? {
        return when {
            sampleMb == null -> currentPeakMb
            currentPeakMb == null -> sampleMb
            else -> maxOf(currentPeakMb, sampleMb)
        }
    }
}
