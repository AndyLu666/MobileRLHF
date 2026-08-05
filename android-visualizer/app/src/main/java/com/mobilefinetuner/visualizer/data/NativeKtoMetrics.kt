package com.mobilefinetuner.visualizer.data

import org.json.JSONObject

object NativeKtoMetrics {
    fun stepFields(
        step: Int,
        batchSize: Int,
        binaryFeedbackRows: Int,
        trainableParams: Int,
        loss: Float,
        accumulatedLoss: Float,
        rewardAccuracy: Float,
        rewardMargin: Float,
        desirableReward: Float,
        undesirableReward: Float,
        klEstimate: Float,
        sampleCount: Int,
        desirableCount: Int,
        undesirableCount: Int,
        validResponseTokens: Int,
        validKlResponseTokens: Int,
        accumulationStep: Int,
        gradientAccumulationSteps: Int,
        optimizerStep: Boolean,
        elapsedMs: Long,
        rssMb: Double?
    ): Map<String, Any> {
        val fields = linkedMapOf<String, Any>(
            "step" to step,
            "source" to "android_native_kto_binary",
            "batch_semantics" to "ordered_binary_feedback",
            "batch_size" to batchSize,
            // Keep train_samples as a compatibility alias. It counts binary
            // prompt/completion/label rows here, never preference pairs.
            "train_samples" to binaryFeedbackRows,
            "binary_feedback_rows" to binaryFeedbackRows,
            "trainable_tensor_count" to trainableParams,
            "reference_gradient_tensor_count" to 0,
            "loss" to loss,
            "accumulated_loss" to accumulatedLoss,
            "reward_accuracy" to rewardAccuracy,
            "reward_margin" to rewardMargin,
            "desirable_reward" to desirableReward,
            "undesirable_reward" to undesirableReward,
            "kl_estimate" to klEstimate,
            "sample_count" to sampleCount,
            "desirable_count" to desirableCount,
            "undesirable_count" to undesirableCount,
            "valid_response_tokens" to validResponseTokens,
            "valid_kl_response_tokens" to validKlResponseTokens,
            "accumulation_step" to accumulationStep,
            "gradient_accumulation_steps" to gradientAccumulationSteps,
            "optimizer_step" to optimizerStep,
            "elapsed_ms" to elapsedMs,
            "reference_logps" to "base_model_exact_ordered_batch_cache"
        )
        if (rssMb != null) {
            fields["rss_mb"] = rssMb
        }
        return fields
    }

    fun stepJson(
        step: Int,
        batchSize: Int,
        binaryFeedbackRows: Int,
        trainableParams: Int,
        loss: Float,
        accumulatedLoss: Float,
        rewardAccuracy: Float,
        rewardMargin: Float,
        desirableReward: Float,
        undesirableReward: Float,
        klEstimate: Float,
        sampleCount: Int,
        desirableCount: Int,
        undesirableCount: Int,
        validResponseTokens: Int,
        validKlResponseTokens: Int,
        accumulationStep: Int,
        gradientAccumulationSteps: Int,
        optimizerStep: Boolean,
        elapsedMs: Long,
        rssMb: Double?
    ): JSONObject {
        val json = JSONObject()
        for ((key, value) in stepFields(
            step = step,
            batchSize = batchSize,
            binaryFeedbackRows = binaryFeedbackRows,
            trainableParams = trainableParams,
            loss = loss,
            accumulatedLoss = accumulatedLoss,
            rewardAccuracy = rewardAccuracy,
            rewardMargin = rewardMargin,
            desirableReward = desirableReward,
            undesirableReward = undesirableReward,
            klEstimate = klEstimate,
            sampleCount = sampleCount,
            desirableCount = desirableCount,
            undesirableCount = undesirableCount,
            validResponseTokens = validResponseTokens,
            validKlResponseTokens = validKlResponseTokens,
            accumulationStep = accumulationStep,
            gradientAccumulationSteps = gradientAccumulationSteps,
            optimizerStep = optimizerStep,
            elapsedMs = elapsedMs,
            rssMb = rssMb
        )) {
            json.put(key, value)
        }
        return json
    }

    fun dpoStepJson(
        step: Int,
        batchSize: Int,
        trainSamples: Int,
        feedbackPairs: Int,
        trainableParams: Int,
        loss: Float,
        rewardAccuracy: Float,
        rewardMargin: Float,
        chosenReward: Float,
        rejectedReward: Float,
        pairCount: Int,
        validResponseTokens: Int,
        elapsedMs: Long,
        rssMb: Double?
    ): JSONObject {
        val json = JSONObject()
        val fields = linkedMapOf<String, Any>(
            "step" to step,
            "source" to "android_native_dpo",
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
            "chosen_reward" to chosenReward,
            "rejected_reward" to rejectedReward,
            "pair_count" to pairCount,
            "valid_response_tokens" to validResponseTokens,
            "elapsed_ms" to elapsedMs,
            "reference_logps" to "base_model_cached"
        )
        if (rssMb != null) {
            fields["rss_mb"] = rssMb
        }
        for ((key, value) in fields) {
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
