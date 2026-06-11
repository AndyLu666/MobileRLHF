/**
 * @file dpo_loss.h
 * @brief Forward DPO helpers for native C++ training/evaluation
 */

#pragma once

#include "backward_functions.h"
#include "tensor.h"
#include <vector>

namespace ops {

struct DPOMetrics {
    float loss = 0.0f;
    float chosen_reward = 0.0f;
    float rejected_reward = 0.0f;
    float reward_margin = 0.0f;
    float reward_accuracy = 0.0f;
};

/**
 * @brief Compute response-only sequence log-probabilities.
 *
 * Aligns with the PyTorch helper:
 * - use logits[:, :-1, :]
 * - use labels/input_ids[:, 1:]
 * - use response_mask[:, 1:]
 * - sum token log-probs over response tokens only
 */
std::vector<float> sequence_logprob_masked(const TensorPtr& logits,
                                           const TensorPtr& input_ids,
                                           const TensorPtr& response_mask);

/**
 * @brief Autograd-enabled scalar DPO loss from policy/reference logits.
 *
 * Gradients are propagated only to policy logits. Reference logits are treated
 * as frozen scores.
 */
TensorPtr dpo_loss(const TensorPtr& policy_chosen_logits,
                   const TensorPtr& policy_rejected_logits,
                   const TensorPtr& ref_chosen_logits,
                   const TensorPtr& ref_rejected_logits,
                   const TensorPtr& chosen_input_ids,
                   const TensorPtr& rejected_input_ids,
                   const TensorPtr& chosen_response_mask,
                   const TensorPtr& rejected_response_mask,
                   float beta);

/**
 * @brief Autograd-enabled scalar DPO loss using cached frozen-reference logps.
 */
TensorPtr dpo_loss_with_ref_logps(const TensorPtr& policy_chosen_logits,
                                  const TensorPtr& policy_rejected_logits,
                                  const std::vector<float>& ref_chosen_logps,
                                  const std::vector<float>& ref_rejected_logps,
                                  const TensorPtr& chosen_input_ids,
                                  const TensorPtr& rejected_input_ids,
                                  const TensorPtr& chosen_response_mask,
                                  const TensorPtr& rejected_response_mask,
                                  float beta);

/**
 * @brief Compute mean DPO metrics from per-sample log-probabilities.
 */
DPOMetrics compute_dpo_metrics_from_logps(const std::vector<float>& policy_chosen_logps,
                                          const std::vector<float>& policy_rejected_logps,
                                          const std::vector<float>& ref_chosen_logps,
                                          const std::vector<float>& ref_rejected_logps,
                                          float beta);

/**
 * @brief Convenience wrapper for computing DPO metrics directly from logits.
 */
DPOMetrics compute_dpo_metrics_from_logits(const TensorPtr& policy_chosen_logits,
                                           const TensorPtr& policy_rejected_logits,
                                           const TensorPtr& ref_chosen_logits,
                                           const TensorPtr& ref_rejected_logits,
                                           const TensorPtr& chosen_input_ids,
                                           const TensorPtr& rejected_input_ids,
                                           const TensorPtr& chosen_response_mask,
                                           const TensorPtr& rejected_response_mask,
                                           float beta);

DPOMetrics compute_dpo_metrics_from_policy_logits_and_ref_logps(
    const TensorPtr& policy_chosen_logits,
    const TensorPtr& policy_rejected_logits,
    const std::vector<float>& ref_chosen_logps,
    const std::vector<float>& ref_rejected_logps,
    const TensorPtr& chosen_input_ids,
    const TensorPtr& rejected_input_ids,
    const TensorPtr& chosen_response_mask,
    const TensorPtr& rejected_response_mask,
    float beta);

}  // namespace ops
