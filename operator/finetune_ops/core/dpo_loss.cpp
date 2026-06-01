/**
 * @file dpo_loss.cpp
 * @brief Forward DPO helpers for native C++ training/evaluation
 */

#include "dpo_loss.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ops {

namespace {

double negative_log_sigmoid(double x) {
    if (x >= 0.0) {
        return std::log1p(std::exp(-x));
    }
    return -x + std::log1p(std::exp(x));
}

}  // namespace

std::vector<float> sequence_logprob_masked(const TensorPtr& logits,
                                           const TensorPtr& input_ids,
                                           const TensorPtr& response_mask) {
    if (logits->ndim() != 3) {
        throw std::runtime_error("sequence_logprob_masked: logits must be [B,S,V]");
    }
    if (input_ids->ndim() != 2 || response_mask->ndim() != 2) {
        throw std::runtime_error("sequence_logprob_masked: input_ids and response_mask must be [B,S]");
    }

    const auto& shape = logits->shape();
    const int64_t B = shape[0];
    const int64_t S = shape[1];
    const int64_t V = shape[2];

    if (input_ids->shape()[0] != B || input_ids->shape()[1] != S ||
        response_mask->shape()[0] != B || response_mask->shape()[1] != S) {
        throw std::runtime_error("sequence_logprob_masked: shape mismatch");
    }

    const float* x = logits->data<float>();
    const int32_t* y = input_ids->data<int32_t>();
    const int32_t* m = response_mask->data<int32_t>();

    std::vector<float> out(static_cast<size_t>(B), 0.0f);
    for (int64_t b = 0; b < B; ++b) {
        double sum_logp = 0.0;
        for (int64_t s = 0; s + 1 < S; ++s) {
            const int32_t label = y[b * S + (s + 1)];
            const int32_t keep = m[b * S + (s + 1)];
            if (!keep) continue;
            if (label < 0 || label >= V) continue;

            const float* row = x + (b * S + s) * V;
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t v = 0; v < V; ++v) {
                max_val = std::max(max_val, row[v]);
            }

            double denom = 0.0;
            for (int64_t v = 0; v < V; ++v) {
                denom += std::exp(row[v] - max_val);
            }
            const double log_sum_exp = max_val + std::log(denom);
            sum_logp += (static_cast<double>(row[label]) - log_sum_exp);
        }
        out[static_cast<size_t>(b)] = static_cast<float>(sum_logp);
    }
    return out;
}

DPOMetrics compute_dpo_metrics_from_logps(const std::vector<float>& policy_chosen_logps,
                                          const std::vector<float>& policy_rejected_logps,
                                          const std::vector<float>& ref_chosen_logps,
                                          const std::vector<float>& ref_rejected_logps,
                                          float beta) {
    const size_t n = policy_chosen_logps.size();
    if (n == 0 || policy_rejected_logps.size() != n ||
        ref_chosen_logps.size() != n || ref_rejected_logps.size() != n) {
        throw std::runtime_error("compute_dpo_metrics_from_logps: inconsistent input sizes");
    }

    DPOMetrics metrics;
    double loss_sum = 0.0;
    double chosen_reward_sum = 0.0;
    double rejected_reward_sum = 0.0;
    double reward_margin_sum = 0.0;
    double reward_accuracy_sum = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const double pi_logratio =
            static_cast<double>(policy_chosen_logps[i]) - static_cast<double>(policy_rejected_logps[i]);
        const double ref_logratio =
            static_cast<double>(ref_chosen_logps[i]) - static_cast<double>(ref_rejected_logps[i]);
        const double logit = static_cast<double>(beta) * (pi_logratio - ref_logratio);

        const double chosen_reward =
            static_cast<double>(beta) * (static_cast<double>(policy_chosen_logps[i]) - static_cast<double>(ref_chosen_logps[i]));
        const double rejected_reward =
            static_cast<double>(beta) * (static_cast<double>(policy_rejected_logps[i]) - static_cast<double>(ref_rejected_logps[i]));
        const double reward_margin = chosen_reward - rejected_reward;

        loss_sum += negative_log_sigmoid(logit);
        chosen_reward_sum += chosen_reward;
        rejected_reward_sum += rejected_reward;
        reward_margin_sum += reward_margin;
        reward_accuracy_sum += (reward_margin > 0.0) ? 1.0 : 0.0;
    }

    const double inv_n = 1.0 / static_cast<double>(n);
    metrics.loss = static_cast<float>(loss_sum * inv_n);
    metrics.chosen_reward = static_cast<float>(chosen_reward_sum * inv_n);
    metrics.rejected_reward = static_cast<float>(rejected_reward_sum * inv_n);
    metrics.reward_margin = static_cast<float>(reward_margin_sum * inv_n);
    metrics.reward_accuracy = static_cast<float>(reward_accuracy_sum * inv_n);
    return metrics;
}

DPOMetrics compute_dpo_metrics_from_logits(const TensorPtr& policy_chosen_logits,
                                           const TensorPtr& policy_rejected_logits,
                                           const TensorPtr& ref_chosen_logits,
                                           const TensorPtr& ref_rejected_logits,
                                           const TensorPtr& chosen_input_ids,
                                           const TensorPtr& rejected_input_ids,
                                           const TensorPtr& chosen_response_mask,
                                           const TensorPtr& rejected_response_mask,
                                           float beta) {
    auto policy_chosen_logps = sequence_logprob_masked(
        policy_chosen_logits, chosen_input_ids, chosen_response_mask);
    auto policy_rejected_logps = sequence_logprob_masked(
        policy_rejected_logits, rejected_input_ids, rejected_response_mask);
    auto ref_chosen_logps = sequence_logprob_masked(
        ref_chosen_logits, chosen_input_ids, chosen_response_mask);
    auto ref_rejected_logps = sequence_logprob_masked(
        ref_rejected_logits, rejected_input_ids, rejected_response_mask);

    return compute_dpo_metrics_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta);
}

}  // namespace ops
