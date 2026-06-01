/**
 * @file dpo_loss.cpp
 * @brief Native DPO loss and metrics for C++ training/evaluation
 */

#include "dpo_loss.h"
#ifdef USE_NEW_AUTOGRAD_ENGINE
#include "autograd_engine.h"
#endif
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

double sigmoid_minus_one(double x) {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return -z / (1.0 + z);
    }
    const double z = std::exp(x);
    return -1.0 / (1.0 + z);
}

void validate_seq_inputs(const TensorPtr& logits,
                         const TensorPtr& input_ids,
                         const TensorPtr& response_mask,
                         const char* name) {
    if (!logits || !input_ids || !response_mask) {
        throw std::runtime_error(std::string(name) + ": null input");
    }
    if (logits->ndim() != 3) {
        throw std::runtime_error(std::string(name) + ": logits must be [B,S,V]");
    }
    if (input_ids->ndim() != 2 || response_mask->ndim() != 2) {
        throw std::runtime_error(std::string(name) + ": input_ids and response_mask must be [B,S]");
    }

    const auto& shape = logits->shape();
    const int64_t B = shape[0];
    const int64_t S = shape[1];
    if (input_ids->shape()[0] != B || input_ids->shape()[1] != S ||
        response_mask->shape()[0] != B || response_mask->shape()[1] != S) {
        throw std::runtime_error(std::string(name) + ": shape mismatch");
    }
}

std::vector<float> compute_sequence_logps(const TensorPtr& logits,
                                          const TensorPtr& input_ids,
                                          const TensorPtr& response_mask) {
    validate_seq_inputs(logits, input_ids, response_mask, "compute_sequence_logps");

    const auto& shape = logits->shape();
    const int64_t B = shape[0];
    const int64_t S = shape[1];
    const int64_t V = shape[2];

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
                denom += std::exp(static_cast<double>(row[v] - max_val));
            }
            const double log_sum_exp = static_cast<double>(max_val) + std::log(denom);
            sum_logp += static_cast<double>(row[label]) - log_sum_exp;
        }
        out[static_cast<size_t>(b)] = static_cast<float>(sum_logp);
    }
    return out;
}

void accumulate_seq_logp_grads(const TensorPtr& grad_logits,
                               const TensorPtr& logits,
                               const TensorPtr& input_ids,
                               const TensorPtr& response_mask,
                               const std::vector<double>& sample_coeffs) {
    validate_seq_inputs(logits, input_ids, response_mask, "accumulate_seq_logp_grads");

    const auto& shape = logits->shape();
    const int64_t B = shape[0];
    const int64_t S = shape[1];
    const int64_t V = shape[2];
    if (static_cast<int64_t>(sample_coeffs.size()) != B) {
        throw std::runtime_error("accumulate_seq_logp_grads: sample coeff size mismatch");
    }

    const float* x = logits->data<float>();
    const int32_t* y = input_ids->data<int32_t>();
    const int32_t* m = response_mask->data<int32_t>();
    float* gx = grad_logits->data<float>();

    for (int64_t b = 0; b < B; ++b) {
        const double coeff = sample_coeffs[static_cast<size_t>(b)];
        if (coeff == 0.0) continue;

        for (int64_t s = 0; s + 1 < S; ++s) {
            const int32_t label = y[b * S + (s + 1)];
            const int32_t keep = m[b * S + (s + 1)];
            if (!keep) continue;
            if (label < 0 || label >= V) continue;

            const float* row = x + (b * S + s) * V;
            float* grad_row = gx + (b * S + s) * V;

            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t v = 0; v < V; ++v) {
                max_val = std::max(max_val, row[v]);
            }

            double denom = 0.0;
            for (int64_t v = 0; v < V; ++v) {
                denom += std::exp(static_cast<double>(row[v] - max_val));
            }
            const double inv_denom = 1.0 / denom;

            for (int64_t v = 0; v < V; ++v) {
                const double p = std::exp(static_cast<double>(row[v] - max_val)) * inv_denom;
                const double one_hot = (v == label) ? 1.0 : 0.0;
                grad_row[v] += static_cast<float>(coeff * (one_hot - p));
            }
        }
    }
}

class DPOBackward : public BackwardFunction {
public:
    DPOBackward(const TensorPtr& policy_chosen_logits,
                const TensorPtr& policy_rejected_logits,
                const TensorPtr& chosen_input_ids,
                const TensorPtr& rejected_input_ids,
                const TensorPtr& chosen_response_mask,
                const TensorPtr& rejected_response_mask,
                std::vector<double> chosen_coeffs,
                std::vector<double> rejected_coeffs)
        : policy_chosen_logits_(policy_chosen_logits),
          policy_rejected_logits_(policy_rejected_logits),
          chosen_input_ids_(chosen_input_ids),
          rejected_input_ids_(rejected_input_ids),
          chosen_response_mask_(chosen_response_mask),
          rejected_response_mask_(rejected_response_mask),
          chosen_coeffs_(std::move(chosen_coeffs)),
          rejected_coeffs_(std::move(rejected_coeffs)) {}

    std::vector<TensorPtr> apply(const TensorPtr& grad_output) override {
        float outer = 1.0f;
        if (grad_output && grad_output->numel() > 0) {
            outer = grad_output->data<float>()[0];
        }

        std::vector<double> chosen_coeffs = chosen_coeffs_;
        std::vector<double> rejected_coeffs = rejected_coeffs_;
        for (double& v : chosen_coeffs) v *= static_cast<double>(outer);
        for (double& v : rejected_coeffs) v *= static_cast<double>(outer);

        auto grad_chosen = zeros(policy_chosen_logits_->shape(), kFloat32, kCPU);
        auto grad_rejected = zeros(policy_rejected_logits_->shape(), kFloat32, kCPU);

        accumulate_seq_logp_grads(
            grad_chosen,
            policy_chosen_logits_,
            chosen_input_ids_,
            chosen_response_mask_,
            chosen_coeffs);
        accumulate_seq_logp_grads(
            grad_rejected,
            policy_rejected_logits_,
            rejected_input_ids_,
            rejected_response_mask_,
            rejected_coeffs);

        return {grad_chosen, grad_rejected};
    }

private:
    TensorPtr policy_chosen_logits_;
    TensorPtr policy_rejected_logits_;
    TensorPtr chosen_input_ids_;
    TensorPtr rejected_input_ids_;
    TensorPtr chosen_response_mask_;
    TensorPtr rejected_response_mask_;
    std::vector<double> chosen_coeffs_;
    std::vector<double> rejected_coeffs_;
};

}  // namespace

std::vector<float> sequence_logprob_masked(const TensorPtr& logits,
                                           const TensorPtr& input_ids,
                                           const TensorPtr& response_mask) {
    return compute_sequence_logps(logits, input_ids, response_mask);
}

TensorPtr dpo_loss(const TensorPtr& policy_chosen_logits,
                   const TensorPtr& policy_rejected_logits,
                   const TensorPtr& ref_chosen_logits,
                   const TensorPtr& ref_rejected_logits,
                   const TensorPtr& chosen_input_ids,
                   const TensorPtr& rejected_input_ids,
                   const TensorPtr& chosen_response_mask,
                   const TensorPtr& rejected_response_mask,
                   float beta) {
    auto policy_chosen_logps = compute_sequence_logps(
        policy_chosen_logits, chosen_input_ids, chosen_response_mask);
    auto policy_rejected_logps = compute_sequence_logps(
        policy_rejected_logits, rejected_input_ids, rejected_response_mask);
    auto ref_chosen_logps = compute_sequence_logps(
        ref_chosen_logits, chosen_input_ids, chosen_response_mask);
    auto ref_rejected_logps = compute_sequence_logps(
        ref_rejected_logits, rejected_input_ids, rejected_response_mask);

    const size_t n = policy_chosen_logps.size();
    if (n == 0 || policy_rejected_logps.size() != n ||
        ref_chosen_logps.size() != n || ref_rejected_logps.size() != n) {
        throw std::runtime_error("dpo_loss: inconsistent batch sizes");
    }

    double loss_sum = 0.0;
    std::vector<double> chosen_coeffs(n, 0.0);
    std::vector<double> rejected_coeffs(n, 0.0);
    const double inv_n = 1.0 / static_cast<double>(n);

    for (size_t i = 0; i < n; ++i) {
        const double pi_logratio =
            static_cast<double>(policy_chosen_logps[i]) - static_cast<double>(policy_rejected_logps[i]);
        const double ref_logratio =
            static_cast<double>(ref_chosen_logps[i]) - static_cast<double>(ref_rejected_logps[i]);
        const double logit = static_cast<double>(beta) * (pi_logratio - ref_logratio);

        loss_sum += negative_log_sigmoid(logit);

        const double dloss_dlogit = sigmoid_minus_one(logit);
        const double dloss_dseq = dloss_dlogit * static_cast<double>(beta) * inv_n;
        chosen_coeffs[i] = dloss_dseq;
        rejected_coeffs[i] = -dloss_dseq;
    }

    auto out = full({1}, static_cast<float>(loss_sum * inv_n), kFloat32, kCPU);
    if (policy_chosen_logits->requires_grad() || policy_rejected_logits->requires_grad()) {
        out->set_requires_grad(true);
        auto backward_fn = std::make_shared<DPOBackward>(
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_input_ids,
            rejected_input_ids,
            chosen_response_mask,
            rejected_response_mask,
            std::move(chosen_coeffs),
            std::move(rejected_coeffs));
#ifdef USE_NEW_AUTOGRAD_ENGINE
        std::vector<TensorPtr> inputs = {policy_chosen_logits, policy_rejected_logits};
        autograd::Engine::instance().register_node(out, inputs, backward_fn);
#else
        out->set_grad_fn([backward_fn, policy_chosen_logits, policy_rejected_logits](const TensorPtr& grad_output)
                             -> std::vector<TensorPtr> {
            auto grads = backward_fn->apply(grad_output);
            if (policy_chosen_logits->requires_grad()) accumulate_gradient(policy_chosen_logits, grads[0]);
            if (policy_rejected_logits->requires_grad()) accumulate_gradient(policy_rejected_logits, grads[1]);
            return grads;
        });
#endif
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
            static_cast<double>(beta) *
            (static_cast<double>(policy_chosen_logps[i]) - static_cast<double>(ref_chosen_logps[i]));
        const double rejected_reward =
            static_cast<double>(beta) *
            (static_cast<double>(policy_rejected_logps[i]) - static_cast<double>(ref_rejected_logps[i]));
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
