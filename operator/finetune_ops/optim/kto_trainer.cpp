#include "kto_trainer.h"

#include "../core/ops.h"
#include "preference_scoring.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ops {

namespace {

struct LegacyKTOBatch {
    TensorPtr input_ids;
    TensorPtr attention_mask;
    TensorPtr response_mask;
    std::vector<float> labels;
    std::vector<float> ref_logps;
    int pair_count = 0;
    int sample_count = 0;
    int sequence_length = 0;
    int valid_response_token_count = 0;
};

void validate_config(const KTOTrainerConfig& config) {
    if (!(config.beta > 0.0f)) {
        throw std::invalid_argument("KTOTrainer requires beta > 0");
    }
    if (!(config.desirable_weight > 0.0f) || !(config.undesirable_weight > 0.0f)) {
        throw std::invalid_argument("KTOTrainer requires positive KTO weights");
    }
    if (config.gradient_accumulation_steps <= 0) {
        throw std::invalid_argument("KTOTrainer requires gradient_accumulation_steps > 0");
    }
}

void validate_batch(const PreferenceBatch& batch) {
    if (!batch.chosen_input_ids || !batch.chosen_attention_mask || !batch.chosen_response_mask ||
        !batch.rejected_input_ids || !batch.rejected_attention_mask || !batch.rejected_response_mask) {
        throw std::invalid_argument("KTOTrainer::train_step received an incomplete PreferenceBatch");
    }
    if (batch.batch_size <= 0 || batch.sequence_length <= 1) {
        throw std::invalid_argument("KTOTrainer::train_step requires at least one preference pair");
    }
}

void validate_batch(const KTOBatch& batch) {
    if (!batch.input_ids || !batch.attention_mask || !batch.response_mask ||
        !batch.kl_input_ids || !batch.kl_attention_mask ||
        !batch.kl_response_mask) {
        throw std::invalid_argument(
            "KTOTrainer::train_step received an incomplete KTOBatch");
    }
    if (batch.batch_size < 2 || batch.sequence_length <= 1) {
        throw std::invalid_argument(
            "KTOTrainer::train_step requires at least two KTO samples");
    }
    if (static_cast<int>(batch.labels.size()) != batch.batch_size) {
        throw std::invalid_argument("KTOTrainer::train_step KTO label count mismatch");
    }
    for (float label : batch.labels) {
        if (label != 1.0f && label != -1.0f) {
            throw std::invalid_argument(
                "KTOTrainer::train_step labels must be exactly +1 or -1");
        }
    }
}

template <typename T>
void copy_rows(T* dst,
               const T* chosen,
               const T* rejected,
               int batch_size,
               int sequence_length) {
    const std::size_t row_count = static_cast<std::size_t>(batch_size) *
                                  static_cast<std::size_t>(sequence_length);
    std::memcpy(dst, chosen, row_count * sizeof(T));
    std::memcpy(dst + row_count, rejected, row_count * sizeof(T));
}

LegacyKTOBatch expand_preference_batch(const PreferenceBatch& batch) {
    validate_batch(batch);
    const int pair_count = batch.batch_size;
    const int seq = batch.sequence_length;
    const int samples = pair_count * 2;
    const std::vector<int64_t> shape{samples, seq};

    LegacyKTOBatch out;
    out.input_ids = std::make_shared<Tensor>(shape, kInt32, kCPU);
    out.attention_mask = std::make_shared<Tensor>(shape, kFloat32, kCPU);
    out.response_mask = std::make_shared<Tensor>(shape, kInt32, kCPU);
    out.labels.reserve(static_cast<std::size_t>(samples));
    out.pair_count = pair_count;
    out.sample_count = samples;
    out.sequence_length = seq;
    out.valid_response_token_count = batch.valid_response_token_count;

    copy_rows(out.input_ids->data<int32_t>(),
              batch.chosen_input_ids->data<int32_t>(),
              batch.rejected_input_ids->data<int32_t>(),
              pair_count,
              seq);
    copy_rows(out.attention_mask->data<float>(),
              batch.chosen_attention_mask->data<float>(),
              batch.rejected_attention_mask->data<float>(),
              pair_count,
              seq);
    copy_rows(out.response_mask->data<int32_t>(),
              batch.chosen_response_mask->data<int32_t>(),
              batch.rejected_response_mask->data<int32_t>(),
              pair_count,
              seq);

    for (int i = 0; i < pair_count; ++i) {
        out.labels.push_back(1.0f);
    }
    for (int i = 0; i < pair_count; ++i) {
        out.labels.push_back(-1.0f);
    }

    if (batch.has_cached_reference_logps()) {
        out.ref_logps.reserve(static_cast<std::size_t>(samples));
        out.ref_logps.insert(out.ref_logps.end(),
                             batch.ref_chosen_logps.begin(),
                             batch.ref_chosen_logps.end());
        out.ref_logps.insert(out.ref_logps.end(),
                             batch.ref_rejected_logps.begin(),
                             batch.ref_rejected_logps.end());
    }
    return out;
}

}  // namespace

KTOTrainer::KTOTrainer(AutoModelForCausalLM& policy_model,
                       const KTOTrainerConfig& config)
    : policy_model_(policy_model), config_(config) {
    validate_config(config_);
    AdamConfig adam_cfg;
    adam_cfg.learning_rate = config_.learning_rate;
    adam_cfg.weight_decay = config_.weight_decay;
    adam_cfg.clip_grad_norm = config_.max_grad_norm;
    optimizer_ = std::make_unique<Adam>(adam_cfg);
}

KTOTrainer::KTOTrainer(AutoModelForCausalLM& policy_model,
                       AutoModelForCausalLM& reference_model,
                       const KTOTrainerConfig& config)
    : policy_model_(policy_model), reference_model_(&reference_model), config_(config) {
    validate_config(config_);
    AdamConfig adam_cfg;
    adam_cfg.learning_rate = config_.learning_rate;
    adam_cfg.weight_decay = config_.weight_decay;
    adam_cfg.clip_grad_norm = config_.max_grad_norm;
    optimizer_ = std::make_unique<Adam>(adam_cfg);
}

KTOTrainStepResult KTOTrainer::train_step(const KTOBatch& batch) {
    validate_batch(batch);
    auto params = policy_model_.trainable_parameters();
    if (params.empty()) {
        throw std::runtime_error(
            "KTOTrainer::train_step requires trainable parameters; call init_lora first");
    }

    std::vector<float> ref_logps = batch.ref_logps;
    std::vector<float> ref_kl_logps = batch.ref_kl_logps;
    if (!batch.has_cached_reference_logps()) {
        if (!reference_model_) {
            throw std::invalid_argument(
                "KTOTrainer::train_step requires complete cached main+KL reference "
                "logps or a reference model");
        }
        const KTOLogps reference_scores = score_kto_logps(*reference_model_, batch);
        ref_logps = reference_scores.response_logps;
        ref_kl_logps = reference_scores.kl_response_logps;
    }

    TensorPtr loss;
    DPOMetrics metrics;
    if (config_.use_streaming_kto_loss) {
        const TensorPtr hidden =
            policy_model_.forward_hidden(batch.input_ids, batch.attention_mask);
        const TensorPtr kl_hidden =
            policy_model_.forward_hidden(batch.kl_input_ids, batch.kl_attention_mask);
        const TensorPtr lm_head_weight = policy_model_.lm_head_weight_for_loss();
        loss = streaming_kto_loss_with_ref_logps_and_kl(
            hidden,
            kl_hidden,
            lm_head_weight,
            ref_logps,
            ref_kl_logps,
            batch.input_ids,
            batch.kl_input_ids,
            batch.response_mask,
            batch.kl_response_mask,
            batch.labels,
            config_.beta,
            config_.desirable_weight,
            config_.undesirable_weight);
        metrics = compute_kto_metrics_from_policy_hidden_and_ref_logps_and_kl(
            hidden,
            kl_hidden,
            lm_head_weight,
            ref_logps,
            ref_kl_logps,
            batch.input_ids,
            batch.kl_input_ids,
            batch.response_mask,
            batch.kl_response_mask,
            batch.labels,
            config_.beta,
            config_.desirable_weight,
            config_.undesirable_weight);
    } else {
        const TensorPtr logits =
            policy_model_.forward(batch.input_ids, batch.attention_mask);
        const TensorPtr kl_logits =
            policy_model_.forward(batch.kl_input_ids, batch.kl_attention_mask);
        loss = kto_loss_with_ref_logps_and_kl(
            logits,
            kl_logits,
            ref_logps,
            ref_kl_logps,
            batch.input_ids,
            batch.kl_input_ids,
            batch.response_mask,
            batch.kl_response_mask,
            batch.labels,
            config_.beta,
            config_.desirable_weight,
            config_.undesirable_weight);
        metrics = compute_kto_metrics_from_logps_and_kl(
            sequence_logprob_masked(logits, batch.input_ids, batch.response_mask),
            sequence_logprob_masked(
                kl_logits, batch.kl_input_ids, batch.kl_response_mask),
            ref_logps,
            ref_kl_logps,
            batch.labels,
            config_.beta,
            config_.desirable_weight,
            config_.undesirable_weight);
    }

    const float raw_loss = loss->data<float>()[0];
    TensorPtr backward_loss = loss;
    if (config_.gradient_accumulation_steps > 1) {
        backward_loss = mul(
            loss, 1.0f / static_cast<float>(config_.gradient_accumulation_steps));
    }
    backward_loss->backward();

    accum_counter_ += 1;
    accum_loss_ += raw_loss;

    KTOTrainStepResult result;
    result.loss = raw_loss;
    result.desirable_reward = metrics.chosen_reward;
    result.undesirable_reward = metrics.rejected_reward;
    result.reward_margin = metrics.reward_margin;
    result.reward_accuracy = metrics.reward_accuracy;
    result.kl_estimate = metrics.kl_estimate;
    result.trainable_tensor_count = static_cast<int>(params.size());
    result.pair_count = 0;
    result.sample_count = batch.batch_size;
    result.valid_response_token_count = batch.valid_response_token_count;
    result.valid_kl_response_token_count = batch.valid_kl_response_token_count;
    for (float label : batch.labels) {
        if (label > 0.0f) {
            ++result.desirable_count;
        } else {
            ++result.undesirable_count;
        }
    }
    result.accumulation_step = accum_counter_;
    result.gradient_accumulation_steps = config_.gradient_accumulation_steps;
    result.optimizer_step = accum_counter_ >= config_.gradient_accumulation_steps;

    if (!result.optimizer_step) {
        return result;
    }

    optimizer_->clip_grad_norm(params, config_.max_grad_norm);
    std::vector<TensorPtr> grads;
    grads.reserve(params.size());
    for (const auto& param : params) {
        grads.push_back(param->grad());
    }
    optimizer_->step(params, grads);
    optimizer_->zero_grad(params);

    result.accumulated_loss =
        accum_loss_ / static_cast<float>(config_.gradient_accumulation_steps);
    accum_counter_ = 0;
    accum_loss_ = 0.0f;
    return result;
}

KTOTrainStepResult KTOTrainer::train_step(const PreferenceBatch& batch) {
    validate_batch(batch);
    auto params = policy_model_.trainable_parameters();
    if (params.empty()) {
        throw std::runtime_error("KTOTrainer::train_step requires trainable parameters; call init_lora first");
    }

    LegacyKTOBatch kto_batch = expand_preference_batch(batch);
    if (!batch.has_cached_reference_logps()) {
        if (!reference_model_) {
            throw std::invalid_argument(
                "KTOTrainer::train_step requires cached reference logps or a reference model");
        }
        auto ref_hidden = reference_model_->forward_hidden(kto_batch.input_ids, kto_batch.attention_mask);
        kto_batch.ref_logps = sequence_logprob_masked_from_hidden(
            ref_hidden,
            reference_model_->lm_head_weight_for_loss(),
            kto_batch.input_ids,
            kto_batch.response_mask);
    }

    TensorPtr loss;
    DPOMetrics metrics;
    if (config_.use_streaming_kto_loss) {
        auto hidden = policy_model_.forward_hidden(kto_batch.input_ids, kto_batch.attention_mask);
        auto lm_head_weight = policy_model_.lm_head_weight_for_loss();
        loss = streaming_kto_loss_with_ref_logps(
            hidden,
            lm_head_weight,
            kto_batch.ref_logps,
            kto_batch.input_ids,
            kto_batch.response_mask,
            kto_batch.labels,
            config_.beta,
            config_.desirable_weight,
            config_.undesirable_weight);
        metrics = compute_kto_metrics_from_policy_hidden_and_ref_logps(
            hidden,
            lm_head_weight,
            kto_batch.ref_logps,
            kto_batch.input_ids,
            kto_batch.response_mask,
            kto_batch.labels,
            config_.beta,
            config_.desirable_weight,
            config_.undesirable_weight);
    } else {
        auto logits = policy_model_.forward(kto_batch.input_ids, kto_batch.attention_mask);
        loss = kto_loss_with_ref_logps(
            logits,
            kto_batch.ref_logps,
            kto_batch.input_ids,
            kto_batch.response_mask,
            kto_batch.labels,
            config_.beta,
            config_.desirable_weight,
            config_.undesirable_weight);
        metrics = compute_kto_metrics_from_logps(
            sequence_logprob_masked(logits, kto_batch.input_ids, kto_batch.response_mask),
            kto_batch.ref_logps,
            kto_batch.labels,
            config_.beta,
            config_.desirable_weight,
            config_.undesirable_weight);
    }

    const float raw_loss = loss->data<float>()[0];
    TensorPtr backward_loss = loss;
    if (config_.gradient_accumulation_steps > 1) {
        backward_loss = mul(loss, 1.0f / static_cast<float>(config_.gradient_accumulation_steps));
    }
    backward_loss->backward();

    accum_counter_ += 1;
    accum_loss_ += raw_loss;

    KTOTrainStepResult result;
    result.loss = raw_loss;
    result.desirable_reward = metrics.chosen_reward;
    result.undesirable_reward = metrics.rejected_reward;
    result.reward_margin = metrics.reward_margin;
    result.reward_accuracy = metrics.reward_accuracy;
    result.trainable_tensor_count = static_cast<int>(params.size());
    result.pair_count = kto_batch.pair_count;
    result.sample_count = kto_batch.sample_count;
    result.desirable_count = kto_batch.pair_count;
    result.undesirable_count = kto_batch.pair_count;
    result.valid_response_token_count = kto_batch.valid_response_token_count;
    result.accumulation_step = accum_counter_;
    result.gradient_accumulation_steps = config_.gradient_accumulation_steps;
    result.optimizer_step = accum_counter_ >= config_.gradient_accumulation_steps;

    if (!result.optimizer_step) {
        return result;
    }

    optimizer_->clip_grad_norm(params, config_.max_grad_norm);
    std::vector<TensorPtr> grads;
    grads.reserve(params.size());
    for (const auto& param : params) {
        grads.push_back(param->grad());
    }
    optimizer_->step(params, grads);
    optimizer_->zero_grad(params);

    result.accumulated_loss = accum_loss_ / static_cast<float>(config_.gradient_accumulation_steps);
    accum_counter_ = 0;
    accum_loss_ = 0.0f;
    return result;
}

}  // namespace ops
