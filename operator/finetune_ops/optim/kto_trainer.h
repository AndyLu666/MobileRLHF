#pragma once

#include "../core/dpo_loss.h"
#include "../data/kto_batch.h"
#include "../data/preference_batch.h"
#include "../graph/auto_model.h"
#include "adam.h"

#include <memory>

namespace ops {

struct KTOTrainerConfig {
    float learning_rate = 2e-4f;
    float weight_decay = 0.0f;
    float max_grad_norm = 1.0f;
    float beta = 0.1f;
    float desirable_weight = 1.0f;
    float undesirable_weight = 1.0f;
    bool use_streaming_kto_loss = true;
    int gradient_accumulation_steps = 1;
};

struct KTOTrainStepResult {
    float loss = 0.0f;
    float accumulated_loss = 0.0f;
    float desirable_reward = 0.0f;
    float undesirable_reward = 0.0f;
    float reward_margin = 0.0f;
    float reward_accuracy = 0.0f;
    float kl_estimate = 0.0f;
    int trainable_tensor_count = 0;
    int pair_count = 0;
    int sample_count = 0;
    int desirable_count = 0;
    int undesirable_count = 0;
    int valid_response_token_count = 0;
    int valid_kl_response_token_count = 0;
    int accumulation_step = 0;
    int gradient_accumulation_steps = 1;
    bool optimizer_step = true;
};

class KTOTrainer {
public:
    KTOTrainer(AutoModelForCausalLM& policy_model,
               const KTOTrainerConfig& config = KTOTrainerConfig());

    KTOTrainer(AutoModelForCausalLM& policy_model,
               AutoModelForCausalLM& reference_model,
               const KTOTrainerConfig& config = KTOTrainerConfig());

    /**
     * @brief Train on canonical unpaired KTO data with a mismatched KL branch.
     */
    KTOTrainStepResult train_step(const KTOBatch& batch);

    /**
     * @brief Legacy pair-to-KTO approximation retained for source compatibility.
     *
     * This path expands chosen/rejected preference pairs and estimates the KL
     * center from the same matched rows. New code should build KTOBatch and use
     * the canonical overload above.
     */
    KTOTrainStepResult train_step(const PreferenceBatch& batch);

private:
    AutoModelForCausalLM& policy_model_;
    AutoModelForCausalLM* reference_model_ = nullptr;
    KTOTrainerConfig config_;
    std::unique_ptr<Adam> optimizer_;
    int accum_counter_ = 0;
    float accum_loss_ = 0.0f;
};

}  // namespace ops
