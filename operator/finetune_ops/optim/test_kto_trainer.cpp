#include "kto_trainer.h"
#include "preference_scoring.h"
#include "../graph/auto_model.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

void write_file(const fs::path& path, const std::string& content) {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to write test file: " + path.string());
    }
    f << content;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

fs::path make_case_dir() {
    const fs::path root = fs::temp_directory_path() / "mft_kto_trainer";
    fs::remove_all(root);
    fs::create_directories(root);
    return root;
}

ops::AutoModelLoadOptions no_weight_load() {
    ops::AutoModelLoadOptions options;
    options.load_weights = false;
    options.verbose = false;
    return options;
}

void fill_parameters(ops::AutoModelForCausalLM& model) {
    int offset = 0;
    for (const auto& param : model.parameters()) {
        if (!param || param->dtype() != ops::kFloat32) {
            continue;
        }
        float* data = param->data<float>();
        for (int64_t i = 0; i < param->numel(); ++i) {
            data[i] = 0.01f + 0.001f * static_cast<float>((offset + i) % 17);
        }
        offset += static_cast<int>(param->numel());
    }
}

ops::AutoLoraConfig tiny_lora() {
    ops::AutoLoraConfig cfg;
    cfg.rank = 2;
    cfg.alpha = 4.0f;
    cfg.dropout = 0.0f;
    cfg.seed = 11;
    return cfg;
}

ops::PreferenceBatch make_cached_ref_pair_batch() {
    ops::PreferenceBatch batch;
    const std::vector<int64_t> shape{1, 4};
    const int32_t chosen_ids[] = {1, 2, 3, 0};
    const int32_t rejected_ids[] = {1, 4, 5, 0};
    const float attention[] = {1.0f, 1.0f, 1.0f, 0.0f};
    const int32_t response_mask[] = {0, 1, 1, 0};

    batch.chosen_input_ids = std::make_shared<ops::Tensor>(shape, chosen_ids, ops::kInt32, ops::kCPU);
    batch.rejected_input_ids = std::make_shared<ops::Tensor>(shape, rejected_ids, ops::kInt32, ops::kCPU);
    batch.chosen_attention_mask = std::make_shared<ops::Tensor>(shape, attention, ops::kFloat32, ops::kCPU);
    batch.rejected_attention_mask = std::make_shared<ops::Tensor>(shape, attention, ops::kFloat32, ops::kCPU);
    batch.chosen_response_mask = std::make_shared<ops::Tensor>(shape, response_mask, ops::kInt32, ops::kCPU);
    batch.rejected_response_mask = std::make_shared<ops::Tensor>(shape, response_mask, ops::kInt32, ops::kCPU);
    batch.ref_chosen_logps = {-2.0f};
    batch.ref_rejected_logps = {-2.1f};
    batch.sample_indices = {0};
    batch.batch_size = 1;
    batch.sequence_length = 4;
    batch.valid_response_token_count = 4;
    return batch;
}

float max_param_delta(const std::vector<ops::TensorPtr>& before,
                      const std::vector<ops::TensorPtr>& after) {
    require(before.size() == after.size(), "parameter vector size mismatch");
    float max_delta = 0.0f;
    for (std::size_t i = 0; i < before.size(); ++i) {
        const float* a = before[i]->data<float>();
        const float* b = after[i]->data<float>();
        for (int64_t j = 0; j < before[i]->numel(); ++j) {
            max_delta = std::max(max_delta, std::fabs(a[j] - b[j]));
        }
    }
    return max_delta;
}

void test_kto_trainer_updates_lora_from_preference_pair() {
    const fs::path root = make_case_dir();
    write_file(root / "config.json",
               R"({"model_type":"gpt2","vocab_size":16,"n_positions":8,"n_embd":8,"n_layer":1,"n_head":2})");

    auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    fill_parameters(*model);
    model->init_lora(tiny_lora());

    std::vector<ops::TensorPtr> before;
    for (const auto& param : model->trainable_parameters()) {
        before.push_back(param->clone());
    }
    require(!before.empty(), "KTOTrainer test missing LoRA parameters");

    ops::KTOTrainerConfig cfg;
    cfg.learning_rate = 1e-2f;
    cfg.beta = 0.1f;
    cfg.use_streaming_kto_loss = true;
    ops::KTOTrainer trainer(*model, cfg);
    const auto result = trainer.train_step(make_cached_ref_pair_batch());

    require(std::isfinite(result.loss), "KTOTrainer loss is not finite");
    require(result.trainable_tensor_count > 0, "KTOTrainer trainable count missing");
    require(result.sample_count == 2, "KTOTrainer should expand one pair to two KTO samples");
    require(result.optimizer_step, "KTOTrainer default step should update optimizer");
    require(max_param_delta(before, model->trainable_parameters()) > 0.0f,
            "KTOTrainer did not update LoRA parameters");

    fs::remove_all(root);
}

void test_reference_logps_are_scored_from_model_batch() {
    const fs::path root = make_case_dir();
    write_file(root / "config.json",
               R"({"model_type":"gpt2","vocab_size":16,"n_positions":8,"n_embd":8,"n_layer":1,"n_head":2})");

    auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), no_weight_load());
    fill_parameters(*model);

    const auto scores = ops::score_preference_logps(*model, make_cached_ref_pair_batch());

    require(scores.chosen_logps.size() == 1, "reference scoring chosen size mismatch");
    require(scores.rejected_logps.size() == 1, "reference scoring rejected size mismatch");
    require(std::isfinite(scores.chosen_logps[0]), "chosen reference logp is not finite");
    require(std::isfinite(scores.rejected_logps[0]), "rejected reference logp is not finite");
    require(scores.valid_response_token_count == 4, "reference scoring valid token count mismatch");

    fs::remove_all(root);
}

}  // namespace

int main() {
    test_kto_trainer_updates_lora_from_preference_pair();
    test_reference_logps_are_scored_from_model_batch();
    return 0;
}
