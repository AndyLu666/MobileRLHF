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

class CharacterTokenizer final : public ops::Tokenizer {
public:
    std::vector<int> encode(const std::string& text) override {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (unsigned char value : text) {
            ids.push_back(static_cast<int>(value));
        }
        return ids;
    }

    std::string decode(const std::vector<int>& tokens) override {
        std::string text;
        text.reserve(tokens.size());
        for (int token : tokens) {
            text.push_back(static_cast<char>(token));
        }
        return text;
    }

    int get_vocab_size() const override { return 256; }
    int get_eos_token() const override { return 255; }
    int get_bos_token() const override { return 254; }
    int get_pad_token() const override { return 0; }
    int get_unk_token() const override { return 253; }
};

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

ops::KTOBatch make_cached_canonical_kto_batch() {
    ops::KTOBatch batch;
    const std::vector<int64_t> shape{2, 4};
    const int32_t ids[] = {1, 2, 3, 0, 1, 4, 5, 0};
    const int32_t kl_ids[] = {1, 4, 5, 0, 1, 2, 3, 0};
    const float attention[] = {
        1.0f, 1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 0.0f,
    };
    const int32_t response_mask[] = {0, 1, 1, 0, 0, 1, 1, 0};

    batch.input_ids =
        std::make_shared<ops::Tensor>(shape, ids, ops::kInt32, ops::kCPU);
    batch.kl_input_ids =
        std::make_shared<ops::Tensor>(shape, kl_ids, ops::kInt32, ops::kCPU);
    batch.attention_mask =
        std::make_shared<ops::Tensor>(shape, attention, ops::kFloat32, ops::kCPU);
    batch.kl_attention_mask =
        std::make_shared<ops::Tensor>(shape, attention, ops::kFloat32, ops::kCPU);
    batch.response_mask =
        std::make_shared<ops::Tensor>(shape, response_mask, ops::kInt32, ops::kCPU);
    batch.kl_response_mask =
        std::make_shared<ops::Tensor>(shape, response_mask, ops::kInt32, ops::kCPU);
    batch.labels = {1.0f, -1.0f};
    batch.ref_logps = {-2.0f, -2.1f};
    batch.ref_kl_logps = {-2.2f, -2.3f};
    batch.sample_indices = {0, 1};
    batch.kl_completion_indices = {1, 0};
    batch.batch_size = 2;
    batch.sequence_length = 4;
    batch.valid_response_token_count = 4;
    batch.valid_kl_response_token_count = 4;
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

void test_kto_batch_rotates_three_completions_and_rejects_singleton() {
    CharacterTokenizer tokenizer;
    ops::KTOBatchConfig config;
    config.sequence_length = 3;
    const std::vector<ops::KTOSample> samples = {
        {"a", "X", true, true, -1.0f, -1.1f},
        {"b", "Y", false, true, -2.0f, -2.1f},
        {"c", "Z", true, true, -3.0f, -3.1f},
    };
    const ops::KTOBatch batch = ops::make_kto_batch(tokenizer, samples, config);

    require(batch.batch_size == 3, "canonical KTO batch size mismatch");
    require(batch.kl_completion_indices == std::vector<std::size_t>({2, 0, 1}),
            "canonical KTO completion rotation mismatch");
    require(batch.labels == std::vector<float>({1.0f, -1.0f, 1.0f}),
            "canonical KTO labels mismatch");
    require(batch.has_cached_reference_logps(),
            "canonical KTO complete reference cache not detected");
    const int32_t* matched = batch.input_ids->data<int32_t>();
    const int32_t* mismatched = batch.kl_input_ids->data<int32_t>();
    for (int i = 0; i < 3; ++i) {
        require(matched[i * 3] == static_cast<int32_t>('a' + i),
                "canonical KTO matched prompt mismatch");
        require(mismatched[i * 3] == static_cast<int32_t>('a' + i),
                "canonical KTO KL prompt mismatch");
    }
    require(matched[1] == 'X' && matched[4] == 'Y' && matched[7] == 'Z',
            "canonical KTO matched completions mismatch");
    require(mismatched[1] == 'Z' && mismatched[4] == 'X' && mismatched[7] == 'Y',
            "canonical KTO mismatched completions mismatch");

    bool rejected_singleton = false;
    try {
        (void)ops::make_kto_batch(tokenizer, {samples.front()}, config);
    } catch (const std::invalid_argument&) {
        rejected_singleton = true;
    }
    require(rejected_singleton, "canonical KTO must reject B=1 batches");
}

void test_canonical_kto_trainer_updates_lora() {
    const fs::path root = make_case_dir();
    write_file(root / "config.json",
               R"({"model_type":"gpt2","vocab_size":16,"n_positions":8,"n_embd":8,"n_layer":1,"n_head":2})");

    auto model = ops::AutoModelForCausalLM::from_pretrained(
        root.string(), no_weight_load());
    fill_parameters(*model);
    model->init_lora(tiny_lora());

    std::vector<ops::TensorPtr> before;
    for (const auto& param : model->trainable_parameters()) {
        before.push_back(param->clone());
    }

    ops::KTOTrainerConfig config;
    config.learning_rate = 1e-2f;
    config.beta = 0.1f;
    config.use_streaming_kto_loss = true;
    ops::KTOTrainer trainer(*model, config);
    const auto result = trainer.train_step(make_cached_canonical_kto_batch());

    require(std::isfinite(result.loss), "canonical KTOTrainer loss is not finite");
    require(std::isfinite(result.kl_estimate) && result.kl_estimate >= 0.0f,
            "canonical KTOTrainer KL estimate is invalid");
    require(result.pair_count == 0, "canonical KTOTrainer must report zero pairs");
    require(result.sample_count == 2, "canonical KTOTrainer sample count mismatch");
    require(result.desirable_count == 1 && result.undesirable_count == 1,
            "canonical KTOTrainer label counts mismatch");
    require(result.valid_response_token_count == 4 &&
                result.valid_kl_response_token_count == 4,
            "canonical KTOTrainer token counts mismatch");
    require(max_param_delta(before, model->trainable_parameters()) > 0.0f,
            "canonical KTOTrainer did not update LoRA parameters");

    fs::remove_all(root);
}

}  // namespace

int main() {
    test_kto_trainer_updates_lora_from_preference_pair();
    test_reference_logps_are_scored_from_model_batch();
    test_kto_batch_rotates_three_completions_and_rejects_singleton();
    test_canonical_kto_trainer_updates_lora();
    return 0;
}
