#define MFT_C_API_ONLY 1
#include "mobile_finetuner/mobile_finetuner.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

using ModelPtr = std::unique_ptr<mft_model, decltype(&mft_model_free)>;
using TokenizerPtr = std::unique_ptr<mft_tokenizer, decltype(&mft_tokenizer_free)>;
using TrainerPtr = std::unique_ptr<mft_trainer, decltype(&mft_trainer_free)>;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_status(mft_status actual,
                    mft_status expected,
                    const std::string& operation) {
    if (actual != expected) {
        const char* error = mft_last_error();
        throw std::runtime_error(
            operation + " returned " + mft_status_string(actual) +
            ", expected " + mft_status_string(expected) +
            (error != nullptr && error[0] != '\0'
                 ? std::string(" (last error: ") + error + ")"
                 : std::string()));
    }
}

mft_string_view view(const char* text) {
    return mft_string_view{text, std::strlen(text)};
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto serial = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = fs::temp_directory_path() /
                ("mft_c_api_runtime_" + std::to_string(serial));
        fs::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

void write_file(const fs::path& path, const std::string& contents) {
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    require(output.is_open(), "failed to create fixture: " + path.string());
    output << contents;
    output.close();
    require(output.good(), "failed to write fixture: " + path.string());
}

void write_tiny_gpt2_assets(const fs::path& root) {
    write_file(
        root / "config.json",
        R"({"model_type":"gpt2","vocab_size":16,"n_positions":8,"n_embd":8,"n_layer":1,"n_head":2})");
    // Every runtime-test string is made only from these one-byte tokens. The
    // fixed sequence length is exact, so GPT-2's conventional 50256 pad id is
    // never presented to the tiny 16-token model.
    write_file(
        root / "vocab.json",
        R"({"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8,"<|endoftext|>":15})");
    write_file(root / "merges.txt", "#version: 0.2\n");
}

void test_thread_local_errors() {
    mft_model_family family = MFT_MODEL_FAMILY_UNKNOWN;
    require_status(
        mft_model_get_family(nullptr, &family),
        MFT_STATUS_INVALID_ARGUMENT,
        "main-thread invalid model call");
    const std::string main_error = mft_last_error();
    require(!main_error.empty(), "main thread did not retain an error");

    bool new_thread_started_empty = false;
    std::string worker_error;
    std::thread worker([&]() {
        const char* initial = mft_last_error();
        new_thread_started_empty = initial == nullptr || initial[0] == '\0';
        mft_trainer_kind kind = MFT_TRAINER_KIND_UNKNOWN;
        const mft_status status = mft_trainer_get_kind(nullptr, &kind);
        if (status == MFT_STATUS_INVALID_ARGUMENT) {
            const char* error = mft_last_error();
            if (error != nullptr) {
                worker_error = error;
            }
        }
    });
    worker.join();

    require(new_thread_started_empty,
            "new thread inherited another thread's last error");
    require(!worker_error.empty(), "worker thread did not retain its own error");
    require(std::string(mft_last_error()) == main_error,
            "worker thread overwrote the main thread's last error");
}

void test_tokenizer_buffer_contract(mft_tokenizer* tokenizer) {
    mft_tokenizer_encode_params encode_params =
        mft_tokenizer_encode_default_params();
    size_t token_count = 0;
    require_status(
        mft_tokenizer_encode(
            tokenizer, view("ab"), &encode_params, nullptr, 0, &token_count),
        MFT_STATUS_BUFFER_TOO_SMALL,
        "tokenizer encode size query");
    require(token_count == 2, "unexpected encoded token count");

    int32_t short_tokens[1] = {-1};
    size_t short_count = 0;
    require_status(
        mft_tokenizer_encode(
            tokenizer,
            view("ab"),
            &encode_params,
            short_tokens,
            1,
            &short_count),
        MFT_STATUS_BUFFER_TOO_SMALL,
        "tokenizer encode short buffer");
    require(short_count == token_count,
            "encode short buffer did not report the required count");

    std::vector<int32_t> tokens(token_count);
    require_status(
        mft_tokenizer_encode(
            tokenizer,
            view("ab"),
            &encode_params,
            tokens.data(),
            tokens.size(),
            &token_count),
        MFT_STATUS_OK,
        "tokenizer encode");
    require(tokens[0] == 1 && tokens[1] == 2,
            "tokenizer returned unexpected IDs");

    mft_tokenizer_decode_params decode_params =
        mft_tokenizer_decode_default_params();
    size_t text_size = 0;
    require_status(
        mft_tokenizer_decode(
            tokenizer,
            tokens.data(),
            tokens.size(),
            &decode_params,
            nullptr,
            0,
            &text_size),
        MFT_STATUS_BUFFER_TOO_SMALL,
        "tokenizer decode size query");
    require(text_size == 2, "unexpected decoded text size");

    char short_text[2] = {'x', 'x'};
    size_t short_text_size = 0;
    require_status(
        mft_tokenizer_decode(
            tokenizer,
            tokens.data(),
            tokens.size(),
            &decode_params,
            short_text,
            sizeof(short_text),
            &short_text_size),
        MFT_STATUS_BUFFER_TOO_SMALL,
        "tokenizer decode short buffer");
    require(short_text_size == text_size,
            "decode short buffer did not report the required size");

    std::vector<char> text(text_size + 1, '\0');
    require_status(
        mft_tokenizer_decode(
            tokenizer,
            tokens.data(),
            tokens.size(),
            &decode_params,
            text.data(),
            text.size(),
            &text_size),
        MFT_STATUS_OK,
        "tokenizer decode");
    require(std::string(text.data()) == "ab", "tokenizer decode mismatch");
}

void test_training_api(const fs::path& model_dir) {
    mft_model* raw_model = nullptr;
    mft_model_params model_params = mft_model_default_params();
    model_params.load_weights = false;
    // Simulate an older caller whose parameter structure ended before the
    // newer validation flags. Missing tail fields must retain library defaults.
    model_params.struct_size = offsetof(mft_model_params, verbose);
    require_status(
        mft_model_load_from_dir(model_dir.c_str(), &model_params, &raw_model),
        MFT_STATUS_OK,
        "load tiny GPT-2 model");
    ModelPtr model(raw_model, &mft_model_free);
    require(model != nullptr, "model load returned a null handle");

    mft_model_family family = MFT_MODEL_FAMILY_UNKNOWN;
    require_status(
        mft_model_get_family(model.get(), &family),
        MFT_STATUS_OK,
        "query model family");
    require(family == MFT_MODEL_FAMILY_GPT2,
            "tiny model did not dispatch to GPT-2");

    mft_lora_params lora_params = mft_lora_default_params();
    lora_params.rank = 2;
    lora_params.alpha = 4.0f;
    lora_params.dropout = 0.0f;
    lora_params.seed = 7;
    require_status(
        mft_model_init_lora(model.get(), &lora_params),
        MFT_STATUS_OK,
        "initialize LoRA");
    size_t trainable_count = 0;
    require_status(
        mft_model_get_trainable_tensor_count(model.get(), &trainable_count),
        MFT_STATUS_OK,
        "query trainable tensor count");
    require(trainable_count > 0, "LoRA did not create trainable tensors");

    mft_tokenizer* raw_tokenizer = nullptr;
    mft_tokenizer_params tokenizer_params = mft_tokenizer_default_params();
    require_status(
        mft_tokenizer_load_from_dir(
            model_dir.c_str(), &tokenizer_params, &raw_tokenizer),
        MFT_STATUS_OK,
        "load tiny GPT-2 tokenizer");
    TokenizerPtr tokenizer(raw_tokenizer, &mft_tokenizer_free);
    require(tokenizer != nullptr, "tokenizer load returned a null handle");
    test_tokenizer_buffer_contract(tokenizer.get());

    mft_trainer* raw_sft = nullptr;
    mft_sft_trainer_params sft_params = mft_sft_trainer_default_params();
    sft_params.learning_rate = 1e-3f;
    require_status(
        mft_sft_trainer_create(model.get(), &sft_params, &raw_sft),
        MFT_STATUS_OK,
        "create SFT trainer");
    TrainerPtr sft(raw_sft, &mft_trainer_free);

    const int32_t input_ids[] = {1, 2, 3};
    const float attention_mask[] = {1.0f, 1.0f, 1.0f};
    const int32_t labels[] = {-100, 2, 3};
    const mft_token_batch token_batch{
        input_ids, attention_mask, labels, 3, 1, 3};
    mft_sft_step_result sft_result{};
    require_status(
        mft_sft_train_token_batch(sft.get(), &token_batch, &sft_result),
        MFT_STATUS_OK,
        "run SFT token batch");
    require(std::isfinite(sft_result.loss), "SFT loss is not finite");
    require(sft_result.valid_label_count == 2,
            "SFT valid-label count mismatch");
    require(sft_result.trainable_tensor_count > 0,
            "SFT result omitted trainable tensors");

    // A policy intentionally permits one active optimizer/trainer at a time.
    sft.reset();

    const fs::path adapter_path = model_dir / "adapter.jsonl";
    require_status(
        mft_model_save_lora_adapter_jsonl(model.get(), adapter_path.c_str()),
        MFT_STATUS_OK,
        "save LoRA adapter JSONL");
    require(fs::is_regular_file(adapter_path) && fs::file_size(adapter_path) > 0,
            "adapter JSONL was not created");
    require_status(
        mft_model_load_lora_adapter_jsonl(model.get(), adapter_path.c_str()),
        MFT_STATUS_OK,
        "load LoRA adapter JSONL");
    const fs::path malformed_adapter_path = model_dir / "malformed-adapter.jsonl";
    write_file(malformed_adapter_path, "{\"format\":\"not-an-mft-adapter\"}\n");
    require_status(
        mft_model_load_lora_adapter_jsonl(
            model.get(), malformed_adapter_path.c_str()),
        MFT_STATUS_DATA_ERROR,
        "reject malformed LoRA adapter JSONL");

    const mft_preference_sample preference{
        view("a"), view("b"), view("c"), true, -2.0f, -2.1f};
    mft_preference_batch_params preference_params =
        mft_preference_batch_default_params();
    preference_params.sequence_length = 2;
    preference_params.truncate = true;
    preference_params.add_special_tokens = false;
    preference_params.append_eos_to_response = false;
    size_t preference_score_count = 0;
    int32_t preference_score_token_count = 0;
    require_status(
        mft_score_preference_batch(
            model.get(),
            tokenizer.get(),
            &preference,
            1,
            &preference_params,
            nullptr,
            nullptr,
            0,
            &preference_score_count,
            &preference_score_token_count),
        MFT_STATUS_BUFFER_TOO_SMALL,
        "query preference score output size");
    require(preference_score_count == 1,
            "preference score query did not report its output count");
    require(preference_score_token_count == 2,
            "preference score query did not report its valid-token count");

    const mft_kto_sample kto_samples[] = {
        {view("a"), view("b"), MFT_KTO_DESIRABLE, true, -2.0f, -2.2f},
        {view("c"), view("d"), MFT_KTO_UNDESIRABLE, true, -2.1f, -2.3f},
    };
    mft_kto_batch_params kto_batch_params = mft_kto_batch_default_params();
    kto_batch_params.sequence_length = 2;
    kto_batch_params.truncate = true;
    kto_batch_params.add_special_tokens = false;
    kto_batch_params.append_eos_to_completion = false;
    size_t kto_score_count = 0;
    int32_t kto_score_token_count = 0;
    int32_t kto_score_kl_token_count = 0;
    require_status(
        mft_score_kto_batch(
            model.get(),
            tokenizer.get(),
            kto_samples,
            2,
            &kto_batch_params,
            nullptr,
            nullptr,
            0,
            &kto_score_count,
            &kto_score_token_count,
            &kto_score_kl_token_count),
        MFT_STATUS_BUFFER_TOO_SMALL,
        "query KTO score output size");
    require(kto_score_count == 2,
            "KTO score query did not report its output count");
    require(kto_score_token_count == 2 && kto_score_kl_token_count == 2,
            "KTO score query did not report both valid-token counts");

    mft_trainer* raw_dpo = nullptr;
    mft_dpo_trainer_params dpo_params = mft_dpo_trainer_default_params();
    dpo_params.learning_rate = 1e-3f;
    require_status(
        mft_dpo_trainer_create(
            model.get(), nullptr, &dpo_params, &raw_dpo),
        MFT_STATUS_OK,
        "create cached-reference DPO trainer");
    TrainerPtr dpo(raw_dpo, &mft_trainer_free);

    require_status(
        mft_sft_train_token_batch(dpo.get(), &token_batch, &sft_result),
        MFT_STATUS_INVALID_STATE,
        "reject SFT call on DPO trainer");

    mft_dpo_step_result dpo_result{};
    require_status(
        mft_dpo_train_preference_batch(
            dpo.get(),
            tokenizer.get(),
            &preference,
            1,
            &preference_params,
            &dpo_result),
        MFT_STATUS_OK,
        "run cached-reference DPO batch");
    require(std::isfinite(dpo_result.loss), "DPO loss is not finite");
    require(dpo_result.pair_count == 1,
            "DPO result pair count mismatch");
    require(dpo_result.valid_response_token_count == 2,
            "DPO valid response-token count mismatch");

    const mft_preference_sample skipped{
        view("a"), view("b"), view("b"), true, -2.0f, -2.0f};
    require_status(
        mft_dpo_train_preference_batch(
            dpo.get(),
            tokenizer.get(),
            &skipped,
            1,
            &preference_params,
            &dpo_result),
        MFT_STATUS_DATA_ERROR,
        "reject a silently skipped preference sample");

    dpo.reset();

    mft_trainer* raw_kto = nullptr;
    mft_kto_trainer_params kto_params = mft_kto_trainer_default_params();
    kto_params.learning_rate = 1e-3f;
    require_status(
        mft_kto_trainer_create(
            model.get(), nullptr, &kto_params, &raw_kto),
        MFT_STATUS_OK,
        "create cached-reference KTO trainer");
    TrainerPtr kto(raw_kto, &mft_trainer_free);

    mft_trainer_kind kind = MFT_TRAINER_KIND_UNKNOWN;
    require_status(
        mft_trainer_get_kind(kto.get(), &kind),
        MFT_STATUS_OK,
        "query KTO trainer kind");
    require(kind == MFT_TRAINER_KIND_KTO, "trainer kind mismatch");

    // A trainer retains shared ownership of its policy implementation. Destroy
    // the public model handle before the canonical KTO step exercises it.
    model.reset();

    mft_kto_step_result kto_result{};
    require_status(
        mft_kto_train_batch(
            kto.get(),
            tokenizer.get(),
            kto_samples,
            2,
            &kto_batch_params,
            &kto_result),
        MFT_STATUS_OK,
        "run canonical cached-reference KTO batch");
    require(std::isfinite(kto_result.loss), "KTO loss is not finite");
    require(std::isfinite(kto_result.kl_estimate),
            "KTO KL estimate is not finite");
    require(kto_result.sample_count == 2 && kto_result.desirable_count == 1 &&
                kto_result.undesirable_count == 1,
            "KTO label counts mismatch");
    require(kto_result.valid_response_token_count == 2,
            "KTO main response-token count mismatch");
    require(kto_result.valid_kl_response_token_count == 2,
            "KTO KL response-token count mismatch");
}

}  // namespace

int main() {
    try {
        test_thread_local_errors();
        TemporaryDirectory temporary_directory;
        write_tiny_gpt2_assets(temporary_directory.path());
        test_training_api(temporary_directory.path());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MobileFineTuner C API runtime test failed: " << error.what()
                  << std::endl;
        return 1;
    }
}
