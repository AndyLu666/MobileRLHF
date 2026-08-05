#ifndef MFT_C_API_ONLY
#define MFT_C_API_ONLY 1
#endif
#include "mobile_finetuner/mobile_finetuner.h"

#include "finetune_ops/core/tensor.h"
#include "finetune_ops/core/tokenizer.h"
#include "finetune_ops/data/causal_lm_batch.h"
#include "finetune_ops/data/kto_batch.h"
#include "finetune_ops/data/preference_batch.h"
#include "finetune_ops/graph/auto_model.h"
#include "finetune_ops/optim/auto_trainer.h"
#include "finetune_ops/optim/dpo_trainer.h"
#include "finetune_ops/optim/kto_trainer.h"
#include "finetune_ops/optim/preference_scoring.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mft_detail {

thread_local std::string last_error;

class ApiError final : public std::runtime_error {
public:
    ApiError(mft_status status, std::string message)
        : std::runtime_error(std::move(message)), status_(status) {}

    mft_status status() const noexcept { return status_; }

private:
    mft_status status_;
};

[[noreturn]] void fail(mft_status status, const std::string& message) {
    throw ApiError(status, message);
}

struct ModelState {
    std::string model_dir;
    std::unique_ptr<ops::AutoModelForCausalLM> model;
    std::atomic<size_t> active_trainers{0};
    bool lora_initialized = false;
};

enum class ReferenceMode {
    None,
    All,
    Mixed,
};

} // namespace mft_detail

struct mft_model {
    std::shared_ptr<mft_detail::ModelState> state;
};

struct mft_tokenizer {
    std::string model_dir;
    std::unique_ptr<ops::Tokenizer> tokenizer;
};

struct mft_trainer {
    mft_trainer_kind kind = MFT_TRAINER_KIND_UNKNOWN;
    std::shared_ptr<mft_detail::ModelState> policy;
    std::shared_ptr<mft_detail::ModelState> reference;
    std::unique_ptr<ops::AutoTrainer> sft;
    std::unique_ptr<ops::DPOTrainer> dpo;
    std::unique_ptr<ops::KTOTrainer> kto;
    int32_t sft_ignore_index = -100;
    bool registered = false;

    ~mft_trainer() {
        if (!registered) return;
        if (policy) policy->active_trainers.fetch_sub(1, std::memory_order_acq_rel);
        if (reference && reference != policy)
            reference->active_trainers.fetch_sub(1, std::memory_order_acq_rel);
    }
};

namespace mft_detail {
template <typename Fn> mft_status guarded(Fn&& fn) noexcept;
template <typename Params, typename DefaultFn>
Params params_or_default(const Params*, DefaultFn&&, const char*);
std::string string_from_view(mft_string_view, const char*);
std::string require_path(const char*, const char*);
void require_finite(float, const char*);
void validate_optimizer_values(float, float, float, int32_t);
mft_model_family to_c_family(ops::ModelFamily);
ModelState& require_model(mft_model*);
const ModelState& require_model(const mft_model*);
ops::Tokenizer& require_tokenizer(mft_tokenizer*);
void validate_trainer_models(const std::shared_ptr<ModelState>&,
                             const std::shared_ptr<ModelState>&);
void register_trainer(mft_trainer&);
std::vector<std::string> lora_targets_from_params(const mft_lora_params&);
void validate_decoder_lora_targets(ops::ModelFamily,
                                   const std::vector<std::string>&);
mft_status copy_string_to_caller(const std::string&, char*, size_t, size_t*);
} // namespace mft_detail

extern "C" {

uint32_t mft_api_version(void) { return MFT_API_VERSION; }
const char* mft_version_string(void) { return "2.1.0"; }

const char* mft_status_string(mft_status status) {
    switch (status) {
        case MFT_STATUS_OK: return "ok";
        case MFT_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case MFT_STATUS_INVALID_STATE: return "invalid state";
        case MFT_STATUS_UNSUPPORTED: return "unsupported";
        case MFT_STATUS_DATA_ERROR: return "data error";
        case MFT_STATUS_IO_ERROR: return "I/O error";
        case MFT_STATUS_OUT_OF_MEMORY: return "out of memory";
        case MFT_STATUS_BUFFER_TOO_SMALL: return "buffer too small";
        case MFT_STATUS_CANCELLED: return "cancelled";
        case MFT_STATUS_INTERNAL_ERROR: return "internal error";
        default: return "unknown status";
    }
}

const char* mft_last_error(void) { return mft_detail::last_error.c_str(); }

size_t mft_get_last_error(char* buffer, size_t capacity) {
    const auto& error = mft_detail::last_error;
    if (buffer != nullptr && capacity > 0) {
        const size_t copied = std::min(error.size(), capacity - 1);
        if (copied > 0) std::memcpy(buffer, error.data(), copied);
        buffer[copied] = '\0';
    }
    return error.size();
}

mft_model_params mft_model_default_params(void) {
    mft_model_params value{};
    value.struct_size = sizeof(value);
    value.load_weights = true;
    value.auto_promote_fp16 = true;
    value.strict_key_check = true;
    value.strict_shape_check = true;
    return value;
}

mft_tokenizer_params mft_tokenizer_default_params(void) {
    mft_tokenizer_params value{}; value.struct_size = sizeof(value); return value;
}

mft_tokenizer_encode_params mft_tokenizer_encode_default_params(void) {
    mft_tokenizer_encode_params value{};
    value.struct_size = sizeof(value); value.truncation = true; return value;
}

mft_tokenizer_decode_params mft_tokenizer_decode_default_params(void) {
    mft_tokenizer_decode_params value{}; value.struct_size = sizeof(value); return value;
}

mft_lora_params mft_lora_default_params(void) {
    mft_lora_params value{};
    value.struct_size = sizeof(value); value.rank = 8; value.alpha = 16.0f;
    value.dropout = 0.05f; value.seed = 42; return value;
}

mft_sft_trainer_params mft_sft_trainer_default_params(void) {
    mft_sft_trainer_params value{};
    value.struct_size = sizeof(value); value.learning_rate = 2e-4f;
    value.max_grad_norm = 1.0f; value.ignore_index = -100;
    value.use_streaming_loss = true; value.gradient_accumulation_steps = 1;
    return value;
}

mft_dpo_trainer_params mft_dpo_trainer_default_params(void) {
    mft_dpo_trainer_params value{};
    value.struct_size = sizeof(value); value.learning_rate = 2e-4f;
    value.max_grad_norm = 1.0f; value.beta = 0.1f;
    value.use_streaming_loss = true; value.gradient_accumulation_steps = 1;
    return value;
}

mft_kto_trainer_params mft_kto_trainer_default_params(void) {
    mft_kto_trainer_params value{};
    value.struct_size = sizeof(value); value.learning_rate = 2e-4f;
    value.max_grad_norm = 1.0f; value.beta = 0.1f;
    value.desirable_weight = 1.0f; value.undesirable_weight = 1.0f;
    value.use_streaming_loss = true; value.gradient_accumulation_steps = 1;
    return value;
}

mft_causal_batch_params mft_causal_batch_default_params(void) {
    mft_causal_batch_params value{};
    value.struct_size = sizeof(value); value.sequence_length = 64;
    value.ignore_index = -100; value.truncate = true;
    value.mask_padding_labels = true; return value;
}

mft_preference_batch_params mft_preference_batch_default_params(void) {
    mft_preference_batch_params value{};
    value.struct_size = sizeof(value); value.sequence_length = 128;
    value.truncate = true; return value;
}

mft_kto_batch_params mft_kto_batch_default_params(void) {
    mft_kto_batch_params value{};
    value.struct_size = sizeof(value); value.sequence_length = 128;
    value.truncate = true; return value;
}

mft_generation_params mft_generation_default_params(void) {
    mft_generation_params value{};
    value.struct_size = sizeof(value); value.max_new_tokens = 32;
    value.top_p = 1.0f; value.seed = 42;
    value.apply_qwen_chat_template = true; value.clean_qwen_output = true;
    return value;
}

mft_status mft_model_load_from_dir(const char* model_dir,
                                   const mft_model_params* params,
                                   mft_model** out_model) {
    if (out_model) *out_model = nullptr;
    return mft_detail::guarded([&] {
        if (!out_model) mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "out_model is null");
        const auto values = mft_detail::params_or_default(
            params, mft_model_default_params, "mft_model_params");
        ops::AutoModelLoadOptions options;
        options.load_weights = values.load_weights;
        options.verbose = values.verbose;
        options.safetensors_options.auto_promote_fp16 = values.auto_promote_fp16;
        options.safetensors_options.strict_key_check = values.strict_key_check;
        options.safetensors_options.strict_shape_check = values.strict_shape_check;
        options.safetensors_options.verbose = values.verbose;
        auto state = std::make_shared<mft_detail::ModelState>();
        state->model_dir = mft_detail::require_path(model_dir, "model_dir");
        state->model = ops::AutoModelForCausalLM::from_pretrained(state->model_dir, options);
        auto handle = std::make_unique<mft_model>();
        handle->state = std::move(state);
        *out_model = handle.release();
    });
}

void mft_model_free(mft_model* model) {
    if (!model) return;
    try { delete model; }
    catch (...) { mft_detail::last_error = "error while freeing model"; }
}

mft_status mft_model_get_family(const mft_model* model, mft_model_family* out_family) {
    if (out_family) *out_family = MFT_MODEL_FAMILY_UNKNOWN;
    return mft_detail::guarded([&] {
        if (!out_family) mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "out_family is null");
        *out_family = mft_detail::to_c_family(mft_detail::require_model(model).model->family());
    });
}

mft_status mft_model_init_lora(mft_model* model, const mft_lora_params* params) {
    return mft_detail::guarded([&] {
        auto& state = mft_detail::require_model(model);
        if (state.active_trainers.load() != 0)
            mft_detail::fail(MFT_STATUS_INVALID_STATE, "cannot init LoRA with an active trainer");
        if (state.lora_initialized)
            mft_detail::fail(MFT_STATUS_INVALID_STATE, "LoRA is already initialized");
        const auto values = mft_detail::params_or_default(
            params, mft_lora_default_params, "mft_lora_params");
        mft_detail::require_finite(values.alpha, "alpha");
        mft_detail::require_finite(values.dropout, "dropout");
        if (values.rank <= 0 || !(values.alpha > 0.0f) ||
            values.dropout < 0.0f || values.dropout >= 1.0f)
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "invalid LoRA parameters");
        auto targets = mft_detail::lora_targets_from_params(values);
        mft_detail::validate_decoder_lora_targets(state.model->family(), targets);
        ops::AutoLoraConfig config;
        config.rank = values.rank; config.alpha = values.alpha;
        config.dropout = values.dropout; config.seed = values.seed;
        config.target_modules = std::move(targets);
        state.model->init_lora(config);
        if (state.model->trainable_parameters().empty())
            mft_detail::fail(MFT_STATUS_INTERNAL_ERROR, "LoRA created no trainable tensors");
        state.lora_initialized = true;
    });
}

mft_status mft_model_get_trainable_tensor_count(const mft_model* model, size_t* out_count) {
    if (out_count) *out_count = 0;
    return mft_detail::guarded([&] {
        if (!out_count) mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "out_count is null");
        *out_count = mft_detail::require_model(model).model->trainable_parameters().size();
    });
}

mft_status mft_tokenizer_load_from_dir(const char* model_dir,
                                       const mft_tokenizer_params* params,
                                       mft_tokenizer** out_tokenizer) {
    if (out_tokenizer) *out_tokenizer = nullptr;
    return mft_detail::guarded([&] {
        if (!out_tokenizer)
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "out_tokenizer is null");
        const auto values = mft_detail::params_or_default(
            params, mft_tokenizer_default_params, "mft_tokenizer_params");
        ops::TokenizerLoadOptions options;
        options.model_type = mft_detail::string_from_view(
            values.model_type_override, "model_type_override");
        auto handle = std::make_unique<mft_tokenizer>();
        handle->model_dir = mft_detail::require_path(model_dir, "model_dir");
        handle->tokenizer = ops::TokenizerFactory::from_pretrained(handle->model_dir, options);
        *out_tokenizer = handle.release();
    });
}

void mft_tokenizer_free(mft_tokenizer* tokenizer) {
    if (!tokenizer) return;
    try { delete tokenizer; }
    catch (...) { mft_detail::last_error = "error while freeing tokenizer"; }
}

mft_status mft_tokenizer_encode(mft_tokenizer* tokenizer,
                                mft_string_view text,
                                const mft_tokenizer_encode_params* params,
                                int32_t* tokens,
                                size_t capacity,
                                size_t* out_count) {
    if (out_count) *out_count = 0;
    return mft_detail::guarded([&] {
        if (!out_count) mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "out_count is null");
        const auto values = mft_detail::params_or_default(
            params, mft_tokenizer_encode_default_params, "encode params");
        if (values.max_length < 0)
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "max_length must be >= 0");
        ops::TokenizerEncodeOptions options;
        options.add_special_tokens = values.add_special_tokens;
        options.max_length = values.max_length;
        options.truncation = values.truncation;
        const auto encoded = mft_detail::require_tokenizer(tokenizer).encode_with_options(
            mft_detail::string_from_view(text, "text"), options);
        *out_count = encoded.size();
        if (!tokens) {
            if (capacity != 0) mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "null buffer with capacity");
            if (!encoded.empty()) mft_detail::fail(MFT_STATUS_BUFFER_TOO_SMALL, "token buffer too small");
            return;
        }
        if (capacity < encoded.size())
            mft_detail::fail(MFT_STATUS_BUFFER_TOO_SMALL, "token buffer too small");
        for (size_t i = 0; i < encoded.size(); ++i) tokens[i] = encoded[i];
    });
}

mft_status mft_tokenizer_decode(mft_tokenizer* tokenizer,
                                const int32_t* tokens,
                                size_t token_count,
                                const mft_tokenizer_decode_params* params,
                                char* text,
                                size_t capacity,
                                size_t* out_size) {
    if (out_size) *out_size = 0;
    return mft_detail::guarded([&] {
        if (token_count && !tokens)
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "tokens is null");
        const auto values = mft_detail::params_or_default(
            params, mft_tokenizer_decode_default_params, "decode params");
        std::vector<int> ids;
        if (token_count > 0) {
            ids.assign(tokens, tokens + token_count);
        }
        ops::TokenizerDecodeOptions options;
        options.skip_special_tokens = values.skip_special_tokens;
        const auto decoded = mft_detail::require_tokenizer(tokenizer).decode_with_options(ids, options);
        const auto status = mft_detail::copy_string_to_caller(decoded, text, capacity, out_size);
        if (status != MFT_STATUS_OK) mft_detail::fail(status, "text buffer too small");
    });
}

static void validate_trainer_params(float lr, float wd, float norm, int32_t steps) {
    mft_detail::validate_optimizer_values(lr, wd, norm, steps);
}

mft_status mft_sft_trainer_create(mft_model* policy,
                                  const mft_sft_trainer_params* params,
                                  mft_trainer** out_trainer) {
    if (out_trainer) *out_trainer = nullptr;
    return mft_detail::guarded([&] {
        if (!out_trainer) mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "out_trainer is null");
        auto& state = mft_detail::require_model(policy);
        mft_detail::validate_trainer_models(policy->state, nullptr);
        const auto values = mft_detail::params_or_default(
            params, mft_sft_trainer_default_params, "SFT params");
        validate_trainer_params(values.learning_rate, values.weight_decay,
                                values.max_grad_norm, values.gradient_accumulation_steps);
        ops::AutoTrainerConfig config;
        config.learning_rate = values.learning_rate; config.weight_decay = values.weight_decay;
        config.max_grad_norm = values.max_grad_norm; config.ignore_index = values.ignore_index;
        config.use_streaming_lm_loss = values.use_streaming_loss;
        config.gradient_accumulation_steps = values.gradient_accumulation_steps;
        auto trainer = std::make_unique<mft_trainer>();
        trainer->kind = MFT_TRAINER_KIND_SFT; trainer->policy = policy->state;
        trainer->sft_ignore_index = values.ignore_index;
        trainer->sft = std::make_unique<ops::AutoTrainer>(*state.model, config);
        mft_detail::register_trainer(*trainer); *out_trainer = trainer.release();
    });
}

mft_status mft_dpo_trainer_create(mft_model* policy,
                                  mft_model* reference_handle,
                                  const mft_dpo_trainer_params* params,
                                  mft_trainer** out_trainer) {
    if (out_trainer) *out_trainer = nullptr;
    return mft_detail::guarded([&] {
        if (!out_trainer) mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "out_trainer is null");
        auto& state = mft_detail::require_model(policy);
        std::shared_ptr<mft_detail::ModelState> reference;
        if (reference_handle) { mft_detail::require_model(reference_handle); reference = reference_handle->state; }
        mft_detail::validate_trainer_models(policy->state, reference);
        const auto values = mft_detail::params_or_default(
            params, mft_dpo_trainer_default_params, "DPO params");
        validate_trainer_params(values.learning_rate, values.weight_decay,
                                values.max_grad_norm, values.gradient_accumulation_steps);
        mft_detail::require_finite(values.beta, "beta");
        if (!(values.beta > 0)) mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "beta must be > 0");
        ops::DPOTrainerConfig config;
        config.learning_rate = values.learning_rate; config.weight_decay = values.weight_decay;
        config.max_grad_norm = values.max_grad_norm; config.beta = values.beta;
        config.use_streaming_dpo_loss = values.use_streaming_loss;
        config.gradient_accumulation_steps = values.gradient_accumulation_steps;
        auto trainer = std::make_unique<mft_trainer>();
        trainer->kind = MFT_TRAINER_KIND_DPO; trainer->policy = policy->state;
        trainer->reference = reference;
        if (reference) trainer->dpo = std::make_unique<ops::DPOTrainer>(*state.model, *reference->model, config);
        else trainer->dpo = std::make_unique<ops::DPOTrainer>(*state.model, config);
        mft_detail::register_trainer(*trainer); *out_trainer = trainer.release();
    });
}

mft_status mft_kto_trainer_create(mft_model* policy,
                                  mft_model* reference_handle,
                                  const mft_kto_trainer_params* params,
                                  mft_trainer** out_trainer) {
    if (out_trainer) *out_trainer = nullptr;
    return mft_detail::guarded([&] {
        if (!out_trainer) mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "out_trainer is null");
        auto& state = mft_detail::require_model(policy);
        std::shared_ptr<mft_detail::ModelState> reference;
        if (reference_handle) { mft_detail::require_model(reference_handle); reference = reference_handle->state; }
        mft_detail::validate_trainer_models(policy->state, reference);
        const auto values = mft_detail::params_or_default(
            params, mft_kto_trainer_default_params, "KTO params");
        validate_trainer_params(values.learning_rate, values.weight_decay,
                                values.max_grad_norm, values.gradient_accumulation_steps);
        mft_detail::require_finite(values.beta, "beta");
        mft_detail::require_finite(values.desirable_weight, "desirable_weight");
        mft_detail::require_finite(values.undesirable_weight, "undesirable_weight");
        if (!(values.beta > 0) || !(values.desirable_weight > 0) || !(values.undesirable_weight > 0))
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "KTO beta and weights must be > 0");
        ops::KTOTrainerConfig config;
        config.learning_rate = values.learning_rate; config.weight_decay = values.weight_decay;
        config.max_grad_norm = values.max_grad_norm; config.beta = values.beta;
        config.desirable_weight = values.desirable_weight;
        config.undesirable_weight = values.undesirable_weight;
        config.use_streaming_kto_loss = values.use_streaming_loss;
        config.gradient_accumulation_steps = values.gradient_accumulation_steps;
        auto trainer = std::make_unique<mft_trainer>();
        trainer->kind = MFT_TRAINER_KIND_KTO; trainer->policy = policy->state;
        trainer->reference = reference;
        if (reference) trainer->kto = std::make_unique<ops::KTOTrainer>(*state.model, *reference->model, config);
        else trainer->kto = std::make_unique<ops::KTOTrainer>(*state.model, config);
        mft_detail::register_trainer(*trainer); *out_trainer = trainer.release();
    });
}

void mft_trainer_free(mft_trainer* trainer) {
    if (!trainer) return;
    try { delete trainer; }
    catch (...) { mft_detail::last_error = "error while freeing trainer"; }
}

mft_status mft_trainer_get_kind(const mft_trainer* trainer, mft_trainer_kind* out_kind) {
    if (out_kind) *out_kind = MFT_TRAINER_KIND_UNKNOWN;
    return mft_detail::guarded([&] {
        if (!trainer || !out_kind)
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "trainer/out_kind is null");
        *out_kind = trainer->kind;
    });
}

} // extern "C"

namespace mft_detail {

template <typename Fn>
mft_status guarded(Fn&& fn) noexcept {
    last_error.clear();
    try {
        fn();
        return MFT_STATUS_OK;
    } catch (const ApiError& error) {
        last_error = error.what();
        return error.status();
    } catch (const std::invalid_argument& error) {
        last_error = error.what();
        return MFT_STATUS_INVALID_ARGUMENT;
    } catch (const std::filesystem::filesystem_error& error) {
        last_error = error.what();
        return MFT_STATUS_IO_ERROR;
    } catch (const std::bad_alloc& error) {
        last_error = error.what();
        return MFT_STATUS_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        last_error = error.what();
        const std::string message = error.what();
        if (message.find("not supported") != std::string::npos ||
            message.find("unsupported") != std::string::npos ||
            message.find("not been passed") != std::string::npos) {
            return MFT_STATUS_UNSUPPORTED;
        }
        if (message.find("Cannot open") != std::string::npos ||
            message.find("cannot open") != std::string::npos ||
            message.find("failed to open") != std::string::npos ||
            message.find("does not exist") != std::string::npos) {
            return MFT_STATUS_IO_ERROR;
        }
        return MFT_STATUS_INTERNAL_ERROR;
    } catch (...) {
        last_error = "Unknown native MobileFineTuner error";
        return MFT_STATUS_INTERNAL_ERROR;
    }
}

template <typename Params, typename DefaultFn>
Params params_or_default(const Params* params,
                         DefaultFn&& default_fn,
                         const char* name) {
    if (params == nullptr) {
        return default_fn();
    }
    if (params->struct_size < sizeof(params->struct_size)) {
        fail(MFT_STATUS_INVALID_ARGUMENT,
             std::string(name) + ".struct_size is too small");
    }
    Params values = default_fn();
    const size_t copied = std::min(
        static_cast<size_t>(params->struct_size), sizeof(Params));
    std::memcpy(&values, params, copied);
    values.struct_size = sizeof(Params);
    return values;
}

std::string string_from_view(mft_string_view view, const char* name) {
    if (view.size > 0 && view.data == nullptr) {
        fail(MFT_STATUS_INVALID_ARGUMENT, std::string(name) + ".data must not be null");
    }
    return view.size == 0 ? std::string() : std::string(view.data, view.size);
}

std::string require_path(const char* path, const char* name) {
    if (path == nullptr || path[0] == '\0') {
        fail(MFT_STATUS_INVALID_ARGUMENT, std::string(name) + " must not be empty");
    }
    return std::string(path);
}

void require_finite(float value, const char* name) {
    if (!std::isfinite(value)) {
        fail(MFT_STATUS_INVALID_ARGUMENT, std::string(name) + " must be finite");
    }
}

void validate_optimizer_values(float learning_rate,
                               float weight_decay,
                               float max_grad_norm,
                               int32_t accumulation_steps) {
    require_finite(learning_rate, "learning_rate");
    require_finite(weight_decay, "weight_decay");
    require_finite(max_grad_norm, "max_grad_norm");
    if (!(learning_rate > 0.0f)) {
        fail(MFT_STATUS_INVALID_ARGUMENT, "learning_rate must be > 0");
    }
    if (weight_decay < 0.0f) {
        fail(MFT_STATUS_INVALID_ARGUMENT, "weight_decay must be >= 0");
    }
    if (!(max_grad_norm > 0.0f)) {
        fail(MFT_STATUS_INVALID_ARGUMENT, "max_grad_norm must be > 0");
    }
    if (accumulation_steps <= 0) {
        fail(MFT_STATUS_INVALID_ARGUMENT,
             "gradient_accumulation_steps must be > 0");
    }
}

size_t checked_element_count(size_t batch_size, size_t sequence_length) {
    if (batch_size == 0 || sequence_length <= 1) {
        fail(MFT_STATUS_INVALID_ARGUMENT,
             "batch_size must be > 0 and sequence_length must be > 1");
    }
    if (batch_size > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        sequence_length > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        batch_size > std::numeric_limits<size_t>::max() / sequence_length) {
        fail(MFT_STATUS_INVALID_ARGUMENT, "batch dimensions overflow");
    }
    return batch_size * sequence_length;
}

mft_status copy_string_to_caller(const std::string& value,
                                 char* output,
                                 size_t capacity,
                                 size_t* out_size) {
    if (out_size == nullptr) {
        fail(MFT_STATUS_INVALID_ARGUMENT, "out_text_size must not be null");
    }
    *out_size = value.size();
    if (output == nullptr) {
        if (capacity != 0) {
            fail(MFT_STATUS_INVALID_ARGUMENT,
                 "a null output buffer requires zero capacity");
        }
        return value.empty() ? MFT_STATUS_OK : MFT_STATUS_BUFFER_TOO_SMALL;
    }
    if (capacity <= value.size()) {
        return MFT_STATUS_BUFFER_TOO_SMALL;
    }
    if (!value.empty()) {
        std::memcpy(output, value.data(), value.size());
    }
    output[value.size()] = '\0';
    return MFT_STATUS_OK;
}

mft_model_family to_c_family(ops::ModelFamily family) {
    switch (family) {
        case ops::ModelFamily::GPT2: return MFT_MODEL_FAMILY_GPT2;
        case ops::ModelFamily::Gemma: return MFT_MODEL_FAMILY_GEMMA;
        case ops::ModelFamily::Llama: return MFT_MODEL_FAMILY_LLAMA;
        case ops::ModelFamily::Mistral: return MFT_MODEL_FAMILY_MISTRAL;
        case ops::ModelFamily::Qwen: return MFT_MODEL_FAMILY_QWEN;
        case ops::ModelFamily::Unknown:
        default: return MFT_MODEL_FAMILY_UNKNOWN;
    }
}

ModelState& require_model(mft_model* handle) {
    if (handle == nullptr || !handle->state || !handle->state->model) {
        fail(MFT_STATUS_INVALID_ARGUMENT, "model handle must not be null");
    }
    return *handle->state;
}

const ModelState& require_model(const mft_model* handle) {
    if (handle == nullptr || !handle->state || !handle->state->model) {
        fail(MFT_STATUS_INVALID_ARGUMENT, "model handle must not be null");
    }
    return *handle->state;
}

ops::Tokenizer& require_tokenizer(mft_tokenizer* handle) {
    if (handle == nullptr || !handle->tokenizer) {
        fail(MFT_STATUS_INVALID_ARGUMENT, "tokenizer handle must not be null");
    }
    return *handle->tokenizer;
}

mft_trainer& require_trainer(mft_trainer* trainer, mft_trainer_kind expected) {
    if (trainer == nullptr) {
        fail(MFT_STATUS_INVALID_ARGUMENT, "trainer handle must not be null");
    }
    if (trainer->kind != expected) {
        fail(MFT_STATUS_INVALID_STATE,
             "trainer objective does not match this operation");
    }
    return *trainer;
}

void validate_trainer_models(const std::shared_ptr<ModelState>& policy,
                             const std::shared_ptr<ModelState>& reference) {
    if (!policy || !policy->model) {
        fail(MFT_STATUS_INVALID_ARGUMENT, "policy model must not be null");
    }
    if (!policy->lora_initialized || policy->model->trainable_parameters().empty()) {
        fail(MFT_STATUS_INVALID_STATE,
             "policy model requires initialized LoRA before trainer creation");
    }
    if (policy->active_trainers.load(std::memory_order_acquire) != 0) {
        fail(MFT_STATUS_INVALID_STATE, "policy model already has an active trainer");
    }
    if (reference) {
        if (reference == policy) {
            fail(MFT_STATUS_INVALID_ARGUMENT,
                 "reference model must be distinct from the policy model");
        }
        if (!reference->model) {
            fail(MFT_STATUS_INVALID_ARGUMENT, "reference model is not initialized");
        }
        if (reference->active_trainers.load(std::memory_order_acquire) != 0) {
            fail(MFT_STATUS_INVALID_STATE,
                 "reference model already has an active trainer");
        }
    }
}

void register_trainer(mft_trainer& trainer) {
    trainer.policy->active_trainers.fetch_add(1, std::memory_order_acq_rel);
    if (trainer.reference && trainer.reference != trainer.policy) {
        trainer.reference->active_trainers.fetch_add(1, std::memory_order_acq_rel);
    }
    trainer.registered = true;
}

std::vector<std::string> lora_targets_from_params(const mft_lora_params& params) {
    if (params.target_module_count > 0 && params.target_modules == nullptr) {
        fail(MFT_STATUS_INVALID_ARGUMENT,
             "target_modules is null with a non-zero target_module_count");
    }
    std::vector<std::string> targets;
    std::unordered_set<std::string> unique;
    targets.reserve(params.target_module_count);
    for (size_t i = 0; i < params.target_module_count; ++i) {
        std::string target = string_from_view(params.target_modules[i], "target module");
        if (target.empty()) {
            fail(MFT_STATUS_INVALID_ARGUMENT, "LoRA target names must not be empty");
        }
        if (!unique.insert(target).second) {
            fail(MFT_STATUS_INVALID_ARGUMENT, "duplicate LoRA target: " + target);
        }
        targets.push_back(std::move(target));
    }
    return targets;
}

bool target_set_equals(const std::vector<std::string>& values,
                       std::initializer_list<const char*> expected) {
    if (values.size() != expected.size()) {
        return false;
    }
    const std::unordered_set<std::string> actual(values.begin(), values.end());
    for (const char* item : expected) {
        if (actual.count(item) == 0) {
            return false;
        }
    }
    return true;
}

void validate_decoder_lora_targets(ops::ModelFamily family,
                                   const std::vector<std::string>& targets) {
    if (family != ops::ModelFamily::Llama && family != ops::ModelFamily::Qwen) {
        return;
    }
    if (targets.empty() ||
        target_set_equals(targets, {"q_proj", "v_proj"}) ||
        target_set_equals(targets, {"q_proj", "k_proj", "v_proj", "o_proj"})) {
        return;
    }
    fail(MFT_STATUS_UNSUPPORTED,
         "Llama/Qwen LoRA targets must be default, q+v, or q+k+v+o");
}

void validate_sample_indices(const std::vector<size_t>& indices,
                             size_t expected_count,
                             const char* context) {
    if (indices.size() != expected_count) {
        fail(MFT_STATUS_DATA_ERROR,
             std::string(context) + " skipped one or more samples");
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] != i) {
            fail(MFT_STATUS_DATA_ERROR,
                 std::string(context) + " changed sample ordering");
        }
    }
}

ReferenceMode reference_mode(bool saw_reference, bool saw_missing) {
    if (saw_reference && saw_missing) {
        return ReferenceMode::Mixed;
    }
    return saw_reference ? ReferenceMode::All : ReferenceMode::None;
}

std::vector<std::string> strings_from_views(const mft_string_view* values,
                                            size_t count,
                                            const char* name) {
    if (values == nullptr || count == 0) {
        fail(MFT_STATUS_INVALID_ARGUMENT, std::string(name) + " must not be empty");
    }
    std::vector<std::string> output;
    output.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        output.push_back(string_from_view(values[i], name));
    }
    return output;
}

struct PreferenceInputs {
    std::vector<ops::PreferenceSample> samples;
    ReferenceMode references = ReferenceMode::None;
};

PreferenceInputs preference_inputs_from_c(const mft_preference_sample* samples,
                                          size_t sample_count) {
    if (samples == nullptr || sample_count == 0) {
        fail(MFT_STATUS_INVALID_ARGUMENT,
             "preference samples must contain at least one row");
    }
    PreferenceInputs result;
    result.samples.reserve(sample_count);
    bool saw_reference = false;
    bool saw_missing = false;
    for (size_t i = 0; i < sample_count; ++i) {
        ops::PreferenceSample item;
        item.prompt = string_from_view(samples[i].prompt, "preference prompt");
        item.chosen = string_from_view(samples[i].chosen, "chosen response");
        item.rejected = string_from_view(samples[i].rejected, "rejected response");
        item.has_reference_logps = samples[i].has_reference_logps;
        if (item.has_reference_logps) {
            require_finite(samples[i].ref_chosen_logp, "ref_chosen_logp");
            require_finite(samples[i].ref_rejected_logp, "ref_rejected_logp");
            item.ref_chosen_logp = samples[i].ref_chosen_logp;
            item.ref_rejected_logp = samples[i].ref_rejected_logp;
            saw_reference = true;
        } else {
            saw_missing = true;
        }
        result.samples.push_back(std::move(item));
    }
    result.references = reference_mode(saw_reference, saw_missing);
    return result;
}

ops::PreferenceBatchConfig preference_config_from_c(
    const mft_preference_batch_params& params) {
    if (params.sequence_length <= 1) {
        fail(MFT_STATUS_INVALID_ARGUMENT, "sequence_length must be > 1");
    }
    ops::PreferenceBatchConfig config;
    config.sequence_length = params.sequence_length;
    config.truncate = params.truncate;
    config.add_special_tokens = params.add_special_tokens;
    config.append_eos_to_response = params.append_eos_to_response;
    return config;
}

ops::PreferenceBatch make_strict_preference_batch(
    ops::Tokenizer& tokenizer,
    const PreferenceInputs& inputs,
    const mft_preference_batch_params& params) {
    ops::PreferenceBatch batch;
    try {
        batch = ops::make_preference_batch(
            tokenizer, inputs.samples, preference_config_from_c(params));
    } catch (const std::invalid_argument& error) {
        fail(MFT_STATUS_DATA_ERROR,
             std::string("invalid preference batch: ") + error.what());
    }
    if (batch.batch_size != static_cast<int>(inputs.samples.size())) {
        fail(MFT_STATUS_DATA_ERROR,
             "preference batch builder skipped one or more samples");
    }
    validate_sample_indices(batch.sample_indices,
                            inputs.samples.size(),
                            "preference batch builder");
    return batch;
}

struct KTOInputs {
    std::vector<ops::KTOSample> samples;
    ReferenceMode references = ReferenceMode::None;
};

KTOInputs kto_inputs_from_c(const mft_kto_sample* samples, size_t sample_count) {
    if (samples == nullptr || sample_count < 2) {
        fail(MFT_STATUS_INVALID_ARGUMENT,
             "canonical KTO requires at least two ordered samples");
    }
    KTOInputs result;
    result.samples.reserve(sample_count);
    bool saw_reference = false;
    bool saw_missing = false;
    for (size_t i = 0; i < sample_count; ++i) {
        if (samples[i].label != MFT_KTO_UNDESIRABLE &&
            samples[i].label != MFT_KTO_DESIRABLE) {
            fail(MFT_STATUS_INVALID_ARGUMENT, "invalid canonical KTO label");
        }
        ops::KTOSample item;
        item.prompt = string_from_view(samples[i].prompt, "KTO prompt");
        item.completion = string_from_view(samples[i].completion, "KTO completion");
        item.desirable = samples[i].label == MFT_KTO_DESIRABLE;
        item.has_reference_logps = samples[i].has_reference_logps;
        if (item.has_reference_logps) {
            require_finite(samples[i].ref_logp, "KTO ref_logp");
            require_finite(samples[i].ref_kl_logp, "KTO ref_kl_logp");
            item.ref_logp = samples[i].ref_logp;
            item.ref_kl_logp = samples[i].ref_kl_logp;
            saw_reference = true;
        } else {
            saw_missing = true;
        }
        result.samples.push_back(std::move(item));
    }
    result.references = reference_mode(saw_reference, saw_missing);
    return result;
}

ops::KTOBatchConfig kto_config_from_c(const mft_kto_batch_params& params) {
    if (params.sequence_length <= 1) {
        fail(MFT_STATUS_INVALID_ARGUMENT, "sequence_length must be > 1");
    }
    ops::KTOBatchConfig config;
    config.sequence_length = params.sequence_length;
    config.truncate = params.truncate;
    config.add_special_tokens = params.add_special_tokens;
    config.append_eos_to_completion = params.append_eos_to_completion;
    return config;
}

ops::KTOBatch make_strict_kto_batch(ops::Tokenizer& tokenizer,
                                    const KTOInputs& inputs,
                                    const mft_kto_batch_params& params) {
    ops::KTOBatch batch;
    try {
        batch = ops::make_kto_batch(tokenizer, inputs.samples, kto_config_from_c(params));
    } catch (const std::invalid_argument& error) {
        fail(MFT_STATUS_DATA_ERROR, std::string("invalid KTO batch: ") + error.what());
    }
    if (batch.batch_size != static_cast<int>(inputs.samples.size())) {
        fail(MFT_STATUS_DATA_ERROR, "KTO batch builder skipped one or more samples");
    }
    validate_sample_indices(batch.sample_indices, inputs.samples.size(), "KTO batch builder");
    if (batch.kl_completion_indices.size() != inputs.samples.size()) {
        fail(MFT_STATUS_DATA_ERROR, "KTO batch builder produced an incomplete KL rotation");
    }
    return batch;
}

void set_sft_result(const ops::AutoTrainStepResult& source,
                    mft_sft_step_result& target) {
    target.loss = source.loss;
    target.accumulated_loss = source.accumulated_loss;
    target.trainable_tensor_count = source.trainable_tensor_count;
    target.valid_label_count = source.valid_label_count;
    target.accumulation_step = source.accumulation_step;
    target.gradient_accumulation_steps = source.gradient_accumulation_steps;
    target.optimizer_step = source.optimizer_step;
}

void set_dpo_result(const ops::DPOTrainStepResult& source,
                    mft_dpo_step_result& target) {
    target.loss = source.loss;
    target.accumulated_loss = source.accumulated_loss;
    target.chosen_reward = source.chosen_reward;
    target.rejected_reward = source.rejected_reward;
    target.reward_margin = source.reward_margin;
    target.reward_accuracy = source.reward_accuracy;
    target.trainable_tensor_count = source.trainable_tensor_count;
    target.pair_count = source.pair_count;
    target.valid_response_token_count = source.valid_response_token_count;
    target.accumulation_step = source.accumulation_step;
    target.gradient_accumulation_steps = source.gradient_accumulation_steps;
    target.optimizer_step = source.optimizer_step;
}

void set_kto_result(const ops::KTOTrainStepResult& source,
                    mft_kto_step_result& target) {
    target.loss = source.loss;
    target.accumulated_loss = source.accumulated_loss;
    target.kl_estimate = source.kl_estimate;
    target.desirable_reward = source.desirable_reward;
    target.undesirable_reward = source.undesirable_reward;
    target.reward_margin = source.reward_margin;
    target.reward_accuracy = source.reward_accuracy;
    target.trainable_tensor_count = source.trainable_tensor_count;
    target.sample_count = source.sample_count;
    target.desirable_count = source.desirable_count;
    target.undesirable_count = source.undesirable_count;
    target.valid_response_token_count = source.valid_response_token_count;
    target.valid_kl_response_token_count = source.valid_kl_response_token_count;
    target.accumulation_step = source.accumulation_step;
    target.gradient_accumulation_steps = source.gradient_accumulation_steps;
    target.optimizer_step = source.optimizer_step;
}

void set_kto_compat_result(const ops::KTOTrainStepResult& source,
                           mft_kto_preference_step_result& target) {
    target.loss = source.loss;
    target.accumulated_loss = source.accumulated_loss;
    target.desirable_reward = source.desirable_reward;
    target.undesirable_reward = source.undesirable_reward;
    target.reward_margin = source.reward_margin;
    target.reward_accuracy = source.reward_accuracy;
    target.trainable_tensor_count = source.trainable_tensor_count;
    target.pair_count = source.pair_count;
    target.sample_count = source.sample_count;
    target.valid_response_token_count = source.valid_response_token_count;
    target.accumulation_step = source.accumulation_step;
    target.gradient_accumulation_steps = source.gradient_accumulation_steps;
    target.optimizer_step = source.optimizer_step;
}

ops::TensorPtr make_input_ids(const std::vector<int>& ids) {
    std::vector<int32_t> values(ids.begin(), ids.end());
    return std::make_shared<ops::Tensor>(
        std::vector<int64_t>{1, static_cast<int64_t>(values.size())},
        values.data(), ops::kInt32, ops::kCPU);
}

ops::TensorPtr make_attention_mask(size_t count) {
    std::vector<float> values(count, 1.0f);
    return std::make_shared<ops::Tensor>(
        std::vector<int64_t>{1, static_cast<int64_t>(count)},
        values.data(), ops::kFloat32, ops::kCPU);
}

void validate_logits(const ops::TensorPtr& logits) {
    if (!logits || logits->dtype() != ops::kFloat32 || logits->ndim() != 3 ||
        logits->size(0) != 1 || logits->size(1) <= 0 || logits->size(2) <= 0) {
        fail(MFT_STATUS_INTERNAL_ERROR,
             "generation expected float32 logits shaped [1, sequence, vocabulary]");
    }
}

int argmax_last_token(const ops::TensorPtr& logits) {
    validate_logits(logits);
    const int64_t vocabulary = logits->size(2);
    const int64_t offset = (logits->size(1) - 1) * vocabulary;
    const float* data = logits->data<float>();
    int best = 0;
    for (int64_t id = 1; id < vocabulary; ++id) {
        if (data[offset + id] > data[offset + best]) {
            best = static_cast<int>(id);
        }
    }
    return best;
}

int sample_last_token(const ops::TensorPtr& logits,
                      float temperature,
                      float top_p,
                      std::mt19937& random) {
    if (temperature == 0.0f) {
        return argmax_last_token(logits);
    }
    validate_logits(logits);
    const int64_t vocabulary = logits->size(2);
    const int64_t offset = (logits->size(1) - 1) * vocabulary;
    const float* data = logits->data<float>();
    float maximum = data[offset];
    for (int64_t id = 1; id < vocabulary; ++id) {
        maximum = std::max(maximum, data[offset + id]);
    }
    struct Candidate { int id; double probability; };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(vocabulary));
    double denominator = 0.0;
    for (int64_t id = 0; id < vocabulary; ++id) {
        const double probability = std::exp(
            static_cast<double>(data[offset + id] - maximum) / temperature);
        candidates.push_back({static_cast<int>(id), probability});
        denominator += probability;
    }
    if (!(denominator > 0.0) || !std::isfinite(denominator)) {
        fail(MFT_STATUS_INTERNAL_ERROR, "invalid generation probabilities");
    }
    for (auto& candidate : candidates) {
        candidate.probability /= denominator;
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& left, const Candidate& right) {
                  return left.probability > right.probability;
              });
    size_t kept = 0;
    double cumulative = 0.0;
    do {
        cumulative += candidates[kept].probability;
        ++kept;
    } while (kept < candidates.size() && cumulative < top_p);
    std::vector<double> weights;
    weights.reserve(kept);
    for (size_t i = 0; i < kept; ++i) {
        weights.push_back(candidates[i].probability);
    }
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return candidates[distribution(random)].id;
}

std::string make_qwen_chat_prompt(const std::string& prompt) {
    return "<|im_start|>system\n"
           "You are a concise assistant. Answer directly in one short sentence."
           "<|im_end|>\n"
           "<|im_start|>user\n" + prompt +
           "<|im_end|>\n"
           "<|im_start|>assistant\n"
           "<think>\n\n</think>\n\n";
}

std::string clean_qwen_output(std::string text) {
    const std::string begin = "<think>";
    const std::string end = "</think>";
    while (true) {
        const size_t start = text.find(begin);
        if (start == std::string::npos) break;
        const size_t finish = text.find(end, start + begin.size());
        if (finish == std::string::npos) text.erase(start, begin.size());
        else text.erase(start, finish + end.size() - start);
    }
    const std::string whitespace = " \t\n\r\f\v";
    const size_t first = text.find_first_not_of(whitespace);
    if (first == std::string::npos) return {};
    const size_t last = text.find_last_not_of(whitespace);
    return text.substr(first, last - first + 1);
}

} // namespace mft_detail

extern "C" {

mft_status mft_sft_train_token_batch(mft_trainer* trainer,
                                     const mft_token_batch* batch,
                                     mft_sft_step_result* out_result) {
    if (out_result) *out_result = {};
    return mft_detail::guarded([&] {
        auto& handle = mft_detail::require_trainer(trainer, MFT_TRAINER_KIND_SFT);
        if (!batch || !out_result)
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "batch/out_result is null");
        const size_t count = mft_detail::checked_element_count(
            batch->batch_size, batch->sequence_length);
        if (batch->element_count != count || !batch->input_ids ||
            !batch->attention_mask || !batch->labels)
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "invalid token batch buffers");
        const std::vector<int64_t> shape{
            static_cast<int64_t>(batch->batch_size),
            static_cast<int64_t>(batch->sequence_length)};
        auto ids = std::make_shared<ops::Tensor>(
            shape, batch->input_ids, ops::kInt32, ops::kCPU);
        auto mask = std::make_shared<ops::Tensor>(
            shape, batch->attention_mask, ops::kFloat32, ops::kCPU);
        auto labels = std::make_shared<ops::Tensor>(
            shape, batch->labels, ops::kInt32, ops::kCPU);
        mft_detail::set_sft_result(handle.sft->train_step(ids, mask, labels), *out_result);
    });
}

mft_status mft_sft_train_text_batch(mft_trainer* trainer,
                                    mft_tokenizer* tokenizer,
                                    const mft_string_view* texts,
                                    size_t text_count,
                                    const mft_causal_batch_params* batch_params,
                                    mft_sft_step_result* out_result) {
    if (out_result) *out_result = {};
    return mft_detail::guarded([&] {
        auto& handle = mft_detail::require_trainer(trainer, MFT_TRAINER_KIND_SFT);
        if (!out_result) mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "out_result is null");
        const auto values = mft_detail::params_or_default(
            batch_params, mft_causal_batch_default_params, "causal batch params");
        if (values.sequence_length <= 1)
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "sequence_length must be > 1");
        if (values.ignore_index != handle.sft_ignore_index)
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT,
                             "batch ignore_index must equal trainer ignore_index");
        ops::CausalLMBatchConfig config;
        config.sequence_length = values.sequence_length;
        config.ignore_index = values.ignore_index;
        config.truncate = values.truncate;
        config.mask_padding_labels = values.mask_padding_labels;
        config.append_eos = values.append_eos;
        const auto rows = mft_detail::strings_from_views(texts, text_count, "texts");
        auto batch = ops::make_causal_lm_batch(
            mft_detail::require_tokenizer(tokenizer), rows, config);
        mft_detail::set_sft_result(handle.sft->train_step(batch), *out_result);
    });
}

mft_status mft_dpo_train_preference_batch(
    mft_trainer* trainer,
    mft_tokenizer* tokenizer,
    const mft_preference_sample* samples,
    size_t sample_count,
    const mft_preference_batch_params* batch_params,
    mft_dpo_step_result* out_result) {
    if (out_result) *out_result = {};
    return mft_detail::guarded([&] {
        auto& handle = mft_detail::require_trainer(trainer, MFT_TRAINER_KIND_DPO);
        if (!out_result) mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "out_result is null");
        const auto values = mft_detail::params_or_default(
            batch_params, mft_preference_batch_default_params, "preference batch params");
        const auto inputs = mft_detail::preference_inputs_from_c(samples, sample_count);
        if (inputs.references == mft_detail::ReferenceMode::Mixed)
            mft_detail::fail(MFT_STATUS_DATA_ERROR,
                             "preference batch mixes cached and uncached reference logps");
        if (inputs.references == mft_detail::ReferenceMode::None && !handle.reference)
            mft_detail::fail(MFT_STATUS_INVALID_STATE,
                             "DPO requires cached reference logps or a reference model");
        auto batch = mft_detail::make_strict_preference_batch(
            mft_detail::require_tokenizer(tokenizer), inputs, values);
        mft_detail::set_dpo_result(handle.dpo->train_step(batch), *out_result);
    });
}

mft_status mft_score_preference_batch(
    mft_model* model,
    mft_tokenizer* tokenizer,
    const mft_preference_sample* samples,
    size_t sample_count,
    const mft_preference_batch_params* batch_params,
    float* out_chosen_logps,
    float* out_rejected_logps,
    size_t output_capacity,
    size_t* out_count,
    int32_t* out_valid_tokens) {
    if (out_count) *out_count = 0;
    if (out_valid_tokens) *out_valid_tokens = 0;
    return mft_detail::guarded([&] {
        if (!out_count || !out_valid_tokens)
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "score output metadata is null");
        const auto values = mft_detail::params_or_default(
            batch_params, mft_preference_batch_default_params, "preference batch params");
        const auto inputs = mft_detail::preference_inputs_from_c(samples, sample_count);
        auto batch = mft_detail::make_strict_preference_batch(
            mft_detail::require_tokenizer(tokenizer), inputs, values);
        auto scores = ops::score_preference_logps(
            *mft_detail::require_model(model).model, batch);
        if (scores.chosen_logps.size() != sample_count ||
            scores.rejected_logps.size() != sample_count)
            mft_detail::fail(MFT_STATUS_INTERNAL_ERROR, "preference score count mismatch");
        *out_count = sample_count;
        *out_valid_tokens = scores.valid_response_token_count;
        if (!out_chosen_logps || !out_rejected_logps) {
            if (output_capacity != 0)
                mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "null score buffer with capacity");
            mft_detail::fail(MFT_STATUS_BUFFER_TOO_SMALL, "preference score buffer too small");
        }
        if (output_capacity < sample_count)
            mft_detail::fail(MFT_STATUS_BUFFER_TOO_SMALL, "preference score buffer too small");
        std::copy(scores.chosen_logps.begin(), scores.chosen_logps.end(), out_chosen_logps);
        std::copy(scores.rejected_logps.begin(), scores.rejected_logps.end(), out_rejected_logps);
    });
}

mft_status mft_kto_train_batch(mft_trainer* trainer,
                               mft_tokenizer* tokenizer,
                               const mft_kto_sample* samples,
                               size_t sample_count,
                               const mft_kto_batch_params* batch_params,
                               mft_kto_step_result* out_result) {
    if (out_result) *out_result = {};
    return mft_detail::guarded([&] {
        auto& handle = mft_detail::require_trainer(trainer, MFT_TRAINER_KIND_KTO);
        if (!out_result) mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "out_result is null");
        const auto values = mft_detail::params_or_default(
            batch_params, mft_kto_batch_default_params, "KTO batch params");
        const auto inputs = mft_detail::kto_inputs_from_c(samples, sample_count);
        if (inputs.references == mft_detail::ReferenceMode::Mixed)
            mft_detail::fail(MFT_STATUS_DATA_ERROR,
                             "KTO batch mixes cached and uncached reference logps");
        auto batch = mft_detail::make_strict_kto_batch(
            mft_detail::require_tokenizer(tokenizer), inputs, values);
        if (inputs.references == mft_detail::ReferenceMode::None) {
            if (!handle.reference)
                mft_detail::fail(MFT_STATUS_INVALID_STATE,
                                 "KTO requires cached reference logps or a reference model");
            const auto scores = ops::score_kto_logps(*handle.reference->model, batch);
            batch.ref_logps = scores.response_logps;
            batch.ref_kl_logps = scores.kl_response_logps;
        }
        if (!batch.has_cached_reference_logps())
            mft_detail::fail(MFT_STATUS_DATA_ERROR, "incomplete KTO reference logps");
        mft_detail::set_kto_result(handle.kto->train_step(batch), *out_result);
    });
}

mft_status mft_score_kto_batch(
    mft_model* model,
    mft_tokenizer* tokenizer,
    const mft_kto_sample* samples,
    size_t sample_count,
    const mft_kto_batch_params* batch_params,
    float* out_logps,
    float* out_kl_logps,
    size_t output_capacity,
    size_t* out_count,
    int32_t* out_valid_tokens,
    int32_t* out_valid_kl_tokens) {
    if (out_count) *out_count = 0;
    if (out_valid_tokens) *out_valid_tokens = 0;
    if (out_valid_kl_tokens) *out_valid_kl_tokens = 0;
    return mft_detail::guarded([&] {
        if (!out_count || !out_valid_tokens || !out_valid_kl_tokens)
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "KTO score metadata is null");
        const auto values = mft_detail::params_or_default(
            batch_params, mft_kto_batch_default_params, "KTO batch params");
        const auto inputs = mft_detail::kto_inputs_from_c(samples, sample_count);
        auto batch = mft_detail::make_strict_kto_batch(
            mft_detail::require_tokenizer(tokenizer), inputs, values);
        const auto scores = ops::score_kto_logps(
            *mft_detail::require_model(model).model, batch);
        if (scores.response_logps.size() != sample_count ||
            scores.kl_response_logps.size() != sample_count)
            mft_detail::fail(MFT_STATUS_INTERNAL_ERROR, "KTO score count mismatch");
        *out_count = sample_count;
        *out_valid_tokens = scores.valid_response_token_count;
        *out_valid_kl_tokens = scores.valid_kl_response_token_count;
        if (!out_logps || !out_kl_logps) {
            if (output_capacity != 0)
                mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "null score buffer with capacity");
            mft_detail::fail(MFT_STATUS_BUFFER_TOO_SMALL, "KTO score buffer too small");
        }
        if (output_capacity < sample_count)
            mft_detail::fail(MFT_STATUS_BUFFER_TOO_SMALL, "KTO score buffer too small");
        std::copy(scores.response_logps.begin(), scores.response_logps.end(), out_logps);
        std::copy(scores.kl_response_logps.begin(), scores.kl_response_logps.end(), out_kl_logps);
    });
}

mft_status mft_kto_train_preference_batch_compat(
    mft_trainer* trainer,
    mft_tokenizer* tokenizer,
    const mft_preference_sample* samples,
    size_t sample_count,
    const mft_preference_batch_params* batch_params,
    mft_kto_preference_step_result* out_result) {
    if (out_result) *out_result = {};
    return mft_detail::guarded([&] {
        auto& handle = mft_detail::require_trainer(trainer, MFT_TRAINER_KIND_KTO);
        if (!out_result) mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "out_result is null");
        const auto values = mft_detail::params_or_default(
            batch_params, mft_preference_batch_default_params, "preference batch params");
        const auto inputs = mft_detail::preference_inputs_from_c(samples, sample_count);
        if (inputs.references == mft_detail::ReferenceMode::Mixed)
            mft_detail::fail(MFT_STATUS_DATA_ERROR,
                             "compat KTO batch mixes cached and uncached reference logps");
        if (inputs.references == mft_detail::ReferenceMode::None && !handle.reference)
            mft_detail::fail(MFT_STATUS_INVALID_STATE,
                             "compat KTO requires cached logps or a reference model");
        auto batch = mft_detail::make_strict_preference_batch(
            mft_detail::require_tokenizer(tokenizer), inputs, values);
        mft_detail::set_kto_compat_result(handle.kto->train_step(batch), *out_result);
    });
}

mft_status mft_model_save_lora_adapter_jsonl(mft_model* model, const char* path) {
    return mft_detail::guarded([&] {
        auto& state = mft_detail::require_model(model);
        if (!state.lora_initialized)
            mft_detail::fail(MFT_STATUS_INVALID_STATE, "LoRA is not initialized");
        state.model->save_lora_adapter_jsonl(mft_detail::require_path(path, "path"));
    });
}

mft_status mft_model_load_lora_adapter_jsonl(mft_model* model, const char* path) {
    return mft_detail::guarded([&] {
        auto& state = mft_detail::require_model(model);
        if (state.active_trainers.load() != 0)
            mft_detail::fail(MFT_STATUS_INVALID_STATE,
                             "cannot load an adapter while a trainer is active");
        if (!state.lora_initialized)
            mft_detail::fail(MFT_STATUS_INVALID_STATE, "initialize LoRA before loading an adapter");
        const std::string adapter_path = mft_detail::require_path(path, "path");
        try {
            state.model->load_lora_adapter_jsonl(adapter_path);
        } catch (const std::filesystem::filesystem_error&) {
            throw;
        } catch (const std::invalid_argument& error) {
            mft_detail::fail(MFT_STATUS_DATA_ERROR, error.what());
        } catch (const std::out_of_range& error) {
            mft_detail::fail(MFT_STATUS_DATA_ERROR, error.what());
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            if (message.find("failed to open") != std::string::npos ||
                message.find("failed while reading") != std::string::npos) {
                mft_detail::fail(MFT_STATUS_IO_ERROR, message);
            }
            mft_detail::fail(MFT_STATUS_DATA_ERROR, message);
        }
    });
}

mft_status mft_model_generate(mft_model* model,
                              mft_tokenizer* tokenizer,
                              mft_string_view prompt,
                              const mft_generation_params* params,
                              char* text,
                              size_t capacity,
                              size_t* out_size) {
    if (out_size) *out_size = 0;
    return mft_detail::guarded([&] {
        auto& state = mft_detail::require_model(model);
        auto& tokenization = mft_detail::require_tokenizer(tokenizer);
        const auto values = mft_detail::params_or_default(
            params, mft_generation_default_params, "generation params");
        mft_detail::require_finite(values.temperature, "temperature");
        mft_detail::require_finite(values.top_p, "top_p");
        if (values.max_new_tokens <= 0 || values.temperature < 0 ||
            !(values.top_p > 0) || values.top_p > 1)
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "invalid generation parameters");
        const std::string original = mft_detail::string_from_view(prompt, "prompt");
        if (original.empty())
            mft_detail::fail(MFT_STATUS_INVALID_ARGUMENT, "prompt must not be empty");
        const bool qwen = state.model->family() == ops::ModelFamily::Qwen;
        const std::string input = qwen && values.apply_qwen_chat_template
            ? mft_detail::make_qwen_chat_prompt(original) : original;
        std::vector<int> tokens = tokenization.encode(input);
        if (tokens.empty()) mft_detail::fail(MFT_STATUS_DATA_ERROR, "prompt encoded to zero tokens");
        const size_t prompt_tokens = tokens.size();
        const int eos = tokenization.get_eos_token();
        std::mt19937 random(static_cast<uint32_t>(values.seed));
        ops::QwenKVCache cache;
        for (int32_t step = 0; step < values.max_new_tokens; ++step) {
            const auto input_ids = qwen && step > 0
                ? mft_detail::make_input_ids({tokens.back()})
                : mft_detail::make_input_ids(tokens);
            ops::TensorPtr logits;
            if (qwen) logits = state.model->forward_qwen_with_cache(input_ids, cache);
            else logits = state.model->forward(
                input_ids, mft_detail::make_attention_mask(tokens.size()));
            const int next = mft_detail::sample_last_token(
                logits, values.temperature, values.top_p, random);
            tokens.push_back(next);
            if ((eos >= 0 && next == eos) ||
                (qwen && (next == 151643 || next == 151645))) break;
        }
        const std::vector<int> completion(
            tokens.begin() + static_cast<std::ptrdiff_t>(prompt_tokens), tokens.end());
        std::string output = tokenization.decode(completion);
        if (qwen && values.clean_qwen_output)
            output = mft_detail::clean_qwen_output(std::move(output));
        const auto status = mft_detail::copy_string_to_caller(output, text, capacity, out_size);
        if (status != MFT_STATUS_OK) mft_detail::fail(status, "generation buffer too small");
    });
}

} // extern "C"
