#ifndef MOBILE_FINETUNER_H
#define MOBILE_FINETUNER_H

// Stable, llama.cpp-style C ABI for MobileFineTuner.
//
// Model paths refer to Hugging Face directories containing config/tokenizer
// assets and safetensors weights. They are not GGUF file paths. All functions
// are synchronous. Unless stated otherwise, input pointers only need to remain
// valid until the function returns.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(MFT_SHARED)
#  if defined(MFT_BUILD)
#    define MFT_API __declspec(dllexport)
#  else
#    define MFT_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define MFT_API __attribute__((visibility("default")))
#else
#  define MFT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MFT_API_VERSION 1u

typedef struct mft_model mft_model;
typedef struct mft_tokenizer mft_tokenizer;
typedef struct mft_trainer mft_trainer;

typedef enum mft_status {
    MFT_STATUS_OK = 0,
    MFT_STATUS_INVALID_ARGUMENT = 1,
    MFT_STATUS_INVALID_STATE = 2,
    MFT_STATUS_UNSUPPORTED = 3,
    MFT_STATUS_DATA_ERROR = 4,
    MFT_STATUS_IO_ERROR = 5,
    MFT_STATUS_OUT_OF_MEMORY = 6,
    MFT_STATUS_BUFFER_TOO_SMALL = 7,
    MFT_STATUS_CANCELLED = 8,
    MFT_STATUS_INTERNAL_ERROR = 9
} mft_status;

typedef enum mft_model_family {
    MFT_MODEL_FAMILY_UNKNOWN = 0,
    MFT_MODEL_FAMILY_GPT2 = 1,
    MFT_MODEL_FAMILY_GEMMA = 2,
    MFT_MODEL_FAMILY_LLAMA = 3,
    MFT_MODEL_FAMILY_MISTRAL = 4,
    MFT_MODEL_FAMILY_QWEN = 5
} mft_model_family;

typedef enum mft_trainer_kind {
    MFT_TRAINER_KIND_UNKNOWN = 0,
    MFT_TRAINER_KIND_SFT = 1,
    MFT_TRAINER_KIND_DPO = 2,
    MFT_TRAINER_KIND_KTO = 3
} mft_trainer_kind;

typedef enum mft_kto_label {
    MFT_KTO_UNDESIRABLE = 0,
    MFT_KTO_DESIRABLE = 1
} mft_kto_label;

typedef struct mft_string_view {
    const char *data;
    size_t size;
} mft_string_view;

typedef struct mft_model_params {
    uint32_t struct_size;
    bool load_weights;
    bool verbose;
    bool auto_promote_fp16;
    bool strict_key_check;
    bool strict_shape_check;
} mft_model_params;

typedef struct mft_tokenizer_params {
    uint32_t struct_size;
    // Empty means infer the tokenizer type from the model directory.
    mft_string_view model_type_override;
} mft_tokenizer_params;

typedef struct mft_tokenizer_encode_params {
    uint32_t struct_size;
    bool add_special_tokens;
    int32_t max_length;
    bool truncation;
} mft_tokenizer_encode_params;

typedef struct mft_tokenizer_decode_params {
    uint32_t struct_size;
    bool skip_special_tokens;
} mft_tokenizer_decode_params;

typedef struct mft_lora_params {
    uint32_t struct_size;
    int32_t rank;
    float alpha;
    float dropout;
    uint64_t seed;
    const mft_string_view *target_modules;
    size_t target_module_count;
} mft_lora_params;

typedef struct mft_sft_trainer_params {
    uint32_t struct_size;
    float learning_rate;
    float weight_decay;
    float max_grad_norm;
    int32_t ignore_index;
    bool use_streaming_loss;
    int32_t gradient_accumulation_steps;
} mft_sft_trainer_params;

typedef struct mft_dpo_trainer_params {
    uint32_t struct_size;
    float learning_rate;
    float weight_decay;
    float max_grad_norm;
    float beta;
    bool use_streaming_loss;
    int32_t gradient_accumulation_steps;
} mft_dpo_trainer_params;

typedef struct mft_kto_trainer_params {
    uint32_t struct_size;
    float learning_rate;
    float weight_decay;
    float max_grad_norm;
    float beta;
    float desirable_weight;
    float undesirable_weight;
    bool use_streaming_loss;
    int32_t gradient_accumulation_steps;
} mft_kto_trainer_params;

typedef struct mft_causal_batch_params {
    uint32_t struct_size;
    int32_t sequence_length;
    int32_t ignore_index;
    bool truncate;
    bool mask_padding_labels;
    bool append_eos;
} mft_causal_batch_params;

typedef struct mft_preference_batch_params {
    uint32_t struct_size;
    int32_t sequence_length;
    bool truncate;
    bool add_special_tokens;
    bool append_eos_to_response;
} mft_preference_batch_params;

typedef struct mft_kto_batch_params {
    uint32_t struct_size;
    int32_t sequence_length;
    bool truncate;
    bool add_special_tokens;
    bool append_eos_to_completion;
} mft_kto_batch_params;

typedef struct mft_generation_params {
    uint32_t struct_size;
    int32_t max_new_tokens;
    float temperature;
    float top_p;
    uint64_t seed;
    bool apply_qwen_chat_template;
    bool clean_qwen_output;
} mft_generation_params;

typedef struct mft_token_batch {
    const int32_t *input_ids;
    const float *attention_mask;
    const int32_t *labels;
    size_t element_count;
    size_t batch_size;
    size_t sequence_length;
} mft_token_batch;

typedef struct mft_preference_sample {
    mft_string_view prompt;
    mft_string_view chosen;
    mft_string_view rejected;
    bool has_reference_logps;
    float ref_chosen_logp;
    float ref_rejected_logp;
} mft_preference_sample;

// Canonical KTO input. KL examples are constructed deterministically by the
// core batch builder from this ordered batch. Cached ref_logp/ref_kl_logp must
// therefore have been scored with the same sample order and batch parameters.
typedef struct mft_kto_sample {
    mft_string_view prompt;
    mft_string_view completion;
    mft_kto_label label;
    bool has_reference_logps;
    float ref_logp;
    float ref_kl_logp;
} mft_kto_sample;

typedef struct mft_sft_step_result {
    float loss;
    float accumulated_loss;
    int32_t trainable_tensor_count;
    int32_t valid_label_count;
    int32_t accumulation_step;
    int32_t gradient_accumulation_steps;
    bool optimizer_step;
} mft_sft_step_result;

typedef struct mft_dpo_step_result {
    float loss;
    float accumulated_loss;
    float chosen_reward;
    float rejected_reward;
    float reward_margin;
    float reward_accuracy;
    int32_t trainable_tensor_count;
    int32_t pair_count;
    int32_t valid_response_token_count;
    int32_t accumulation_step;
    int32_t gradient_accumulation_steps;
    bool optimizer_step;
} mft_dpo_step_result;

typedef struct mft_kto_step_result {
    float loss;
    float accumulated_loss;
    float kl_estimate;
    float desirable_reward;
    float undesirable_reward;
    float reward_margin;
    float reward_accuracy;
    int32_t trainable_tensor_count;
    int32_t sample_count;
    int32_t desirable_count;
    int32_t undesirable_count;
    int32_t valid_response_token_count;
    int32_t valid_kl_response_token_count;
    int32_t accumulation_step;
    int32_t gradient_accumulation_steps;
    bool optimizer_step;
} mft_kto_step_result;

// Result for the explicitly named legacy pair-to-binary approximation.
typedef struct mft_kto_preference_step_result {
    float loss;
    float accumulated_loss;
    float desirable_reward;
    float undesirable_reward;
    float reward_margin;
    float reward_accuracy;
    int32_t trainable_tensor_count;
    int32_t pair_count;
    int32_t sample_count;
    int32_t valid_response_token_count;
    int32_t accumulation_step;
    int32_t gradient_accumulation_steps;
    bool optimizer_step;
} mft_kto_preference_step_result;

MFT_API uint32_t mft_api_version(void);
MFT_API const char *mft_version_string(void);
MFT_API const char *mft_status_string(mft_status status);

// The returned pointer is thread-local and remains valid until the next
// fallible MobileFineTuner call on the same thread.
MFT_API const char *mft_last_error(void);

// Copies the thread-local error with NUL termination when capacity > 0.
// Returns the required UTF-8 byte count excluding the terminating NUL.
MFT_API size_t mft_get_last_error(char *buffer, size_t capacity);

MFT_API mft_model_params mft_model_default_params(void);
MFT_API mft_tokenizer_params mft_tokenizer_default_params(void);
MFT_API mft_tokenizer_encode_params mft_tokenizer_encode_default_params(void);
MFT_API mft_tokenizer_decode_params mft_tokenizer_decode_default_params(void);
MFT_API mft_lora_params mft_lora_default_params(void);
MFT_API mft_sft_trainer_params mft_sft_trainer_default_params(void);
MFT_API mft_dpo_trainer_params mft_dpo_trainer_default_params(void);
MFT_API mft_kto_trainer_params mft_kto_trainer_default_params(void);
MFT_API mft_causal_batch_params mft_causal_batch_default_params(void);
MFT_API mft_preference_batch_params mft_preference_batch_default_params(void);
MFT_API mft_kto_batch_params mft_kto_batch_default_params(void);
MFT_API mft_generation_params mft_generation_default_params(void);

MFT_API mft_status mft_model_load_from_dir(
    const char *model_dir,
    const mft_model_params *params,
    mft_model **out_model);
MFT_API void mft_model_free(mft_model *model);
MFT_API mft_status mft_model_get_family(
    const mft_model *model,
    mft_model_family *out_family);
MFT_API mft_status mft_model_init_lora(
    mft_model *model,
    const mft_lora_params *params);
MFT_API mft_status mft_model_get_trainable_tensor_count(
    const mft_model *model,
    size_t *out_count);

MFT_API mft_status mft_tokenizer_load_from_dir(
    const char *model_dir,
    const mft_tokenizer_params *params,
    mft_tokenizer **out_tokenizer);
MFT_API void mft_tokenizer_free(mft_tokenizer *tokenizer);

// Pass tokens=NULL and token_capacity=0 to query the required token count.
// A non-empty result returns MFT_STATUS_BUFFER_TOO_SMALL for this query.
MFT_API mft_status mft_tokenizer_encode(
    mft_tokenizer *tokenizer,
    mft_string_view text,
    const mft_tokenizer_encode_params *params,
    int32_t *tokens,
    size_t token_capacity,
    size_t *out_token_count);

// Pass text=NULL and text_capacity=0 to query the required byte count. The
// reported count excludes the NUL; a real output buffer needs count + 1 bytes.
// A non-empty result returns MFT_STATUS_BUFFER_TOO_SMALL for this query.
MFT_API mft_status mft_tokenizer_decode(
    mft_tokenizer *tokenizer,
    const int32_t *tokens,
    size_t token_count,
    const mft_tokenizer_decode_params *params,
    char *text,
    size_t text_capacity,
    size_t *out_text_size);

MFT_API mft_status mft_sft_trainer_create(
    mft_model *policy,
    const mft_sft_trainer_params *params,
    mft_trainer **out_trainer);
MFT_API mft_status mft_dpo_trainer_create(
    mft_model *policy,
    mft_model *reference_or_null,
    const mft_dpo_trainer_params *params,
    mft_trainer **out_trainer);
MFT_API mft_status mft_kto_trainer_create(
    mft_model *policy,
    mft_model *reference_or_null,
    const mft_kto_trainer_params *params,
    mft_trainer **out_trainer);
MFT_API void mft_trainer_free(mft_trainer *trainer);
MFT_API mft_status mft_trainer_get_kind(
    const mft_trainer *trainer,
    mft_trainer_kind *out_kind);

MFT_API mft_status mft_sft_train_token_batch(
    mft_trainer *trainer,
    const mft_token_batch *batch,
    mft_sft_step_result *out_result);
MFT_API mft_status mft_sft_train_text_batch(
    mft_trainer *trainer,
    mft_tokenizer *tokenizer,
    const mft_string_view *texts,
    size_t text_count,
    const mft_causal_batch_params *batch_params,
    mft_sft_step_result *out_result);

MFT_API mft_status mft_dpo_train_preference_batch(
    mft_trainer *trainer,
    mft_tokenizer *tokenizer,
    const mft_preference_sample *samples,
    size_t sample_count,
    const mft_preference_batch_params *batch_params,
    mft_dpo_step_result *out_result);
MFT_API mft_status mft_score_preference_batch(
    mft_model *model,
    mft_tokenizer *tokenizer,
    const mft_preference_sample *samples,
    size_t sample_count,
    const mft_preference_batch_params *batch_params,
    float *out_chosen_logps,
    float *out_rejected_logps,
    size_t output_capacity,
    size_t *out_count,
    int32_t *out_valid_response_token_count);

MFT_API mft_status mft_kto_train_batch(
    mft_trainer *trainer,
    mft_tokenizer *tokenizer,
    const mft_kto_sample *samples,
    size_t sample_count,
    const mft_kto_batch_params *batch_params,
    mft_kto_step_result *out_result);
MFT_API mft_status mft_score_kto_batch(
    mft_model *model,
    mft_tokenizer *tokenizer,
    const mft_kto_sample *samples,
    size_t sample_count,
    const mft_kto_batch_params *batch_params,
    float *out_logps,
    float *out_kl_logps,
    size_t output_capacity,
    size_t *out_count,
    int32_t *out_valid_response_token_count,
    int32_t *out_valid_kl_response_token_count);

// Compatibility only: converts chosen/rejected pairs into desirable and
// undesirable rows. This is not the canonical KTO binary-feedback API.
MFT_API mft_status mft_kto_train_preference_batch_compat(
    mft_trainer *trainer,
    mft_tokenizer *tokenizer,
    const mft_preference_sample *samples,
    size_t sample_count,
    const mft_preference_batch_params *batch_params,
    mft_kto_preference_step_result *out_result);

MFT_API mft_status mft_model_save_lora_adapter_jsonl(
    mft_model *model,
    const char *path);
MFT_API mft_status mft_model_load_lora_adapter_jsonl(
    mft_model *model,
    const char *path);

// Batch-one generation. Querying the required output size performs generation;
// callers should normally provide a suitably sized buffer in the first call.
MFT_API mft_status mft_model_generate(
    mft_model *model,
    mft_tokenizer *tokenizer,
    mft_string_view prompt,
    const mft_generation_params *params,
    char *text,
    size_t text_capacity,
    size_t *out_text_size);

#ifdef __cplusplus
} // extern "C"
#endif

// Preserve source compatibility for existing C++ users that historically
// included mobile_finetuner.h as an umbrella header. Define MFT_C_API_ONLY to
// request the pure C surface from C++.
#if defined(__cplusplus) && !defined(MFT_C_API_ONLY)
#include "mobile_finetuner.hpp"
#endif

#endif // MOBILE_FINETUNER_H
