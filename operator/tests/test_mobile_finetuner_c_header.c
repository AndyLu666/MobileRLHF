#include "mobile_finetuner/mobile_finetuner.h"

#include <stdio.h>
#include <string.h>

_Static_assert(MFT_API_VERSION == 1u, "unexpected MobileFineTuner ABI version");
_Static_assert(sizeof(mft_string_view) >= sizeof(const char *) + sizeof(size_t),
               "mft_string_view is not a C-compatible pointer/size pair");

static int check(int condition, const char *message) {
    if (!condition) {
        fprintf(stderr, "MobileFineTuner C header smoke failed: %s\n", message);
        return 0;
    }
    return 1;
}

int main(void) {
    mft_model_params model_params = mft_model_default_params();
    mft_tokenizer_params tokenizer_params = mft_tokenizer_default_params();
    mft_tokenizer_encode_params encode_params = mft_tokenizer_encode_default_params();
    mft_tokenizer_decode_params decode_params = mft_tokenizer_decode_default_params();
    mft_lora_params lora_params = mft_lora_default_params();
    mft_sft_trainer_params sft_params = mft_sft_trainer_default_params();
    mft_dpo_trainer_params dpo_params = mft_dpo_trainer_default_params();
    mft_kto_trainer_params kto_params = mft_kto_trainer_default_params();
    mft_causal_batch_params causal_params = mft_causal_batch_default_params();
    mft_preference_batch_params preference_params =
        mft_preference_batch_default_params();
    mft_kto_batch_params kto_batch_params = mft_kto_batch_default_params();
    mft_generation_params generation_params = mft_generation_default_params();
    mft_model_family family = MFT_MODEL_FAMILY_UNKNOWN;
    const char *version = mft_version_string();
    const char *error;
    size_t required;
    char short_buffer[4] = {'x', 'x', 'x', 'x'};
    mft_status status;

    if (!check(mft_api_version() == MFT_API_VERSION, "API version mismatch") ||
        !check(version != NULL && version[0] != '\0', "empty version string") ||
        !check(mft_status_string(MFT_STATUS_OK) != NULL &&
                   mft_status_string(MFT_STATUS_OK)[0] != '\0',
               "empty OK status string")) {
        return 1;
    }

    if (!check(model_params.struct_size == sizeof(model_params),
               "model defaults struct_size mismatch") ||
        !check(tokenizer_params.struct_size == sizeof(tokenizer_params),
               "tokenizer defaults struct_size mismatch") ||
        !check(encode_params.struct_size == sizeof(encode_params),
               "encode defaults struct_size mismatch") ||
        !check(decode_params.struct_size == sizeof(decode_params),
               "decode defaults struct_size mismatch") ||
        !check(lora_params.struct_size == sizeof(lora_params),
               "LoRA defaults struct_size mismatch") ||
        !check(sft_params.struct_size == sizeof(sft_params),
               "SFT defaults struct_size mismatch") ||
        !check(dpo_params.struct_size == sizeof(dpo_params),
               "DPO defaults struct_size mismatch") ||
        !check(kto_params.struct_size == sizeof(kto_params),
               "KTO defaults struct_size mismatch") ||
        !check(causal_params.struct_size == sizeof(causal_params),
               "causal-batch defaults struct_size mismatch") ||
        !check(preference_params.struct_size == sizeof(preference_params),
               "preference-batch defaults struct_size mismatch") ||
        !check(kto_batch_params.struct_size == sizeof(kto_batch_params),
               "KTO-batch defaults struct_size mismatch") ||
        !check(generation_params.struct_size == sizeof(generation_params),
               "generation defaults struct_size mismatch")) {
        return 1;
    }

    if (!check(lora_params.rank > 0 && lora_params.alpha > 0.0f,
               "invalid LoRA defaults") ||
        !check(sft_params.learning_rate > 0.0f &&
                   sft_params.gradient_accumulation_steps > 0,
               "invalid SFT defaults") ||
        !check(dpo_params.beta > 0.0f &&
                   dpo_params.gradient_accumulation_steps > 0,
               "invalid DPO defaults") ||
        !check(kto_params.beta > 0.0f && kto_params.desirable_weight > 0.0f &&
                   kto_params.undesirable_weight > 0.0f,
               "invalid KTO defaults")) {
        return 1;
    }

    status = mft_model_get_family(NULL, &family);
    if (!check(status == MFT_STATUS_INVALID_ARGUMENT,
               "NULL model did not return invalid_argument")) {
        return 1;
    }
    error = mft_last_error();
    if (!check(error != NULL && error[0] != '\0', "missing thread-local error")) {
        return 1;
    }

    required = mft_get_last_error(NULL, 0);
    if (!check(required == strlen(error), "error size query mismatch") ||
        !check(required >= sizeof(short_buffer), "error fixture unexpectedly short")) {
        return 1;
    }
    if (!check(mft_get_last_error(short_buffer, sizeof(short_buffer)) == required,
               "short error copy changed required size") ||
        !check(short_buffer[sizeof(short_buffer) - 1] == '\0',
               "short error copy was not NUL-terminated")) {
        return 1;
    }

    /* Freeing a null opaque handle is intentionally a no-op. These calls also
       ensure the three destroy symbols are linkable from a C translation unit. */
    mft_model_free(NULL);
    mft_tokenizer_free(NULL);
    mft_trainer_free(NULL);
    return 0;
}
