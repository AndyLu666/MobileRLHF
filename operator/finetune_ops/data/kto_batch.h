#pragma once

#include "../core/tensor.h"
#include "../core/tokenizer.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ops {

struct KTOBatchConfig {
    int sequence_length = 128;
    bool truncate = true;
    bool add_special_tokens = false;
    bool append_eos_to_completion = false;
};

/**
 * @brief One unpaired KTO example.
 *
 * ref_logp is the frozen-reference score for prompt + completion. ref_kl_logp
 * is the score for this row's prompt combined with the rotated completion used
 * by the KL estimator. Cached reference scores are usable only when both are
 * supplied for every sample in the batch.
 */
struct KTOSample {
    std::string prompt;
    std::string completion;
    bool desirable = true;
    bool has_reference_logps = false;
    float ref_logp = 0.0f;
    float ref_kl_logp = 0.0f;
};

struct KTOBatch {
    TensorPtr input_ids;       // int32 [B,S], prompt_i + completion_i
    TensorPtr attention_mask;  // float32 [B,S]
    TensorPtr response_mask;   // int32 [B,S]

    // Canonical KTO KL branch: prompt_i + completion_(i-1 mod B).
    TensorPtr kl_input_ids;       // int32 [B,S]
    TensorPtr kl_attention_mask;  // float32 [B,S]
    TensorPtr kl_response_mask;   // int32 [B,S]

    std::vector<float> labels;  // +1 desirable, -1 undesirable
    std::vector<float> ref_logps;
    std::vector<float> ref_kl_logps;
    std::vector<std::size_t> sample_indices;
    std::vector<std::size_t> kl_completion_indices;

    int batch_size = 0;
    int sequence_length = 0;
    int valid_response_token_count = 0;
    int valid_kl_response_token_count = 0;

    bool has_cached_reference_logps() const;
};

/**
 * @brief Build a canonical KTO batch with a one-position completion rotation.
 *
 * At least two samples are required so every KL row uses a mismatched
 * completion. Row i uses the completion from row (i - 1 + B) % B.
 */
KTOBatch make_kto_batch(Tokenizer& tokenizer,
                        const std::vector<KTOSample>& samples,
                        const KTOBatchConfig& config = KTOBatchConfig());

}  // namespace ops
