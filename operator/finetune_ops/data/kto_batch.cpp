#include "kto_batch.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace ops {

namespace {

struct EncodedBranch {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> attention_mask;
    std::vector<int32_t> response_mask;
    int valid_response_token_count = 0;
};

void validate_config(const KTOBatchConfig& config) {
    if (config.sequence_length <= 1) {
        throw std::invalid_argument("KTOBatchConfig.sequence_length must be > 1");
    }
}

int resolve_pad_token(const Tokenizer& tokenizer) {
    int pad = tokenizer.get_pad_token();
    if (pad < 0) {
        pad = tokenizer.get_eos_token();
    }
    return pad >= 0 ? pad : 0;
}

std::vector<int> encode_segment(Tokenizer& tokenizer,
                                const std::string& text,
                                const KTOBatchConfig& config) {
    TokenizerEncodeOptions options;
    options.add_special_tokens = config.add_special_tokens;
    options.max_length = 0;
    options.truncation = false;
    return tokenizer.encode_with_options(text, options);
}

EncodedBranch encode_branch(Tokenizer& tokenizer,
                            const std::string& prompt,
                            const std::string& completion,
                            const KTOBatchConfig& config) {
    std::vector<int> prompt_ids = encode_segment(tokenizer, prompt, config);
    std::vector<int> completion_ids = encode_segment(tokenizer, completion, config);
    if (config.append_eos_to_completion) {
        const int eos = tokenizer.get_eos_token();
        if (eos < 0) {
            throw std::invalid_argument(
                "append_eos_to_completion=true requires tokenizer EOS token");
        }
        if (completion_ids.empty() || completion_ids.back() != eos) {
            completion_ids.push_back(eos);
        }
    }
    if (completion_ids.empty()) {
        throw std::invalid_argument("KTO samples require a non-empty completion");
    }

    std::vector<int32_t> ids;
    std::vector<int32_t> response;
    ids.reserve(prompt_ids.size() + completion_ids.size());
    response.reserve(prompt_ids.size() + completion_ids.size());
    for (int id : prompt_ids) {
        ids.push_back(static_cast<int32_t>(id));
        response.push_back(0);
    }
    for (int id : completion_ids) {
        ids.push_back(static_cast<int32_t>(id));
        response.push_back(1);
    }

    const int seq = config.sequence_length;
    if (config.truncate && static_cast<int>(ids.size()) > seq) {
        ids.resize(static_cast<std::size_t>(seq));
        response.resize(static_cast<std::size_t>(seq));
    }
    if (static_cast<int>(ids.size()) > seq) {
        throw std::invalid_argument("KTO sample exceeds sequence_length and truncate=false");
    }

    EncodedBranch encoded;
    for (std::size_t s = 1; s < response.size(); ++s) {
        if (response[s] != 0) {
            ++encoded.valid_response_token_count;
        }
    }
    if (encoded.valid_response_token_count == 0) {
        throw std::invalid_argument(
            "KTO sample has no completion token after causal shift/truncation");
    }

    encoded.input_ids.assign(static_cast<std::size_t>(seq),
                             static_cast<int32_t>(resolve_pad_token(tokenizer)));
    encoded.attention_mask.assign(static_cast<std::size_t>(seq), 0);
    encoded.response_mask.assign(static_cast<std::size_t>(seq), 0);
    const int len = static_cast<int>(ids.size());
    const int offset = tokenizer.default_left_padding() ? seq - len : 0;
    for (int i = 0; i < len; ++i) {
        const int dst = offset + i;
        encoded.input_ids[static_cast<std::size_t>(dst)] = ids[static_cast<std::size_t>(i)];
        encoded.attention_mask[static_cast<std::size_t>(dst)] = 1;
        encoded.response_mask[static_cast<std::size_t>(dst)] =
            response[static_cast<std::size_t>(i)];
    }
    return encoded;
}

void copy_branch_row(KTOBatch& batch,
                     int row,
                     const EncodedBranch& encoded,
                     bool kl_branch) {
    const std::size_t seq = static_cast<std::size_t>(batch.sequence_length);
    const std::size_t offset = static_cast<std::size_t>(row) * seq;
    TensorPtr ids = kl_branch ? batch.kl_input_ids : batch.input_ids;
    TensorPtr attention = kl_branch ? batch.kl_attention_mask : batch.attention_mask;
    TensorPtr response = kl_branch ? batch.kl_response_mask : batch.response_mask;
    std::memcpy(ids->data<int32_t>() + offset,
                encoded.input_ids.data(), seq * sizeof(int32_t));
    for (std::size_t s = 0; s < seq; ++s) {
        attention->data<float>()[offset + s] =
            static_cast<float>(encoded.attention_mask[s]);
    }
    std::memcpy(response->data<int32_t>() + offset,
                encoded.response_mask.data(), seq * sizeof(int32_t));
}

}  // namespace

bool KTOBatch::has_cached_reference_logps() const {
    return batch_size > 0 &&
           static_cast<int>(ref_logps.size()) == batch_size &&
           static_cast<int>(ref_kl_logps.size()) == batch_size;
}

KTOBatch make_kto_batch(Tokenizer& tokenizer,
                        const std::vector<KTOSample>& samples,
                        const KTOBatchConfig& config) {
    validate_config(config);
    if (samples.size() < 2) {
        throw std::invalid_argument(
            "make_kto_batch requires at least two samples for mismatched KL completions");
    }

    const int batch_size = static_cast<int>(samples.size());
    const int seq = config.sequence_length;
    const std::vector<int64_t> shape{batch_size, seq};

    KTOBatch batch;
    batch.input_ids = std::make_shared<Tensor>(shape, kInt32, kCPU);
    batch.attention_mask = std::make_shared<Tensor>(shape, kFloat32, kCPU);
    batch.response_mask = std::make_shared<Tensor>(shape, kInt32, kCPU);
    batch.kl_input_ids = std::make_shared<Tensor>(shape, kInt32, kCPU);
    batch.kl_attention_mask = std::make_shared<Tensor>(shape, kFloat32, kCPU);
    batch.kl_response_mask = std::make_shared<Tensor>(shape, kInt32, kCPU);
    batch.batch_size = batch_size;
    batch.sequence_length = seq;
    batch.labels.reserve(samples.size());
    batch.sample_indices.reserve(samples.size());
    batch.kl_completion_indices.reserve(samples.size());

    bool all_have_reference_logps = true;
    batch.ref_logps.reserve(samples.size());
    batch.ref_kl_logps.reserve(samples.size());

    for (int i = 0; i < batch_size; ++i) {
        const std::size_t row = static_cast<std::size_t>(i);
        const std::size_t completion_row =
            (row + samples.size() - 1) % samples.size();
        const KTOSample& sample = samples[row];
        const EncodedBranch matched =
            encode_branch(tokenizer, sample.prompt, sample.completion, config);
        const EncodedBranch mismatched =
            encode_branch(tokenizer, sample.prompt,
                          samples[completion_row].completion, config);

        copy_branch_row(batch, i, matched, false);
        copy_branch_row(batch, i, mismatched, true);
        batch.valid_response_token_count += matched.valid_response_token_count;
        batch.valid_kl_response_token_count += mismatched.valid_response_token_count;
        batch.labels.push_back(sample.desirable ? 1.0f : -1.0f);
        batch.sample_indices.push_back(row);
        batch.kl_completion_indices.push_back(completion_row);

        all_have_reference_logps =
            all_have_reference_logps && sample.has_reference_logps;
        if (sample.has_reference_logps) {
            if (!std::isfinite(sample.ref_logp) || !std::isfinite(sample.ref_kl_logp)) {
                throw std::invalid_argument("KTO cached reference logps must be finite");
            }
            batch.ref_logps.push_back(sample.ref_logp);
            batch.ref_kl_logps.push_back(sample.ref_kl_logp);
        }
    }

    if (!all_have_reference_logps) {
        batch.ref_logps.clear();
        batch.ref_kl_logps.clear();
    }
    return batch;
}

}  // namespace ops
