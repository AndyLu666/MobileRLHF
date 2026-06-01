/**
 * @file preference_dataset.h
 * @brief Native preference-pair dataset for C++ DPO training
 *
 * Expected JSONL row format:
 * {"prompt":"...","chosen":" ...","rejected":" ..."}
 */

#pragma once

#include "../core/tensor.h"
#include "../core/tokenizer_bpe.h"
#include <random>
#include <string>
#include <vector>

namespace ops {

struct PreferenceDatasetConfig {
    std::string train_path;
    std::string valid_path;
    std::string test_path;
    int seq_len = 128;
    int batch_size = 2;
    uint64_t seed = 2025;
};

struct PreferenceEncodedSample {
    std::vector<int32_t> chosen_input_ids;
    std::vector<int32_t> chosen_attention_mask;
    std::vector<int32_t> chosen_response_mask;

    std::vector<int32_t> rejected_input_ids;
    std::vector<int32_t> rejected_attention_mask;
    std::vector<int32_t> rejected_response_mask;
};

struct PreferenceBatch {
    TensorPtr chosen_input_ids;       // [B, S] int32
    TensorPtr chosen_attention_mask;  // [B, S] int32
    TensorPtr chosen_response_mask;   // [B, S] int32

    TensorPtr rejected_input_ids;       // [B, S] int32
    TensorPtr rejected_attention_mask;  // [B, S] int32
    TensorPtr rejected_response_mask;   // [B, S] int32

    size_t num_samples = 0;
};

enum class PreferenceSplit {
    Train,
    Valid,
    Test,
};

class PreferenceDataset {
public:
    PreferenceDataset(const PreferenceDatasetConfig& cfg, GPT2BPETokenizer* tok);

    void load(PreferenceSplit split);
    void shuffle();
    void reset_cursor() { cursor_ = 0; }

    size_t num_samples() const { return samples_.size(); }
    size_t num_batches() const;
    size_t num_kept() const { return kept_samples_; }
    size_t num_skipped() const { return skipped_samples_; }

    PreferenceBatch get_batch(size_t batch_idx) const;
    PreferenceBatch next_batch(bool loop = true);

    void print_stats() const;

private:
    struct JsonRow {
        std::string prompt;
        std::string chosen;
        std::string rejected;
    };

    bool parse_jsonl_row(const std::string& line, JsonRow& row) const;
    bool encode_pair(const std::string& prompt,
                     const std::string& response,
                     std::vector<int32_t>& input_ids,
                     std::vector<int32_t>& attention_mask,
                     std::vector<int32_t>& response_mask) const;
    std::string path_for_split(PreferenceSplit split) const;

    PreferenceDatasetConfig cfg_;
    GPT2BPETokenizer* tok_;

    std::vector<PreferenceEncodedSample> samples_;
    std::vector<size_t> order_;
    mutable size_t cursor_ = 0;
    mutable std::mt19937_64 rng_;

    size_t kept_samples_ = 0;
    size_t skipped_samples_ = 0;
};

}  // namespace ops
