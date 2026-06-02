/**
 * @file preference_dataset.cpp
 * @brief Native preference-pair dataset for C++ DPO training
 */

#include "preference_dataset.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

namespace ops {

namespace {

static size_t skip_ws(const std::string& s, size_t pos) {
    while (pos < s.size()) {
        const char c = s[pos];
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
        ++pos;
    }
    return pos;
}

static bool parse_json_string_literal(const std::string& s,
                                      size_t quote_start,
                                      std::string& out,
                                      size_t* next_pos) {
    if (quote_start >= s.size() || s[quote_start] != '"') return false;

    std::string value;
    bool escape = false;
    for (size_t i = quote_start + 1; i < s.size(); ++i) {
        const char c = s[i];
        if (escape) {
            switch (c) {
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                default: value.push_back(c); break;
            }
            escape = false;
            continue;
        }
        if (c == '\\') {
            escape = true;
            continue;
        }
        if (c == '"') {
            out = std::move(value);
            if (next_pos) *next_pos = i + 1;
            return true;
        }
        value.push_back(c);
    }
    return false;
}

static bool extract_json_string_field(const std::string& line,
                                      const std::string& key,
                                      std::string& out) {
    const std::string needle = "\"" + key + "\"";
    size_t key_pos = line.find(needle);
    if (key_pos == std::string::npos) return false;

    size_t colon = line.find(':', key_pos + needle.size());
    if (colon == std::string::npos) return false;

    size_t value_pos = skip_ws(line, colon + 1);
    return parse_json_string_literal(line, value_pos, out, nullptr);
}

}  // namespace

PreferenceDataset::PreferenceDataset(const PreferenceDatasetConfig& cfg, GPT2BPETokenizer* tok)
    : cfg_(cfg), tok_(tok), rng_(cfg.seed) {}

std::string PreferenceDataset::path_for_split(PreferenceSplit split) const {
    if (split == PreferenceSplit::Train) return cfg_.train_path;
    if (split == PreferenceSplit::Valid) return cfg_.valid_path;
    return cfg_.test_path;
}

bool PreferenceDataset::parse_jsonl_row(const std::string& line, JsonRow& row) const {
    return extract_json_string_field(line, "prompt", row.prompt) &&
           extract_json_string_field(line, "chosen", row.chosen) &&
           extract_json_string_field(line, "rejected", row.rejected);
}

bool PreferenceDataset::encode_pair(const std::string& prompt,
                                    const std::string& response,
                                    std::vector<int32_t>& input_ids,
                                    std::vector<int32_t>& attention_mask,
                                    std::vector<int32_t>& response_mask) const {
    std::vector<int> prompt_ids = tok_->encode(prompt, false, 0, true);
    std::vector<int> response_ids = tok_->encode(response, false, 0, true);

    if (prompt_ids.empty() || response_ids.empty()) return false;

    std::vector<int32_t> flat_ids;
    std::vector<int32_t> flat_response_mask;
    flat_ids.reserve(prompt_ids.size() + response_ids.size());
    flat_response_mask.reserve(prompt_ids.size() + response_ids.size());

    for (int id : prompt_ids) {
        flat_ids.push_back(static_cast<int32_t>(id));
        flat_response_mask.push_back(0);
    }
    for (int id : response_ids) {
        flat_ids.push_back(static_cast<int32_t>(id));
        flat_response_mask.push_back(1);
    }

    if (flat_ids.size() > static_cast<size_t>(cfg_.seq_len)) {
        flat_ids.resize(cfg_.seq_len);
        flat_response_mask.resize(cfg_.seq_len);
    }

    int response_count = 0;
    for (int32_t v : flat_response_mask) response_count += (v != 0);
    if (response_count == 0) return false;

    const int32_t pad_id = static_cast<int32_t>(tok_->get_pad_token_id());
    input_ids.assign(cfg_.seq_len, pad_id);
    attention_mask.assign(cfg_.seq_len, 0);
    response_mask.assign(cfg_.seq_len, 0);

    for (size_t i = 0; i < flat_ids.size(); ++i) {
        input_ids[i] = flat_ids[i];
        attention_mask[i] = 1;
        response_mask[i] = flat_response_mask[i];
    }
    return true;
}

void PreferenceDataset::load(PreferenceSplit split) {
    samples_.clear();
    order_.clear();
    kept_samples_ = 0;
    skipped_samples_ = 0;
    identical_after_truncation_ = 0;
    cursor_ = 0;

    const std::string path = path_for_split(split);
    if (path.empty()) {
        std::cout << "[PreferenceDataset] Empty path for split\n";
        return;
    }

    std::ifstream fin(path);
    if (!fin.is_open()) {
        throw std::runtime_error("PreferenceDataset: cannot open " + path);
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        JsonRow row;
        if (!parse_jsonl_row(line, row)) {
            skipped_samples_++;
            continue;
        }

        PreferenceEncodedSample sample;
        bool chosen_ok = encode_pair(
            row.prompt, row.chosen,
            sample.chosen_input_ids,
            sample.chosen_attention_mask,
            sample.chosen_response_mask);
        bool rejected_ok = encode_pair(
            row.prompt, row.rejected,
            sample.rejected_input_ids,
            sample.rejected_attention_mask,
            sample.rejected_response_mask);

        if (!chosen_ok || !rejected_ok) {
            skipped_samples_++;
            continue;
        }

        if (sample.chosen_input_ids == sample.rejected_input_ids &&
            sample.chosen_attention_mask == sample.rejected_attention_mask &&
            sample.chosen_response_mask == sample.rejected_response_mask) {
            identical_after_truncation_++;
            skipped_samples_++;
            continue;
        }

        samples_.push_back(std::move(sample));
        kept_samples_++;
    }

    order_.resize(samples_.size());
    for (size_t i = 0; i < samples_.size(); ++i) order_[i] = i;
}

size_t PreferenceDataset::num_batches() const {
    return (samples_.size() + cfg_.batch_size - 1) / cfg_.batch_size;
}

void PreferenceDataset::shuffle() {
    std::shuffle(order_.begin(), order_.end(), rng_);
    cursor_ = 0;
}

PreferenceBatch PreferenceDataset::get_batch(size_t batch_idx) const {
    PreferenceBatch batch;

    const size_t start = batch_idx * cfg_.batch_size;
    const size_t end = std::min(start + cfg_.batch_size, samples_.size());
    const size_t actual_batch_size = end - start;
    if (actual_batch_size == 0) return batch;

    const std::vector<int64_t> shape = {
        static_cast<int64_t>(actual_batch_size),
        static_cast<int64_t>(cfg_.seq_len),
    };

    batch.chosen_input_ids = std::make_shared<Tensor>(shape, kInt32, kCPU);
    batch.chosen_attention_mask = std::make_shared<Tensor>(shape, kFloat32, kCPU);
    batch.chosen_response_mask = std::make_shared<Tensor>(shape, kInt32, kCPU);

    batch.rejected_input_ids = std::make_shared<Tensor>(shape, kInt32, kCPU);
    batch.rejected_attention_mask = std::make_shared<Tensor>(shape, kFloat32, kCPU);
    batch.rejected_response_mask = std::make_shared<Tensor>(shape, kInt32, kCPU);

    int32_t* c_ids = batch.chosen_input_ids->data<int32_t>();
    float* c_attn = batch.chosen_attention_mask->data<float>();
    int32_t* c_resp = batch.chosen_response_mask->data<int32_t>();
    int32_t* r_ids = batch.rejected_input_ids->data<int32_t>();
    float* r_attn = batch.rejected_attention_mask->data<float>();
    int32_t* r_resp = batch.rejected_response_mask->data<int32_t>();

    std::memset(c_ids, 0, actual_batch_size * cfg_.seq_len * sizeof(int32_t));
    std::memset(c_attn, 0, actual_batch_size * cfg_.seq_len * sizeof(float));
    std::memset(c_resp, 0, actual_batch_size * cfg_.seq_len * sizeof(int32_t));
    std::memset(r_ids, 0, actual_batch_size * cfg_.seq_len * sizeof(int32_t));
    std::memset(r_attn, 0, actual_batch_size * cfg_.seq_len * sizeof(float));
    std::memset(r_resp, 0, actual_batch_size * cfg_.seq_len * sizeof(int32_t));

    for (size_t b = 0; b < actual_batch_size; ++b) {
        const auto& sample = samples_[order_[start + b]];
        for (int i = 0; i < cfg_.seq_len; ++i) {
            c_ids[b * cfg_.seq_len + i] = sample.chosen_input_ids[i];
            c_attn[b * cfg_.seq_len + i] = static_cast<float>(sample.chosen_attention_mask[i]);
            c_resp[b * cfg_.seq_len + i] = sample.chosen_response_mask[i];

            r_ids[b * cfg_.seq_len + i] = sample.rejected_input_ids[i];
            r_attn[b * cfg_.seq_len + i] = static_cast<float>(sample.rejected_attention_mask[i]);
            r_resp[b * cfg_.seq_len + i] = sample.rejected_response_mask[i];
        }
    }

    batch.num_samples = actual_batch_size;
    return batch;
}

PreferenceBatch PreferenceDataset::next_batch(bool loop) {
    if (cursor_ >= num_batches()) {
        if (loop) cursor_ = 0;
        else return PreferenceBatch{};
    }
    return get_batch(cursor_++);
}

void PreferenceDataset::print_stats() const {
    std::cout << "[PreferenceDataset] samples=" << samples_.size()
              << " kept=" << kept_samples_
              << " skipped=" << skipped_samples_
              << " identical_after_truncation=" << identical_after_truncation_
              << " batch_size=" << cfg_.batch_size
              << " seq_len=" << cfg_.seq_len
              << std::endl;
}

}  // namespace ops
