#include "auto_model.h"

#include "../core/memory_manager.h"
#include "model_family_adapter.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace ops {

namespace {

FamilyLoraRequest make_family_lora_request(const AutoLoraConfig& cfg) {
    FamilyLoraRequest request;
    request.rank = cfg.rank;
    request.alpha = cfg.alpha;
    request.dropout = cfg.dropout;
    request.seed = cfg.seed;
    request.target_modules = cfg.target_modules;
    return request;
}

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char ch : value) {
        if (ch == '\\' || ch == '"') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    return out;
}

std::string shape_to_json(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

std::string tensor_data_to_json(const TensorPtr& tensor) {
    if (!tensor || tensor->dtype() != kFloat32) {
        throw std::runtime_error("LoRA adapter JSONL only supports float32 tensors");
    }
    std::ostringstream oss;
    oss << "[";
    const float* data = tensor->data<float>();
    for (int64_t i = 0; i < tensor->numel(); ++i) {
        if (i != 0) {
            oss << ",";
        }
        oss << data[i];
    }
    oss << "]";
    return oss.str();
}

std::string extract_json_string_field(const std::string& line, const std::string& key) {
    const std::string marker = "\"" + key + "\":\"";
    const std::size_t start = line.find(marker);
    if (start == std::string::npos) {
        return {};
    }
    std::string out;
    bool escaped = false;
    for (std::size_t i = start + marker.size(); i < line.size(); ++i) {
        const char ch = line[i];
        if (escaped) {
            out.push_back(ch);
            escaped = false;
        } else if (ch == '\\') {
            escaped = true;
        } else if (ch == '"') {
            return out;
        } else {
            out.push_back(ch);
        }
    }
    return {};
}

std::string extract_json_array_field(const std::string& line, const std::string& key) {
    const std::string marker = "\"" + key + "\":[";
    const std::size_t start = line.find(marker);
    if (start == std::string::npos) {
        return {};
    }
    const std::size_t value_start = start + marker.size();
    const std::size_t end = line.find(']', value_start);
    if (end == std::string::npos) {
        return {};
    }
    return line.substr(value_start, end - value_start);
}

std::vector<int64_t> parse_shape_list(const std::string& text) {
    std::vector<int64_t> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        values.push_back(std::stoll(item));
    }
    return values;
}

std::vector<float> parse_float_list(const std::string& text) {
    std::vector<float> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        const float value = std::stof(item);
        if (!std::isfinite(value)) {
            throw std::runtime_error("adapter tensor data contains a non-finite value");
        }
        values.push_back(value);
    }
    return values;
}

std::filesystem::path make_adapter_temp_path(const std::filesystem::path& target) {
    static std::atomic<std::uint64_t> serial{0};
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    return target.string() + ".tmp." + std::to_string(now) + "." +
           std::to_string(thread_id) + "." + std::to_string(serial.fetch_add(1));
}

class TemporaryAdapterFile {
public:
    explicit TemporaryAdapterFile(std::filesystem::path path) : path_(std::move(path)) {}
    ~TemporaryAdapterFile() {
        if (!committed_) {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
    }

    const std::filesystem::path& path() const { return path_; }
    void commit() { committed_ = true; }

private:
    std::filesystem::path path_;
    bool committed_ = false;
};

}  // namespace

AutoModelForCausalLM::AutoModelForCausalLM(std::string model_dir,
                                           ModelArchitectureSpec spec)
    : model_dir_(std::move(model_dir)), spec_(std::move(spec)) {}

std::unique_ptr<AutoModelForCausalLM>
AutoModelForCausalLM::from_pretrained(const std::string& model_dir,
                                      const AutoModelLoadOptions& options) {
    auto spec = ModelRegistry::inspect_pretrained(model_dir);
    auto model = std::unique_ptr<AutoModelForCausalLM>(
        new AutoModelForCausalLM(model_dir, spec));

    switch (model->spec_.family) {
        case ModelFamily::GPT2: {
            GPT2Config cfg = GPT2Config::from_pretrained(model_dir);
            model->gpt2_ = std::make_unique<GPT2Model>(cfg);
            if (cfg.tie_word_embeddings) {
                model->gpt2_->tie_weights();
            }
            break;
        }
        case ModelFamily::Gemma: {
            GemmaTextConfig cfg = GemmaTextConfig::from_pretrained(model_dir);
            model->gemma_ = std::make_unique<GemmaModel>(cfg);
            break;
        }
        case ModelFamily::Llama: {
            LlamaConfig cfg = LlamaConfig::from_pretrained(model_dir + "/config.json");
            model->llama_ = std::make_unique<LlamaModel>(cfg);
            break;
        }
        case ModelFamily::Mistral:
            throw std::runtime_error(
                "AutoModelForCausalLM: Mistral is recognized by ModelRegistry, "
                "but the Mistral graph/tokenizer gates have not passed yet");
        case ModelFamily::Qwen: {
            if (!model->spec_.tie_word_embeddings) {
                throw std::runtime_error(
                    "AutoModelForCausalLM: Qwen checkpoints with tie_word_embeddings=false "
                    "are not supported yet; provide a tied Qwen checkpoint or add lm_head.weight support");
            }
            QwenConfig cfg = QwenConfig::from_pretrained(model_dir + "/config.json");
            model->qwen_ = std::make_unique<QwenModel>(cfg);
            break;
        }
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }

    if (options.load_weights) {
        model->load_pretrained_weights(options);
    }
    return model;
}

void AutoModelForCausalLM::load_pretrained_weights(const AutoModelLoadOptions& options) {
    SafeTensorsModelReader reader(model_dir_);
    reader.parse_headers();
    safetensors_load_report_.clear();

    const auto load_opts =
        load_options_for_family(spec_.family, options.safetensors_options, options.verbose);

    switch (spec_.family) {
        case ModelFamily::GPT2: {
            auto mapping = GPT2KeyMapper::generate_gpt2_mapping(gpt2_->config().n_layer);
            auto tensors = reader.load_tensors_mapped(mapping, load_opts, &safetensors_load_report_);
            for (const auto& kv : tensors) {
                gpt2_->assign_weight(kv.first, kv.second, load_opts.strict_shape_check);
            }
            break;
        }
        case ModelFamily::Gemma: {
            auto mapping = GemmaKeyMapper::generate_gemma_mapping(gemma_->config().num_hidden_layers);
            const auto tensor_names = reader.get_tensor_names();
            const bool has_lm_head =
                std::find(tensor_names.begin(), tensor_names.end(), "lm_head.weight") != tensor_names.end();
            if (!has_lm_head) {
                mapping.erase("lm_head.weight");
            }
            auto tensors = reader.load_tensors_mapped(mapping, load_opts, &safetensors_load_report_);
            for (const auto& kv : tensors) {
                gemma_->assign_weight(kv.first, kv.second, load_opts.strict_shape_check);
            }
            break;
        }
        case ModelFamily::Llama: {
            auto mapping = LlamaKeyMapper::generate_llama_mapping(
                llama_->config().num_hidden_layers,
                llama_->config().tie_word_embeddings,
                llama_->config().attention_bias);
            auto tensors = reader.load_tensors_mapped(mapping, load_opts, &safetensors_load_report_);
            for (const auto& kv : tensors) {
                llama_->assign_weight(kv.first, kv.second, load_opts.strict_shape_check);
            }
            MemoryManager::instance().clear_unused_memory();
            break;
        }
        case ModelFamily::Qwen: {
            auto mapping = QwenKeyMapper::generate_qwen_mapping(qwen_->config().num_hidden_layers);
            auto tensors = reader.load_tensors_mapped(mapping, load_opts, &safetensors_load_report_);
            for (const auto& kv : tensors) {
                qwen_->assign_weight(kv.first, kv.second, load_opts.strict_shape_check);
            }
            MemoryManager::instance().clear_unused_memory();
            break;
        }
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

TensorPtr AutoModelForCausalLM::forward(const TensorPtr& input_ids,
                                        const TensorPtr& attention_mask) {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->forward(input_ids, attention_mask);
        case ModelFamily::Gemma:
            return gemma_->forward(input_ids, attention_mask);
        case ModelFamily::Llama:
            return llama_->forward(input_ids, attention_mask);
        case ModelFamily::Qwen:
            return qwen_->forward(input_ids, attention_mask);
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

TensorPtr AutoModelForCausalLM::forward_qwen_with_cache(const TensorPtr& input_ids,
                                                        QwenKVCache& cache) {
    if (spec_.family != ModelFamily::Qwen || !qwen_) {
        throw std::runtime_error("AutoModelForCausalLM::forward_qwen_with_cache requires a Qwen model");
    }
    return qwen_->forward_with_cache(input_ids, cache);
}

TensorPtr AutoModelForCausalLM::forward_hidden(const TensorPtr& input_ids,
                                               const TensorPtr& attention_mask) {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->forward_hidden(input_ids, attention_mask);
        case ModelFamily::Gemma:
            return gemma_->forward_hidden(input_ids, attention_mask);
        case ModelFamily::Llama:
            return llama_->forward_hidden(input_ids, attention_mask);
        case ModelFamily::Qwen:
            return qwen_->forward_hidden(input_ids, attention_mask);
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

TensorPtr AutoModelForCausalLM::lm_head_weight_for_loss() const {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->lm_head_weight();
        case ModelFamily::Gemma:
            return gemma_->lm_head_weight_for_loss();
        case ModelFamily::Llama:
            return llama_->lm_head_weight_for_loss();
        case ModelFamily::Qwen:
            return qwen_->embedding_weight();
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

void AutoModelForCausalLM::init_lora(const AutoLoraConfig& config) {
    switch (spec_.family) {
        case ModelFamily::GPT2: {
            gpt2_->init_lora_modules();
            gpt2_lora_ = std::make_unique<LoraInjector>();
            gpt2_lora_->inject(*gpt2_, make_gpt2_lora_spec(make_family_lora_request(config)));
            break;
        }
        case ModelFamily::Gemma: {
            gemma_lora_ = std::make_unique<GemmaLoraInjector>();
            gemma_lora_->inject(*gemma_, make_gemma_lora_spec(make_family_lora_request(config)));
            break;
        }
        case ModelFamily::Llama:
            llama_->init_lora(config.rank, config.alpha, config.dropout,
                              is_qv_only_attention_request(make_family_lora_request(config)),
                              config.seed);
            llama_->freeze_base();
            break;
        case ModelFamily::Qwen:
            qwen_->init_lora(config.rank, config.alpha, config.dropout,
                             is_qv_only_attention_request(make_family_lora_request(config)),
                             config.seed);
            qwen_->freeze_base();
            break;
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

std::vector<TensorPtr> AutoModelForCausalLM::parameters() {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->parameters();
        case ModelFamily::Gemma:
            return gemma_->parameters();
        case ModelFamily::Llama:
            return llama_->parameters();
        case ModelFamily::Qwen:
            return qwen_->parameters();
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

std::vector<TensorPtr> AutoModelForCausalLM::trainable_parameters() {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->get_lora_parameters();
        case ModelFamily::Gemma:
            return gemma_->get_lora_parameters();
        case ModelFamily::Llama:
            return llama_->get_lora_parameters();
        case ModelFamily::Qwen:
            return qwen_->get_lora_parameters();
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

std::vector<std::pair<std::string, TensorPtr>> AutoModelForCausalLM::named_trainable_parameters() {
    switch (spec_.family) {
        case ModelFamily::GPT2:
            return gpt2_->named_lora_parameters();
        case ModelFamily::Gemma:
            return gemma_->named_lora_parameters();
        case ModelFamily::Llama:
            return llama_->named_lora_parameters();
        case ModelFamily::Qwen:
            return qwen_->named_lora_parameters();
        default:
            throw std::runtime_error("AutoModelForCausalLM: unsupported model family");
    }
}

void AutoModelForCausalLM::save_lora_adapter_jsonl(const std::string& path) {
    if (path.empty()) {
        throw std::invalid_argument("save_lora_adapter_jsonl: path must not be empty");
    }
    const auto tensors = named_trainable_parameters();
    if (tensors.empty()) {
        throw std::runtime_error("save_lora_adapter_jsonl: no LoRA trainable parameters to save");
    }

    // Validate the entire snapshot before touching the destination. This also
    // prevents a late unsupported dtype from leaving a partially written file.
    for (const auto& kv : tensors) {
        if (!kv.second || kv.second->dtype() != kFloat32) {
            throw std::runtime_error(
                "save_lora_adapter_jsonl: trainable parameter is not float32: " + kv.first);
        }
    }

    std::filesystem::path adapter_path(path);
    if (adapter_path.has_parent_path()) {
        std::filesystem::create_directories(adapter_path.parent_path());
    }
    TemporaryAdapterFile temp_file(make_adapter_temp_path(adapter_path));
    std::ofstream out(temp_file.path(), std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error(
            "save_lora_adapter_jsonl: failed to open temporary file for " + path);
    }
    out << "{\"format\":\"mobile_finetuner.lora_adapter_jsonl.v1\"}\n";
    for (const auto& kv : tensors) {
        out << "{\"name\":\"" << json_escape(kv.first)
            << "\",\"shape\":" << shape_to_json(kv.second->shape())
            << ",\"data\":" << tensor_data_to_json(kv.second)
            << "}\n";
    }
    out.flush();
    if (!out.good()) {
        throw std::runtime_error("save_lora_adapter_jsonl: failed while writing " + path);
    }
    out.close();

    // The temporary file lives beside the target, so POSIX rename is an
    // atomic replacement on the macOS/Linux/Android platforms we support.
    if (std::rename(temp_file.path().string().c_str(), adapter_path.string().c_str()) != 0) {
        throw std::runtime_error("save_lora_adapter_jsonl: failed to replace " + path);
    }
    temp_file.commit();
}

void AutoModelForCausalLM::load_lora_adapter_jsonl(const std::string& path) {
    if (path.empty()) {
        throw std::invalid_argument("load_lora_adapter_jsonl: path must not be empty");
    }
    auto tensors = named_trainable_parameters();
    if (tensors.empty()) {
        throw std::runtime_error("load_lora_adapter_jsonl: initialize LoRA before loading adapter");
    }
    std::unordered_map<std::string, TensorPtr> by_name;
    for (const auto& kv : tensors) {
        by_name.emplace(kv.first, kv.second);
    }

    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("load_lora_adapter_jsonl: failed to open " + path);
    }

    struct PendingTensor {
        TensorPtr target;
        std::vector<float> data;
    };
    std::unordered_map<std::string, PendingTensor> pending;
    pending.reserve(tensors.size());

    std::string line;
    bool saw_format = false;
    bool saw_tensor = false;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::string format = extract_json_string_field(line, "format");
        if (!format.empty()) {
            if (saw_format || saw_tensor ||
                format != "mobile_finetuner.lora_adapter_jsonl.v1") {
                throw std::runtime_error(
                    "load_lora_adapter_jsonl: invalid or misplaced format record");
            }
            saw_format = true;
            continue;
        }
        if (!saw_format) {
            throw std::runtime_error(
                "load_lora_adapter_jsonl: missing format record before tensor data");
        }
        saw_tensor = true;
        const std::string name = extract_json_string_field(line, "name");
        const auto shape = parse_shape_list(extract_json_array_field(line, "shape"));
        const auto data = parse_float_list(extract_json_array_field(line, "data"));
        if (name.empty() || shape.empty()) {
            throw std::runtime_error("load_lora_adapter_jsonl: malformed tensor record");
        }
        auto it = by_name.find(name);
        if (it == by_name.end()) {
            throw std::runtime_error("load_lora_adapter_jsonl: adapter tensor not present in model: " + name);
        }
        if (pending.find(name) != pending.end()) {
            throw std::runtime_error("load_lora_adapter_jsonl: duplicate tensor record: " + name);
        }
        TensorPtr tensor = it->second;
        if (tensor->dtype() != kFloat32) {
            throw std::runtime_error("load_lora_adapter_jsonl: target tensor is not float32: " + name);
        }
        if (tensor->shape() != shape) {
            throw std::runtime_error("load_lora_adapter_jsonl: shape mismatch for " + name);
        }
        if (static_cast<int64_t>(data.size()) != tensor->numel()) {
            throw std::runtime_error("load_lora_adapter_jsonl: data length mismatch for " + name);
        }
        pending.emplace(name, PendingTensor{tensor, data});
    }
    if (!in.eof() && in.fail()) {
        throw std::runtime_error("load_lora_adapter_jsonl: failed while reading " + path);
    }
    if (!saw_format) {
        throw std::runtime_error("load_lora_adapter_jsonl: adapter file is missing its format record");
    }
    if (pending.empty()) {
        throw std::runtime_error("load_lora_adapter_jsonl: adapter file did not contain any tensors");
    }
    if (pending.size() != by_name.size()) {
        std::string missing;
        for (const auto& kv : by_name) {
            if (pending.find(kv.first) == pending.end()) {
                if (!missing.empty()) {
                    missing += ", ";
                }
                missing += kv.first;
            }
        }
        throw std::runtime_error(
            "load_lora_adapter_jsonl: adapter is incomplete; missing tensors: " + missing);
    }

    // Commit only after every record has been parsed and validated. A corrupt
    // file therefore leaves all live LoRA tensors untouched.
    for (auto& kv : pending) {
        float* dst = kv.second.target->data<float>();
        std::copy(kv.second.data.begin(), kv.second.data.end(), dst);
    }
}

}  // namespace ops
