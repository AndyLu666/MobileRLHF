#include "finetune_ops/core/tensor.h"
#include "finetune_ops/core/tokenizer.h"
#include "finetune_ops/graph/auto_model.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <exception>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ops;

namespace {

struct Args {
    std::string model_dir;
    std::string prompt;
    int max_new_tokens = 64;
    float temperature = 0.0f;
    float top_p = 1.0f;
    float repetition_penalty = 1.08f;
    bool chat_template = true;
    bool thinking = false;
    bool use_kv_cache = true;
    bool echo_prompt = false;
};

void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --model_dir PATH --prompt TEXT [options]\n"
        << "\nOptions:\n"
        << "  --max_new_tokens N        Number of tokens to generate (default: 64)\n"
        << "  --temperature F           0 means greedy, >0 samples (default: 0)\n"
        << "  --top_p F                 Nucleus sampling cutoff when temperature >0 (default: 1.0)\n"
        << "  --repetition_penalty F    Penalize already-generated tokens (default: 1.08)\n"
        << "  --chat_template 0|1       Wrap prompt in Qwen chat template (default: 1)\n"
        << "  --thinking 0|1            Enable Qwen3 thinking trace before answer (default: 0)\n"
        << "  --use_kv_cache 0|1        Cache attention keys/values during generation (default: 1)\n"
        << "  --echo_prompt 0|1         Decode prompt + completion instead of completion only\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (key == "--model_dir") {
            args.model_dir = need_value("--model_dir");
        } else if (key == "--prompt") {
            args.prompt = need_value("--prompt");
        } else if (key == "--max_new_tokens") {
            args.max_new_tokens = std::stoi(need_value("--max_new_tokens"));
        } else if (key == "--temperature") {
            args.temperature = std::stof(need_value("--temperature"));
        } else if (key == "--top_p") {
            args.top_p = std::stof(need_value("--top_p"));
        } else if (key == "--repetition_penalty") {
            args.repetition_penalty = std::stof(need_value("--repetition_penalty"));
        } else if (key == "--chat_template") {
            args.chat_template = std::stoi(need_value("--chat_template")) != 0;
        } else if (key == "--thinking") {
            args.thinking = std::stoi(need_value("--thinking")) != 0;
        } else if (key == "--use_kv_cache") {
            args.use_kv_cache = std::stoi(need_value("--use_kv_cache")) != 0;
        } else if (key == "--echo_prompt") {
            args.echo_prompt = std::stoi(need_value("--echo_prompt")) != 0;
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }

    if (args.model_dir.empty()) {
        throw std::runtime_error("--model_dir is required");
    }
    if (args.prompt.empty()) {
        throw std::runtime_error("--prompt is required");
    }
    if (args.max_new_tokens <= 0) {
        throw std::runtime_error("--max_new_tokens must be positive");
    }
    if (args.temperature < 0.0f) {
        throw std::runtime_error("--temperature must be non-negative");
    }
    if (args.top_p <= 0.0f || args.top_p > 1.0f) {
        throw std::runtime_error("--top_p must be in (0, 1]");
    }
    if (args.repetition_penalty < 1.0f) {
        throw std::runtime_error("--repetition_penalty must be >= 1");
    }
    return args;
}

std::string make_chat_prompt(const std::string& prompt, bool thinking) {
    std::string text =
        "<|im_start|>system\n"
        "You are a careful health and fitness assistant. Give practical, safe, concise advice."
        "<|im_end|>\n"
        "<|im_start|>user\n" +
        prompt +
        "<|im_end|>\n"
        "<|im_start|>assistant\n";
    if (!thinking) {
        text += "<think>\n\n</think>\n\n";
    }
    return text;
}

TensorPtr make_input_ids(const std::vector<int>& ids) {
    std::vector<int32_t> values(ids.begin(), ids.end());
    return std::make_shared<Tensor>(
        std::vector<int64_t>{1, static_cast<int64_t>(values.size())},
        values.data(),
        kInt32,
        kCPU);
}

TensorPtr make_attention_mask(size_t length) {
    std::vector<float> values(length, 1.0f);
    return std::make_shared<Tensor>(
        std::vector<int64_t>{1, static_cast<int64_t>(values.size())},
        values.data(),
        kFloat32,
        kCPU);
}

int select_next_token(const TensorPtr& logits,
                      const std::vector<int>& existing_tokens,
                      float temperature,
                      float top_p,
                      float repetition_penalty,
                      std::mt19937& rng) {
    if (!logits || logits->ndim() != 3) {
        throw std::runtime_error("Expected logits with shape [batch, seq, vocab]");
    }
    if (logits->size(0) != 1) {
        throw std::runtime_error("infer_qwen currently supports batch size 1 only");
    }
    const int64_t seq_len = logits->size(1);
    const int64_t vocab = logits->size(2);
    if (seq_len <= 0 || vocab <= 0) {
        throw std::runtime_error("Invalid logits shape");
    }

    const float* data = logits->data<float>();
    const int64_t offset = (seq_len - 1) * vocab;
    std::vector<float> scores(static_cast<size_t>(vocab));
    for (int64_t i = 0; i < vocab; ++i) {
        scores[static_cast<size_t>(i)] = data[offset + i];
    }

    if (repetition_penalty > 1.0f) {
        for (int id : existing_tokens) {
            if (id < 0 || id >= vocab) continue;
            float& value = scores[static_cast<size_t>(id)];
            value = value >= 0.0f ? value / repetition_penalty : value * repetition_penalty;
        }
    }

    if (temperature <= 0.0f) {
        int best_id = 0;
        float best = scores[0];
        for (int64_t i = 1; i < vocab; ++i) {
            const float value = scores[static_cast<size_t>(i)];
            if (value > best) {
                best = value;
                best_id = static_cast<int>(i);
            }
        }
        return best_id;
    }

    struct Candidate {
        int id;
        float score;
        float prob;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(vocab));
    float max_score = scores[0];
    for (int64_t i = 1; i < vocab; ++i) {
        max_score = std::max(max_score, scores[static_cast<size_t>(i)]);
    }
    double denom = 0.0;
    for (int64_t i = 0; i < vocab; ++i) {
        const float scaled = (scores[static_cast<size_t>(i)] - max_score) / temperature;
        const float prob = std::exp(scaled);
        candidates.push_back({static_cast<int>(i), scores[static_cast<size_t>(i)], prob});
        denom += prob;
    }
    for (auto& c : candidates) {
        c.prob = static_cast<float>(static_cast<double>(c.prob) / denom);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.prob > b.prob;
    });

    double cumulative = 0.0;
    size_t keep = 0;
    for (; keep < candidates.size(); ++keep) {
        cumulative += candidates[keep].prob;
        if (cumulative >= top_p) {
            ++keep;
            break;
        }
    }
    keep = std::max<size_t>(1, std::min(keep, candidates.size()));
    std::vector<double> weights;
    weights.reserve(keep);
    for (size_t i = 0; i < keep; ++i) {
        weights.push_back(candidates[i].prob);
    }
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    return candidates[dist(rng)].id;
}

bool is_stop_token(int id, int eos) {
    if (eos >= 0 && id == eos) return true;
    return id == 151643 || id == 151645;
}

std::string clean_qwen3_output(std::string text) {
    const std::string begin = "<think>";
    const std::string end = "</think>";
    while (true) {
        const size_t start = text.find(begin);
        if (start == std::string::npos) break;
        const size_t finish = text.find(end, start + begin.size());
        if (finish == std::string::npos) {
            text.erase(start, begin.size());
            continue;
        }
        text.erase(start, finish + end.size() - start);
    }
    size_t left = 0;
    while (left < text.size() && std::isspace(static_cast<unsigned char>(text[left]))) {
        ++left;
    }
    size_t right = text.size();
    while (right > left && std::isspace(static_cast<unsigned char>(text[right - 1]))) {
        --right;
    }
    return text.substr(left, right - left);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);

        auto tokenizer = TokenizerFactory::from_pretrained(args.model_dir);

        AutoModelLoadOptions load_options;
        load_options.load_weights = true;
        load_options.verbose = false;
        auto model = AutoModelForCausalLM::from_pretrained(args.model_dir, load_options);
        if (model->family() != ModelFamily::Qwen) {
            throw std::runtime_error("infer_qwen requires a Qwen/Qwen2 model directory");
        }

        const std::string model_prompt =
            args.chat_template ? make_chat_prompt(args.prompt, args.thinking) : args.prompt;
        std::vector<int> tokens = tokenizer->encode(model_prompt);
        if (tokens.empty()) {
            throw std::runtime_error("Tokenizer produced no prompt tokens");
        }
        const size_t prompt_token_count = tokens.size();
        const int eos = tokenizer->get_eos_token();
        std::mt19937 rng(42);

        QwenKVCache kv_cache;
        for (int step = 0; step < args.max_new_tokens; ++step) {
            auto input_ids = (args.use_kv_cache && step > 0)
                ? make_input_ids(std::vector<int>{tokens.back()})
                : make_input_ids(tokens);
            TensorPtr logits;
            if (args.use_kv_cache) {
                logits = model->forward_qwen_with_cache(input_ids, kv_cache);
            } else {
                auto attention_mask = make_attention_mask(tokens.size());
                logits = model->forward(input_ids, attention_mask);
            }
            const int next_id = select_next_token(
                logits,
                tokens,
                args.temperature,
                args.top_p,
                args.repetition_penalty,
                rng);
            tokens.push_back(next_id);
            if (is_stop_token(next_id, eos)) {
                break;
            }
        }

        std::vector<int> output_tokens;
        if (args.echo_prompt) {
            output_tokens = tokens;
        } else {
            output_tokens.assign(tokens.begin() + static_cast<long>(prompt_token_count), tokens.end());
        }

        std::string output = tokenizer->decode(output_tokens);
        if (!args.thinking) {
            output = clean_qwen3_output(output);
        }
        std::cout << output << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[infer_qwen] error: " << e.what() << std::endl;
        usage(argv[0]);
        return 1;
    }
}
