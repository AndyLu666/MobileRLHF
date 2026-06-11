#include "finetune_ops/core/tensor.h"
#include "finetune_ops/core/tokenizer.h"
#include "finetune_ops/graph/auto_model.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ops;

namespace {

struct Args {
    std::string model_dir;
    std::string prompt;
    int max_new_tokens = 64;
    bool echo_prompt = false;
};

void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --model_dir PATH --prompt TEXT [options]\n"
        << "\nOptions:\n"
        << "  --max_new_tokens N   Number of tokens to generate (default: 64)\n"
        << "  --echo_prompt 0|1    Decode prompt + completion instead of completion only\n";
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
    return args;
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

int argmax_last_token(const TensorPtr& logits) {
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
    int best_id = 0;
    float best = data[offset];
    for (int64_t i = 1; i < vocab; ++i) {
        const float value = data[offset + i];
        if (value > best) {
            best = value;
            best_id = static_cast<int>(i);
        }
    }
    return best_id;
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

        std::vector<int> tokens = tokenizer->encode(args.prompt);
        if (tokens.empty()) {
            throw std::runtime_error("Tokenizer produced no prompt tokens");
        }
        const size_t prompt_token_count = tokens.size();
        const int eos = tokenizer->get_eos_token();

        for (int step = 0; step < args.max_new_tokens; ++step) {
            auto input_ids = make_input_ids(tokens);
            auto attention_mask = make_attention_mask(tokens.size());
            auto logits = model->forward(input_ids, attention_mask);
            const int next_id = argmax_last_token(logits);
            tokens.push_back(next_id);
            if (eos >= 0 && next_id == eos) {
                break;
            }
        }

        std::vector<int> output_tokens;
        if (args.echo_prompt) {
            output_tokens = tokens;
        } else {
            output_tokens.assign(tokens.begin() + static_cast<long>(prompt_token_count), tokens.end());
        }

        std::cout << tokenizer->decode(output_tokens) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[infer_qwen] error: " << e.what() << std::endl;
        usage(argv[0]);
        return 1;
    }
}
