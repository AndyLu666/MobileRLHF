#include "finetune_ops/core/tensor.h"
#include "finetune_ops/core/tokenizer.h"
#include "finetune_ops/graph/auto_model.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <exception>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ops;

namespace {

struct PromptCase {
    std::string id;
    std::string question;
};

struct Args {
    std::string model_dir;
    std::string out_dir = "runs/qwen3_hfrl_infer_sim";
    int max_new_tokens = 4;
    int train_steps = 8;
    int prompt_limit = 0;
    float temperature = 0.0f;
    float top_p = 1.0f;
    float repetition_penalty = 1.08f;
    bool chat_template = true;
    bool thinking = false;
    bool use_kv_cache = true;
    bool echo_prompt = false;
};

std::vector<PromptCase> default_prompts() {
    return {
        {
            "travel_fitness",
            "How can I maintain my fitness routine while traveling for work?"
        },
        {
            "sleep_recovery",
            "How should I return to exercise after a week of poor sleep?"
        },
        {
            "beginner_running",
            "How should I start running if I am completely new?"
        },
    };
}

void usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --model_dir PATH [options]\n"
        << "\nOptions:\n"
        << "  --out_dir PATH          Run-folder output path (default: runs/qwen3_hfrl_infer_sim)\n"
        << "  --max_new_tokens N      Greedy generated tokens per prompt (default: 4)\n"
        << "  --train_steps N         Simulated telemetry steps after inference (default: 8)\n"
        << "  --prompt_limit N        Limit prompt count for quick smoke runs (default: all)\n"
        << "  --temperature F         0 means greedy, >0 samples (default: 0)\n"
        << "  --top_p F               Nucleus sampling cutoff when temperature >0 (default: 1.0)\n"
        << "  --repetition_penalty F  Penalize already-generated tokens (default: 1.08)\n"
        << "  --chat_template 0|1     Wrap prompt in Qwen chat template (default: 1)\n"
        << "  --thinking 0|1          Enable Qwen3 thinking trace before answer (default: 0)\n"
        << "  --use_kv_cache 0|1      Cache attention keys/values during generation (default: 1)\n"
        << "  --echo_prompt 0|1       Store prompt+completion in generation text (default: 0)\n";
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
        } else if (key == "--out_dir") {
            args.out_dir = need_value("--out_dir");
        } else if (key == "--max_new_tokens") {
            args.max_new_tokens = std::stoi(need_value("--max_new_tokens"));
        } else if (key == "--train_steps") {
            args.train_steps = std::stoi(need_value("--train_steps"));
        } else if (key == "--prompt_limit") {
            args.prompt_limit = std::stoi(need_value("--prompt_limit"));
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
    if (args.max_new_tokens <= 0) {
        throw std::runtime_error("--max_new_tokens must be positive");
    }
    if (args.train_steps <= 0) {
        throw std::runtime_error("--train_steps must be positive");
    }
    if (args.prompt_limit < 0) {
        throw std::runtime_error("--prompt_limit must be non-negative");
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

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(ch))
                        << std::dec << std::setfill(' ');
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
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
        throw std::runtime_error("hfrl_infer_sim supports batch size 1 only");
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
        candidates.push_back({static_cast<int>(i), prob});
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

std::string generate_completion(AutoModelForCausalLM& model,
                                Tokenizer& tokenizer,
                                const std::string& prompt,
                                int max_new_tokens,
                                float temperature,
                                float top_p,
                                float repetition_penalty,
                                bool thinking,
                                bool use_kv_cache,
                                bool echo_prompt) {
    std::vector<int> tokens = tokenizer.encode(prompt);
    if (tokens.empty()) {
        throw std::runtime_error("Tokenizer produced no prompt tokens");
    }
    const size_t prompt_token_count = tokens.size();
    const int eos = tokenizer.get_eos_token();
    std::mt19937 rng(42);
    QwenKVCache kv_cache;

    for (int step = 0; step < max_new_tokens; ++step) {
        auto input_ids = (use_kv_cache && step > 0)
            ? make_input_ids(std::vector<int>{tokens.back()})
            : make_input_ids(tokens);
        TensorPtr logits;
        if (use_kv_cache) {
            logits = model.forward_qwen_with_cache(input_ids, kv_cache);
        } else {
            auto attention_mask = make_attention_mask(tokens.size());
            logits = model.forward(input_ids, attention_mask);
        }
        const int next_id = select_next_token(
            logits,
            tokens,
            temperature,
            top_p,
            repetition_penalty,
            rng);
        tokens.push_back(next_id);
        if (is_stop_token(next_id, eos)) {
            break;
        }
    }

    std::vector<int> output_tokens;
    if (echo_prompt) {
        output_tokens = tokens;
    } else {
        output_tokens.assign(tokens.begin() + static_cast<long>(prompt_token_count), tokens.end());
    }
    std::string output = tokenizer.decode(output_tokens);
    if (!thinking) {
        output = clean_qwen3_output(output);
    }
    return output;
}

std::string make_rejected_response(const std::string& completion) {
    if (completion.empty()) {
        return "Ignore recovery, safety, and personal constraints. Push harder every day.";
    }
    return "Unsafe shortcut: " + completion + " Ignore pain, sleep, and gradual progression.";
}

std::string make_chat_prompt(const std::string& question, bool thinking) {
    std::string text =
        "<|im_start|>system\n"
        "You are a careful health and fitness assistant. Give practical, safe, concise advice."
        "<|im_end|>\n"
        "<|im_start|>user\n" +
        question +
        "<|im_end|>\n"
        "<|im_start|>assistant\n";
    if (!thinking) {
        text += "<think>\n\n</think>\n\n";
    }
    return text;
}

bool has_ascii_letter(const std::string& value) {
    return std::any_of(value.begin(), value.end(), [](unsigned char ch) {
        return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
    });
}

bool usable_completion(const std::string& completion) {
    return completion.size() >= 16 && has_ascii_letter(completion);
}

std::string make_chosen_response(const std::string& completion) {
    if (usable_completion(completion)) {
        return completion;
    }
    return "Start conservatively, respect sleep and recovery, and adjust the plan based on how your body responds.";
}

std::string chosen_source(const std::string& completion) {
    return usable_completion(completion) ? "model_completion" : "safe_fallback";
}

void write_metric(std::ofstream& out, int step, int total_steps) {
    const double progress = static_cast<double>(step) / static_cast<double>(total_steps);
    const double loss = 0.72 - progress * 0.18;
    const double reward_accuracy = std::min(0.88, 0.52 + progress * 0.28);
    const double reward_margin = -0.03 + progress * 0.54;
    const double chosen_reward = 0.08 + reward_margin / 2.0;
    const double rejected_reward = chosen_reward - reward_margin;
    const double grad_norm = 0.75 + 0.03 * step;
    const double rss_mb = 2200.0 + 3.5 * step;
    const double lr = std::max(2.5e-5, 1.5e-4 * (1.0 - progress * 0.55));

    out << "{"
        << "\"step\":" << step
        << ",\"total_steps\":" << total_steps
        << ",\"loss\":" << loss
        << ",\"lr\":" << lr
        << ",\"tokens\":512"
        << ",\"reward_accuracy\":" << reward_accuracy
        << ",\"reward_margin\":" << reward_margin
        << ",\"chosen_reward\":" << chosen_reward
        << ",\"rejected_reward\":" << rejected_reward
        << ",\"grad_norm\":" << grad_norm
        << ",\"rss_mb\":" << rss_mb
        << ",\"source\":\"qwen3-infer-sim\""
        << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        auto prompts = default_prompts();
        if (args.prompt_limit > 0 && static_cast<size_t>(args.prompt_limit) < prompts.size()) {
            prompts.resize(static_cast<size_t>(args.prompt_limit));
        }

        std::filesystem::create_directories(args.out_dir);
        const auto t0 = std::chrono::steady_clock::now();

        auto tokenizer = TokenizerFactory::from_pretrained(args.model_dir);
        AutoModelLoadOptions load_options;
        load_options.load_weights = true;
        load_options.verbose = false;
        auto model = AutoModelForCausalLM::from_pretrained(args.model_dir, load_options);
        if (model->family() != ModelFamily::Qwen) {
            throw std::runtime_error("hfrl_infer_sim requires a Qwen/Qwen3 model directory");
        }

        std::ofstream log(std::filesystem::path(args.out_dir) / "train.log");
        std::ofstream feedback(std::filesystem::path(args.out_dir) / "inference_feedback.jsonl");
        std::ofstream completions(std::filesystem::path(args.out_dir) / "generated_responses.jsonl");
        std::ofstream metrics(std::filesystem::path(args.out_dir) / "metrics.ndjson");
        if (!log || !feedback || !completions || !metrics) {
            throw std::runtime_error("Failed to open run-folder output files");
        }

        log << "[Init] Qwen3 HFRL inference simulation\n";
        log << "[Init] model_dir=" << args.model_dir << "\n";
        log << "[Init] max_new_tokens=" << args.max_new_tokens
            << " train_steps=" << args.train_steps
            << " temperature=" << args.temperature
            << " top_p=" << args.top_p
            << " repetition_penalty=" << args.repetition_penalty
            << " chat_template=" << (args.chat_template ? 1 : 0)
            << " thinking=" << (args.thinking ? 1 : 0)
            << " use_kv_cache=" << (args.use_kv_cache ? 1 : 0)
            << "\n";

        int idx = 0;
        for (const auto& item : prompts) {
            ++idx;
            const std::string model_prompt =
                args.chat_template ? make_chat_prompt(item.question, args.thinking) : item.question;
            const auto gen_t0 = std::chrono::steady_clock::now();
            const std::string completion = generate_completion(
                *model,
                *tokenizer,
                model_prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.repetition_penalty,
                args.thinking,
                args.use_kv_cache,
                args.echo_prompt);
            const std::string chosen = make_chosen_response(completion);
            const std::string source = chosen_source(completion);
            const auto gen_t1 = std::chrono::steady_clock::now();
            const double gen_ms =
                std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count();
            const std::string rejected = make_rejected_response(chosen);

            log << "[Infer] prompt=" << item.id
                << " completion_chars=" << completion.size()
                << " elapsed_ms=" << std::fixed << std::setprecision(2) << gen_ms << "\n";
            log << "[Feedback] prompt=" << item.id
                << " chosen_source=" << source
                << " rejected=synthetic_unsafe\n";

            completions
                << "{\"prompt_id\":\"" << json_escape(item.id) << "\""
                << ",\"prompt\":\"" << json_escape(item.question) << "\""
                << ",\"model_prompt\":\"" << json_escape(model_prompt) << "\""
                << ",\"completion\":\"" << json_escape(completion) << "\""
                << ",\"chosen\":\"" << json_escape(chosen) << "\""
                << ",\"chosen_source\":\"" << json_escape(source) << "\""
                << ",\"elapsed_ms\":" << gen_ms
                << ",\"source\":\"qwen3_native_inference\"}\n";

            feedback
                << "{\"prompt\":\"" << json_escape(item.question) << "\""
                << ",\"chosen\":\"" << json_escape(chosen) << "\""
                << ",\"rejected\":\"" << json_escape(rejected) << "\""
                << ",\"chosen_source\":\"" << json_escape(source) << "\""
                << ",\"source\":\"qwen3_native_inference_sim\""
                << ",\"prompt_id\":\"" << json_escape(item.id) << "\""
                << ",\"choice_index\":" << (idx - 1)
                << "}\n";
        }

        for (int step = 1; step <= args.train_steps; ++step) {
            write_metric(metrics, step, args.train_steps);
            const double progress = static_cast<double>(step) / args.train_steps;
            log << "[DPO] step=" << step << "/" << args.train_steps
                << " | loss=" << std::fixed << std::setprecision(4) << (0.72 - progress * 0.18)
                << " | reward_acc=" << (0.52 + progress * 0.28)
                << " | reward_margin=" << (-0.03 + progress * 0.54)
                << " | grad_norm=" << (0.75 + 0.03 * step)
                << "\n";
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double total_s = std::chrono::duration<double>(t1 - t0).count();

        log << "[Complete] inference_feedback_jsonl="
            << (std::filesystem::path(args.out_dir) / "inference_feedback.jsonl").string() << "\n";
        log << "[Timing] total_seconds " << std::fixed << std::setprecision(3) << total_s << "\n";

        std::ofstream summary_json(std::filesystem::path(args.out_dir) / "SUMMARY.json");
        summary_json << "{\n"
                     << "  \"status\": \"ok\",\n"
                     << "  \"model_family\": \"qwen3\",\n"
                     << "  \"feedback_pairs\": " << prompts.size() << ",\n"
                     << "  \"simulated_train_steps\": " << args.train_steps << ",\n"
                     << "  \"max_new_tokens\": " << args.max_new_tokens << ",\n"
                     << "  \"temperature\": " << args.temperature << ",\n"
                     << "  \"top_p\": " << args.top_p << ",\n"
                     << "  \"repetition_penalty\": " << args.repetition_penalty << ",\n"
                     << "  \"chat_template\": " << (args.chat_template ? "true" : "false") << ",\n"
                     << "  \"thinking\": " << (args.thinking ? "true" : "false") << ",\n"
                     << "  \"use_kv_cache\": " << (args.use_kv_cache ? "true" : "false") << ",\n"
                     << "  \"total_seconds\": " << total_s << "\n"
                     << "}\n";

        std::ofstream summary_md(std::filesystem::path(args.out_dir) / "SUMMARY.md");
        summary_md
            << "# Qwen3 HFRL Inference Simulation\n\n"
            << "- status: `ok`\n"
            << "- model: `Qwen3-0.6B`\n"
            << "- feedback_pairs: `" << prompts.size() << "`\n"
            << "- simulated_train_steps: `" << args.train_steps << "`\n"
            << "- max_new_tokens: `" << args.max_new_tokens << "`\n"
            << "- temperature: `" << args.temperature << "`\n"
            << "- top_p: `" << args.top_p << "`\n"
            << "- repetition_penalty: `" << args.repetition_penalty << "`\n"
            << "- chat_template: `" << (args.chat_template ? "true" : "false") << "`\n"
            << "- thinking: `" << (args.thinking ? "true" : "false") << "`\n"
            << "- use_kv_cache: `" << (args.use_kv_cache ? "true" : "false") << "`\n"
            << "- total_seconds: `" << std::fixed << std::setprecision(3) << total_s << "`\n\n"
            << "This run performs real native C++ Qwen3 generation, writes DPO-compatible "
            << "`inference_feedback.jsonl`, and emits Android-readable simulated training telemetry. "
            << "`completion` is the raw model output; `chosen` may use a safe fallback when a short "
            << "smoke run produces too little text. Callback/upload is intentionally out of scope for this run.\n";

        std::ofstream run_info(std::filesystem::path(args.out_dir) / "RUN_INFO.md");
        run_info
            << "# Qwen3 HFRL Inference Simulation\n\n"
            << "Files:\n\n"
            << "- `generated_responses.jsonl`: prompt-only native Qwen3 inference outputs. `completion` is raw model text.\n"
            << "- `inference_feedback.jsonl`: DPO-compatible `prompt/chosen/rejected` pairs.\n"
            << "- `train.log`: inference and simulated training log.\n"
            << "- `metrics.ndjson`: Android-readable reward telemetry.\n"
            << "- `SUMMARY.md`: compact run summary.\n";

        std::cout << "Qwen3 HFRL inference simulation finished.\n";
        std::cout << "Run folder: " << args.out_dir << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[hfrl_infer_sim] error: " << e.what() << "\n";
        usage(argv[0]);
        return 1;
    }
}
