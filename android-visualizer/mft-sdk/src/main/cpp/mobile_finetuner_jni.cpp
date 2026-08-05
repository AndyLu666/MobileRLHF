#include <jni.h>

#include "mobile_finetuner/mobile_finetuner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct NativeSession {
    std::string model_dir;
    std::unique_ptr<ops::AutoModelForCausalLM> model;
    std::unique_ptr<ops::Tokenizer> tokenizer;
    std::unique_ptr<ops::AutoTrainer> trainer;
    std::unique_ptr<ops::DPOTrainer> dpo_trainer;
    std::unique_ptr<ops::KTOTrainer> kto_trainer;
};

void throw_java(JNIEnv* env, const char* class_name, const std::string& message) {
    jclass clazz = env->FindClass(class_name);
    if (clazz != nullptr) {
        env->ThrowNew(clazz, message.c_str());
    }
}

void throw_illegal_state(JNIEnv* env, const std::string& message) {
    throw_java(env, "java/lang/IllegalStateException", message);
}

void throw_illegal_argument(JNIEnv* env, const std::string& message) {
    throw_java(env, "java/lang/IllegalArgumentException", message);
}

std::string to_string(JNIEnv* env, jstring value) {
    if (value == nullptr) {
        return {};
    }
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return {};
    }
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

std::vector<std::string> to_string_vector(JNIEnv* env, jobjectArray values) {
    std::vector<std::string> out;
    if (values == nullptr) {
        return out;
    }

    const jsize length = env->GetArrayLength(values);
    out.reserve(static_cast<size_t>(length));
    for (jsize i = 0; i < length; ++i) {
        auto item = static_cast<jstring>(env->GetObjectArrayElement(values, i));
        if (env->ExceptionCheck()) {
            return {};
        }
        if (item != nullptr) {
            out.push_back(to_string(env, item));
            env->DeleteLocalRef(item);
        }
    }
    return out;
}

std::vector<std::string> copy_string_array(JNIEnv* env, jobjectArray values, const char* name) {
    if (values == nullptr) {
        throw std::invalid_argument(std::string(name) + " must not be null");
    }
    const jsize length = env->GetArrayLength(values);
    if (length <= 0) {
        throw std::invalid_argument(std::string(name) + " must contain at least one text");
    }
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(length));
    for (jsize i = 0; i < length; ++i) {
        auto item = static_cast<jstring>(env->GetObjectArrayElement(values, i));
        if (env->ExceptionCheck()) {
            return {};
        }
        if (item == nullptr) {
            throw std::invalid_argument(std::string(name) + " must not contain null entries");
        }
        out.push_back(to_string(env, item));
        env->DeleteLocalRef(item);
    }
    return out;
}

NativeSession* require_session(JNIEnv* env, jlong handle) {
    if (handle == 0) {
        throw_illegal_state(env, "MobileFineTuner native session is closed");
        return nullptr;
    }
    return reinterpret_cast<NativeSession*>(handle);
}

ops::Tokenizer& require_tokenizer(NativeSession& session) {
    if (!session.tokenizer) {
        if (session.model_dir.empty()) {
            throw std::runtime_error("Model directory is not available for tokenizer loading");
        }
        session.tokenizer = ops::TokenizerFactory::from_pretrained(session.model_dir);
    }
    return *session.tokenizer;
}

template <typename Body, typename Result>
Result guarded(JNIEnv* env, Body&& body, Result fallback) {
    try {
        return body();
    } catch (const std::invalid_argument& e) {
        throw_illegal_argument(env, e.what());
    } catch (const std::exception& e) {
        throw_illegal_state(env, e.what());
    } catch (...) {
        throw_illegal_state(env, "Unknown native MobileFineTuner error");
    }
    return fallback;
}

template <typename Body>
void guarded_void(JNIEnv* env, Body&& body) {
    try {
        body();
    } catch (const std::invalid_argument& e) {
        throw_illegal_argument(env, e.what());
    } catch (const std::exception& e) {
        throw_illegal_state(env, e.what());
    } catch (...) {
        throw_illegal_state(env, "Unknown native MobileFineTuner error");
    }
}

std::vector<jint> copy_int_array(JNIEnv* env, jintArray array, int64_t count, const char* name) {
    if (array == nullptr) {
        throw_illegal_argument(env, std::string(name) + " must not be null");
        return {};
    }
    if (env->GetArrayLength(array) != count) {
        throw_illegal_argument(env, std::string(name) + " has unexpected length");
        return {};
    }
    std::vector<jint> values(static_cast<size_t>(count));
    env->GetIntArrayRegion(array, 0, static_cast<jsize>(count), values.data());
    return values;
}

std::vector<jfloat> copy_float_array(JNIEnv* env, jfloatArray array, int64_t count, const char* name) {
    if (array == nullptr) {
        throw_illegal_argument(env, std::string(name) + " must not be null");
        return {};
    }
    if (env->GetArrayLength(array) != count) {
        throw_illegal_argument(env, std::string(name) + " has unexpected length");
        return {};
    }
    std::vector<jfloat> values(static_cast<size_t>(count));
    env->GetFloatArrayRegion(array, 0, static_cast<jsize>(count), values.data());
    return values;
}

std::vector<jboolean> copy_boolean_array(
        JNIEnv* env,
        jbooleanArray array,
        int64_t count,
        const char* name) {
    if (array == nullptr) {
        throw_illegal_argument(env, std::string(name) + " must not be null");
        return {};
    }
    if (env->GetArrayLength(array) != count) {
        throw_illegal_argument(env, std::string(name) + " has unexpected length");
        return {};
    }
    std::vector<jboolean> values(static_cast<size_t>(count));
    env->GetBooleanArrayRegion(array, 0, static_cast<jsize>(count), values.data());
    return values;
}

jdoubleArray make_double_array(JNIEnv* env, const jdouble* values, jsize count) {
    jdoubleArray out = env->NewDoubleArray(count);
    if (out == nullptr) {
        return nullptr;
    }
    env->SetDoubleArrayRegion(out, 0, count, values);
    return out;
}

std::shared_ptr<ops::Tensor> make_input_ids(const std::vector<int>& ids) {
    std::vector<int32_t> values(ids.begin(), ids.end());
    return std::make_shared<ops::Tensor>(
        std::vector<int64_t>{1, static_cast<int64_t>(values.size())},
        values.data(),
        ops::kInt32,
        ops::kCPU);
}

std::shared_ptr<ops::Tensor> make_attention_mask(size_t length) {
    std::vector<float> values(length, 1.0f);
    return std::make_shared<ops::Tensor>(
        std::vector<int64_t>{1, static_cast<int64_t>(values.size())},
        values.data(),
        ops::kFloat32,
        ops::kCPU);
}

int argmax_last_token(const ops::TensorPtr& logits) {
    if (!logits || logits->ndim() != 3) {
        throw std::runtime_error("Expected logits with shape [batch, seq, vocab]");
    }
    if (logits->size(0) != 1) {
        throw std::runtime_error("Generation currently supports batch size 1 only");
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

int sample_last_token(const ops::TensorPtr& logits,
                      float temperature,
                      float top_p,
                      std::mt19937& rng) {
    if (temperature <= 0.0f) {
        return argmax_last_token(logits);
    }
    if (!logits || logits->ndim() != 3) {
        throw std::runtime_error("Expected logits with shape [batch, seq, vocab]");
    }
    if (logits->size(0) != 1) {
        throw std::runtime_error("Generation currently supports batch size 1 only");
    }
    const int64_t seq_len = logits->size(1);
    const int64_t vocab = logits->size(2);
    if (seq_len <= 0 || vocab <= 0) {
        throw std::runtime_error("Invalid logits shape");
    }

    const float* data = logits->data<float>();
    const int64_t offset = (seq_len - 1) * vocab;
    std::vector<float> scores(static_cast<size_t>(vocab));
    float max_score = data[offset];
    for (int64_t i = 0; i < vocab; ++i) {
        const float value = data[offset + i];
        scores[static_cast<size_t>(i)] = value;
        if (value > max_score) {
            max_score = value;
        }
    }

    struct Candidate {
        int id;
        float prob;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(vocab));
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

std::string make_qwen_chat_prompt(const std::string& prompt) {
    return
        "<|im_start|>system\n"
        "You are a concise assistant. Answer directly in one short sentence."
        "<|im_end|>\n"
        "<|im_start|>user\n" +
        prompt +
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n";
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

void fill_parameters(ops::AutoModelForCausalLM& model, float value) {
    for (const auto& param : model.parameters()) {
        if (!param || param->dtype() != ops::kFloat32) {
            continue;
        }
        float* data = param->data<float>();
        for (int64_t i = 0; i < param->numel(); ++i) {
            data[i] = value;
        }
    }
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeBuildInfo(JNIEnv* env, jclass) {
    return env->NewStringUTF("MobileFineTuner Android SDK JNI: AutoModelForCausalLM + AutoTrainer + DPOTrainer + KTOTrainer");
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeSelfTest(
        JNIEnv* env,
        jclass,
        jstring working_dir) {
    return guarded(env, [&]() -> jdoubleArray {
        const std::string working_dir_string = to_string(env, working_dir);
        if (working_dir_string.empty()) {
            throw std::invalid_argument("workingDir must not be empty");
        }

        namespace fs = std::filesystem;
        const fs::path root = fs::path(working_dir_string) / "mft_sdk_native_self_test";
        fs::create_directories(root);
        {
            std::ofstream config(root / "config.json");
            if (!config.is_open()) {
                throw std::runtime_error("Failed to create self-test config.json");
            }
            config << R"({"model_type":"gpt2","vocab_size":16,"n_positions":8,"n_embd":8,"n_layer":1,"n_head":2})";
        }

        const auto t0 = std::chrono::steady_clock::now();

        ops::AutoModelLoadOptions options;
        options.load_weights = false;
        options.verbose = false;

        auto model = ops::AutoModelForCausalLM::from_pretrained(root.string(), options);
        fill_parameters(*model, 0.01f);

        ops::AutoLoraConfig lora;
        lora.rank = 2;
        lora.alpha = 4.0f;
        lora.dropout = 0.0f;
        lora.seed = 7;
        model->init_lora(lora);

        int32_t ids[] = {1, 2, 3};
        float mask[] = {1.0f, 1.0f, 1.0f};
        int32_t labels[] = {-100, 2, 3};
        auto input_ids = std::make_shared<ops::Tensor>(
            std::vector<int64_t>{1, 3}, ids, ops::kInt32, ops::kCPU);
        auto attention_mask = std::make_shared<ops::Tensor>(
            std::vector<int64_t>{1, 3}, mask, ops::kFloat32, ops::kCPU);
        auto label_tensor = std::make_shared<ops::Tensor>(
            std::vector<int64_t>{1, 3}, labels, ops::kInt32, ops::kCPU);

        ops::AutoTrainerConfig trainer_cfg;
        trainer_cfg.learning_rate = 1e-3f;
        ops::AutoTrainer trainer(*model, trainer_cfg);
        const ops::AutoTrainStepResult step =
            trainer.train_step(input_ids, attention_mask, label_tensor);

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        jdouble values[3] = {
            static_cast<jdouble>(step.loss),
            static_cast<jdouble>(step.trainable_tensor_count),
            static_cast<jdouble>(elapsed_ms)
        };
        return make_double_array(env, values, 3);
    }, static_cast<jdoubleArray>(nullptr));
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeCreate(
        JNIEnv* env,
        jclass,
        jstring model_dir,
        jboolean load_weights) {
    return guarded(env, [&]() -> jlong {
        const std::string model_dir_string = to_string(env, model_dir);
        if (model_dir_string.empty()) {
            throw std::invalid_argument("modelDir must not be empty");
        }

        ops::AutoModelLoadOptions options;
        options.load_weights = (load_weights == JNI_TRUE);
        options.verbose = false;

        auto session = std::make_unique<NativeSession>();
        session->model_dir = model_dir_string;
        session->model = ops::AutoModelForCausalLM::from_pretrained(model_dir_string, options);
        return reinterpret_cast<jlong>(session.release());
    }, static_cast<jlong>(0));
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeInitLora(
        JNIEnv* env,
        jclass,
        jlong handle,
        jint rank,
        jfloat alpha,
        jfloat dropout,
        jlong seed,
        jobjectArray target_modules) {
    guarded_void(env, [&]() {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return;
        }
        if (!session->model) {
            throw std::runtime_error("Model is not initialized");
        }

        ops::AutoLoraConfig config;
        config.rank = rank;
        config.alpha = alpha;
        config.dropout = dropout;
        config.seed = static_cast<uint64_t>(seed);
        config.target_modules = to_string_vector(env, target_modules);
        if (env->ExceptionCheck()) {
            return;
        }
        session->model->init_lora(config);
    });
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeCreateTrainer(
        JNIEnv* env,
        jclass,
        jlong handle,
        jfloat learning_rate,
        jfloat weight_decay,
        jfloat max_grad_norm,
        jint ignore_index,
        jboolean use_streaming_lm_loss,
        jint gradient_accumulation_steps) {
    guarded_void(env, [&]() {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return;
        }
        if (!session->model) {
            throw std::runtime_error("Model is not initialized");
        }

        ops::AutoTrainerConfig config;
        config.learning_rate = learning_rate;
        config.weight_decay = weight_decay;
        config.max_grad_norm = max_grad_norm;
        config.ignore_index = ignore_index;
        config.use_streaming_lm_loss = use_streaming_lm_loss == JNI_TRUE;
        config.gradient_accumulation_steps = gradient_accumulation_steps;
        session->trainer = std::make_unique<ops::AutoTrainer>(*session->model, config);
    });
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeCreateDpoTrainer(
        JNIEnv* env,
        jclass,
        jlong handle,
        jfloat learning_rate,
        jfloat weight_decay,
        jfloat max_grad_norm,
        jfloat beta,
        jboolean use_streaming_dpo_loss,
        jint gradient_accumulation_steps) {
    guarded_void(env, [&]() {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return;
        }
        if (!session->model) {
            throw std::runtime_error("Model is not initialized");
        }

        ops::DPOTrainerConfig config;
        config.learning_rate = learning_rate;
        config.weight_decay = weight_decay;
        config.max_grad_norm = max_grad_norm;
        config.beta = beta;
        config.use_streaming_dpo_loss = use_streaming_dpo_loss == JNI_TRUE;
        config.gradient_accumulation_steps = gradient_accumulation_steps;
        session->dpo_trainer = std::make_unique<ops::DPOTrainer>(*session->model, config);
    });
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeCreateKtoTrainer(
        JNIEnv* env,
        jclass,
        jlong handle,
        jfloat learning_rate,
        jfloat weight_decay,
        jfloat max_grad_norm,
        jfloat beta,
        jfloat desirable_weight,
        jfloat undesirable_weight,
        jboolean use_streaming_kto_loss,
        jint gradient_accumulation_steps) {
    guarded_void(env, [&]() {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return;
        }
        if (!session->model) {
            throw std::runtime_error("Model is not initialized");
        }

        ops::KTOTrainerConfig config;
        config.learning_rate = learning_rate;
        config.weight_decay = weight_decay;
        config.max_grad_norm = max_grad_norm;
        config.beta = beta;
        config.desirable_weight = desirable_weight;
        config.undesirable_weight = undesirable_weight;
        config.use_streaming_kto_loss = use_streaming_kto_loss == JNI_TRUE;
        config.gradient_accumulation_steps = gradient_accumulation_steps;
        session->kto_trainer = std::make_unique<ops::KTOTrainer>(*session->model, config);
    });
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeTrainStep(
        JNIEnv* env,
        jclass,
        jlong handle,
        jintArray input_ids,
        jfloatArray attention_mask,
        jintArray labels,
        jint batch_size,
        jint sequence_length) {
    return guarded(env, [&]() -> jdoubleArray {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return nullptr;
        }
        if (!session->trainer) {
            throw std::runtime_error("Trainer is not initialized; call createTrainer first");
        }
        if (batch_size <= 0 || sequence_length <= 1) {
            throw std::invalid_argument("batchSize must be positive and sequenceLength must be > 1");
        }

        const int64_t count = static_cast<int64_t>(batch_size) * sequence_length;
        auto ids = copy_int_array(env, input_ids, count, "inputIds");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto mask = copy_float_array(env, attention_mask, count, "attentionMask");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto y = copy_int_array(env, labels, count, "labels");
        if (env->ExceptionCheck()) {
            return nullptr;
        }

        const std::vector<int64_t> shape{
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(sequence_length)
        };
        auto input_tensor = std::make_shared<ops::Tensor>(
            shape, ids.data(), ops::kInt32, ops::kCPU);
        auto mask_tensor = std::make_shared<ops::Tensor>(
            shape, mask.data(), ops::kFloat32, ops::kCPU);
        auto label_tensor = std::make_shared<ops::Tensor>(
            shape, y.data(), ops::kInt32, ops::kCPU);

        const ops::AutoTrainStepResult result =
            session->trainer->train_step(input_tensor, mask_tensor, label_tensor);

        jdouble values[7] = {
            static_cast<jdouble>(result.loss),
            static_cast<jdouble>(result.trainable_tensor_count),
            static_cast<jdouble>(result.valid_label_count),
            static_cast<jdouble>(result.accumulated_loss),
            static_cast<jdouble>(result.accumulation_step),
            static_cast<jdouble>(result.gradient_accumulation_steps),
            result.optimizer_step ? 1.0 : 0.0
        };
        return make_double_array(env, values, 7);
    }, static_cast<jdoubleArray>(nullptr));
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeTrainTextBatch(
        JNIEnv* env,
        jclass,
        jlong handle,
        jobjectArray texts,
        jint sequence_length,
        jboolean append_eos) {
    return guarded(env, [&]() -> jdoubleArray {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return nullptr;
        }
        if (!session->trainer) {
            throw std::runtime_error("Trainer is not initialized; call createTrainer first");
        }
        if (sequence_length <= 1) {
            throw std::invalid_argument("sequenceLength must be > 1");
        }

        auto rows = copy_string_array(env, texts, "texts");
        if (env->ExceptionCheck()) {
            return nullptr;
        }

        ops::CausalLMBatchConfig batch_cfg;
        batch_cfg.sequence_length = sequence_length;
        batch_cfg.append_eos = append_eos == JNI_TRUE;
        auto batch = ops::make_causal_lm_batch(require_tokenizer(*session), rows, batch_cfg);
        const ops::AutoTrainStepResult result = session->trainer->train_step(batch);

        jdouble values[7] = {
            static_cast<jdouble>(result.loss),
            static_cast<jdouble>(result.trainable_tensor_count),
            static_cast<jdouble>(result.valid_label_count),
            static_cast<jdouble>(result.accumulated_loss),
            static_cast<jdouble>(result.accumulation_step),
            static_cast<jdouble>(result.gradient_accumulation_steps),
            result.optimizer_step ? 1.0 : 0.0
        };
        return make_double_array(env, values, 7);
    }, static_cast<jdoubleArray>(nullptr));
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeTrainPreferenceBatch(
        JNIEnv* env,
        jclass,
        jlong handle,
        jobjectArray prompts,
        jobjectArray chosen,
        jobjectArray rejected,
        jfloatArray ref_chosen_logps,
        jfloatArray ref_rejected_logps,
        jint sequence_length,
        jboolean append_eos_to_response) {
    return guarded(env, [&]() -> jdoubleArray {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return nullptr;
        }
        if (!session->dpo_trainer) {
            throw std::runtime_error("DPO trainer is not initialized; call createDpoTrainer first");
        }
        if (sequence_length <= 1) {
            throw std::invalid_argument("sequenceLength must be > 1");
        }

        auto prompt_rows = copy_string_array(env, prompts, "prompts");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto chosen_rows = copy_string_array(env, chosen, "chosen");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto rejected_rows = copy_string_array(env, rejected, "rejected");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        if (chosen_rows.size() != prompt_rows.size() || rejected_rows.size() != prompt_rows.size()) {
            throw std::invalid_argument("preference text arrays must have the same length");
        }

        const int64_t count = static_cast<int64_t>(prompt_rows.size());
        auto ref_chosen = copy_float_array(env, ref_chosen_logps, count, "refChosenLogps");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto ref_rejected = copy_float_array(env, ref_rejected_logps, count, "refRejectedLogps");
        if (env->ExceptionCheck()) {
            return nullptr;
        }

        std::vector<ops::PreferenceSample> samples;
        samples.reserve(prompt_rows.size());
        for (size_t i = 0; i < prompt_rows.size(); ++i) {
            ops::PreferenceSample sample;
            sample.prompt = std::move(prompt_rows[i]);
            sample.chosen = std::move(chosen_rows[i]);
            sample.rejected = std::move(rejected_rows[i]);
            sample.has_reference_logps = true;
            sample.ref_chosen_logp = ref_chosen[i];
            sample.ref_rejected_logp = ref_rejected[i];
            samples.push_back(std::move(sample));
        }

        ops::PreferenceBatchConfig batch_cfg;
        batch_cfg.sequence_length = sequence_length;
        batch_cfg.append_eos_to_response = append_eos_to_response == JNI_TRUE;
        auto batch = ops::make_preference_batch(require_tokenizer(*session), samples, batch_cfg);
        const ops::DPOTrainStepResult result = session->dpo_trainer->train_step(batch);

        jdouble values[12] = {
            static_cast<jdouble>(result.loss),
            static_cast<jdouble>(result.trainable_tensor_count),
            static_cast<jdouble>(result.pair_count),
            static_cast<jdouble>(result.valid_response_token_count),
            static_cast<jdouble>(result.accumulated_loss),
            static_cast<jdouble>(result.chosen_reward),
            static_cast<jdouble>(result.rejected_reward),
            static_cast<jdouble>(result.reward_margin),
            static_cast<jdouble>(result.reward_accuracy),
            static_cast<jdouble>(result.accumulation_step),
            static_cast<jdouble>(result.gradient_accumulation_steps),
            result.optimizer_step ? 1.0 : 0.0
        };
        return make_double_array(env, values, 12);
    }, static_cast<jdoubleArray>(nullptr));
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeTrainKtoPreferenceBatch(
        JNIEnv* env,
        jclass,
        jlong handle,
        jobjectArray prompts,
        jobjectArray chosen,
        jobjectArray rejected,
        jfloatArray ref_chosen_logps,
        jfloatArray ref_rejected_logps,
        jint sequence_length,
        jboolean append_eos_to_response) {
    return guarded(env, [&]() -> jdoubleArray {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return nullptr;
        }
        if (!session->kto_trainer) {
            throw std::runtime_error("KTO trainer is not initialized; call createKtoTrainer first");
        }
        if (sequence_length <= 1) {
            throw std::invalid_argument("sequenceLength must be > 1");
        }

        auto prompt_rows = copy_string_array(env, prompts, "prompts");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto chosen_rows = copy_string_array(env, chosen, "chosen");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto rejected_rows = copy_string_array(env, rejected, "rejected");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        if (chosen_rows.size() != prompt_rows.size() || rejected_rows.size() != prompt_rows.size()) {
            throw std::invalid_argument("preference text arrays must have the same length");
        }

        const int64_t count = static_cast<int64_t>(prompt_rows.size());
        auto ref_chosen = copy_float_array(env, ref_chosen_logps, count, "refChosenLogps");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto ref_rejected = copy_float_array(env, ref_rejected_logps, count, "refRejectedLogps");
        if (env->ExceptionCheck()) {
            return nullptr;
        }

        std::vector<ops::PreferenceSample> samples;
        samples.reserve(prompt_rows.size());
        for (size_t i = 0; i < prompt_rows.size(); ++i) {
            ops::PreferenceSample sample;
            sample.prompt = std::move(prompt_rows[i]);
            sample.chosen = std::move(chosen_rows[i]);
            sample.rejected = std::move(rejected_rows[i]);
            sample.has_reference_logps = true;
            sample.ref_chosen_logp = ref_chosen[i];
            sample.ref_rejected_logp = ref_rejected[i];
            samples.push_back(std::move(sample));
        }

        ops::PreferenceBatchConfig batch_cfg;
        batch_cfg.sequence_length = sequence_length;
        batch_cfg.append_eos_to_response = append_eos_to_response == JNI_TRUE;
        auto batch = ops::make_preference_batch(require_tokenizer(*session), samples, batch_cfg);
        const ops::KTOTrainStepResult result = session->kto_trainer->train_step(batch);

        // Keep the historical 13-value prefix stable for compatibility and
        // append canonical KTO diagnostics after it.
        jdouble values[17] = {
            static_cast<jdouble>(result.loss),
            static_cast<jdouble>(result.trainable_tensor_count),
            static_cast<jdouble>(result.pair_count),
            static_cast<jdouble>(result.sample_count),
            static_cast<jdouble>(result.valid_response_token_count),
            static_cast<jdouble>(result.accumulated_loss),
            static_cast<jdouble>(result.desirable_reward),
            static_cast<jdouble>(result.undesirable_reward),
            static_cast<jdouble>(result.reward_margin),
            static_cast<jdouble>(result.reward_accuracy),
            static_cast<jdouble>(result.accumulation_step),
            static_cast<jdouble>(result.gradient_accumulation_steps),
            result.optimizer_step ? 1.0 : 0.0,
            static_cast<jdouble>(result.kl_estimate),
            static_cast<jdouble>(result.desirable_count),
            static_cast<jdouble>(result.undesirable_count),
            static_cast<jdouble>(result.valid_kl_response_token_count)
        };
        return make_double_array(env, values, 17);
    }, static_cast<jdoubleArray>(nullptr));
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeTrainKtoBatch(
        JNIEnv* env,
        jclass,
        jlong handle,
        jobjectArray prompts,
        jobjectArray completions,
        jbooleanArray desirable,
        jfloatArray ref_logps,
        jfloatArray ref_kl_logps,
        jint sequence_length,
        jboolean append_eos_to_completion) {
    return guarded(env, [&]() -> jdoubleArray {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return nullptr;
        }
        if (!session->kto_trainer) {
            throw std::runtime_error("KTO trainer is not initialized; call createKtoTrainer first");
        }
        if (sequence_length <= 1) {
            throw std::invalid_argument("sequenceLength must be > 1");
        }

        auto prompt_rows = copy_string_array(env, prompts, "prompts");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto completion_rows = copy_string_array(env, completions, "completions");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        if (completion_rows.size() != prompt_rows.size()) {
            throw std::invalid_argument("KTO text arrays must have the same length");
        }

        const int64_t count = static_cast<int64_t>(prompt_rows.size());
        auto label_rows = copy_boolean_array(env, desirable, count, "desirable");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto reference_rows = copy_float_array(env, ref_logps, count, "refLogps");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto reference_kl_rows = copy_float_array(env, ref_kl_logps, count, "refKlLogps");
        if (env->ExceptionCheck()) {
            return nullptr;
        }

        std::vector<ops::KTOSample> samples;
        samples.reserve(prompt_rows.size());
        for (size_t i = 0; i < prompt_rows.size(); ++i) {
            ops::KTOSample sample;
            sample.prompt = std::move(prompt_rows[i]);
            sample.completion = std::move(completion_rows[i]);
            sample.desirable = label_rows[i] == JNI_TRUE;
            sample.has_reference_logps = true;
            sample.ref_logp = reference_rows[i];
            sample.ref_kl_logp = reference_kl_rows[i];
            samples.push_back(std::move(sample));
        }

        ops::KTOBatchConfig batch_cfg;
        batch_cfg.sequence_length = sequence_length;
        batch_cfg.append_eos_to_completion = append_eos_to_completion == JNI_TRUE;
        auto batch = ops::make_kto_batch(require_tokenizer(*session), samples, batch_cfg);
        const ops::KTOTrainStepResult result = session->kto_trainer->train_step(batch);

        jdouble values[17] = {
            static_cast<jdouble>(result.loss),
            static_cast<jdouble>(result.trainable_tensor_count),
            static_cast<jdouble>(result.pair_count),
            static_cast<jdouble>(result.sample_count),
            static_cast<jdouble>(result.valid_response_token_count),
            static_cast<jdouble>(result.accumulated_loss),
            static_cast<jdouble>(result.desirable_reward),
            static_cast<jdouble>(result.undesirable_reward),
            static_cast<jdouble>(result.reward_margin),
            static_cast<jdouble>(result.reward_accuracy),
            static_cast<jdouble>(result.accumulation_step),
            static_cast<jdouble>(result.gradient_accumulation_steps),
            result.optimizer_step ? 1.0 : 0.0,
            static_cast<jdouble>(result.kl_estimate),
            static_cast<jdouble>(result.desirable_count),
            static_cast<jdouble>(result.undesirable_count),
            static_cast<jdouble>(result.valid_kl_response_token_count)
        };
        return make_double_array(env, values, 17);
    }, static_cast<jdoubleArray>(nullptr));
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeScorePreferenceBatch(
        JNIEnv* env,
        jclass,
        jlong handle,
        jobjectArray prompts,
        jobjectArray chosen,
        jobjectArray rejected,
        jint sequence_length,
        jboolean append_eos_to_response) {
    return guarded(env, [&]() -> jdoubleArray {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return nullptr;
        }
        if (!session->model) {
            throw std::runtime_error("Model is not initialized");
        }
        if (sequence_length <= 1) {
            throw std::invalid_argument("sequenceLength must be > 1");
        }

        auto prompt_rows = copy_string_array(env, prompts, "prompts");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto chosen_rows = copy_string_array(env, chosen, "chosen");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto rejected_rows = copy_string_array(env, rejected, "rejected");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        if (chosen_rows.size() != prompt_rows.size() || rejected_rows.size() != prompt_rows.size()) {
            throw std::invalid_argument("preference text arrays must have the same length");
        }

        std::vector<ops::PreferenceSample> samples;
        samples.reserve(prompt_rows.size());
        for (size_t i = 0; i < prompt_rows.size(); ++i) {
            ops::PreferenceSample sample;
            sample.prompt = std::move(prompt_rows[i]);
            sample.chosen = std::move(chosen_rows[i]);
            sample.rejected = std::move(rejected_rows[i]);
            samples.push_back(std::move(sample));
        }

        ops::PreferenceBatchConfig batch_cfg;
        batch_cfg.sequence_length = sequence_length;
        batch_cfg.append_eos_to_response = append_eos_to_response == JNI_TRUE;
        auto batch = ops::make_preference_batch(require_tokenizer(*session), samples, batch_cfg);
        const ops::PreferenceLogps scores = ops::score_preference_logps(*session->model, batch);

        std::vector<jdouble> values;
        values.reserve(1 + scores.chosen_logps.size() + scores.rejected_logps.size());
        values.push_back(static_cast<jdouble>(scores.valid_response_token_count));
        for (float value : scores.chosen_logps) {
            values.push_back(static_cast<jdouble>(value));
        }
        for (float value : scores.rejected_logps) {
            values.push_back(static_cast<jdouble>(value));
        }
        return make_double_array(env, values.data(), static_cast<jsize>(values.size()));
    }, static_cast<jdoubleArray>(nullptr));
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeScoreKtoBatch(
        JNIEnv* env,
        jclass,
        jlong handle,
        jobjectArray prompts,
        jobjectArray completions,
        jbooleanArray desirable,
        jint sequence_length,
        jboolean append_eos_to_completion) {
    return guarded(env, [&]() -> jdoubleArray {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return nullptr;
        }
        if (!session->model) {
            throw std::runtime_error("Model is not initialized");
        }
        if (sequence_length <= 1) {
            throw std::invalid_argument("sequenceLength must be > 1");
        }

        auto prompt_rows = copy_string_array(env, prompts, "prompts");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        auto completion_rows = copy_string_array(env, completions, "completions");
        if (env->ExceptionCheck()) {
            return nullptr;
        }
        if (completion_rows.size() != prompt_rows.size()) {
            throw std::invalid_argument("KTO text arrays must have the same length");
        }

        const int64_t count = static_cast<int64_t>(prompt_rows.size());
        auto label_rows = copy_boolean_array(env, desirable, count, "desirable");
        if (env->ExceptionCheck()) {
            return nullptr;
        }

        std::vector<ops::KTOSample> samples;
        samples.reserve(prompt_rows.size());
        for (size_t i = 0; i < prompt_rows.size(); ++i) {
            ops::KTOSample sample;
            sample.prompt = std::move(prompt_rows[i]);
            sample.completion = std::move(completion_rows[i]);
            sample.desirable = label_rows[i] == JNI_TRUE;
            samples.push_back(std::move(sample));
        }

        ops::KTOBatchConfig batch_cfg;
        batch_cfg.sequence_length = sequence_length;
        batch_cfg.append_eos_to_completion = append_eos_to_completion == JNI_TRUE;
        auto batch = ops::make_kto_batch(require_tokenizer(*session), samples, batch_cfg);
        const ops::KTOLogps scores = ops::score_kto_logps(*session->model, batch);

        std::vector<jdouble> values;
        values.reserve(2 + scores.response_logps.size() + scores.kl_response_logps.size());
        values.push_back(static_cast<jdouble>(scores.valid_response_token_count));
        values.push_back(static_cast<jdouble>(scores.valid_kl_response_token_count));
        for (float value : scores.response_logps) {
            values.push_back(static_cast<jdouble>(value));
        }
        for (float value : scores.kl_response_logps) {
            values.push_back(static_cast<jdouble>(value));
        }
        return make_double_array(env, values.data(), static_cast<jsize>(values.size()));
    }, static_cast<jdoubleArray>(nullptr));
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeSaveLoraAdapter(
        JNIEnv* env,
        jclass,
        jlong handle,
        jstring path) {
    guarded_void(env, [&]() {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return;
        }
        if (!session->model) {
            throw std::runtime_error("Model is not initialized");
        }
        const std::string adapter_path = to_string(env, path);
        if (adapter_path.empty()) {
            throw std::invalid_argument("adapter path must not be empty");
        }
        session->model->save_lora_adapter_jsonl(adapter_path);
    });
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeLoadLoraAdapter(
        JNIEnv* env,
        jclass,
        jlong handle,
        jstring path) {
    guarded_void(env, [&]() {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return;
        }
        if (!session->model) {
            throw std::runtime_error("Model is not initialized");
        }
        const std::string adapter_path = to_string(env, path);
        if (adapter_path.empty()) {
            throw std::invalid_argument("adapter path must not be empty");
        }
        session->model->load_lora_adapter_jsonl(adapter_path);
    });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeTrainableTensorCount(
        JNIEnv* env,
        jclass,
        jlong handle) {
    return guarded(env, [&]() -> jint {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return 0;
        }
        if (!session->model) {
            throw std::runtime_error("Model is not initialized");
        }
        return static_cast<jint>(session->model->trainable_parameters().size());
    }, static_cast<jint>(0));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeGenerate(
        JNIEnv* env,
        jclass,
        jlong handle,
        jstring prompt,
        jint max_new_tokens,
        jfloat temperature,
        jfloat top_p,
        jlong seed) {
    return guarded(env, [&]() -> jstring {
        NativeSession* session = require_session(env, handle);
        if (session == nullptr) {
            return nullptr;
        }
        if (!session->model) {
            throw std::runtime_error("Model is not initialized");
        }
        if (max_new_tokens <= 0) {
            throw std::invalid_argument("maxNewTokens must be positive");
        }
        if (temperature < 0.0f) {
            throw std::invalid_argument("temperature must be non-negative");
        }
        if (top_p <= 0.0f || top_p > 1.0f) {
            throw std::invalid_argument("topP must be in (0, 1]");
        }

        if (!session->tokenizer) {
            session->tokenizer = ops::TokenizerFactory::from_pretrained(session->model_dir);
        }
        if (!session->tokenizer) {
            throw std::runtime_error("Tokenizer is not initialized");
        }

        const std::string prompt_text = to_string(env, prompt);
        if (prompt_text.empty()) {
            throw std::invalid_argument("prompt must not be empty");
        }

        const bool is_qwen = session->model->family() == ops::ModelFamily::Qwen;
        const std::string model_prompt = is_qwen ? make_qwen_chat_prompt(prompt_text) : prompt_text;
        std::vector<int> tokens = session->tokenizer->encode(model_prompt);
        if (tokens.empty()) {
            throw std::runtime_error("Tokenizer produced no prompt tokens");
        }
        const size_t prompt_token_count = tokens.size();
        const int eos = session->tokenizer->get_eos_token();
        std::mt19937 rng(static_cast<uint32_t>(seed));

        ops::QwenKVCache qwen_cache;
        for (int step = 0; step < max_new_tokens; ++step) {
            auto input_ids = (is_qwen && step > 0)
                ? make_input_ids(std::vector<int>{tokens.back()})
                : make_input_ids(tokens);
            ops::TensorPtr logits;
            if (is_qwen) {
                logits = session->model->forward_qwen_with_cache(input_ids, qwen_cache);
            } else {
                auto attention_mask = make_attention_mask(tokens.size());
                logits = session->model->forward(input_ids, attention_mask);
            }
            const int next_id = sample_last_token(logits, temperature, top_p, rng);
            tokens.push_back(next_id);
            if (is_stop_token(next_id, eos)) {
                break;
            }
        }

        std::vector<int> output_tokens(tokens.begin() + static_cast<long>(prompt_token_count), tokens.end());
        std::string completion = session->tokenizer->decode(output_tokens);
        if (is_qwen) {
            completion = clean_qwen3_output(completion);
        }
        return env->NewStringUTF(completion.c_str());
    }, static_cast<jstring>(nullptr));
}

extern "C" JNIEXPORT void JNICALL
Java_com_mobilefinetuner_sdk_MobileFineTuner_nativeClose(JNIEnv* env, jclass, jlong handle) {
    guarded_void(env, [&]() {
        delete reinterpret_cast<NativeSession*>(handle);
    });
}
