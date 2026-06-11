#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <filesystem>

#include "finetune_ops/graph/gpt2_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include "finetune_ops/graph/lora_injector.h"
#include "finetune_ops/graph/lora_saver.h"
#include "finetune_ops/data/wikitext2_dataset.h"
#include "finetune_ops/data/preference_dataset.h"
#include "finetune_ops/core/tokenizer_bpe.h"
#include "finetune_ops/core/autograd_engine.h"
#include "finetune_ops/core/dpo_loss.h"
#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/optim/adam.h"
#include "finetune_ops/optim/smoke_utils.h"
#include "finetune_ops/core/memory_manager.h"

using namespace std;
using namespace ops;

struct CmdArgs {
    string train_mode = "sft";
    string data_dir;
    string pretrained_dir;
    string lora_out;
    string resume_from;
    string eval_out;
    string metrics_out;
    // Optional: masked JSONL inputs for supervised LM (ids/mask)
    string jsonl_train;
    string jsonl_valid;
    string jsonl_test;
    string preference_train;
    string preference_valid;
    string preference_test;
    
    int epochs = 0;
    int steps = 0;
    int batch_size = 1;
    int grad_accum_steps = 1;
    int seq_len = 128;
    int rank = 8;
    float alpha = 16.0f;
    float lr = 1e-4f;
    float weight_decay = 0.0f;
    int warmup_steps = 0;
    float clip_grad_norm = 1.0f;
    float lora_dropout = 0.0f;
    float dpo_beta = 0.1f;
    bool dpo_pair_forward = true;
    bool cache_ref_logps = false;
    
    int log_interval = 1;
    int eval_interval = 0;
    int eval_batches = 50;
    int eval_batch_size = 2;
    int save_every = 0;
    float ema_beta = 0.9f;
    int seed = 42;
    bool synthetic_smoke = false;
    int smoke_steps = 2;
};

static void print_usage(const char* prog) {
    cerr << "Usage: " << prog << " [options]\n"
         << "Options:\n"
         << "  --data_dir PATH          Data directory (WikiText-2 raw)\n"
         << "  --pretrained_dir PATH    Pretrained model directory\n"
         << "  --train_mode MODE        sft | dpo\n"
         << "  --jsonl_train PATH       JSONL train (optional; contains ids/mask)\n"
         << "  --jsonl_valid PATH       JSONL valid (optional; contains ids/mask)\n"
         << "  --jsonl_test PATH        JSONL test  (optional; contains ids/mask)\n"
         << "  --preference_train PATH  Preference JSONL train (prompt/chosen/rejected)\n"
         << "  --preference_valid PATH  Preference JSONL valid (prompt/chosen/rejected)\n"
         << "  --preference_test PATH   Preference JSONL test  (prompt/chosen/rejected)\n"
         << "  --lora_out PATH          Output path for LoRA adapter\n"
         << "  --resume_from PATH       Resume from LoRA checkpoint\n"
         << "  --eval_out PATH          Write eval metrics to JSONL\n"
         << "  --metrics_out PATH       Write per-step Android metrics NDJSON\n"
         << "  --epochs N               Num epochs (overrides steps if >0)\n"
         << "  --steps N                Num training steps\n"
         << "  --batch_size N           Micro-batch size\n"
         << "  --grad_accum_steps N     Gradient accumulation steps\n"
         << "  --seq_len N              Sequence length\n"
         << "  --rank R                 LoRA rank\n"
         << "  --alpha A                LoRA alpha\n"
         << "  --dpo_beta F             DPO beta\n"
         << "  --dpo_pair_forward 0|1   Batch chosen/rejected into paired DPO forwards\n"
         << "  --cache_ref_logps 0|1    Precompute frozen-reference logps for DPO\n"
         << "  --lr LR                  Learning rate\n"
         << "  --weight_decay WD        Weight decay\n"
         << "  --warmup_steps N         Warmup steps\n"
         << "  --clip_grad_norm F       Gradient clipping threshold\n"
         << "  --lora_dropout F         LoRA dropout\n"
         << "  --log_interval N         Logging interval (steps)\n"
         << "  --eval_interval N        Eval interval (steps; 0=disable)\n"
         << "  --eval_batches N         Num eval batches\n"
         << "  --eval_batch_size N      Eval micro-batch size\n"
         << "  --save_every N           Checkpoint interval (steps)\n"
         << "  --ema_beta F             EMA smoothing factor\n"
         << "  --seed N                 Random seed\n"
         << "  --synthetic_smoke        Run a self-contained 2-step smoke train\n"
         << "  --smoke_steps N          Override synthetic smoke steps\n";
}

static CmdArgs parse_args(int argc, char** argv) {
    CmdArgs args;
    args.data_dir = "data/wikitext2/wikitext-2-raw";

    for (int i = 1; i < argc; ++i) {
        string k = argv[i];
        auto get_val = [&](const string& key)->string {
            if (i + 1 >= argc) { print_usage(argv[0]); exit(1); }
            if (k != key) { print_usage(argv[0]); exit(1); }
            return string(argv[++i]);
        };
        if (k == "--data_dir") args.data_dir = get_val("--data_dir");
        else if (k == "--pretrained_dir") args.pretrained_dir = get_val("--pretrained_dir");
        else if (k == "--train_mode") args.train_mode = get_val("--train_mode");
        else if (k == "--jsonl_train") args.jsonl_train = get_val("--jsonl_train");
        else if (k == "--jsonl_valid") args.jsonl_valid = get_val("--jsonl_valid");
        else if (k == "--jsonl_test") args.jsonl_test = get_val("--jsonl_test");
        else if (k == "--preference_train") args.preference_train = get_val("--preference_train");
        else if (k == "--preference_valid") args.preference_valid = get_val("--preference_valid");
        else if (k == "--preference_test") args.preference_test = get_val("--preference_test");
        else if (k == "--lora_out") args.lora_out = get_val("--lora_out");
        else if (k == "--resume_from") args.resume_from = get_val("--resume_from");
        else if (k == "--eval_out") args.eval_out = get_val("--eval_out");
        else if (k == "--metrics_out") args.metrics_out = get_val("--metrics_out");
        else if (k == "--epochs") args.epochs = stoi(get_val("--epochs"));
        else if (k == "--steps") args.steps = stoi(get_val("--steps"));
        else if (k == "--batch_size") args.batch_size = stoi(get_val("--batch_size"));
        else if (k == "--grad_accum_steps") args.grad_accum_steps = stoi(get_val("--grad_accum_steps"));
        else if (k == "--seq_len") args.seq_len = stoi(get_val("--seq_len"));
        else if (k == "--rank") args.rank = stoi(get_val("--rank"));
        else if (k == "--alpha") args.alpha = stof(get_val("--alpha"));
        else if (k == "--dpo_beta") args.dpo_beta = stof(get_val("--dpo_beta"));
        else if (k == "--dpo_pair_forward") args.dpo_pair_forward = (stoi(get_val("--dpo_pair_forward")) != 0);
        else if (k == "--cache_ref_logps") args.cache_ref_logps = (stoi(get_val("--cache_ref_logps")) != 0);
        else if (k == "--lr") args.lr = stof(get_val("--lr"));
        else if (k == "--weight_decay") args.weight_decay = stof(get_val("--weight_decay"));
        else if (k == "--warmup_steps") args.warmup_steps = stoi(get_val("--warmup_steps"));
        else if (k == "--clip_grad_norm") args.clip_grad_norm = stof(get_val("--clip_grad_norm"));
        else if (k == "--lora_dropout") args.lora_dropout = stof(get_val("--lora_dropout"));
        else if (k == "--log_interval") args.log_interval = stoi(get_val("--log_interval"));
        else if (k == "--eval_interval") args.eval_interval = stoi(get_val("--eval_interval"));
        else if (k == "--eval_batches") args.eval_batches = stoi(get_val("--eval_batches"));
        else if (k == "--eval_batch_size") args.eval_batch_size = stoi(get_val("--eval_batch_size"));
        else if (k == "--save_every") args.save_every = stoi(get_val("--save_every"));
        else if (k == "--ema_beta") args.ema_beta = stof(get_val("--ema_beta"));
        else if (k == "--seed") args.seed = stoi(get_val("--seed"));
        else if (k == "--synthetic_smoke") args.synthetic_smoke = true;
        else if (k == "--smoke_steps") args.smoke_steps = stoi(get_val("--smoke_steps"));
        else if (k == "--help" || k == "-h") { print_usage(argv[0]); exit(0); }
        else { cerr << "Unknown arg: " << k << endl; print_usage(argv[0]); exit(1); }
    }
    if (args.train_mode != "sft" && args.train_mode != "dpo") {
        cerr << "Unsupported --train_mode: " << args.train_mode << endl;
        print_usage(argv[0]);
        exit(1);
    }
    return args;
}

static void append_jsonl(const string& path, const string& json_str) {
    ofstream ofs(path, ios::app);
    if (ofs) {
        ofs << json_str << "\n";
    }
}

static void truncate_file_if_requested(const string& path) {
    if (path.empty()) return;
    ofstream ofs(path, ios::trunc);
}

static size_t dtype_num_bytes(DType dtype) {
    const size_t bytes = DTypeUtils::size_of(dtype);
    if (bytes == 0) {
        throw runtime_error("Unsupported dtype in dtype_num_bytes");
    }
    return bytes;
}

static TensorPtr concat_batch_dim0(const TensorPtr& first, const TensorPtr& second) {
    if (!first || !second) {
        throw runtime_error("concat_batch_dim0: null tensor");
    }
    if (first->dtype() != second->dtype() || first->device() != second->device()) {
        throw runtime_error("concat_batch_dim0: dtype/device mismatch");
    }
    if (first->ndim() != second->ndim() || first->ndim() < 1) {
        throw runtime_error("concat_batch_dim0: ndim mismatch");
    }

    auto out_shape = first->shape();
    if (out_shape[0] != second->shape()[0]) {
        // Different batch sizes are not needed for current DPO batches and make
        // split bookkeeping error-prone.
        throw runtime_error("concat_batch_dim0: batch-size mismatch");
    }
    for (int64_t d = 1; d < first->ndim(); ++d) {
        if (first->shape()[static_cast<size_t>(d)] != second->shape()[static_cast<size_t>(d)]) {
            throw runtime_error("concat_batch_dim0: shape mismatch");
        }
    }
    out_shape[0] = first->shape()[0] + second->shape()[0];

    auto out = std::make_shared<Tensor>(out_shape, first->dtype(), first->device());
    const size_t bytes = static_cast<size_t>(first->numel()) * dtype_num_bytes(first->dtype());
    std::memcpy(out->data_ptr(), first->data_ptr(), bytes);
    std::memcpy(static_cast<uint8_t*>(out->data_ptr()) + bytes, second->data_ptr(), bytes);
    return out;
}

struct DPOForwardPack {
    TensorPtr policy_chosen_logits;
    TensorPtr policy_rejected_logits;
    TensorPtr ref_chosen_logits;
    TensorPtr ref_rejected_logits;
};

struct RefLogpCache {
    std::vector<float> chosen;
    std::vector<float> rejected;
    bool enabled = false;
};

static string make_checkpoint_path(const string& base, int step) {
    size_t dot_pos = base.find_last_of('.');
    string stem = (dot_pos == string::npos) ? base : base.substr(0, dot_pos);
    string ext = (dot_pos == string::npos) ? "" : base.substr(dot_pos);
    stringstream ss;
    ss << stem << "_step" << step << ext;
    return ss.str();
}

static void require_file(const string& path, const string& desc) {
    if (!std::filesystem::exists(path)) {
        throw runtime_error(desc + " not found: " + path);
    }
}

static int run_synthetic_smoke(const CmdArgs& args) {
    cout << "\n========== GPT-2 LoRA Synthetic Smoke ==========\n" << endl;

    GPT2Config cfg;
    cfg.vocab_size = 64;
    cfg.n_positions = max(16, args.seq_len + 2);
    cfg.n_embd = 16;
    cfg.n_layer = 1;
    cfg.n_head = 4;
    cfg.use_memory_efficient_attention = true;

    GPT2Model model(cfg);
    std::mt19937 rng(static_cast<uint32_t>(args.seed));
    smoke::initialize_tiny_gpt2(model, rng);
    model.tie_weights();

    LoraSpec lora_spec;
    lora_spec.rank = args.rank;
    lora_spec.alpha = args.alpha;
    lora_spec.dropout = args.lora_dropout;
    lora_spec.split_qkv = false;
    lora_spec.targets = {LoraTarget::AttnQKV, LoraTarget::AttnProj};

    LoraInjector injector;
    injector.inject(model, lora_spec);
    auto trainable = injector.get_trainable_params();
    if (trainable.empty()) {
        throw runtime_error("synthetic smoke created no trainable LoRA parameters");
    }

    auto before = smoke::clone_tensors(trainable);
    const int batch_size = max(1, min(args.batch_size, 2));
    const int seq_len = max(4, min(args.seq_len, 8));
    auto input_ids = smoke::make_input_ids(batch_size, seq_len, cfg.vocab_size, 1);
    auto labels = smoke::make_shifted_labels(input_ids, cfg.vocab_size);
    auto attention_mask = smoke::make_attention_mask(batch_size, seq_len);

    AdamConfig opt_cfg;
    opt_cfg.learning_rate = max(args.lr, 1e-2f);
    opt_cfg.weight_decay = 0.0f;
    Adam optimizer(opt_cfg);

    vector<float> losses;
    const int smoke_steps = max(1, args.smoke_steps);
    for (int step = 0; step < smoke_steps; ++step) {
        auto logits = model.forward(input_ids, attention_mask);
        auto loss = lm_cross_entropy(logits, labels, -100, "mean");
        if (!smoke::is_finite_scalar(loss)) {
            throw runtime_error("non-finite loss during GPT-2 synthetic smoke");
        }
        losses.push_back(loss->data<float>()[0]);
        loss->backward();

        vector<TensorPtr> grads;
        grads.reserve(trainable.size());
        for (const auto& param : trainable) {
            grads.push_back(param->grad());
        }
        double grad_norm = smoke::grad_l2_norm(grads);
        if (!(grad_norm > 0.0) || !std::isfinite(grad_norm)) {
            throw runtime_error("invalid gradient norm during GPT-2 synthetic smoke");
        }

        optimizer.step(trainable, grads);
        smoke::zero_grads(trainable);
        smoke::cleanup_step_memory();

        cout << "[smoke step " << (step + 1) << "/" << smoke_steps
             << "] loss=" << losses.back()
             << " grad_norm=" << grad_norm << endl;
    }

    double param_delta = smoke::max_param_delta(before, trainable);
    if (!args.lora_out.empty()) {
        injector.save_lora_safetensors(args.lora_out);
    }

    cout << "[SyntheticSmoke] param_delta=" << param_delta << endl;
    const bool ok = std::isfinite(param_delta) && param_delta > 0.0;
    cout << (ok ? "[PASS]" : "[FAIL]") << endl;
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto args = parse_args(argc, argv);
    if (args.synthetic_smoke) {
        try {
            return run_synthetic_smoke(args);
        } catch (const exception& e) {
            cerr << "\n❌ Exception: " << e.what() << endl;
            return 1;
        }
    }

    try {
        if (args.pretrained_dir.empty()) {
            throw runtime_error("--pretrained_dir is required unless --synthetic_smoke is used");
        }
        require_file(args.pretrained_dir + "/config.json", "GPT-2 config");
        if (args.epochs <= 0 && args.steps <= 0) {
            args.epochs = 1;
        }
        if (args.jsonl_train.empty() && args.jsonl_valid.empty() && args.jsonl_test.empty()) {
            require_file(args.data_dir + "/wiki.train.raw", "WikiText-2 train split");
            require_file(args.data_dir + "/wiki.valid.raw", "WikiText-2 valid split");
        }

        cout << "\n========== GPT-2 LoRA Finetune (Pro) ==========\n" << endl;
        cout << "[Config]" << endl;
        // Build-time BLAS toggle (informational only; actual kernels may choose paths at runtime)
#if defined(USE_BLAS)
        cout << "  BLAS          : ON (compiled)" << endl;
#else
        cout << "  BLAS          : OFF (compiled)" << endl;
#endif
        // Data source: prefer masked JSONL (MMLU-style samples); otherwise fall back to raw text under data_dir
        cout << "  train_mode     : " << args.train_mode << endl;
        if (args.train_mode == "dpo") {
            cout << "  preference_train: " << args.preference_train << endl;
            cout << "  preference_valid: " << args.preference_valid << endl;
            if (!args.preference_test.empty()) cout << "  preference_test : " << args.preference_test << endl;
            cout << "  dpo_beta       : " << args.dpo_beta << endl;
            cout << "  dpo_pair_forward: " << (args.dpo_pair_forward ? "ON" : "OFF") << endl;
            cout << "  cache_ref_logps: " << (args.cache_ref_logps ? "ON" : "OFF") << endl;
        } else if (!args.jsonl_train.empty() || !args.jsonl_valid.empty() || !args.jsonl_test.empty()) {
            cout << "  data_source    : JSONL (masked Causal LM)" << endl;
            if (!args.jsonl_train.empty()) cout << "  jsonl_train    : " << args.jsonl_train << endl;
            if (!args.jsonl_valid.empty()) cout << "  jsonl_valid    : " << args.jsonl_valid << endl;
            if (!args.jsonl_test.empty())  cout << "  jsonl_test     : " << args.jsonl_test  << endl;
        } else {
            cout << "  data_dir       : " << args.data_dir << endl;
        }
        cout << "  pretrained_dir : " << args.pretrained_dir << endl;
        cout << "  lora_out       : " << args.lora_out << endl;
        if (!args.resume_from.empty()) {
            cout << "  resume_from    : " << args.resume_from << endl;
        }
        if (!args.eval_out.empty()) {
            cout << "  eval_out       : " << args.eval_out << endl;
        }
        if (!args.metrics_out.empty()) {
            cout << "  metrics_out    : " << args.metrics_out << endl;
            truncate_file_if_requested(args.metrics_out);
        }
        cout << "  epochs         : " << args.epochs << endl;
        cout << "  batch_size     : " << args.batch_size << endl;
        cout << "  grad_accum     : " << args.grad_accum_steps << endl;
        cout << "  seq_len        : " << args.seq_len << endl;
        cout << "  rank/alpha     : " << args.rank << "/" << args.alpha << endl;
        cout << "  lr/wd          : " << args.lr << "/" << args.weight_decay << endl;
        cout << "  warmup_steps   : " << args.warmup_steps << endl;
        cout << "  clip_grad_norm : " << args.clip_grad_norm << endl;
        cout << "  lora_dropout   : " << args.lora_dropout << endl;
        cout << "  log_interval   : " << args.log_interval << endl;
        cout << "  eval_interval  : " << args.eval_interval << endl;
        cout << "  save_every     : " << args.save_every << endl;
        cout << "  ema_beta       : " << args.ema_beta << endl;
        cout << "  seed           : " << args.seed << endl;

        // 1) Construct model and load pretrained weights
        cout << "\n[1/6] Load pretrained model..." << endl;
        // Load size dynamically from HF config.json (supports gpt2 / gpt2-medium / gpt2-large / gpt2-xl)
        GPT2Config cfg = GPT2Config::from_pretrained(args.pretrained_dir + "/config.json");
        GPT2Model model(cfg);
        // Native memory-efficient attention backward is still incomplete.
        // Force the standard path for all LoRA finetuning modes so SFT/DPO
        // comparisons use the same valid gradient path.
        model.set_use_memory_efficient_attention(false);
        model.tie_weights();
        model.print_model_info();

        SafeTensorsModelReader reader(args.pretrained_dir);
        reader.parse_headers();
        auto key_map = GPT2KeyMapper::generate_gpt2_mapping(cfg.n_layer);
        // Linear weights in GPT-2 safetensors are stored as [in, out] (e.g., [768, 2304]); no transpose needed.
        // Explicitly disable load-time transpose to avoid runtime transposition overhead.
        SafeTensorsLoadOptions load_opts;
        load_opts.transpose_linear = false;
        load_opts.verbose = false;        // reduce loader verbosity during training
        auto loaded = reader.load_tensors_mapped(key_map, load_opts);
        for (const auto& kv : loaded) {
            model.assign_weight(kv.first, kv.second);
        }
        cout << "  ✓ Model loaded" << endl;

        // 2) Initialize LoRA and inject / resume
        cout << "\n[2/6] Inject/Resume LoRA..." << endl;
        model.init_lora_modules();

        // Utility: clear existing LoRA slices to avoid duplicate attachments
        auto clear_all_lora = [&]() {
            const auto& cfg_local = model.config();
            for (int li = 0; li < cfg_local.n_layer; ++li) {
                auto& blk = model.get_block(li);
                if (blk.qkv_lin) blk.qkv_lin->clear_lora();
                if (blk.proj_lin) blk.proj_lin->clear_lora();
                if (blk.fc_in_lin) blk.fc_in_lin->clear_lora();
                if (blk.fc_out_lin) blk.fc_out_lin->clear_lora();
            }
        };

        LoraSpec active_lora_spec;
        bool has_active_lora_spec = false;

        int start_step = 0;
        std::vector<TensorPtr> trainable;
        if (!args.resume_from.empty()) {
            cout << "  ↪ Found resume file, loading checkpoint: " << args.resume_from << endl;
            clear_all_lora();
            auto lora_state = LoraSaver::load_safetensors(args.resume_from);
            LoraSaver::attach_from_state(model, lora_state);
            // After resuming, refresh trainable parameters and ensure they require grad
            trainable = model.get_lora_parameters();
            for (auto& p : trainable) { p->set_requires_grad(true); p->zero_grad(); }
            cout << "  ✓ LoRA weights loaded and attached (trainable=" << trainable.size() << ")" << endl;

            active_lora_spec.rank = lora_state.rank;
            active_lora_spec.alpha = lora_state.alpha;
            active_lora_spec.dropout = lora_state.dropout;
            active_lora_spec.split_qkv = lora_state.split_qkv;
            if (!lora_state.targets.empty()) {
                active_lora_spec.targets = lora_state.targets;
            } else {
                active_lora_spec.targets = { LoraTarget::AttnQKV, LoraTarget::AttnProj };
            }
            has_active_lora_spec = true;
        } else {
            LoraSpec lora_spec;
            lora_spec.rank = args.rank;
            lora_spec.alpha = args.alpha;
            lora_spec.dropout = args.lora_dropout;
            // Align with PyTorch/PEFT default topology:
            // - do not split Q/K/V for fused c_attn
            // - only target attention QKV (c_attn) and attention proj (c_proj)
            lora_spec.split_qkv = false;
            lora_spec.targets = { LoraTarget::AttnQKV, LoraTarget::AttnProj };

            LoraInjector injector;
            injector.inject(model, lora_spec);
            trainable = model.get_lora_parameters();
            for (auto& p : trainable) p->zero_grad();
            cout << "  ✓ LoRA injected: " << trainable.size() << " trainable parameters" << endl;

            active_lora_spec = lora_spec;
            has_active_lora_spec = true;
        }

        // 3) Prepare tokenizer and dataset
        cout << "\n[3/6] Load dataset..." << endl;
        auto tok_cfg = BPEConfig::from_pretrained(args.pretrained_dir);
        GPT2BPETokenizer tokenizer(tok_cfg);
        tokenizer.load();

        if (args.train_mode == "dpo") {
            if (args.preference_train.empty() || args.preference_valid.empty()) {
                throw runtime_error("DPO mode requires --preference_train and --preference_valid");
            }

            PreferenceDatasetConfig pref_cfg;
            pref_cfg.train_path = args.preference_train;
            pref_cfg.valid_path = args.preference_valid;
            pref_cfg.test_path = args.preference_test;
            pref_cfg.seq_len = args.seq_len;
            pref_cfg.batch_size = args.batch_size;
            pref_cfg.seed = static_cast<uint64_t>(args.seed);

            PreferenceDataset train_pref_dataset(pref_cfg, &tokenizer);
            train_pref_dataset.load(PreferenceSplit::Train);
            train_pref_dataset.shuffle();
            train_pref_dataset.print_stats();

            PreferenceDataset valid_pref_dataset(pref_cfg, &tokenizer);
            valid_pref_dataset.load(PreferenceSplit::Valid);
            valid_pref_dataset.print_stats();

            GPT2Model ref_model(cfg);
            ref_model.set_use_memory_efficient_attention(false);
            ref_model.tie_weights();
            for (const auto& kv : loaded) {
                ref_model.assign_weight(kv.first, kv.second);
            }
            cout << "  ✓ Reference model loaded (frozen scoring only)" << endl;

            const int64_t train_pairs = static_cast<int64_t>(train_pref_dataset.num_samples());
            const int64_t micro_bs = args.batch_size;
            const int64_t accum = max(1, args.grad_accum_steps);
            const int64_t steps_per_epoch = (train_pairs + micro_bs * accum - 1) / (micro_bs * accum);
            if (args.epochs > 0) {
                args.steps = steps_per_epoch * args.epochs;
            }

            cout << "\n[Training plan]" << endl;
            cout << "  epochs         : " << args.epochs << endl;
            cout << "  steps_per_epoch: " << steps_per_epoch << endl;
            cout << "  total_steps    : " << args.steps << endl;
            cout << "  Effective batch: " << (micro_bs * accum) << " (micro=" << micro_bs
                 << " × accum=" << accum << ")" << endl;

            cout << "\n[4/6] Init optimizer..." << endl;
            AdamConfig opt_cfg;
            opt_cfg.learning_rate = args.lr;
            opt_cfg.beta1 = 0.9f;
            opt_cfg.beta2 = 0.999f;
            opt_cfg.epsilon = 1e-8f;
            opt_cfg.weight_decay = args.weight_decay;
            opt_cfg.clip_grad_norm = args.clip_grad_norm;
            Adam optimizer(opt_cfg);
            cout << "  ✓ Adam optimizer ready" << endl;

            auto count_response_tokens = [](const TensorPtr& mask) -> int64_t {
                if (!mask) return 0;
                const int32_t* ptr = mask->data<int32_t>();
                int64_t total = 0;
                for (int64_t i = 0; i < mask->numel(); ++i) {
                    total += (ptr[i] != 0) ? 1 : 0;
                }
                return total;
            };

            auto lr_schedule = [&](int step) -> float {
                const float base_lr = args.lr;
                const int T = args.steps;
                const int W = max(0, args.warmup_steps);
                if (step < W) {
                    return base_lr * (float(step + 1) / float(max(1, W)));
                }
                const int s = step - W;
                const int d = max(1, T - W);
                const float min_lr = 0.1f * base_lr;
                const float PI = acosf(-1.0f);
                const float cosv = 0.5f * (1.0f + cos(PI * float(s) / float(d)));
                return min_lr + (base_lr - min_lr) * cosv;
            };

            auto clip_and_get_grad_norm = [&](float max_norm) -> float {
                double norm_sq = 0.0;
                for (const auto& p : trainable) {
                    auto grad = p->grad();
                    if (!grad) continue;
                    const float* g = grad->data<float>();
                    for (int64_t i = 0; i < grad->numel(); ++i) {
                        norm_sq += double(g[i]) * double(g[i]);
                    }
                }
                float norm = float(sqrt(norm_sq));
                if (max_norm > 0.0f && norm > max_norm) {
                    float scale = max_norm / (norm + 1e-6f);
                    for (const auto& p : trainable) {
                        auto grad = p->grad();
                        if (!grad) continue;
                        float* g = grad->data<float>();
                        for (int64_t i = 0; i < grad->numel(); ++i) {
                            g[i] *= scale;
                        }
                    }
                    norm = max_norm;
                }
                return norm;
            };

            auto count_nonzero_grad_tensors = [&]() -> int {
                int count = 0;
                for (const auto& p : trainable) {
                    auto grad = p->grad();
                    if (!grad) continue;
                    const float* g = grad->data<float>();
                    bool nonzero = false;
                    for (int64_t i = 0; i < grad->numel(); ++i) {
                        if (std::fabs(g[i]) > 1e-12f) {
                            nonzero = true;
                            break;
                        }
                    }
                    if (nonzero) count++;
                }
                return count;
            };

            auto run_four_dpo_forward = [&](const PreferenceBatch& batch) -> DPOForwardPack {
                DPOForwardPack pack;
                pack.policy_chosen_logits = model.forward(
                    batch.chosen_input_ids, batch.chosen_attention_mask);
                pack.policy_rejected_logits = model.forward(
                    batch.rejected_input_ids, batch.rejected_attention_mask);
                pack.ref_chosen_logits = ref_model.forward(
                    batch.chosen_input_ids, batch.chosen_attention_mask);
                pack.ref_rejected_logits = ref_model.forward(
                    batch.rejected_input_ids, batch.rejected_attention_mask);
                return pack;
            };

            auto run_batched_dpo_forward = [&](const PreferenceBatch& batch) -> DPOForwardPack {
                const int64_t batch_n = static_cast<int64_t>(batch.num_samples);
                if (batch_n <= 0) {
                    throw runtime_error("run_batched_dpo_forward: empty batch");
                }

                auto pair_input_ids = concat_batch_dim0(
                    batch.chosen_input_ids, batch.rejected_input_ids);
                auto pair_attention_mask = concat_batch_dim0(
                    batch.chosen_attention_mask, batch.rejected_attention_mask);

                auto policy_pair_logits = model.forward(pair_input_ids, pair_attention_mask);
                auto ref_pair_logits = ref_model.forward(pair_input_ids, pair_attention_mask);

                DPOForwardPack pack;
                pack.policy_chosen_logits = policy_pair_logits->slice(0, 0, batch_n);
                pack.policy_rejected_logits = policy_pair_logits->slice(0, batch_n, 2 * batch_n);
                pack.ref_chosen_logits = ref_pair_logits->slice(0, 0, batch_n);
                pack.ref_rejected_logits = ref_pair_logits->slice(0, batch_n, 2 * batch_n);
                return pack;
            };

            auto run_policy_forward_for_cached_ref = [&](const PreferenceBatch& batch) -> DPOForwardPack {
                const int64_t batch_n = static_cast<int64_t>(batch.num_samples);
                if (batch_n <= 0) {
                    throw runtime_error("run_policy_forward_for_cached_ref: empty batch");
                }

                DPOForwardPack pack;
                if (args.dpo_pair_forward) {
                    auto pair_input_ids = concat_batch_dim0(
                        batch.chosen_input_ids, batch.rejected_input_ids);
                    auto pair_attention_mask = concat_batch_dim0(
                        batch.chosen_attention_mask, batch.rejected_attention_mask);
                    auto policy_pair_logits = model.forward(pair_input_ids, pair_attention_mask);
                    pack.policy_chosen_logits = policy_pair_logits->slice(0, 0, batch_n);
                    pack.policy_rejected_logits = policy_pair_logits->slice(0, batch_n, 2 * batch_n);
                } else {
                    pack.policy_chosen_logits = model.forward(
                        batch.chosen_input_ids, batch.chosen_attention_mask);
                    pack.policy_rejected_logits = model.forward(
                        batch.rejected_input_ids, batch.rejected_attention_mask);
                }
                return pack;
            };

            auto run_dpo_forward = [&](const PreferenceBatch& batch) -> DPOForwardPack {
                if (args.dpo_pair_forward) {
                    return run_batched_dpo_forward(batch);
                }
                return run_four_dpo_forward(batch);
            };

            auto build_ref_logp_cache = [&](PreferenceDataset& dataset, const string& name) -> RefLogpCache {
                struct NoGradGuard {
                    NoGradGuard() { ops::autograd::Engine::instance().set_enabled(false); }
                    ~NoGradGuard() {
                        ops::autograd::Engine::instance().clear_graph();
                        ops::autograd::Engine::instance().set_enabled(true);
                    }
                } _nograd_guard;

                RefLogpCache cache;
                cache.enabled = true;
                cache.chosen.assign(dataset.num_samples(), 0.0f);
                cache.rejected.assign(dataset.num_samples(), 0.0f);

                dataset.reset_cursor();
                size_t cached = 0;
                while (true) {
                    auto batch = dataset.next_batch(false);
                    if (!batch.chosen_input_ids) break;

                    DPOForwardPack fwd;
                    if (args.dpo_pair_forward) {
                        const int64_t batch_n = static_cast<int64_t>(batch.num_samples);
                        auto pair_input_ids = concat_batch_dim0(
                            batch.chosen_input_ids, batch.rejected_input_ids);
                        auto pair_attention_mask = concat_batch_dim0(
                            batch.chosen_attention_mask, batch.rejected_attention_mask);
                        auto ref_pair_logits = ref_model.forward(pair_input_ids, pair_attention_mask);
                        fwd.ref_chosen_logits = ref_pair_logits->slice(0, 0, batch_n);
                        fwd.ref_rejected_logits = ref_pair_logits->slice(0, batch_n, 2 * batch_n);
                    } else {
                        fwd.ref_chosen_logits = ref_model.forward(
                            batch.chosen_input_ids, batch.chosen_attention_mask);
                        fwd.ref_rejected_logits = ref_model.forward(
                            batch.rejected_input_ids, batch.rejected_attention_mask);
                    }

                    auto chosen_logps = sequence_logprob_masked(
                        fwd.ref_chosen_logits, batch.chosen_input_ids, batch.chosen_response_mask);
                    auto rejected_logps = sequence_logprob_masked(
                        fwd.ref_rejected_logits, batch.rejected_input_ids, batch.rejected_response_mask);

                    if (chosen_logps.size() != batch.sample_indices.size() ||
                        rejected_logps.size() != batch.sample_indices.size()) {
                        throw runtime_error("build_ref_logp_cache: logp/index size mismatch");
                    }
                    for (size_t i = 0; i < batch.sample_indices.size(); ++i) {
                        const size_t sample_idx = batch.sample_indices[i];
                        cache.chosen[sample_idx] = chosen_logps[i];
                        cache.rejected[sample_idx] = rejected_logps[i];
                        cached++;
                    }
                }
                dataset.reset_cursor();
                cout << "  ✓ Cached " << cached << " reference logp pairs for " << name << endl;
                return cache;
            };

            auto gather_cached_ref_logps = [](const PreferenceBatch& batch,
                                              const RefLogpCache& cache,
                                              std::vector<float>& chosen,
                                              std::vector<float>& rejected) {
                if (!cache.enabled) {
                    throw runtime_error("gather_cached_ref_logps: cache disabled");
                }
                chosen.clear();
                rejected.clear();
                chosen.reserve(batch.sample_indices.size());
                rejected.reserve(batch.sample_indices.size());
                for (size_t sample_idx : batch.sample_indices) {
                    if (sample_idx >= cache.chosen.size() || sample_idx >= cache.rejected.size()) {
                        throw runtime_error("gather_cached_ref_logps: sample index out of range");
                    }
                    chosen.push_back(cache.chosen[sample_idx]);
                    rejected.push_back(cache.rejected[sample_idx]);
                }
            };

            RefLogpCache train_ref_cache;
            RefLogpCache valid_ref_cache;
            if (args.cache_ref_logps) {
                cout << "\n[RefCache] Precompute frozen reference logps..." << endl;
                auto cache_t0 = std::chrono::steady_clock::now();
                train_ref_cache = build_ref_logp_cache(train_pref_dataset, "train");
                valid_ref_cache = build_ref_logp_cache(valid_pref_dataset, "valid");
                auto cache_t1 = std::chrono::steady_clock::now();
                double cache_seconds = std::chrono::duration<double>(cache_t1 - cache_t0).count();
                cout << "[Timing] ref_cache_seconds " << fixed << setprecision(4)
                     << cache_seconds << endl;
            }

            auto evaluate_valid_dpo = [&]() -> DPOMetrics {
                struct NoGradGuard {
                    NoGradGuard() { ops::autograd::Engine::instance().set_enabled(false); }
                    ~NoGradGuard() {
                        ops::autograd::Engine::instance().clear_graph();
                        ops::autograd::Engine::instance().set_enabled(true);
                    }
                } _nograd_guard;

                valid_pref_dataset.reset_cursor();
                double loss_sum = 0.0;
                double chosen_reward_sum = 0.0;
                double rejected_reward_sum = 0.0;
                double reward_margin_sum = 0.0;
                double reward_acc_sum = 0.0;
                int64_t sample_sum = 0;
                int processed = 0;

                while (processed < args.eval_batches) {
                    auto batch = valid_pref_dataset.next_batch(false);
                    if (!batch.chosen_input_ids) break;

                    DPOMetrics metrics;
                    if (args.cache_ref_logps) {
                        auto fwd = run_policy_forward_for_cached_ref(batch);
                        std::vector<float> ref_chosen_logps;
                        std::vector<float> ref_rejected_logps;
                        gather_cached_ref_logps(
                            batch, valid_ref_cache, ref_chosen_logps, ref_rejected_logps);
                        metrics = compute_dpo_metrics_from_policy_logits_and_ref_logps(
                            fwd.policy_chosen_logits,
                            fwd.policy_rejected_logits,
                            ref_chosen_logps,
                            ref_rejected_logps,
                            batch.chosen_input_ids,
                            batch.rejected_input_ids,
                            batch.chosen_response_mask,
                            batch.rejected_response_mask,
                            args.dpo_beta);
                    } else {
                        auto fwd = run_dpo_forward(batch);
                        metrics = compute_dpo_metrics_from_logits(
                            fwd.policy_chosen_logits,
                            fwd.policy_rejected_logits,
                            fwd.ref_chosen_logits,
                            fwd.ref_rejected_logits,
                            batch.chosen_input_ids,
                            batch.rejected_input_ids,
                            batch.chosen_response_mask,
                            batch.rejected_response_mask,
                            args.dpo_beta);
                    }

                    const int64_t batch_samples = static_cast<int64_t>(batch.num_samples);
                    loss_sum += static_cast<double>(metrics.loss) * static_cast<double>(batch_samples);
                    chosen_reward_sum += static_cast<double>(metrics.chosen_reward) * static_cast<double>(batch_samples);
                    rejected_reward_sum += static_cast<double>(metrics.rejected_reward) * static_cast<double>(batch_samples);
                    reward_margin_sum += static_cast<double>(metrics.reward_margin) * static_cast<double>(batch_samples);
                    reward_acc_sum += static_cast<double>(metrics.reward_accuracy) * static_cast<double>(batch_samples);
                    sample_sum += batch_samples;
                    processed++;
                }

                if (sample_sum == 0) {
                    DPOMetrics empty;
                    empty.loss = 1e9f;
                    return empty;
                }

                const double inv_n = 1.0 / static_cast<double>(sample_sum);
                DPOMetrics out;
                out.loss = static_cast<float>(loss_sum * inv_n);
                out.chosen_reward = static_cast<float>(chosen_reward_sum * inv_n);
                out.rejected_reward = static_cast<float>(rejected_reward_sum * inv_n);
                out.reward_margin = static_cast<float>(reward_margin_sum * inv_n);
                out.reward_accuracy = static_cast<float>(reward_acc_sum * inv_n);
                return out;
            };

            cout << "\n[5/6] Start training..." << endl;
            cout << "========================================\n" << endl;

            double ema_loss = 0.0;
            bool ema_initialized = false;
            int64_t total_tokens = 0;
            double rss_peak_mb = 0.0;
            DPOMetrics last_train_metrics;
            auto train_t0 = std::chrono::steady_clock::now();

            for (int step = start_step; step < args.steps; ++step) {
                const int cur_epoch = step / steps_per_epoch + 1;
                const int step_in_epoch = step % steps_per_epoch + 1;

                double accum_loss = 0.0;
                int64_t accum_tokens = 0;
                auto rss_to_mb = []() -> double {
                    size_t bytes = MemoryMonitor::get_system_memory_usage();
                    return static_cast<double>(bytes) / (1024.0 * 1024.0);
                };
                double rss_pre_max = 0.0, rss_fwd_max = 0.0, rss_bwd_max = 0.0, rss_opt_max = 0.0, rss_post_max = 0.0;
                rss_pre_max = std::max(rss_pre_max, rss_to_mb());

                for (int acc = 0; acc < accum; ++acc) {
                    auto batch = train_pref_dataset.next_batch();
                    auto fwd = args.cache_ref_logps
                        ? run_policy_forward_for_cached_ref(batch)
                        : run_dpo_forward(batch);
                    rss_fwd_max = std::max(rss_fwd_max, rss_to_mb());

                    TensorPtr loss;
                    DPOMetrics train_metrics;
                    if (args.cache_ref_logps) {
                        std::vector<float> ref_chosen_logps;
                        std::vector<float> ref_rejected_logps;
                        gather_cached_ref_logps(
                            batch, train_ref_cache, ref_chosen_logps, ref_rejected_logps);
                        loss = dpo_loss_with_ref_logps(
                            fwd.policy_chosen_logits,
                            fwd.policy_rejected_logits,
                            ref_chosen_logps,
                            ref_rejected_logps,
                            batch.chosen_input_ids,
                            batch.rejected_input_ids,
                            batch.chosen_response_mask,
                            batch.rejected_response_mask,
                            args.dpo_beta);
                        train_metrics = compute_dpo_metrics_from_policy_logits_and_ref_logps(
                            fwd.policy_chosen_logits,
                            fwd.policy_rejected_logits,
                            ref_chosen_logps,
                            ref_rejected_logps,
                            batch.chosen_input_ids,
                            batch.rejected_input_ids,
                            batch.chosen_response_mask,
                            batch.rejected_response_mask,
                            args.dpo_beta);
                    } else {
                        loss = dpo_loss(
                            fwd.policy_chosen_logits,
                            fwd.policy_rejected_logits,
                            fwd.ref_chosen_logits,
                            fwd.ref_rejected_logits,
                            batch.chosen_input_ids,
                            batch.rejected_input_ids,
                            batch.chosen_response_mask,
                            batch.rejected_response_mask,
                            args.dpo_beta);
                        train_metrics = compute_dpo_metrics_from_logits(
                            fwd.policy_chosen_logits,
                            fwd.policy_rejected_logits,
                            fwd.ref_chosen_logits,
                            fwd.ref_rejected_logits,
                            batch.chosen_input_ids,
                            batch.rejected_input_ids,
                            batch.chosen_response_mask,
                            batch.rejected_response_mask,
                            args.dpo_beta);
                    }
                    last_train_metrics = train_metrics;
                    if (step == start_step && acc == 0) {
                        if (!fwd.policy_chosen_logits->requires_grad() ||
                            !fwd.policy_rejected_logits->requires_grad() ||
                            !loss->requires_grad()) {
                            throw runtime_error(
                                "DPO debug: policy logits or loss are not attached to autograd");
                        }
                    }
                    float loss_val = loss->data<float>()[0];
                    accum_loss += loss_val;
                    accum_tokens += count_response_tokens(batch.chosen_response_mask);
                    accum_tokens += count_response_tokens(batch.rejected_response_mask);

                    float inv_accum = 1.0f / float(accum);
                    auto scaled_loss = mul(loss, inv_accum);
                    scaled_loss->backward();
                    rss_bwd_max = std::max(rss_bwd_max, rss_to_mb());
                }

                float grad_norm = clip_and_get_grad_norm(args.clip_grad_norm);
                int nonzero_grad_tensors = count_nonzero_grad_tensors();
                float cur_lr = lr_schedule(step);
                optimizer.set_learning_rate(cur_lr);

                vector<TensorPtr> grads;
                grads.reserve(trainable.size());
                for (const auto& p : trainable) {
                    grads.push_back(p->grad());
                }
                optimizer.step(trainable, grads);
                for (auto& p : trainable) {
                    p->zero_grad();
                }
                rss_opt_max = std::max(rss_opt_max, rss_to_mb());

                if (((step + 1) % 50) == 0) {
                    MemoryManager::instance().force_cleanup();
                }
                if (((step + 1) % 100) == 0) {
                    MemoryManager::instance().clear_unused_memory();
                }
                MemoryManager::instance().cleanup_dead_references();
                rss_post_max = std::max(rss_post_max, rss_to_mb());
                double rss_step_max = std::max({rss_pre_max, rss_fwd_max, rss_bwd_max, rss_opt_max, rss_post_max});
                rss_peak_mb = std::max(rss_peak_mb, rss_step_max);

                float avg_loss = float(accum_loss / double(accum));
                total_tokens += accum_tokens;
                if (!ema_initialized) {
                    ema_loss = avg_loss;
                    ema_initialized = true;
                } else {
                    double beta = max(0.0, min(0.9999, double(args.ema_beta)));
                    ema_loss = beta * ema_loss + (1.0 - beta) * avg_loss;
                }

                if ((step + 1) % max(1, args.log_interval) == 0) {
                    cout << "[Train] epoch " << cur_epoch << "/" << args.epochs
                         << " | step " << step_in_epoch << "/" << steps_per_epoch
                         << " (global " << (step + 1) << "/" << args.steps << ")"
                         << " | lr " << fixed << setprecision(6) << cur_lr
                         << " | dpo_loss " << setprecision(4) << avg_loss
                         << " | reward_acc " << setprecision(4) << last_train_metrics.reward_accuracy
                         << " | reward_margin " << setprecision(4) << last_train_metrics.reward_margin
                         << " | grad_norm " << setprecision(6) << grad_norm
                         << " | nonzero_grad_tensors " << nonzero_grad_tensors
                         << " | resp_tokens " << accum_tokens
                         << " | RSS(pre/fwd/bwd/opt/post)="
                         << fixed << setprecision(1)
                         << rss_pre_max << "/" << rss_fwd_max << "/"
                         << rss_bwd_max << "/" << rss_opt_max << "/"
                         << rss_post_max << " MB"
                         << " | step_max=" << rss_step_max << " MB"
                         << " | peak=" << rss_peak_mb << " MB"
                         << endl;
                    if (!args.metrics_out.empty()) {
                        stringstream ss;
                        ss << "{\"step\":" << (step + 1)
                           << ",\"total_steps\":" << args.steps
                           << ",\"epoch\":" << cur_epoch
                           << ",\"loss\":" << avg_loss
                           << ",\"lr\":" << cur_lr
                           << ",\"tokens\":" << accum_tokens
                           << ",\"total_tokens\":" << total_tokens
                           << ",\"reward_accuracy\":" << last_train_metrics.reward_accuracy
                           << ",\"reward_margin\":" << last_train_metrics.reward_margin
                           << ",\"chosen_reward\":" << last_train_metrics.chosen_reward
                           << ",\"rejected_reward\":" << last_train_metrics.rejected_reward
                           << ",\"grad_norm\":" << grad_norm
                           << ",\"rss_mb\":" << rss_step_max
                           << ",\"peak_rss_mb\":" << rss_peak_mb
                           << ",\"source\":\"native_dpo\""
                           << "}";
                        append_jsonl(args.metrics_out, ss.str());
                    }
                }

                if ((step + 1) % 100 == 0) {
                    MemoryManager::instance().print_memory_stats();
                }

                if (args.eval_interval > 0 && (step + 1) % args.eval_interval == 0) {
                    auto valid_metrics = evaluate_valid_dpo();
                    cout << "\n[Eval] epoch " << cur_epoch
                         << " | step " << (step + 1)
                         << " | loss " << fixed << setprecision(4) << valid_metrics.loss
                         << " | reward_acc " << setprecision(4) << valid_metrics.reward_accuracy
                         << " | reward_margin " << setprecision(4) << valid_metrics.reward_margin
                         << " | ema_loss " << setprecision(4) << float(ema_loss)
                         << " | total_tokens " << total_tokens
                         << "\n" << endl;

                    if (!args.eval_out.empty()) {
                        stringstream ss;
                        ss << "{\"step\":" << (step + 1)
                           << ",\"epoch\":" << cur_epoch
                           << ",\"loss\":" << valid_metrics.loss
                           << ",\"reward_accuracy\":" << valid_metrics.reward_accuracy
                           << ",\"reward_margin\":" << valid_metrics.reward_margin
                           << ",\"chosen_reward\":" << valid_metrics.chosen_reward
                           << ",\"rejected_reward\":" << valid_metrics.rejected_reward
                           << ",\"ema_loss\":" << float(ema_loss)
                           << ",\"total_tokens\":" << total_tokens
                           << "}";
                        append_jsonl(args.eval_out, ss.str());
                    }
                }

                if (args.save_every > 0 && (step + 1) % args.save_every == 0 && !args.lora_out.empty()) {
                    string ckpt_path = make_checkpoint_path(args.lora_out, step + 1);
                    cout << "\n[Checkpoint] saving to " << ckpt_path << endl;
                    if (has_active_lora_spec) {
                        LoraSaver::save_safetensors(ckpt_path, model, active_lora_spec);
                        cout << "  ✓ Checkpoint saved\n" << endl;
                    } else {
                        cout << "  ⚠️ No active LoRA config detected, skip saving" << endl;
                    }
                }
            }
            auto train_t1 = std::chrono::steady_clock::now();
            double train_seconds = std::chrono::duration<double>(train_t1 - train_t0).count();
            cout << "[Timing] train_loop_seconds " << fixed << setprecision(4)
                 << train_seconds << endl;

            cout << "\n[6/6] Save final LoRA weights..." << endl;
            if (!args.lora_out.empty()) {
                if (has_active_lora_spec) {
                    LoraSaver::save_safetensors(args.lora_out, model, active_lora_spec);
                    cout << "  ✓ LoRA saved to: " << args.lora_out << endl;
                } else {
                    cout << "  ⚠️ No LoRA weights found to save, skipped" << endl;
                }
            }

            cout << "\n========================================" << endl;
            cout << "✅ Training complete!" << endl;
            cout << "  Total steps: " << args.steps << endl;
            cout << "  Total response tokens: " << total_tokens << endl;
            cout << "  Final EMA loss: " << fixed << setprecision(4) << float(ema_loss) << endl;
            cout << "========================================\n" << endl;
            return 0;
        }

        WT2Config data_cfg;
        // Prefer JSONL when provided (masked labels applied only on answer spans)
        if (!args.jsonl_train.empty() || !args.jsonl_valid.empty() || !args.jsonl_test.empty()) {
            data_cfg.jsonl_train = args.jsonl_train;
            data_cfg.jsonl_valid = args.jsonl_valid;
            data_cfg.jsonl_test  = args.jsonl_test;
            // Disable streaming scan in JSONL mode
            data_cfg.streaming_mode = false;
            // Enable shuffling for training
            data_cfg.shuffle_train = true;
            // Ignore other text paths when JSONL is present
        } else {
            data_cfg.train_path = args.data_dir + "/wiki.train.raw";
            data_cfg.valid_path = args.data_dir + "/wiki.valid.raw";
            data_cfg.test_path  = args.data_dir + "/wiki.test.raw";
        }
        data_cfg.seq_len = args.seq_len;
        data_cfg.stride  = -1;
        data_cfg.eos_id  = 50256;
        data_cfg.seed = args.seed;
        
        WikiText2Dataset train_dataset(data_cfg, &tokenizer);
        train_dataset.load(Split::Train);
        
        WT2Config valid_cfg = data_cfg;
        valid_cfg.drop_last = false;
        WikiText2Dataset valid_dataset(valid_cfg, &tokenizer);
        valid_dataset.load(Split::Valid);
        
        cout << "  ✓ Train set: " << train_dataset.num_sequences() << " sequences" << endl;
        cout << "  ✓ Valid set: " << valid_dataset.num_sequences() << " sequences" << endl;

        // Derive the training schedule
        const int64_t train_seqs = train_dataset.num_sequences();
        const int64_t micro_bs = args.batch_size;
        const int64_t accum = max(1, args.grad_accum_steps);
        const int64_t steps_per_epoch = (train_seqs + micro_bs * accum - 1) / (micro_bs * accum);
        
        if (args.epochs > 0) {
            args.steps = steps_per_epoch * args.epochs;
        }
        
        cout << "\n[Training plan]" << endl;
        cout << "  epochs         : " << args.epochs << endl;
        cout << "  steps_per_epoch: " << steps_per_epoch << endl;
        cout << "  total_steps    : " << args.steps << endl;
        cout << "  Effective batch: " << (micro_bs * accum) << " (micro=" << micro_bs 
             << " × accum=" << accum << ")" << endl;

        // 4) Optimizer
        cout << "\n[4/6] Init optimizer..." << endl;
        AdamConfig opt_cfg;
        opt_cfg.learning_rate = args.lr;
        opt_cfg.beta1 = 0.9f;
        opt_cfg.beta2 = 0.999f;
        opt_cfg.epsilon = 1e-8f;
        opt_cfg.weight_decay = args.weight_decay;
        opt_cfg.clip_grad_norm = args.clip_grad_norm;
        Adam optimizer(opt_cfg);
        cout << "  ✓ Adam optimizer ready" << endl;

        // Helper: count valid (masked) tokens in attention mask
        auto count_tokens = [](const TensorPtr& mask) -> int64_t {
            if (!mask) return 0;
            const float* ptr = mask->data<float>();
            int64_t total = 0;
            for (int64_t i = 0; i < mask->numel(); ++i) {
                if (ptr[i] > 0.5f) total++;
            }
            return total;
        };

        // Learning-rate schedule: linear warmup followed by cosine decay
        auto lr_schedule = [&](int step) -> float {
            const float base_lr = args.lr;
            const int T = args.steps;
            const int W = max(0, args.warmup_steps);
            
            if (step < W) {
                // Linear warmup
                return base_lr * (float(step + 1) / float(max(1, W)));
            }
            
            // Cosine decay down to 10% of the base LR
            const int s = step - W;
            const int d = max(1, T - W);
            const float min_lr = 0.1f * base_lr;
            // Avoid portability issues with non-standard M_PI macro
            const float PI = acosf(-1.0f);
            const float cosv = 0.5f * (1.0f + cos(PI * float(s) / float(d)));
            return min_lr + (base_lr - min_lr) * cosv;
        };

        // Gradient clipping and L2 norm computation
        auto clip_and_get_grad_norm = [&](float max_norm) -> float {
            double norm_sq = 0.0;
            for (const auto& p : trainable) {
                auto grad = p->grad();
                if (!grad) continue;
                const float* g = grad->data<float>();
                for (int64_t i = 0; i < grad->numel(); ++i) {
                    norm_sq += double(g[i]) * double(g[i]);
                }
            }
            float norm = float(sqrt(norm_sq));
            
            if (max_norm > 0.0f && norm > max_norm) {
                float scale = max_norm / (norm + 1e-6f);
                for (const auto& p : trainable) {
                    auto grad = p->grad();
                    if (!grad) continue;
                    float* g = grad->data<float>();
                    for (int64_t i = 0; i < grad->numel(); ++i) {
                        g[i] *= scale;
                    }
                }
                norm = max_norm;
            }
            return norm;
        };

        // Validation pass
        auto evaluate_valid = [&]() -> float {
            // Disable autograd during evaluation; restore state and clear graph on exit
            struct NoGradGuard {
                NoGradGuard() { ops::autograd::Engine::instance().set_enabled(false); }
                ~NoGradGuard() {
                    ops::autograd::Engine::instance().clear_graph();
                    ops::autograd::Engine::instance().set_enabled(true);
                }
            } _nograd_guard;
            
            valid_dataset.reset_cursor();
            double loss_sum = 0.0;
            int64_t token_sum = 0;
            int processed = 0;
            
            while (processed < args.eval_batches) {
                auto batch = valid_dataset.next_batch(args.eval_batch_size, false);
                if (!batch.input_ids) break;
                
                auto logits = model.forward(batch.input_ids, batch.attention_mask);
                auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
                float loss_val = loss->data<float>()[0];
                int64_t tokens = count_tokens(batch.attention_mask);
                
                loss_sum += double(loss_val) * double(tokens);
                token_sum += tokens;
                processed++;
            }
            
            if (token_sum == 0) return 1e9f;
            float mean_loss = float(loss_sum / double(token_sum));
            return perplexity_from_loss(mean_loss);
        };

        // 5) Training loop
        cout << "\n[5/6] Start training..." << endl;
        cout << "========================================\n" << endl;
        
        double ema_loss = 0.0;
        bool ema_initialized = false;
        int64_t total_tokens = 0;
        // Track global RSS peak across steps
        double rss_peak_mb = 0.0;

        for (int step = start_step; step < args.steps; ++step) {
            const int cur_epoch = step / steps_per_epoch + 1;
            const int step_in_epoch = step % steps_per_epoch + 1;
            
            // Gradient accumulation
            double accum_loss = 0.0;
            int64_t accum_tokens = 0;
            // Per-step RSS trackers
            auto rss_to_mb = []() -> double {
                size_t bytes = MemoryMonitor::get_system_memory_usage();
                return static_cast<double>(bytes) / (1024.0 * 1024.0);
            };
            double rss_pre_max = 0.0, rss_fwd_max = 0.0, rss_bwd_max = 0.0, rss_opt_max = 0.0, rss_post_max = 0.0;
            rss_pre_max = std::max(rss_pre_max, rss_to_mb());
            
            for (int acc = 0; acc < accum; ++acc) {
                auto batch = train_dataset.next_batch(micro_bs);
                auto logits = model.forward(batch.input_ids, batch.attention_mask);
                rss_fwd_max = std::max(rss_fwd_max, rss_to_mb());
                auto loss = lm_cross_entropy(logits, batch.labels, -100, "mean");
                float loss_val = loss->data<float>()[0];
                
                accum_loss += loss_val;
                accum_tokens += count_tokens(batch.attention_mask);
                // Strategy A: scale loss by 1/accum at each micro step, then backprop
                {
                    float inv_accum = 1.0f / float(accum);
                    auto scaled_loss = mul(loss, inv_accum);
                    scaled_loss->backward();
                }
                rss_bwd_max = std::max(rss_bwd_max, rss_to_mb());
            }
            
            // Gradient clipping
            float grad_norm = clip_and_get_grad_norm(args.clip_grad_norm);
            
            // Update learning rate
            float cur_lr = lr_schedule(step);
            optimizer.set_learning_rate(cur_lr);
            
            // Optimizer step and zero grads
            vector<TensorPtr> grads;
            grads.reserve(trainable.size());
            for (const auto& p : trainable) {
                grads.push_back(p->grad());
            }
            optimizer.step(trainable, grads);
            for (auto& p : trainable) {
                p->zero_grad();
            }
            rss_opt_max = std::max(rss_opt_max, rss_to_mb());
            
            // Periodic light cleanup of free blocks (default every 50 steps) to reduce overhead
            if (((step + 1) % 50) == 0) {
                MemoryManager::instance().force_cleanup();
            }
            // Aggressive release of free memory back to OS every 100 steps to prevent RSS drift
            if (((step + 1) % 100) == 0) {
                MemoryManager::instance().clear_unused_memory();
            }
            // Also do a light cleanup each step to stabilize post-step RSS observation
            MemoryManager::instance().cleanup_dead_references();
            rss_post_max = std::max(rss_post_max, rss_to_mb());
            double rss_step_max = std::max({rss_pre_max, rss_fwd_max, rss_bwd_max, rss_opt_max, rss_post_max});
            rss_peak_mb = std::max(rss_peak_mb, rss_step_max);
            
            // Update running statistics
            float avg_loss = float(accum_loss / double(accum));
            float ppl = perplexity_from_loss(avg_loss);
            total_tokens += accum_tokens;
            
            // EMA
            if (!ema_initialized) {
                ema_loss = avg_loss;
                ema_initialized = true;
            } else {
                double beta = max(0.0, min(0.9999, double(args.ema_beta)));
                ema_loss = beta * ema_loss + (1.0 - beta) * avg_loss;
            }
            
            // Logging
            if ((step + 1) % max(1, args.log_interval) == 0) {
                cout << "[Train] epoch " << cur_epoch << "/" << args.epochs
                     << " | step " << step_in_epoch << "/" << steps_per_epoch
                     << " (global " << (step + 1) << "/" << args.steps << ")"
                     << " | lr " << fixed << setprecision(6) << cur_lr
                     << " | loss " << setprecision(4) << avg_loss
                     << " | ppl " << setprecision(2) << ppl
                     << " | grad_norm " << setprecision(3) << grad_norm
                     << " | tokens " << accum_tokens
                     << " | RSS(pre/fwd/bwd/opt/post)="
                     << fixed << setprecision(1)
                     << rss_pre_max << "/" << rss_fwd_max << "/"
                     << rss_bwd_max << "/" << rss_opt_max << "/"
                     << rss_post_max << " MB"
                     << " | step_max=" << rss_step_max << " MB"
                     << " | peak=" << rss_peak_mb << " MB"
                     << endl;
                if (!args.metrics_out.empty()) {
                    stringstream ss;
                    ss << "{\"step\":" << (step + 1)
                       << ",\"total_steps\":" << args.steps
                       << ",\"epoch\":" << cur_epoch
                       << ",\"loss\":" << avg_loss
                       << ",\"ppl\":" << ppl
                       << ",\"lr\":" << cur_lr
                       << ",\"tokens\":" << accum_tokens
                       << ",\"total_tokens\":" << total_tokens
                       << ",\"grad_norm\":" << grad_norm
                       << ",\"rss_mb\":" << rss_step_max
                       << ",\"peak_rss_mb\":" << rss_peak_mb
                       << ",\"source\":\"native_sft\""
                       << "}";
                    append_jsonl(args.metrics_out, ss.str());
                }
            }
            
            // Every 100 steps, print memory stats (monitor total_allocated / total_in_use)
            if ((step + 1) % 100 == 0) {
                MemoryManager::instance().print_memory_stats();
            }
            
            // Periodic evaluation
            if (args.eval_interval > 0 && (step + 1) % args.eval_interval == 0) {
                float valid_ppl = evaluate_valid();
                cout << "\n[Eval] epoch " << cur_epoch 
                     << " | step " << (step + 1)
                     << " | valid_ppl " << fixed << setprecision(2) << valid_ppl
                     << " | ema_loss " << setprecision(4) << float(ema_loss)
                     << " | total_tokens " << total_tokens
                     << "\n" << endl;
                
                // Append metrics to JSONL
                if (!args.eval_out.empty()) {
                    stringstream ss;
                    ss << "{\"step\":" << (step + 1)
                       << ",\"epoch\":" << cur_epoch
                       << ",\"valid_ppl\":" << valid_ppl
                       << ",\"ema_loss\":" << float(ema_loss)
                       << ",\"total_tokens\":" << total_tokens
                       << "}";
                    append_jsonl(args.eval_out, ss.str());
                }
            }
            
            // Periodic checkpointing
            if (args.save_every > 0 && (step + 1) % args.save_every == 0 && !args.lora_out.empty()) {
                string ckpt_path = make_checkpoint_path(args.lora_out, step + 1);
                cout << "\n[Checkpoint] saving to " << ckpt_path << endl;
                if (has_active_lora_spec) {
                    LoraSaver::save_safetensors(ckpt_path, model, active_lora_spec);
                    cout << "  ✓ Checkpoint saved\n" << endl;
                } else {
                    cout << "  ⚠️ No active LoRA config detected, skip saving" << endl;
                }
            }
        }

        // 6) Save final LoRA adapter
        cout << "\n[6/6] Save final LoRA weights..." << endl;
        if (!args.lora_out.empty()) {
            if (has_active_lora_spec) {
                LoraSaver::save_safetensors(args.lora_out, model, active_lora_spec);
                cout << "  ✓ LoRA saved to: " << args.lora_out << endl;
            } else {
                cout << "  ⚠️ No LoRA weights found to save, skipped" << endl;
            }
        }

        cout << "\n========================================" << endl;
        cout << "✅ Training complete!" << endl;
        cout << "  Total steps: " << args.steps << endl;
        cout << "  Total tokens: " << total_tokens << endl;
        cout << "  Final EMA loss: " << fixed << setprecision(4) << float(ema_loss) << endl;
        cout << "========================================\n" << endl;
        
        return 0;

    } catch (const exception& e) {
        cerr << "\n❌ Exception: " << e.what() << endl;
        return 1;
    }
}
