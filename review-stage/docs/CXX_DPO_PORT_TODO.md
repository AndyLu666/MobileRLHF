# C++ DPO Port TODO

Date: `2026-06-01`

## 2026-06-02 checkpoint
- Native DPO smoke now has a standard build target: `gpt2_small_lora_finetune` builds `test_dpo_native_smoke`.
- Repeatable validation path is now:
  - `cmake --build . --target test_dpo_native_smoke train -j2`
  - `./test_dpo_native_smoke`
  - `./train --train_mode dpo ...`
- Real-MMLU tokenizer retention is now aligned with PyTorch/HF at `seq_len=384`:
  - native keeps `222/480` train and `117/120` valid
- BLAS / Accelerate full real-MMLU native smoke is now practical:
  - build dir: `gpt2_small_lora_finetune/build_release_blas`
  - one-command entrypoint: `review-stage/scripts/run_native_real_mmlu_full_smoke_blas.sh`
  - observed result:
    - `dpo_loss=0.6931`
    - `grad_norm=1.000000`
    - `nonzero_grad_tensors=24`
    - wall time about `11.78s`
- A repeatable short real-data native smoke now exists:
  - generator: `review-stage/scripts/build_real_mmlu_short_pref_subset.py`
  - outputs:
    - `review-stage/data/real_mmlu_pref_short_train.jsonl`
    - `review-stage/data/real_mmlu_pref_short_eval.jsonl`
  - one-command entrypoint: `review-stage/scripts/run_native_real_mmlu_short_smoke.sh`
  - smoke config:
    - `seq_len=128`
    - `train=16`
    - `eval=8`
- Remaining work under issue `#16` is now closer to:
  - treat the BLAS full real-MMLU 1-step smoke as the primary repeatable validation target
  - keep the short real-data smoke as the faster fallback path

## Goal
- Final goal: run `DPO` training through the repo's native `C++` / mobile-style training path, not only through `pytorch_alignment/`.

## Current State
- `PyTorch` path already supports `DPO`:
  - `pytorch_alignment/dpo_utils.py`
  - `pytorch_alignment/gpt2_lora_finetune.py`
  - `pytorch_alignment/gemma_lora_finetune.py`
  - `pytorch_alignment/qwen_lora_finetune.py`
- Native `C++` path currently supports `SFT`-style masked LM training:
  - `gpt2_small_lora_finetune/src/main.cpp`
  - `gpt2_medium_lora_finetune/src/main.cpp`
  - `operator/finetune_ops/data/mmlu_dataset.*`
- There is no native `C++` preference-pair dataset, no DPO loss, and no frozen reference-model scoring path yet.

## Work Breakdown

### 1. Add native preference-pair dataset support
- Create a `PreferenceDataset` in `operator/finetune_ops/data/`
- Input format:
  - JSONL rows with `prompt`, `chosen`, `rejected`
- Required outputs per sample:
  - `chosen_input_ids`
  - `chosen_attention_mask`
  - `chosen_response_mask`
  - `rejected_input_ids`
  - `rejected_attention_mask`
  - `rejected_response_mask`
- Match the tokenization/truncation behavior of `pytorch_alignment/dpo_utils.py`
- Track kept/skipped sample counts after truncation

### 2. Port DPO loss and metrics to C++
- Implement sequence log-prob computation over only response tokens
- Implement DPO core math:
  - policy chosen logprob
  - policy rejected logprob
  - reference chosen logprob
  - reference rejected logprob
  - `-log(sigmoid(beta * ((pi_c-pi_r) - (ref_c-ref_r))))`
- Log native C++ metrics:
  - `loss`
  - `reward_accuracy`
  - `reward_margin`
  - `chosen_reward`
  - `rejected_reward`

### 3. Add frozen reference-model support
- Load a second GPT-2 model in inference-only mode
- Ensure reference parameters do not receive gradients
- Reuse the same tokenizer and prompt formatting as the policy model
- Check memory cost carefully, because this is the main on-device risk

### 4. Extend GPT-2 C++ trainer CLI
- Add CLI flags to `gpt2_small_lora_finetune/src/main.cpp`
  - `--train_mode sft|dpo`
  - `--preference_jsonl`
  - `--preference_valid_jsonl`
  - `--dpo_beta`
- Preserve the existing SFT path
- Route batches to either:
  - masked LM loss
  - DPO loss
- Save logs in a format the Android visualizer can still read

### 5. Add minimal validation assets
- Add a tiny synthetic preference JSONL for native C++ smoke tests
- Add one end-to-end command for `gpt2_small_lora_finetune`
- Verify:
  - finite loss
  - non-zero LoRA gradients on policy
  - zero gradients on reference
  - held-out reward metrics improve over steps

### 6. Add real-data validation
- Reuse:
  - `review-stage/data/real_mmlu_pref_train.jsonl`
  - `review-stage/data/real_mmlu_pref_eval.jsonl`
- Start with the same practical configuration learned from PyTorch:
  - `seq_len=384`
  - limited-step smoke first
- Compare native C++ results against the PyTorch DPO baseline

## Suggested Execution Order
1. Native preference dataset
2. DPO loss + metrics functions
3. Frozen reference-model inference path
4. GPT-2 small C++ CLI integration
5. Tiny synthetic smoke test
6. Real-MMLU validation
7. Android / on-device run

## Files Most Likely To Change
- `gpt2_small_lora_finetune/src/main.cpp`
- `gpt2_medium_lora_finetune/src/main.cpp`
- `operator/finetune_ops/data/` (new preference dataset files)
- `operator/finetune_ops/core/` (new DPO helpers or loss helpers)
- `android-visualizer/` only if log format changes

## Main Risks
- Reference model doubles forward cost and increases memory pressure
- Long prompts still reduce usable sample count
- C++ path must match PyTorch token masking exactly, or comparisons will be misleading

## Definition Of Done
- A native `C++` GPT-2 command can run `--train_mode dpo`
- It consumes preference JSONL directly
- It logs DPO metrics
- It passes a tiny smoke test
- It runs on the same real-MMLU preference split used in PyTorch
