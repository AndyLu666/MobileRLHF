# Native DPO Checkpoint

Date: `2026-06-02`

## Status
- Native GPT-2 small `C++` DPO path is now functional beyond toy-loss validation.
- A standard native smoke target now exists and builds under `gpt2_small_lora_finetune`.
- The major real-MMLU blocker was tokenizer parity, not the DPO loss.

## What Landed
- `gpt2_small_lora_finetune/CMakeLists.txt`
  - added build target `test_dpo_native_smoke`
- `operator/finetune_ops/core/tokenizer_bpe.cpp`
  - fixed GPT-2 encode order to match HuggingFace flow:
    - raw-text pre-tokenization first
    - byte-to-unicode conversion per token
    - BPE merges after that
  - replaced the older ASCII regex approximation with a manual Unicode-aware GPT-2-like splitter
- `operator/finetune_ops/data/preference_dataset.cpp`
  - hardened JSON string extraction for escaped content
- `gpt2_small_lora_finetune/README.md`
  - documented native DPO smoke commands

## Verified Results

### 1. Native autograd smoke is repeatable
- Command:
  - `cmake --build . --target test_dpo_native_smoke train -j2`
  - `./test_dpo_native_smoke`
- Result:
  - `loss=0.693147`
  - `grad_A_max=0.03`
  - `grad_B_max=0.0225`

### 2. Native end-to-end synthetic DPO train step completes
- Dataset:
  - `review-stage/data/synthetic_pref_longform_train.jsonl`
  - `review-stage/data/synthetic_pref_longform_eval.jsonl`
- Config:
  - `seq_len=128`
  - `batch_size=1`
  - `steps=1`
- Result:
  - `dpo_loss=0.6931`
  - `grad_norm=1.000000`
  - `nonzero_grad_tensors=24`
  - `resp_tokens=108`
  - `peak RSS=2918.7 MB`
  - wall time `106.35s`

### 3. Real-MMLU sample retention is restored
- Before tokenizer fixes:
  - native kept only `71/480` train samples at `seq_len=384`
- After tokenizer fixes:
  - native now keeps `222/480` train and `117/120` valid
  - this matches PyTorch / HuggingFace on the same real-MMLU preference split

### 4. The last tokenizer parity bug was fixed
- The final remaining mismatch came from ASCII pre-tokenization on runs of spaces before non-whitespace text.
- A native splitter change now preserves the last space in a multi-space run so it can be absorbed into the following token, matching HuggingFace GPT-2 behavior more closely.
- This resolved the last native-only dropped rows and restored full real-MMLU retention parity.

### 5. A short real-data smoke target now exists
- New lightweight real-MMLU subsets:
  - `review-stage/data/real_mmlu_pref_short_train.jsonl`
  - `review-stage/data/real_mmlu_pref_short_eval.jsonl`
- One-command entrypoint:
  - `review-stage/scripts/run_native_real_mmlu_short_smoke.sh`
- Construction rule:
  - shortest real-MMLU prompts that still preserve chosen/rejected response tokens at `seq_len=128`
- Size:
  - train `16`
  - eval `8`
- Native 1-step DPO smoke on this real-data subset completed with:
  - `dpo_loss=0.6931`
  - `grad_norm=1.000000`
  - `nonzero_grad_tensors=24`
  - `resp_tokens=2`
  - `peak RSS=2918.7 MB` on the one-command rerun
  - wall time `103.20s` on the one-command rerun
- Separate BLAS / Accelerate build experiment:
  - build dir: `gpt2_small_lora_finetune/build_release_blas`
  - one-command short smoke rerun completed in `7.28s` wall time
  - train metrics stayed qualitatively the same (`dpo_loss=0.6931`, non-zero grads)
  - the CLI banner issue is now addressed by propagating BLAS compile definitions through the `operators` target
  - new full-data one-command entrypoint:
    - `review-stage/scripts/run_native_real_mmlu_full_smoke_blas.sh`
  - full real-MMLU 1-step smoke on the BLAS build completed with:
    - retained `222/117`
    - `dpo_loss=0.6931`
    - `grad_norm=1.000000`
    - `nonzero_grad_tensors=24`
    - `peak RSS=4018.7 MB`
    - wall time `11.43s`

## Remaining Gaps
- Native real-MMLU 1-step DPO smoke still runs slowly on CPU.
- The real-data path now enters training correctly, but a short smoke was manually interrupted after `199.72s` wall time before the first step log printed.

## Main Takeaways

### 1. Native DPO is no longer blocked by objective correctness
- Loss wiring, backward, reference scoring, and LoRA update path all execute.

### 2. Tokenizer parity was the main hidden blocker
- The biggest native/PyTorch discrepancy came from GPT-2 tokenization behavior.
- Fixing tokenizer order and pre-tokenization semantics restored full real-MMLU retention parity.

### 3. The current bottleneck is runtime practicality
- The native path is now correct enough to validate.
- The next problem is how to make real-data smoke runs complete quickly enough for repeatable iteration.
- BLAS / Accelerate already reduces short-smoke runtime by more than an order of magnitude and also makes the full real-MMLU 1-step smoke complete quickly.

## Recommended Next Step
- After that, choose one practical validation path:
  - use the BLAS full real-MMLU smoke as the primary repeatable validation target
  - keep the short real-data subset as the faster fallback smoke
