# Native DPO PR Summary

Date: `2026-06-02`

## Scope
- Finish native GPT-2 `C++` DPO validation work so the path is not limited to toy-loss checks or PyTorch-only experiments.

## Main Changes

### 1. Native smoke targets are now first-class build outputs
- Added `test_dpo_native_smoke` to `gpt2_small_lora_finetune/CMakeLists.txt`
- Added `inspect_pref_lengths` as a native tokenizer / dataset diagnostic utility

### 2. GPT-2 tokenizer parity with HuggingFace was fixed
- `operator/finetune_ops/core/tokenizer_bpe.cpp`
  - encode path now tokenizes raw text before byte-to-unicode conversion
  - replaced ASCII regex approximation with a manual GPT-2-like splitter
  - fixed remaining multi-space pre-tokenization edge case that was over-splitting real-MMLU prompts

### 3. Preference dataset parsing was hardened
- `operator/finetune_ops/data/preference_dataset.cpp`
  - improved JSON string extraction for escaped content
  - kept/skipped statistics remain visible for truncation diagnostics

### 4. BLAS compile state now propagates correctly
- `operator/CMakeLists.txt`
  - switched BLAS-related compile definitions to `target_compile_definitions(operators PUBLIC ...)`
  - top-level native CLI now reports `BLAS ON (compiled)` correctly in BLAS builds

### 5. Repeatable real-data validation paths were added
- Short real-data subset generator:
  - `review-stage/scripts/build_real_mmlu_short_pref_subset.py`
- Fast fallback smoke:
  - `review-stage/scripts/run_native_real_mmlu_short_smoke.sh`
- Primary BLAS full-data smoke:
  - `review-stage/scripts/run_native_real_mmlu_full_smoke_blas.sh`

## Key Validation Results

### Native autograd smoke
- `./test_dpo_native_smoke`
- Result:
  - `loss=0.693147`
  - non-zero grads on toy parameters

### Native synthetic 1-step DPO smoke
- Dataset:
  - `synthetic_pref_longform_*`
- Result:
  - `dpo_loss=0.6931`
  - `grad_norm=1.000000`
  - `nonzero_grad_tensors=24`

### Real-MMLU tokenizer retention parity
- Before fixes:
  - native kept only `71/480` train samples at `seq_len=384`
- After fixes:
  - native keeps `222/480` train and `117/120` valid
  - this matches PyTorch / HuggingFace

### Fallback real-data smoke
- Command:
  - `bash review-stage/scripts/run_native_real_mmlu_short_smoke.sh`
- Result:
  - `16/16` train kept
  - `8/8` valid kept
  - finite `dpo_loss`

### Primary BLAS full real-data smoke
- Command:
  - `bash review-stage/scripts/run_native_real_mmlu_full_smoke_blas.sh`
- Result:
  - `BLAS ON (compiled)`
  - `222/480` train kept
  - `117/120` valid kept
  - `dpo_loss=0.6931`
  - `grad_norm=1.000000`
  - `nonzero_grad_tensors=24`
  - wall time about `11-12s`

## Acceptance
- `PASS`: native tokenizer parity on real-MMLU retention
- `PASS`: native autograd DPO smoke target
- `PASS`: native synthetic end-to-end smoke
- `PASS`: native real-data fallback smoke
- `PASS`: native BLAS full real-MMLU 1-step smoke

## Recommended Reviewer Focus
- Check tokenizer changes in `operator/finetune_ops/core/tokenizer_bpe.cpp`
- Check BLAS compile-definition propagation in `operator/CMakeLists.txt`
- Re-run:
  - `bash review-stage/scripts/run_native_real_mmlu_full_smoke_blas.sh`
  - optionally `bash review-stage/scripts/run_native_real_mmlu_short_smoke.sh`
