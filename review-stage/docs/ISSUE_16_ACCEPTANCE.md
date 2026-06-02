# Issue 16 Acceptance

Date: `2026-06-02`

## Goal
- Make native GPT-2 `C++` DPO validation repeatable on real preference data.

## Acceptance Status
- `PASS`: native tokenizer retention now matches PyTorch / HuggingFace on the real-MMLU preference split.
- `PASS`: native DPO autograd smoke is a standard build target.
- `PASS`: native end-to-end DPO smoke completes on synthetic data.
- `PASS`: native end-to-end DPO smoke completes on real data.
- `PASS`: a stronger BLAS / Accelerate full real-MMLU validation path now completes in practical time.

## Primary Validation
- Command:
  - `bash review-stage/scripts/run_native_real_mmlu_full_smoke_blas.sh`
- Expected:
  - config prints `BLAS          : ON (compiled)`
  - `PreferenceDataset` reports `222/480` train and `117/120` valid kept at `seq_len=384`
  - first train step prints finite `dpo_loss`
  - `nonzero_grad_tensors` is non-zero
- Latest observed result:
  - `dpo_loss=0.6931`
  - `grad_norm=1.000000`
  - `nonzero_grad_tensors=24`
  - `peak RSS=3891.6 MB`
  - wall time `11.78s`

## Fallback Validation
- Command:
  - `bash review-stage/scripts/run_native_real_mmlu_short_smoke.sh`
- Expected:
  - `PreferenceDataset` reports `16/16` train and `8/8` valid kept at `seq_len=128`
  - first train step prints finite `dpo_loss`
  - `nonzero_grad_tensors` is non-zero
- Latest observed result:
  - `dpo_loss=0.6931`
  - `grad_norm=1.000000`
  - `nonzero_grad_tensors=24`
  - wall time about `7-103s` depending on build path

## What Was Fixed
- Native GPT-2 tokenizer parity with HuggingFace:
  - encode order corrected
  - ASCII whitespace / punctuation pre-tokenization aligned more closely
- Native preference dataset robustness:
  - escaped JSON strings handled correctly
- Build wiring:
  - `test_dpo_native_smoke`
  - BLAS compile definitions now propagate to the top-level CLI banner

## Residual Risk
- Full real-MMLU validation is now practical only on the BLAS / Accelerate build.
- Pure C++ non-BLAS full real-MMLU smoke remains much slower and is not the recommended validation path.
