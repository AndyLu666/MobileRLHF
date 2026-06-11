# Native C++ DPO vs SFT Results Summary

Date: `2026-06-03`

## Scope
- Model: `gpt2_small_lora_finetune/pretrained/gpt2`
- Runtime: native C++ `gpt2_small_lora_finetune/build_release_blas/train`
- Main comparison target: native `DPO` vs native chosen-only `SFT`
- Main report boundary: matched `60-step` budget on full real-MMLU preference data

## Important Validation Note
- This report uses the native C++ finetune path end-to-end for both training runs.
- Before this comparison, native `SFT` was corrected to force standard attention as well:
  - `gpt2_small_lora_finetune/src/main.cpp`
  - reason: native `memory_efficient_attention` backward is still incomplete, so `SFT` and `DPO` must use the same valid gradient path
- Held-out pairwise comparison reuses the existing Python preference evaluator by converting native LoRA `.safetensors` into a minimal PEFT-compatible adapter directory:
  - `review-stage/scripts/native_lora_to_peft_adapter.py`
  - `review-stage/scripts/run_native_gpt2_dpo_vs_sft.sh`

## Timing

| Run | Dataset | Key config | Wall time |
| --- | --- | --- | --- |
| `native_gpt2_dpo_vs_sft_real_mmlu_s384` DPO | real MMLU preference pairs | `seq_len=384`, `steps=60`, `lr=5e-4`, `batch_size=1` | `1135.66s` |
| `native_gpt2_dpo_vs_sft_real_mmlu_s384` SFT | chosen-only masked real MMLU | `seq_len=384`, `steps=60`, `lr=5e-4`, `batch_size=1` | `279.97s` |

## Dataset
- Source preference train set: `review-stage/data/real_mmlu_pref_train.jsonl`
- Source preference eval set: `review-stage/data/real_mmlu_pref_eval.jsonl`
- Native DPO effective kept split at `seq_len=384`:
  - train `222 / 480`
  - eval `117 / 120`
- Chosen-only SFT JSONL converted from the same preference split:
  - train kept `222`, skipped `258`
  - eval kept `117`, skipped `3`

## Results Table

| Run dir | Data | Effective train / eval | Config | Trainer eval | Held-out preference eval | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `runs/native_gpt2_dpo_vs_sft_real_mmlu_s384` DPO | full real MMLU preference pairs | `222 / 117` | `seq_len=384`, `steps=60`, `lr=5e-4`, `beta=0.1` | `loss=0.6862`, `reward_acc=0.6200`, `reward_margin=0.0149` | `pair_accuracy=0.5726`, `mean_pair_margin=0.5648` | native policy/reference DPO loop; `48` nonzero LoRA grad tensors at step 60 |
| `runs/native_gpt2_dpo_vs_sft_real_mmlu_s384` SFT | chosen-only masked real MMLU | `222 / 117` | `seq_len=384`, `steps=60`, `lr=5e-4` | `valid_ppl=4.65` | `pair_accuracy=0.5556`, `mean_pair_margin=0.0689` | native chosen-only SFT on the same retained examples |

## Main Takeaways

### 1. Native C++ DPO vs SFT comparison is now runnable end-to-end
- The repo now has a repeatable native comparison harness:
  - `review-stage/scripts/run_native_gpt2_dpo_vs_sft.sh`
- Native LoRA export can now be evaluated with the existing held-out preference script through a PEFT adapter conversion step.

### 2. At a `60-step` budget on full real-MMLU, native DPO is slightly better than native SFT on held-out preference evaluation
- Held-out pair accuracy:
  - `DPO 0.5726`
  - `SFT 0.5556`
- Held-out mean pair margin:
  - `DPO 0.5648`
  - `SFT 0.0689`
- Interpretation:
  - current native `DPO` is modestly better on both held-out pair accuracy and pair margin
  - the advantage is still small in absolute accuracy terms
  - this supports functionality and early trainer-level competitiveness, not a final quality claim

### 3. Under the native DPO trainer objective, DPO itself is behaving sanely
- Final native DPO trainer eval at step `60`:
  - `loss=0.6862`
  - `reward_acc=0.6200`
  - `reward_margin=0.0149`
- Step logs show non-zero gradients throughout:
  - `nonzero_grad_tensors=48`
- So the remaining question is not "does native DPO backprop work"; that part is now validated.

### 4. Runtime cost is materially different
- Native DPO was much slower than native SFT on the same `60-step` real-MMLU budget:
  - `1135.66s` vs `279.97s`
- This is expected because DPO performs four model forwards per batch:
  - policy chosen
  - policy rejected
  - reference chosen
  - reference rejected

## Caveats
- This is a `60-step` comparison, not a convergence study.
- Held-out evaluation uses a PEFT adapter conversion layer for native LoRA outputs.
- Native DPO trainer logs report only `120` response tokens across `60` steps because the current metric counts response-mask tokens, not full prompt tokens.
  - That metric is still useful for consistency checks
  - but it should not be read as total training throughput

## Recommendation For Group Report
- Present this result as:
  - `native C++ DPO is now functionally ported, trainable, and directly comparable against native SFT on real MMLU prompts`
  - `at a 60-step budget, native DPO is modestly ahead of native SFT on held-out preference discrimination`
- Do not claim:
  - native DPO is already decisively better than SFT on real MMLU
  - the current 60-step result is the final quality ceiling

## Recommended Next Step
- Re-run the same native full real-MMLU comparison with multiple seeds to test whether the current small DPO advantage is stable.
- If the result is stable, then try one controlled budget extension such as `steps=120` while keeping `seq_len=384`, `lr=5e-4`, and `batch_size=1`.
