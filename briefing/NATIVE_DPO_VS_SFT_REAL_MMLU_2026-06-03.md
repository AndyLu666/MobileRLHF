# Native C++ DPO vs SFT Results Summary

Date: `2026-06-03`

## Scope
- Model: `gpt2_small_lora_finetune/pretrained/gpt2`
- Runtime: native C++ `gpt2_small_lora_finetune/build_release_blas/train`
- Main comparison target: native `DPO` vs native chosen-only `SFT`
- Main report boundary: matched `20-step` budget on full real-MMLU preference data

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
| `native_gpt2_dpo_vs_sft_real_mmlu_s384_s20` DPO | real MMLU preference pairs | `seq_len=384`, `steps=20`, `lr=5e-4`, `batch_size=1` | `623.01s` |
| `native_gpt2_dpo_vs_sft_real_mmlu_s384_s20` SFT | chosen-only masked real MMLU | `seq_len=384`, `steps=20`, `lr=5e-4`, `batch_size=1` | `169.00s` |

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
| `runs/native_gpt2_dpo_vs_sft_real_mmlu_s384_s20` DPO | full real MMLU preference pairs | `222 / 117` | `seq_len=384`, `steps=20`, `lr=5e-4`, `beta=0.1` | `loss=0.6913`, `reward_acc=0.6400`, `reward_margin=0.0038` | `pair_accuracy=0.5556`, `mean_pair_margin=0.5236` | native policy/reference DPO loop; `48` nonzero LoRA grad tensors at step 20 |
| `runs/native_gpt2_dpo_vs_sft_real_mmlu_s384_s20` SFT | chosen-only masked real MMLU | `222 / 117` | `seq_len=384`, `steps=20`, `lr=5e-4` | `valid_ppl=37.07` | `pair_accuracy=0.5726`, `mean_pair_margin=0.4300` | native chosen-only SFT on the same retained examples |

## Main Takeaways

### 1. Native C++ DPO vs SFT comparison is now runnable end-to-end
- The repo now has a repeatable native comparison harness:
  - `review-stage/scripts/run_native_gpt2_dpo_vs_sft.sh`
- Native LoRA export can now be evaluated with the existing held-out preference script through a PEFT adapter conversion step.

### 2. At a `20-step` budget on full real-MMLU, DPO is competitive but not better than SFT on held-out pair accuracy
- Held-out pair accuracy:
  - `DPO 0.5556`
  - `SFT 0.5726`
- Held-out mean pair margin:
  - `DPO 0.5236`
  - `SFT 0.4300`
- Interpretation:
  - current native `DPO` is slightly worse on accuracy
  - but slightly better on margin
  - this is not a decisive win for either method at this short budget

### 3. Under the native DPO trainer objective, DPO itself is behaving sanely
- Final native DPO trainer eval at step `20`:
  - `loss=0.6913`
  - `reward_acc=0.6400`
  - `reward_margin=0.0038`
- Step logs show non-zero gradients throughout:
  - `nonzero_grad_tensors=48`
- So the remaining question is not "does native DPO backprop work"; that part is now validated.

### 4. Runtime cost is materially different
- Native DPO was much slower than native SFT on the same `20-step` real-MMLU budget:
  - `623.01s` vs `169.00s`
- This is expected because DPO performs four model forwards per batch:
  - policy chosen
  - policy rejected
  - reference chosen
  - reference rejected

## Caveats
- This is a `20-step` comparison, not a convergence study.
- Held-out evaluation uses a PEFT adapter conversion layer for native LoRA outputs.
- The converted native adapters trigger PEFT warnings about missing `mlp.c_proj` adapter keys.
  - This comes from PEFT matching GPT-2 `c_proj` in both attention and MLP modules.
  - Native GPT-2 training here only targets attention `c_attn` and attention `c_proj`, which is the intended topology.
  - The warning does not block evaluation, but it should be documented when presenting the result.

## Recommendation For Group Report
- Present this result as:
  - `native C++ DPO is now functionally ported, trainable, and directly comparable against native SFT on real MMLU prompts`
  - `at a short 20-step budget, native DPO is competitive with native SFT but does not yet beat it on held-out pair accuracy`
- Do not claim:
  - native DPO is already better than SFT on real MMLU
  - the current 20-step result is the final quality ceiling

## Recommended Next Step
- Rerun the same native full real-MMLU comparison at a slightly longer but still practical budget:
  - `steps=60`
  - keep `seq_len=384`, `lr=5e-4`, `batch_size=1`
- In parallel, tighten the PEFT conversion metadata so the GPT-2 `c_proj` warning disappears and the evaluation bridge is cleaner.
