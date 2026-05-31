# DPO vs SFT Results Summary

Date: `2026-05-31`

## Scope
- Model: `gpt2_small_lora_finetune/pretrained/gpt2`
- Trainer: `pytorch_alignment/gpt2_lora_finetune.py`
- Main comparison target: `DPO` vs chosen-only `SFT`
- Main report boundary: `220 steps`

## Timing
- Wall-clock time was not logged consistently in the earliest runs.
- To provide report-quality timing, the two main `220-step` baselines were re-run under `/usr/bin/time -p`.

| Run | Dataset | Key config | Wall time |
| --- | --- | --- | --- |
| `gpt2_dpo_vs_sft_mmlu_style_timed220` | synthetic MMLU-style proxy | `seq_len=128`, `steps=220`, `lr=5e-4` | `225.72s` |
| `gpt2_dpo_vs_sft_real_mmlu_s384_timed220` | real MMLU prompts | `seq_len=384`, `steps=220`, `lr=5e-4` | `1045.06s` |

## Dataset Variants
- `synthetic_pref_from_sft_*`: tiny matched synthetic preference pairs
- `synthetic_pref_longform_*`: tiny long-form synthetic preference pairs
- `synthetic_mmlu_style_pref_*`: larger synthetic multiple-choice proxy, `480/120`
- `real_mmlu_pref_*`: real MMLU prompt text, sampled as `480/120`

## Results Table

| Run dir | Data | Effective train / eval | Config | DPO trainer eval | SFT trainer eval | DPO held-out | SFT held-out | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `runs/gpt2_dpo_vs_sft` | tiny synthetic | very small | `seq_len=128`, `steps=60`, `lr=5e-4` | `loss=0.0100`, `acc=1.0000`, `margin=6.4107` | initial SFT path hit the old dataclass bug | n/a | n/a | pre-fix smoke path only |
| `runs/gpt2_dpo_vs_sft_longform` | tiny long-form synthetic | `12 / 6` | `seq_len=128`, `steps=60`, `lr=5e-4` | `loss=0.5994`, `acc=1.0000`, `margin=0.2053` | `valid_ppl=15.33` | `acc=0.5000`, `margin=-19.8201` | `acc=0.5000`, `margin=-19.7202` | policy-only held-out not very informative here |
| `runs/gpt2_dpo_vs_sft_mmlu_style` | synthetic MMLU-style proxy | `480 / 120` | `seq_len=128`, `steps=220`, `lr=5e-4` | `loss=0.4062`, `acc=0.8400`, `margin=2.0862` | `valid_ppl=2.92` | `acc=0.8583`, `margin=23.8520` | `acc=0.8417`, `margin=2.5100` | strongest clear DPO win |
| `runs/gpt2_dpo_vs_sft_real_mmlu` | real MMLU, too short | `43 / 92` | `seq_len=128`, `steps=220`, `lr=5e-4` | `loss=0.9928`, `acc=0.4891`, `margin=-0.0069` | `valid_ppl=2.50` | `acc=0.5109`, `margin=0.3859` | `acc=0.5000`, `margin=-0.0248` | invalid baseline; severe truncation |
| `runs/gpt2_dpo_vs_sft_real_mmlu_s384` | real MMLU | `222 / 117` | `seq_len=384`, `steps=220`, `lr=5e-4` | `loss=0.6955`, `acc=0.4200`, `margin=-0.0039` | `valid_ppl=3.81` | `acc=0.5726`, `margin=0.4588` | `acc=0.5299`, `margin=-0.0253` | first usable real-MMLU comparison |
| `runs/gpt2_dpo_vs_sft_real_mmlu_s384_lr3e4_s400` | real MMLU | `222 / 117` | `seq_len=384`, `steps=400`, `lr=3e-4` | `loss=0.6916`, `acc=0.5600`, `margin=0.0055` | `valid_ppl=3.67` | `acc=0.5556`, `margin=0.5583` | `acc=0.5556`, `margin=0.0198` | more stable than 220-step trainer metric |
| `runs/gpt2_dpo_vs_sft_real_mmlu_s384_lr3e4_s600` | real MMLU | `222 / 117` | `seq_len=384`, `steps=600`, `lr=3e-4` | `loss=0.6872`, `acc=0.5400`, `margin=0.0332` | `valid_ppl=3.51` | `acc=0.5641`, `margin=0.8014` | `acc=0.5812`, `margin=0.0583` | more separation margin, no clear accuracy win |

## Main Takeaways

### 1. DPO integration works
- The PyTorch DPO path is functional end-to-end.
- On synthetic MMLU-style data, DPO clearly outperforms chosen-only SFT on pairwise preference discrimination.

### 2. Real MMLU is much harder mainly because of context length
- `seq_len=128` is not sufficient for real MMLU prompts.
- Moving from `128` to `384` increased usable train examples from `43` to `222`.
- The core bottleneck is effective context coverage, not whether the DPO loss is implemented correctly.

### 3. For a `220-step` budget, real-MMLU DPO is competitive
- On the first usable real-MMLU run (`seq_len=384`, `220` steps), DPO achieved better held-out pair accuracy than SFT:
  - `DPO 0.5726` vs `SFT 0.5299`
- DPO also achieved a positive held-out pair margin while SFT stayed slightly negative.

### 4. More than `220` steps did not create a decisive real-data win
- `400` steps improved DPO trainer `reward_acc` to `0.5600`.
- `600` steps increased DPO held-out margin to `0.8014`.
- But SFT caught up on held-out pair accuracy by `600` steps (`0.5812` vs `0.5641`).
- Conclusion: longer training helped stability, but did not produce a clean accuracy win for DPO.

## Recommendation For Group Report
- Use `220 steps` as the main comparison budget.
- Present `seq_len=384` as the minimum usable real-MMLU setting for GPT-2 small.
- Claim the following, and not more:
  - `DPO works and is competitive with SFT on real MMLU prompts.`
  - `DPO is clearly stronger on synthetic preference-style proxies.`
  - `The main real-data bottleneck is context-length / sample retention, not the DPO objective itself.`

## Recommended Next Step
- Do not keep increasing steps on the same setup.
- Instead, build a filtered real-MMLU subset that keeps shorter prompts while preserving real question text.
- Then rerun `DPO vs SFT` once on that filtered subset.

## Files Added For The Real-MMLU Path
- `review-stage/scripts/restore_mmlu_from_hf.py`
- `review-stage/scripts/build_mmlu_preference_from_csv.py`
- `review-stage/data/real_mmlu_pref_train.jsonl`
- `review-stage/data/real_mmlu_pref_eval.jsonl`
- `gpt2_small_lora_finetune/prepare_data.py`
- `gpt2_medium_lora_finetune/prepare_data.py`
