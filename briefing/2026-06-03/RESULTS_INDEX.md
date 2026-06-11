# Results Index

Date: `2026-06-03`

## Docs

- 中文汇报稿:
  - [WEEKLY_PROGRESS_CN_2026-06-03.md](/Users/anakinju/DKU/MobileFineTuner/briefing/2026-06-03/WEEKLY_PROGRESS_CN_2026-06-03.md:1)
- 英文 showcase:
  - [WEEKLY_DEMO_EN_2026-06-03.md](/Users/anakinju/DKU/MobileFineTuner/briefing/2026-06-03/WEEKLY_DEMO_EN_2026-06-03.md:1)
- native C++ 总结:
  - [NATIVE_DPO_VS_SFT_REAL_MMLU_2026-06-03.md](/Users/anakinju/DKU/MobileFineTuner/briefing/2026-06-03/NATIVE_DPO_VS_SFT_REAL_MMLU_2026-06-03.md:1)

## PyTorch Main Results

主总结:
- [review-stage/reports/RESULTS_SUMMARY_2026-05-31.md](/Users/anakinju/DKU/MobileFineTuner/review-stage/reports/RESULTS_SUMMARY_2026-05-31.md:1)

关键 run:
- `220-step synthetic MMLU-style proxy`
  - [runs/gpt2_dpo_vs_sft_mmlu_style_timed220](/Users/anakinju/DKU/MobileFineTuner/runs/gpt2_dpo_vs_sft_mmlu_style_timed220:1)
  - DPO `acc=0.8583`, SFT `acc=0.8417`
  - DPO `margin=23.8520`, SFT `margin=2.5100`
- `220-step real MMLU, seq_len=384`
  - [runs/gpt2_dpo_vs_sft_real_mmlu_s384_timed220](/Users/anakinju/DKU/MobileFineTuner/runs/gpt2_dpo_vs_sft_real_mmlu_s384_timed220:1)
  - DPO `acc=0.5726`, SFT `acc=0.5299`
  - DPO `margin=0.4588`, SFT `margin=-0.0253`
- `400-step real MMLU, seq_len=384`
  - [runs/gpt2_dpo_vs_sft_real_mmlu_s384_lr3e4_s400](/Users/anakinju/DKU/MobileFineTuner/runs/gpt2_dpo_vs_sft_real_mmlu_s384_lr3e4_s400:1)
  - DPO `acc=0.5556`, SFT `acc=0.5556`
  - DPO `margin=0.5583`, SFT `margin=0.0198`
- `600-step real MMLU, seq_len=384`
  - [runs/gpt2_dpo_vs_sft_real_mmlu_s384_lr3e4_s600](/Users/anakinju/DKU/MobileFineTuner/runs/gpt2_dpo_vs_sft_real_mmlu_s384_lr3e4_s600:1)
  - DPO `acc=0.5641`, SFT `acc=0.5812`
  - DPO `margin=0.8014`, SFT `margin=0.0583`

## Native C++ Main Results

当前推荐主结果:
- [runs/native_gpt2_dpo_vs_sft_real_mmlu_s384](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384:1)

关键文件:
- held-out DPO:
  - [dpo_eval.json](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384/dpo_eval.json:1)
  - `pair_accuracy=0.5726`
  - `mean_pair_margin=0.5648`
- held-out SFT:
  - [sft_eval.json](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384/sft_eval.json:1)
  - `pair_accuracy=0.5556`
  - `mean_pair_margin=0.0689`
- DPO trainer eval:
  - [dpo_eval_metrics.jsonl](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384/dpo_eval_metrics.jsonl:1)
  - `loss=0.6862`
  - `reward_accuracy=0.6200`
  - `reward_margin=0.0149`
- SFT trainer eval:
  - [sft_eval_metrics.jsonl](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384/sft_eval_metrics.jsonl:1)
  - `valid_ppl=4.6478`

运行时间:
- DPO `1135.66s`
- SFT `279.97s`

## Native C++ Older Checkpoints

- `20-step real MMLU`
  - [runs/native_gpt2_dpo_vs_sft_real_mmlu_s384_s20](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384_s20:1)
- `smoke / short smoke`
  - [runs/native_gpt2_dpo_vs_sft_real_mmlu_s384_smoke](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384_smoke:1)
  - [runs/native_gpt2_dpo_vs_sft_short_smoke](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_short_smoke:1)
  - [runs/native_gpt2_dpo_vs_sft_short_smoke_v2](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_short_smoke_v2:1)

## Dataset

真实 preference 数据:
- train:
  - [review-stage/data/real_mmlu_pref_train.jsonl](/Users/anakinju/DKU/MobileFineTuner/review-stage/data/real_mmlu_pref_train.jsonl:1)
- eval:
  - [review-stage/data/real_mmlu_pref_eval.jsonl](/Users/anakinju/DKU/MobileFineTuner/review-stage/data/real_mmlu_pref_eval.jsonl:1)
