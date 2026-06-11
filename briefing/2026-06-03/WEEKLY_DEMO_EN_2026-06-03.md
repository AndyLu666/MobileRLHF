# DPO Showcase

Date: `2026-06-03`

## PyTorch: Why This Is the Main Comparison

- more stable training/eval pipeline
- better suited for algorithm-level `DPO vs SFT` comparison
- native `C++` this week is mainly an integration milestone, not the final benchmark path

## Dataset Used

- real MMLU-style preference pairs
- files:
  - [review-stage/data/real_mmlu_pref_train.jsonl](/Users/anakinju/DKU/MobileFineTuner/review-stage/data/real_mmlu_pref_train.jsonl:1)
  - [review-stage/data/real_mmlu_pref_eval.jsonl](/Users/anakinju/DKU/MobileFineTuner/review-stage/data/real_mmlu_pref_eval.jsonl:1)
- format:
  - `prompt`
  - `chosen`
  - `rejected`
- train/eval raw size:
  - `480 / 120`

## Real Data Examples

- `human_sexuality_val`
  - prompt: hormone feedback loop question
  - chosen: `A`
  - rejected: `C`
- `high_school_physics_val`
  - prompt: swimmer force question
  - chosen: `A`
  - rejected: `B`
- `astronomy_val`
  - prompt: blackbody cooling question
  - chosen: `D`
  - rejected: `B`

## PyTorch Main Result

Source:
- [review-stage/reports/RESULTS_SUMMARY_2026-05-31.md](/Users/anakinju/DKU/MobileFineTuner/review-stage/reports/RESULTS_SUMMARY_2026-05-31.md:1)

### Synthetic MMLU-style proxy, `220 steps`

- DPO held-out accuracy: `0.8583`
- SFT held-out accuracy: `0.8417`
- DPO held-out margin: `23.8520`
- SFT held-out margin: `2.5100`

### Real MMLU, `seq_len=384`, `220 steps`

- DPO held-out accuracy: `0.5726`
- SFT held-out accuracy: `0.5299`
- DPO held-out margin: `0.4588`
- SFT held-out margin: `-0.0253`

### Longer runs

- `400 steps`
  - DPO acc: `0.5556`
  - SFT acc: `0.5556`
  - DPO margin: `0.5583`
  - SFT margin: `0.0198`
- `600 steps`
  - DPO acc: `0.5641`
  - SFT acc: `0.5812`
  - DPO margin: `0.8014`
  - SFT margin: `0.0583`

## Metric Meaning

- `pair_accuracy`
  - fraction of pairs where model scores `chosen > rejected`
- `mean_pair_margin`
  - average of `logp(chosen) - logp(rejected)`
- `valid_ppl`
  - chosen-only SFT perplexity
- `DPO loss`
  - trainer objective, not `1 - accuracy`

## DPO vs SFT Setup

- same GPT-2 LoRA base
- same matched real-MMLU split
- SFT:
  - chosen-only masked supervision
- DPO:
  - `(prompt, chosen, rejected)` preference supervision
- key difference:
  - objective and input form
  - not a completely unrelated parameter setup

## Was Backward Used?

- yes
- DPO is not forward-only
- policy model receives gradients from DPO loss
- reference model is frozen

## Native C++: What Was Achieved

Source:
- [briefing/NATIVE_DPO_VS_SFT_REAL_MMLU_2026-06-03.md](/Users/anakinju/DKU/MobileFineTuner/briefing/NATIVE_DPO_VS_SFT_REAL_MMLU_2026-06-03.md:1)

- CLI path integrated:
  - `--train_mode dpo`
  - `--preference_train`
  - `--preference_valid`
  - `--dpo_beta`
- preference dataset loading integrated
- policy/reference scoring integrated
- native DPO loss integrated
- backward pass integrated
- LoRA update path verified

## Native C++ Data and Retention

- same real-MMLU preference data
- `seq_len=384`
- effective kept split:
  - train `222 / 480`
  - eval `117 / 120`

## Native C++ Latest Result

Run dir:
- [runs/native_gpt2_dpo_vs_sft_real_mmlu_s384](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384:1)

Held-out eval:
- DPO: [dpo_eval.json](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384/dpo_eval.json:1)
  - accuracy: `0.5726`
  - margin: `0.5648`
- SFT: [sft_eval.json](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384/sft_eval.json:1)
  - accuracy: `0.5556`
  - margin: `0.0689`

Trainer eval:
- DPO: [dpo_eval_metrics.jsonl](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384/dpo_eval_metrics.jsonl:1)
  - loss: `0.6862`
  - reward accuracy: `0.6200`
  - reward margin: `0.0149`
- SFT: [sft_eval_metrics.jsonl](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384/sft_eval_metrics.jsonl:1)
  - valid ppl: `4.65`

## Runtime

- native DPO, `60 steps`: `1135.66s`
- native SFT, `60 steps`: `279.97s`

## What We Found on C++

- native C++ DPO is functionally working
- not just toy loss validation
- end-to-end training path is connected
- current branch should be treated as a checkpoint, not the final stable version

## Room for Improvement

- multi-seed validation
- cleaner post-merge stability check
- longer-budget rerun after teammate code merge
- possible speed/memory optimization in native DPO path
