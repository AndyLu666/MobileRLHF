# Health Advice DPO Dataset Research

Date: 2026-06-09

## Recommendation

Use an external preference dataset first, then use larger medical QA/SFT datasets only after constructing rejected responses or collecting A/B labels.

## Candidate 1: `victor203/fitness-preferences`

- Source: Hugging Face dataset `victor203/fitness-preferences`
- Domain: fitness and health advice preference pairs
- Rows: 54 total; train/validation/test split appears as 42/6/6 in the viewer
- Fields: `prompt`, `chosen`, `rejected`, `metadata`
- License: MIT
- Fit for MobileFineTuner DPO: high. It already matches the native trainer's required `prompt/chosen/rejected` JSONL contract.
- Limitation: small dataset, so it is useful for a realistic smoke and demo but not a strong generalization claim.
- Current blocker: this machine repeatedly hit `Connection reset by peer` when downloading from Hugging Face via `datasets`, `huggingface_hub`, `curl`, parquet API, and dataset-viewer rows API.

## Candidate 2: `FreedomIntelligence/medical-o1-reasoning-SFT`

- Source: Hugging Face dataset `FreedomIntelligence/medical-o1-reasoning-SFT`
- Domain: medical reasoning supervised fine-tuning
- Fit for MobileFineTuner DPO: medium/low as-is. It is SFT-style data, not pairwise preference data.
- Good use: future SFT baseline, or raw material for constructing A/B preference pairs with a carefully designed rejected-answer generator.
- Risk: medical reasoning/diagnosis claims are higher-risk than general wellness coaching. Do not use this to claim clinical safety.

## Candidate 3: HealthcareMagic / ChatDoctor-style QA

- Source family: HealthcareMagic / ChatDoctor medical QA datasets on Kaggle/Hugging Face mirrors
- Domain: patient question and doctor answer style QA
- Fit for MobileFineTuner DPO: low as-is. These datasets are usually question/answer corpora, not chosen/rejected preference pairs.
- Good use: future SFT baseline or RAG-style evaluation, after license and provenance checks.

## Current Fallback

Until external download succeeds, use the existing deterministic health A/B generator:

- Script: `review-stage/scripts/build_health_ab_preference_dataset.py`
- Generated local train/eval: `/private/tmp/mft_health_ab_pref_train_128.jsonl`, `/private/tmp/mft_health_ab_pref_eval_128.jsonl`
- Shape: 128 train rows, 32 eval rows, `prompt/chosen/rejected`
- Status on 8GB AVD smoke:
  - `seq_len=192`
  - train kept/skipped: `128/0`
  - eval kept/skipped: `32/0`
  - `identical_after_truncation=0`
  - peak RSS so far: about `3927.72 MB`

## Next Action

1. Finish the 10-step AVD smoke on the deterministic health A/B dataset.
2. If stable, run an 80-step health A/B AVD run before any 220-step run.
3. Retry `victor203/fitness-preferences` download later, then convert it directly to the same JSONL contract.
4. Do not treat medical-o1 or HealthcareMagic as direct DPO data until a rejected-answer construction or human preference labeling step exists.
