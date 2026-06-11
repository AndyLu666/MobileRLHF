# Android HFRL Demo Showcase

Date: `2026-06-10`

## One-Line Summary

This week we moved MobileFineTuner from command-line native DPO validation to an Android app-driven on-device GPT-2 LoRA DPO demo, with live logs, charts, stop control, and Android arm64 BLAS enabled.

## Main Contributions

- Added an in-app `Native Demo Runner`.
- Packaged the native trainer into the APK as `libmft_train.so`.
- Launches native GPT-2 DPO runs from app-private storage.
- Writes built-in fitness/health preference data on device.
- Streams `train.log` and `metrics.ndjson` into the dashboard.
- Shows pre-first-step status, logs, event timeline, and parsed metrics.
- Added stop support via native pid tracking.
- Built Android arm64 OpenBLAS and linked it into the native trainer.

## Current Android Demo Dataset

- Domain: fitness / health advice preference pairs.
- Format:
  - `prompt`
  - `chosen`
  - `rejected`
  - `source`
- Built-in app demo size:
  - train: `5` pairs
  - eval: `2` pairs
- External converted dataset also exists:
  - `victor203/fitness-preferences`
  - converted train/eval: `42 / 12` rows

## App Demo Flow

- User opens the Android app.
- User selects a native demo preset:
  - `Smoke`
  - `Phone 80`
  - `Long 220`
- User can edit:
  - steps
  - sequence length
  - LoRA rank / alpha
  - learning rate
  - eval interval
  - DPO forward mode
  - reference logp cache mode
- App launches native trainer.
- Dashboard updates live.

## Verified Native Runtime

- Previous Android binary:
  - `BLAS : OFF (compiled)`
- Current real-phone APK:
  - `BLAS : ON (compiled)`
- OpenBLAS build:
  - Android arm64
  - static CBLAS
  - installed under `third_party/openblas-android-arm64`

## Why Loading Still Takes Time

The app demo is a full GPT-2 DPO run, not the tiny DPO unit smoke test.

The full app path loads:

- GPT-2 config
- full GPT-2 small model
- `model.safetensors` around `523MB`
- tokenizer vocab / merges
- LoRA modules
- policy and reference scoring path

The fast `test_dpo_native_smoke` only tests small tensors and DPO backward correctness. It does not load GPT-2 weights.

## Model Pre-Initialization Plan

Model pre-initialization is feasible.

Short-term:

- keep the current CLI native process path for reliability
- use BLAS-enabled APK for demo
- show the load phase explicitly in the event timeline

Medium-term:

- add a persistent native worker process
- load GPT-2 once
- send run configs from the app to the worker
- reuse the loaded model across runs

Long-term:

- expose a JNI `ModelSession`
- use `initModel`, `startDpoRun`, `stopRun`, and `releaseModel`
- reduce startup cost and improve app integration

## Completed Local HFRL Runs

### Health A/B DPO 220-step

- run: `runs/hfrl_health_ab_dpo_s220`
- steps: `220`
- final eval reward accuracy: `1.0`
- final eval reward margin: `8.96158`
- note: reference-cache preprocessing was expensive

### Health A/B no-cache rerun

- run: `runs/hfrl_health_ab_dpo_s220_nocache`
- steps: `220`
- final eval reward accuracy: `1.0`
- final eval reward margin: `8.96158`
- reference-cache seconds: `0.0`

### Sparse-feedback DPO 220-step

- run: `runs/hfrl_sparse_feedback_dpo_s220`
- steps: `220`
- final eval reward accuracy: `0.5`
- DPO length-normalized margin: `5.1937`

## Demo Claim

Safe claim:

`The Android vertical slice is now functional: the app can launch a native GPT-2 LoRA DPO run on device, visualize progress, stop the native trainer, and run with Android arm64 BLAS enabled.`

Avoid claiming:

- large-scale Android benchmark completion
- medically reliable health advice generation
- final optimized startup latency

## Next Steps

- Add model pre-initialization via native worker or JNI model session.
- Replace the tiny built-in demo set with sampled external fitness preference pairs.
- Add a quick-check preset that confirms BLAS, model loading, and first metric in one flow.
- Continue Android UX improvements around load-phase progress and run comparison.
