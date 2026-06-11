# Results Index

Date: `2026-06-10`

## Docs

- 中文详细汇报稿:
  - [WEEKLY_PROGRESS_CN_2026-06-10.md](/Users/anakinju/DKU/MobileFineTuner/briefing/2026-06-10/WEEKLY_PROGRESS_CN_2026-06-10.md:1)
- English showcase:
  - [WEEKLY_DEMO_EN_2026-06-10.md](/Users/anakinju/DKU/MobileFineTuner/briefing/2026-06-10/WEEKLY_DEMO_EN_2026-06-10.md:1)
- Android Studio / AVD demo runbook:
  - [ANDROID_DEMO_RUNBOOK_2026-06-10.md](/Users/anakinju/DKU/MobileFineTuner/briefing/2026-06-10/ANDROID_DEMO_RUNBOOK_2026-06-10.md:1)
- 口头汇报稿:
  - [.briefing/2026-06-10/组会汇报稿_2026-06-10_android_hfrl.md](/Users/anakinju/DKU/MobileFineTuner/.briefing/2026-06-10/组会汇报稿_2026-06-10_android_hfrl.md:1)

## Main Progress This Week

- Android app now has an in-app `Native Demo Runner` instead of only command-line runs.
- Native trainer is packaged into the APK as `libmft_train.so`.
- Android app can launch native GPT-2 DPO runs from app-private storage.
- UI now exposes run presets, editable training parameters, live status, event timeline, charts, logs, and stop control.
- Android arm64 OpenBLAS was built and linked into the native trainer.
- Installed APK on real phone `3B158M00F6N00000` was verified to print `BLAS : ON (compiled)`.
- Current app demo data is a small built-in fitness/health preference dataset: 5 train pairs and 2 eval pairs.

## Key Files

- App native runner:
  - [ExperimentViewModel.kt](/Users/anakinju/DKU/MobileFineTuner/android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/viewmodel/ExperimentViewModel.kt:211)
  - [NativeDemoCommandBuilder.kt](/Users/anakinju/DKU/MobileFineTuner/android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/data/NativeDemoCommandBuilder.kt:20)
  - [TrainingModels.kt](/Users/anakinju/DKU/MobileFineTuner/android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/model/TrainingModels.kt:121)
  - [DashboardScreen.kt](/Users/anakinju/DKU/MobileFineTuner/android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/DashboardScreen.kt:1681)
- Android native trainer package path:
  - [libmft_train.so](/Users/anakinju/DKU/MobileFineTuner/android-visualizer/app/src/main/jniLibs/arm64-v8a/libmft_train.so:1)
- Android OpenBLAS integration:
  - [operator/CMakeLists.txt](/Users/anakinju/DKU/MobileFineTuner/operator/CMakeLists.txt:13)
  - [third_party/openblas-android-arm64/lib/libopenblas_armv8-r0.3.33.a](/Users/anakinju/DKU/MobileFineTuner/third_party/openblas-android-arm64/lib/libopenblas_armv8-r0.3.33.a:1)
- Current APK:
  - [app-debug.apk](/Users/anakinju/DKU/MobileFineTuner/android-visualizer/app/build/outputs/apk/debug/app-debug.apk:1)

## Validated Evidence

- Android arm64 OpenBLAS static library exists:
  - `third_party/openblas-android-arm64/lib/libopenblas_armv8-r0.3.33.a`
- Current debug APK was built and installed:
  - `android-visualizer/app/build/outputs/apk/debug/app-debug.apk`
- On the real phone, executing the APK-packaged trainer printed:
  - `BLAS          : ON (compiled)`
- No long training run was started by this validation; the verification used an early-exit config print.

## Completed Local HFRL Runs

- Health A/B DPO 220-step baseline:
  - [runs/hfrl_health_ab_dpo_s220](/Users/anakinju/DKU/MobileFineTuner/runs/hfrl_health_ab_dpo_s220:1)
  - `steps_logged=220`
  - `final_eval_reward_accuracy=1`
  - `final_eval_reward_margin=8.96158`
  - note: reference-cache preprocessing cost was high.
- Health A/B DPO 220-step no-cache rerun:
  - [runs/hfrl_health_ab_dpo_s220_nocache](/Users/anakinju/DKU/MobileFineTuner/runs/hfrl_health_ab_dpo_s220_nocache:1)
  - `steps_logged=220`
  - `final_eval_reward_accuracy=1`
  - `final_eval_reward_margin=8.96158`
- Sparse-feedback DPO 220-step run:
  - [runs/hfrl_sparse_feedback_dpo_s220](/Users/anakinju/DKU/MobileFineTuner/runs/hfrl_sparse_feedback_dpo_s220:1)
  - `steps_logged=220`
  - `final_eval_reward_accuracy=0.5`
  - `dpo_length_norm_margin=5.1937`

## Current Caution

- The app demo is now a real end-to-end GPT-2 DPO run, not the tiny `test_dpo_native_smoke` unit test.
- Slow `Load pretrained model` is mostly GPT-2 weight/model initialization, not dataset loading.
- Model pre-initialization is feasible, but requires keeping a native worker process or JNI model session alive; current app launches a fresh native process per run.
