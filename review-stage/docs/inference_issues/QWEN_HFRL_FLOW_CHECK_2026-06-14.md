# Qwen HFRL Flow Check

Date: `2026-06-14`

## Scope

Check whether the current Qwen/HFRL path is complete except for callback/upload,
which is intentionally left for teammate integration.

## Completed

- Native Qwen3 single-prompt generation exists through `infer_qwen`.
- Qwen3 generation supports chat-template prompting, optional thinking trace,
  greedy/sampling decode parameters, repetition penalty, and KV-cache decoding.
- Desktop HFRL inference simulation exists through `hfrl_infer_sim` and writes:
  - `generated_responses.jsonl`
  - `inference_feedback.jsonl`
  - `train.log`
  - `metrics.ndjson`
  - `SUMMARY.md`
  - `SUMMARY.json`
- Android `mft-sdk` exposes `MobileFineTuner.generate(...)` through JNI.
- Android Experiments tab has a Qwen inference panel for:
  - entering a model directory and prompt,
  - generating one answer,
  - generating A/B answers,
  - choosing A or B,
  - saving the choice as DPO-compatible JSONL.
- Android native demo launcher prefers generated Qwen A/B feedback when
  `mft_demo_data/inference_feedback.jsonl` exists, otherwise it falls back to the
  built-in HF fitness preference dataset.
- The run-folder parser/UI can display inference artifacts and their chosen
  source, so short smoke fallback text is visible instead of being confused with
  model-quality evidence.
- Local bridge script `run_infer_then_dpo.sh` proves the desktop flow shape:
  Qwen inference feedback JSONL -> native GPT-2 DPO `--preference_train`.

## Emulator Validation

Validated on Android Studio emulator `emulator-5554` with the debug APK and real
Qwen assets pushed to `/data/local/tmp/mft_qwen3`.

- Installed `android-visualizer/app/build/outputs/apk/debug/app-debug.apk`.
- Pushed 31 Qwen asset files from `examples/qwen_lora_finetune/pretrained`.
- Opened the Android Experiments tab and used the Qwen inference panel.
- Generated visible A/B candidates through native Qwen inference.
- Selected a preference, which wrote DPO-compatible rows to
  `files/mft_demo_data/inference_feedback.jsonl`.
- The native demo launcher detected `Qwen A/B feedback` as its training source
  and reported `Pairs: 3`.
- Started `Train on A/B`, producing run
  `native/phone_qwen_ab_dpo_8s_0614_154335`.
- The run reached `Step 8 / 8`, wrote `metrics.ndjson` and
  `eval_metrics.jsonl`, and saved `hf_fitness_adapter_8s.safetensors`.

Final training evidence:

```text
Total steps: 8
Total response tokens: 270
Final EMA loss: 0.6581
Last train step loss: 0.574235
Last train reward_accuracy: 1.0000
Feedback rows copied into run folder: 3
```

## Not In Scope / Still Open

- Real callback/upload after feedback selection.
- Post-training LoRA adapter loading for Qwen inference.
- Replacing simulated training telemetry in `hfrl_infer_sim` with real Qwen-side
  DPO backprop.

## Checks Run

```bash
cd android-visualizer
JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" \
  ./gradlew :app:compileDebugKotlin :mft-sdk:assembleDebug

cd android-visualizer
JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" \
  ./gradlew :app:testDebugUnitTest

cd android-visualizer
JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" \
  ./gradlew :app:assembleDebug

ADB=/Users/anakinju/Library/Android/sdk/platform-tools/adb
$ADB install -r android-visualizer/app/build/outputs/apk/debug/app-debug.apk
$ADB push examples/qwen_lora_finetune/pretrained/. /data/local/tmp/mft_qwen3/

cmake -S examples/qwen_lora_finetune \
  -B examples/qwen_lora_finetune/build \
  -DUSE_BLAS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build examples/qwen_lora_finetune/build \
  --target infer_qwen hfrl_infer_sim \
  -j2
```

All checks passed on this workstation.

## Verdict

Everything needed before callback/upload is implemented at code level and passes
local build checks. The Android Qwen prompt panel was also exercised end-to-end
on an emulator with real Qwen assets at `/data/local/tmp/mft_qwen3`, including
A/B generation, preference logging, and native DPO training from the generated
feedback file.
