# Android Studio / AVD Demo Runbook

Date: `2026-06-10`

This runbook is for showing the Android HFRL demo from Android Studio, an emulator, or a real Android phone.

## 0. Paths

Repo:

```bash
cd /Users/anakinju/DKU/MobileFineTuner
```

ADB:

```bash
export ADB="$HOME/Library/Android/sdk/platform-tools/adb"
```

Android Studio JBR:

```bash
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
```

## 1. Open Android Studio

```bash
open -a "Android Studio" /Users/anakinju/DKU/MobileFineTuner/android-visualizer
```

In Android Studio:

- Open the `android-visualizer/` project.
- Select an emulator or connected phone.
- Use the green Run button if you want to launch from Studio.

## 2. Start / Check AVD

List AVDs:

```bash
"$HOME/Library/Android/sdk/emulator/emulator" -list-avds
```

Start the current Pixel AVD:

```bash
"$HOME/Library/Android/sdk/emulator/emulator" -avd Pixel_7a -no-snapshot-load
```

Wait for boot:

```bash
"$ADB" devices -l
"$ADB" -s emulator-5554 shell getprop sys.boot_completed
```

Expected:

```text
1
```

## 3. Device Selection

For emulator:

```bash
export SERIAL="emulator-5554"
```

For the current real phone:

```bash
export SERIAL="3B158M00F6N00000"
```

Check devices:

```bash
"$ADB" devices -l
```

## 4. Build Debug APK

```bash
cd /Users/anakinju/DKU/MobileFineTuner/android-visualizer
JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" ./gradlew assembleDebug
```

APK:

```text
/Users/anakinju/DKU/MobileFineTuner/android-visualizer/app/build/outputs/apk/debug/app-debug.apk
```

## 5. Install APK

```bash
cd /Users/anakinju/DKU/MobileFineTuner
"$ADB" -s "$SERIAL" install -r android-visualizer/app/build/outputs/apk/debug/app-debug.apk
```

## 6. Push GPT-2 Files

The app expects GPT-2 files under:

```text
/data/local/tmp/mft_gpt2
```

Push files:

```bash
"$ADB" -s "$SERIAL" shell mkdir -p /data/local/tmp/mft_gpt2

"$ADB" -s "$SERIAL" push gpt2_small_lora_finetune/pretrained/gpt2/config.json /data/local/tmp/mft_gpt2/
"$ADB" -s "$SERIAL" push gpt2_small_lora_finetune/pretrained/gpt2/merges.txt /data/local/tmp/mft_gpt2/
"$ADB" -s "$SERIAL" push gpt2_small_lora_finetune/pretrained/gpt2/vocab.json /data/local/tmp/mft_gpt2/
"$ADB" -s "$SERIAL" push gpt2_small_lora_finetune/pretrained/gpt2/model.safetensors /data/local/tmp/mft_gpt2/
```

The largest file is `model.safetensors`, around `523MB`.

## 7. Launch App

```bash
"$ADB" -s "$SERIAL" shell am start -n com.mobilefinetuner.visualizer/.MainActivity
```

Alternative:

```bash
"$ADB" -s "$SERIAL" shell monkey -p com.mobilefinetuner.visualizer 1
```

## 8. Verify APK-Packaged Native Trainer Is BLAS ON

Find native library dir:

```bash
"$ADB" -s "$SERIAL" shell dumpsys package com.mobilefinetuner.visualizer | grep -E "legacyNativeLibraryDir|primaryCpuAbi"
```

Example output:

```text
legacyNativeLibraryDir=/data/app/.../com.mobilefinetuner.visualizer-.../lib
primaryCpuAbi=arm64-v8a
```

Run a config-only early-exit check. Replace `<APP_LIB_DIR>` with the printed path plus `/arm64`.

```bash
"$ADB" -s "$SERIAL" shell "<APP_LIB_DIR>/libmft_train.so --train_mode sft --jsonl_train /data/local/tmp/no.jsonl --pretrained_dir /data/local/tmp/mft_missing --steps 1 --seq_len 8 2>&1 | head -n 16"
```

Expected:

```text
BLAS          : ON (compiled)
```

## 9. Run Demo From App

In the app:

1. Open `Experiments`.
2. Find `Native Demo Runner`.
3. Select `Smoke`.
4. Keep defaults:
   - `steps=8`
   - `seq_len=128`
   - `rank=1`
   - `alpha=2`
   - `lr=0.001`
   - `dpo_pair_forward=OFF`
   - `cache_ref_logps=OFF`
5. Tap `Start`.
6. Watch the event timeline:
   - `[1/6] Load pretrained model`
   - `[2/6] Setup LoRA`
   - `[3/6] Load tokenizer/data`
   - `[5/6] Start training`
7. Wait for the first `[Train]` line.
8. Open `Train` and `Logs` tabs to show metrics and raw output.

## 10. Longer Demo

After Smoke works:

- Try `Phone 80`.
- Only use `Long 220` if the meeting/demo has enough time.

Recommended order:

```text
Smoke -> Phone 80 -> Long 220
```

## 11. Stop A Run

Use the app's `Stop` button.

Manual fallback:

```bash
"$ADB" -s "$SERIAL" shell "run-as com.mobilefinetuner.visualizer sh -c 'cat files/mft_runs/*/native.pid 2>/dev/null | tail -n 1'"
```

Then:

```bash
"$ADB" -s "$SERIAL" shell "run-as com.mobilefinetuner.visualizer sh -c 'kill -TERM <PID> 2>/dev/null; sleep 1; kill -KILL <PID> 2>/dev/null'"
```

## 12. Inspect Latest App Run Logs

List latest run logs:

```bash
"$ADB" -s "$SERIAL" shell "run-as com.mobilefinetuner.visualizer sh -c 'find files/mft_runs -maxdepth 2 -type f -name train.log -printf \"%T@ %p\n\" 2>/dev/null | sort -n | tail -n 10'"
```

Tail latest run log after choosing a run path:

```bash
"$ADB" -s "$SERIAL" shell "run-as com.mobilefinetuner.visualizer sh -c 'tail -n 120 files/mft_runs/<RUN_NAME>/train.log'"
```

Check metrics:

```bash
"$ADB" -s "$SERIAL" shell "run-as com.mobilefinetuner.visualizer sh -c 'wc -l files/mft_runs/<RUN_NAME>/metrics.ndjson; tail -n 5 files/mft_runs/<RUN_NAME>/metrics.ndjson'"
```

## 13. Troubleshooting

### App says no training steps parsed

Usually the run is still before the first `[Train]` line.

Check event timeline:

- If it is at `Load pretrained model`, the model is still loading.
- If it reached `Start training`, wait for first step.

### Load pretrained model is slow

This is expected for the current process-per-run design.

The app runs full GPT-2 DPO:

- loads `model.safetensors`
- builds GPT-2 small
- initializes LoRA
- creates reference scoring path

This is not the same as `test_dpo_native_smoke`, which only tests tiny tensors.

### BLAS shows OFF

Rebuild native trainer with:

```bash
cmake -S gpt2_small_lora_finetune \
  -B gpt2_small_lora_finetune/build_android_arm64_blas \
  -DCMAKE_TOOLCHAIN_FILE="$HOME/Library/Android/sdk/ndk/27.3.13750724/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-35 \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_BLAS=ON \
  -DOPENBLAS_ROOT="/Users/anakinju/DKU/MobileFineTuner/third_party/openblas-android-arm64" \
  -DBUILD_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_LEGACY_CLI=OFF

cmake --build gpt2_small_lora_finetune/build_android_arm64_blas --target train -j4
cp gpt2_small_lora_finetune/build_android_arm64_blas/train android-visualizer/app/src/main/jniLibs/arm64-v8a/libmft_train.so
```

Then rebuild and reinstall the APK.
