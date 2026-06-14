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

## Not In Scope / Still Open

- Real callback/upload after feedback selection.
- Post-training LoRA adapter loading for Qwen inference.
- Full phone-side validation of the new Qwen prompt panel with real Qwen assets
  pushed to the device path.
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
local build checks. The main caveat is that the new Android Qwen prompt panel
has not yet been exercised end-to-end on a real phone with Qwen assets pushed to
`/data/local/tmp/mft_qwen3`.
