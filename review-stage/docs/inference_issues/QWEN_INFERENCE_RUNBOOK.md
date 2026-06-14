# Qwen Inference Runbook

## Purpose

This is the first real inference step for the HFRL flow: provide a prompt and let the native Qwen model generate text. It is not pairwise evaluation and it does not use callback/upload yet.

## Required Assets

Put a local Qwen model directory in one of these locations:

- `examples/qwen_lora_finetune/pretrained`
- `$MFT_MODEL_ROOT/Qwen3-0.6B`
- `$MFT_MODEL_ROOT/qwen3-0.6b`
- `$MFT_MODEL_ROOT/qwen`

The directory must contain:

- `config.json`
- `tokenizer.json`
- `vocab.json`
- `merges.txt`
- `model.safetensors`, or `model.safetensors.index.json` plus shard files

## Build

```bash
cmake -S examples/qwen_lora_finetune -B examples/qwen_lora_finetune/build -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build examples/qwen_lora_finetune/build --target infer_qwen -j
cmake --build examples/qwen_lora_finetune/build --target hfrl_infer_sim -j
```

On macOS this links the operator kernels against Accelerate. The CMake output
should show `BLAS acceleration: ON` and `Linked Accelerate Framework`.

## Run Single-Prompt Inference

```bash
QWEN_MODEL_DIR=/path/to/Qwen3-0.6B \
PROMPT="How can I maintain my fitness routine while traveling for work?" \
MAX_NEW_TOKENS=64 \
THINKING=0 \
USE_KV_CACHE=1 \
examples/qwen_lora_finetune/run_infer.sh
```

Or call the binary directly:

```bash
examples/qwen_lora_finetune/build/infer_qwen \
  --model_dir /path/to/Qwen3-0.6B \
  --prompt "How can I maintain my fitness routine while traveling for work?" \
  --max_new_tokens 64 \
  --thinking 0
```

Useful generation parameters:

- `MAX_NEW_TOKENS`: how many new tokens to generate. Very small values such as
  `2` or `8` only test the first words, not a full answer.
- `THINKING=0`: default demo mode. It disables Qwen3's visible thinking trace
  and returns the answer directly.
- `THINKING=1`: show Qwen3's `<think>...</think>` trace before the answer.
- `TEMPERATURE=0`: deterministic greedy decoding.
- `TEMPERATURE=0.7 TOP_P=0.9`: sampled decoding.
- `REPETITION_PENALTY=1.08`: light repeat penalty to avoid short repeated text.
- `USE_KV_CACHE=1`: default fast path. The full prompt is processed once, then
  later tokens reuse cached attention keys/values.
- `USE_KV_CACHE=0`: slow comparison path. Every generated token recomputes the
  full prompt and all previously generated tokens.

## KV Cache Speed Check

Measured on the local macOS desktop build with Qwen3-0.6B, BLAS/Accelerate
enabled, `MAX_NEW_TOKENS=32`, `THINKING=0`, greedy decoding:

| Path | Wall time | HFRL sim total | HFRL inference only |
| --- | ---: | ---: | ---: |
| `USE_KV_CACHE=0` | `17.08s` | `16.703s` | `13586.97ms` |
| `USE_KV_CACHE=1` | `5.18s` | `5.030s` | `1889.88ms` |

The decoded 32-token output was identical in this greedy test. In plain terms:
before this change, generation recalculated the whole conversation for every new
token; now it calculates the prompt once and reuses previous attention state.

## Run Desktop HFRL Inference Simulation

This is the current computer-side end-to-end simulation before the Android update.
It performs real native C++ Qwen3 generation, writes DPO-compatible generated
preference pairs, and emits Android-readable simulated training telemetry.

Quick smoke:

```bash
MAX_NEW_TOKENS=8 \
TRAIN_STEPS=4 \
PROMPT_LIMIT=1 \
THINKING=0 \
USE_KV_CACHE=1 \
OUT_DIR=runs/qwen3_hfrl_infer_sim_smoke \
examples/qwen_lora_finetune/run_hfrl_infer_sim.sh
```

Full local demo:

```bash
MAX_NEW_TOKENS=32 \
TRAIN_STEPS=8 \
THINKING=0 \
USE_KV_CACHE=1 \
OUT_DIR=runs/qwen3_hfrl_infer_sim \
examples/qwen_lora_finetune/run_hfrl_infer_sim.sh
```

Run folder outputs:

- `generated_responses.jsonl`: raw prompt/completion records from C++ Qwen3 inference. `completion` is the real model output.
- `inference_feedback.jsonl`: DPO-compatible `prompt/chosen/rejected` pairs for the next training stage.
- `train.log`: log lines that Android can display.
- `metrics.ndjson`: Android-readable training/reward telemetry.
- `SUMMARY.md` and `SUMMARY.json`: compact run summary.

`chosen` normally copies `completion`. If a very short smoke run produces too
little text, `chosen` may use a safe fallback sentence; check `chosen_source`.

## Android BLAS

The Android SDK build is pure C++ by default because the Android NDK does not
ship OpenBLAS. To enable BLAS on a real phone build, provide an arm64 OpenBLAS
directory containing `include/cblas.h` and `lib/libopenblas.so` or `.a`, then run:

```bash
cd android-visualizer
JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home" \
./gradlew :mft-sdk:assembleDebug \
  -Pandroid.injected.cmake.arguments="-DMFT_ANDROID_USE_OPENBLAS=ON -DOPENBLAS_ROOT=/path/to/android-arm64-openblas"
```

Without that OpenBLAS package, keep Android at the default pure C++ path.

## Current Limitation

This first step uses base Qwen3 weights only. The desktop HFRL run performs real
native C++ Qwen3 inference, then simulates training metrics after inference; it
does not perform real DPO backprop yet. The current KV cache still concatenates
cache tensors as generation grows, so it is much faster than full recomputation
but not the final possible speed. Adapter-aware inference, prompt-entry Android
UI, and callback/upload are separate follow-up issues.
