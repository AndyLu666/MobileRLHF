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
cmake -S examples/qwen_lora_finetune -B examples/qwen_lora_finetune/build
cmake --build examples/qwen_lora_finetune/build --target infer_qwen -j
cmake --build examples/qwen_lora_finetune/build --target hfrl_infer_sim -j
```

## Run Single-Prompt Inference

```bash
QWEN_MODEL_DIR=/path/to/Qwen3-0.6B \
PROMPT="How can I maintain my fitness routine while traveling for work?" \
MAX_NEW_TOKENS=64 \
examples/qwen_lora_finetune/run_infer.sh
```

Or call the binary directly:

```bash
examples/qwen_lora_finetune/build/infer_qwen \
  --model_dir /path/to/Qwen3-0.6B \
  --prompt "How can I maintain my fitness routine while traveling for work?" \
  --max_new_tokens 64
```

## Run Desktop HFRL Inference Simulation

This is the current computer-side end-to-end simulation before the Android update.
It performs real native C++ Qwen3 generation, writes DPO-compatible generated
preference pairs, and emits Android-readable simulated training telemetry.

Quick smoke:

```bash
MAX_NEW_TOKENS=2 \
TRAIN_STEPS=4 \
PROMPT_LIMIT=1 \
OUT_DIR=runs/qwen3_hfrl_infer_sim_smoke \
examples/qwen_lora_finetune/run_hfrl_infer_sim.sh
```

Full local demo:

```bash
MAX_NEW_TOKENS=4 \
TRAIN_STEPS=8 \
OUT_DIR=runs/qwen3_hfrl_infer_sim \
examples/qwen_lora_finetune/run_hfrl_infer_sim.sh
```

Run folder outputs:

- `generated_responses.jsonl`: raw prompt/completion records from C++ Qwen3 inference.
- `inference_feedback.jsonl`: DPO-compatible `prompt/chosen/rejected` pairs for the next training stage.
- `train.log`: log lines that Android can display.
- `metrics.ndjson`: Android-readable training/reward telemetry.
- `SUMMARY.md` and `SUMMARY.json`: compact run summary.

## Current Limitation

This first step uses greedy generation and base Qwen3 weights only. The desktop
HFRL run simulates training metrics after inference; it does not perform real DPO
backprop yet. Adapter-aware inference, Android UI integration, and callback/upload
are separate follow-up issues.
