# Qwen Inference Runbook

## Purpose

This is the first real inference step for the HFRL flow: provide a prompt and let the native Qwen model generate text. It is not pairwise evaluation and it does not use callback/upload yet.

## Required Assets

Put a local Qwen model directory in one of these locations:

- `examples/qwen_lora_finetune/pretrained`
- `$MFT_MODEL_ROOT/Qwen2.5-0.5B`
- `$MFT_MODEL_ROOT/Qwen3-0.6B`
- `$MFT_MODEL_ROOT/qwen2.5-0.5b`
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
```

## Run

```bash
QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B \
PROMPT="How can I maintain my fitness routine while traveling for work?" \
MAX_NEW_TOKENS=64 \
examples/qwen_lora_finetune/run_infer.sh
```

Or call the binary directly:

```bash
examples/qwen_lora_finetune/build/infer_qwen \
  --model_dir /path/to/Qwen2.5-0.5B \
  --prompt "How can I maintain my fitness routine while traveling for work?" \
  --max_new_tokens 64
```

## Current Limitation

This first step uses greedy generation and base Qwen weights only. Adapter-aware inference, JNI/SDK exposure, Android UI, and callback/upload are separate follow-up issues.
