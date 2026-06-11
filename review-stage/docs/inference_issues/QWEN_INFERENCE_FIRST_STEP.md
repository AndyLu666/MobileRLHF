# Qwen Inference First Step Issues

## Goal

Build the missing generation-inference step for the HFRL loop. For now, ignore callback/upload. The immediate target is: given a prompt, run the on-device/native model to generate text and return/display the generated answer.

## Current State

- Offline DPO training works from `prompt/chosen/rejected` JSONL.
- Android visualizer can launch a native DPO demo and show training/eval metrics.
- LoRA adapters are saved after training.
- The merged upstream adds `mft-sdk` with `open`, `initLora`, `createTrainer`, `trainStep`, and `selfTest`.
- Qwen model/tokenizer support exists in `operator` and `examples/qwen_lora_finetune`.

## Gap

The system does not yet expose generation inference:

- no native `generate(prompt)` loop,
- no Java/Kotlin `generate` SDK method,
- no Android UI to enter a prompt and display generated text,
- no post-training adapter loading path for inference.

## Issues

### Issue 1: Native Qwen generation CLI

Implement a minimal native CLI that loads Qwen model/tokenizer, accepts a prompt, runs greedy token generation, decodes output, and prints the response. This proves the core inference path before UI/JNI.

Acceptance:

- Build target exists under `examples/qwen_lora_finetune`.
- Command accepts `--model_dir`, `--prompt`, `--max_new_tokens`.
- Works with a local Qwen directory when available.
- Does not require DPO or feedback.

### Issue 2: JNI/SDK Generate API

Expose generation through `mft-sdk`.

Acceptance:

- Java API: `String generate(String prompt, int maxNewTokens, float temperature)` or a simple greedy variant first.
- JNI calls native generation and returns generated text.
- SDK sample can run one prompt and display output.

### Issue 3: Android Inference Panel

Add a visualizer UI panel for prompt input and generated response.

Acceptance:

- User enters prompt.
- App runs inference and displays output.
- No callback/upload required yet.

### Issue 4: Adapter-Aware Inference

Load a trained LoRA adapter before generation.

Acceptance:

- Inference can run base model and optionally adapter model.
- UI can show base vs trained LoRA output.

### Issue 5: Online HFRL Loop After Inference

Use generated A/B answers as feedback pairs for DPO.

Acceptance:

- Generate A/B responses.
- User picks one.
- App writes `prompt/chosen/rejected` JSONL.
- DPO training consumes the generated feedback data.
