# Qwen Inference First Step Issues

## Goal

Build the missing generation-inference step for the HFRL loop. For now, ignore callback/upload. The immediate target is: given a prompt, run the on-device/native model to generate text and return/display the generated answer.

## Current State

- Offline DPO training works from `prompt/chosen/rejected` JSONL.
- Android visualizer can launch a native DPO demo and show training/eval metrics.
- LoRA adapters are saved after training.
- The merged upstream adds `mft-sdk` with `open`, `initLora`, `createTrainer`, `trainStep`, and `selfTest`.
- Qwen3-0.6B model/tokenizer support exists in `operator` and `examples/qwen_lora_finetune`.
- Native C++ single-prompt Qwen3 generation exists through `infer_qwen`; Qwen3
  q/k norm and tied/output lm head loading are wired for normal text output.
- Native C++ Qwen3 generation now uses KV cache by default for faster token
  generation; `USE_KV_CACHE=0` remains available for speed/correctness checks.
- JNI/SDK exposes a basic greedy `generate(prompt, maxNewTokens)` API and uses
  the same Qwen KV-cache path.
- Desktop C++ HFRL inference simulation exists through `hfrl_infer_sim`.
- Desktop Qwen examples enable BLAS on macOS through Accelerate.
- Android visualizer can read and display generated inference artifacts from a run folder,
  including whether `chosen` came from model output or a short-run fallback.

## Gap

The system does not yet expose generation inference:

- no Android UI to enter a prompt and display generated text,
- no post-training adapter loading path for inference.
- no real callback/upload path for inference-generated preference pairs.

## Issues

### Issue 1: Native Qwen generation CLI

Implement a minimal native CLI that loads Qwen model/tokenizer, accepts a prompt, runs greedy token generation, decodes output, and prints the response. This proves the core inference path before UI/JNI.

Status: done for Qwen3-0.6B base-model generation. Demo mode disables visible
Qwen3 thinking traces by default (`THINKING=0`), supports optional sampling
parameters, and uses KV cache by default.

Acceptance:

- Build target exists under `examples/qwen_lora_finetune`.
- Command accepts `--model_dir`, `--prompt`, `--max_new_tokens`, `--temperature`,
  `--top_p`, `--repetition_penalty`, `--chat_template`, and `--thinking`.
- Works with a local Qwen directory when available.
- Does not require DPO or feedback.
- `USE_KV_CACHE=1` and `USE_KV_CACHE=0` produce matching greedy output on the
  8-token smoke prompt; 32-token local timing improved from `17.08s` to `5.18s`.

### Issue 2: JNI/SDK Generate API

Expose generation through `mft-sdk`.

Status: done for a basic greedy `generate(prompt, maxNewTokens)` API. Qwen
generation uses KV cache; non-Qwen models keep the previous full-forward path.

Acceptance:

- Java API: `String generate(String prompt, int maxNewTokens, float temperature)` or a simple greedy variant first.
- JNI calls native generation and returns generated text.
- SDK sample can run one prompt and display output.

### Issue 3: Android Inference Panel

Add a visualizer UI panel for prompt input and generated response.

Status: partially done. The app can display run-folder inference artifacts
(`generated_responses.jsonl` and `inference_feedback.jsonl`), but it does not yet
provide an in-app prompt box that launches Qwen3 generation on the phone.

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

### Issue 6: Desktop HFRL Inference Simulation

Before updating Android UI, run a computer-side HFRL simulation that proves the
new inference stage can feed the existing run-folder/visualizer contract.

Status: implemented for base Qwen3 inference plus simulated telemetry.

Acceptance:

- Build target exists under `examples/qwen_lora_finetune`: `hfrl_infer_sim`.
- Runner exists: `examples/qwen_lora_finetune/run_hfrl_infer_sim.sh`.
- Run folder contains `generated_responses.jsonl`, `inference_feedback.jsonl`,
  `train.log`, `metrics.ndjson`, `SUMMARY.md`, and `SUMMARY.json`.
- `inference_feedback.jsonl` uses DPO-compatible `prompt/chosen/rejected` rows.
- `metrics.ndjson` contains Android-readable `loss`, `reward_accuracy`,
  `reward_margin`, `chosen_reward`, `rejected_reward`, `grad_norm`, and `rss_mb`.
- Android Overview displays parsed inference feedback records when the run folder
  contains `generated_responses.jsonl` or `inference_feedback.jsonl`.
- `generated_responses.jsonl` records raw `completion` and `chosen_source`, so
  short smoke output is not confused with real full-answer quality.
- `USE_KV_CACHE=1` is the default. The local 32-token one-prompt HFRL smoke
  improved from `16.703s` total / `13586.97ms` inference to `5.030s` total /
  `1889.88ms` inference.
