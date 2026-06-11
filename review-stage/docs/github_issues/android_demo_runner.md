# Android Demo Runner

## Problem

Native phone/AVD experiments currently require adb commands for pushing assets, choosing parameters, launching `mft_train`, and monitoring `train.log` / `metrics.ndjson`. This is hard to demo and makes the Android app feel like a viewer rather than an interactive HFRL product.

## Goal

Add an in-app Demo Runner that lets a user configure and launch an on-device native DPO run from the visualizer UI, then monitor the resulting run with the existing dashboard charts and logs.

## Implementation Steps

1. Add run configuration models for steps, sequence length, LoRA rank/alpha, learning rate, eval interval, cache-ref toggle, and preset metadata.
2. Add ViewModel actions to update config, create a run directory under the app external files directory, launch `/data/local/tmp/mft_train`, stop a running process, and attach the generated run folder as the active dashboard run.
3. Add an Experiments tab control surface with parameter controls, launch/stop buttons, prerequisite status, generated command preview, and live run summary.
4. Reuse the existing `train.log` / `metrics.ndjson` parser path for monitoring.
5. Add focused unit tests for command construction.

## Notes

- This first version assumes the native binary and GPT-2 assets are already staged at `/data/local/tmp/mft_train` and `/data/local/tmp/mft_gpt2`.
- Android BLAS support remains a separate build-system issue; this UI exposes whether cache-ref is enabled but does not make BLAS available.
