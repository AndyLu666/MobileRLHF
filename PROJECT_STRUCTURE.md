# Project Structure

This repository is organized around the maintained MobileRLHF phone-training
path. Research outputs, old review notes, build caches, and run artifacts are
kept outside the Git-tracked tree.

| Path | Purpose |
| --- | --- |
| `android-visualizer/` | Android app, SDK/AAR module, JNI integration, UI, and device tests. |
| `operator/` | Native C++ MobileFineTuner training core. |
| `examples/` | Native model-family examples and CLI entrypoints. |
| `pytorch_alignment/` | Lightweight Python reference checks for DPO/KTO data and loss behavior. |
| `scripts/android/` | adb helpers for real-phone federation. |
| `scripts/federated/` | LoRA adapter FedAvg and round summarization. |
| `scripts/prepare_fedrl_datasets.py` | Validates DPO JSONL and regenerates KTO JSONL assets. |
| `docs/` | Stable public documentation. |
| `tests/` | Python unit tests for maintained helper code. |
| `third_party/` | Small vendored native dependencies needed by the Android/native build. |

## Maintained MobileRLHF Flow

1. Validate bundled preference assets:

```bash
python3 scripts/prepare_fedrl_datasets.py
```

2. Build and install the Android app.

3. Run DPO or KTO federation:

```bash
export FEDRL_SERIALS="PHONE_A PHONE_B PHONE_C"
OBJECTIVE=dpo ROUND=001 scripts/android/run_fedrl_round_with_monitor.sh
OBJECTIVE=kto ROUND=001 scripts/android/run_fedrl_round_with_monitor.sh
```

4. Continue with `PREVIOUS_GLOBAL_ADAPTER` for later rounds.

## Artifact Policy

- `runs/` is for local metrics, adapter outputs, monitor CSVs, and summaries.
- `review-stage/`, dated briefings, presentations, IDE files, and build caches
  are not part of the public source tree.
- Model weights and external datasets must stay outside Git-tracked paths.
- Small JSONL assets under `android-visualizer/app/src/main/assets/` are
  tracked because they make a clean clone runnable without the old research
  workspace.
