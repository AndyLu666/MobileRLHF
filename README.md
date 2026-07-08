# MobileRLHF

MobileRLHF is a real-device federated preference post-training stack for
MobileFineTuner. The current maintained path runs DPO and KTO LoRA training on
Android phones, collects only local adapter updates and run metadata, and
aggregates adapters with FedAvg on the host.

## What Is Included

| Path | Purpose |
| --- | --- |
| `android-visualizer/` | Android app, SDK module, JNI bridge, and phone-side DPO/KTO runner. |
| `operator/` | Native C++ training core used by the Android SDK. |
| `scripts/android/` | adb automation for installing, starting, monitoring, collecting, and distributing adapters. |
| `scripts/federated/` | FedAvg aggregation and round summarization. |
| `scripts/prepare_fedrl_datasets.py` | Validates DPO data and regenerates KTO JSONL assets. |
| `android-visualizer/app/src/main/assets/mft_demo_data/` | Small tracked DPO/KTO JSONL assets for clean-clone demos. |
| `docs/FEDERATED_DPO_KTO_PHONE_RUNBOOK.md` | Full phone runbook. |
| `docs/FEDERATED_DPO_KTO_DATA_DESIGN.md` | Dataset, model, parameter, and artifact schema. |

Generated outputs are intentionally not tracked. Local runs go under `runs/`,
which is ignored by Git.

## DPO and KTO

MobileRLHF supports two preference objectives on device:

| Objective | Android native trainer | Input view | Output adapter |
| --- | --- | --- | --- |
| DPO | `createDpoTrainer()` + `trainPreferenceBatch()` | `prompt/chosen/rejected` pairs | `qwen_dpo_lora_adapter.jsonl` |
| KTO | `createKtoTrainer()` + `trainKtoPreferenceBatch()` | binary desirable/undesirable feedback, expanded from pairs when needed | `qwen_kto_lora_adapter.jsonl` |

Both objectives cache base-model reference log-probs, initialize a LoRA adapter,
optionally load the previous global LoRA adapter, train locally, then save a
JSONL LoRA adapter for aggregation.

## Data Assets

Prepare or validate the tracked preference assets:

```bash
python3 scripts/prepare_fedrl_datasets.py
```

Current bundled assets:

| Dataset | DPO train/eval | KTO train/eval |
| --- | ---: | ---: |
| `fedrl_health` | 24 / 6 pairs | 48 / 12 binary rows |
| `fedrl_hf_fitness` | 42 / 12 pairs | 84 / 24 binary rows |

The Android runner uses the DPO-pair JSONL as its runtime input. KTO rounds
expand those pairs internally before calling the native KTO trainer. The KTO
JSONL assets are kept as validation/reference data for Python checks and smoke
runs.

Phone-collected A/B feedback remains in the app private directory and is used
instead of the bundled dataset when present.

## Phone Setup

Build and install the Android app:

```bash
cd android-visualizer
./gradlew :app:assembleDebug
cd ..

ADB="$HOME/Library/Android/sdk/platform-tools/adb"
"$ADB" devices
"$ADB" -s PHONE_A install -r android-visualizer/app/build/outputs/apk/debug/app-debug.apk
"$ADB" -s PHONE_B install -r android-visualizer/app/build/outputs/apk/debug/app-debug.apk
"$ADB" -s PHONE_C install -r android-visualizer/app/build/outputs/apk/debug/app-debug.apk
```

Each phone must have the same base model at:

```text
/data/local/tmp/mft_qwen3
```

The current wireless three-phone form is:

```bash
export FEDRL_SERIALS="10.203.22.73:5555 10.203.34.125:5556 10.203.4.60:5556"
```

## Run One Federated Round

DPO round 001:

```bash
OBJECTIVE=dpo ROUND=001 STEPS=10 SEQ_LEN=64 RANK=2 ALPHA=4 LR=0.001 \
  scripts/android/run_fedrl_round_with_monitor.sh
```

KTO round 001:

```bash
OBJECTIVE=kto ROUND=001 STEPS=10 SEQ_LEN=64 RANK=2 ALPHA=4 LR=0.001 \
  scripts/android/run_fedrl_round_with_monitor.sh
```

Round 2 and later load the previous global adapter before local training:

```bash
OBJECTIVE=dpo ROUND=002 STEPS=10 SEQ_LEN=64 RANK=2 ALPHA=4 LR=0.001 \
  PREVIOUS_GLOBAL_ADAPTER="runs/fedrl_round_001_dpo/global_dpo_lora_adapter.jsonl" \
  scripts/android/run_fedrl_round_with_monitor.sh
```

The runner writes:

```text
runs/fedrl_round_<ROUND>_<OBJECTIVE>/
  <serial>/<run_name>/{report.json,metrics.ndjson,reference_logps.jsonl,qwen_<objective>_lora_adapter.jsonl}
  monitor/<serial>.csv
  global_<objective>_lora_adapter.jsonl
  global_<objective>_lora_adapter.jsonl.manifest.json
  device_summary.csv
  summary.json
```

## Run Many Rounds

Use the same loop for 100 or 200 rounds. Change `seq 1 200` to `seq 1 100`
for a 100-round run.

```bash
export FEDRL_SERIALS="10.203.22.73:5555 10.203.34.125:5556 10.203.4.60:5556"
OBJECTIVE=dpo
PREV=""

for n in $(seq 1 200); do
  ROUND="$(printf '%03d' "$n")"
  if [ -n "$PREV" ]; then
    OBJECTIVE="$OBJECTIVE" ROUND="$ROUND" PREVIOUS_GLOBAL_ADAPTER="$PREV" \
      STEPS=10 SEQ_LEN=64 RANK=2 ALPHA=4 LR=0.001 \
      scripts/android/run_fedrl_round_with_monitor.sh
  else
    OBJECTIVE="$OBJECTIVE" ROUND="$ROUND" \
      STEPS=10 SEQ_LEN=64 RANK=2 ALPHA=4 LR=0.001 \
      scripts/android/run_fedrl_round_with_monitor.sh
  fi
  PREV="runs/fedrl_round_${ROUND}_${OBJECTIVE}/global_${OBJECTIVE}_lora_adapter.jsonl"
done
```

Set `OBJECTIVE=kto` for the KTO federation.

## Default Training Parameters

| Parameter | Value |
| --- | --- |
| Base model | Qwen3-0.6B under `/data/local/tmp/mft_qwen3` |
| Adapter | LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Rank / alpha | `2 / 4` |
| Steps per local round | `10` by default |
| Sequence length | `64` |
| Learning rate | `0.001` |
| Beta | `0.1` |
| Batch size | `4` when `steps <= 5`, otherwise `2` |
| Aggregation | FedAvg weighted by `feedback_pairs` |

## Validation

Run lightweight checks before pushing:

```bash
python3 scripts/prepare_fedrl_datasets.py
bash -n scripts/android/*.sh scripts/federated/*.sh scripts/run_kto_health_smoke.sh
python3 -m py_compile scripts/prepare_fedrl_datasets.py scripts/federated/*.py scripts/convert_dpo_to_kto_jsonl.py
cd android-visualizer && ./gradlew :app:testDebugUnitTest
```

Full Android builds require a local Android SDK/NDK and are intentionally not
committed as artifacts.
