# Federated DPO/KTO Phone Runbook

This runbook is for real three-phone federated preference post-training.

## Claim Language

Use this wording:

- MobileFineTuner supports two on-device preference objectives: native DPO and native KTO.
- DPO consumes `prompt/chosen/rejected` preference pairs.
- KTO consumes binary desirable/undesirable feedback; DPO-style pairs can be expanded into KTO rows.
- In federated mode, each phone trains a local LoRA adapter from private feedback, uploads only adapter weights and run metadata, and the server aggregates LoRA adapters with FedAvg.

Avoid saying raw health data, prompts, or rejected/chosen text are uploaded unless the experiment explicitly exports them.

## Android Objective Selection

The Android demo runner uses `NativeDemoConfig.dpoPairForward`:

- `false`: native KTO path, `createKtoTrainer()` + `trainKtoPreferenceBatch()`
- `true`: native DPO path, `createDpoTrainer()` + `trainPreferenceBatch()`

The DPO path writes:

```text
report.json
metrics.ndjson
reference_logps.jsonl
qwen_dpo_lora_adapter.jsonl
before_after_generation.json
before_after_eval.json
train.log
```

The KTO path writes the same structure with `objective: "kto"` and `qwen_kto_lora_adapter.jsonl`.

## Three-Phone Round

1. Build and install the Android visualizer on each phone.

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

2. Stage the same base model on all phones, for example:

```bash
scripts/android/stage_qwen_qnli_phone_assets.sh
```

3. On each phone, collect private feedback in the app.

For DPO, use A/B comparison feedback and enable the runner's `DPO objective` toggle before training.

For KTO, use single thumbs feedback or DPO-compatible pairs with `DPO objective` disabled.

4. Run one local training round on all three phones.

Recommended smoke settings before longer runs:

```text
steps=5
seq_len=64
rank=2
alpha=4
batch=4 for <=5 steps, batch=2 for >5 steps
```

The maintained end-to-end runner starts the app on each phone by adb intent,
records per-device RSS/time series, collects the exact run it started, performs
FedAvg, and writes `device_summary.csv`.

```bash
export FEDRL_SERIALS="10.203.22.73:5555 10.203.34.125:5556 10.203.4.60:5556"

OBJECTIVE=dpo ROUND=001 STEPS=10 SEQ_LEN=64 RANK=2 ALPHA=4 LR=0.001 \
  scripts/android/run_fedrl_round_with_monitor.sh

OBJECTIVE=kto ROUND=001 STEPS=10 SEQ_LEN=64 RANK=2 ALPHA=4 LR=0.001 \
  scripts/android/run_fedrl_round_with_monitor.sh
```

For round 2 and later, pass the previous global LoRA adapter:

```bash
OBJECTIVE=dpo ROUND=002 STEPS=10 SEQ_LEN=64 RANK=2 ALPHA=4 LR=0.001 \
  PREVIOUS_GLOBAL_ADAPTER="runs/fedrl_round_001_dpo/global_dpo_lora_adapter.jsonl" \
  scripts/android/run_fedrl_round_with_monitor.sh
```

For 100 or 200 rounds, keep the previous round's global adapter as the next
round's initialization:

```bash
export FEDRL_SERIALS="10.203.22.73:5555 10.203.34.125:5556 10.203.4.60:5556"
OBJECTIVE=dpo
PREV=""
for n in $(seq 1 200); do
  ROUND="$(printf '%03d' "$n")"
  if [ -n "$PREV" ]; then
    OBJECTIVE="$OBJECTIVE" ROUND="$ROUND" PREVIOUS_GLOBAL_ADAPTER="$PREV" \
      scripts/android/run_fedrl_round_with_monitor.sh
  else
    OBJECTIVE="$OBJECTIVE" ROUND="$ROUND" \
      scripts/android/run_fedrl_round_with_monitor.sh
  fi
  PREV="runs/fedrl_round_${ROUND}_${OBJECTIVE}/global_${OBJECTIVE}_lora_adapter.jsonl"
done
```

5. Manual collection path, if local training was started from the UI.

```bash
FEDRL_SERIALS="PHONE_A PHONE_B PHONE_C" OBJECTIVE=dpo ROUND=001 \
  scripts/android/collect_fedrl_round.sh

FEDRL_SERIALS="PHONE_A PHONE_B PHONE_C" OBJECTIVE=kto ROUND=001 \
  scripts/android/collect_fedrl_round.sh
```

6. Aggregate the adapters.

```bash
python3 scripts/federated/aggregate_lora_jsonl.py \
  --input runs/fedrl_round_001_dpo/PHONE_A/*/qwen_dpo_lora_adapter.jsonl \
  --input runs/fedrl_round_001_dpo/PHONE_B/*/qwen_dpo_lora_adapter.jsonl \
  --input runs/fedrl_round_001_dpo/PHONE_C/*/qwen_dpo_lora_adapter.jsonl \
  --output runs/fedrl_round_001_dpo/global_dpo_lora_adapter.jsonl
```

The default FedAvg weight is `feedback_pairs` from each phone's `report.json`. To override weights:

```bash
python3 scripts/federated/aggregate_lora_jsonl.py \
  --input phone_a/qwen_dpo_lora_adapter.jsonl::24 \
  --input phone_b/qwen_dpo_lora_adapter.jsonl::18 \
  --input phone_c/qwen_dpo_lora_adapter.jsonl::30 \
  --output global_dpo_lora_adapter.jsonl
```

7. Distribute the global adapter for the next round.

Push `global_dpo_lora_adapter.jsonl` or `global_kto_lora_adapter.jsonl` to each phone and load it before the next local round. The current SDK already supports `loadLoraAdapter(File)`.

The runner distributes the previous adapter automatically when
`PREVIOUS_GLOBAL_ADAPTER` is set:

```bash
FEDRL_SERIALS="PHONE_A PHONE_B PHONE_C" OBJECTIVE=dpo ROUND=002 \
  PREVIOUS_GLOBAL_ADAPTER="runs/fedrl_round_001_dpo/global_dpo_lora_adapter.jsonl" \
  scripts/android/run_fedrl_round_with_monitor.sh
```

The monitored runner records per-device time series in `runs/fedrl_round_<ROUND>_<OBJECTIVE>/monitor/*.csv`,
collects the exact run artifacts it started, writes `device_summary.csv`, and aggregates
`global_<OBJECTIVE>_lora_adapter.jsonl`.

## Evidence Checklist

For each objective and each phone, keep:

- `report.json` with `objective`, `feedback_pairs`, `max_steps`, `batch_size`, `lora_rank`, `lora_alpha`
- `metrics.ndjson` with finite `loss`, `reward_accuracy`, and `reward_margin`
- `reference_logps.jsonl` proving the reference side was cached before local training
- `qwen_dpo_lora_adapter.jsonl` or `qwen_kto_lora_adapter.jsonl`
- server `global_*_lora_adapter.jsonl`
- server aggregation manifest

## Paper Figure Wording

Recommended label for the training block:

```text
On-device LoRA Preference Post-Training
(DPO and KTO objectives)
```

Recommended label for the federation block:

```text
Federated LoRA aggregation
Only adapter updates leave the phone
```
