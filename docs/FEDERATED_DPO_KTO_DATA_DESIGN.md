# Federated DPO/KTO Data Design

This file defines the concrete data, model, training parameters, and run metadata
for the three-phone federated preference experiment.

## Objectives

We run two independent federated preference objectives:

| Objective | Local trainer | Local input schema | Adapter artifact |
| --- | --- | --- | --- |
| DPO | `createDpoTrainer()` + `trainPreferenceBatch()` | `prompt/chosen/rejected` | `qwen_dpo_lora_adapter.jsonl` |
| KTO | `createKtoTrainer()` + `trainKtoBatch()` | unpaired `prompt/response/label`; DPO pairs are explicitly unpaired by the data adapter | `qwen_kto_lora_adapter.jsonl` |

## Dataset Spec

Prepared formal health dataset:

```kotlin
PreferenceDatasetSpec(
    id = "fedrl_health",
    displayName = "Curated health preference dataset",
    kind = PreferenceDatasetKind.HEALTH_PREFERENCES,
    jsonlPath = "app/src/main/assets/mft_demo_data/fedrl_health_dpo_train.jsonl",
    schema = "jsonl: prompt/chosen/rejected",
    privateOnDevice = false,
    promptField = "prompt",
    chosenField = "chosen",
    rejectedField = "rejected",
    labelField = null,
    sourceDescription = "Curated health preference pairs for FedRL DPO/KTO"
)
```

Prepared files:

| Dataset | DPO train/eval | KTO train/eval |
| --- | ---: | ---: |
| `fedrl_health` | 24 / 6 pairs | 48 / 12 binary rows |
| `fedrl_hf_fitness` | 42 / 12 pairs | 84 / 24 binary rows |

Android runtime input may be either DPO-pair or KTO-binary JSONL. For
`OBJECTIVE=kto`, the data adapter keeps each `prompt/response/label` payload and
label unchanged and explicitly unpairs each `prompt/chosen/rejected` row into
one desirable and one undesirable sample. Like TRL's unpairing utility,
conversion writes the
desirable block before the undesirable block; the Android batch planner then
selects balanced, deterministic slices and prefers rotations across distinct
prompts. The runner calls canonical `trainKtoBatch()` and caches both matched
and rotated KL-branch reference scores for the exact batch order. The tracked
`*_kto_*.jsonl` files are also used by the Python smoke path and dataset checks.

Android assets:

```text
app/src/main/assets/mft_demo_data/fedrl_health_dpo_train.jsonl
app/src/main/assets/mft_demo_data/fedrl_health_dpo_eval.jsonl
app/src/main/assets/mft_demo_data/fedrl_health_kto_train.jsonl
app/src/main/assets/mft_demo_data/fedrl_health_kto_eval.jsonl
app/src/main/assets/mft_demo_data/fedrl_hf_fitness_dpo_train.jsonl
app/src/main/assets/mft_demo_data/fedrl_hf_fitness_dpo_eval.jsonl
app/src/main/assets/mft_demo_data/fedrl_hf_fitness_kto_train.jsonl
app/src/main/assets/mft_demo_data/fedrl_hf_fitness_kto_eval.jsonl
app/src/main/assets/mft_demo_data/fedrl_preference_manifest.json
```

Phone-collected formal dataset:

```kotlin
PreferenceDatasetSpec(
    id = "phone_qwen_feedback",
    displayName = "Private on-phone Qwen health preference feedback",
    kind = PreferenceDatasetKind.APP_AB_FEEDBACK,
    jsonlPath = "files/mft_runs/<run>/inference_feedback.jsonl",
    schema = "mixed jsonl: prompt/chosen/rejected or prompt/response/label",
    privateOnDevice = true,
    promptField = "prompt",
    chosenField = "chosen",
    rejectedField = "rejected",
    labelField = "label",
    sourceDescription = "A/B and thumbs feedback collected locally in the Android app"
)
```

Bundled smoke fallback:

```kotlin
PreferenceDatasetSpec(
    id = "fedrl_hf_fitness",
    displayName = "HF fitness preference pairs",
    kind = PreferenceDatasetKind.HF_FITNESS_PREFERENCES,
    jsonlPath = "app/src/main/assets/mft_demo_data/fedrl_hf_fitness_dpo_train.jsonl",
    schema = "jsonl: prompt/chosen/rejected",
    privateOnDevice = false,
    sourceDescription = "Bundled fitness preference pairs for smoke validation"
)
```

Privacy rule:

- Raw health sensor streams stay on device.
- Local prompts, responses, labels, and chosen/rejected text should be treated
  as private experiment data.
- Federated aggregation uploads only LoRA adapter weights plus run metadata.

## Model Spec

```kotlin
OnDeviceModelSpec(
    id = "qwen3_0_6b_android",
    displayName = "Qwen3-0.6B Android local model",
    family = "Qwen3",
    modelDir = "/data/local/tmp/mft_qwen3",
    tokenizerPath = null,
    quantization = null,
    baseModelFrozen = true,
    adapterType = "LoRA",
    lora = LoraSpec(
        rank = 2,
        alpha = 4,
        dropout = 0.0,
        targetModules = listOf("q_proj", "k_proj", "v_proj", "o_proj")
    )
)
```

## Training Specs

Smoke settings before longer runs:

```kotlin
PreferenceTrainingSpec(
    objective = PreferenceObjective.DPO,
    steps = 5,
    batchSize = 4,
    seqLen = 64,
    learningRate = 0.001,
    beta = 0.1,
    cacheReferenceLogps = true,
    referenceFree = false,
    evalInterval = 4,
    evalBatches = 1,
    seed = 42L
)
```

Recommended formal first round:

```kotlin
PreferenceTrainingSpec(
    objective = PreferenceObjective.DPO, // or KTO
    steps = 10,
    batchSize = 2,
    seqLen = 64,
    learningRate = 0.001,
    beta = 0.1,
    cacheReferenceLogps = true,
    referenceFree = false,
    evalInterval = 4,
    evalBatches = 1,
    seed = 42L
)
```

The Android UI currently maps this through `NativeDemoConfig`:

| Field | Value |
| --- | --- |
| `steps` | `10` formal, `5` smoke |
| `seqLen` | `64` |
| `rank` | `2` |
| `alpha` | `4` |
| `learningRate` | `0.001` |
| `dpoPairForward` | `true` for DPO, `false` for KTO |
| `globalAdapterPath` | empty for round 1, `files/federated/global_lora_adapter.jsonl` for later rounds |
| native batch | `4` when `steps <= 5`, otherwise `2` |

## Phone Specs

Current three wireless clients:

```kotlin
listOf(
    FederatedPhoneSpec(
        adbSerial = "10.203.22.73:5555",
        deviceAlias = "phone_a_iqoo",
        modelName = "V2505A",
        hardwareSerial = "10AG181KTH005F7",
        androidId = "a1d3f7bbccd924c5",
        wlanIp = "10.203.22.73"
    ),
    FederatedPhoneSpec(
        adbSerial = "10.203.34.125:5556",
        deviceAlias = "phone_b_hebe",
        modelName = "Hebe-BD00",
        hardwareSerial = "TNEBB22408200892",
        androidId = "4850072e178b0ba9",
        wlanIp = "10.203.34.125"
    ),
    FederatedPhoneSpec(
        adbSerial = "10.203.4.60:5556",
        deviceAlias = "phone_c_jad",
        modelName = "JAD-AL50",
        hardwareSerial = "T9Y0221B09002522",
        androidId = "abfd55c93e3d4ab3",
        wlanIp = "10.203.4.60"
    )
)
```

Shell form:

```bash
export FEDRL_SERIALS="10.203.22.73:5555 10.203.34.125:5556 10.203.4.60:5556"
```

## Aggregation Spec

```kotlin
FederatedAggregationSpec(
    strategy = "fedavg_jsonl_v1",
    weightKey = "feedback_pairs (DPO) / binary_feedback_rows (KTO)",
    uploadedArtifacts = listOf(
        "qwen_dpo_lora_adapter.jsonl or qwen_kto_lora_adapter.jsonl",
        "report.json"
    ),
    privateArtifacts = listOf(
        "inference_feedback.jsonl",
        "reference_logps.jsonl",
        "metrics.ndjson",
        "train.log"
    )
)
```

FedAvg command:

```bash
python3 scripts/federated/aggregate_lora_jsonl.py \
  --input runs/fedrl_round_001_dpo/10.203.22.73:5555/*/qwen_dpo_lora_adapter.jsonl \
  --input runs/fedrl_round_001_dpo/10.203.34.125:5556/*/qwen_dpo_lora_adapter.jsonl \
  --input runs/fedrl_round_001_dpo/10.203.4.60:5556/*/qwen_dpo_lora_adapter.jsonl \
  --output runs/fedrl_round_001_dpo/global_dpo_lora_adapter.jsonl
```

The same shape applies to KTO with `OBJECTIVE=kto` and `qwen_kto_lora_adapter.jsonl`.

## Run Spec Shape

```kotlin
FederatedPreferenceRunSpec(
    roundId = "001",
    objective = PreferenceObjective.DPO,
    dataset = phoneFeedbackDataset,
    model = qwenAndroidModel,
    training = dpoTrainingSpec,
    phones = threeWirelessPhones,
    aggregation = FederatedAggregationSpec(),
    artifacts = PreferenceRunArtifactSpec()
)
```

Required evidence per phone:

- `report.json`
- `metrics.ndjson`
- `reference_logps.jsonl`
- `qwen_dpo_lora_adapter.jsonl` or `qwen_kto_lora_adapter.jsonl`
- `train.log`

Required server evidence per objective:

- `global_dpo_lora_adapter.jsonl` / `global_kto_lora_adapter.jsonl`
- aggregation manifest
