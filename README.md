# MobileRLHF

MobileRLHF is a real-device federated preference post-training stack for
MobileFineTuner. The current maintained path runs DPO and KTO LoRA training on
Android phones, collects only local adapter updates and run metadata, and
aggregates adapters with FedAvg on the host.

## What Is Included

| Path | Purpose |
| --- | --- |
| `android-visualizer/` | Android app, SDK module, JNI bridge, and phone-side DPO/KTO runner. |
| `operator/` | Native C++ training core plus stable C ABI used by downstream integrations. |
| `scripts/android/` | adb automation for installing, starting, monitoring, collecting, and distributing adapters. |
| `scripts/federated/` | FedAvg aggregation and round summarization. |
| `scripts/prepare_fedrl_datasets.py` | Validates DPO data and regenerates KTO JSONL assets. |
| `android-visualizer/app/src/main/assets/mft_demo_data/` | Small tracked DPO/KTO JSONL assets for clean-clone demos. |
| `docs/FEDERATED_DPO_KTO_PHONE_RUNBOOK.md` | Full phone runbook. |
| `docs/FEDERATED_DPO_KTO_DATA_DESIGN.md` | Dataset, model, parameter, and artifact schema. |

Generated outputs are intentionally not tracked. Local runs go under `runs/`,
which is ignored by Git.

## Integration Guide

### 1. Choose an entry point

| Consumer | Supported entry point | Recommended use |
| --- | --- | --- |
| C or another FFI language | [`mobile_finetuner.h`](operator/include/mobile_finetuner/mobile_finetuner.h) | Stable opaque-handle ABI for cross-repository integrations and language bindings. |
| C++ | [`mobile_finetuner.hpp`](operator/include/mobile_finetuner/mobile_finetuner.hpp) | C++ umbrella API for native applications. |
| Android | [`android-visualizer/mft-sdk`](android-visualizer/mft-sdk) | AAR with a Java API, JNI bridge, and the native training core. |
| Federated phone experiments | [`scripts/android/`](scripts/android) | ADB orchestration scripts for multi-device experiments, artifact collection, and adapter distribution. |

New direct native/FFI bindings should use the stable C API instead of depending
on internal `finetune_ops/*` classes; Android applications may use the packaged
Java/AAR API. The current C ABI version is `MFT_API_VERSION == 1`. The header
contains declarations only: copying `mobile_finetuner.h` into another project
is not sufficient. Native consumers must also link
`MobileFineTuner::operators`, while Android consumers can use the packaged AAR.

### 2. Prepare the model assets

MobileFineTuner does not bundle or download model weights. Before opening a
model, place a Hugging Face-style snapshot in a local directory readable by the
application:

```text
/path/to/Qwen3-0.6B/
  config.json
  tokenizer.json                 # or the tokenizer files required by the family
  tokenizer_config.json          # when supplied by the snapshot
  model.safetensors              # or an index plus all referenced shards
```

The native and Android APIs accept the directory path, not a model ID or a
single weight-file path. See [`docs/MODEL_ASSETS.md`](docs/MODEL_ASSETS.md) for
the complete model-family and asset contract.

> **Format boundary:** `llama.cpp`-style describes the shape of the C API only.
> MobileFineTuner does not currently load GGUF models or GGUF adapters. If an
> application already uses `model.gguf` for inference, keep that inference
> asset separate from the Hugging Face/SafeTensors assets used for fine-tuning.
> The public C and Android APIs save LoRA adapters separately in MobileFineTuner's
> versioned JSONL format. That JSONL adapter is not directly loadable by a
> GGUF/`llama.rn` inference path.

### 3. Build and link the native library

Build and install the native library from a clean clone. The commands below
require CMake 3.15 or newer:

```bash
cmake -S operator -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX=/absolute/path/to/mft-install
cmake --build build/native --config Release -j
cmake --install build/native --config Release
```

In a downstream CMake project:

```cmake
cmake_minimum_required(VERSION 3.16)
project(mft_consumer LANGUAGES C CXX)

find_package(MobileFineTuner 2.1 REQUIRED)
add_executable(mft_consumer main.c)
set_target_properties(mft_consumer PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
  LINKER_LANGUAGE CXX
)
target_link_libraries(mft_consumer PRIVATE MobileFineTuner::operators)
```

Configure that consumer with the installation prefix:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/absolute/path/to/mft-install
cmake --build build
```

When both projects are in one source tree, the operator can instead be vendored
with `add_subdirectory(...)` and linked through the same
`MobileFineTuner::operators` target.

### 4. Use the C API

Every configuration structure must come from its matching
`mft_*_default_params()` helper. Check every returned `mft_status`, read
`mft_last_error()` on failure, and release the opaque handles when finished.
The minimal SFT lifecycle is:

```c
#include <mobile_finetuner/mobile_finetuner.h>
#include <stdio.h>
#include <string.h>

#define MFT_CHECK(call) do {                                              \
    mft_status status_ = (call);                                          \
    if (status_ != MFT_STATUS_OK) {                                       \
        fprintf(stderr, "%s: %s (%s)\n", #call,                          \
                mft_status_string(status_), mft_last_error());             \
        goto cleanup;                                                     \
    }                                                                     \
} while (0)

int main(int argc, char **argv) {
    int exit_code = 1;
    mft_model *model = NULL;
    mft_tokenizer *tokenizer = NULL;
    mft_trainer *trainer = NULL;

    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_DIR ADAPTER.jsonl\n", argv[0]);
        return 2;
    }

    mft_model_params model_params = mft_model_default_params();
    mft_tokenizer_params tokenizer_params = mft_tokenizer_default_params();
    MFT_CHECK(mft_model_load_from_dir(argv[1], &model_params, &model));
    MFT_CHECK(mft_tokenizer_load_from_dir(
        argv[1], &tokenizer_params, &tokenizer));

    mft_lora_params lora = mft_lora_default_params();
    MFT_CHECK(mft_model_init_lora(model, &lora));

    mft_sft_trainer_params trainer_params = mft_sft_trainer_default_params();
    MFT_CHECK(mft_sft_trainer_create(model, &trainer_params, &trainer));

    const char *line = "Question: How should I recover? Answer: Rest and hydrate.";
    mft_string_view text = {line, strlen(line)};
    mft_causal_batch_params batch = mft_causal_batch_default_params();
    batch.sequence_length = 64;
    batch.append_eos = true;

    mft_sft_step_result result = {0};
    MFT_CHECK(mft_sft_train_text_batch(
        trainer, tokenizer, &text, 1, &batch, &result));

    mft_trainer_free(trainer);
    trainer = NULL;
    MFT_CHECK(mft_model_save_lora_adapter_jsonl(model, argv[2]));
    printf("loss=%f adapter=%s\n", result.loss, argv[2]);
    exit_code = 0;

cleanup:
    mft_trainer_free(trainer);
    mft_tokenizer_free(tokenizer);
    mft_model_free(model);
    return exit_code;
}
```

The objective-specific sequence is:

| Objective | Required sequence |
| --- | --- |
| SFT | Load model/tokenizer, initialize LoRA, create an SFT trainer, then train text or token batches. |
| DPO | Obtain frozen-reference chosen/rejected log-probabilities, initialize LoRA, create a DPO trainer, then train the same preference batch. |
| KTO | Score the exact ordered binary-feedback batch before changing LoRA, initialize LoRA, create a KTO trainer, then reuse both cached score arrays with the same batch settings. |

Only one trainer may own a policy model at a time. Keep the tokenizer alive for
text and batch calls. Load an existing adapter after initializing LoRA and
before creating a trainer; save the updated adapter after training. Then free
the trainer, tokenizer, and model. See
[`docs/C_API.md`](docs/C_API.md) for error/buffer contracts and a complete
canonical KTO example.

### 5. Use the Android SDK

Build the release AAR:

```bash
bash scripts/android/build_mft_sdk_aar.sh
```

The output is
`android-visualizer/mft-sdk/build/outputs/aar/mft-sdk-release.aar`. To consume
it through Gradle, publish it to the repository-local Maven directory:

```bash
bash scripts/android/publish_mft_sdk_local.sh
```

Then add the repository and dependency to the Android application:

```kotlin
repositories {
    maven {
        url = uri("/absolute/path/to/MobileRLHF/android-visualizer/mft-sdk/build/repo")
    }
}

dependencies {
    implementation("com.mobilefinetuner:mobilefinetuner-android:0.2.0")
}
```

Minimal Java call flow:

```java
import com.mobilefinetuner.sdk.MobileFineTuner;
import java.io.File;

File modelDir = new File(context.getFilesDir(), "models/Qwen3-0.6B");
File adapterFile = new File(context.getFilesDir(), "mft_lora_adapter.jsonl");
try (MobileFineTuner mf = MobileFineTuner.open(
        modelDir.getAbsolutePath(), true)) {
    mf.initLora(MobileFineTuner.LoraConfig.attentionQkvo());
    mf.createTrainer(MobileFineTuner.TrainerConfig.defaults());
    MobileFineTuner.TrainStepResult result = mf.trainTextBatch(
            new String[]{"A training sentence."},
            64,
            true
    );
    mf.saveLoraAdapter(adapterFile); // use a .jsonl path
}
```

Place the model directory and output adapter in storage readable by the app;
app-private storage is recommended for production. The packaged Android path
is:

```text
Android application
  -> com.mobilefinetuner.sdk.MobileFineTuner
  -> libmobilefinetuner_jni.so
  -> MobileFineTuner C++ core
```

The existing AAR's JNI layer calls the C++ core directly; it is not a wrapper
around the newer public C ABI. A direct language binding instead follows:

```text
Swift / Rust / Python / another native binding
  -> mobile_finetuner.h (stable C ABI)
  -> mobile_finetuner_c.cpp
  -> MobileFineTuner C++ core
```

For FedCampus or another React Native application, the application repository
still owns the Android native-module/TurboModule binding:

```text
React Native TypeScript
  -> app-owned Android native module
  -> MobileFineTuner Java API
  -> JNI
  -> MobileFineTuner C++ core
```

This repository supplies the stable C API and the Android AAR/JNI layer; it does
not currently ship a ready-made React Native or iOS package. Native calls are
synchronous and CPU-oriented, so a React Native integration should run them off
the JavaScript/UI thread and serialize access to each native handle. See
[`docs/ANDROID_SDK.md`](docs/ANDROID_SDK.md) for Android requirements, storage,
call order, and device smoke tests.

Adapter aggregation does not require a C/C++ binding. In the supplied host-side
workflow, compatible local JSONL adapters and run metadata can be collected,
aggregated, and redistributed as a global JSONL adapter for the next round. A
production app/server upload and distribution path remains integration work for
the consuming system, such as FedCampus.

### 6. Integration checklist

- Choose the stable C API, C++ convenience API, or Android AAR.
- Provision a readable local Hugging Face/SafeTensors model directory.
- Keep any existing GGUF inference assets on a separate path.
- Cache reference scores before LoRA changes for DPO/KTO.
- Check every status/error and respect opaque-handle lifetimes.
- Treat native calls as synchronous and CPU-oriented; do not use one handle
  concurrently from multiple threads.
- Package the current Android SDK for `arm64-v8a`; other Android ABIs are not
  validated yet.
- Save or load adapters through the versioned JSONL adapter API.
- Run the native or Android smoke tests before connecting the app workflow.

Additional references:

- [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md): native and Android quick start.
- [`docs/PUBLIC_API.md`](docs/PUBLIC_API.md): supported C++ surface and objective details.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): library and packaging architecture.
- [`docs/FEDERATED_DPO_KTO_PHONE_RUNBOOK.md`](docs/FEDERATED_DPO_KTO_PHONE_RUNBOOK.md): complete multi-phone workflow.

## DPO and KTO

MobileRLHF supports two preference objectives on device:

| Objective | Android native trainer | Input view | Output adapter |
| --- | --- | --- | --- |
| DPO | `createDpoTrainer()` + `trainPreferenceBatch()` | `prompt/chosen/rejected` pairs | `qwen_dpo_lora_adapter.jsonl` |
| KTO | `createKtoTrainer()` + `trainKtoBatch()` | unpaired `prompt/response/label` rows; pairs are explicitly unpaired by the data adapter | `qwen_kto_lora_adapter.jsonl` |

Both objectives cache base-model reference log-probs, initialize a LoRA adapter,
optionally load the previous global LoRA adapter, train locally, then save a
JSONL LoRA adapter for aggregation. Canonical KTO caches both the matched score
and the rotated mismatched-completion KL score for each exact ordered batch.

Native and FFI consumers should follow the integration guide above. The C API
uses a stable opaque-handle boundary, while the C++ header exposes the
convenience surface over the same training core.

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

The Android runner accepts either DPO-pair JSONL or KTO binary JSONL. KTO rounds
preserve binary rows directly; a pair is explicitly converted into one
desirable and one undesirable row before canonical KTO batching. The tracked
KTO JSONL assets are also used by Python checks and smoke runs.

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

For the adb-driven experiment runner, each phone must have the same base model
at:

```text
/data/local/tmp/mft_qwen3
```

Set `FEDRL_SERIALS` to the serials reported by `adb devices`, for example:

```bash
export FEDRL_SERIALS="<PHONE_A_SERIAL> <PHONE_B_SERIAL> <PHONE_C_SERIAL>"
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
export FEDRL_SERIALS="<PHONE_A_SERIAL> <PHONE_B_SERIAL> <PHONE_C_SERIAL>"
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
| Aggregation | FedAvg weighted by `feedback_pairs` for DPO or `binary_feedback_rows` for KTO |

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
