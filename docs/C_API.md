# Stable C API

MobileFineTuner 2.1 exposes a C11-compatible, `llama.cpp`-style public
boundary from:

```c
#include <mobile_finetuner/mobile_finetuner.h>
```

The C ABI version is `MFT_API_VERSION == 1`. Runtime callers can compare it
with `mft_api_version()` and can display the library release returned by
`mft_version_string()`.

This interface loads a Hugging Face model **directory** containing
`config.json`, tokenizer assets, and SafeTensors weights. It does not load GGUF
files. The current native execution path is synchronous and CPU-oriented.

## Build and link

```cmake
find_package(MobileFineTuner 2.1 REQUIRED)
target_link_libraries(my_app PRIVATE MobileFineTuner::operators)
```

The implementation is C++, so a pure C executable must ultimately be linked
with the C++ runtime. CMake handles that transitively through the exported
target.

Existing C++ consumers should include the convenience umbrella instead:

```cpp
#include <mobile_finetuner/mobile_finetuner.hpp>
```

For source compatibility, including `mobile_finetuner.h` from C++ still pulls
in that umbrella unless `MFT_C_API_ONLY` is defined.

## Handle lifecycle

The API uses three opaque handles:

- `mft_model`: model graph, weights, and optional LoRA parameters.
- `mft_tokenizer`: tokenizer loaded from a model directory.
- `mft_trainer`: one SFT, DPO, or KTO optimizer session.

Destroy handles with `mft_model_free`, `mft_tokenizer_free`, and
`mft_trainer_free`. Passing `NULL` to a destroy function is safe. A trainer
retains its model state, but the tokenizer passed to text/batch operations must
remain alive for the duration of each call. Do not use one handle concurrently
from multiple threads.

Only one active trainer may own a policy model at a time. Initialize LoRA
before creating a trainer. Adapter load/save operations use the versioned
`mobile_finetuner.lora_adapter_jsonl.v1` format; loading is transactional and
saving replaces the destination atomically on supported POSIX filesystems.

## Errors and output buffers

Every fallible function returns `mft_status`. On failure,
`mft_last_error()` returns a thread-local diagnostic string that remains valid
until the next fallible API call on that thread. `mft_get_last_error()` copies
it into caller-owned storage.

Tokenizer encode/decode operations use caller-owned buffers. Pass a null
buffer with zero capacity to query the required size. A non-empty result uses
`MFT_STATUS_BUFFER_TOO_SMALL` for that query. Decode and generation byte counts
exclude the terminating NUL, so allocate `required + 1` bytes.

All parameter structures start with `struct_size`. Initialize them with the
matching `mft_*_default_params()` function, then override only the desired
fields. Do not zero-initialize a parameter structure in place of calling its
default helper. The library accepts shorter structures from older callers and
fills a missing tail from current defaults; a newer caller's unknown tail is
ignored.

## Canonical KTO example

KTO consumes unpaired binary feedback: one prompt, one completion, and one
desirable/undesirable label per row. A canonical batch contains at least two
rows. MobileFineTuner deterministically rotates completions within the ordered
batch to build the mismatched KL-estimation branch.

Reference scores must be captured **before** LoRA changes the policy. Both
`ref_logp` and `ref_kl_logp` are tied to the exact sample order, batch
membership, tokenizer settings, and sequence length used during scoring.

```c
#include <mobile_finetuner/mobile_finetuner.h>
#include <stdio.h>
#include <string.h>

static mft_string_view sv(const char *text) {
    mft_string_view value = {text, strlen(text)};
    return value;
}

int run_kto(const char *model_dir) {
    mft_model *model = NULL;
    mft_tokenizer *tokenizer = NULL;
    mft_trainer *trainer = NULL;

    mft_model_params model_params = mft_model_default_params();
    mft_tokenizer_params tokenizer_params = mft_tokenizer_default_params();
    if (mft_model_load_from_dir(model_dir, &model_params, &model) != MFT_STATUS_OK ||
        mft_tokenizer_load_from_dir(model_dir, &tokenizer_params, &tokenizer) != MFT_STATUS_OK) {
        fprintf(stderr, "%s\n", mft_last_error());
        goto fail;
    }

    mft_kto_sample rows[2] = {
        {sv("How should I recover?"), sv("Take an easy walk."),
         MFT_KTO_DESIRABLE, false, 0.0f, 0.0f},
        {sv("How should I train?"), sv("Ignore fatigue and sprint."),
         MFT_KTO_UNDESIRABLE, false, 0.0f, 0.0f},
    };
    mft_kto_batch_params batch = mft_kto_batch_default_params();
    batch.sequence_length = 64;
    batch.append_eos_to_completion = true;

    float ref_logps[2];
    float ref_kl_logps[2];
    size_t score_count = 0;
    int32_t response_tokens = 0;
    int32_t kl_response_tokens = 0;
    if (mft_score_kto_batch(
            model, tokenizer, rows, 2, &batch,
            ref_logps, ref_kl_logps, 2, &score_count,
            &response_tokens, &kl_response_tokens) != MFT_STATUS_OK) {
        fprintf(stderr, "%s\n", mft_last_error());
        goto fail;
    }
    for (size_t i = 0; i < score_count; ++i) {
        rows[i].has_reference_logps = true;
        rows[i].ref_logp = ref_logps[i];
        rows[i].ref_kl_logp = ref_kl_logps[i];
    }

    mft_lora_params lora = mft_lora_default_params();
    if (mft_model_init_lora(model, &lora) != MFT_STATUS_OK) {
        fprintf(stderr, "%s\n", mft_last_error());
        goto fail;
    }
    mft_kto_trainer_params kto = mft_kto_trainer_default_params();
    if (mft_kto_trainer_create(model, NULL, &kto, &trainer) != MFT_STATUS_OK) {
        fprintf(stderr, "%s\n", mft_last_error());
        goto fail;
    }

    mft_kto_step_result result;
    if (mft_kto_train_batch(
            trainer, tokenizer, rows, 2, &batch, &result) != MFT_STATUS_OK) {
        fprintf(stderr, "%s\n", mft_last_error());
        goto fail;
    }
    printf("loss=%f kl=%f samples=%d\n",
           result.loss, result.kl_estimate, result.sample_count);

    mft_trainer_free(trainer);
    mft_tokenizer_free(tokenizer);
    mft_model_free(model);
    return 0;

fail:
    mft_trainer_free(trainer);
    mft_tokenizer_free(tokenizer);
    mft_model_free(model);
    return 1;
}
```

`mft_kto_train_preference_batch_compat()` is retained only for old
chosen/rejected callers. It expands each pair into two labeled rows but keeps
the historical matched-response approximation; new integrations should use
`mft_score_kto_batch()` and `mft_kto_train_batch()`.

## Capability boundary

The stable API currently covers model/tokenizer loading, LoRA initialization,
SFT, DPO, canonical KTO, JSONL LoRA adapter round trips, and batch-one text
generation. Supported graph/tokenizer combinations are GPT-2, Gemma, Llama,
and Qwen variants validated by the repository tests. Mistral may be recognized
from `config.json` but is rejected until its graph and tokenizer path is
validated.

This interface is a MobileFineTuner contract. A downstream application can
integrate it before or after its own branch work without changing this ABI.
