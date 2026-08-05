#pragma once

// C++ convenience entrypoint for MobileFineTuner consumers.
//
// New language bindings should use mobile_finetuner.h, whose public surface is
// a stable C ABI. This header intentionally preserves the historical C++
// umbrella include for existing native consumers and advanced users.

#include "finetune_ops/core/dtype.h"
#include "finetune_ops/core/dpo_loss.h"
#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/memory_efficient_attention.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/tensor.h"
#include "finetune_ops/core/tokenizer.h"

#include "finetune_ops/data/batch_provider.h"
#include "finetune_ops/data/mmlu_dataset.h"
#include "finetune_ops/data/causal_lm_batch.h"
#include "finetune_ops/data/kto_batch.h"
#include "finetune_ops/data/preference_batch.h"
#include "finetune_ops/data/wikitext2_dataset.h"

#include "finetune_ops/graph/auto_model.h"
#include "finetune_ops/graph/gemma_lora_injector.h"
#include "finetune_ops/graph/gemma_model.h"
#include "finetune_ops/graph/gpt2_model.h"
#include "finetune_ops/graph/llama_model.h"
#include "finetune_ops/graph/lora_injector.h"
#include "finetune_ops/graph/lora_saver.h"
#include "finetune_ops/graph/model_registry.h"
#include "finetune_ops/graph/qwen_model.h"
#include "finetune_ops/graph/safetensors_loader.h"

#include "finetune_ops/nn/lora_linear.h"

#include "finetune_ops/optim/adam.h"
#include "finetune_ops/optim/auto_trainer.h"
#include "finetune_ops/optim/dpo_trainer.h"
#include "finetune_ops/optim/kto_trainer.h"
#include "finetune_ops/optim/preference_scoring.h"
#include "finetune_ops/optim/gemma_trainer.h"
#include "finetune_ops/optim/optimizer.h"
#include "finetune_ops/optim/trainer.h"
