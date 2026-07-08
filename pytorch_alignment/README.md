# PyTorch Alignment

This directory is the lightweight Python reference layer for alignment methods.
It is used to validate data formats, loss behavior, metrics, and backpropagation
before the native C++ or Android paths are changed.

## What Lives Here

| File | Purpose |
| --- | --- |
| `dpo_utils.py` | Shared DPO/KTO datasets, collators, sequence log-prob utilities, losses, metrics, and eval helpers. |
| `kto_trainer.py` | Reusable tiny PyTorch KTO training loop with metrics writing, gradient checks, reference freezing, and eval reporting. |
| `kto_cli.py` | Maintained CLI wrapper for Qwen3-0.6B KTO JSONL smoke via `python -m pytorch_alignment.kto_cli`. |

## Data Formats

### DPO Pair JSONL

Used for pairwise preference data:

```json
{"prompt":"User: ...\nAssistant:","chosen":" better answer","rejected":" weaker answer"}
```

### KTO Binary JSONL

Used for binary desirable/undesirable feedback:

```json
{"prompt":"User: ...\nAssistant:","response":" answer text","label":true}
{"prompt":"User: ...\nAssistant:","response":" answer text","label":false}
```

Convert DPO pairs into KTO binary records with:

```bash
python3 scripts/convert_dpo_to_kto_jsonl.py \
  --input android-visualizer/app/src/main/assets/mft_demo_data/fedrl_health_dpo_train.jsonl \
  --output android-visualizer/app/src/main/assets/mft_demo_data/fedrl_health_kto_train.jsonl
```

## Current KTO Smoke

The maintained one-command smoke entrypoint uses Qwen3-0.6B:

```bash
bash scripts/run_kto_health_smoke.sh
```

It performs three actions:

1. Converts curated health DPO pairs into KTO binary train/eval JSONL.
2. Calls `python -m pytorch_alignment.kto_cli`, which loads Qwen3-0.6B and
   delegates the training loop to `pytorch_alignment.kto_trainer`.
3. Runs a Qwen3-0.6B LoRA KTO smoke with batch size 4 for 5 steps.
4. Runs a Qwen3-0.6B LoRA KTO smoke with batch size 2 for 10 steps.

You can also call the CLI directly:

```bash
python3 -m pytorch_alignment.kto_cli \
  --backend qwen \
  --model_dir examples/qwen_lora_finetune/pretrained \
  --kto_jsonl android-visualizer/app/src/main/assets/mft_demo_data/fedrl_health_kto_train.jsonl \
  --eval_jsonl android-visualizer/app/src/main/assets/mft_demo_data/fedrl_health_kto_eval.jsonl \
  --out_dir runs/kto_qwen3_0_6b_manual \
  --max_steps 5 \
  --batch_size 4 \
  --seq_len 128 \
  --lora_rank 2 \
  --lora_alpha 4
```

Success means:

- finite KTO loss;
- nonzero trainable policy gradients;
- no reference-model gradients;
- metrics include `loss`, `reward_accuracy`, `reward_margin`,
  `desirable_reward`, `undesirable_reward`, and `kl_center`.

This smoke proves the data/loss/backprop path. It does not prove final model
quality.
