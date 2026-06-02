以下脚本封装了从数据准备到开始 LoRA 微调与评测的最小可运行流程（基于 GPT-2 small 分词器与权重）。

要求：
- 已安装 Python 依赖（pytorch、transformers、peft、safetensors 等）
- 本仓库根目录含有数据集 `data/mmlu/`
- 预训练权重：`gpt2_lora_finetune/pretrained/gpt2/`

使用顺序：
1) 生成 JSONL（masked 标签，仅答案位置参与损失）
   bash run_prepare_data.sh
   输出目录：仓库根目录 `runs/mmlu_jsonl_gpt2_s128/`

2) 启动微调（LoRA）
   bash run_train.sh
   产物：当前目录 `outputs/lora_adapter/`

3) 评测（MMLU MCQ）
   # DEV 验证
   bash run_eval.sh
   # TEST（可选 few-shot，默认 5-shot）
   FEWSHOT=5 SPLIT=test bash run_eval.sh

可调参数：
- 在 run_train.sh 中通过环境变量覆盖：STEPS、BATCH_SIZE、SEQ_LEN、LR、RANK、ALPHA、EPOCHS、LOG_EVERY
- 在 run_eval.sh 中通过环境变量覆盖：SPLIT、FEWSHOT、DEVICE

## Native DPO validation

构建：
```bash
cd /Users/anakinju/DKU/MobileFineTuner/gpt2_small_lora_finetune/build
cmake --build . --target test_dpo_native_smoke train -j2
```

运行最小 native DPO autograd smoke：
```bash
./test_dpo_native_smoke
```

期望：
- 打印有限 `loss`
- `grad_A_max` 和 `grad_B_max` 都非零

运行一条端到端 native GPT-2 DPO smoke：
```bash
./train --train_mode dpo \
  --pretrained_dir /Users/anakinju/DKU/MobileFineTuner/gpt2_small_lora_finetune/pretrained/gpt2 \
  --preference_train /Users/anakinju/DKU/MobileFineTuner/review-stage/data/synthetic_pref_longform_train.jsonl \
  --preference_valid /Users/anakinju/DKU/MobileFineTuner/review-stage/data/synthetic_pref_longform_eval.jsonl \
  --seq_len 128 \
  --batch_size 1 \
  --steps 1 \
  --log_interval 1 \
  --eval_interval 0
```

如果要避免真实偏好数据在截断后 chosen/rejected 坍塌，优先提高 `--seq_len`，再检查 `PreferenceDataset` 的 kept/skipped 统计。

Primary validation: BLAS / Accelerate full real-MMLU smoke
```bash
cd /Users/anakinju/DKU/MobileFineTuner
bash review-stage/scripts/run_native_real_mmlu_full_smoke_blas.sh
```

期望：
- 启动配置打印 `BLAS          : ON (compiled)`
- `PreferenceDataset` 显示 `222/480` train 与 `117/120` valid kept
- 首步打印有限 `dpo_loss`
- `nonzero_grad_tensors` 非零

Fallback validation: real-MMLU short smoke
```bash
cd /Users/anakinju/DKU/MobileFineTuner
bash review-stage/scripts/run_native_real_mmlu_short_smoke.sh
```

期望：
- `PreferenceDataset` 显示 `16/16` train 与 `8/8` valid kept
- 首步打印有限 `dpo_loss`
- `nonzero_grad_tensors` 非零
