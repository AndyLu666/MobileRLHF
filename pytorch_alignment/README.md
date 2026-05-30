# PyTorch 对齐基线

这个目录包含一组 PyTorch 参考实现，用来和 C++ 训练流水线做数值对齐、行为比较和迁移验证。

## 可用脚本

### GPT-2 LoRA 微调
**文件：** `gpt2_lora_finetune.py`
**对应：** `gpt2_lora_finetune/main.cpp`

这是一个基于 PyTorch + PEFT 的 GPT-2 LoRA 训练实现，主要用于和 C++ 版本做对齐验证。

**用法（SFT）：**
```bash
python pytorch_alignment/gpt2_lora_finetune.py \
  --data_dir data/wikitext2/wikitext-2-raw \
  --pretrained_dir gpt2_lora_finetune/pretrained/gpt2 \
  --lora_out pytorch_runs/gpt2_lora \
  --epochs 1 --batch_size 8 --grad_accum_steps 1 --seq_len 128 \
  --rank 8 --alpha 16 --lr 2e-4 --lora_dropout 0.0
```

**GPT-2 Medium 示例：**
```bash
python pytorch_alignment/gpt2_lora_finetune.py \
  --pretrained_dir gpt2_lora_finetune/pretrained/gpt2-medium \
  --lora_out pytorch_runs/gpt2_medium_lora \
  --batch_size 4 --epochs 1
```

**用法（DPO）：**
```bash
python pytorch_alignment/gpt2_lora_finetune.py \
  --pretrained_dir gpt2_lora_finetune/pretrained/gpt2 \
  --data_dir data/wikitext2/wikitext-2-raw \
  --train_mode dpo \
  --preference_jsonl review-stage/data/synthetic_dpo_smoke.jsonl \
  --lora_out pytorch_runs/gpt2_dpo_lora \
  --steps 10 --batch_size 2 --grad_accum_steps 1 \
  --seq_len 128 --rank 8 --alpha 16 --dpo_beta 0.1
```

---

### GPT-2 全参数微调
**文件：** `gpt2_full_finetune.py`
**用途：** 作为全参数微调对照基线

**用法：**
```bash
python pytorch_alignment/gpt2_full_finetune.py \
  --data_dir data/wikitext2/wikitext-2-raw \
  --pretrained_dir gpt2_lora_finetune/pretrained/gpt2 \
  --output_dir pytorch_runs/gpt2_full \
  --epochs 1 --batch_size 4 --seq_len 128
```

---

### Gemma LoRA 微调
**文件：** `gemma_lora_finetune.py`
**对应：** `operators/finetune_ops/optim/train_lora_gemma.cpp`

这是一个基于 PyTorch + PEFT 的 Gemma LoRA 训练实现。

**用法（SFT）：**
```bash
python pytorch_alignment/gemma_lora_finetune.py \
  --model_dir gemma-3-270m \
  --data_dir data/wikitext2/wikitext-2-raw \
  --output_dir pytorch_runs/gemma_270m_lora \
  --epochs 1 --batch 4 --grad_accum 1 --seq_len 256 \
  --learning_rate 2e-4 --warmup_ratio 0.03 \
  --target_mode full --lora_r 8 --lora_alpha 32 \
  --torch_dtype float32
```

**Gemma 1B 示例：**
```bash
python pytorch_alignment/gemma_lora_finetune.py \
  --model_dir gemma-3-1b \
  --batch 2 --grad_accum 2 \
  --output_dir pytorch_runs/gemma_1b_lora
```

**用法（DPO）：**
```bash
python pytorch_alignment/gemma_lora_finetune.py \
  --model_dir gemma-3-270m \
  --data_dir data/wikitext2/wikitext-2-raw \
  --train_mode dpo \
  --preference_jsonl review-stage/data/synthetic_dpo_smoke.jsonl \
  --output_dir pytorch_runs/gemma_dpo_lora \
  --max_steps 10 --batch 2 --seq_len 256 \
  --target_mode attn --lora_r 8 --lora_alpha 32 --dpo_beta 0.1
```

---

### Qwen2.5-0.5B LoRA 微调
**文件：** `qwen_lora_finetune.py`
**对应：** `qwen_lora_finetune/src/wikitext_main.cpp` 和 `src/mmlu_main.cpp`

默认支持 WikiText-2，也支持用于 masked causal LM 的 MMLU JSONL（`ids/mask`）模式。

**用法（WikiText-2 / SFT）：**
```bash
python pytorch_alignment/qwen_lora_finetune.py \
  --model_dir Qwen2.5-0.5B \
  --data_dir data/wikitext2/wikitext-2-raw \
  --output_dir pytorch_runs/qwen_wikitext_lora \
  --epochs 1 --batch 1 --seq_len 1024 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05
```

**用法（MMLU JSONL，ids/mask）：**
```bash
python pytorch_alignment/qwen_lora_finetune.py \
  --model_dir Qwen2.5-0.5B \
  --jsonl_train runs/mmlu_jsonl_qwen/train.jsonl \
  --jsonl_valid runs/mmlu_jsonl_qwen/valid.jsonl \
  --seq_len 128 --batch 8 --grad_accum 1 \
  --output_dir pytorch_runs/qwen_mmlu_lora \
  --target_mode qv
```

**用法（DPO）：**
```bash
python pytorch_alignment/qwen_lora_finetune.py \
  --model_dir Qwen2.5-0.5B \
  --train_mode dpo \
  --preference_jsonl review-stage/data/synthetic_dpo_smoke.jsonl \
  --output_dir pytorch_runs/qwen_dpo_lora \
  --steps 10 --batch 2 --seq_len 256 \
  --target_mode qv --dpo_beta 0.1
```

---

## DPO 偏好训练

现在三套 LoRA trainer 都支持 `--train_mode dpo`，同时保留原来的 SFT 路径。

### Preference JSONL 格式

每一行是一条偏好样本，格式如下：

```json
{"prompt":"...","chosen":"preferred answer","rejected":"less preferred answer"}
```

字段含义：
- `prompt`：共享输入上下文
- `chosen`：希望策略模型偏好的回答
- `rejected`：希望策略模型压低分数的回答

### DPO CLI 参数

- `--train_mode dpo`：把 trainer 从 token-label SFT 切换成偏好优化
- `--preference_jsonl`：必填，训练集，格式为 `(prompt, chosen, rejected)`
- `--preference_valid_jsonl`：可选，验证集；不填时默认回退到训练文件
- `--reference_model_dir`：可选，冻结 reference model 路径；默认使用底座模型
- `--dpo_beta`：DPO 温度 / margin 缩放系数

### 指标说明

在 DPO 模式下，`perplexity` 不再是核心指标。当前会记录：
- `loss`
- `reward_acc`
- `reward_margin`
- `chosen_reward`
- `rejected_reward`

### 从 SFT 迁移到 DPO 的区别

- SFT 使用 `labels`，对单个目标回答做 token-level cross entropy。
- DPO 使用成对 completion，比的是 policy 与 reference 的 log-ratio margin。
- 现有 SFT 路径默认保持兼容，不会被 DPO 改坏。

---

## 配置对齐

这些脚本默认尽量和 C++ 实现保持一致：

| 参数 | GPT-2 | Gemma |
|------|-------|-------|
| **LoRA Rank** | 8 | 8 |
| **LoRA Alpha** | 16 | 32 |
| **LoRA Targets** | AttnQKV, AttnProj | q_proj, k_proj, v_proj, o_proj |
| **Optimizer** | AdamW | AdamW |
| **LR Schedule** | Warmup + Cosine | Warmup + Linear/Cosine |
| **Gradient Clip** | 1.0 | 1.0 |
| **Data Format** | Raw text files | Raw text files |

---

## 关键特性

### 一致的训练流程
1. 从本地目录加载预训练权重
2. 用 PEFT 注入 LoRA adapter
3. 冻结 base model 参数
4. 使用一致的 tokenizer 在 WikiText-2 上训练
5. 以 SafeTensors 格式保存 LoRA adapter

### 梯度累积
所有实现都支持正确缩放的梯度累积：

```text
effective_batch_size = batch_size x grad_accum_steps
```

### 学习率调度
- **Warmup：** 前若干步线性升温
- **Decay：** GPT-2 使用 cosine 衰减到基准学习率的 10%，Gemma 使用 linear/cosine

### Loss 计算
SFT 路径仍然使用标准 shifted cross-entropy：
- Logits: `[:, :-1, :]`
- Labels: `[:, 1:]`
- 和 C++ 的 `lm_cross_entropy` 实现保持一致

---

## 验证场景

### 1. 数值对齐检查
比较 loss 曲线和最终 adapter：

```bash
# 运行 C++ 版本
./build/gpt2_lora_finetune --steps 100 --lora_out cpp_lora.safetensors

# 运行 PyTorch 版本
python pytorch_alignment/gpt2_lora_finetune.py --steps 100 --lora_out pt_lora

# 对比输出（loss 一般应在 1-2% 以内）
```

### 2. 梯度验证
在做 alignment debugging 时，可以导出梯度进行对比。

### 3. 性能基线
PyTorch 版本可以作为 C++ 优化前的速度和显存基线。

---

## 环境要求

### Python 环境
```bash
python3 -m venv pt_env
source pt_env/bin/activate
pip install torch transformers peft datasets
```

### 数据准备
WikiText-2 原始文本建议放在：

```text
data/wikitext2/wikitext-2-raw/
├── wiki.train.raw
├── wiki.valid.raw
└── wiki.test.raw
```

### 模型权重
预训练模型建议提前下载到本地：
- GPT-2：`gpt2_lora_finetune/pretrained/gpt2/`
- GPT-2 Medium：`gpt2_lora_finetune/pretrained/gpt2-medium/`
- Gemma 270M：`gemma-3-270m/`
- Gemma 1B：`gemma-3-1b/`
