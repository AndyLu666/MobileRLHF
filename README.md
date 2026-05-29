# MobileFineTuner

**一个面向手机端大语言模型微调的端到端统一框架**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-iOS%20%7C%20Android-green.svg)]()
[![C++](https://img.shields.io/badge/C%2B%2B-17-orange.svg)]()

<img width="2250" height="1370" alt="framework" src="https://github.com/user-attachments/assets/b4a6768c-d644-4ec8-8d3c-10d6ada04c3e" />

<img width="1992" height="1112" alt="image" src="https://github.com/user-attachments/assets/35cb5c4e-a3bb-4a21-aac2-d38d50899a17" />

https://github.com/user-attachments/assets/6ba63485-34ba-42b6-b9b1-ff980717c77b

## 论文

论文已发布在 arXiv：

**MobileFineTuner: A Unified End-to-End Framework for Fine-Tuning LLMs on Mobile Phones**  
Jiaxiang Geng*, Lunyu Zhao*, Yiyi Lu*, and Bing Luo

[![arXiv](https://img.shields.io/badge/arXiv-2512.08211-b31b1b.svg)](https://arxiv.org/abs/2512.08211)

如果这个项目对你有帮助，可以引用论文。

## 项目简介

MobileFineTuner 是一个开源框架，目标是在普通手机上直接完成大语言模型微调。训练数据留在本地设备，不依赖桌面端模拟，也不强制依赖 Python 运行时，适合做隐私敏感场景下的个性化模型适配。

当前仓库同时包含：

- 原生 C++17 训练与算子实现
- GPT-2、Gemma、Qwen 等模型的 LoRA / 全参数微调入口
- 用于数值对齐和迁移验证的 PyTorch 参考实现
- Android 可视化监控面板

## 主要特性

- **纯 C++ 训练链路**：包含自动求导、反向传播、LoRA 注入和训练调度
- **面向移动端约束优化**：提供内存分片、梯度累积、能耗感知调度等能力
- **多模型支持**：覆盖 GPT-2、Gemma、Qwen 等主流 decoder-only 架构
- **对齐验证完善**：保留 PyTorch baseline，便于对照 loss、梯度和 adapter 结果
- **配套可视化**：`android-visualizer/` 可实时查看训练日志、loss、RSS 和事件流

## 目录说明

- `operator/`：核心 C++ 算子、训练器、图结构与测试目标
- `gpt2_small_lora_finetune/`、`gpt2_medium_lora_finetune/`：GPT-2 训练入口
- `gemma_3_270m_lora_finetune/`、`gemma_3_1b_pt_lora_finetune/`：Gemma 训练入口
- `qwen_lora_finetune/`：Qwen 训练入口
- `pytorch_alignment/`：PyTorch 对齐基线与 DPO/SFT 实现
- `tests/`：Python 单元测试
- `android-visualizer/`：Android 训练监控面板
- `scripts/`：对齐、benchmark、批处理脚本

## 环境要求

### C++ 部分

- C++17 或更高版本
- CMake 3.10 及以上
- pthread
- 可选 BLAS：macOS 下可用 Accelerate，Linux 下可用 OpenBLAS / MKL

### Python 部分

- Python 3.10 及以上
- 若要运行对齐脚本或 Python 测试，需要安装 `torch`、`transformers`、`peft`

推荐使用虚拟环境：

```bash
python3 -m venv pt_env
source pt_env/bin/activate
pip install -U pip
pip install torch transformers peft datasets
```

## 构建方法

根目录下的 C++ 核心库位于 `operator/`，不是 `operators/`。

```bash
cd operator
mkdir -p build
cd build
cmake .. -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

常见产物包括：

- `operators` 静态库
- `gpt2_lora_finetune`
- `gpt2_full_finetune`
- `train_lora_gemma`
- `qwen_lora_finetune`
- 若开启测试，还会生成若干 `test_*` 可执行文件

## 快速开始

### 1. 准备数据和模型

WikiText-2 原始文本默认放在：

```bash
data/wikitext2/wikitext-2-raw/
```

目录中应至少包含：

- `wiki.train.raw`
- `wiki.valid.raw`
- `wiki.test.raw`

各模型的 Hugging Face 权重需要分别放到对应目录，例如：

- `gpt2_small_lora_finetune/pretrained/gpt2/`
- `gpt2_medium_lora_finetune/pretrained/gpt2-medium/`
- `gemma_3_270m_lora_finetune/pretrained/`
- `qwen_lora_finetune/pretrained/`

### 2. 运行微调

以 GPT-2 Small 的 WikiText-2 LoRA 为例：

```bash
./operator/build/gpt2_lora_finetune \
  --data_dir data/wikitext2/wikitext-2-raw \
  --pretrained_dir gpt2_small_lora_finetune/pretrained/gpt2 \
  --lora_out runs/gpt2_small_wt2_lora.safetensors \
  --epochs 1 --batch_size 4 --grad_accum_steps 2 --seq_len 128 \
  --rank 8 --alpha 16 --lr 2e-4 --warmup_steps 100
```

Gemma 270M：

```bash
./operator/build/train_lora_gemma \
  --model_dir gemma_3_270m_lora_finetune/pretrained \
  --data_dir data/wikitext2/wikitext-2-raw \
  --output_dir runs/gemma_270m_wt2_lora \
  --epochs 1 --batch 4 --grad_accum 1 --seq_len 256 \
  --learning_rate 2e-4 --warmup_ratio 0.03 \
  --lora_r 8 --lora_alpha 32 --targets full
```

Qwen2.5-0.5B：

```bash
./operator/build/qwen_lora_finetune \
  --data_dir data/wikitext2/wikitext-2-raw \
  --pretrained_dir qwen_lora_finetune/pretrained \
  --lora_out runs/qwen_wt2_lora.safetensors \
  --seq_len 1024 --batch_size 1 --grad_accum_steps 1 \
  --rank 8 --alpha 16 --lr 2e-4
```

更多 PyTorch 对齐示例见 [pytorch_alignment/README.md](/Users/anakinju/DKU/MobileFineTuner/pytorch_alignment/README.md)。

## 测试怎么跑

这个仓库目前至少有三类测试入口。

### 1. Python 单测

当前现成的 Python 测试是 `tests/test_dpo_utils.py`：

```bash
python -m unittest tests.test_dpo_utils
```

如果你的环境还没装依赖，先执行：

```bash
pip install torch transformers peft
```

### 2. C++ 测试

`operator/CMakeLists.txt` 默认 `BUILD_TESTS=ON`。先构建：

```bash
cd operator
mkdir -p build
cd build
cmake .. -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

如果已经注册到 CTest，可以直接跑：

```bash
ctest --output-on-failure
```

仓库里也有一些独立测试二进制，例如：

- `test_gemma_config`
- `test_gemma_forward`
- `test_tokenizer_gemma`
- `test_qkt_softmax_grad`
- `test_rmsnorm_unit`
- `test_train_minimal`

这些通常会在 `operator/build/` 下生成，必要时可以单独执行：

```bash
./test_gemma_config
```

### 3. Android 单测

```bash
cd android-visualizer
./gradlew :app:testDebugUnitTest
```

如果要顺带跑 lint 和打 debug 包：

```bash
./gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebug
```

## PyTorch 对齐

`pytorch_alignment/` 目录提供了 GPT-2、Gemma、Qwen 的 PyTorch 参考实现，用来和 C++ 路径做：

- loss 曲线对齐
- 梯度行为对齐
- LoRA adapter 导出结果对齐
- SFT / DPO 迁移验证

这个子目录的 README 已经是中文文档。

## Android 可视化

`android-visualizer/` 是配套的 Android 监控面板，可以实时查看：

- 训练进度
- loss / ppl / lr / RSS
- 日志流
- run 对比结果

构建命令：

```bash
cd android-visualizer
./gradlew assembleDebug
```

## 支持的模型

- GPT-2 Small / Medium
- Gemma 3 270M
- Gemma 3 1B-PT
- Qwen2.5-0.5B

## 引用

```bibtex
@article{geng2025mobilefinetuner,
  title={MobileFineTuner: A Unified End-to-End Framework for Fine-Tuning LLMs on Mobile Phones},
  author={Geng, Jiaxiang and Zhao, Lunyu and Lu, Yiyi and Luo, Bing},
  journal={arXiv preprint arXiv:2512.08211},
  year={2025}
}
```

## License

本项目采用 Apache 2.0 许可证，详见 [LICENSE](/Users/anakinju/DKU/MobileFineTuner/LICENSE)。
