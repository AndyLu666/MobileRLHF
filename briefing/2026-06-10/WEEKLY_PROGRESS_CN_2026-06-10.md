# 本周进度汇报稿

日期：`2026-06-10`

## 一、这周主线是什么

这周的重点从“DPO 算法能不能跑通”进一步推进到“能不能在 Android 上像一个产品 demo 一样跑起来”。

上周我们主要证明了 native `C++ DPO` 训练链路本身是通的：preference dataset 能读，policy/reference 能打分，`DPO loss` 能算，backward 能传到 LoRA 参数。

这周的目标更偏系统和演示：

- 把 native trainer 接进 Android app，而不是只靠命令行。
- 让用户可以在 app 里选择训练参数和启动 demo。
- 让 app 里能看到运行状态、日志、图表和指标。
- 解决真机上 native executable 的安装和执行权限问题。
- 解决 Android arm64 版本没有 BLAS 导致训练极慢的问题。
- 准备一套可以在 Android Studio / AVD / 真机上演示的完整流程。

所以这周的核心结论可以这样讲：

`我们已经把 MobileFineTuner 从命令行 native DPO 验证推进到了 Android app 内可操作的 on-device DPO demo，并且把 Android arm64 版 BLAS 打开了。`

## 二、Android app 现在能做什么

app 里现在有一个 `Native Demo Runner`。它不是单纯展示旧日志，而是可以从 app 内部启动 native trainer。

当前入口做了这些事：

1. 在 app 私有目录里创建 run folder。
2. 写入内置的 fitness/health preference JSONL。
3. 从 APK 的 native library 目录找到 `libmft_train.so`。
4. 组装 native GPT-2 DPO 命令。
5. 后台启动 native process。
6. 把 stdout/stderr 写入 `train.log`。
7. 轮询 `train.log` 和 `metrics.ndjson`。
8. 更新 Home / Train / Versus / Logs 的 UI。
9. 支持 Stop，向 native pid 发送 TERM/KILL。

关键代码：

- app 启动 native run:
  - [ExperimentViewModel.kt](/Users/anakinju/DKU/MobileFineTuner/android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/viewmodel/ExperimentViewModel.kt:211)
- native shell command builder:
  - [NativeDemoCommandBuilder.kt](/Users/anakinju/DKU/MobileFineTuner/android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/data/NativeDemoCommandBuilder.kt:20)
- 参数 model:
  - [TrainingModels.kt](/Users/anakinju/DKU/MobileFineTuner/android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/model/TrainingModels.kt:121)
- UI runner:
  - [DashboardScreen.kt](/Users/anakinju/DKU/MobileFineTuner/android-visualizer/app/src/main/java/com/mobilefinetuner/visualizer/ui/DashboardScreen.kt:1681)

## 三、现在 app 里用的是什么数据集

当前 app demo 用的是内置的 fitness/health preference 数据。

app 启动 demo 时会写两个文件：

- `files/mft_demo_data/hf_fitness_pref_train.jsonl`
- `files/mft_demo_data/hf_fitness_pref_eval.jsonl`

当前内置数据规模是：

- train: `5` 条 preference pairs
- eval: `2` 条 preference pairs

每条数据都是这种结构：

```json
{
  "prompt": "What are the signs of overtraining and how can I prevent it?",
  "chosen": "Watch for persistent fatigue, declining performance, poor sleep...",
  "rejected": "The main sign of overtraining is laziness. Push harder every day...",
  "source": "android_builtin_fitness_demo"
}
```

这里的数据不是 MMLU letter choice，而是更接近健康建议 / fitness advice 的长回答偏好对。它更适合演示 HFRL 的直觉：用户更偏好安全、具体、合理的建议，而不是危险或草率的回答。

这和 Hugging Face 上下载过的 `victor203/fitness-preferences` 是同一类方向。当前本地已经有转换后的外部数据文件：

- [review-stage/data/hf_fitness_pref_train.jsonl](/Users/anakinju/DKU/MobileFineTuner/review-stage/data/hf_fitness_pref_train.jsonl:1)
- [review-stage/data/hf_fitness_pref_eval.jsonl](/Users/anakinju/DKU/MobileFineTuner/review-stage/data/hf_fitness_pref_eval.jsonl:1)
- report:
  - [hf_fitness_pref_report.json](/Users/anakinju/DKU/MobileFineTuner/review-stage/data/hf_fitness_pref_report.json:1)

本周 app 里先用 5+2 内置数据，是为了演示稳定和可控。下一步可以把 app demo 改成从外部 fitness preference 文件里采样更多条，而不是只用内置小集合。

## 四、为什么 `test_dpo_native_smoke` 很快，而 app demo 加载慢

这个问题非常关键，因为两者名字都像 DPO，但实际工作量完全不同。

### `test_dpo_native_smoke` 是什么

`test_dpo_native_smoke` 是一个 C++ 单元烟测。它只做了很小的事情：

- 手写几个小 tensor。
- 矩阵大小只有 `2x2`、`2x3`。
- 调一次 `dpo_loss`。
- 调一次 `backward`。
- 检查梯度是不是非零。

它不加载 GPT-2，不读 tokenizer，不读 `model.safetensors`，不构建 12 层 Transformer。

代码在：

- [test_dpo_native_smoke.cpp](/Users/anakinju/DKU/MobileFineTuner/operator/finetune_ops/optim/test_dpo_native_smoke.cpp:8)

所以它快是正常的。它证明的是：

`DPO loss 和 autograd 最小链路能不能工作。`

### app demo 是什么

app demo 跑的是完整 GPT-2 DPO 训练流程：

1. 读取 GPT-2 config。
2. 构建 GPT-2 small 模型。
3. 从 `/data/local/tmp/mft_gpt2/model.safetensors` 读取完整权重。
4. 初始化 LoRA。
5. 加载 tokenizer vocab/merges。
6. 读取 preference JSONL。
7. 建 reference model。
8. 对 chosen/rejected 做 DPO 训练。
9. 输出 metrics 和 train log。

这里真正慢的是前面模型加载和初始化，尤其是 `model.safetensors` 大约 `523MB`。所以看到 `Load pretrained model` 阶段慢，不是数据集没导入，也不是 preference JSONL 太大，而是每次新 native process 都要重新把完整 GPT-2 初始化一遍。

## 五、模型可以预初始化吗

可以，而且这是下一步很合理的优化方向。

但要先说明当前实现方式：

当前 app 每次点 Start，都会启动一个新的 native process：

```text
app -> libmft_train.so process -> load GPT-2 -> train -> exit
```

这种方式简单、可靠、容易停掉，也方便把日志和 run folder 对齐。但缺点是每次都要重新 load GPT-2。

预初始化有几种方案：

### 方案 1：常驻 native worker process

启动 app 后先启动一个 native worker，让它提前 load GPT-2，然后等待命令。

后续用户点 Start 时，只需要告诉 worker：

- 用哪个 dataset
- steps 多少
- seq_len 多少
- rank/alpha/lr 多少

优点：

- 第一次加载后，后续 run 的启动延迟会明显降低。
- 更接近真正的 on-device service。

缺点：

- Stop / crash recovery 更复杂。
- 要设计 IPC，例如 socket、pipe、stdin protocol 或 file command queue。
- Android 后台进程和权限管理要处理好。

### 方案 2：JNI model session

把 trainer 从“可执行程序”改成 JNI library，在 app 进程里创建一个 `ModelSession`：

```text
initModel()
loadWeights()
startDpoRun(config)
stopRun()
releaseModel()
```

优点：

- UI 和 native session 交互更直接。
- 模型可以真正挂在 app 生命周期里。
- 更像正式 Android 产品。

缺点：

- 改动更大。
- 需要把现有 CLI trainer 重构成可复用 API。
- native crash 可能直接影响 app 进程。

### 方案 3：保留 CLI，但做 warm-up / dry-run

app 打开后先跑一个极小 dry-run，只加载模型、打印配置、然后退出。

这个只能 warm page cache，不能保留 C++ 对象，所以优化有限：

- 第二次读 `model.safetensors` 可能更快一些。
- 但 GPT-2 对象还是要重新构建。

所以它不是根本方案。

### 我的建议

短期为了组会演示：

- 保持当前 CLI process 方案。
- 用 BLAS ON 版 APK。
- 先演示 Smoke / 80 steps。
- 报告中明确说 load 阶段是完整 GPT-2 初始化成本。

中期优化：

- 做 `native worker process`。
- 先用 file-based command queue 或 local socket。
- 目标是让 app 打开后先 `Load model once`，后续训练复用同一个 model session。

长期产品化：

- 再考虑 JNI `ModelSession`。

## 六、BLAS 问题解决了什么

之前手机上训练慢，有两个原因混在一起：

1. 完整 GPT-2 每次都要重新加载。
2. Android arm64 trainer 没有链接 BLAS，矩阵乘法走纯 C++。

上周本地 macOS/native SFT 和 DPO 很多日志里是 `BLAS : ON (compiled)`，所以跑得比手机顺。手机 APK 里的 Android arm64 trainer 之前是 `BLAS : OFF (compiled)`。

这周我们做了：

- 下载 OpenBLAS `0.3.33`。
- 用 Android NDK 编译 arm64 静态库。
- 安装到 `third_party/openblas-android-arm64/`。
- 修改 [operator/CMakeLists.txt](/Users/anakinju/DKU/MobileFineTuner/operator/CMakeLists.txt:13)，支持 `OPENBLAS_ROOT`。
- 重新编译 Android arm64 native trainer。
- 打包进 APK。
- 安装到真机验证。

真机上已经验证 APK 内的 trainer 打印：

```text
BLAS          : ON (compiled)
```

这说明现在手机上运行的是 BLAS ON 版 native trainer。

## 七、Android UI 这周具体改善了什么

这周之前的问题是：即使 native process 开始跑，app 也不够用户友好。

主要问题包括：

- Home / Train / Versus 没有清楚展示 pre-first-step 状态。
- 用户只看到 running，但不知道卡在 load model、load dataset 还是训练。
- `Smoke` 不能可靠停止。
- 日志解析不到 `[1/6]` 这类阶段信息，显示 `no training steps parsed`。
- native trainer 原来通过 `/data/local/tmp/mft_train` 执行，在真机上遇到执行权限限制。

现在已经改善：

- trainer 打包到 APK 的 native library 目录：
  - `android-visualizer/app/src/main/jniLibs/arm64-v8a/libmft_train.so`
- APK packaging 开启 legacy native lib extraction，让它可以作为 executable 运行。
- app 使用 `applicationInfo.nativeLibraryDir` 找 trainer。
- app run data/logs 放到 app 私有目录：
  - `files/mft_demo_data/...`
  - `files/mft_runs/...`
- UI 能显示 pre-first-step 阶段。
- parser 能识别：
  - `[1/6]`
  - `[Stopped]`
  - `Training complete!`
  - `total_steps : ...`
- Stop 会记录 native pid 并发送 TERM/KILL。
- Native Demo Runner 提供：
  - `Smoke`
  - `Phone 80`
  - `Long 220`
  - editable steps / seq_len / rank / alpha / lr
  - `dpo_pair_forward`
  - `cache_ref_logps`

## 八、现在默认参数是什么

当前 `Smoke` 默认参数：

- `steps=8`
- `seq_len=128`
- `rank=1`
- `alpha=2`
- `lr=0.001`
- `eval_interval=4`
- `eval_batches=1`
- `cache_ref_logps=false`
- `dpo_pair_forward=false`

`Phone 80` preset:

- `steps=80`
- `seq_len=128`
- `rank=1`
- `alpha=2`
- `lr=0.001`
- `eval_interval=20`
- `cache_ref_logps=false`

`Long 220` preset:

- `steps=220`
- `seq_len=128`
- `rank=1`
- `alpha=2`
- `lr=0.001`
- `eval_interval=55`
- `cache_ref_logps=false`

为什么现在默认 `dpo_pair_forward=false`：

- DPO 每步本来就要处理 chosen/rejected/reference。
- paired forward 会把 chosen/rejected 合起来算，但一次调用的内存压力更大。
- 真机上先用 separated chosen/rejected forward 更稳。

## 九、本地 HFRL 实验结果怎么讲

除了 app demo，本地还完成了几组 HFRL-style DPO run。

### Health A/B 220-step baseline

路径：

- [runs/hfrl_health_ab_dpo_s220](/Users/anakinju/DKU/MobileFineTuner/runs/hfrl_health_ab_dpo_s220:1)

结果：

- `steps_logged=220`
- `final_eval_loss=0.00234474`
- `final_eval_reward_accuracy=1`
- `final_eval_reward_margin=8.96158`
- `train_loop_seconds=801.1588`
- `ref_cache_seconds=1036.218`

这里说明 DPO 在这个 deterministic health A/B 数据上能非常明确地区分 chosen/rejected。

但也发现一个系统问题：

`reference cache` 的预处理时间比训练本身还长。

所以后续默认改成 `cache_ref_logps=false`，除非同一套静态数据要重复训练很多次，cache 才值得。

### Health A/B no-cache rerun

路径：

- [runs/hfrl_health_ab_dpo_s220_nocache](/Users/anakinju/DKU/MobileFineTuner/runs/hfrl_health_ab_dpo_s220_nocache:1)

结果：

- `steps_logged=220`
- `final_eval_reward_accuracy=1`
- `final_eval_reward_margin=8.96158`
- `train_loop_seconds=1349.5051`
- `ref_cache_seconds=0.0`

这条 run 说明不做 reference cache 也能完成完整 220-step 训练，只是训练 loop 内部时间更长。

### Sparse-feedback 220-step run

路径：

- [runs/hfrl_sparse_feedback_dpo_s220](/Users/anakinju/DKU/MobileFineTuner/runs/hfrl_sparse_feedback_dpo_s220:1)

结果：

- `steps_logged=220`
- `final_eval_reward_accuracy=0.5`
- `final_eval_reward_margin=1.84861`
- `dpo_length_norm_accuracy=1.0`
- `dpo_length_norm_margin=5.1937`

这条 run 说明稀疏用户点击反馈可以转成 DPO 训练数据，但比 deterministic A/B 数据更难，reward accuracy 不如完整偏好对稳定。

## 十、组会 demo 建议怎么演示

如果用 Android Studio / AVD 演示，我建议流程是：

1. 打开 Android Studio。
2. 启动 AVD 或连接真机。
3. 安装当前 APK。
4. 推 GPT-2 权重到 `/data/local/tmp/mft_gpt2`。
5. 启动 app。
6. 进入 `Experiments` / `Native Demo Runner`。
7. 先选 `Smoke`。
8. 点 Start。
9. 展示 event timeline 从 `[1/6] Load pretrained model` 到 `[5/6] Start training`。
10. 等第一条 `[Train]` 出来后，展示 Home / Train / Logs。
11. 如果时间允许，再切 `Phone 80`。

完整命令在：

- [ANDROID_DEMO_RUNBOOK_2026-06-10.md](/Users/anakinju/DKU/MobileFineTuner/briefing/2026-06-10/ANDROID_DEMO_RUNBOOK_2026-06-10.md:1)

## 十一、这周可以怎么总结

可以这样汇报：

`This week, we moved from native DPO command-line validation to an Android app demo. The app can now launch a GPT-2 LoRA DPO run directly from the UI, write fitness/health preference data, stream logs and metrics into the dashboard, stop native training, and visualize progress. We also fixed the Android arm64 performance path by building and linking OpenBLAS; the installed phone APK now prints BLAS ON. The remaining startup delay is mainly full GPT-2 model initialization, so the next system optimization is a persistent native worker or JNI model session that preloads the model once and reuses it across runs.`

中文可以说：

`这周的重点不是再证明 DPO loss 能不能算，而是把它变成一个手机上能点、能跑、能看、能停的 demo。现在 app 里已经能启动 native GPT-2 DPO，数据是 fitness/health preference pairs，日志和指标能进入 UI，并且 Android arm64 的 BLAS 已经打开。当前最主要的剩余问题是每次 run 都要重新 load GPT-2，后续可以通过常驻 native worker 或 JNI session 做模型预初始化。`

## 十二、不能夸大的地方

汇报时建议避免这几种说法：

- 不要说 Android app 上已经完成大规模 220-step 稳定 benchmark。
- 不要说内置 5+2 fitness demo 能证明模型真实健康建议能力。
- 不要说 `test_dpo_native_smoke` 代表完整 GPT-2 训练速度。
- 不要说 BLAS 解决了所有慢的问题；它解决的是训练矩阵计算慢，不能消除每次完整模型加载。

更稳妥的说法是：

- Android end-to-end demo path 已经打通。
- BLAS ON 已在真机 APK 内验证。
- 当前 app demo 是小型 health/fitness preference vertical slice。
- 下一步要做 model pre-initialization 和更大的 external fitness preference dataset sampling。
