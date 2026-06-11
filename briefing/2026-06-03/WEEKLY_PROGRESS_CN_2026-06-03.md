# 本周进度汇报稿

日期：`2026-06-03`

## 一、这周我做了什么

这周我的工作主线有两条。

第一条是 `PyTorch` 路径下的 `DPO vs SFT` 对比实验。这条线的重点是拿到一组比较可信、比较完整、适合汇报的结果，所以我重点看的是高步数设置，尤其是：
- `220 steps`
- `400 steps`
- `600 steps`

第二条是 native `C++` 路径下的 `DPO` 接入。这条线这周的目标不是追求最终精度，而是先把训练链路打通，也就是：
- preference 数据能读进来
- `DPO loss` 能算
- policy/reference 前向打分能跑
- backward 能跑
- 参数能更新
- 命令行训练入口能直接用

所以这周组会上，我建议把重点分成两层：
- 主 highlight：`PyTorch DPO vs SFT` 的效果结果
- 次 highlight：native `C++ DPO` 已经功能打通

## 二、这些指标应该怎么解读

这是汇报时最容易被问到的部分，我先把定义讲清楚。

### 1. accuracy 不是 `1 - loss`

这里的 `accuracy` 不是分类里的 `1 - loss`，也不是 token accuracy。

在我们的 held-out preference evaluation 里，`pair_accuracy` 的定义是：

对于每一个 `(prompt, chosen, rejected)` 三元组，
- 先计算模型对 `chosen` 的 response log-prob
- 再计算模型对 `rejected` 的 response log-prob
- 如果 `chosen` 的得分更高，就算这条 pair 预测正确

最后：

`pair_accuracy = 正确 pair 数 / 总 pair 数`

所以它表达的是：

`模型在多少比例的 preference pair 上，确实更偏向 chosen 而不是 rejected。`

这和 loss 是两件事，不能直接写成 `1 - loss`。

### 2. held-out margin 是什么

`mean_pair_margin` 的定义是：

`margin = logp(chosen) - logp(rejected)`

然后对所有 held-out pair 取平均。

它表达的是：
- 如果 margin 为正，说明模型平均上更偏向 chosen
- 如果 margin 越大，说明这种偏好越强
- 如果 accuracy 接近，但 DPO 的 margin 更高，通常说明 DPO 的 preference separation 更明显

所以可以这样理解：
- `accuracy` 看“赢了多少题”
- `margin` 看“赢得有多明显”

### 3. DPO trainer 里的 reward_acc / reward_margin 是什么

native `C++` 和 PyTorch trainer 内部还有一组 DPO 指标：
- `reward_accuracy`
- `reward_margin`

它们和 held-out eval 的直觉类似，但它们是在 DPO trainer 自己的 reference-relative 指标空间里算的，不是简单的外部 pairwise exact match。

你汇报时可以这样说：

`trainer reward_acc / reward_margin 用来判断 DPO 目标本身是否在按预期优化；held-out pair_accuracy / mean_pair_margin 用来做更直接的模型对比。`

### 4. valid_ppl 是什么

`SFT` 那边没有 preference pair loss，所以 trainer 侧一般报的是：
- `valid_ppl`

也就是 perplexity。

它反映的是 chosen-only 语言建模拟合情况，但它和 pairwise preference discrimination 不是一个维度，所以：

`SFT 的 valid_ppl 不能直接和 DPO 的 loss 做一一对应比较。`

## 三、我们用的数据集长什么样

这次最重要的数据集是：
- `review-stage/data/real_mmlu_pref_train.jsonl`
- `review-stage/data/real_mmlu_pref_eval.jsonl`

它的每一行基本是这种结构：

```json
{
  "prompt": "Question: ... A. ... B. ... C. ... D. ... Answer:",
  "chosen": " A",
  "rejected": " C",
  "subject": "high_school_physics_val",
  "source_split": "val",
  "generator_mode": "mcq_letter_real_mmlu"
}
```

也就是说，它不是 open-ended 长回答，而是：
- 一个真实 `MMLU` 风格的多选题 prompt
- 一个正确选项 letter 作为 `chosen`
- 一个错误选项 letter 作为 `rejected`

### 数据样例 1

题目来自 `human_sexuality_val`：

- prompt: `The feedback loop in women which regulates the production of sex hormones includes all but which of the following?`
- options: `A/B/C/D`
- chosen: `A`
- rejected: `C`

### 数据样例 2

题目来自 `high_school_physics_val`：

- prompt: `A swimmer is able to propel himself forward through the water by moving his arms. Which force explains the forward acceleration?`
- chosen: `A`
- rejected: `B`

### 数据样例 3

题目来自 `astronomy_val`：

- prompt: `You cool a blackbody to half its original temperature. How does its spectrum change?`
- chosen: `D`
- rejected: `B`

训练集里也有更长的真实英文阅读题，样例如：
- 长篇 cell phone recycling 阅读理解题
- 幼儿园 scratcher 行为改造阅读题
- 家庭故事型阅读理解题

所以这份数据的特点是：

1. prompt 本身是真实 `MMLU` 风格文本，不是合成短句
2. `chosen/rejected` 很短，通常只是选项字母
3. 真正困难的部分不是答案长，而是 prompt 长

这也正是为什么 `seq_len=128` 在 real MMLU 上不够用。

## 四、为什么主结果先用 PyTorch，而不是直接用 C++

这个问题你组会上很可能会被问。

我建议直接这样回答：

`因为 PyTorch 路径更成熟、更稳定，适合先做算法效果对比；C++ 路径这周的主要目标是功能打通，而不是作为最终效果结论。`

更具体一点，有三个原因。

### 1. PyTorch 路径先天更适合做算法对比

PyTorch 这边：
- DPO utils 已经比较完整
- trainer-level eval 更成熟
- 数据处理、evaluation、held-out comparison 也已经形成稳定流程

所以如果我要回答“DPO 和 SFT 到底比起来怎么样”，用 PyTorch 更可信、更快、更方便分析。

### 2. C++ 这周的重点不是效果天花板，而是训练链路能不能跑通

native `C++` 这周要回答的核心问题其实是：

- loss 接进去了吗
- backward 通了吗
- LoRA 参数真的更新了吗
- CLI 能不能直接跑

所以它更像 systems / integration checkpoint，而不是最终算法 benchmark。

### 3. 当前 C++ 代码还有待 merge 的更新

你同学已经说了现在的代码还有问题，所以当前 native `C++` 版本我们不应该过度精调。

更合理的策略是：
- 先把“已经打通”的阶段性结果汇报出去
- 等他 merge 最新代码后，再基于新版本做下一轮稳定验证

## 五、PyTorch 这周最重要的结果怎么讲

主结果文件在这里：
- [review-stage/reports/RESULTS_SUMMARY_2026-05-31.md](/Users/anakinju/DKU/MobileFineTuner/review-stage/reports/RESULTS_SUMMARY_2026-05-31.md:1)

### 1. synthetic MMLU-style proxy，`220 steps`

- `DPO held-out acc = 0.8583`
- `SFT held-out acc = 0.8417`
- `DPO held-out margin = 23.8520`
- `SFT held-out margin = 2.5100`

这组结果最清楚，说明：

`在标准 preference-style proxy 上，DPO 明显优于 chosen-only SFT。`

### 2. real MMLU，`seq_len=384`，`220 steps`

- `DPO held-out acc = 0.5726`
- `SFT held-out acc = 0.5299`
- `DPO held-out margin = 0.4588`
- `SFT held-out margin = -0.0253`

这是我建议组会上重点展示的 real-data 主结果。

因为它说明两件事：

1. 把 `seq_len` 提到 `384` 以后，真实 MMLU 才真正可用
2. 在这个第一个可用设置下，`DPO` 是优于 `SFT` 的

### 3. 更高步数，`400 / 600 steps`

`400 steps`：
- DPO acc `0.5556`
- SFT acc `0.5556`
- DPO margin `0.5583`
- SFT margin `0.0198`

`600 steps`：
- DPO acc `0.5641`
- SFT acc `0.5812`
- DPO margin `0.8014`
- SFT margin `0.0583`

这一段结果的解读要谨慎。

正确说法是：

`更长训练让 DPO 的 margin 提升了，说明 preference separation 变强；但在 accuracy 上没有形成稳定、决定性的优势。`

所以这周最稳妥的口径是：

`220-step real-MMLU 是当前最适合汇报的主结果，高步数结果主要说明趋势和稳定性，而不是最终 conclusively better。`

## 六、PyTorch 里 DPO 和 SFT 的比较，是不是换了全新的参数设置

不是“完全另起一套”，而是：

`尽量保持可比设置，只替换训练目标和对应输入形式。`

具体来说：
- 同样是 `GPT-2 LoRA`
- 同样是 matched train/eval split
- SFT 的数据是从 preference 数据里转成 chosen-only masked JSONL
- DPO 的数据保留 `(prompt, chosen, rejected)` 形式

所以变化的核心不是“模型大改”或者“参数完全不同”，而是：

1. 输入形式不同
   - SFT 只看 chosen
   - DPO 同时看 chosen/rejected
2. loss 不同
   - SFT 是 masked LM / chosen-only supervision
   - DPO 是 pairwise preference objective

这正是我们想比较的东西。

## 七、我们是怎么实现 DPO 的，为什么要这么做

这一段建议你讲清楚，因为这能证明我们做的是“真的 DPO”，不是“换个名字的 SFT”。

### 1. DPO 不是单答案拟合，而是相对偏好优化

`SFT` 的目标是让 chosen answer 的 likelihood 更高。

`DPO` 的目标不是只提高 chosen，而是：

`让模型对 chosen 相对于 rejected 的偏好更强，并且这个偏好要比 reference model 更强。`

### 2. 为什么每条样本要做四次 forward

对于每个 pair，我们都需要：
- `policy(chosen)`
- `policy(rejected)`
- `reference(chosen)`
- `reference(rejected)`

原因是 DPO 比较的是：
- policy 对 chosen/rejected 的偏好差
- 相对于 reference 原来的偏好差，有没有更好

所以四路打分是必要的，不是重复计算。

### 3. loss 是怎么计算的

直觉上是这样：

1. 分别算 chosen 和 rejected 的 response token log-prob
2. 算 policy 偏好差：
   - `logp_policy(chosen) - logp_policy(rejected)`
3. 算 reference 偏好差：
   - `logp_ref(chosen) - logp_ref(rejected)`
4. 用 policy 差减去 reference 差
5. 再乘 `beta`，送进 logistic DPO objective

所以它优化的是 preference gap，而不是单个答案分数。

### 4. backward 是怎么做的

在 native `C++` 实现里：
- reference model 只做 frozen scoring
- policy model 的 logits 会进入 `dpo_loss`
- `dpo_loss` 算出来后直接 `backward`
- 后面接 gradient clipping、optimizer step、zero grad

日志里持续看到：
- `nonzero_grad_tensors = 48`

这说明 LoRA 参数确实拿到了有效梯度，不是只做了前向。

### 5. 为什么要这么做

因为我们的目标不是“背答案”，而是“学偏好”。

这套实现更适合未来的真实用户偏好场景，因为很多时候我们拿到的是：
- 这个回答更好
- 那个回答更差

而不是一个标准唯一 label。

## 八、C++ 这周到什么程度了

native `C++` 当前总结文件在这里：
- [briefing/NATIVE_DPO_VS_SFT_REAL_MMLU_2026-06-03.md](/Users/anakinju/DKU/MobileFineTuner/briefing/NATIVE_DPO_VS_SFT_REAL_MMLU_2026-06-03.md:1)

最重要的结论不是“C++ 精度已经很高”，而是：

`native C++ DPO 训练链路已经打通。`

具体已经做到：
- `--train_mode dpo` CLI 已接入
- preference 数据已接入
- policy/reference scoring 已接入
- native `DPO loss` 已接入
- backward 已接入
- LoRA 参数更新已验证
- held-out eval bridge 已接通

换句话说，如果学长的要求是：

`先把 DPO loss、前向打分、反传和 CLI 训练链路打通`

那这个 checkpoint 已经达到要求。

## 九、C++ 这次跑了什么数据，结果怎么解读

native `C++` 这次跑的是同一套 real-MMLU preference 数据：
- train: `480` 原始 pair，`seq_len=384` 下保留 `222`
- eval: `120` 原始 pair，保留 `117`

运行目录：
- [runs/native_gpt2_dpo_vs_sft_real_mmlu_s384](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384:1)

### held-out 结果

DPO：
- [dpo_eval.json](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384/dpo_eval.json:1)
- `pair_accuracy = 0.5726`
- `mean_pair_margin = 0.5648`

SFT：
- [sft_eval.json](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384/sft_eval.json:1)
- `pair_accuracy = 0.5556`
- `mean_pair_margin = 0.0689`

### trainer 结果

DPO：
- [dpo_eval_metrics.jsonl](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384/dpo_eval_metrics.jsonl:1)
- `loss = 0.6862`
- `reward_acc = 0.6200`
- `reward_margin = 0.0149`

SFT：
- [sft_eval_metrics.jsonl](/Users/anakinju/DKU/MobileFineTuner/runs/native_gpt2_dpo_vs_sft_real_mmlu_s384/sft_eval_metrics.jsonl:1)
- `valid_ppl = 4.65`

### 时间

在同一个 `60-step` real-MMLU budget 下：
- DPO: `1135.66s`
- SFT: `279.97s`

这个时间差是正常的，因为 DPO 每个 pair 本来就要做四次 forward。

## 十、在 C++ 上发现了什么

### 1. DPO 的核心训练逻辑已经能跑

这周最重要的发现是：

`native C++ 不再只是 toy smoke，而是真的能在真实 preference 数据上训练。`

### 2. C++ 和 SFT 对比时，必须保证梯度路径一致

为了做可比实验，我们在 native `main.cpp` 里把 `SFT` 和 `DPO` 都切到了标准 attention 路径，因为 memory-efficient attention 的 backward 还不完整。

这一步很重要，因为否则比较会不公平，甚至会出现“一个能反传，一个不能反传”的问题。

### 3. C++ 结果目前可以作为功能验证，但不建议继续在旧代码上深调

因为你同学已经说当前代码还有问题，所以现在最合理的状态判断是：

- 功能接入完成
- 当前结果可汇报
- 暂不继续在当前 native 代码上做大量调参
- 等 merge 最新代码后，再继续稳定版验证

## 十一、这周汇报时最稳妥的结论

### 可以明确说的

- `PyTorch DPO vs SFT` 已经跑通，并完成了多组系统对比。
- `220-step real-MMLU` 是当前最适合展示的主结果。
- `DPO` 在 synthetic preference-style proxy 上明显优于 `SFT`。
- native `C++ DPO` 已经完成了 loss、forward scoring、backward 和 CLI 训练链路打通。

### 不建议说太满的

- 不建议说 `DPO` 已经稳定显著优于 `SFT`
- 不建议说当前 native `C++` 就是最终稳定版本
- 不建议说现在还要继续在旧 C++ 代码上做大量精调

## 十二、后续评估框架

后面我建议把评估拆成三层，不再只看现在这一组 `held-out acc / margin`。

### 1. 训练是否打通

- `DPO loss`
- `reward_acc / reward_margin`
- `held-out pair_accuracy / mean_pair_margin`

这一层回答的是：

`模型有没有学会 preference discrimination。`

### 2. 真实生成效果

- 对同一批 prompt 真实生成
- 比较 `DPO vs SFT`
- 报告 win rate 或 judged preference rate

这一层回答的是：

`模型最后生成出来的结果，到底有没有更符合用户偏好。`

### 3. 真实业务场景一致性

- 按用户分组
- 按偏好强弱分组
- 尽量加入 `weak preference / tie` 情况

这一层回答的是：

`在用户只是“略偏好 A”而不是“讨厌 B”的真实场景下，DPO 还能不能稳定成立。`

所以我的建议是：

`保留当前这套离线指标作为基础 benchmark，但后续一定要补真实 generation eval，不能只靠 acc 和 margin 下结论。`

## 十三、一句话总结

`这周我们在 PyTorch 路径上完成了 DPO 和 SFT 的主对比，其中 220-step real-MMLU 是当前最值得汇报的结果；同时我们已经把 native C++ DPO 的 loss、前向打分、反向传播和 CLI 训练链路完整打通，当前可以作为功能完成的 checkpoint，后续等待最新代码 merge 后再继续做稳定版本验证。`
