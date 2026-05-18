# DeepFactCite 可复现性问题日志

日期：2026-05-18

本文记录评估 Qwen3-8B Soft SFT、合并 checkpoint 和继续训练时发现的问题。目标是避免混用不兼容的评测路径，并保证每个报告数字都能复现。

## 成功标准

最终项目交付件必须包括：

1. 一张符合项目目标的结果表：保留 Search-R1 风格的答案/搜索行为，同时提高 DeepFactCite 引用真实性和声明支持。
2. 一份失败台账：充分解释每次失败或未胜出的 SFT 尝试，达到外部审核也能看懂的程度。

仅完成训练是不够的。只有当工件路径、服务路径、数据集、指标和失败分析都能从本日志或链接报告中恢复时，一次运行才算有用证据。

## 当前基线

固定评测集：

- ShortQA 防护评测：`data/shortqa_防护评测/rl/test.parquet`，32 行，有 gold answer。
- DeepFactCite 严格引用评测：`data/deepfactcite_strict/sft/test.parquet`，47 行，无 gold answer；`answer_subem` 无意义。

当前报告：

| 模型/服务路径 | Eval | Answer | 总分 | URL 有效性 | 引用精确度 | 声明支持度 | 不支持 | 搜索轮数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B、vLLM、bf16 | ShortQA32 | 0.219 | 0.189 | 0.031 | 0.038 | 0.031 | 0.906 | 2.625 |
| `Base + soft LoRA`、vLLM、bf16 dynamic LoRA | ShortQA32 | 0.438 | 0.391 | 0.812 | 0.292 | 0.292 | 0.557 | 1.156 |
| `Base + mixclean200 LoRA`、vLLM、bf16 dynamic LoRA | ShortQA32 | 0.500 | 0.427 | 0.969 | 0.333 | 0.333 | 0.495 | 1.125 |
| `soft` bf16 合并完整模型 | ShortQA32 | 0.438 | 0.444 | 0.844 | 0.375 | 0.375 | 0.479 | 1.125 |
| `soft` fp32 合并完整模型，vLLM fp32 | ShortQA32 | 0.406 | 0.361 | 0.688 | 0.240 | 0.237 | 0.615 | 1.000 |
| `soft` bf16 合并 + mix50 LoRA | ShortQA32 | 0.375 | 0.324 | 0.625 | 0.190 | 0.190 | 0.750 | 1.156 |
| Base Qwen3-8B、vLLM、bf16 | Strict47 | N/A | 0.120 | 0.064 | 0.062 | 0.021 | 0.574 | 1.745 |
| `Base + soft LoRA`、vLLM、bf16 dynamic LoRA | Strict47 | N/A | 0.137 | 0.319 | 0.070 | 0.064 | 0.851 | 1.000 |
| `Base + mixclean200 LoRA`、vLLM、bf16 dynamic LoRA | Strict47 | N/A | 0.142 | 0.418 | 0.074 | 0.072 | 0.879 | 1.234 |
| `soft` bf16 合并完整模型 | Strict47 | N/A | 0.138 | 0.234 | 0.070 | 0.055 | 0.762 | 0.979 |
| `soft` bf16 合并 + mix50 LoRA | Strict47 | N/A | 0.150 | 0.312 | 0.095 | 0.087 | 0.778 | 0.894 |

到目前为止的决定：

- 当前最佳 SFT 候选是 `Base + mixclean200 LoRA`，在 vLLM bf16 中以 dynamic LoRA adapter 方式服务。
- `mixclean200` 在 ShortQA32 的答案、URL 有效性、引用精确度、声明支持、不支持率和搜索轮数上优于 `soft100`。
- `mixclean200` 在 Strict47 的 URL 有效性、引用精确度和声明支持上也优于 `soft100`，但 Strict47 的不支持率和虚假 URL 率仍然很高。因此它只是候选 SFT 初始化，不是最终可信引用结论。
- `mixclean200` 通过了 Search-R1 核心 BM25 防护评测：200 行 NQ/HotpotQA，subEM 0.490，高于 `soft100` 的 0.450 和 Base 的 0.290；搜索成功率 1.000，没有搜索轮数峰值问题。
- `mix50`不是赢家，不应继续。
- 旧的 bf16 合并 parent 不得用作继续训练 parent。

## SFT 故障台账

### F-SFT-001：Strict SFT 没有自动击败 Soft SFT

假说：

```text
按 URL 有效性和声明支持过滤 SFT 轨迹，应该改善引用行为，并成为默认 SFT 胜出方案。
```

工件：

```text
Soft adapter：
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100

Strict adapter：
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-strict-100/global_step_100
```

观察结果：

```text
ShortQA32:
  Soft dynamic LoRA:   answer 0.469, URL validity 0.812, claim support 0.240
  Strict dynamic LoRA: answer 0.469, URL validity 0.781, claim support 0.234

Strict47:
  Soft dynamic LoRA:   total 0.156, URL validity 0.346, claim support 0.103
  Strict dynamic LoRA: total 0.137, URL validity 0.282, claim support 0.063
```

分析：

严格过滤改善了预期的数据清洁度约束，但也减少了训练多样性，并可能缩小了答案/搜索行为覆盖范围。
在这个阶段，严格的数据是一种有用的消融，而不是默认的赢家。

结论：

```text
除非 strict SFT 在相同服务路径和相同评测下胜出，否则不要提升为默认方案。
继续把 Soft SFT 100 dynamic LoRA 作为当前 SFT 基线。
```

防护措施：

```text
未来任何“更高质量”的 SFT 数据集，在启动更多训练之前，
都必须同时在 answer/search 防护评测数据和 DeepFactCite 引用数据上与 Soft SFT 100 对比。
```

### F-SFT-002：bf16 LoRA 合并不等同于 dynamic LoRA

假说：

```text
Base Qwen3-8B + Soft LoRA dynamic serving 应该等价于合并后的 Soft 完整模型，
因此合并模型可以作为继续训练 parent。
```

工件：

```text
Dynamic LoRA：
Base model + outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100

旧 bf16 merged full model：
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged
```

观察结果：

```text
HF logits:
  PEFT forward vs in-memory bf16 merge:
    max abs logit diff = 3.328 / 8.547 on two prompts
  in-memory bf16 merge vs saved bf16 merge:
    max abs logit diff = 0.0
  PEFT forward vs in-memory fp32 merge:
    max abs logit diff < 0.0002
```

分析：

保存的 bf16 合并模型内部一致，但不等价于 PEFT dynamic LoRA forward。
失败原因是在 `merge_and_unload()` 期间，把 fp32 LoRA 增量加到了 bf16 base weight 上。

结论：

```text
删除或弃用旧的 bf16 merged parent。
修补 merge_lora_adapter.py，使 float32 成为默认合并 dtype。
信任 merged parent 之前，先做 HF logits 等价性检查。
```

防护措施：

```text
在完成并记录 PEFT-vs-merged HF logits 检查前，merged checkpoint 不能视为有效的继续训练 parent。
```

### F-SFT-003：从无效 parent 继续训练的 MIX50

假说：

```text
从 merged Soft parent 出发，在 mix 数据上继续 SFT 50 步，以改善或稳定引用行为。
```

工件：

```text
原始 mix50 adapter：
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged-mix-50/global_step_50

Parent：
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged
```

观察结果：

```text
训练完成：
  train steps = 50
  val/loss = 1.016
  exit code = 0

ShortQA32:
  answer 0.375, total 0.324, URL validity 0.625, claim support 0.190

Strict47:
  total 0.150, URL validity 0.312, claim support 0.087
```

分析：

训练本身没有崩溃，但它的 parent checkpoint 是 F-SFT-002 中那个有损的 bf16 合并模型。
因此，这个结果不能干净回答 mix-data 继续训练是否有帮助。

结论：

```text
不要从 mix50 继续训练。
不要把 mix50 报告为候选胜出模型。
仅把它保留为失败/消融记录。
```

防护措施：

```text
继续 SFT 之前，先验证 parent model，并记录精确血统：
base -> adapter 或 fp32 merged parent -> continuation adapter。
```

### S-SFT-001：mix-clean 200 成为当前 SFT 候选，但不是最终答案

假说：

```text
直接从 Base Qwen3-8B 训练一个干净的 mixed SFT adapter 200 步，
而不是从 merged Soft parent 继续训练；目标是在保留 answer/search 防护评测行为的同时改善引用行为。
```

工件：

```text
训练脚本：
scripts/deepfactcite/launch_mix_clean_sft_200.sh

Adapter：
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200/global_step_200

Eval 启动器：
scripts/deepfactcite/wait_then_eval_mixclean_sft.sh
scripts/deepfactcite/launch_vllm_base_soft_mixclean.sh
scripts/deepfactcite/run_mixclean_sft_eval_suite.sh

报告：
reports/base_qwen3_8b_mixclean_rerun_vllm_shortqa_防护评测32.json
reports/base_qwen3_8b_mixclean_rerun_vllm_deepfactcite_strict_sft_test47.json
reports/soft100_qwen3_8b_mixclean_rerun_vllm_shortqa_防护评测32.json
reports/soft100_qwen3_8b_mixclean_rerun_vllm_deepfactcite_strict_sft_test47.json
reports/mixclean200_qwen3_8b_vllm_shortqa_防护评测32.json
reports/mixclean200_qwen3_8b_vllm_deepfactcite_strict_sft_test47.json
```

观察结果：

```text
ShortQA32:
  Base:     answer 0.219, total 0.189, URL 0.031, support 0.031, unsupported 0.906
  Soft100:  answer 0.438, total 0.391, URL 0.812, support 0.292, unsupported 0.557
  Mix200:   answer 0.500, total 0.427, URL 0.969, support 0.333, unsupported 0.495

Strict47:
  Base:     total 0.120, URL 0.064, precision 0.062, support 0.021, unsupported 0.574, fake 0.000
  Soft100:  total 0.137, URL 0.319, precision 0.070, support 0.064, unsupported 0.851, fake 0.064
  Mix200:   total 0.142, URL 0.418, precision 0.074, support 0.072, unsupported 0.879, fake 0.071
```

分析：

该运行是在受控 dynamic-LoRA bf16 服务路径下完成的有意义 SFT 改进。
它提升了 ShortQA 答案/搜索防护评测指标，也改善了两个 eval 集上的 URL 有效性和支持性。
但是 Strict47 的 unsupported 与 fake URL 率仍然很高，因为模型引用更积极了。
这支持原始项目判断：SFT 可以教会引用行为，但可信引用还需要在 reward 阶段加入来源真实性和声明支持约束。

结论：

```text
把 mixclean200 作为下一阶段当前 SFT 初始化候选。
不要仅凭 SFT 宣称已经实现最终可信引用。
下一阶段必须加入对 fake/unsupported citation 的硬惩罚，
并运行 outcome-only vs citation-aware GRPO。
```

Search-R1 答案/搜索防护评测：

```text
数据集：
  data/searchr1_core_防护评测/test.parquet
  200 rows = NQ 100 + HotpotQA 100

检索器：
  wiki-18 BM25 + extracted wiki_dump.jsonl

结果：
  Base:     subEM 0.290, search 0.960, budget fail 0.170
  Soft100:  subEM 0.450, search 0.990, budget fail 0.045
  Mix200:   subEM 0.490, search 1.000, budget fail 0.025
```

这张防护评测由 BM25 控制，不是论文官方 E5 设置的复现。
它适合同一骨干下的本地比较，因为唯一变化的是被服务的模型。

### F-SFT-004：最初误把无 gold answer 的引用集当作答案评测集

假说：

```text
DeepFactCite strict47 可以同时用于 answer/search 指标和引用指标。
```

观察结果：

```text
data/deepfactcite_strict/sft/test.parquet 没有 gold answer target。
这个集合上的 answer_subem 没有意义。
```

分析：

DeepFactCite strict47 是一个引用行为评估集。它可以衡量引用存在性、URL 有效性、引用精确度、声明支持、不支持引用率、fake URL rate、搜索轮数和响应长度。
但由于没有 gold answer，它不能支持可靠的 `answer_subem` 结论。

结论：

```text
使用 ShortQA32 和 Search-R1 NQ/HotpotQA 风格数据做 answer/search 防护评测。
strict47 只用于引用行为评测。
```

防护措施：

```text
每张 eval 表都必须标明 `answer_subem` 是否适用于该数据集。
```

### F-SFT-005：服务路径会改变闭环搜索智能体指标

假说：

```text
如果两个 checkpoint 在 logits 层面很接近，greedy vLLM eval 应该给出相同的聚合指标。
```

观察结果：

```text
Soft dynamic LoRA bf16、soft bf16 merged full model 和 soft fp32 merged full model
在闭环搜索评测中产生了不同聚合指标。
```

分析：

Search-agent eval 会放大小的生成差异：

```text
早期 token 差异 -> 不同搜索查询 -> 不同检索片段
-> 不同答案/引用轨迹 -> 不同指标
```

贪婪解码不会消除 dynamic LoRA、合并权重、dtype、attention 后端或 vLLM 加载路径带来的差异。

结论：

```text
把 serving path 视为模型身份的一部分。
不要把 dynamic-LoRA 结果和 merged-model 结果混在同一张基线表里。
```

防护措施：

```text
每张结果表都必须包含 serving path 标签，例如：
base_bf16, soft_dynamic_lora_bf16, soft_merged_bf16, soft_merged_fp32.
```

## 已确认的问题

### 1. bf16 LoRA 合并在数值上不等价

旧合并脚本用 `torch_dtype=torch.bfloat16` 加载 base model，然后调用 `merge_and_unload()`。

直接 HF logits 检查显示：

| 对比 | 最大绝对 logit 差异 | Top1 |
|---|---:|---|
| `PEFT forward(Base+LoRA)` 与内存中 bf16 合并 | 两个 prompt 上为 3.328/8.547 | 在测试 prompt 中相同，但 logit 差异很大 |
| 内存中 bf16 合并与保存的 bf16 合并模型 | 0.0 | 相同 |
| `PEFT forward(Base+LoRA)` 与内存中 fp32 合并 | < 0.0002 | 相同 |

结论：

- 保存的 bf16 合并在内部一致，但它是有损合并。
- 损失来自于将 fp32 LoRA 增量添加到 bf16 base weight 中。
- 合并脚本现在默认为`--torch-dtype float32`。

### 2. 训练 dtype 和合并 dtype 是不同概念

SFT 训练器使用 `torch_dtype=torch.float32` 加载 `partial_pretrain`，然后使用 FSDP 混合精度：

- 加载参数使用 float32，
- 计算/混合精度由 FSDP 控制，
- LoRA adapter 张量保存为 fp32。

因此，用 fp32 合并训练 parent 并不矛盾。它可以防止在合并步骤中丢失 LoRA 增量，而训练仍然可以使用 bf16 混合精度。

### 3. vLLM dynamic LoRA 不支持 float32 LoRA kernel

在 LoRA 图模式分析期间，尝试用 `--dtype float32` 在 vLLM 中运行 `Base + soft LoRA` 失败：

```text
assert weight.dtype in [torch.float16, torch.bfloat16]
AssertionError
```

所以不能用 vLLM 比较：

- `Base + soft LoRA` 的 float32 dynamic LoRA 模式
- vs `soft` fp32 合并完整模型

最接近精确等价性的检查是 HF logits，而不是 vLLM 生成。

### 4. 即使使用贪婪解码，vLLM 服务路径也会改变输出

搜索智能体评估是一个闭环：

1. 模型发出搜索查询，
2. retriever 返回片段，
3. 模型从检索到的片段中继续，
4. 答案/引用指标取决于之前的所有文本。

小的早期 token 差异可以改变搜索查询，从而改变整个轨迹。贪婪解码不能保证以下路径的 rollouts 相同：

- dynamic LoRA 与合并完整模型，
- bf16 与 fp32，
- FlashAttention 与 Triton attention，
- 不同 vLLM 模型加载路径。

这就是为什么即使直接单步 logits 看起来很接近，聚合 eval 也可能移动几个点。

## 必须记录的变量

对于每次运行，记录以下所有内容：

### 模型标识

- base model 路径。
- adapter 路径（如果有）。
- merged full model 路径（如果有）。
- 模型是 dynamic LoRA 还是 merged。
- 合并 dtype 是 bf16 还是 fp32。
- adapter 的 `base_model_name_or_path`。
- 模型提交/校验和（如果可用）。

### 服务后端

- vLLM 版本。
- `--dtype`.
- `--tensor-parallel-size`.
- `--max-model-len`.
- `--gpu-memory-utilization`.
- 是否使用`--enable-lora`。
- 日志中的 attention 后端：FlashAttention、Triton 等。
- 模型是作为 base model、LoRA module 还是 merged full model 加载。
- 精确的 vLLM 启动日志路径。

### 提示和令牌生成器

- 传给 eval 脚本的 tokenizer 路径。
- 模板来源。
- prompt 构造函数/版本。
- 停止标记。
- `max_new_tokens`、`temperature`、`top_p` （如果使用）。

### 检索和代理循环

- eval parquet 路径。
- 语料库 JSONL 路径。
- 检索器实现/版本。
- `topk`.
- `max_turns`.
- 是否从 JSONL 恢复结果。
- 搜索查询规范化（如果更改）。

### 奖励和指标

- 奖励代码版本。
- 是否包含提示令牌或仅对响应进行评分。
- ground-truth schema。
- `answer_subem` 对该数据集是否有意义。
- 聚合脚本/版本。

### 环境

- Conda env。
- PyTorch、Transformers、PEFT、vLLM、verl 版本。
- CUDA 可见设备。
- GPU 类型。
- `CUDA_VISIBLE_DEVICES`、`TOKENIZERS_PARALLELISM`、`WANDB_MODE`等相关环境变量。

## 不可混合的结果类型

请勿将它们视为相同的模型进行比较：

- `Base + soft LoRA` dynamic vLLM bf16 vs `soft` bf16 merged full model。
- `Base + soft LoRA` dynamic vLLM bf16 vs `soft` fp32 merged full model（vLLM fp32）。
- 旧的仅聚合报告 vs 当前有 JSONL 支撑的报告。
- Strict47 引用评测 vs ShortQA 答案防护评测。
- DeepFactCite strict 测试的 `answer_subem` vs ShortQA `answer_subem`；strict47 没有 gold answer。

## 所需的重现性方案

每个评估必须生成：

1. JSONL rollouts。
2. 聚合 JSON。
3. vLLM 启动日志。
4. 已使用命令行。
5. Git 状态/差异摘要。
6. 数据集/语料库路径。
7. serving path 标签：
   - `base_bf16`
   - `soft_dynamic_lora_bf16`
   - `soft_merged_bf16`
   - `soft_merged_fp32`
   - 等等

每个比较表都必须说明使用了哪个服务路径。

## 当前安全操作规则

1. 对于当前 SFT 基线报告，使用 `Base + soft LoRA` 作为 vLLM bf16 下的 dynamic LoRA。
2. 不要把 bf16 merged parent 用作训练 parent。
3. 如果继续 SFT 需要完整 parent，请用 fp32 合并创建。
4. 除非在相同 serving path 和 eval 脚本下击败同一基线，否则不要宣称改进。
5. 保留 ShortQA32 作为答案/搜索防护评测，Strict47 作为引用行为评测。
6. 如果结果意外变化，先运行单 prompt HF logits 对比，再把 GPU 时间花在完整评测上。
7. 对于该机器上的大型 Hugging Face 工件，先运行 `unvpn`，优先使用国内镜像加 `aria2c`；观察到直接 HF/Xet 传输慢得多。
8. 验证前，把 `wiki-18.jsonl.gz` 当作压缩存档；下载到的工件是 gzip 压缩的 tar payload，而不是裸 JSONL gzip。

## 基础设施备注

## GRPO 故障台账

### F-GRPO-001：检索命中 GRPO 证明管道可用，但没有解决引用支持

假说：

```text
如果 GRPO 样例保证目标 URL 可被检索到，DeepFactCite reward 应该能改善引用质量。
```

观察结果：

```text
2-GPU smoke 完成 16 steps / 32 samples。
reward 0.218, search 0.938, URL validity 0.594, citation precision 0.126,
claim support 0.122, unsupported citation rate 0.755.
```

分析：

```text
检索到相关 URL 不等于引用旁边的局部声明得到了支持。
模型可以引用真实 URL，同时写出比片段所能证明范围更宽的句子。
```

结论：

```text
不要基于 retrieval-hit 数据直接跑长 GRPO。
先构建 claim-level support-filtered 数据。
```

### F-GRPO-002：v1 声明过滤数据上的引用感知没有击败 outcome-only

假说：

```text
在同一份 claim-filtered 数据上，citation-aware reward 应该相对 outcome-only reward
改善 URL validity、citation precision、claim support 和 unsupported citation rate。
```

工件：

```text
reports/deepfactcite_claimfiltered_grpo_ablation_2gpu_20260518.md
reports/dfc_mixclean200_claimfiltered_outcome_only_2gpu_rollout_summary.md
reports/dfc_mixclean200_claimfiltered_citation_aware_2gpu_rollout_summary.md
logs/grpo/rollouts/dfc-mixclean200-claimfiltered-outcome-only-20260518_claimfiltered_2gpu/
logs/grpo/rollouts/dfc-mixclean200-claimfiltered-citation-aware-20260518_claimfiltered_2gpu/
```

观察结果：

| 指标 | Outcome-Only | Citation-Aware |
|---|---:|---:|
| search | 1.0000 | 1.0000 |
| URL 有效性 | 0.7656 | 0.6979 |
| 引用精确度 | 0.4219 | 0.3828 |
| 声明支持 | 0.4219 | 0.3828 |
| 不支持的引用率 | 0.4219 | 0.4844 |
| 虚假 URL 率 | 0.0469 | 0.0521 |
| 平均步长响应剪辑比 | 0.1563 | 0.2813 |

分析：

```text
这次运行不足以支撑扩展规模。可能的失败原因是 data/prompt 不匹配：
部分原始问题很宽，而被选中的受支持声明很窄。
模型回答宽问题时会加入额外事实，然后把引用附在一个检索片段并不能完全支持的局部声明上。
```

结论：

```text
不要切到 4 GPUs。
不要在同一份 v1 数据上跑更久。
构建更严格的 one-claim / one-citation 数据集，加入 query-length 过滤和 one-sentence prompt；
在任何更大消融前，先跑一个小规模修复 smoke。
```

### S-GRPO-001：v2 one-citation 数据是修复路径

目的：

```text
把每一行改成一个受支持声明、一条检索引用和一个 one-sentence prompt，
从而减少不受支持的宽泛扩写。
```

工件：

```text
data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite/train.parquet
data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite/test.parquet
data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite/corpus.jsonl
data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite/summary.json
```

生成结果：

```text
kept rows 32
train/test 28/4
corpus docs 32
avg selected citations 1.0
top-2 URL hit train 28/28, test 4/4
reject reasons: weak_claim_support 43, query_too_broad 19,
too_few_supported_claims 3, claim_too_broad 1
```

下一次检查：

```text
先跑一个短的 v2 citation-aware smoke。
如果它减少 unsupported/no-citation/clip 失败，就跑公平的 v2 outcome-only vs citation-aware 消融。
如果没有减少，先修 reward 或 prompt，再继续花 GPU。
```

### S-GRPO-002：v2 引用感知在同一数据上击败 v2 outcome-only

假说：

```text
如果 one-claim / one-citation 数据能减少宽问题过度回答，
显式 citation/support reward 应该在保持搜索的同时，在引用质量上击败 outcome-only reward。
```

工件：

```text
reports/deepfactcite_v2_onecite_grpo_ablation_2gpu_20260518.md
reports/dfc_mixclean200_claimfiltered_v2_onecite_citation_aware_2gpu_rollout_summary.md
reports/dfc_mixclean200_claimfiltered_v2_onecite_outcome_only_2gpu_rollout_summary.md
```

观察结果：

| 指标 | Outcome-Only | Citation-Aware |
|---|---:|---:|
| search | 0.9688 | 1.0000 |
| URL 有效性 | 0.1562 | 0.7188 |
| 引用精确度 | 0.1125 | 0.3937 |
| 声明支持 | 0.1094 | 0.3906 |
| 不支持的引用率 | 0.8438 | 0.3750 |
| no_citation 失败 | 27 | 9 |
| 响应剪辑比率 | 0.0000 | 0.0000 |

分析：

```text
这是当前阶段第一个干净的正向 GRPO 信号。
关键胜利不是 raw reward，而是在同一数据上 URL validity、support、unsupported rate 和 citation count 的诊断指标改善。
```

结论：

```text
下一轮 citation GRPO 应使用 v2 数据替代 v1。
暂时不要提升到 4 GPUs，因为这次没有保存 checkpoint，且 no_citation 仍为 9/32。
```

### F-GRPO-003：更强引用权重没有修复引用遗漏

假说：

```text
提高 citation/support 权重应该能减少 v2 上的 no_citation 失败。
```

工件：

```text
reports/dfc_mixclean200_v2_onecite_citation_strong_2gpu_rollout_summary.md
logs/grpo/rollouts/dfc-mixclean200-v2-onecite-citation-strong-20260518_2gpu/
```

观察结果：

| 指标 | 默认引用感知 | Citation-Strong |
|---|---:|---:|
| search | 1.0000 | 1.0000 |
| URL 有效性 | 0.7188 | 0.7188 |
| 引用精确度 | 0.3937 | 0.4219 |
| 声明支持 | 0.3906 | 0.4219 |
| 不支持的引用率 | 0.3750 | 0.4375 |
| no_citation失败 | 9 | 9 |
| 响应剪辑比率 | 0.0000 | 0.0000 |

分析：

```text
更高的 citation/support 权重改善了平均 support，但 no_citation 没有变化，unsupported rate 反而变差。
剩余失败更像离散格式问题：模型有时不输出 markdown citation，
或者输出裸 [S_xxx]、[1]、原始 URL，而不是 [label](URL)。
单纯调标量 reward 权重对此太粗。
```

结论：

```text
继续训练前，先修 reward/prompt，明确要求精确的 markdown URL citation。
直接惩罚 no-citation 和 bare-label citation。
修补后重新跑 16-step smoke。
```

### INF-001：大型 HF 工件下载路径很重要

日期：2026-05-18

内容：

- 需要 `PeterJinGo/wiki-18-corpus/wiki-18.jsonl.gz` 来做 Search-R1 BM25 文档到文本查找。
- 官方 BM25 索引只存储 `id`，不存储原始文档内容。
- 最初通过 `aria2c` 直接下载 Hugging Face/Xet 较慢，速度低于 1 MB/s 到约 0.7 MB/s。

处理方式：

- 用户在命令行启用 `unvpn`。
- 通过 `hf-mirror.com` 和 `aria2c` 重试下载。
- 观察到速度提升到约 12 MiB/s。

首选命令模式：

```bash
unvpn
aria2c -d data/wiki-18-corpus -o wiki-18.jsonl.gz -x 8 -s 8 -j 4 -k 1M \
  --continue=true \
  --file-allocation=none \
  --auto-file-renaming=false \
  --allow-overwrite=true \
  --max-tries=0 \
  --retry-wait=5 \
  --connect-timeout=30 \
  --timeout=60 \
  --summary-interval=10 \
  https://hf-mirror.com/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz
```

经验：

- 开始多 GB 下载前，先测试网络路径和镜像。
- 不要假设只用 `aria2c` 就足够；在这个环境中，路由选择影响更大。

### INF-002：wiki-18 语料库文件虽然名为 `.jsonl.gz`，实际是 tar payload

日期：2026-05-18

观察结果：

```text
gzip -t data/wiki-18-corpus/wiki-18.jsonl.gz
  passed

按 UTF-8 JSONL 读取：
  UnicodeDecodeError at byte 0x80

解压后的开头字节：
  tar header with ustar marker
```

实际 payload：

```text
data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl
uncompressed size: 14393573105 bytes
record shape: {"id": "...", "contents": "..."}
```

处理方式：

```bash
tar -xOzf data/wiki-18-corpus/wiki-18.jsonl.gz \
  data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl \
  > data/wiki-18-corpus/wiki_dump.jsonl.tmp
mv data/wiki-18-corpus/wiki_dump.jsonl.tmp data/wiki-18-corpus/wiki_dump.jsonl
```

经验：

- 接到检索器之前，先验证压缩格式和逻辑文件格式。
- Search-R1 BM25 服务器应使用 `data/wiki-18-corpus/wiki_dump.jsonl` 作为 `--corpus-path`，
  而不是下载到的 `wiki-18.jsonl.gz` 存档。

## 开放式问题

### GRPO-001：旧 SGLang 后端的 `custom` 奖励管理器不使用 `custom_reward_function.path`

日期：2026-05-18

观察结果：

```text
scripts/train_grpo_2xa800.sh sets:
  reward_model.reward_manager=custom
  custom_reward_function.path=...

但是 /root/autodl-tmp/SearchShortQA/verl/verl/workers/reward_manager/custom.py
直接调用 agentic_rl_searchqa.rewards.reward_manager.RewardManager，
没有调用已加载的 compute_score function。
```

风险：

- 运行看起来可能像是在使用 DeepFactCite reward，但实际仍在使用旧项目 reward。
- 这会让 outcome-only 与 citation-aware 的 GRPO 比较失效。

处理方式：

- 保持旧的后端代码不变。
- 注册 `deepfactcite_custom`，来源为 `scripts/deepfactcite/verl_deepfactcite_reward.py`。
- 通过以下方式启动 GRPO：

```text
reward_model.reward_manager=deepfactcite_custom
custom_reward_function.path=/root/autodl-tmp/Search-R1-DeepFactCite/scripts/deepfactcite/verl_deepfactcite_reward.py
custom_reward_function.name=compute_score
```

2-GPU smoke 运行验证：

```text
reward_manager: deepfactcite_custom
using customized reward function 'compute_score' from .../verl_deepfactcite_reward.py
RewardManagerWorker 加载了同一路径
rollout_data_step_1.jsonl 包含 DeepFactCite 细节：
  url_validity, citation_precision, claim_support, unsupported_citation_rate
```

经验：

- 不要只相信 Hydra `custom_reward_function.path`；要检查实际选中的奖励管理器实现，并在日志中证明 reward 来源。

### GRPO-002：GRPO smoke 行必须经过检索命中过滤

日期：2026-05-18

在第一个 2-GPU smoke rollout 中观察到：

```text
Query: Bach BWV 171 structure/scoring/features
Rollout 1: No relevant search results were found.
Rollout 2: retrieved irrelevant Magnificat / acoustics snippets.
Rewards: 0.08 to 0.10, citation_count=0, claim_support=0.
```

解释：

- GRPO 机制正在发挥作用：弱/无证据获得低奖励。
- 作为训练数据，盲目 `head(24)` 选择长问题效率很低，因为离线 smoke 语料库可能不包含或检索不到支持证据。

下次运行的处理方式：

- 在花费更长 GPU 时间前，构建 `retrieval-hit` GRPO 子集。
- 查询与预期证据之间，至少需要一个有意义词汇 overlap 的 top-k 检索 URL/text。
- 保留短 QA 防护评测行，但避免证据为空或明显偏题的长行。

经验：

- 对可信引用 RL 而言，数据质量 = prompt 质量 + 检索命中质量。
  如果环境很少提供有效证据，即使 reward 正确，模型也很难学到有用的引用行为。

### GRPO-003：短工具响应会截断 URL 并破坏 URL 有效性

日期：2026-05-18

在第一次 2-GPU smoke 运行中观察到：

```text
max_tool_response_length=256
tool response URL: https://pixe...(truncated)...
```

风险：

- 模型无法复制准确检索到的 URL。
- 即使检索器找到了相关页面，URL 有效性也可能被评为 fake/invalid。

处理方式：

- 使用更少的结果和更长的工具响应进行引用训练：

```text
SEARCHQA_TOPK=2
GRPO_MAX_TOOL_RESPONSE_LENGTH=768
```

经验：

- 在引用 RL 中，工具响应截断不只是上下文长度问题；
  它会破坏 URL 字符串，从而改变标签。

### GRPO-004：SGLang 内存分数过低或过高都会出问题

日期：2026-05-18

观察结果：

```text
GRPO_GPU_MEMORY_UTILIZATION=0.12
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

解释：

- SGLang 内存分数降得太低时，静态内存池反而不足，无法支撑 8B、TP=2 的 rollout server。
- 经验值 `0.15` 在 2 张 A800 GPU 上可以成功初始化。

处理方式：

```text
GRPO_GPU_MEMORY_UTILIZATION=0.15
```

经验：

- 把 SGLang 内存分数视为静态池大小参数，而不只是降低显存压力的旋钮。

### GRPO-005：检索命中提高了 URL 有效性，但声明支持仍不足

日期：2026-05-18

32 个采样轨迹上的检索命中 smoke 结果：

```text
format = 1.000
search = 0.938
url_validity = 0.594
citation_precision = 0.126
claim_support = 0.122
unsupported_citation_rate = 0.755
```

观察到的最佳支持样例：

```text
reward ~= 0.625
url_validity = 1.0
claim_support = 0.75
unsupported_citation_rate = 0.0
```

解释：

- reward 和 SGLang 后端工作正常。
- 当片段是直接的，模型可以生成有效的、受支持的引用。
- 大多数采样声明仍然比检索证据更宽，因此支持指标保持低位。

下一个修复：

- 下一份 GRPO 数据集要按声明级别构建，而不是只按查询级别构建：
  选择那些最终引用声明能被 compact 检索片段直接支持的 SFT traces。
  答案尽量短，确保引用出现在回复截断之前。

### GRPO-006：声明级过滤必须拒绝低信息支持的片段

日期：2026-05-18

观察结果：

```text
第一版 claim-level support filter 产出的行满足引用 URL 有效、overlap support 为 1.0，
但少量被选中声明的信息量太低，不适合 RL；
例如它们更像标题片段，而不是有用的答案声明。
```

风险：

- 词汇支持过滤器可能过度奖励标题/小标题片段。
- GRPO 可能学会引用狭窄但无益的片段，而不是简洁、有答案信息的声明。

处理方式：

```text
scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py
```

现在过滤器：

```text
1-2 条引用
只允许检索返回的 URL
不允许截断或虚假的引用 URL
claim support score >= 1.0
声明长度上限
过滤低信息量引用片段
```

生成的工件：

```text
data/deepfactcite_sglang_grpo_claim_filtered/train.parquet
data/deepfactcite_sglang_grpo_claim_filtered/test.parquet
data/deepfactcite_sglang_grpo_claim_filtered/corpus.jsonl
data/deepfactcite_sglang_grpo_claim_filtered/summary.json
data/deepfactcite_sglang_grpo_claim_filtered/preview.jsonl
data/deepfactcite_sglang_grpo_claim_filtered/reject_samples.jsonl
```

验证结果：

```text
kept rows: 32
train/test: 28/4
corpus docs: 42
train top-2 retriever URL hit: 28/28
test top-2 retriever URL hit: 4/4
reject reasons while collecting: weak_claim_support=54, too_few_supported_claims=5
```

结论：

- 使用此数据集进行第一次 2-GPU outcome-only vs citation-aware 消融。
- 在磁盘清理前保留 `save_freq=0`；`/root/autodl-tmp` 大约只有 14G 可用。
- 在 2-GPU 消融显示真正的引用指标收益，且 answer/search 没有崩溃之前，不要扩展到 4-GPU。

## 开放式问题

1. 能否添加一个无需完整合并、直接从 `Base + existing LoRA` 恢复的 Trainer 路径？
2. 能否在不改变基线 serving path 的情况下评估 continuation adapter？
3. 当精确的 LoRA/merge 等价性比速度更重要时，是否应该用 HF generation 做小规模确定性 sanity eval？
4. vLLM 评测是否只标准化到 dynamic LoRA bf16，而把 fp32 merge 仅保留作训练初始化？

## 立即下一步

不要从 `mix50` 继续。

`mixclean200` 已通过本地 Search-R1 BM25 answer/search 防护评测，
是当前 SFT 基线候选。

下一步：

1. 使用 `mixclean200` 作为可能的初始化 checkpoint。
2. 移植或重用 Qwen3/SGLang GRPO 后端。
3. 使用相同 backbone/数据运行 outcome-only GRPO 和 citation-aware GRPO。
4. 在引用感知奖励中保留虚假/不受支持的引用硬性处罚。
5. 除非复现官方 E5 设置，否则原始 Search-R1 论文数字只作为外部参考。

### F-GRPO-004：markdown 引用解析器没有计数带嵌套括号的标签

日期：2026-05-18

预期标准：

```text
任何形如 [label](URL) 的答案链接，只要 URL 语法有效，就应该计为一个 markdown citation，
即使 label 里包含年份这类普通 bracket 文本。
```

在扣分上限 smoke 期间观察到：

```text
[NCDAS: Substance Abuse and Addiction Statistics [2025]](https://drugabusestatistics.org)
```

被报告为：

```text
citation_count = 0
```

根本原因：

```text
deepfactcite/reward.py 使用了一个扁平正则：
\[([^\[\]]+)\]\(([^()\s]+)\)

该正则会拒绝包含另一个 `[` 或 `]` 的链接 label。
如果在未剥离合法链接的答案上运行 bare-bracket 检查，
内部的 [2025] 还可能被当作裸 bracket 诊断。
```

修复：

```text
deepfactcite/reward.py 现在用一个小型扫描器抽取 markdown 链接：
寻找后接 `(` 的闭合 `]`，因此带 bracket 年份的 label 也会被计数。
它还会在 bare-bracket 和 raw-URL 检查前先剥离合法 markdown 链接。
```

验证结果：

```text
嵌套 label 引用：
citation_count=1, raw_url_count=0, bare_citation_count=0

裸 [S_1] 加原始 URL：
citation_count=0, raw_url_count=1, bare_citation_count=1
```

经验：

```text
Reward parser 也是实验的一部分。
如果解析器太脆弱，GRPO 可能会被 parser artifact 而不是真实模型行为误判。
```

### F-GRPO-005：第一次 markdown-cap 运行在计划的 16 步之前停止

日期：2026-05-18

运行：

```text
dfc-mixclean200-v2-onecite-markdowncap-20260518_2gpu
```

观察结果：

```text
只产生了 rollout_data_step_1.jsonl 到 rollout_data_step_6.jsonl
no traceback / OOM / NCCL error in trainer log
GPU idle after stop
```

部分聚合结果：

```text
samples=12
steps=6
reward=0.2319
search=0.7500
url_validity=0.2500
claim_support=0.2083
unsupported_citation_rate=0.5833
citation_count=0.4167
```

使用固定解析器重新计算诊断后：

```text
url_validity=0.3333
citation_count=0.5000
no_citation 失败数从 7 变为 6
```

结论：

```text
不要把它当成可比较的 16-step training result。
只把它作为 parser 问题的调试证据，并重新跑一次干净的 16-step smoke。
```

### S-GRPO-003：对 v2 one-citation 数据启动 parser-fix markdown-cap 重跑

日期：2026-05-18

运行：

```bash
RUN_TAG=20260518_v2_markdowncap_parserfix2_2gpu \
MODE=citation-aware \
EXPERIMENT_NAME=dfc-mixclean200-v2-onecite-markdowncap-parserfix2-20260518_2gpu \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite \
DRY_RUN=0 \
GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

启动验证：

```text
SGLang/GRPO 已进入 async rollout。
SearchQAVerlTool 已使用 v2 one-citation 离线语料初始化。
rollout_data_step_1.jsonl 已创建。
save_freq=0.
```

验收标准：

```text
no_citation < 9/32
claim_support > 0.3906
unsupported_citation_rate <= 0.3750
search 保持接近 1.0
response clip ratio 保持 0.0
```

待决

```text
如果验收通过，先清理磁盘，再运行一次会保存 checkpoint 的 2-GPU 实验。
如果被拒绝，继续留在 2-GPU，先改进 prompt/data/reward，再考虑任何 4-GPU 扩展。
```
