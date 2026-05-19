# DeepFactCite 实验记录

日期：2026-05-18

## 项目目标

构建一个 Search-R1 风格的搜索智能体：在保留答案/搜索行为的同时，提高引用真实性和声明级支持。

该项目在两个轴上进行评估：

1. answer/search 防护评测：模型不能在 ShortQA 或 NQ/HotpotQA 风格的 Search-R1 BM25 评测上崩掉；
2. 引用质量：被引用的 URL 必须来自检索证据，并且被引用的局部声明必须由对应片段支持。

GRPO 的核心实验问题是：

```text
在相同数据、模型、检索器和 rollout 栈下，
citation-aware reward 相比 outcome-only reward，
是否能提高 URL validity、citation precision、claim support，
并降低 unsupported citation rate？
```

## 当前约束

硬件：

```text
2 x A800 80G currently allocated in this workspace
```

存储：

```text
/root/autodl-tmp 大约有 14G 可用空间
```

操作约束：

```text
磁盘清理完成前，不保存完整 checkpoint。
smoke/ablation run 使用 save_freq=0。
短消融产生有用信号之前，不启动长 GRPO。
```

## 外部定位

除非复现了原始模型/检索器设置，否则本项目不应被表述为“击败 Search-R1”。更可辩护的定位是：

```text
DeepFactCite 在 Search-R1 风格 RL 搜索智能体上增加引用真实性和声明支持奖励，
并在相同 backbone、数据和检索器下，把这些奖励与 outcome-only baseline 对比。
```

参考锚点：

```text
Search-R1: https://arxiv.org/abs/2503.09516
ALCE citation evaluation: https://arxiv.org/abs/2305.14627
Correctness vs attribution faithfulness: https://arxiv.org/abs/2412.18004
OpenAI Deep Research system card: https://openai.com/index/deep-research-system-card/
```

## 如何阅读此记录

这个项目分为三层。对新手来说，阅读每个实验时都可以问三个问题：

```text
1. 模型能否正确回答并搜索？
2. 如果它引用来源，URL 是否真的是检索返回的 URL？
3. 每个被引用的局部声明，是否真的能从被引用片段推出？
```

重要术语：

| Term | 在此项目中的含义 | 为什么重要 |
|---|---|---|
| SFT | 在示例搜索/引用轨迹上做监督微调 | 在 RL 之前先教会模型基本行为模式 |
| GRPO | 一种 RL 训练方式，用 reward 比较多个采样答案 | 让 reward 推动那些单靠 SFT 不一定稳定学到的行为 |
| Rollout | 一条完整智能体轨迹：prompt、搜索调用、片段、最终答案 | 调试时检查的基本单元 |
| Retriever | 返回片段/URL 的离线搜索组件 | 只有引用检索证据，引用才可信 |
| URL validity | 被引用 URL 是否来自检索证据，而不是幻觉链接 | 防止虚假或不相关引用 |
| Citation precision | 有效且局部有用的引用比例 | 惩罚过度引用或低质量引用 |
| Claim support | 被引用片段是否明确支持附近声明 | 衡量归因忠实性，而不只是链接是否存在 |
| Unsupported rate | 被引用片段不支持其声明的比例 | 本项目试图降低的主要失败模式 |

最重要的工程原理是：

```text
不要因为一次 run 跑完了就扩大 GPU 规模。
只有当这次 run 隔离了问题、记录了失败模式，并且在不破坏防护指标的情况下
改善了它本来要改善的指标，才值得扩容。
```

## 阶段1：数据和SFT基线

从 DeepCiteFact 轨迹准备 DeepFactCite 风格 SFT/RL 数据：

```text
data/deepfactcite/sft/train.parquet
data/deepfactcite/sft/test.parquet
data/deepfactcite/rl/train.parquet
data/deepfactcite/rl/test.parquet
```

Strict SFT 过滤只保留带检索 URL 且有非平凡 claim-support 的轨迹：

```text
data/deepfactcite_strict/sft/train.parquet
data/deepfactcite_strict/sft/test.parquet
```

关键经验：

```text
仅靠 strict filtering 并不会自动击败 Soft SFT。
更干净的引用数据可能降低多样性，因此晋升前必须用 answer/search 防护评测检查。
```

初学者说明：

```text
“更干净的数据”不自动等于“更好的训练数据”。
如果过滤掉大量多样化样例，模型看到的回答/搜索方式会更少。
即使剩余样例的引用更漂亮，通用行为也可能变差。
所以正确测试不是“数据集看起来是否干净”，而是
“在同一评测路径下，用它训练出的模型是否击败旧模型”。
```

当前 SFT baseline 是 MixClean200：

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200/global_step_200
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16
```

训练脚本：

```text
scripts/deepfactcite/launch_mix_clean_sft_200.sh
```

主要报告：

```text
reports/deepfactcite_mixclean200_eval_summary.md
```

## 第2阶段：基线评估

MixClean200 与 Base、Soft100 在相同 vLLM dynamic-LoRA serving path 下评测。

ShortQA32：

| 模型 | 答案 subEM | URL 有效性 | 声明支持度 | 不支持 |
|---|---:|---:|---:|---:|
| Base | 0.219 | 0.031 | 0.031 | 0.906 |
| Soft100 | 0.438 | 0.812 | 0.292 | 0.557 |
| MixClean200 | 0.500 | 0.969 | 0.333 | 0.495 |

Search-R1 BM25 200：

| 模型 | subEM | 搜索成功 | 搜索轮数 | 预算失败 |
|---|---:|---:|---:|---:|
| Base | 0.290 | 0.960 | 1.720 | 0.170 |
| Soft100 | 0.450 | 0.990 | 1.195 | 0.045 |
| MixClean200 | 0.490 | 1.000 | 1.160 | 0.025 |

结论：

```text
使用 MixClean200 作为 GRPO 的 SFT 初始化。
不要仅凭 SFT 就宣称最终实现了可信引用。
```

为何重要：

```text
SFT 成功教会模型更频繁地搜索和引用，
但还没有完全教会更严格的行为：“只引用检索片段支持的声明”。
这正是 GRPO reward 应该瞄准的缺口。
```

## 第3阶段：GRPO 后端和奖励管道

发现的问题：

```text
旧 SGLang backend 的 reward_manager=custom 不一定会调用新的 DeepFactCite reward function，
即使已经设置 custom_reward_function.path。
```

修复：

```text
通过下面文件注册 reward_manager=deepfactcite_custom：
scripts/deepfactcite/verl_deepfactcite_reward.py
```

已验证日志：

```text
reward_model.reward_manager = deepfactcite_custom
custom_reward_function.path = scripts/deepfactcite/verl_deepfactcite_reward.py
rollout JSONL 包含 url_validity、citation_precision、claim_support、
unsupported_citation_rate
```

## 第4阶段：2-GPU 检索命中 GRPO smoke

目的：

```text
在投入更长 GPU 时间之前，验证 2-GPU Qwen3-8B SGLang GRPO、
DeepFactCite reward、离线检索和 rollout 日志记录链路。
```

运行：

```text
data/deepfactcite_sglang_grpo_retrieval_hit/train.parquet
logs/grpo_mixclean200_2gpu_hit_smoke_retry.log
logs/grpo/rollouts/mixclean200_2gpu_hit_smoke/
```

设置：

```text
TP=2
rollout.n=2
train_batch_size=1
total_steps=16
topk=2
max_tool_response_length=768
max_response_length=384
save_freq=0
```

32 条采样轨迹上的结果：

| 指标 | 值 |
|---|---:|
| reward | 0.218 |
| search | 0.938 |
| URL 有效性 | 0.594 |
| 引用精确度 | 0.126 |
| 声明支持 | 0.122 |
| 不支持的引用率 | 0.755 |

解释：

```text
工程链路可以跑通。瓶颈不是 SGLang plumbing，而是 evidence/claim 质量。
query 级 retrieval-hit 还不够，因为模型仍然会写出比片段支持范围更宽的声明。
```

失败分析：

```text
这是一次成功的系统 smoke test，但引用质量结果失败。
模型通常会搜索，很多被引用 URL 也确实来自检索结果；
但答案句子经常提出比片段支持范围更宽的声明。
典型模式是：片段只支持“X 发生在 2012 年”，答案却说“X 在 2010 到 2013 年改变了整个组织”。
URL 不是假的，但被引用声明仍然没有被支持。
```

经验： 

```text
query 级 retrieval hit 对引用 RL 来说太弱。
数据必须在 claim 级过滤：短答案、少量引用、只允许检索返回的 URL，
并且被引用声明必须被片段明确支持。
```

结论

```text
在更长训练前，先构建 claim-level support-filtered GRPO 数据。
```

## 简单语言的失败台账

本节用更直白的方式解释先前失败或未获胜的实验，方便新接手项目的人检查。

### 失败1：严格SFT更干净，但效果不佳

我们尝试的方法：

```text
在 URL/support 属性更好的严格 SFT 样例上训练。
```

发生了什么：

```text
Strict SFT 在相同 eval path 下没有击败 Soft SFT，因此没有被提升。
```

它可能失败的原因：

```text
过滤提高了数据干净程度，但降低了多样性。搜索智能体不仅要学习引用格式，
还要学习什么时候搜索、如何组织查询、如何回答不同问题。
过度过滤会让这些行为变窄。
```

技术价值

```text
它证明了数据集质量必须由下游防护评测指标衡量，
不能只看过滤规则有多严格。
```

### 失败2：bf16 LoRA 合并改变了模型

我们尝试的方法：

```text
把 LoRA adapter 合并进 base model，并把 merged full model 当作等价于 dynamic LoRA serving 来使用。
```

发生了什么：

```text
HF logits 显示，PEFT dynamic LoRA 与保存的 bf16 merged model 之间存在很大差异，
即使保存后的模型内部是自洽的。
```

失败的原因：

```text
merge 过程中，LoRA delta 实际上被加进了 bf16 base weight。
这会丢失数值细节。在闭环搜索智能体里，很小的 token 变化就可能导致不同搜索查询，
进而改变检索证据和最终指标。
```

技术价值

```text
Serving path 是模型身份的一部分。除非明确标注，
否则 dynamic LoRA、bf16 merged 和 fp32 merged 的结果不能混在同一张 baseline 表里。
```

### 失败3：mix50 continuation 使用了无效 parent

我们尝试的方法：

```text
从旧 merged Soft parent 继续 SFT 50 步。
```

发生了什么：

```text
这次 run 完成了，但 parent 是上一个失败中的有损 bf16 merged model。
结果更差，也不能干净地回答 mix-data continuation 是否有效。
```

失败的原因：

```text
训练数据不是唯一变量。parent checkpoint 的行为已经发生变化，
所以这个实验无法回答 continuation recipe 本身是否有效。
```

技术价值

```text
每次训练都需要明确 lineage：base model、adapter、merge dtype、serving path 和 eval path。
```

### 失败4：Strict47 答案指标没有意义

我们尝试的方法：

```text
把 DeepFactCite strict47 当作通用答案与引用评测集使用。
```

发生了什么：

```text
这个集合没有适合 answer_subem 的 gold answer 字段。
它的引用指标有用，但 answer_subem 没有意义。
```

为什么重要

```text
如果数据集 schema 不支持某个指标，那么这个指标即使看起来漂亮也没有用。
这就是为什么项目要把 answer/search 防护评测（ShortQA/Search-R1 BM25）
和 citation-behavior 评测（strict47）分开。
```

### 失败5：检索命中GRPO未解决支持问题

我们尝试的方法：

```text
给 GRPO 提供检索器能够找到目标 URL 的样例。
```

发生了什么：

```text
2-GPU 栈跑通了，但 claim support 仍然很低，unsupported citation rate 仍然很高。
```

失败的原因：

```text
找到正确 URL 不等于写出了被支持的声明。
模型可以引用真实 URL，同时写出片段并不能证明的句子。
```

技术价值

```text
这个失败证明了当前 claim-level support-filtered 数据构建、
以及 outcome-only vs citation-aware 消融是必要的。
```

## 可重复使用的实验流程

未来实验中应重复使用的过程是：

```text
1. 固定问题：这次 run 只测试哪一个假设？
2. 固定模型身份：base、adapter、merge dtype、serving path。
3. 固定数据身份：train/test parquet、corpus path、retriever top-k。
4. 运行能验证完整工程路径的最小 smoke。
5. 解析 rollouts，而不是只看 stdout reward。
6. 同时记录成功和失败。
7. 只有当失败分析说明下一次更大规模运行是合理的，才扩大规模。
```


## 第5阶段：声明级支持过滤的GRPO数据

脚本 

```text
scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py
```

命令：

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python \
  scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py \
  --rows 32 \
  --out-dir data/deepfactcite_sglang_grpo_claim_filtered
```

工件：

```text
data/deepfactcite_sglang_grpo_claim_filtered/train.parquet
data/deepfactcite_sglang_grpo_claim_filtered/test.parquet
data/deepfactcite_sglang_grpo_claim_filtered/corpus.jsonl
data/deepfactcite_sglang_grpo_claim_filtered/summary.json
data/deepfactcite_sglang_grpo_claim_filtered/preview.jsonl
data/deepfactcite_sglang_grpo_claim_filtered/reject_samples.jsonl
```

生成结果：

| 项目 | 数值 |
|---|---:|
| 保留行数 | 32 |
| train/test | 28/4 |
| 语料库文档 | 42 |
| 平均所选引用数 | 1.3125 |
| 最小支持分 | 1.0 |
| 平均支持分 | 1.0 |

收集时拒绝原因：

| 原因 | 数量 |
|---|---:|
| weak_claim_support | 54 |
| too_few_supported_claims | 5 |

检索器验证：

```text
OfflineSearchTool, topk=2
train top-2 URL hit: 28/28
test top-2 URL hit: 4/4
```

筛选规则

```text
1-2 条引用
只允许检索返回的 URL
不允许截断或虚假的引用 URL
claim support score >= 1.0
声明长度上限
过滤低信息量引用片段
prompt 要求输出 1-3 个简洁句子
```

## 第6阶段：仅限结果与引用感知GRPO

入口点：

```text
scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

常用设置

```text
actor = outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16
data = data/deepfactcite_sglang_grpo_claim_filtered
rollout 后端 = SGLang
tensor parallel = 2
rollout.n = 2
train_batch_size = 1
total_training_steps = 16
topk = 2
max_tool_response_length = 768
max_response_length = 384
save_freq = 0
reward_manager = deepfactcite_custom
```

仅限结果的奖励：

| 组件 | 权重 |
|---|---:|
| answer | 0.80 |
| citation | 0.00 |
| support | 0.00 |
| format | 0.15 |
| search | 0.10 |
| cost | 0.05 |

引用感知奖励：

| 组件 | 权重 |
|---|---:|
| answer | 0.15 |
| citation | 0.35 |
| support | 0.30 |
| format | 0.10 |
| search | 0.05 |
| cost | 0.05 |

命令：

```bash
MODE=outcome-only DRY_RUN=0 GRPO_TOTAL_STEPS=16 \
  bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh

MODE=citation-aware DRY_RUN=0 GRPO_TOTAL_STEPS=16 \
  bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

已完成的运行：

```text
MODE=outcome-only
RUN_TAG=20260518_claimfiltered_2gpu
log=logs/dfc-mixclean200-claimfiltered-outcome-only-20260518_claimfiltered_2gpu.log
rollouts=logs/grpo/rollouts/dfc-mixclean200-claimfiltered-outcome-only-20260518_claimfiltered_2gpu/

MODE=citation-aware
RUN_TAG=20260518_claimfiltered_2gpu
log=logs/dfc-mixclean200-claimfiltered-citation-aware-20260518_claimfiltered_2gpu.log
rollouts=logs/grpo/rollouts/dfc-mixclean200-claimfiltered-citation-aware-20260518_claimfiltered_2gpu/
```

启动验证：

```text
Hydra config 校验通过。
reward_manager=deepfactcite_custom.
reward function 已从 scripts/deepfactcite/verl_deepfactcite_reward.py 加载。
数据集规模 = 28 train, 4 val。
总训练步数 = 16。
离线检索器使用 data/deepfactcite_sglang_grpo_claim_filtered/corpus.jsonl。
SGLang 后端已在 CUDA_VISIBLE_DEVICES=0,1 上初始化。
checkpoint 保存已关闭。
```

结果：

| 指标 | Outcome-Only | Citation-Aware | 解读 |
|---|---:|---:|---|
| samples | 32 | 32 | 预算相同 |
| search | 1.0000 | 1.0000 | 搜索未收起 |
| format | 0.9812 | 0.9750 | 都很高 |
| answer_subem | 0.0000 | 0.0312 | 引用感知略高，但样本量很小 |
| URL 有效性 | 0.7656 | 0.6979 | 引用感知更差 |
| 引用精确度 | 0.4219 | 0.3828 | 引用感知更差 |
| 声明支持 | 0.4219 | 0.3828 | 引用感知更差 |
| 不支持的引用率 | 0.4219 | 0.4844 | 引用感知更差 |
| 虚假 URL 率 | 0.0469 | 0.0521 | 引用感知更差 |
| 引用计数 | 1.0625 | 1.0000 | 相近 |
| 平均步长响应剪辑比 | 0.1563 | 0.2813 | 引用感知更差 |

请仔细阅读此表：

```text
citation-aware 的 reward 数字本身更高，但不同模式的 reward 权重不同，
所以 reward 数字不能直接横向比较。公平对比应该看 URL validity、
citation precision、claim support、unsupported rate、search、format 和 answer 防护评测。
按这些指标看，v1 citation-aware 没有获胜。
```

故障分析

```text
v1 claim-filtered 数据只保证某个选中的声明被片段支持，
但有些原始用户问题比这个选中声明更宽。
模型经常仍然回答宽问题，额外添加背景信息，
然后把引用放在一个片段并不能完全证明的句子后面。
这会让 URL 真实，但被引用的局部声明仍然不受支持。
```

结论

```text
不要基于这个结果切到 4 GPU。
不要在同一份 v1 数据上跑更长 GRPO。
先构建并测试更严格的 one-claim / one-citation 数据集。
```

后续报告：

```text
reports/deepfactcite_claimfiltered_grpo_ablation_2gpu_20260518.md
```

## 第7阶段：单声明/单引用修复

原因：

```text
v1 的失败并不是“citation-aware reward 没用”，而是“当前数据和 prompt
仍然允许模型过度回答宽问题”。修复方向是让下一版数据更窄，
使被支持的行为更容易学习，也更容易衡量。
```

脚本已更新

```text
scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py
```

新筛选条件/选项：

```text
max_citations = 1
max_claim_tokens = 30
max_query_tokens = 18
max_answer_sentences = 1
strict_one_citation_prompt = true
```

命令：

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python \
  scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py \
  --rows 32 \
  --out-dir data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite \
  --max-citations 1 \
  --max-claim-tokens 30 \
  --max-query-tokens 18 \
  --max-answer-sentences 1 \
  --strict-one-citation-prompt
```

生成结果：

| 项目 | 数值 |
|---|---:|
| 保留行数 | 32 |
| train/test | 28/4 |
| 语料库文档 | 32 |
| 平均所选引用数 | 1.0000 |
| 最小支持分 | 1.0000 |
| 平均支持分 | 1.0000 |

拒绝原因：

| 原因 | 数量 |
|---|---:|
| weak_claim_support | 43 |
| query_too_broad | 19 |
| too_few_supported_claims | 3 |
| claim_too_broad | 1 |

检索器验证：

```text
OfflineSearchTool, topk=2
train top-2 URL hit: 28/28
test top-2 URL hit: 4/4
```

下一次小规模运行：

```bash
RUN_TAG=20260518_v2_onecite_2gpu \
MODE=citation-aware \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite \
DRY_RUN=0 \
GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

此维修运行的成功标准：

```text
这次修复运行不需要证明最终优越性。它应该说明更严格的 one-claim 数据
是否能减少 no-citation、fake URL、response clipping 和 unsupported claim 行为，
从而支撑一次公平的 v2 outcome-only vs citation-aware 消融。
```

已完成 v2 运行：

```text
v2 citation-aware:
logs/grpo/rollouts/dfc-mixclean200-claimfiltered-citation-aware-20260518_v2_onecite_2gpu/
reports/dfc_mixclean200_claimfiltered_v2_onecite_citation_aware_2gpu_rollout_summary.md

v2 outcome-only:
logs/grpo/rollouts/dfc-mixclean200-claimfiltered-outcome-only-20260518_v2_onecite_2gpu/
reports/dfc_mixclean200_claimfiltered_v2_onecite_outcome_only_2gpu_rollout_summary.md

v2 citation-strong:
logs/grpo/rollouts/dfc-mixclean200-v2-onecite-citation-strong-20260518_2gpu/
reports/dfc_mixclean200_v2_onecite_citation_strong_2gpu_rollout_summary.md
```

公平 v2 消融结果：

| 指标 | Outcome-Only | Citation-Aware | 解读 |
|---|---:|---:|---|
| search | 0.9688 | 1.0000 | 引用感知保留搜索 |
| format | 0.9812 | 0.9625 | 都可以使用 |
| URL 有效性 | 0.1562 | 0.7188 | 引用感知能力更强 |
| 引用精确度 | 0.1125 | 0.3937 | 引用感知能力更强 |
| 声明支持 | 0.1094 | 0.3906 | 引用感知能力更强 |
| 不支持的引用率 | 0.8438 | 0.3750 | 引用感知能力更强 |
| 虚假 URL 率 | 0.0000 | 0.0000 | v2 中两者都没有虚假 URL |
| 引用计数 | 0.1562 | 0.7188 | 引用感知引用更多 |
| 响应剪辑比率 | 0.0000 | 0.0000 | v2固定截断 |
| no_citation 失败 | 27 | 9 | 引用感知能力更强 |

这是本阶段第一个干净的正向 GRPO 信号：

```text
在同一份 v2 one-citation 数据上，显式 citation/support reward 相比 outcome-only reward
显著改善了引用行为，同时保持搜索行为并消除了 response clipping。
```

Citation-Strong 后续实验：

| 指标 | v2 引用感知 | v2 Citation-Strong |
|---|---:|---:|
| search | 1.0000 | 1.0000 |
| format | 0.9625 | 0.9812 |
| URL 有效性 | 0.7188 | 0.7188 |
| 引用精确度 | 0.3937 | 0.4219 |
| 声明支持 | 0.3906 | 0.4219 |
| 不支持的引用率 | 0.3750 | 0.4375 |
| no_citation 失败 | 9 | 9 |
| 响应剪辑比率 | 0.0000 | 0.0000 |

解释：

```text
更强的 citation/support 权重提高了平均 support，
但没有减少 no_citation，反而让 unsupported rate 变差。
单纯调标量权重不是下一步最干净的修复。
```

更新后的决定：

```text
暂时不要上 4 GPU。
不要再盲目跑更长 GRPO。
下一步工程任务是修补 reward/prompt 对精确 markdown URL 引用存在性的处理：
惩罚 no citation、裸 `[S_xxx]`、裸 `[1]`，以及不是 markdown link 的原始 URL。
```

详细报告

```text
reports/deepfactcite_v2_onecite_grpo_ablation_2gpu_20260518.md
```

## 要报告的指标

主要引用指标：

```text
url_validity
citation_precision
claim_support
unsupported_citation_rate
fake_url_rate
citation_count
```

护栏指标：

```text
answer_subem / target subEM
format
search
search turns
response_length clip ratio
no-search 与 no-citation 失败
```

摘要生成器：

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python \
  scripts/deepfactcite/summarize_grpo_rollouts.py \
  logs/grpo/rollouts/<run_name> \
  --out reports/<run_name>_rollout_summary.md
```

## 扩展规则

仅在以下情况下转到更长或 4-GPU 运行：

```text
1. citation-aware 相比 outcome-only 改善 URL validity、claim support 和 unsupported rate；
2. answer/search 行为没有实质性崩塌；
3. response truncation 不是主要失败模式；
4. 启用 checkpoint 保存前已经完成磁盘清理。
```

如果出现以下情况，请停止或修改：

```text
1. citation-aware 只是增加引用数量，但 unsupported rate 仍然很高；
2. outcome-only 在 answer/search 上获胜，而 citation-aware 崩掉；
3. rollout 主要被 no-search/no-citation 格式问题支配；
4. max_response_length=384 截断了太多答案。
```

## 面试级叙述

最有力的项目叙述是：

```text
我先建立了可复现的 SFT baseline 和 answer/search 防护评测，
然后验证了 2-GPU SGLang GRPO 工程路径。第一次 retrieval-hit smoke 表明
backend plumbing 能跑通，但 citation support 仍然很弱。
我没有盲目扩大 GPU，而是构建了 claim-level support-filtered RL 数据，
在真实 rollout 检索器下验证 retriever top-k 命中，并设置了受控的
outcome-only vs citation-aware reward 消融。这个实验干净地检验了：
显式 citation authenticity 和 claim-support reward 是否能在不牺牲
Search-R1 风格 answer/search 目标的情况下改善 attribution。
```

## 第8阶段：奖励解析器和扣分引用修复

为什么存在此阶段：

```text
公平 v2 消融给出了有价值的结果：与 outcome-only 相比，
citation-aware GRPO 改善了 URL validity、citation precision、claim support、
unsupported rate 和 no-citation count。但这个结果还不足以直接扩容，
因为 citation-aware 在 32 个样本中仍有 9 个 no-citation 失败。
```

天真的下一步会是跑更久或使用更多 GPU。我们没有这样做，
因为故障分析指出了一个更狭隘的问题：

```text
模型经常知道答案，也会执行搜索，但有时没有在最终答案中输出严格的 markdown URL 引用。
```

这是一个奖励指定问题，而不仅仅是容量问题。

### 此处被视为有效引用的内容

对于此项目，引用不仅仅是一个看起来像源的令牌。它必须通过
三项检查：

```text
1. Markdown 形式：[some label](https://retrieved-url)
2. URL 来源：URL 必须出现在该 rollout 的检索证据中
3. 声明支持：引用附近的局部声明必须被该 URL 背后的文本片段支持
```

示例：

```text
有效形式：
该洞穴保存有人物形象绘画 [Chauvet Cave](https://example.org/chauvet)。

无效形式：
该洞穴保存有人物形象绘画 [S_abc123]。

无效形式：
该洞穴保存有人物形象绘画 [1]。

无效形式：
该洞穴保存有人物形象绘画 https://example.org/chauvet。
```

无效形式在模型 rollout 中很常见，因为模型会在工具输出中看到片段 ID 和 URL。
如果 reward 接受这些松散形式，
模型可以在不学习目标归因行为的情况下显示为“引用”。

### 奖励/渠道补丁

修补文件：

```text
deepfactcite/reward.py
scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py
scripts/deepfactcite/summarize_grpo_rollouts.py
```

变化：

```text
1. citation-aware 模式现在设置 DFC_REQUIRE_MARKDOWN_CITATION=true。
2. outcome-only 模式设置 DFC_REQUIRE_MARKDOWN_CITATION=false，以保留消融边界。
3. 如果存在检索证据，但最终答案没有 markdown URL 引用，则 reward 上限为 0.08。
4. 如果答案使用裸 bracket 引用或 markdown 外的原始 URL，则 reward 上限为 0.12。
5. v2 prompt 现在要求只输出一个短句，并且只输出一个从检索证据中复制的
   markdown URL 引用。
6. rollout 汇总器可以通过 --recompute-details 使用当前 reward 代码重算诊断细节，
   同时保留日志里记录的 reward。
```

### 发现解析器错误

在扣分上限smoke期间，一个故障示例如下所示：

```text
[NCDAS: Substance Abuse and Addiction Statistics [2025]](https://drugabusestatistics.org)
```

这是markdown。旧的正则表达式无法计数，因为标签包含
内括号年份`[2025]`。这使得指标显示
`citation_count=0`，即使模型产生了扣分链接。

为何重要：

```text
如果解析器少计引用，reward 和报告就可能惩罚或诊断错误的行为。
这会污染对 GRPO 运行结果的解释。
```

修复：

```text
deepfactcite/reward.py 现在用一个小型解析器扫描 markdown 链接，
允许链接标签内部出现 bracket 文本。它还会先移除合法的 markdown 链接，
再检查裸 bracket 或原始 URL，因此有效引用标签里的 `[2025]`
不会被误判为不合格的裸引用。
```

健全性检查通过：

```text
嵌套标签 markdown 链接：
citation_count=1, raw_url_count=0, bare_citation_count=0

裸 [S_1] 加原始 URL：
citation_count=0, raw_url_count=1, bare_citation_count=1
```

### 不完整的Markdown-Cap运行

命令：

```bash
RUN_TAG=20260518_v2_markdowncap_2gpu \
MODE=citation-aware \
EXPERIMENT_NAME=dfc-mixclean200-v2-onecite-markdowncap-20260518_2gpu \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite \
DRY_RUN=0 \
GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

观察结果：

```text
计划 16 steps，只写出了 6 steps。
日志末尾没有 traceback、OOM、NCCL failure 或 Python exception。
停止后 GPU 处于空闲状态。
```

解析器重新计算前的部分结果：

```text
samples = 12
steps = 6
reward = 0.2319
search = 0.7500
url_validity = 0.2500
claim_support = 0.2083
unsupported_citation_rate = 0.5833
citation_count = 0.4167
no_citation failures = 7
```

解析器修复后的诊断重新计算：

```text
total = 0.2492
url_validity = 0.3333
citation_count = 0.5000
no_citation failures = 6
```

解释：

```text
这次 partial run 可作为调试证据，但不是最终实验。
它暴露了 parser 问题，也确认新的 no-citation cap 已生效；
但它不能与完整 16-step run 做公平比较。
```

### 当前解析器修复重新运行

命令：

```bash
RUN_TAG=20260518_v2_markdowncap_parserfix2_2gpu \
MODE=citation-aware \
EXPERIMENT_NAME=dfc-mixclean200-v2-onecite-markdowncap-parserfix2-20260518_2gpu \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite \
DRY_RUN=0 \
GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

启动状态：

```text
训练进入 async rollout。
SearchQAVerlTool 使用 v2 one-citation 离线语料完成初始化。
rollout_data_step_1.jsonl 已创建。
GPU 显存被 2-card GRPO/SGLang 栈占用。
save_freq=0，所以这次仍然是 smoke/ablation，不会消耗大量 checkpoint 磁盘。
```

完成后需要检查的内容：

```text
1. no_citation 失败数：目标低于 9/32。
2. claim_support：目标高于 0.3906。
3. unsupported_citation_rate：目标不高于 0.3750。
4. search：应保持接近 1.0000。
5. response clip ratio：应保持 0.0000。
```

如果这些检查通过，下一步是在磁盘清理后做一次会保存 checkpoint 的 2-GPU 运行。
如果没有通过，不要扩展到 4-GPU；先继续修复数据、prompt 或 reward。

## 第9阶段：小规模与扩展规则

重要区别：

```text
小规模 run 不用于估计最终效果，只用于机制测试。
```

在32个rollout样本处，一个样本将总速率改变3.125个点。
因此：

```text
no_citation 9 vs 10 is not statistically decisive.
unsupported 0.3750 vs 0.4062 is also only about one sample of movement.
```

但小批量运行仍然能说明实验方向是否正确。v2 结果已经足以算作机制信号：

```text
outcome-only -> citation-aware
claim_support: 0.1094 -> 0.3906
unsupported_citation_rate: 0.8438 -> 0.3750
no_citation: 27 -> 9
```

这就是为什么v2值得继续的原因。解析器修复运行有混合证据：

```text
claim_support 提升到 0.4375
citation_precision 提升到 0.4406
format/search 保持 1.0000
但 no_citation 为 10，unsupported 为 0.4062
```

解释：

```text
这不能证明修复不好，但也不足以支撑现在就花 4/8 张 GPU。
它说明下一步便宜实验应该瞄准确切的残余失败：
模型在某些情况下仍然忽略 markdown citation 格式。
```

放大策略：

```text
1. 2-GPU tiny run：验证工程链路和机制。
2. 2-GPU 或 4-GPU medium run：机制成立后，用保存 checkpoint 的运行验证趋势。
3. 8-GPU run：留给最终吞吐或已确认的大规模训练，不用于调试 reward/data 定义。
```

扩展门槛基于最完整的 v2 引用感知基线：

```text
no_citation < 9/32
claim_support > 0.3906
unsupported_citation_rate <= 0.3750
search near 1.0
response clip ratio 0.0
disk cleaned before checkpoint save
```

这些不是通用科学阈值，而是本地工程门槛：更昂贵的运行应该先击败当前最便宜基线中的失败模式，
同时保留 Search-R1 行为。

## 第10阶段： v3提示-修复数据

原因：

```text
v2 parquet 是在 prompt 补丁之前生成的。
这意味着 parser-fix 运行使用了新的 reward 上限，
但旧 prompt 并没有明确禁止裸 [S_xxx]、裸 [1] 和原始 URL。
```

修复：

```text
用当前更严格的 prompt 重新生成同样的 32 行 one-citation 数据。
除此之外，行集合、检索语料、支持阈值、模型、reward 和 2-GPU 设置都保持不变。
```

命令：

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python \
  scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py \
  --rows 32 \
  --out-dir data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite \
  --max-citations 1 \
  --max-claim-tokens 30 \
  --max-query-tokens 18 \
  --max-answer-sentences 1 \
  --strict-one-citation-prompt \
  --citation-format-template
```

数据摘要：

```text
kept_rows = 32
train_rows = 28
test_rows = 4
corpus_docs = 32
avg_selected_citations = 1.0
min_support_score = 1.0
avg_support_score = 1.0
```

检索器验证：

```text
train top-2 URL hit: 28/28
test top-2 URL hit: 4/4
```

当前运行：

```bash
RUN_TAG=20260518_v3_promptfix_2gpu \
MODE=citation-aware \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite \
DRY_RUN=0 \
GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

启动验证：

```text
trainer accepted train=28 / val=4
SearchQAVerlTool 已使用 v3 prompt-fix 语料初始化
reward_manager=deepfactcite_custom
save_freq=0
```

结果：

```text
reports/dfc_mixclean200_v3_promptfix_onecite_2gpu_rollout_summary.md
logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu/
```

| 指标 | v2 引用感知 | v3 prompt-fix | 解读 |
|---|---:|---:|---|
| search | 1.0000 | 0.9375 | v3有2次未搜索失败 |
| format | 0.9625 | 0.9875 | v3更好 |
| URL 有效性 | 0.7188 | 0.9375 | 好得多 |
| 引用精确度 | 0.3937 | 0.6312 | 好得多 |
| 声明支持 | 0.3906 | 0.6250 | 好得多 |
| 不支持的引用率 | 0.3750 | 0.1250 | 好得多 |
| 虚假 URL 率 | 0.0000 | 0.0000 | 干净 |
| 引用计数 | 0.7188 | 0.9375 | v3引用更可靠 |
| no_citation 失败 | 9 | 2 | v3 修复了主要失败模式 |
| 响应剪辑比率 | 0.0000 | 0.0000 | 仍然干净 |

解释：

```text
这是第一个通过本地扩展门槛的结果。它说明主要瓶颈不只是数据小或 GPU 规模不足；
缺失的一环是训练数据 prompt 与 reward parser/cap 之间的对齐。
```

剩余警告：

```text
两个 no-search/no-answer 失败说明下一轮必须关注搜索行为。
模型没有崩，但下一次保存 checkpoint 的 medium run 应该至少保持这个搜索水平。
```

下一个决定：

```text
不要用 8 GPU 做调试。磁盘清理后做一次会保存 checkpoint 的 medium run：
如果有 4 GPU 就用 4 GPU，否则用 2 GPU。目标是产出可评测 checkpoint，
而不是再做一次不保存 checkpoint 的 smoke。
```

审核:

```text
检查 rows：32
检查 steps：16
训练 reward 重算：完全匹配，max diff = 0.0
诊断指标重算：与 logged details 完全匹配
Prompt target leakage：0/28 train, 0/4 test
离线检索器目标 URL top-1 hit：28/28 train, 4/4 test
bare_citation_count mean: 0.0
raw_url_count mean: 0.0
```

审核失败：

```text
2 no_search/no_citation rows:
  Boredoms
  Chauvet-Pont d'Arc Cave

4 unsupported citation rows:
  climate-policy question, two samples
  Jim Umbricht 问题，两个样本
```

引用标签警告：

```text
模型基本学会了附上真实 URL，但很多 label 仍然是 snippet ID。
30 个 markdown citation 中，22 个 label 是 S_xxx 风格，8 个是人类可读 label。
这对当前 URL 来源和支持性目标是可接受的；
但如果最终产品需要人类可读引用标签，就应该加入 label-quality 指标。
```

## 第7阶段：CPU 模式整合和 4-GPU 准备

在 v3 prompt-fix 结果之后，GPU 工作可以安全暂停。CPU-only 阶段仍有有用工作：

```text
1. 把 rollout 样例保存成可读 gallery；
2. 写出适合新手阅读的路径；
3. 准备 4-GPU saved-checkpoint 命令；
4. 再次租 GPU 前检查 data/report 一致性；
5. 只清理安全的临时或空训练 artifact。
```

新工件：

```text
reports/deepfactcite_v3_promptfix_rollout_gallery_20260518.md
reports/deepfactcite_v3_promptfix_rollout_gallery_samples_20260518.jsonl
docs/deepfactcite_learning_index_20260518.md
docs/deepfactcite_4gpu_saved_run_plan_20260518.md
scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

4-GPU 包装脚本默认为 `DRY_RUN=1`，并且只保存模型 checkpoint：

```text
GRPO_ACTOR_CKPT_SAVE_CONTENTS=[model]
GRPO_SAVE_FREQ=16
GRPO_MAX_ACTOR_CKPT_TO_KEEP=1
```

原因：

```text
磁盘仍然紧张。model-only checkpoint 已足够评测，
并且比带 extra state、可恢复训练的 checkpoint 节省很多磁盘。
```

下一个 GPU 操作：

```text
先运行一个短的 4-GPU checkpoint 写入 sanity 检查。如果能干净启动、写出 checkpoint，
并且 search/citation 指标正常，再继续 64-step saved medium run。
```
