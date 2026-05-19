# Search-R1-DeepFactCite 项目详细文档

日期：2026-05-19（仓库环境 UTC）

## 0. 摘要结论

Search-R1-DeepFactCite 是一个基于 Search-R1 / veRL 的搜索增强问答训练项目。它的目标不是简单做一个 RAG 应用，而是训练一个会“先搜索、再基于证据回答、并且给出可信引用”的语言模型智能体。

项目研究的问题可以概括为：

> 如何让 Qwen3-8B 这类开源大模型在开放域问题中主动搜索证据，并在最终回答里只引用真实检索到、且能支撑当前声明的来源。

普通问答模型常见的问题是“答案看起来对，但来源不可信”。本项目把这个问题拆成三层：

- 答案是否正确：模型有没有回答到问题本身。
- 搜索是否有效：模型有没有发起搜索，搜索结果是否能提供证据。
- 引用是否可信：答案里的 URL 是否来自当前检索轨迹，引用附近的 claim 是否真的被该片段支持。

因此，本项目的核心不是“让模型输出更多链接”，而是让模型学会：

```text
问题
-> 生成搜索动作
-> 工具返回证据
-> 读取证据
-> 生成带 markdown citation 的回答
-> reward 检查答案、搜索、引用、声明支持和格式
-> GRPO 强化更好的搜索/引用行为
```

一句话心智模型：

```text
Ray/veRL 负责训练调度，FSDP 负责训练模型，SGLang 负责高吞吐 rollout，
Search-R1 工具环境负责搜索，DeepFactCite reward 负责判断引用是否可信。
```

当前阶段已经完成的主线能力：

- 从 DeepCiteFact 轨迹构建 DeepFactCite 风格 SFT / RL 数据。
- 训练 Qwen3-8B LoRA SFT baseline。
- 构建轻量词汇检索器和 Search-R1 风格离线检索 corpus。
- 在 vLLM 路径上评测 SFT 模型的答案、搜索和引用指标。
- 在 SGLang / veRL 多轮工具调用路径上跑通 DeepFactCite GRPO。
- 实现 citation-aware reward，并与 outcome-only reward 做过受控消融。
- 形成 MixClean200 SFT baseline 和 v3 prompt-fix GRPO 正向信号。

当前最重要的阶段性结论：

| 阶段 | 结论 |
|---|---|
| SFT | `mixclean200` 是当前较好的 SFT 初始化，能提升答案/搜索行为，也改善 URL 有效性 |
| 早期 GRPO smoke | 工程链路能跑通，但 query 级 retrieval hit 不足以保证 claim support |
| v2 one-citation | citation-aware reward 明显优于 outcome-only reward |
| v3 prompt-fix | 显著改善 URL validity、citation precision、claim support，并降低 unsupported citation rate |
| 4 GPU 保存训练 | saved32 checkpoint 已完成转换与评测，工程链路成立；held-out eval 暂未证明相对 MixClean200 SFT 稳定提效 |

需要特别强调：

```text
训练跑完不等于项目成功。
只有保存下来的 checkpoint 在答案/搜索防护评测和引用质量评测上都通过，
才可以把一次 GRPO 训练提升为有效模型结果。
```

### 0.1 Search-R1 先解决了什么

Search-R1 的出发点是：大模型不是天生会用搜索引擎。你在 prompt 里告诉它“可以搜索”，它也可能搜错关键词、搜太多轮、读不到关键证据，或者明明需要搜索却直接凭记忆回答。

Search-R1 把问答过程改造成一个可训练的工具交互轨迹：

```text
用户问题
-> 模型在 <think> 里推理
-> 模型用 <search> 生成搜索 query
-> 检索环境把结果放进 <information>
-> 模型阅读结果，继续搜索或给出 <answer>
-> reward 根据最终答案和轨迹质量打分
-> PPO/GRPO 等 RL 方法更新模型
```

和普通 RAG 的关键差别是：普通 RAG 的搜索通常由系统固定执行；Search-R1 让搜索 query、搜索轮次和何时停止都成为模型策略的一部分。模型不是只学“根据文档回答”，而是学习“什么时候查、查什么、查完怎么答”。

Search-R1 论文和代码中最值得保留的思想有三点：

| 思想 | 小白版理解 | 对本项目的意义 |
|---|---|---|
| 搜索是 action | 搜索不是预处理，而是模型自己做出的动作 | 本项目沿用 search/tool 轨迹 |
| reward 看完整轨迹 | 好坏不只看单个 token，而看一次交互是否完成任务 | 本项目把引用质量也放进轨迹奖励 |
| 检索器可替换 | 本地 BM25、稠密检索、在线搜索都可以接入 | 本项目补充了带 URL 的本地检索和评测服务 |

因此，Search-R1 已经搭好了“会搜索的 RL 智能体”底座。

### 0.2 本项目相比 Search-R1 改进了什么

本项目的核心改进不是换一个检索器，也不是简单把回答写长，而是把 Search-R1 的训练目标从“搜索后答对”扩展为“搜索后答对，并且引用可信”。

可以用一个例子理解：

```text
证据只说：Chauvet-Pont d'Arc Cave 有旧石器时代洞穴壁画。
模型却回答：它是欧洲最早、保存最完整的史前艺术遗址之一 [source](URL)。
```

这里 URL 可能来自检索结果，所以 `url_validity` 是高的；但证据并不支持“最早、保存最完整”这些更强声明，所以 `claim_support` 是低的。Search-R1 风格的答案奖励未必能抓住这个问题，本项目的 DeepFactCite reward 会把它作为重点失败模式。

本项目相对 Search-R1 的主要增量如下：

| 增量 | 做了什么 | 解决的问题 |
|---|---|---|
| DeepFactCite prompt | 要求用 markdown citation `[短说明](URL)`，且 URL 必须来自当前检索 | 防止模型随手编链接 |
| URL 保留检索器 | 检索返回的 snippet 中保留 `URL:` 字段 | 让引用可验证 |
| DeepCiteFact 数据转换 | 构建 SFT/RL parquet 和 citation corpus | 让模型先学会搜索引用格式 |
| strict / mix 数据过滤 | 按 URL validity、claim support、unsupported rate 过滤轨迹 | 减少坏引用样本污染 |
| citation-aware reward | 同时奖励答案、搜索、格式、URL 真实性和 claim support | 防止“答案对但引用乱贴” |
| hard cap 规则 | 对无 citation、裸 `[1]`、raw URL、fake URL、不支持引用设低分上限 | 让严重引用错误无法拿高分 |
| one-citation GRPO 消融 | 对比 outcome-only 和 citation-aware 两种奖励 | 证明引用奖励本身是否有效 |
| guardrail eval | 同时跑 Search-R1 BM25、ShortQA、DeepFactCite strict 评测 | 防止引用训练破坏原有问答/搜索能力 |

最重要的变化可以压缩成一句话：

```text
Search-R1 训练模型“会查资料再答题”；
本项目训练模型“查到资料后，只引用真实来源，并且引用要支撑旁边那句话”。
```

这也是本项目对外最稳妥的表述。除非在相同模型、相同检索器、相同数据、相同 prompt 和相同评测下完成严格复现，否则不要把项目说成“击败 Search-R1”。更准确的定位是：在 Search-R1 风格搜索智能体上，增加引用真实性和声明级支持优化。

还需要说明一个容易被误解的点：本项目没有必要强行复现原始 Search-R1 的重型搜索库。原始 Search-R1 的论文设置通常围绕 Wikipedia 语料、E5 向量检索、FAISS / ANN、Pyserini / BM25 等完整开放域检索栈；本项目的核心问题是 citation faithfulness，检索库必须保留 URL、片段和可验证引用证据。两者的语料目标、检索器形态和基座模型都不同。

| 对比项 | 原始 Search-R1 常见设置 | 本项目设置 | 为什么这样做 |
|---|---|---|---|
| 搜索库 | 面向 NQ/HotpotQA 等开放域 QA 的 Wikipedia / wiki-18 检索库 | DeepFactCite / ShortQA / 本地 BM25 guardrail / 带 URL 的 citation corpus | 引用训练必须知道 URL 是否来自当前 evidence |
| 检索目标 | 让模型搜到能回答短事实问题的段落 | 让模型搜到能支撑具体 claim 的来源片段 | URL 真实不等于 claim 被支持 |
| 基座模型 | 主要参考 Qwen2.5 / Llama3.2 系列结果 | 本项目主线是 Qwen3-8B | 不能把 backbone 差异误当成 reward 差异 |
| 工程成本 | 大索引、向量库、检索服务、版本 pin 和资源依赖都较重 | 优先使用轻量、可审计、可复现的检索库 | 收窄变量，先验证引用 reward 是否有效 |
| 评测口径 | 论文 benchmark 数字可作为背景参考 | 本地同环境 benchmark 是主要依据 | 相同检索器、prompt、数据和 evaluator 下的对比更干净 |

因此，本项目的合理目标不是“完整复刻 Search-R1 检索栈”，而是：

```text
在不同搜索库和 Qwen3-8B 基模上，
用较轻量但可审计的检索环境达到有竞争力的 benchmark 准确率，
同时把 Search-R1 没有重点处理的引用真实性和 claim support 纳入训练目标。
```

当前本地 benchmark 已经能支撑这个定位：MixClean200 SFT 在 Search-R1 BM25 200 上达到 `answer_subem=0.490`、`search_success=1.000`，在 ShortQA32 上达到 `answer_subem=0.500`。这些数字不能拿来直接对比 Search-R1 论文表格，但说明在本项目受控环境里，答案/搜索准确率有竞争力，后续引用训练必须守住这条基线。

## 1. 项目概述

### 1.1 背景与问题动机

大模型在事实问答里有两类常见问题。

第一类是事实错误。模型可能凭参数记忆回答，但记忆过期、混淆实体，或者在长文本里编造细节。

第二类是引用错误。模型可能给出一个看似权威的链接，但这个链接并不是检索返回的来源；或者链接是真的，但它只支持很窄的事实，模型却把它放在一个更宽泛的声明后面。

举例说：

```text
检索片段只说：某球员 1962 年加入某队。
模型回答却说：该球员从先发投手转型为救援投手，并在两支球队间角色逐渐变化。
```

这时 URL 可能是真实的，但被引用的声明并没有被该 URL 的片段充分支持。这就是本项目最关注的问题：citation faithfulness，也就是引用忠实性。

### 1.2 为什么不是普通 RAG

普通 RAG 通常是固定流程：

```text
用户问题
-> 系统检索 top-k 文档
-> 把文档拼进 prompt
-> LLM 生成答案
```

本项目的流程更接近 Search Agent 训练：

```text
用户问题
-> 模型决定是否搜索
-> 模型生成 <search> 或 <google_search> 动作
-> 外部工具返回 <information> 或 <tool_response>
-> 模型继续搜索或输出 <answer>
-> reward 对完整轨迹打分
-> GRPO 优化模型策略
```

差异如下：

| 维度 | 普通 RAG | Search-R1-DeepFactCite |
|---|---|---|
| 检索时机 | 系统固定检索 | 模型作为 action 主动搜索 |
| 搜索 query | 用户原问题或规则改写 | 模型自己生成 |
| 搜索轮次 | 通常固定 | 模型学习何时停止 |
| 训练目标 | 模仿答案或最小化 loss | 优化轨迹级 reward |
| 引用检查 | 多数只检查格式 | 检查 URL 是否来自证据、claim 是否被支持 |
| 成本控制 | 工程参数控制 | reward 惩罚过长输出和无效搜索 |
| 调试单元 | 单次答案 | 完整 rollout 轨迹 |

### 1.3 项目最终想训练什么能力

理想模型应该具备这些能力：

- 遇到开放域事实问题时，先用搜索动作找证据。
- 搜索 query 尽量简洁、有效，不无限重复搜索。
- 只根据工具返回的证据写事实声明。
- 用 markdown citation `[短说明](URL)` 标注来源。
- URL 必须来自当前检索轨迹，不能编造。
- 引用要紧贴它支持的 claim，不能把所有链接堆在答案最后。
- 如果证据不足，要缩小回答范围，而不是硬编背景。

### 1.4 SFT 和 GRPO 分别解决什么

本项目不是直接上 RL。它分两步：

```text
SFT：先教模型基本格式和搜索/引用轨迹。
GRPO：再用 reward 优化 SFT 难以直接学稳的行为。
```

SFT 适合教模型：

- 怎样输出 `<think>`、`<search>`、`<answer>`。
- 搜索返回后再回答，而不是直接凭记忆回答。
- 长答案里大致怎样放 markdown citation。
- 最终答案应该放在 `<answer>...</answer>` 中。

但 SFT 不擅长直接优化这些行为：

- 哪个搜索 query 更好。
- 搜索几轮合适。
- 哪个 URL 应该引用。
- 引用附近的声明是否真的被片段支持。
- 答案正确、引用真实、输出简洁之间如何权衡。

这些行为都更适合用 reward 反馈，因此本项目在 SFT 后引入 GRPO。

### 1.5 当前实现范围

当前仓库主要覆盖以下范围：

- DeepFactCite 数据转换。
- DeepFactCite prompt 和工具协议。
- 轻量 lexical retriever。
- DeepFactCite reward 和 diagnostics。
- Qwen3 LoRA SFT 启动脚本。
- LoRA merge 工具。
- vLLM 评测脚本。
- SGLang GRPO 训练 wrapper。
- rollout summary 生成器。
- 多份实验记录、失败台账和运行计划。

当前不应夸大的范围：

- 这不是一个完整产品级在线 Deep Research 系统。
- 当前 action space 主要围绕搜索，没有网页全文浏览、点击、代码执行等复杂工具。
- 当前 claim support 主要基于 snippet / local overlap / 可选 judge，不等同于完整人工事实核查。
- 已有 GRPO 正向信号主要来自小规模受控实验，最终模型结论必须依赖保存 checkpoint 后的系统评测。

## 2. 数据与协议设计

### 2.1 工具协议

本仓库里存在两套相近协议。

DeepFactCite 本地 prompt 使用：

```xml
<think>reasoning</think>
<search>plain text query</search>
<information>retrieved evidence</information>
<answer>final answer with citations</answer>
```

Search-R1 / agentic-rl-searchqa 的 SGLang 工具路径使用：

```xml
<google_search>plain text query</google_search>
<tool_response>retrieved evidence</tool_response>
```

`deepfactcite.reward.convert_deepcite_tags()` 会把两套协议对齐：

```text
<google_search> -> <search>
<tool_response> -> <information>
```

这使得 reward 可以统一评估两种 rollout 格式。

关键边界：

```text
模型可以生成 search / google_search 和 answer。
模型不能生成 information / tool_response。
information / tool_response 必须由检索环境注入。
```

这个边界很重要。如果模型被允许自己生成工具返回，它就可能学会伪造网页标题、URL 和摘要，整个引用训练就失去意义。

### 2.2 一条轨迹长什么样

简化后的轨迹如下：

```text
User: Tell me about the Chauvet-Pont d'Arc Cave.

Assistant:
<think>I need a concise supported fact.</think>
<search>Chauvet-Pont d'Arc Cave what is it</search>

Environment:
<information>
<snippet id=S_xxx>
Title: ...
URL: https://...
Text: ...
</snippet>
</information>

Assistant:
<answer>The Chauvet-Pont d'Arc Cave is a prehistoric cave site known for Paleolithic paintings [source](https://...).</answer>
```

reward 会检查：

- 有没有 `<answer>`。
- 有没有搜索。
- URL 是否在 `<information>` 里出现过。
- citation 附近的 claim 是否能从对应 snippet 推出。
- 是否存在裸 `[S_xxx]`、裸 `[1]` 或 markdown 之外的 raw URL。
- 输出是否过长或结构错误。

### 2.3 数据来源

本项目围绕 DeepCiteFact 数据构建长文本引用问答训练数据，并保留 Search-R1 / ShortQA 风格评测作为防护指标。

主要数据目录：

| 数据目录 | 用途 | 当前规模或特点 |
|---|---|---|
| `data/deepfactcite` | 基础 DeepFactCite SFT/RL 数据 | SFT 1200，RL 1200，corpus 6118 |
| `data/deepfactcite_strict` | 更严格过滤后的 SFT/RL 数据 | SFT 947，RL 1200，corpus 4766 |
| `data/deepfactcite_mix/sft` | soft + strict 混合 SFT 数据 | train 1693，test 105 |
| `data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite` | v3 GRPO one-citation 数据 | kept 32，train 28，test 4，corpus 32 |
| `data/shortqa_guardrail` | ShortQA 防护评测 | 检查答案/搜索行为是否崩坏 |
| `data/searchr1_core_guardrail` | Search-R1 核心 BM25 防护评测 | NQ/HotpotQA 风格问题 |

`data/deepfactcite_strict/summary.json` 记录了 strict 数据的过滤过程：

```text
seen: 2916
kept: 947
skipped: 1969
主要跳过原因：
  weak_claim_support: 1947
  too_many_unsupported_citations: 22
```

这说明原始轨迹里大量引用并不能通过声明级支持检查。这个现象正是项目要解决的问题。

### 2.4 为什么要做 claim-filtered one-citation 数据

早期 GRPO smoke 发现：只保证检索器能命中相关 URL 还不够。模型仍然会把很宽的答案写出来，然后贴上一个真实但支撑不足的 URL。

因此 v2 / v3 数据构造把问题缩小：

- 每条样本只选择 1 个高支持 claim。
- 答案要求 1 句或少量句子。
- query token 数有限制，避免问题太宽。
- claim token 数有限制，避免被支持范围太大。
- 目标 URL 必须能被 top-2 检索命中。
- prompt 明确要求 exactly 1 markdown citation。

v3 数据摘要：

```text
source_sft: data/deepfactcite_mix/sft/train.parquet
kept_rows: 32
train_rows: 28
test_rows: 4
corpus_docs: 32
avg_selected_citations: 1.0
min_support_score: 1.0
avg_support_score: 1.0
```

这批数据不是最终大规模训练集，而是用于受控验证：

```text
当 evidence 已经很窄、目标 citation 很明确时，
citation-aware reward 是否能推动模型输出真实、贴近证据的引用？
```

## 3. 系统架构

### 3.1 三层结构

可以把项目理解为三层：

| 层 | 作用 | 主要文件 |
|---|---|---|
| 任务逻辑层 | prompt、检索、reward、citation 解析 | `deepfactcite/` |
| 数据和实验脚本层 | 数据构建、训练启动、评测、汇总 | `scripts/deepfactcite/` |
| 训练框架层 | SFT、PPO/GRPO、Ray worker、rollout | `verl/` 和外部 SearchShortQA/verl |

新增核心代码：

| 文件 | 作用 |
|---|---|
| `deepfactcite/prompts.py` | 定义 DeepFactCite 搜索引用 prompt |
| `deepfactcite/reward.py` | 解析答案、搜索、引用，并计算 reward |
| `deepfactcite/retriever_server.py` | 轻量 lexical retriever，提供 `/retrieve` API |
| `verl/utils/reward_score/deepfactcite.py` | 把 reward 接入本仓库 veRL reward score |
| `verl/trainer/main_ppo_deepfactcite.py` | 本仓库内 DeepFactCite GRPO 入口 |
| `scripts/deepfactcite/prepare_data.py` | 从 DeepCiteFact 构建 SFT/RL 数据 |
| `scripts/deepfactcite/build_sft_mix.py` | 混合 soft/strict SFT 数据 |
| `scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py` | 构建 one-citation GRPO 数据 |
| `scripts/deepfactcite/verl_deepfactcite_reward.py` | SGLang GRPO 自定义 reward manager |
| `scripts/deepfactcite/summarize_grpo_rollouts.py` | 汇总 rollout JSONL 指标 |

### 3.2 训练与评测流水线

整体流程：

```text
DeepCiteFact 原始轨迹
-> prepare_data.py
-> deepfactcite SFT / RL parquet
-> train_sft_qwen3.sh
-> Qwen3-8B LoRA SFT adapter
-> merge_lora_adapter.py
-> vLLM / guardrail eval
-> prepare_sglang_grpo_claim_filtered.py
-> SGLang 多轮工具 rollout
-> DeepFactCite reward
-> GRPO actor update
-> rollout summary
-> checkpoint eval
```

### 3.3 检索组件

本项目用过两类检索路径。

第一类是本仓库内轻量 lexical retriever：

```text
deepfactcite/retriever_server.py
scripts/deepfactcite/start_lexical_retriever.sh
```

它读取 `corpus.jsonl`，对 query 和文档 contents 做简单词汇匹配，返回 Search-R1 兼容的 `/retrieve` 响应。

第二类是 SGLang GRPO 里使用的 Search-R1 offline search tool。它通过外部 `agentic-rl-searchqa` 配置读取本地 corpus，并在多轮 agent loop 中注入工具返回。

两者的共同要求是：

```text
返回内容必须带 URL。
否则模型即使想引用，也无法生成可验证的 markdown citation。
```

### 3.4 4 GPU GRPO 运行里的角色

在 SGLang GRPO 路径中，训练系统大致由以下角色组成：

| 组件 | 作用 |
|---|---|
| Ray | 调度训练 worker、SGLang server、agent loop、reward manager |
| FSDP actor | 负责训练、logprob、actor update |
| SGLang engine | 负责高吞吐 rollout 生成 |
| AgentLoopWorker | 负责多轮 search 工具调用和轨迹拼接 |
| DeepFactCiteRewardManager | 对完整 response/trajectory 打分 |

一次 GRPO step 可以理解为：

```text
1. 取一个 prompt。
2. 复制成 n 条候选轨迹。
3. SGLang 生成多轮搜索/回答。
4. AgentLoop 调用 google_search 工具并拼接 tool response。
5. DeepFactCite reward 给每条轨迹打分。
6. 对同一 prompt 的 n 条结果做组内标准化 advantage。
7. FSDP actor 根据 advantage 更新参数。
8. 记录 rollout JSONL 和训练指标。
```

GRPO 的直觉：

```text
同一个问题生成多条答案。
比组内平均更好的答案被鼓励。
比组内平均更差的答案被抑制。
```

它比 PPO 少维护一个 value critic，更适合当前 8B 模型和结果型 reward 的资源约束。

## 4. 奖励机制

### 4.1 总体公式

`deepfactcite/reward.py` 中的默认权重是：

```text
R =
  0.25 * answer
+ 0.35 * citation
+ 0.25 * support
+ 0.10 * format
+ 0.05 * search
- 0.05 * cost
```

SGLang citation-aware 模式常用权重是：

```text
answer=0.15
citation=0.35
support=0.30
format=0.10
search=0.05
cost=0.05
```

outcome-only 消融会弱化或关闭 citation/support 权重，用来检查：

```text
同样数据、同样模型、同样 rollout 栈下，
显式引用奖励是否真的改善 citation 行为。
```

### 4.2 奖励组件含义

| 组件 | 检查什么 | 为什么重要 |
|---|---|---|
| `answer_subem` | 最终答案是否覆盖 gold target | 防止只学会引用格式却不回答问题 |
| `format` | `<answer>`、搜索标签、结尾结构是否合规 | 防止训练轨迹不可解析 |
| `search` | 是否在答案前搜索、搜索次数是否超预算 | 保持 Search-R1 风格行为 |
| `url_validity` | citation URL 是否来自当前 evidence | 防止伪造链接 |
| `citation_precision` | citation 附近 claim 是否被支持 | 防止链接真实但不支撑答案 |
| `claim_support` | claim 与对应文档的词汇/语义支持度 | 衡量引用忠实性 |
| `cost` | 输出长度和搜索成本 | 防止无限长回答或无效多搜 |

### 4.3 硬上限和格式惩罚

reward 不只是加权求和，还会对一些严重问题设置上限。

例如：

- 如果格式错误，总分会被 capped。
- 如果 citation URL 不来自 evidence，总分最高只能到较低区间。
- 如果有 evidence 却没有 markdown citation，citation-aware 模式会把总分压到很低。
- 如果出现裸 `[S_xxx]`、裸 `[1]` 或 markdown 外 raw URL，也会触发低分上限。
- 如果 citation 大量不被支持，总分会被进一步限制。

这些 hard cap 的目的不是让 reward 更复杂，而是让模型明确知道：

```text
“答案像是对的，但没有可信引用”不是高质量轨迹。
```

### 4.4 URL 有效不等于引用可信

这是项目最重要的概念。

| 情况 | URL validity | Claim support | 说明 |
|---|---:|---:|---|
| URL 是模型编的 | 低 | 低 | 典型虚假引用 |
| URL 来自检索，但 claim 过宽 | 高 | 低 | 本项目重点失败模式 |
| URL 来自检索，claim 被片段支持 | 高 | 高 | 目标行为 |

因此只统计 URL 是否真实是不够的。DeepFactCite reward 还要看 citation 附近的局部声明。

## 5. 训练路径

### 5.1 SFT 路径

SFT 入口：

```bash
MODEL_SIZE=8B N_GPUS=2 TOTAL_STEPS=200 \
  bash scripts/deepfactcite/train_sft_qwen3.sh
```

常用 wrapper：

```text
scripts/deepfactcite/launch_mix_clean_sft_200.sh
```

当前 SFT 基线：

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200/global_step_200
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16
```

SFT 使用 LoRA：

```text
target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
默认 lora_rank=32
```

SFT 主要教模型基础行为：

- 输出工具协议。
- 先搜索再回答。
- 使用 markdown citation。
- 控制回答格式。

### 5.2 LoRA 合并

合并工具：

```text
scripts/deepfactcite/merge_lora_adapter.py
```

重要经验：

```text
bf16 merge 不一定等价于 dynamic LoRA forward。
继续训练或评估 merged parent 前，应做 logits 等价性检查。
```

已有失败台账记录了旧 bf16 merged parent 与 PEFT dynamic LoRA 不等价的问题。当前脚本默认把 `--torch-dtype` 设为 `float32`，避免把 fp32 LoRA 增量直接加到 bf16 base weight 上造成明显差异。

### 5.3 GRPO 路径

本仓库内有一个早期 DeepFactCite GRPO 入口：

```text
verl/trainer/main_ppo_deepfactcite.py
scripts/deepfactcite/train_grpo_deepfactcite_qwen3.sh
```

但 Qwen3 与原版 Search-R1 的 vLLM pin 存在兼容限制。当前实际更重要的 GRPO 路径是 SGLang 多轮工具调用路径：

```text
scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
scripts/deepfactcite/run_sglang_grpo_v3_promptfix_2gpu_saved.sh
scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
scripts/deepfactcite/verl_deepfactcite_reward.py
```

4 GPU v3 保存版 wrapper 默认是 dry-run：

```bash
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

真正运行需要显式关闭 dry-run：

```bash
DRY_RUN=0 \
GRPO_TOTAL_STEPS=64 \
GRPO_SAVE_FREQ=16 \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260518_4gpu_saved \
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

当前 4 GPU saved32 实验已经证明了保存、转换和评测链路可以闭环，但这不等于模型效果已经胜出。后续正式 GRPO 仍必须把 checkpoint 评测作为 gate，而不能只根据 rollout 训练日志下结论。

## 6. 评测与当前结果

### 6.1 SFT baseline 结果

ShortQA32 防护评测：

| 模型 | answer subEM | total | URL validity | claim support | unsupported | search turns |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B | 0.219 | 0.189 | 0.031 | 0.031 | 0.906 | 2.625 |
| Soft100 LoRA | 0.438 | 0.391 | 0.812 | 0.292 | 0.557 | 1.156 |
| MixClean200 LoRA | 0.500 | 0.427 | 0.969 | 0.333 | 0.495 | 1.125 |

Search-R1 BM25 200 防护评测：

| 模型 | subEM | 搜索成功 | 搜索轮数 | 预算失败 |
|---|---:|---:|---:|---:|
| Base | 0.290 | 0.960 | 1.720 | 0.170 |
| Soft100 | 0.450 | 0.990 | 1.195 | 0.045 |
| MixClean200 | 0.490 | 1.000 | 1.160 | 0.025 |

这说明 MixClean200 不只是让模型更会贴链接，也没有破坏 Search-R1 风格的答案/搜索能力。因此它被选为后续 GRPO 的 SFT 初始化。

### 6.2 早期 GRPO smoke

2 GPU retrieval-hit smoke 主要验证工程链路：

```text
SGLang rollout
DeepFactCite reward
offline corpus
rollout JSONL
summary report
```

结果：

| 指标 | 值 |
|---|---:|
| reward | 0.218 |
| search | 0.938 |
| URL validity | 0.594 |
| citation precision | 0.126 |
| claim support | 0.122 |
| unsupported citation rate | 0.755 |

结论：

```text
链路跑通了，但引用质量不够。
问题不是模型完全不会搜索，而是它会把真实 URL 放在不受支持的宽声明后面。
```

### 6.3 v2 one-citation 消融

v2 使用同一份 one-citation 数据，对比 outcome-only 与 citation-aware reward：

| Metric | v2 Outcome-Only | v2 Citation-Aware |
|---|---:|---:|
| search | 0.9688 | 1.0000 |
| format | 0.9812 | 0.9625 |
| URL validity | 0.1562 | 0.7188 |
| citation precision | 0.1125 | 0.3937 |
| claim support | 0.1094 | 0.3906 |
| unsupported citation rate | 0.8438 | 0.3750 |
| fake URL rate | 0.0000 | 0.0000 |
| citation count | 0.1562 | 0.7188 |

这个结果说明：

```text
在相同数据、相同模型、相同 rollout 栈下，
citation-aware reward 明显改善了引用行为。
```

但 v2 仍有较多 no-citation 失败，因此还不能直接扩容。

### 6.4 v3 prompt-fix 结果

v3 在数据 prompt 中更明确地要求：

- exactly 1 markdown citation。
- URL 必须从工具返回复制。
- 禁止裸 `[S_xxx]`、裸 `[1]` 和 raw URL。
- 答案只写 narrow supported fact。

v3 2 GPU 16-step 结果：

| Metric | Mean |
|---|---:|
| reward | 0.5120 |
| total | 0.4807 |
| format | 0.9875 |
| search | 0.9375 |
| URL validity | 0.9375 |
| citation precision | 0.6312 |
| claim support | 0.6250 |
| unsupported citation rate | 0.1250 |
| fake URL rate | 0.0000 |
| citation count | 0.9375 |

与 v2 相比，v3 是当前最清晰的正向信号：

```text
URL validity、citation precision、claim support 明显提升；
unsupported citation rate 明显下降；
fake URL rate 仍为 0。
```

需要注意：

```text
v3 2 GPU 结果是 no-save 小规模诊断，不是最终模型 checkpoint 结果。
```

### 6.5 4 GPU 诊断与保存训练

4 GPU no-save N=4 diagnostic 结果：

| Metric | Mean |
|---|---:|
| reward | 0.5162 |
| total | 0.4845 |
| format | 0.9938 |
| search | 0.9688 |
| URL validity | 0.9375 |
| citation precision | 0.6359 |
| claim support | 0.6328 |
| unsupported citation rate | 0.1719 |
| fake URL rate | 0.0000 |
| citation count | 0.9375 |

这说明 v3 在 4 GPU / rollout.n=4 设定下仍有可用信号。保存型训练的目标是产生可以评测的 actor checkpoint，而不是再做一次 smoke test。

saved32 运行已产出并评测过一个 actor checkpoint：

```text
logs/dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2.log
logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2/
outputs/deepfactcite/grpo/dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2/
outputs/deepfactcite/grpo/dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2_hf/
```

评测结论要谨慎：

| 评测 | MixClean200 SFT | GRPO saved32 | 结论 |
|---|---:|---:|---|
| Search-R1 BM25 200 `answer_subem` | 0.490 | 0.410 | GRPO 下降 |
| ShortQA32 `total` | 0.4268 | 0.4319 | 小幅上升，幅度不足 |
| ShortQA32 `answer_subem` | 0.5000 | 0.5000 | 持平 |
| DeepFactCite strict47 `total` | 0.1416 | 0.1521 | 小幅上升 |
| DeepFactCite strict47 `unsupported_citation_rate` | 0.8787 | 0.8918 | 仍然很高且略差 |

因此，saved32 的正确定位是：

```text
保存型 GRPO、checkpoint 转换和三类评测 pipeline 已经跑通；
但该 checkpoint 不能作为“GRPO 相比 MixClean200 SFT 明显提效”的最终结果。
```

下一轮保存型运行必须继续检查：

- Search-R1 BM25 200 是否守住答案/搜索能力。
- ShortQA32 是否不退化。
- DeepFactCite Strict47 的 `claim_support` 是否实质上升。
- `unsupported_citation_rate` 是否下降，而不是只提高 citation 数量。
- rollout summary 是否保持 search、citation、support 指标。

## 7. 如何复现主要流程

### 7.1 准备数据

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite

python scripts/deepfactcite/prepare_data.py \
  --deepcitefact-dir /root/autodl-tmp/DeepCiteFact \
  --output-dir data/deepfactcite
```

构建混合 SFT：

```bash
python scripts/deepfactcite/build_sft_mix.py \
  --soft-dir data/deepfactcite/sft \
  --strict-dir data/deepfactcite_strict/sft \
  --output-dir data/deepfactcite_mix/sft
```

构建 v3 one-citation GRPO 数据：

```bash
python scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py \
  --rows 32 \
  --out-dir data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite \
  --max-citations 1 \
  --max-claim-tokens 30 \
  --max-query-tokens 18 \
  --max-answer-sentences 1 \
  --strict-one-citation-prompt \
  --citation-format-template
```

### 7.2 启动轻量检索器

```bash
CORPUS=data/deepfactcite/corpus.jsonl PORT=8000 \
  bash scripts/deepfactcite/start_lexical_retriever.sh
```

### 7.3 训练 SFT

```bash
MODEL_SIZE=8B \
N_GPUS=2 \
TOTAL_STEPS=200 \
DATA_DIR=data/deepfactcite_mix/sft \
EXPERIMENT_NAME=deepfactcite-sft-qwen3-8b-lora-mix-clean-200 \
bash scripts/deepfactcite/train_sft_qwen3.sh
```

### 7.4 合并 LoRA

```bash
python scripts/deepfactcite/merge_lora_adapter.py \
  --base-model /root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base \
  --adapter outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200/global_step_200 \
  --output-dir outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16 \
  --torch-dtype float32 \
  --save-dtype bfloat16
```

### 7.5 运行 GRPO dry-run

```bash
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

默认 `DRY_RUN=1`，只打印命令，不真正占用 GPU 训练。

### 7.6 真正运行 4 GPU 保存训练

```bash
DRY_RUN=0 \
GRPO_TOTAL_STEPS=32 \
GRPO_SAVE_FREQ=32 \
GRPO_N=4 \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32 \
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

运行前应确认：

- 4 张 GPU 可用。
- `/root/autodl-tmp` 有足够空间。
- `ACTOR_MODEL_PATH` 指向可用 merged SFT actor。
- `DATA_DIR` 指向 v3 prompt-fix one-citation 数据。
- `GRPO_REQUIRE_FINAL_CHECKPOINT=1` 时，`GRPO_SAVE_FREQ` 必须能覆盖最终 step。

### 7.7 汇总 rollout

```bash
python scripts/deepfactcite/summarize_grpo_rollouts.py \
  logs/grpo/rollouts/<EXPERIMENT_NAME> \
  --out reports/<EXPERIMENT_NAME>_rollout_summary.md \
  --recompute-details
```

## 8. 常见坑与项目纪律

### 8.1 不要让模型生成工具返回

错误方式：

```text
模型一次性输出 <search>、<information>、<answer>
```

正确方式：

```text
模型输出 <search>
环境注入 <information>
模型再输出 <answer>
```

### 8.2 不要把 URL validity 当成最终引用质量

真实 URL 只是第一步。真正要看的是：

```text
这个 URL 对应的 snippet 是否支持 citation 附近的具体 claim。
```

### 8.3 严格过滤不一定带来更好模型

`deepfactcite_strict` 数据更干净，但早期 Strict SFT 没有自动击败 Soft SFT。原因可能是过滤降低了多样性。项目因此保留了 MixClean200 这种兼顾干净度和行为覆盖的 baseline。

### 8.4 bf16 merge 不是天然等价

旧实验发现：

```text
PEFT dynamic LoRA forward 与 bf16 merge 后模型可能出现明显 logits diff。
```

因此 merged parent 不能只看“能加载”，还要验证等价性和下游评测。

### 8.5 no-save run 不能作为最终模型结果

no-save run 只能说明训练信号是否值得继续。它不能回答：

```text
训练后的 checkpoint 在真实评测集上是否更好。
```

所以报告里区分：

- rollout 诊断信号。
- 保存 checkpoint。
- checkpoint 评测结果。

### 8.6 失败也要记录

本仓库已经有多份失败台账。它们的价值在于说明：

- 哪些假说被证伪。
- 哪些运行只是工程 smoke。
- 哪些结果不能继续作为 parent。
- 哪些路径可以安全复现。

这比只保留成功命令更重要。

## 9. 目录导览

### 9.1 核心代码

```text
deepfactcite/
  prompts.py
  reward.py
  retriever_server.py

scripts/deepfactcite/
  prepare_data.py
  build_sft_mix.py
  prepare_sglang_grpo_claim_filtered.py
  train_sft_qwen3.sh
  merge_lora_adapter.py
  eval_citation_agent_vllm.py
  eval_searchr1_agent_vllm.py
  run_sglang_grpo_ablation_2gpu.sh
  run_sglang_grpo_v3_promptfix_4gpu_saved.sh
  verl_deepfactcite_reward.py
  summarize_grpo_rollouts.py

verl/
  trainer/main_ppo_deepfactcite.py
  utils/reward_score/deepfactcite.py
```

### 9.2 数据目录

```text
data/deepfactcite/
data/deepfactcite_strict/
data/deepfactcite_mix/
data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite/
data/shortqa_guardrail/
data/searchr1_core_guardrail/
data/wiki-18-bm25-index/
data/wiki-18-corpus/
```

### 9.3 实验产物

```text
outputs/deepfactcite/
outputs/deepfactcite/grpo/
logs/
logs/grpo/rollouts/
reports/
tensorboard_log/DeepFactCite-GRPO/
```

### 9.4 推荐阅读顺序

新接手项目的人建议按这个顺序读：

```text
docs/project_report.md
docs/deepfactcite_learning_index_20260518.md
docs/deepfactcite_experiment_record_20260518.md
docs/deepfactcite_reproducibility_issue_log.md
reports/deepfactcite_mixclean200_eval_summary.md
reports/deepfactcite_v2_onecite_grpo_ablation_2gpu_20260518.md
reports/dfc_mixclean200_v3_promptfix_onecite_2gpu_rollout_summary.md
docs/deepfactcite_grpo_processes_interview_20260519.md
```

## 10. 关键术语

| 术语 | 简单解释 |
|---|---|
| Search-R1 | 让模型学会边推理边调用搜索工具的 RL 框架 |
| DeepFactCite | 本项目中的长答案引用可信任务方向 |
| SFT | 监督微调，用已有轨迹教模型基本格式和行为 |
| GRPO | 不训练 critic 的 RL 方法，用同题多答案的组内相对分数做 advantage |
| rollout | 一条完整模型轨迹，包括搜索、工具返回和最终答案 |
| citation | 答案里的引用，主要是 `[短说明](URL)` |
| URL validity | URL 是否来自当前检索结果 |
| claim support | citation 附近的声明是否被所引片段支持 |
| unsupported citation | URL 可能是真的，但它不支持当前 claim |
| guardrail eval | 防护评测，确保引用训练没有破坏答案/搜索基础能力 |
| outcome-only | 主要看答案或结果的 reward 设置 |
| citation-aware | 显式奖励 URL、citation、support 的 reward 设置 |

## 11. 当前状态与下一步

当前可以比较有把握地说：

```text
本项目已经把 Search-R1 风格搜索智能体训练扩展到了 DeepFactCite 引用可信场景，
并在小规模受控 GRPO 实验中证明 citation-aware reward 比 outcome-only 更能改善引用行为。
```

但还不能说：

```text
已经得到最终可信引用模型。
```

下一步优先级：

1. 保留 saved32 作为“保存、转换、评测链路已打通”的工程里程碑。
2. 下一次 GRPO 先做 16/32 step checkpoint gate：Search-R1 BM25 200 不能明显低于 MixClean200，ShortQA32 不能退，DeepFactCite Strict47 的 `claim_support` 要实质上升。
3. 继续使用本地同环境 benchmark 作为主要比较口径，不把不同搜索库、不同基模下的数字直接和 Search-R1 论文表格做胜负比较。
4. 对比 SFT MixClean200、outcome-only GRPO、citation-aware GRPO。
5. 抽样分析好样例和失败样例，尤其关注 unsupported citation。
6. 如果 checkpoint 评测稳定，再扩大数据规模和训练步数。

后续可能改进：

- 引入更强的 claim-level judge。
- 扩大 one-citation 数据到多 citation、多句答案。
- 在保持 URL 可验证和实验变量可控的前提下，逐步升级检索器；不盲目复刻重型 Search-R1 搜索库。
- 增加 citation presence 的正向奖励，而不只是 hard cap。
- 对中文问题和跨语言 evidence 做专门评测。
- 将 runtime citation verifier 与训练 reward 统一。

## 12. 参考入口

- Search-R1 原始 README：`docs/english_originals/README.md`
- veRL 原始说明：`VERL_README.md`
- 本项目学习索引：`docs/deepfactcite_learning_index_20260518.md`
- 实验主记录：`docs/deepfactcite_experiment_record_20260518.md`
- 复现问题日志：`docs/deepfactcite_reproducibility_issue_log.md`
- 4 GPU 保存计划：`docs/deepfactcite_4gpu_saved_run_plan_20260518.md`
- 训练进程解释：`docs/deepfactcite_grpo_processes_interview_20260519.md`
- 参考写法来源：`https://github.com/HanyunReady/searchQA/blob/main/docs/project_report.md`
