# Search-R1-DeepFactCite

面向新手的中文项目文档。原始 Search-R1 英文 README 已归档在
`docs/english_originals/README.md`。

本项目基于 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) 和
[veRL](https://github.com/volcengine/verl)，目标是训练一个会搜索、会读证据、
会给可信引用的开放域问答智能体。

一句话概括：

```text
Search-R1 教模型学会“边想边搜索”；
本项目在这个基础上继续教模型“引用必须真实，而且要真的支持答案里的那句话”。
```

## 先看结论

Search-R1 解决的是：

```text
问题来了
-> 模型自己决定搜什么
-> 检索器返回资料
-> 模型继续推理或继续搜索
-> 最后回答
-> 用答案是否正确等规则奖励做强化学习
```

它把大模型从“被动读 RAG 塞进来的文档”变成“主动调用搜索工具的智能体”。

本项目 Search-R1-DeepFactCite 继续解决的是：

```text
答案看起来对，不代表引用可信。
URL 真实，也不代表它支撑了 URL 前面的那句话。
```

所以本项目把训练目标从“答对问题”扩展成：

```text
答对问题 + 搜到证据 + URL 来自当前搜索结果 + citation 附近的 claim 被证据支持
```

当前项目定位要讲清楚：

- 不能简单宣称“击败 Search-R1”。
- 更准确的说法是：本项目在 Search-R1 风格搜索智能体上增加了引用真实性和声明级支持训练。
- 原始 Search-R1 的主线结果主要围绕 `Qwen2.5-3B/7B` 和 `Llama3.2-3B`；本项目主线是 `Qwen3-8B-Base`，另保留 `Qwen3-4B-Base` 作为快速实验目标。
- 原始 Search-R1 的典型搜索库是 2018 Wikipedia / `wiki-18`：本地 `wiki_dump.jsonl` 约 `21,015,324` 条、`19G`，BM25 索引约 `2.2G`，也可接 E5 向量索引。它适合开放域短问答，但不天然保存“哪个 URL 支撑答案里的哪句话”。
- 本项目主线搜索库来自 DeepCiteFact/DeepFactCite：`data/deepfactcite/corpus.jsonl` 有 `6,118` 条、`19M`，strict 版有 `4,766` 条、`16M`，v3 one-citation GRPO 受控集只有 `32` 条。它小得多，但每条都保留 URL 和 citation evidence，更适合训练引用可信。
- 因为搜索库、检索器和基座模型都不同，本文档里的结果按“本地同环境 benchmark”理解，不和 Search-R1 论文表格做直接因果比较。
- 没有必要为了本项目强行照搬 Search-R1 的重型 E5 / Wikipedia / ANN / BM25 工程栈；复现成本高、变量多，换到 Qwen3 和引用任务后不一定提升 claim support。
- 已跑通 Qwen3-8B SFT、Search-R1 风格 BM25 防护评测、DeepFactCite 引用评测、SGLang/veRL GRPO 路径。
- 在当前本地 benchmark 上，MixClean200 SFT 已达到有竞争力的准确率：Search-R1 BM25 200 的 `answer_subem=0.490`、`search_success=1.000`，ShortQA32 的 `answer_subem=0.500`。（`answer_subem=0.490` 表示 200 道 NQ/HotpotQA 风格题里约 49% 的回答包含标准答案；`search_success=1.000` 表示全部样本都成功发起搜索；ShortQA32 的 `answer_subem=0.500` 表示 32 道短问答约一半命中。）
- 和 Search-R1 论文里最相近的 NQ/HotpotQA 场景相比，本项目结果在同一量级：Search-R1 Qwen2.5-7B-base 报告 NQ `EM=0.480`、HotpotQA `EM=0.433`；本项目 Qwen3-8B MixClean200 在本地 BM25 200 上是 NQ `subEM=0.500`、HotpotQA `subEM=0.480`。注意这里一个是论文 E5 检索 + EM，一个是本地 BM25 + subEM，只能说明“没有明显低一个量级”，不能直接宣称胜负。
- 粗略看，差异会双向影响数字：E5 dense retrieval 通常更擅长语义匹配，可能比 BM25 更容易找回改写后的相关段落；但 `subEM` 比严格 `EM` 宽松，可能把“答案出现在长句里”也算命中；另外 Qwen3-8B 比 Qwen2.5-7B 更新且更大，本地 200 题子集也比论文完整 benchmark 更小。因此这些因素有的压低、有的抬高分数，最稳妥的结论就是“量级接近，可以作为答案/搜索能力没有崩掉的证据”。
- 4 GPU saved GRPO 的工程链路已经成立，但当前 saved32 checkpoint 还不能作为最终效果提升结论。

## Search-R1 到底干了什么

很多人第一次看 Search-R1 会把它理解成“RAG 加强化学习”。这个理解不够准确。

普通 RAG 通常是系统先搜好文档，再把文档塞进 prompt：

```text
用户问题
-> 系统检索 top-k 文档
-> LLM 根据这些文档回答
```

Search-R1 更像是在训练一个会用搜索工具的学生：

```text
用户问题
-> 模型先想一想
-> 模型自己写搜索 query
-> 搜索工具返回 information
-> 模型阅读 information
-> 模型决定继续搜索还是回答
-> reward 根据最终答案和轨迹质量打分
-> RL 鼓励更好的搜索和回答策略
```

Search-R1 的核心贡献可以拆成 5 点：

| 点 | 小白版解释 | 技术含义 |
|---|---|---|
| 1. 主动搜索 | 模型不是等别人喂资料，而是自己决定搜什么 | search action 由 policy 生成 |
| 2. 多轮交互 | 一次搜不够，可以继续搜 | interleaved reasoning and search |
| 3. 工具协议 | 用标签区分思考、搜索、工具返回和答案 | `<think>`、`<search>`、`<information>`、`<answer>` |
| 4. 强化学习 | 不直接标注每一步怎么搜，而是用结果奖励训练 | PPO / GRPO / reinforce 等 |
| 5. 可替换检索器 | 可以接本地稀疏/稠密检索，也可以接在线搜索 | `/retrieve` API |

Search-R1 论文中强调：只是在推理时 prompt 模型“你可以用搜索”并不够，因为模型并没有真正学会怎样和搜索引擎交互。Search-R1 用 RL 让模型在逐步推理时学会自己生成一个或多个搜索查询，并利用实时检索结果完成问答。

### 为什么本项目没有照搬 Search-R1 的重型搜索库

原始 Search-R1 常见设置依赖较完整的开放域检索栈。以本仓库已经准备过的 `wiki-18` BM25 防护评测为例，`data/wiki-18-corpus/wiki_dump.jsonl` 约 `21,015,324` 条、`19G`，`data/wiki-18-bm25-index` 约 `2.2G`；如果走原论文常用的 E5 dense retrieval，还要额外下载和加载向量索引。这个路线适合复现 NQ/HotpotQA 这类开放域短问答，但对 DeepFactCite 引用任务不一定是最优投入。

差异可以这样看：

| 对比项 | 原始 Search-R1 常见设置 | 本项目主线设置 |
|---|---|---|
| 基座模型 | `Qwen2.5-3B/7B`、`Llama3.2-3B` 等 | `Qwen3-8B-Base` 为主，`Qwen3-4B-Base` 用于快速实验 |
| 训练问题 | NQ/HotpotQA 风格短事实问答 | DeepFactCite 长答案、引用和 claim support |
| 搜索库 | 2018 Wikipedia / `wiki-18`，本地约 `21.0M` 段、`19G`，另有 BM25/E5 索引 | DeepFactCite citation corpus：基础版 `6,118` 条，strict 版 `4,766` 条，v3 GRPO 受控集 `32` 条 |
| 检索目标 | 找到能回答问题的段落 | 找到能支撑某个具体 claim 的 URL 片段 |
| 复现成本 | 需要大语料、大索引、检索服务和版本 pin | 轻量词汇检索即可复现实验主线，重点检查 URL 和 claim |

因此本项目采用更可控的策略：

```text
用轻量、可审计、带 URL 的检索库验证引用训练；
再用 Search-R1 BM25 200 / ShortQA32 / DeepFactCite strict47 做同环境 benchmark。
```

这不是降低标准，而是把实验变量收窄。只要同一个检索环境、同一套 prompt、同一套评测脚本下，模型能达到有竞争力的答案准确率，同时改善引用真实性，这个结论就更干净。

参考资料：

- Search-R1 论文：https://arxiv.org/abs/2503.09516
- Search-R1 代码：https://github.com/PeterGriffinJin/Search-R1

## 本项目相比 Search-R1 改进了什么

Search-R1 主要优化答案正确性和搜索行为。本项目继续问一个更细的问题：

```text
模型给出的引用，真的可信吗？
```

举个简单例子。

检索结果只说：

```text
某洞穴有旧石器时代壁画。
URL: https://example.org/cave
```

模型回答：

```text
这个洞穴是欧洲最早发现的、保存最完整的史前艺术遗址之一 [source](https://example.org/cave)。
```

这个 URL 可能是真的，但检索片段并没有支持“最早发现”“保存最完整”这些更强的说法。普通答案奖励可能看不出问题，本项目的 citation-aware reward 会把它当成低质量引用。

### 改进 1：从“会搜索”升级到“会可信引用”

Search-R1 关注：

```text
模型有没有学会搜索？
答案有没有命中 gold answer？
```

本项目额外关注：

```text
答案里的 markdown citation 是否存在？
URL 是否来自当前检索轨迹？
citation 前后的 claim 是否被对应 snippet 支持？
有没有裸 [1]、裸 [S_xxx] 或模型编造的 raw URL？
```

### 改进 2：新增 DeepFactCite reward

核心代码在 `deepfactcite/reward.py`。

reward 不只看答案，还看这些指标：

| 指标 | 检查什么 | 为什么重要 |
|---|---|---|
| `answer_subem` | 答案是否覆盖目标答案 | 防止只学引用格式却不回答问题 |
| `search` | 是否先搜索、搜索是否超预算 | 保留 Search-R1 搜索能力 |
| `format` | 标签结构是否正确 | 训练轨迹要能解析 |
| `url_validity` | URL 是否来自工具返回 | 防止编造链接 |
| `citation_precision` | 引用附近 claim 是否被支持 | 防止真实链接乱贴 |
| `claim_support` | claim 和证据片段的支持度 | 衡量引用忠实性 |
| `unsupported_citation_rate` | 不受支持的引用比例 | 本项目最想降低的错误 |
| `cost` | 输出长度和搜索成本 | 防止无限长回答或乱搜 |

直觉上，reward 在告诉模型：

```text
只答对不够。
只贴链接也不够。
必须把正确链接放在它真正能支持的那句话旁边。
```

### 改进 3：新增 DeepFactCite 数据构建

核心脚本：

```text
scripts/deepfactcite/prepare_data.py
scripts/deepfactcite/build_sft_mix.py
scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py
```

数据处理做了几件事：

- 从 DeepCiteFact 轨迹构建 Search-R1 风格 SFT/RL parquet。
- 把原始轨迹里的 `<google_search>`、`<tool_response>` 对齐到 `<search>`、`<information>`。
- 过滤掉 URL 不真实、claim support 太弱、unsupported citation 太多的 SFT 样本。
- 构造 one-citation GRPO 小数据，用来干净地验证 citation-aware reward 是否有效。

这里的 one-citation 数据不是最终大规模训练集，而是一个受控实验：

```text
如果每个样本只要求 1 条窄 claim 和 1 个真实 URL，
模型能不能学会把 citation 放对？
```

### 改进 4：检索结果必须带 URL

Search-R1 原始任务里，检索片段主要用于回答问题。本项目要训练引用，因此检索返回必须保留 URL。

相关代码：

```text
deepfactcite/retriever_server.py
scripts/deepfactcite/start_lexical_retriever.sh
scripts/deepfactcite/serve_searchr1_bm25.py
```

边界非常重要：

```text
模型只能生成 <search> 和 <answer>。
<information> 必须由检索环境注入。
模型不能自己编造工具返回。
```

如果模型能自己写 `<information>`，它就能伪造网页标题、URL 和摘要，引用训练会失去意义。

### 改进 5：建立 answer/search 防护评测

训练引用能力时，很容易把原本的问答能力训坏。所以本项目不只看引用指标，还保留 Search-R1 风格防护评测。

主要评测有三类：

| 评测 | 用途 |
|---|---|
| `Search-R1 BM25 200` | 检查 NQ/HotpotQA 风格答案和搜索能力是否退化 |
| `ShortQA32` | 检查短答案、搜索和引用基础行为 |
| `DeepFactCite strict47` | 检查长答案引用真实性、URL validity 和 claim support |

当前 MixClean200 SFT baseline 在本地 BM25 防护评测上比 Base 更稳定：

| 模型 | Answer subEM | Search Success | Search Turns |
|---|---:|---:|---:|
| Base Qwen3-8B | 0.290 | 0.960 | 1.720 |
| Soft SFT 100 | 0.450 | 0.990 | 1.195 |
| MixClean SFT 200 | 0.490 | 1.000 | 1.160 |

这说明 SFT 没有只学会“贴链接”，也保住了 Search-R1 风格的搜索问答行为。

这里的数字不是原始 Search-R1 论文设置的直接复现，因为搜索库、检索器和基模都不同。它的意义是本地同环境对比：在相同 benchmark 和检索服务下，本项目模型已经有竞争力的答案准确率，后续 citation-aware 训练不能以牺牲这条基线为代价。

### 改进 6：做 outcome-only vs citation-aware 消融

为了证明“引用奖励真的有用”，本项目做了对照实验。

同一份 v2 one-citation 数据上：

| Metric | Outcome-Only | Citation-Aware |
|---|---:|---:|
| URL validity | 0.1562 | 0.7188 |
| citation precision | 0.1125 | 0.3937 |
| claim support | 0.1094 | 0.3906 |
| unsupported citation rate | 0.8438 | 0.3750 |
| citation count | 0.1562 | 0.7188 |

结论很清楚：

```text
只看最终答案的 reward 不足以学好引用；
显式 citation/support reward 能明显改善引用行为。
```

v3 prompt-fix 进一步把 markdown citation 要求写清楚，在小规模 no-save 诊断中把 URL validity、claim support 提高到更可用的水平。但 saved checkpoint 的最终有效性仍要以 held-out eval 为准。

## 新手怎么理解整个流程

可以把训练模型想象成教学生开卷考试。

Search-R1 教的是：

```text
不会就去查资料；
查资料时自己想关键词；
查完再回答。
```

本项目继续教的是：

```text
你引用的资料必须真的来自这次查到的资料；
你引用的资料必须支撑旁边那句话；
证据不够就少说，不能硬编。
```

一条训练轨迹长这样：

```text
User: 问题

Assistant:
<think>我需要查证这个事实。</think>
<search>搜索关键词</search>

Environment:
<information>
<snippet id=S_1>
Title: ...
URL: https://...
Text: ...
</snippet>
</information>

Assistant:
<answer>被证据支持的回答 [source](https://...).</answer>
```

reward 会检查：

```text
有没有搜索？
有没有最终答案？
URL 是不是工具返回过的？
citation 前面的 claim 是否被这个 URL 的片段支持？
格式是不是可解析？
回答是不是过长或乱搜？
```

## 目录怎么读

建议按这个顺序读：

1. `README.md`：先理解项目目标和 Search-R1 对比。
2. `docs/project_report.md`：看完整项目报告、实验结论和复现路径。
3. `docs/deepfactcite_grpo.md`：看 DeepFactCite-GRPO 的训练入口。
4. `docs/deepfactcite_experiment_record_20260518.md`：看实验过程和失败记录。
5. `reports/searchr1_core_bm25_eval_summary.md`：看答案/搜索防护评测。
6. `docs/deepfactcite_reproducibility_issue_log.md`：看为什么不能只保留成功实验。

核心代码：

```text
deepfactcite/
  prompts.py              # 搜索引用 prompt
  reward.py               # citation-aware reward
  retriever_server.py     # 轻量词汇检索器

scripts/deepfactcite/
  prepare_data.py                         # DeepCiteFact -> SFT/RL 数据
  build_sft_mix.py                        # 构建 MixClean SFT
  prepare_sglang_grpo_claim_filtered.py   # one-citation GRPO 数据
  train_sft_qwen3.sh                      # Qwen3 LoRA SFT
  merge_lora_adapter.py                   # LoRA 合并
  eval_citation_agent_vllm.py             # 引用评测
  eval_searchr1_agent_vllm.py             # Search-R1 风格评测
  run_sglang_grpo_v3_promptfix_4gpu_saved.sh
  summarize_grpo_rollouts.py
```

## 最小复现路径

准备 DeepFactCite 数据：

```bash
python scripts/deepfactcite/prepare_data.py \
  --deepcitefact-dir /root/autodl-tmp/DeepCiteFact \
  --output-dir data/deepfactcite
```

启动轻量检索器：

```bash
CORPUS=data/deepfactcite/corpus.jsonl PORT=8000 \
  bash scripts/deepfactcite/start_lexical_retriever.sh
```

训练 Qwen3-8B LoRA SFT：

```bash
MODEL_SIZE=8B \
N_GPUS=2 \
TOTAL_STEPS=200 \
DATA_DIR=data/deepfactcite_mix/sft \
EXPERIMENT_NAME=deepfactcite-sft-qwen3-8b-lora-mix-clean-200 \
  bash scripts/deepfactcite/train_sft_qwen3.sh
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

先 dry-run 4 GPU GRPO 命令：

```bash
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

真正训练时显式关闭 dry-run：

```bash
DRY_RUN=0 \
GRPO_TOTAL_STEPS=32 \
GRPO_SAVE_FREQ=32 \
GRPO_N=4 \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32 \
  bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

## 关键术语

| 术语 | 小白解释 |
|---|---|
| RAG | 先检索资料，再让模型根据资料回答 |
| Search Agent | 模型自己决定什么时候搜索、搜什么、什么时候停止 |
| SFT | 监督微调，先让模型模仿正确格式和轨迹 |
| GRPO | 强化学习方法，同一个问题生成多条答案，奖励更好的那条 |
| rollout | 一条完整交互轨迹，包括搜索、工具返回和答案 |
| citation | 答案里的引用，比如 `[source](https://...)` |
| URL validity | URL 是否来自当前检索结果 |
| claim support | 引用旁边那句话是否被所引片段支持 |
| unsupported citation | 链接是真的，但不能支撑当前说法 |
| guardrail eval | 防护评测，确认新能力没有把旧能力训坏 |

## 当前限制

- 当前项目不是产品级 Deep Research 系统，工具主要围绕搜索。
- claim support 主要基于 snippet、词汇支持度和可选 judge，不等于人工事实核查。
- v3 小规模 GRPO 有正向信号，但 saved32 checkpoint 在 held-out eval 上还没有证明稳定超过 MixClean200 SFT。
- 对外表述时应强调“扩展 Search-R1 的引用可信训练”，不要夸大为“全面击败 Search-R1”。
