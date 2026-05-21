# DeepFactCite 面试问题逐题答复

日期：2026-05-19

本文只回答前面问题清单中的题目，不展开后续“草稿思路”里的补充模块。回答口径基于本仓库当前文档、脚本和已完成实验记录。需要注意：本项目不是直接宣称击败原始 Search-R1 论文设置，而是在 Search-R1/veRL 风格搜索智能体上，受控验证 citation-aware reward 是否能改善 URL 真实性和 claim support。

## 0. 实际运行口径

面试中先把口径讲清楚，后面的技术细节才站得住：

- 当前 DeepFactCite 主线是 `Qwen3-8B-Base -> MixClean200 SFT -> citation-aware GRPO`，不是官方 Search-R1 checkpoint 的直接续训主线。
- Qwen3 GRPO 实际使用的是 `/root/autodl-tmp/SearchShortQA/verl` 里的新版 SGLang multi-turn veRL 路径，原因是当前仓库原始 Search-R1 本地 `verl/` 和 vLLM <= 0.6.3 对 Qwen3 rollout 不天然兼容。
- 之前老 Search-R1/legacy 路径更接近 `Qwen2.5/Llama + vLLM + wiki-18/E5/BM25`，环境里会涉及 `flash-attn`、`vLLM 0.6.3`、`torch 2.4.0/cu121`；当前 SGLang Qwen3 GRPO 主线则默认走 `sdpa/triton`，不要把两条栈混成同一个实验。
- 当前最完整的 saved GRPO 运行是 `4GPU / GRPO_N=4 / 32 steps / model-only checkpoint`，checkpoint 在 `outputs/deepfactcite/grpo/dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2/global_step_32/actor`，但它的 rollout 指标不能直接当 held-out 泛化指标。

## 1. 项目概览与方法设计

### 1. DeepFactCite 相比原始 Search-R1 增加了哪些能力？为什么必要？

DeepFactCite 在 Search-R1 的“搜索增强问答”基础上增加了可信引用能力。原始 Search-R1 更关注模型是否会搜索、是否能通过搜索提高最终答案命中率；DeepFactCite 进一步要求最终答案中的 markdown citation 必须来自当前检索轨迹，并且 citation 附近的局部声明要被对应证据片段支持。

面试时可以这样说：Search-R1 解决的是“会不会搜、答案对不对”，但长答案场景还会出现“答案看起来对、引用却不支持”的问题。DeepFactCite 的新增点是把 `URL validity`、`citation precision`、`claim support`、`unsupported citation rate` 纳入 reward 和评测，这样模型不能只靠编一个看起来合理的链接拿高分。

这里 `unsupported citation rate` 可以先用中文理解成“无效支撑引用比例”：模型确实写了引用，但这个引用对应的证据不能证明它前面的那句话，或者 URL 根本不是本次检索返回的。它越低越好。比如 10 个引用里有 4 个不能支撑对应句子，unsupported rate 大约就是 0.4。

它代表的不是“答案对不对”的能力，而是“模型会不会负责任地引用证据”的能力。低 unsupported rate 说明模型更会把引用放在真正能被证据支持的句子后面；高 unsupported rate 说明模型可能会贴链接、会装出有来源，但来源和它说的话对不上。这是 citation faithfulness 或 attribution faithfulness 的核心指标。

### 2. Search-R1 与 DeepFactCite 的基座模型有什么区别？为什么选 Qwen3-8B？

原始 Search-R1 常见基座是 `Qwen2.5-3B/7B`、`Llama3.2-3B` 等，配合 wiki-18/E5/vLLM 搜索 rollout。DeepFactCite 主线使用 `Qwen3-8B-Base`，`Qwen3-4B-Base` 只作为快速 MVP 目标。

选择 Qwen3-8B 的原因是：8B 在答案组织、工具调用和长文本引用格式上明显比 4B 更适合展示；同时 8B 又还能在 2 到 4 张 A800 80G 上完成 LoRA SFT 和小中规模 GRPO。当前资源规划是：Qwen3-8B LoRA SFT 用 2 张 A800，GRPO 稳定目标是 4 张 A800。

### 3. 为什么没有复刻 Search-R1 的 E5/Wiki-18 大检索栈，而使用轻量 citation corpus？

因为本项目的核心变量不是“大检索系统是否复刻成功”，而是“在可控证据下，citation-aware reward 是否改善引用行为”。wiki-18 语料约 `21,015,324` 条、`19G`，BM25 索引约 `2.2G`，如果再引入 E5/ANN/flat index，会把模型、检索器、索引质量、吞吐和 citation reward 多个变量混在一起。

DeepFactCite citation corpus 的优势是每条证据都有可审计 URL，reward 可以检查模型引用的 URL 是否来自当前检索结果。当前常用规模包括：基础 corpus `6,118` 条，strict corpus `4,766` 条，v3 one-citation GRPO 受控 corpus `32` 条。面试里要强调这是受控实验选择，不是说大检索栈不重要。

### 4. 从 outcome-only 到 citation-aware reward 的直观理解是什么？

`outcome-only` 主要奖励答案、格式和搜索行为，不显式奖励引用是否真实、是否支持声明。它容易鼓励“答对即可”的策略，模型可能输出真实或看似真实的 URL，但局部声明并没有被证据支持。

`citation-aware` 把 reward 拆成答案、引用 URL、claim support、格式、搜索和成本。直观说，模型不仅要答出内容，还要“拿得出证据”，并且证据要支撑它刚说的那句话。当前 citation-aware 默认权重是 `answer=0.15`、`citation=0.35`、`support=0.30`、`format=0.10`、`search=0.05`、`cost=0.05`。

`Claim support` 可以用小白语言解释成：引用不是装饰品，而是要能证明它前面那句话。比如检索片段只说“Chauvet-Pont d'Arc Cave 位于法国南部，并保存了早期具象洞穴绘画”，那么答案写“Chauvet-Pont d'Arc Cave is in southern France and contains early figurative cave paintings [source](URL)”就是被支持的；但如果答案写“这个洞穴改变了 1990 年代欧洲考古政策 [source](同一个 URL)”，URL 可能是真的，但这句话不是片段能证明的，所以 claim support 应该低。这个例子能说明 DeepFactCite 关心的不是“有没有链接”，而是“链接能不能证明这句话”。

如果面试官追问 `answer=0.15` 会不会太低，稳妥回答是：它不是理论最优，而是当前消融的工程选择。MixClean200 SFT 已经先保证了 answer/search baseline，GRPO 这一阶段主要想修复“答案会写、但引用不可信”的问题，所以把主要优化压力放到 citation/support 上。风险是 answer 太低可能让模型变得过短、过保守；因此是否调高不能凭直觉，要做小规模权重消融，例如比较 `answer=0.15/0.25/0.35`，同时看 `answer_subEM`、search、URL validity、claim support、unsupported rate 和 no-citation，而不是只看 raw reward。这里的 unsupported rate 就是“引用不支持对应句子”的失败比例，调高 answer 后如果它变高，说明模型可能又开始写更宽但证据撑不住的答案。

### 5. URL validity 和 claim support 如何定义？训练中怎么用？

`URL validity` 指最终答案里的 markdown citation URL 是否出现在本次 rollout 的检索证据中。它防止模型凭空编链接，或者引用没有检索到的来源。这里的“本次 rollout 的检索证据”不是另一个隐藏数据库，而是模型调用 search tool 后，环境返回给模型的 tool response。

具体流程是：检索器返回若干 `SearchSnippet`，每条有 `sid/title/url/text`；工具层把它格式化进 `<tool_response>...</tool_response>` 或 `<information>...</information>`，每个 snippet 里都有 `URL:` 和 `Text:`。所以 URL 一方面会塞进 tool response，让模型看得见、可以复制；另一方面 reward 会从完整 trajectory 里重新解析这些 `URL:` 和 `Text:`，维护一个“本次 rollout 允许引用的 URL -> 证据文本”映射。最终答案里的 markdown URL 只有出现在这个映射里，才算 URL valid。

一个 URL validity 的例子：

```text
tool response:
<snippet id="S_1">
Title: Chauvet Cave
URL: https://example.org/chauvet
Text: Chauvet-Pont d'Arc Cave is in southern France and contains early figurative cave paintings.
</snippet>

答案 A：
Chauvet-Pont d'Arc Cave is in southern France [source](https://example.org/chauvet).

答案 B：
Chauvet-Pont d'Arc Cave is in southern France [source](https://fake.example/chauvet).
```

答案 A 的 URL valid，因为这个 URL 出现在本次 tool response 里。答案 B 的 URL invalid，因为这个 URL 是模型自己编的，或者至少不是这次检索返回的。

`Claim support` 指 citation 附近的局部声明是否能由被引用 URL 对应的证据片段明确支持。它比 URL validity 更严格，因为“链接是真的”不等于“链接能证明这句话”。

还是用同一个 tool response：

```text
tool response:
URL: https://example.org/chauvet
Text: Chauvet-Pont d'Arc Cave is in southern France and contains early figurative cave paintings.

答案 A：
Chauvet-Pont d'Arc Cave is in southern France and contains early figurative cave paintings
[source](https://example.org/chauvet).

答案 C：
Chauvet-Pont d'Arc Cave completely changed European archaeology policy in the 1990s
[source](https://example.org/chauvet).
```

答案 A 的 URL validity 高，claim support 也高，因为证据文本直接支持这句话。答案 C 的 URL validity 也高，因为 URL 确实来自 tool response；但 claim support 低，因为证据只说它的位置和洞穴绘画，没有说它改变了 1990 年代欧洲考古政策。面试时可以把这句话说得更白：URL validity 问“这个链接是不是模型刚搜到的？”，claim support 问“这个链接里的文字能不能证明模型这句话？”

训练时，自定义 reward 会解析完整 agent 轨迹，抽取检索返回的 URL/Text 和最终答案引用，计算 `url_validity`、`citation_precision`、`claim_support`、`unsupported_citation_rate` 等指标，并把最终标量 reward 放到最后一个有效 response token 上供 GRPO 更新。

`claim_support` 的默认计算不是直接问一个大模型，而是规则化的词汇支持分，核心逻辑在 `deepfactcite/reward.py`：

1. 先从最终 `<answer>` 中解析 markdown citation，例如 `[source](https://example.org/chauvet)`。
2. 对每个 citation，取它附近的一小段句子作为 `claim`。简单理解就是“这个链接前后那句需要被证明的话”，不是整篇答案。
3. 用 citation URL 去本次 tool response 解析出的 URL/Text 映射里找证据文本。URL 找不到时，这条 citation 直接算 unsupported。
4. 对 URL 找到的 citation，把 claim 和证据文本都 tokenize，去掉停用词，计算 claim token 有多少比例出现在证据文本里。
5. 重叠比例 `>=0.65` 记 `1.0`，表示强支持；`>=0.25` 记 `0.5`，表示部分支持；低于 `0.25` 记 `0.0`。
6. `claim_support` 是所有 citation 的 support score 之和除以总 citation 数。注意分母包含 invalid URL citation，所以编链接会同时拉低 URL validity 和 claim support。
7. 如果某条 citation URL 无效，或者 support score `<0.5`，会计入 `unsupported_citation_rate`。

用上面的例子，答案 A 的 claim tokens 主要是 `Chauvet-Pont d'Arc Cave / southern France / early figurative cave paintings`，这些都能在证据文本里找到，所以 support score 接近 `1.0`。答案 C 的关键 claim 是 `changed European archaeology policy in the 1990s`，证据里没有 `policy/1990s/changed` 这些支撑信息，所以 support score 会很低。

也可以启用 `DEEPFACTCITE_JUDGE_BASE_URL` 和 `DEEPFACTCITE_JUDGE_MODEL` 接一个 OpenAI-compatible judge，但当前主线默认使用规则分数，优点是便宜、可复现；缺点是它只是词汇/近似支持，不等于真正完整的语义蕴含判断。

还要承认一个边界：当前 DeepFactCite 更强的是局部可信引用，即“URL 不假、局部 claim 被证据支持”。更高层的“引用是否最相关、是否覆盖答案核心论点、是否真正回答用户问题”更难评估。它涉及 evidence-query relevance、关键 claim coverage、多跳推理和引用遗漏问题，很难完全用一个规则 reward 精确量化。后续更适合先作为 held-out eval、LLM/human judge 或 reranker 目标，再谨慎纳入训练。

## 2. 数据集与 SFT 基线

### 1. DeepFactCite SFT/RL 数据集包括哪些子集？规模是多少？

基础 DeepFactCite 数据来自 DeepCiteFact 轨迹转换：

| 数据 | 规模 | 用途 |
|---|---:|---|
| `data/deepfactcite/sft/train.parquet` | 1,140 | 基础 SFT 训练 |
| `data/deepfactcite/sft/test.parquet` | 60 | 基础 SFT 测试 |
| `data/deepfactcite/rl/train.parquet` | 1,140 | 基础 RL 训练 |
| `data/deepfactcite/rl/test.parquet` | 60 | 基础 RL 测试 |
| `data/deepfactcite/corpus.jsonl` | 6,118 | 基础引用语料 |
| `data/deepfactcite_strict/sft/train.parquet` | 900 | 严格过滤 SFT 训练 |
| `data/deepfactcite_strict/sft/test.parquet` | 47 | 严格引用评测 |
| `data/deepfactcite_strict/corpus.jsonl` | 4,766 | 严格语料 |
| `data/deepfactcite_mix/sft/train.parquet` | 1,693 | MixClean SFT 训练 |
| `data/deepfactcite_mix/sft/test.parquet` | 105 | MixClean SFT 测试 |
| `data/shortqa_guardrail/rl/test.parquet` | 32 | ShortQA answer/search 防护评测 |
| `data/searchr1_core_guardrail/test.parquet` | 200 | NQ 100 + HotpotQA 100 BM25 防护评测 |
| `data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite/train.parquet` | 28 | v3 GRPO 训练 |
| `data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite/test.parquet` | 4 | v3 GRPO 测试 |

数据样本长这样：

| 数据类型 | 主要字段 | 内容 |
|---|---|---|
| SFT 样本 | `prompt`, `answer`, `query` | `prompt` 是用户问题和搜索/引用协议；`answer` 是完整示范轨迹，里面包含 `<think>`、`<search>`、`<information>` 和最终 `<answer>`；`query` 是原始问题 |
| 基础 RL 样本 | `data_source`, `prompt`, `ability`, `reward_model`, `extra_info` | 主要给 GRPO rollout 用，`reward_model.ground_truth` 里有 query，有些样本没有短答案 target |
| v3 one-citation GRPO 样本 | `query`, `answer`, `prompt`, `reward_model`, `extra_info` | `answer` 是被筛出来的短 claim；`extra_info.selected_claims` 里有 claim、URL、title、support_score；`reward_model.ground_truth.reference_answer` 有带 markdown URL 的参考答案 |

一个 v3 样本可以概括成：

```text
query:
What happens to my $15/hr job if the minimum wage is increased to $15?

prompt:
先搜索；只输出 1 个短句；只用 1 个 markdown URL citation；
URL 必须从 tool response 复制；不要裸 [S_xxx] 或裸 URL。

selected_claim:
Research suggests that increasing the minimum wage to around $9.25 can erode employer labor-market power...

selected_url:
https://www.minneapolisfed.org/article/2023/better-together-pairing-smaller-minimum-wage-increases-with-tax-policy-to-reduce-inequality
```

所以 DeepFactCite 不是传统“问题 -> 唯一标准短答案”的数据集。SFT 阶段有示范答案轨迹；GRPO 阶段更多是让模型自己 rollout，再用规则 reward 检查它是否搜索、是否引用真实 URL、引用是否支持局部 claim。

### 2. Strict SFT 与 Soft SFT 的差异是什么？为什么 strict filtering 不一定赢？

Soft SFT 使用较宽的 DeepFactCite 轨迹，覆盖更多问题和行为模式。Strict SFT 只保留带检索 URL、URL 有效性高、claim support 达到阈值、unsupported rate 受控的样例。strict 数据更干净，但规模和多样性更低。

实际结果说明 strict filtering 不能自动成为赢家。Strict SFT 在相同 dynamic-LoRA vLLM 服务路径下没有击败 Soft SFT：ShortQA32 上 Soft 和 Strict 的 answer 都约 `0.469`，但 Strict 的 URL validity 和 claim support 更低；Strict47 上 Soft 的 total、URL validity、claim support 也更好。面试时要强调：数据清洁度只是训练前假设，是否提升要看同口径 eval，而不是看过滤规则本身。

### 3. MixClean200 是如何生成的？为什么选为当前 SFT baseline？

MixClean200 是从 `Base Qwen3-8B` 直接训练的 LoRA SFT，不是从旧 merged parent 继续训练。训练数据是 `data/deepfactcite_mix/sft`，混合了 soft 和 strict 样例；训练脚本是 `scripts/deepfactcite/launch_mix_clean_sft_200.sh`，配置为 `N_GPUS=2`、`TOTAL_STEPS=200`、`TRAIN_BATCH_SIZE=4`、`MICRO_BATCH_SIZE=2`，最终 validation loss `0.992`。

它被选为当前 SFT baseline，是因为同口径 dynamic-LoRA vLLM 评测下，它在三个不同评测口径上都比 Base 和 Soft100 更稳。这里要特别区分：`Search-R1 BM25 200` 不是 Strict47，它是 NQ/HotpotQA 的 answer/search 防护评测；Strict47 是 DeepFactCite 引用行为评测。

ShortQA32，用于看短问答和引用行为是否退化：

| 指标 | Base | Soft100 | MixClean200 |
|---|---:|---:|---:|
| answer_subEM | 0.219 | 0.438 | 0.500 |
| URL validity | 0.031 | 0.812 | 0.969 |
| claim support | 0.031 | 0.292 | 0.333 |

Search-R1 BM25 200，用于看 NQ/HotpotQA 风格 answer/search 能力是否保住：

| 指标 | Base | Soft100 | MixClean200 |
|---|---:|---:|---:|
| answer subEM | 0.290 | 0.450 | 0.490 |
| search success | 0.960 | 0.990 | 1.000 |
| budget fail | 0.170 | 0.045 | 0.025 |

Strict47，用于看 DeepFactCite 引用行为；这个集合没有可用 gold answer，所以不看 answer_subEM。这里的 `unsupported rate` 越低越好，代表模型引用的来源越能支撑它写出的局部声明：

| 指标 | Base | Soft100 | MixClean200 |
|---|---:|---:|---:|
| total | 0.120 | 0.137 | 0.142 |
| URL validity | 0.064 | 0.319 | 0.418 |
| claim support | 0.021 | 0.064 | 0.072 |
| unsupported rate | 0.574 | 0.851 | 0.879 |

所以结论是：MixClean200 适合作为 GRPO 初始化，因为它保住了 answer/search，并提升了 URL validity。但它不是最终可信引用结论，因为 Strict47 上 unsupported rate 仍高，说明 SFT 学会了更积极引用，但还没完全学会“只引用被支持的声明”。

### 4. ShortQA32 和 Strict47 分别有什么作用？Strict47 上 answer_subEM 有意义吗？

ShortQA32 是小规模 answer/search 防护评测，有 gold answer，可以看 `answer_subEM`、搜索成功、引用质量等。它用于检查模型有没有因为引用训练而损坏基本问答能力。

Search-R1 BM25 200 是另一套 answer/search 防护评测，来自 NQ 100 + HotpotQA 100，使用 wiki-18 BM25 检索器。它有 gold answer，所以 `subEM` 有意义；它不是 Strict47。

Strict47 是 DeepFactCite 严格引用评测集，重点看 citation behavior，例如 URL validity、citation precision、claim support、unsupported rate。它没有适合 `answer_subEM` 的 gold answer 字段，所以 Strict47 上的 `answer_subEM` 没有意义。面试时可以说：指标必须匹配数据 schema，否则数字再漂亮也不能解释模型能力。

更具体地说，Strict47 是 strict-filtered SFT 数据构建时按 `val_ratio=0.05` 留出的 test split：strict 过滤后一共 `947` 条 SFT 样本，随机打乱后 `900` 条写入 `data/deepfactcite_strict/sft/train.parquet`，`47` 条写入 `data/deepfactcite_strict/sft/test.parquet`。所以它是同源 strict 数据里的 holdout，不是从训练用的 900 条里再抽出来评测。MixClean200 的训练数据使用了 strict train 部分，Strict47 保留作引用行为测试。

因此，对“可信引用数据集是不是没标准答案，只能用 reward 测”要分开说：

- 如果“标准答案”指 NQ/ShortQA 那种可算 `answer_subEM` 的短 gold answer，Strict47 确实没有，所以不能用 Strict47 判断答案 EM。
- 如果“标准答案”指可参考的示范回答和检索证据轨迹，Strict47 是有的：每条样本有 SFT `answer`，里面包含检索返回的 `<information>`、URL、Text 和最终 `<answer>`。只是它的作用不是拿来做唯一答案匹配，而是拿来评估引用行为。
- 所以 Strict47 的测试主要依赖规则 reward/诊断指标，例如 URL 是否来自证据、claim 是否被证据支持、有没有 fake URL、有没有 no-citation。这不等于“只能看训练 reward”；它是在 holdout split 上对模型输出重新打分。

DeepFactCite 的测试拆成两条线：

1. 答案能力用有 gold answer 的数据测，例如 ShortQA32 和 Search-R1 BM25 200，看 `answer_subEM`、search success、budget fail。
2. 引用可信度用 reward/诊断指标测，例如 Strict47、v2/v3 rollout summary，看 URL validity、claim support、unsupported rate、fake URL、no-citation。

也就是说，Strict47 不是用来判断“答案是否和唯一标准答案一致”，而是用来判断“模型写出的引用是否真实、是否支撑它自己的局部声明”。这确实主要靠规则 reward 打分，但不是只看训练时的 reward 数字；要在 holdout split 或 saved checkpoint 的独立 eval 上，用同一套诊断指标比较不同模型。

`claim_support` 和 `unsupported_citation_rate` 确实有关系，二者来自同一套 citation-support 计算，但含义不同：

| 指标 | 含义 | 方向 |
|---|---|---|
| `claim_support` | 所有引用的平均支持分。每条引用大致是 `1.0/0.5/0.0`，再求平均 | 越高越好 |
| `unsupported_citation_rate` | 有多少比例的引用是失败引用。URL 无效，或者 support score `<0.5`，都算 unsupported | 越低越好 |

举例：如果一个答案有 4 个引用，support score 分别是 `[1.0, 1.0, 0.5, 0.0]`，那么 `claim_support=(1+1+0.5+0)/4=0.625`；其中只有 `0.0` 这个低于 `0.5`，所以 `unsupported_citation_rate=1/4=0.25`。如果 4 个引用是 `[0.5, 0.5, 0.5, 0.5]`，`claim_support=0.5`，但 `unsupported_citation_rate=0`，因为没有一条低于失败阈值。

所以它们不是完全重复：`claim_support` 看平均质量，`unsupported_citation_rate` 看失败比例。面试里可以说，一个衡量“整体支撑强不强”，另一个衡量“坏引用占多少”。

### 5. 构造数据时如何保证 URL 来源可验证？

数据构造时把检索返回的证据块保存为带 `URL:` 的 snippet，训练 prompt 和 reward 都围绕这些 URL。对于 claim-filtered/v2/v3 数据，还会检查所选引用必须来自检索返回 URL，并验证 OfflineSearchTool 在 `topk=2` 下能命中目标 URL。v2/v3 one-citation 数据都达到了 train `28/28`、test `4/4` 的 top-2 URL hit。

同时，reward 只认可 markdown URL 引用，例如 `[label](https://...)`。裸 `[S_xxx]`、裸 `[1]` 或非 markdown 的原始 URL 会被 cap 或视为无效格式。这保证模型不能只复制 snippet id 来假装引用。

## 3. 检索器与工具交互

### 1. 本地 BM25 和 e5 检索器的启动和区别是什么？

BM25 是稀疏检索，不需要 GPU，启动时指定 BM25 index、corpus 和 `retriever_name=bm25`。它速度快、工程简单，适合本地复现和私有领域语料。

e5 是稠密检索，需要 embedding 模型和 FAISS 索引。flat e5 做精确向量匹配，最好开 `--faiss_gpu`；ANN/HNSW64 可以在 CPU 上跑，速度更快但召回可能弱一些。

本项目主线 DeepFactCite 用轻量词汇检索和 offline corpus，是为了保证 URL 可审计。Search-R1 BM25 200 防护评测则使用 wiki-18 BM25，本地语料 `21,015,324` 条，topk 通常为 3。

### 2. 平面索引和 ANN 索引在 GPU/CPU 使用上有什么区别？如何选择？

flat index 是精确最近邻搜索，质量高但计算重。在线 RL 训练里如果要高吞吐，建议用 GPU FAISS flat index。

ANN/HNSW64 是近似最近邻，适合 CPU 或 GPU 不足时使用，吞吐更好但可能牺牲 top-k 召回。选择标准是：如果检索命中率直接影响 reward，优先选择可承受的最高召回；如果工程资源有限，先用 BM25 或 ANN 做可复现实验，再把检索器升级作为单独变量。

### 3. 为什么多节点训练每个节点都要启动检索服务器？如何保持一致性？

这个说法大方向对，但要把 “worker” 说清楚：搜索不是每张 GPU 训练 rank 自己单独做一次，而是 rollout/AgentLoop 里的环境工具会调用检索器。多节点时，Ray 可能把 rollout 相关进程调度到不同机器；如果只有 head node 有检索器，其他节点都要跨节点访问，延迟、网络故障和单点瓶颈都会放大。

如果用 Search-R1 风格的 HTTP retriever，通常每个节点启动一份相同检索服务；如果用 DeepFactCite 当前 offline tool，则每个节点要有相同的 `SEARCHQA_CORPUS`、index/top-k 和工具版本。面试里可以说：检索器是 RL 环境的一部分，不一致会导致同一个 prompt 在不同 rollout 上看到不同证据，进而污染 reward 和 advantage。

4GPU 单机 saved run 不是“几张卡专门生成、几张卡专门训练”。实际是同一组 4 张 GPU 同时承载两类进程：`ray::WorkerDict` 是 FSDP（Fully Sharded Data Parallel，把模型参数、梯度和 optimizer state 分片到多张 GPU 的训练并行方式）actor rank，负责 logprob/update/checkpoint；`sglang::scheduler_TP0-TP3` 是 SGLang TP=4 推理分片，负责 rollout 生成。训练循环上是先生成、再打分、再更新；显存上两套进程会共存。

### 4. DeepFactCite 如何确保模型只能引用实际检索到的 URL？

协议上，工具返回的 `<information>` 块带有 `URL:` 行，模型被要求只引用这些 URL。reward 上，最终答案的 markdown URL 会和当前 trajectory 中检索返回的 URL 集合做匹配，不在集合内就是 invalid/fake URL。

这里说的 v3 prompt-fix “进一步要求”，不是只在评测脚本里事后加过滤，而是在 v3 one-citation 数据构建时重写了 prompt 模板：训练/rollout 看到的指令就要求只输出一个短句、只复制检索证据里的 markdown URL citation，并禁止裸 snippet label、裸 `[1]` 和 raw URL。reward parser/cap 也按这个协议检查，所以数据约束、运行时 prompt 和 reward 是对齐的。

v3 2GPU 诊断的 fake URL rate `0.0000`，是在 `data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite` 这套 v3 one-citation GRPO 数据上做的 smoke rollout 诊断，规模是 32 条生成样本，不是 Strict47。4GPU saved32 的 fake URL rate `0.0000` 也是同一 v3 one-citation 训练设置下的 rollout 诊断，规模是 128 条生成样本；它说明这条受控训练轨迹里没有编造 URL，但不能直接当成 held-out 泛化结论。

## 4. SFT 训练与 LoRA 合并

### 1. Qwen3-8B LoRA SFT 的主要训练参数有哪些？

MixClean200 的主要参数是：

| 参数 | 值 |
|---|---|
| base model | `Qwen3-8B-Base` |
| 训练数据 | `data/deepfactcite_mix/sft` |
| GPU | 2 x A800 80G |
| total steps | 200 |
| max length | 8192 |
| train batch size | 4 |
| micro batch size | 2 |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| target modules | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| learning rate | `1e-5` |
| attention | 默认 `sdpa` |

### 2. Soft SFT adapter 与 bf16 merged full model 为什么数值上不等价？有什么影响？

先把几个词解释清楚：

| 术语 | 含义 |
|---|---|
| LoRA adapter | SFT 没有直接改完整 base 权重，而是训练一组低秩增量矩阵。推理时等价于在某些线性层上使用 `W_base + delta_lora` |
| PEFT dynamic LoRA forward | Hugging Face/PEFT 加载 base model 和 LoRA adapter，forward 时动态把 LoRA 增量加到对应层上，不提前把 adapter 写死进 base 权重 |
| bf16 merged full model | 先用 bf16 加载 base，再把 LoRA delta 合进 base weight，保存成一个完整模型目录 |
| fp32 merge | 用 fp32 加载 base 和 adapter，先在 fp32 精度下合并，再保存完整模型；这是更接近 PEFT dynamic LoRA 的合并方式 |

Soft100 的 LoRA adapter tensor 是 fp32。旧 merge 过程的问题是：base 先按 bf16 加载，随后把 fp32 LoRA delta 加进 bf16 base weight。这样做会把 adapter 里很多细小更新提前舍入到 bf16 权重里，等于损失了 SFT 学到的部分数值信息。

HF logits 检查证明这不是理论担心：同一个输入下，PEFT dynamic LoRA forward 与 in-memory bf16 merge 的 max abs logit diff 达到 `3.328 / 8.547`；而 PEFT dynamic LoRA forward 与 fp32 merge 的 max abs diff 小于 `0.0002`。也就是说，bf16 merge 后的模型已经不是原来 adapter path 的同一个函数；fp32 merge 才基本保持等价。

影响是：在搜索智能体里，早期 token 很关键。logit 差异可能改变第一个 `<search>` query，query 一变，检索结果、引用 URL、claim support 和最终 reward 都会跟着变。因此 serving path 是模型身份的一部分，dynamic LoRA、bf16 merged 和 fp32 merged 的结果不能混在一张 baseline 表里。

### 3. fp32 merged full model 在 vLLM 上如何使用？与 dynamic LoRA 有什么差异？

fp32 merged full model 是把 base 和 LoRA delta 合成一个完整 HF 模型目录，可以直接作为 serving/training parent 加载。它适合需要完整模型路径的训练或推理栈。

dynamic LoRA 是 vLLM 加载 base model，并通过 `--enable-lora`、`--lora-modules` 动态挂 adapter。它更接近 PEFT forward 的语义，也是当前 SFT baseline 评测的固定口径。二者不能默认等价；如果要比较，必须在同一 dtype、同一 serving path 下做 logits 或 eval 对齐检查。

### 4. 合并 LoRA adapter 的正确流程是什么？为什么 bf16 merge 有风险？

正确流程是：用 fp32 加载 base model，加载 LoRA adapter，执行 `merge_and_unload()`，保存完整模型，然后做 PEFT-vs-merged 的 HF logits 等价性检查。仓库脚本 `scripts/deepfactcite/merge_lora_adapter.py` 已把说明写清楚：训练 parent 推荐 `--torch-dtype float32`。

bf16 merge 的风险是把 fp32 LoRA 增量直接加到 bf16 base weight 上，导致 adapter 的细粒度更新被量化吞掉。它不是“不能跑”，而是不能假设和 dynamic LoRA 数值等价。

## 5. GRPO 训练与奖励设计

### 1. GRPO 相比 PPO 的优势是什么？为什么本项目选 GRPO？

PPO 通常需要训练 value critic 来估计 baseline，对 8B 模型来说等于多维护一个大模型副本，显存和工程复杂度都更高。GRPO 对同一个 prompt 采样多条回答，用组内均值和方差做 baseline，不需要 critic。

本项目的 reward 是结果型且可规则化计算：答案是否命中、是否搜索、URL 是否来自证据、声明是否被支持。GRPO 很适合这种“同一个问题下比较多条候选轨迹”的设置。4GPU `GRPO_N=4` saved32 运行中，`32/32` 个 group 都有非零 reward std，说明组内确实有可学习的相对差异。

更具体地说，4GPU saved32 的执行不是固定一部分 GPU 生成、一部分 GPU 训练，而是同一组 4 张卡同时放训练副本和推理副本：

| 角色 | 4 卡上的形态 | 负责什么 |
|---|---|---|
| FSDP actor | 4 个 `ray::WorkerDict` rank，每张 GPU 一个训练 rank | 持有分片后的 actor 参数/梯度/optimizer state，负责 logprob、loss、反向传播、参数更新和 checkpoint |
| SGLang rollout engine | `sglang::scheduler_TP0-TP3`，TP=4 横跨 4 张 GPU | 持有推理分片和 KV cache，负责多轮生成 rollout |
| AgentLoop / tool / reward | Ray actor 进程调度，不等于每张 GPU 一个检索器 | 解析 `<search>`，调用检索工具，把 tool response 塞回轨迹，并计算 reward |

FSDP 全称是 Fully Sharded Data Parallel。可以把它理解成“训练用的分片并行”：Qwen3-8B 太大，训练时不只需要权重，还需要梯度、optimizer state 和 activation；FSDP 让 4 张卡共同持有一个 actor，每张卡常驻一部分状态，需要 forward/backward 时再 all-gather 或 reduce-scatter。它和 SGLang TP 不同：FSDP 是训练副本，SGLang TP 是推理/rollout 副本。

训练流程可以按 step 讲：

1. dataloader 取一个 batch 的 prompt。4GPU saved32 中 `GRPO_N=4`，所以每个 prompt 会采样 4 条候选轨迹，32 个 group 总共形成 128 条 rollout 样本。
2. 当前 actor 权重同步到 SGLang rollout engine，SGLang 用 TP=4 在 4 张 GPU 上共同生成。
3. AgentLoop 监控模型输出；模型写出 `<search>...</search>` 时，工具调用本地 retriever/offline corpus，返回带 `URL:` 和 `Text:` 的 evidence，再继续让模型生成 `<answer>`。
4. rollout 结束后，DeepFactCite reward 解析完整轨迹，计算 answer/search/format、URL validity、claim support、unsupported citation rate 等分数。
5. 对同一个 prompt 的 4 条轨迹做组内比较：用这 4 个 reward 的均值和方差标准化 advantage。saved32 里 `32/32` 个 group 都有非零 reward std，说明组内确实有可学习差异。
6. FSDP actor 在 4 张 GPU 上重算模型生成 token 的 logprob，并用 `response_mask` 只训练 LLM 自己生成的 token，不训练工具返回文本。
7. 用 PPO-style clipped loss 更新 actor。更新发生在 FSDP 训练副本上；下一轮 rollout 前再把新 actor 权重同步给 SGLang 推理副本。

一句话总结：GRPO 提供“同题多样本相对比较”的学习信号；FSDP 让 8B actor 能在 4 卡上训练；SGLang TP 让同一个 actor 的 rollout 生成更快。

### 2. Rollout 中 LLM token 和工具返回 token 如何区分？为什么要 response_mask？

多轮 agent 轨迹里既有模型自己生成的 token，也有搜索工具返回的 observation。训练时只能优化模型生成的 token，不能让模型为检索器返回的文本背梯度。

因此 rollout 会构造 `response_mask`：LLM 生成 token 标为 1，工具返回 token 标为 0。GRPO advantage 会乘上 `response_mask`，只广播到模型真实生成的 token 上。这一点是搜索 agent RL 里很关键的工程细节。

### 3. Citation-aware reward 的组成和权重如何？各部分作用是什么？

默认 citation-aware 权重：

| 组件 | 权重 | 作用 |
|---|---:|---|
| answer | 0.15 | 保留答案相关性 |
| citation | 0.35 | 奖励引用 URL 来自检索证据 |
| support | 0.30 | 奖励被引用声明得到证据支持 |
| format | 0.10 | 约束 `<answer>`、markdown citation 等格式 |
| search | 0.05 | 鼓励合理搜索 |
| cost | 0.05 | 惩罚过多搜索和冗余 |

outcome-only 对照组则把 citation/support 权重置 0，主要看 answer、format、search、cost。

`answer=0.15` 不应被解释成“答案不重要”，而是因为当前 GRPO 阶段的主要缺口是 citation faithfulness。SFT 阶段已经用 MixClean200 和 ShortQA/Search-R1 BM25 guardrail 保住了基本答案与搜索能力；如果 GRPO 里继续把 answer 权重放得很高，模型容易回到“答得更宽、更像完整答案，但引用不一定支撑”的策略。更合理的做法是把 `0.15` 当作当前消融设置，而不是最终最优值；后续可以比较 `answer=0.15/0.25/0.35`，只有在 `answer_subEM` 提升且 `unsupported_citation_rate`、`no_citation` 不变差时，才考虑调高。

### 4. 为什么 outcome-only reward 可能导致引用不可信？

因为 outcome-only 只要答案、格式和搜索行为看起来对，就可能给较高分。模型可以学会“搜索一下、写一个答案、附一个看似相关的链接”，但这个链接未必来自当前检索证据，也未必支持 citation 附近的那句话。

v2 消融说明了这个问题：同一份 one-citation 数据上，outcome-only 的 URL validity 只有 `0.1562`，claim support `0.1094`，no-citation 失败 `27/32`；citation-aware 分别提升到 `0.7188`、`0.3906`，no-citation 降到 `9/32`。

## 6. GRPO 消融与小规模实验

### 1. v1、v2、v3 数据分别如何修改？

v1 是 claim-filtered 数据，保证选中的某些声明有支持，但原始问题仍可能很宽，模型容易回答宽问题并附上不完全支持的 URL。v1 结果中 citation-aware 没赢，说明只做 query-level 或部分 claim filtering 不够。

v2 改成 one-citation/one-claim：每行最多一个引用，最多一个短句，`max_query_tokens=18`、`max_claim_tokens=30`、`max_answer_sentences=1`，并验证 top-2 检索命中。

v3 在 v2 基础上重生成 prompt，加入更严格的 citation format template，明确要求输出一个短句、一个从检索证据复制的 markdown URL citation，禁止裸 `[S_xxx]`、裸 `[1]` 和原始 URL。

### 2. v2 one-citation 数据集如何构建？规模是多少？

v2 来自 `data/deepfactcite_mix/sft/train.parquet`，通过 `scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py` 过滤。核心条件是：1 个 citation、claim support 分数至少 1.0、query 和 claim 不过宽、答案最多 1 句、工具文本长度受控。

构建结果：保留 `32` 行，train/test 为 `28/4`，corpus docs `32`，平均所选引用数 `1.0`，train top-2 URL hit `28/28`，test top-2 URL hit `4/4`。主要拒绝原因是 `weak_claim_support=43`、`query_too_broad=19`、`too_few_supported_claims=3`、`claim_too_broad=1`。

### 3. v2 小规模 smoke 中 citation-aware 指标变化如何？

公平 v2 消融使用 2 x A800、SGLang rollout、`rollout.n=2`、`train_batch_size=1`、`total_steps=16`、`SEARCHQA_TOPK=2`、`save_freq=0`。

| 指标 | v2 Outcome-Only | v2 Citation-Aware |
|---|---:|---:|
| search | 0.9688 | 1.0000 |
| format | 0.9812 | 0.9625 |
| URL validity | 0.1562 | 0.7188 |
| citation precision | 0.1125 | 0.3937 |
| claim support | 0.1094 | 0.3906 |
| unsupported citation rate | 0.8438 | 0.3750 |
| fake URL rate | 0.0000 | 0.0000 |
| no_citation failures | 27 | 9 |
| response clip ratio | 0.0000 | 0.0000 |

结论是：v2 是第一个干净的正向 GRPO 信号，但它只是 16 step / 32 samples，不能直接当最终泛化结果。

### 4. 为什么 Citation-Strong 没能完全减少 no-citation 失败？

Citation-Strong 把权重改成更偏 citation/support：`answer=0.05`、`citation=0.45`、`support=0.35`、`format=0.10`、`search=0.05`、`cost=0.05`。结果 claim support 从 `0.3906` 提升到 `0.4219`，但 no-citation 仍是 `9/32`，unsupported rate 反而从 `0.3750` 变差到 `0.4375`。

这说明剩余问题不是简单调大 citation 权重就能解决，而是 prompt/reward 对“必须输出严格 markdown URL citation”的约束不够精确。因此后续做了 markdown citation cap、parser 修复和 v3 prompt-fix。

### 5. v3 prompt-fix 的改进点和效果是什么？

v3 的关键改进是让训练数据 prompt、运行时 rollout prompt 和 reward parser/cap 对齐：数据构建时重写 prompt 模板，明确要求一个短句、一个从检索证据复制的 markdown URL citation，并禁止裸 snippet label、裸 `[1]` 和 raw URL。数据本身仍是 32 行 one-citation，train/test `28/4`，corpus `32`。

v3 2GPU 诊断是在这套 v3 one-citation GRPO 数据上做的 smoke rollout，统计的是 32 条生成样本；它不是 Strict47，也不是完整 held-out 泛化评测。相对 v2 citation-aware 的变化是：

| 指标 | v2 Citation-Aware | v3 Prompt-Fix |
|---|---:|---:|
| search | 1.0000 | 0.9375 |
| format | 0.9625 | 0.9875 |
| URL validity | 0.7188 | 0.9375 |
| citation precision | 0.3937 | 0.6312 |
| claim support | 0.3906 | 0.6250 |
| unsupported citation rate | 0.3750 | 0.1250 |
| fake URL rate | 0.0000 | 0.0000 |
| no_citation failures | 9 | 2 |

这是第一个通过本地扩展门槛的结果。警告是 search 从 1.0 降到 0.9375，有 2 个 no-search/no-answer 失败，后续 saved run 需要盯住搜索行为。

## 7. 多节点训练与显存占用

### 1. Ray 在多节点训练中起什么作用？如何调度 actor、rollout 和 reward？

Ray 是分布式执行层，不是算法本身。它负责拉起本地或多节点 worker、分配 GPU、维护 placement group、管理 Ray actor 生命周期，并让主训练器通过 RPC 调用 `generate_sequences`、`compute_log_prob`、`update_actor` 等远端方法。

在当前 SGLang/veRL 栈中，Ray 管理 `WorkerDict` 训练 worker、`SGLangHttpServer`、`AgentLoopWorker`、`RewardManagerWorker` 等角色。多节点时，head node 提交训练任务，worker nodes 加入 Ray 集群；每个节点应启动一致的检索服务，或至少保证 offline corpus/index/tool 配置完全一致。

### 2. 8B 模型训练中的显存优化机制是什么？

Qwen3-8B bf16 权重本身约十几 GB，但训练时还需要参数、梯度、optimizer state、activation、logprob/update 临时张量和 NCCL buffer。当前使用 FSDP 分片 actor：多个 rank 共同持有一个 actor，每个 rank 只常驻部分参数/状态，forward/backward 时再 all-gather 或 reduce-scatter。

配置上还使用 `optimizer_offload=True` 降低训练显存，`gradient_checkpointing=True` 用额外计算换 activation 显存，`param_offload=False` 避免频繁 CPU/GPU 拷贝。rollout 侧则通过 SGLang TP、`max_num_seqs=1`、较小 `gpu_memory_utilization` 和限制 response/tool 长度控制显存。

### 3. TP engine 的作用是什么？为什么训练和推理要分开？

SGLang TP engine 是推理副本，负责高吞吐 rollout 生成。`tensor_model_parallel_size=4` 时，Qwen3-8B 推理模型被切到 4 张 GPU，每张卡持有一部分权重和 KV cache。

训练和推理分开，是因为二者优化目标不同。FSDP actor 适合反向传播、optimizer 和 checkpoint；SGLang 适合 prefix/KV cache、动态调度、多轮请求生成。如果直接用 FSDP actor 做多轮生成，吞吐和工具调用调度都会更差。当前 hybrid 流程是：actor 更新 -> 权重同步给 SGLang -> SGLang rollout -> reward -> actor 更新。

### 4. 每张卡上 actor rank 与 SGLang TP 分片的显存大概是多少？

4GPU 运行快照中，每张卡上同时有 `ray::WorkerDict` 和 `sglang::scheduler_TP*`：

| GPU | FSDP actor rank 显存 | SGLang TP 显存 |
|---:|---:|---:|
| 0 | 8,962 MiB | 13,838 MiB |
| 1 | 13,766 MiB | 14,030 MiB |
| 2 | 13,766 MiB | 14,030 MiB |
| 3 | 13,670 MiB | 13,838 MiB |

可以概括为：actor rank 约 9 到 14GB，SGLang TP rank 约 14GB。显存不完全一致，是因为 FSDP 分片、optimizer offload、临时通信 buffer、rank0 逻辑和 CUDA allocator cache 都会影响占用。

### 5. 训练和 rollout 中哪些操作最耗显存或计算？

当前瓶颈主要在 rollout 生成，而不是 actor backward。日志里 `timing_s/gen` 往往是 20 到 50 秒以上，`update_actor` 约 3 秒级。多轮 agent、SGLang 生成、工具调用、KV cache 和 response length 是主要成本。

显存上，SGLang 侧主要受模型 TP 分片和 KV cache 影响；actor 侧主要受 logprob、backward、activation、optimizer state 和 micro batch 影响。排查时如果 `scheduler_TP*` OOM，优先降 response length、`max_num_seqs`、SGLang gpu memory utilization；如果 `WorkerDict` OOM，优先降 micro batch、开更多 offload 或缩短 response。

## 8. Checkpoint 保存与可复现性

### 1. 4-GPU 保存 checkpoint 的流程是什么？为什么不直接用 8 GPU？

4GPU saved run 使用 `scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh`，默认 `DATA_DIR=v3_promptfix_onecite`、`MODE=citation-aware`、`N_GPUS_PER_NODE=4`、`TENSOR_MODEL_PARALLEL_SIZE=4`、`GRPO_SAVE_FREQ=16`、`GRPO_ACTOR_CKPT_SAVE_CONTENTS=[model]`。正式运行前先 `DRY_RUN=1` 检查命令，再跑短 sanity，最后跑中等规模保存实验。

不直接用 8GPU，是因为当时瓶颈不是并行度，而是实验控制、reward/prompt 定义、checkpoint/评测纪律和磁盘空间。v3 过了 2GPU 门槛后，4GPU 足够验证 saved checkpoint 链路。实际 4GPU `GRPO_N=4` saved32 运行已保存 checkpoint，目录约 `31G`。

### 2. DRY_RUN 与 GRPO_SAVE_FREQ 的意义是什么？

`DRY_RUN=1` 用于打印和检查最终 Hydra/veRL 命令，不真正启动训练，适合在 CPU 或无 GPU 状态下检查路径、参数和数据。saved wrapper 默认 dry-run，就是为了避免误开昂贵训练。

`GRPO_SAVE_FREQ` 控制每隔多少 step 保存 checkpoint。脚本还会预检查 train rows、total steps、epochs 和 save freq，避免“训练自然结束但没跑到保存点”。`GRPO_REQUIRE_FINAL_CHECKPOINT=1` 会在训练结束后强制检查 tracker 和 actor checkpoint。

### 3. checkpoint 需要多少磁盘？model-only 和 extra state 差异是什么？

计划文档给出的最低建议是：model-only checkpoint 至少 `35G` 可用空间；带 extra state、可恢复训练的 checkpoint 至少 `50G` 可用空间。

model-only 主要保存可用于评测/serving 的模型权重，磁盘更省。带 `extra` 会保存优化器、scheduler、训练状态等，适合断点恢复，但更占空间。实际 4GPU saved32 checkpoint 是 4 个 model shard，每个约 `8.19G`，总目录约 `31G`。

### 4. 如何验证 serving path 和模型 artifact 一致性？为什么 dynamic LoRA、bf16 merge、fp32 merge 不能混用？

验证步骤包括：记录 base model、adapter、merge dtype、serving backend、tensor parallel、dtype、eval 数据和脚本；必要时做 PEFT dynamic LoRA vs merged model 的 HF logits 检查；评测表只比较同一 serving path 下的结果。

不能混用的原因是它们不是同一个模型身份。Soft100 上已经证明 bf16 merged 与 dynamic LoRA logits 差异很大，而 fp32 merge 才接近 dynamic LoRA。搜索智能体对首个搜索 query 很敏感，serving path 差异会放大成不同检索轨迹。

### 5. 如果 checkpoint 或复现路径异常，应该如何执行复现？

先不要扩大训练。复现流程应该是：固定假设 -> 固定模型身份 -> 固定数据和 corpus -> dry-run 检查命令 -> 小规模 smoke -> 解析 rollout JSONL -> 对比同口径指标 -> 再决定是否保存 checkpoint。

如果是 saved run，先检查 `latest_checkpointed_iteration.txt` 是否等于预期 step，再检查 `global_step_x/actor` 是否存在、shard 是否完整、磁盘是否足够。若是 eval 结果异常，优先检查 serving path、dtype、LoRA/merge 身份、retriever URL/topk 和 prompt 版本，不要直接归因于训练算法。

## 9. 面试技巧与核心问题

### 1. 如何通俗解释 citation-aware reward 的必要性？为什么 SFT 无法直接教会搜索 query？

可以这样说：SFT 像给学生看标准答案和标准引用格式，它能教会“长什么样”。但真实搜索任务里，老师很难提前标注每个问题最应该搜哪几个关键词、应该选哪个证据、引用是否刚好支持当前句子。这些是结果导向的行为。

所以我用 citation-aware GRPO，把答案正确性、URL 真实性和声明支持性做成 reward。模型生成多条搜索轨迹后，系统比较哪条证据更可靠、哪条引用更支持声明，间接训练“什么时候搜索、搜什么、引用什么”。

### 2. 解释 GRPO 的组内 advantage 计算，以及为什么不需要 critic。

对同一个 prompt 采样 `n` 条回答，分别得到 reward score。然后计算：

```text
adv_i = (score_i - mean(score_group)) / (std(score_group) + epsilon)
```

比组内平均高的回答 advantage 为正，训练提高这些 token 的概率；比组内平均低的回答 advantage 为负，训练降低这些 token 的概率。这个组内均值就是 baseline，所以不需要额外训练 value critic。

多条 rollout 的奖励可比较，是因为它们来自同一个 prompt、同一检索器、同一 reward 函数和同一 decoding 配置。跨不同 prompt 的 raw reward 不一定直接可比，但 GRPO 主要使用组内相对比较。

### 3. 如何解释 SGLang、FSDP、Ray、AgentLoop 与 GRPO 的协作关系？

30 秒答法：Ray 负责把训练系统里的角色调度起来；FSDP actor 负责训练、logprob 和参数更新；SGLang 负责高吞吐生成多轮搜索 rollout；AgentLoop 把 LLM 生成和搜索工具调用串成完整轨迹；DeepFactCite reward 给轨迹打分；GRPO 用同一 prompt 下多条轨迹的相对分数更新 actor。

如果面试官追问进程，说明每张 GPU 上会看到两类主要进程：`ray::WorkerDict` 是 FSDP actor rank，`sglang::scheduler_TP*` 是 SGLang tensor parallel rank。训练副本负责学习，推理副本负责生成样本。

### 4. 为什么多轮 agent rollout 中一个小 token 差异会导致指标大幅波动？

搜索 agent 是闭环系统。第一个 token 差异可能改变搜索 query；query 改变会改变检索证据；证据改变会改变最终答案和引用；引用改变又会影响 URL validity、claim support 和 unsupported rate。

这也是为什么 bf16 merge、serving path、temperature、topk、prompt 细节都必须固定。它们看似只是底层实现差异，但在多轮工具调用里会通过检索环境放大，最后表现成完全不同的 rollout 轨迹和指标。

## 10. 数据集扩展与 Prompt 设计

### 1. DeepFactCite 数据集如何构建？基础版、strict 版和 one-citation v3 有什么区别？

基础版从 DeepCiteFact 轨迹转换而来，保留 prompt、answer、query、检索轨迹和带 URL 的 corpus。它的目标是先让模型学会 Search-R1 风格的搜索、读取证据和带引用回答。规模是 SFT `1,140/60`，RL `1,140/60`，corpus `6,118` 条。

strict 版是在基础版上做过滤，只保留 URL validity 和 claim support 更好的 SFT 轨迹。过滤条件包括 `min_url_validity=1.0`、`min_claim_support=0.35`、`max_unsupported_rate=0.5`、至少 1 个引用。它留下 SFT `900/47`，corpus `4,766` 条。strict 的作用是做“干净引用数据”消融，但它不自动等于更好训练数据，因为过滤会损失行为多样性。

v3 one-citation 数据是 GRPO 受控数据，不是通用大训练集。它从 mix SFT 数据里筛出 32 行，train/test `28/4`，corpus `32` 条。每行只围绕一个短 claim、一个 URL citation，并用 prompt-fix 明确要求输出一个短句和一个 markdown URL citation。它的目的是隔离 citation reward 是否有效，而不是覆盖所有真实问答形态。

### 2. 平均引用数量、句子长度和 claim token 限制是多少？为什么这样设？

基础 SFT train 平均每条约 `4.14` 个 markdown citation，strict train 约 `3.99`，mix train 约 `4.10`。这些样本更接近长答案引用训练，但在 RL 阶段变量太多：多句、多引用、多 claim 会让 reward 很难判断到底是哪一句、哪个链接出了问题。

v2/v3 one-citation 数据的构建约束更窄：

| 约束 | 值 | 目的 |
|---|---:|---|
| `max_citations` | 1 | 让 reward 能明确对应一个 claim 和一个 URL |
| `max_claim_tokens` | 30 | 避免 claim 太宽，证据只支持一部分 |
| `max_query_tokens` | 18 | 避免问题过宽，检索目标不清楚 |
| `max_answer_sentences` | 1 | 降低多句答案里局部 attribution 的歧义 |
| `max_tool_text_chars` | 700 | 控制工具返回长度和显存/上下文压力 |

面试里可以说：这些限制不是最终产品形态，而是为了先做机制验证。先让模型在一个短 claim 上学会“搜到证据、复制真实 URL、只写证据支持的话”，再考虑扩展到多句多引用。

### 3. Prompt 如何确保输出 `<search>`、`<information>` 和 `<answer>` 格式？SFT 中 prompt 影响多大？

Prompt 会明确告诉模型：如果缺知识，先用 `<search> query </search>` 调工具；环境会把检索结果插入 `<information>` 或 `<tool_response>`；最终答案必须放在 `<answer>...</answer>` 中，并使用 markdown citation `[label](URL)`。

格式不是只靠 prompt，SFT 也在示例轨迹里反复展示这种协议。也就是说，prompt 给规则，SFT 给示范，reward 再惩罚偏离格式的行为。SFT 阶段 prompt 影响很大，因为搜索 agent 的动作空间不是普通续写：模型必须学会什么时候输出 search tag、什么时候停下来写 answer、如何把 URL 放进 markdown citation。prompt 稍微不明确，就可能出现裸 `[S_xxx]`、裸 URL、没有 `<answer>` 或没有搜索。

### 4. v2 与 v3 prompt-fix 有哪些改进？对指标有什么影响？

v2 已经把数据收窄到 one-citation，但 prompt 仍不够强，模型有时输出裸 snippet id、没有 markdown URL，或者回答比证据支持范围更宽。v3 的改进不是事后筛掉坏输出，而是在数据构建阶段重新生成 prompt，并在训练/rollout 时使用这个更严格的 prompt：加入 citation format template，明确要求只输出一个短句、只输出一个从检索证据复制的 markdown URL citation，禁止裸 `[S_xxx]`、裸 `[1]` 和 raw URL。

v3 2GPU 诊断是在 v3 one-citation GRPO 数据上统计 32 条 rollout 样本，不是 Strict47。相对 v2 citation-aware 的变化是：

| 指标 | v2 citation-aware | v3 prompt-fix |
|---|---:|---:|
| URL validity | 0.7188 | 0.9375 |
| citation precision | 0.3937 | 0.6312 |
| claim support | 0.3906 | 0.6250 |
| unsupported citation rate | 0.3750 | 0.1250 |
| no-citation failures | 9/32 | 2/32 |
| fake URL rate | 0.0000 | 0.0000 |

这说明 prompt-fix 不只是让格式更漂亮，而是让模型更稳定地引用真实 URL，并减少“URL 真实但 claim 不被支持”的情况。

### 5. 如何限制多句或多引用回答？为什么 GRPO 用 one-citation 数据？

限制来自两层：数据构建时设置 `max_citations=1`、`max_answer_sentences=1`、`max_claim_tokens=30`；prompt 中再要求只输出一个短句和一个 markdown URL citation。reward 侧还会惩罚 no citation、裸 snippet id、raw URL 和 unsupported citation。

GRPO 先用 one-citation，是为了降低 credit assignment 难度。多句多引用答案里，一个总 reward 很难告诉模型是哪一句不该写、哪个 URL 不支持、哪条引用缺失。one-citation 把问题变成最小闭环：一个问题、一个搜索证据、一个 claim、一个 citation。这个闭环跑通后，再扩到多引用更可控。

## 11. 检索器性能与 Top-k / 响应长度

### 1. 本地 BM25 与 E5 dense retriever 有什么差别？GPU vs CPU 怎么选？

BM25 是稀疏检索，不需要 GPU，启动简单，适合本地可复现和小语料实验。E5 dense retriever 要先把 query 和文档映射成向量，再用 FAISS 检索；flat index 更准确但计算重，建议用 GPU；HNSW/ANN 可以在 CPU 上跑，速度快但可能牺牲召回。

选择原则：如果目标是先验证 RL reward 和 citation 行为，BM25/轻量 offline retriever 更稳，因为变量少；如果目标是通用开放域问答性能，E5 flat/ANN 更接近 Search-R1 大检索设定，但要单独控制索引、召回和吞吐变量。

### 2. Top-k 对搜索成功率和 citation precision 有什么影响？

Top-k 越大，模型看到正确证据的概率通常越高，但也会引入更多干扰片段。对 citation reward 来说，top-k 太小可能搜不到目标 URL，导致模型无法引用正确来源；top-k 太大可能让模型引用相关但不支持当前 claim 的片段，降低 citation precision。

当前口径里，SFT/guardrail eval 常用 `topk=3`，v2/v3 GRPO 使用 `SEARCHQA_TOPK=2`。v2/v3 构建后专门验证了 top-2 命中：train `28/28`，test `4/4`。这说明在受控 GRPO 数据上，`topk=2` 已足够让目标 URL 出现在工具返回中，同时减少干扰。

### 3. `max_tool_response_length` 和 `max_response_length` 对 URL validity 与 claim support 有什么影响？

`max_tool_response_length` 控制每次搜索返回给模型的证据文本长度。太短会截断关键句，模型可能拿到 URL 但看不到支持 claim 的文字，claim support 会受损；太长会增加上下文、KV cache 和生成成本，也可能让模型被无关内容干扰。

`max_response_length` 控制模型最终回答和中间生成预算。太短会导致 answer 被截断、引用丢失或格式不完整；太长则鼓励模型写更多背景句，增加 unsupported claim 的概率。当前 2GPU GRPO 常用 `max_tool_response_length=768`、`max_response_length=384`，v2/v3 的 response clip ratio 为 `0.0000`，说明这个长度对受控 one-citation 任务够用。

### 4. 为什么每个节点要启动相同检索服务器？多节点如何保证一致性？

多节点训练时，不是每张 GPU 都单独维护一个检索器，而是 rollout/AgentLoop 的环境工具会调用检索。Ray 可能把这些 rollout 相关进程调度到不同节点；如果不同节点连到不同检索器、不同索引或不同 corpus，同一个 prompt 的证据就可能不同，reward 不再可比，GRPO 组内 advantage 也会被污染。

一致性做法是：HTTP retriever 模式下每个节点启动相同检索服务；offline tool 模式下每个节点放同一份 corpus/index，并固定 top-k、版本、端口或环境变量。训练日志里记录 `SEARCHQA_CORPUS`、`SEARCHQA_TOPK` 和 retriever backend。面试里可以说：检索器是 RL 环境的一部分，环境不一致就等于训练任务不一致。

### 5. 离线检索器如何验证命中率和 top-k 合理性？

对受控数据，构建脚本会把目标 URL 写入 corpus，并用同一个 OfflineSearchTool 检查 top-k 是否能返回目标 URL。v2/v3 都验证了 `topk=2` 下 train `28/28`、test `4/4` 命中。

训练后还要看 rollout JSONL：search 是否接近 1、URL validity 是否高、no-search/no-citation 是否低。如果检索命中高但 claim support 低，问题不在检索器，而在模型把真实 URL 挂到了不被支持的句子后面；这正是 v1/v2 迭代要解决的失败模式。

## 12. 消融实验与 Small-scale / Medium-scale / 4-GPU Run 对比

### 1. v1、v2、v3 和奖励消融的设计思路是什么？

v1 先做 claim-filtered 数据，验证工程链路和 citation-aware reward 能否跑通，但结果显示宽问题仍会诱导模型写 unsupported 背景句。v2 把问题收窄到 one-citation/one-claim，用同一数据比较 outcome-only 与 citation-aware。v3 再修 prompt 和 parser，让“必须输出 markdown URL citation”与 reward cap 对齐。

small-scale run 的作用不是估计最终泛化，而是验证机制：同一模型、同一数据、同一检索器下，citation-aware 是否比 outcome-only 改善 URL validity、claim support、unsupported rate，并且不破坏 search/format。

### 2. 2-GPU smoke run 如何控制每个 prompt 的 sample 数和最大 turn？

2GPU ablation 默认 `train_batch_size=1`、`rollout.n=2`，所以每个 step 一个 prompt，生成 2 条候选轨迹供 GRPO 组内比较。SGLang rollout 使用 async multi-turn，`agent.num_workers=2`，`max_assistant_turns=3`，`max_user_turns=3`，`max_num_seqs=1`，并限制 `max_response_length=384`、`max_tool_response_length=768`。

这个设置保守但稳定：吞吐不是最高，但能让每个样本的行为可审计，便于排查 no-search、no-citation、unsupported citation。

### 3. 4-GPU saved run 的目标是什么？和 2GPU smoke 有何差别？

2GPU smoke 的目标是机制验证，通常 `save_freq=0`，不保留 checkpoint。4GPU saved run 的目标是打通“可保存、可复评”的中等规模训练链路，产出可以做 held-out eval 的 actor checkpoint。

实际 4GPU `GRPO_N=4` saved32 运行配置是 32 steps、每 prompt 4 条 rollout、model-only checkpoint。它完成后保存到 `global_step_32/actor`，checkpoint 约 `31G`。训练 rollout 指标为：search `0.9688`、URL validity `0.9453`、claim support `0.6367`、unsupported `0.1016`；后半程 search/URL/citation_count 达到 `1.0000`，unsupported 降到 `0.0312`。这些说明训练轨迹改善，但仍需 held-out eval 才能下最终结论。

### 4. 为什么 mix50 continuation 不应作为有效 baseline？指标有什么变化？

mix50 从旧的 bf16 merged Soft parent 继续训练，而这个 parent 已经被证明不等价于 dynamic LoRA。它训练完成不代表实验有效，因为 lineage 不干净。

指标上，mix50 在 ShortQA32 上 answer `0.375`、URL validity `0.625`、claim support `0.190`，比 Soft/MixClean200 弱；Strict47 上 total `0.150`、URL validity `0.312`、claim support `0.087`，也不能作为可靠胜出证据。正确做法是把它记录为失败台账，而不是继续从它出发训练。

### 5. v2 与 v3 上 citation-aware 相比 outcome-only 的改善差异如何？

v2 上，citation-aware 相比 outcome-only 有明显正向信号：

| 指标 | v2 outcome-only | v2 citation-aware |
|---|---:|---:|
| URL validity | 0.1562 | 0.7188 |
| claim support | 0.1094 | 0.3906 |
| unsupported citation rate | 0.8438 | 0.3750 |
| no-citation | 27/32 | 9/32 |

v3 没有再跑完整同口径 outcome-only，对比重点是 v3 prompt-fix 相对 v2 citation-aware 的改进：URL validity `0.9375`、claim support `0.6250`、unsupported `0.1250`、no-citation `2/32`。这些数来自 v3 one-citation GRPO 数据上的 2GPU smoke rollout 诊断，不是 Strict47。它说明剩余瓶颈不只是 reward 权重，而是 prompt、parser 和 reward cap 必须对齐。

## 13. SFT Baseline 对 GRPO 收敛的影响

### 1. MixClean200 为什么被选作 GRPO 初始化？ShortQA32 和 Strict47 指标是多少？

MixClean200 是当前最干净的 SFT 初始化，因为它在同一 dynamic-LoRA vLLM serving path 下，同时改善 answer/search guardrail 和引用行为。

ShortQA32：

| 模型 | answer_subEM | URL validity | claim support | unsupported |
|---|---:|---:|---:|---:|
| Base | 0.219 | 0.031 | 0.031 | 0.906 |
| Soft100 | 0.438 | 0.812 | 0.292 | 0.557 |
| MixClean200 | 0.500 | 0.969 | 0.333 | 0.495 |

Strict47：

| 模型 | total | URL validity | claim support | unsupported | fake URL |
|---|---:|---:|---:|---:|---:|
| Base | 0.120 | 0.064 | 0.021 | 0.574 | 0.000 |
| Soft100 | 0.137 | 0.319 | 0.064 | 0.851 | 0.064 |
| MixClean200 | 0.142 | 0.418 | 0.072 | 0.879 | 0.071 |

MixClean200 不是最终可信引用模型，因为 Strict47 unsupported 仍高；但它是更好的 GRPO 起点。

### 2. 为什么 strict SFT 过滤更多轨迹却不一定击败 soft SFT？

strict filtering 提高了样例干净程度，但减少了行为多样性。搜索 agent 不只学引用格式，还要学什么时候搜索、怎么构造 query、如何组织答案。过度过滤可能让模型见到的问题类型和表达方式变窄。

实际结果也支持这一点：Strict SFT 没有在相同 eval path 下击败 Soft SFT。因此 strict 数据适合作为消融和引用评测，不应自动晋升为默认 baseline。

### 3. SFT 如何教搜索/引用行为？GRPO 什么时候依赖 SFT baseline？

SFT 给模型提供“动作模板”：看到问题后如何输出 `<search>`，如何读 `<information>`，如何在 `<answer>` 中写 markdown citation。没有 SFT，GRPO 初始策略可能根本不会搜索或不会按协议输出，reward 稀疏且难学。

GRPO 特别依赖一个能基本完成任务的 SFT baseline：它需要同一 prompt 下多条 rollout 有可比较的好坏。如果初始模型大多数样本都 no-search/no-answer/no-citation，组内 reward 方差低，优势估计就没有训练信号。

### 4. LoRA 与 merged full model 的差异如何影响 GRPO 收敛？

dynamic LoRA、bf16 merged、fp32 merged 不是可以随便互换的模型身份。bf16 merge 已经被 HF logits 检查证明会明显偏离 PEFT dynamic LoRA；搜索 agent 对首个 query 和早期 token 很敏感，小的 logit 差异会变成不同检索轨迹。

如果 GRPO 初始化用了错误 merged parent，训练可能仍能跑，但收敛曲线和指标解释不可信。正确做法是固定 serving/training path，并在需要 merge 时优先 fp32 merge，再做 logits 等价性检查。

### 5. SFT batch size、sequence length、LoRA rank 对 GRPO 初始探索有什么影响？

MixClean200 SFT 使用 `train_batch_size=4`、`micro_batch_size=2`、`max_length=8192`、`LoRA rank=32`。这些参数影响 GRPO 的初始策略质量。

`max_length=8192` 让 SFT 能看到较完整的搜索轨迹和长引用答案；太短会截断 evidence/answer，使模型学不到完整协议。LoRA rank 影响可学习容量，rank 太低可能学不到工具调用和引用格式，太高则更占显存和更容易过拟合。batch size 影响训练稳定性；太小噪声大，太大可能牺牲探索多样性。当前参数是工程折中，不是理论最优。

## 14. vLLM、Tensor Parallel 和 `rollout.n` 对吞吐量与显存占用

### 1. 为什么训练 actor 和 SGLang rollout engine 共用 GPU？显存大致如何？

训练 actor 负责 logprob 和参数更新，SGLang rollout engine 负责快速生成多轮搜索轨迹。两者共用同一组 GPU，是为了在有限 A800 资源上同时保留训练副本和推理副本，并在 rollout 前同步权重。

4GPU saved run 不是固定几张卡生成、几张卡训练，而是 4 张卡都参与 actor 训练，也都参与 SGLang TP=4 rollout 推理。快照中每张卡上大致有两类进程：`ray::WorkerDict` 的 FSDP actor rank 约 `9-14GB`，`sglang::scheduler_TP*` 约 `14GB`。这说明显存不是只被模型权重占用，还包括 KV cache、activation、optimizer/offload buffer、NCCL 和 allocator cache。

### 2. `tensor_model_parallel_size` 如何影响显存分片和 NCCL 通信？

TP size 越大，每张 GPU 持有的推理权重越少，单卡显存压力降低；但每次 forward 需要更多跨卡通信，NCCL 开销增加。Qwen3-8B 4GPU saved run 用 `tensor_model_parallel_size=4`，2GPU smoke 用 TP=2。

不要随便用 3GPU/TP=3，因为 head 数、SGLang 支持、通信拓扑和脚本都未验证。项目文档里明确建议：没有 4 张卡时用稳定 TP=2 fallback，而不是临时发明 3GPU 拓扑。

### 3. `rollout.n` 对 GRPO advantage 和稳定性有什么作用？

`rollout.n` 是每个 prompt 采样多少条候选轨迹。GRPO 用同一 prompt 下这些轨迹的 reward 均值和方差计算 advantage。`n` 越大，组内比较越稳定，也更容易出现有差异的候选，但生成成本和显存/时间也会上升。

2GPU smoke 默认 `n=2`，够做便宜机制测试。4GPU saved32 用 `GRPO_N=4`，`32/32` 个 group 都有非零 reward std，mean group reward std `0.0799`，说明每组 4 条 rollout 确实提供了可学习的相对信号。

### 4. vLLM KV cache 与 prefix cache 有什么作用？如何影响多轮吞吐？

在 vLLM/SGLang 这类推理引擎里，KV cache 保存已生成上下文的 attention key/value，避免每生成一个 token 都重新计算全部历史。prefix cache 则复用多个请求共享的前缀，例如相同 system prompt、工具协议和部分对话上下文。

多轮 search agent 中，prompt 很长且多次调用工具，KV/prefix cache 对吞吐很重要。代价是 cache 占 GPU 显存；如果 `max_response_length`、tool response 或并发数增加，KV cache 会变大，可能导致 `scheduler_TP*` OOM 或生成变慢。

### 5. 增大 tool/response length 会如何影响显存和生成时间？怎么平衡？

增大 `max_tool_response_length` 会给模型更多证据，可能提高 claim support，但也会增加上下文长度、prefill 时间和 KV cache。增大 `max_response_length` 可以减少截断，但也让模型更容易写多余背景，增加 unsupported claim。

平衡方法是看失败类型：如果很多样本因为证据被截断而 unsupported，可以适当增大 tool length；如果 no-citation 或格式丢失来自 response 截断，可以增大 response length；如果 unsupported 来自模型乱扩展答案，就不应盲目增大长度，而应收紧 prompt 或 reward。

## 15. Checkpoint 保存与可复现性补充

### 1. 4-GPU saved checkpoint 流程是什么？`DRY_RUN`、`GRPO_SAVE_FREQ` 有什么意义？

流程是先用 wrapper dry-run 检查命令和路径，再短跑 sanity 确认能启动和写 checkpoint，最后跑 saved medium run。`DRY_RUN=1` 只打印命令，不启动训练；`GRPO_SAVE_FREQ` 控制每多少 step 保存一次；`GRPO_REQUIRE_FINAL_CHECKPOINT=1` 会在结束后检查 tracker 和 actor checkpoint 是否存在。

saved32 实际运行使用 `GRPO_TOTAL_STEPS=32`、`GRPO_SAVE_FREQ=32`、`GRPO_N=4`，最终 `latest_checkpointed_iteration.txt=32`，说明保存点和预期一致。

### 2. model-only checkpoint 与 extra state 的磁盘差别？

model-only 只保存评测/serving 所需的 actor model 权重，磁盘压力较小；带 extra state 会保存 optimizer、scheduler、训练状态，能恢复训练但更占空间。项目建议 model-only 至少准备 `35G`，带 extra state 至少 `50G`。

saved32 的 actor checkpoint 是 4 个 shard，每个约 `8.19G`，总目录约 `31G`。保存后 `/root/autodl-tmp` 只剩约 `16G`，所以继续 saved GRPO 前必须先规划磁盘。

### 3. 如何验证 serving path、artifact 与 rollout JSONL 一致性？

需要记录并核对：模型路径、base/adapter/merge dtype、serving backend、TP size、dtype、数据集、corpus、top-k、prompt 版本、reward 版本、rollout JSONL 路径和 checkpoint step。

rollout JSONL 用于解释训练时模型到底搜了什么、看到了什么 URL、最终引用了什么。held-out eval 则要用保存的 checkpoint 单独跑，不能把训练 rollout 指标当泛化结果。若 eval 数字异常，先查 serving path 和数据路径，而不是直接怀疑算法。

### 4. 为什么 dynamic LoRA、bf16 merge、fp32 merge 不能混在一张评测表？

因为它们可能不是同一函数。Soft100 的诊断显示，PEFT dynamic LoRA 与 bf16 merged logits 差异很大；fp32 merge 才接近 dynamic LoRA。搜索 agent 的 rollout 对早期 token 很敏感，不同 serving path 可能生成不同 query，进而看到不同证据。

因此评测表必须标注 serving path。当前 SFT baseline 表固定使用 vLLM dynamic LoRA bf16；如果用 merged full model，要单独成表或先做等价性检查。

### 5. checkpoint 保存失败或指标异常时，复现和审计步骤是什么？

先查基础工件：`latest_checkpointed_iteration.txt`、`global_step_x/actor`、shard 数量和大小、磁盘剩余空间、训练日志末尾是否有 OOM/NCCL/traceback。再查配置：`GRPO_TOTAL_STEPS` 是否真的能跑到保存点，`GRPO_SAVE_FREQ` 是否整除或覆盖最终 step。

指标异常时按顺序排查：模型身份、serving path、prompt 版本、retriever corpus/top-k、reward parser、rollout JSONL。原则是先复现最小 smoke，再比较同口径指标，不要在原因不明时扩大 GPU 或继续长跑。

## 16. 当前可防守结论

面试中最稳的总结是：

```text
我没有把项目包装成直接击败 Search-R1 论文结果，而是做了一个受控的 citation-aware GRPO 扩展。先用 MixClean200 建立 SFT baseline，并用 ShortQA32 和 Search-R1 BM25 200 保证 answer/search 不崩；然后通过 v1/v2/v3 小规模消融逐步定位问题。v2 证明 citation-aware reward 相比 outcome-only 能显著改善 URL validity 和 claim support；v3 进一步证明 prompt/reward/parser 对齐后，no-citation 和 unsupported citation 明显下降。最后用 4GPU N=4 saved32 跑通了保存 checkpoint 链路，但训练 rollout 指标不等于 held-out 泛化指标，下一步必须做同口径 checkpoint eval。
```
