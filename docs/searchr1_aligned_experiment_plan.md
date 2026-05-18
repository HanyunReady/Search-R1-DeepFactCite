# DeepFactCite 与 Search-R1 对齐的实验计划

这个计划保留原始 Search-R1 的工作流结构，同时把优化目标从“答案正确”扩展为“答案正确 + 引用可信”。

## 项目成功标准

项目不能只因为一次训练跑完就算成功。只有同时满足下面两个条件，才算成功：

```text
1. 评测目标达成：
   - Search-R1 风格的答案/搜索 benchmark 没有明显退化。
   - DeepFactCite 的引用指标有明显提升。

2. 工程学习过程可审计：
   - 每一次失败的、或没有获胜的 SFT/merge/eval 尝试都被记录。
   - 每条记录都包含假设、精确 artifact 路径、评测设置、
     观测指标、根因分析，以及下一条防护措施。
```

失败记录本身是最终交付物的一部分。它应该足够回答资深面试官会问的问题，例如：

```text
SFT 阶段失败了什么，为什么失败，你如何证明根因，
以及你做了什么改变让结果变得可复现？
```

## 定位

除非严格复现了原始 benchmark baseline，否则不要宣称 DeepFactCite “击败 Search-R1”。更准确的项目表述是：

DeepFactCite 在 Search-R1 风格搜索智能体目标上扩展了 URL 真实性和声明支持奖励，在保持答案/搜索性能的同时，提升引用有效性和引用支持性。

## 外部 Search-R1 参考

以下内容只能作为参考点，不能作为主要因果 baseline：

```text
PeterJinGo/Search-R1 和 Search-R1-v0.3 collections：
  原始发布 checkpoint 基于 Qwen2.5/Llama3.2，而不是 Qwen3。

orbit-ai/searchr1-repro-4b：
  在 NQ/HotpotQA 上复现的 Qwen3-4B Search-R1 风格 GRPO，
  训练 200 steps，使用 <search>/<information> 工具协议。

jiulaikankan/Qwen3-4B-Thinking-Search-R1-baseline：
  来自 OpenRLHF-Agent/Search-R1 的小型 Qwen3-4B-Thinking adapter/checkpoint，
  可用于 sanity check，但不足以作为主要 8B baseline。
```

公平的 8B 对比仍然应该是：相同 backbone、相同检索器、相同 prompt/data，只改变 reward 目标：
outcome-only/Search-R1-style 对比 citation-aware DeepFactCite。

## 原始 Search-R1 结果参考

最终报告应该包含原始 Search-R1 数字，作为“只看结果的参考区间”。这不是主要因果对比，因为 backbone 是 Qwen2.5 而不是 Qwen3；但它可以防止我们接受一个答案/搜索行为远低于已知 Search-R1 水平的 DeepFactCite 模型。

已发布的 Search-R1 Table 2 报告了 7 个 QA 数据集上的 EM，设置为 Search-R1 协议、NQ+HotpotQA 训练、2018 Wikipedia、E5 检索器、top-3 检索段落。

报告中需要保留的参考数字：

| Method | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B RAG | 0.349 | 0.585 | 0.392 | 0.299 | 0.235 | 0.058 | 0.208 | 0.304 |
| Qwen2.5-7B Rejection Sampling | 0.360 | 0.592 | 0.380 | 0.331 | 0.296 | 0.123 | 0.355 | 0.348 |
| Qwen2.5-7B Search-R1-base | 0.480 | 0.638 | 0.457 | 0.433 | 0.382 | 0.196 | 0.432 | 0.431 |
| Qwen2.5-7B Search-R1-instruct | 0.393 | 0.610 | 0.397 | 0.370 | 0.414 | 0.146 | 0.368 | 0.385 |
| Qwen2.5-3B Search-R1-base | 0.406 | 0.587 | 0.435 | 0.284 | 0.273 | 0.049 | 0.088 | 0.303 |
| Qwen2.5-3B Search-R1-instruct | 0.341 | 0.545 | 0.378 | 0.324 | 0.319 | 0.103 | 0.264 | 0.325 |

来源参考：

```text
Paper: https://arxiv.org/abs/2503.09516
HTML table: https://arxiv.org/html/2503.09516
HF Search-R1 collection: https://huggingface.co/collections/PeterJinGo/search-r1
HF Search-R1-v0.3 collection: https://huggingface.co/collections/PeterJinGo/search-r1-v03
HF nq_hotpotqa_train dataset: https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train
```

这张表的使用方式：

```text
把它作为外部背景：
  “原始 Search-R1 Qwen2.5-7B-base 报告的平均 EM 是 0.431。”

不要把它作为主要胜负结论：
  除非我们运行相同模型家族、相同检索器、相同数据、相同 prompt、
  相同指标和相同运行时，否则“DeepFactCite 击败 Search-R1”是不成立的。

把它作为 sanity check：
  如果 DeepFactCite Qwen3-8B 在 NQ/HotpotQA 风格答案/搜索评测上崩掉，
  只有引用提升并不足以支撑项目 claim。
```

## SFT 失败记录要求

每一次 SFT 相关失败或未获胜结果，都必须先记录到
`docs/deepfactcite_reproducibility_issue_log.md`，然后才能启动下一次昂贵运行。

必填字段：

```text
failure id
date
hypothesis
model/input artifacts
training command or eval command
serving path
dataset/corpus
observed metrics
what failed
root cause or current best explanation
evidence used to prove or narrow the cause
decision: discard / keep as ablation / rerun / fix code
guardrail added for future runs
```

最终报告中必须保留的当前 SFT 经验：

```text
1. Strict SFT 不会自动成为赢家。
   高质量过滤能提升数据干净程度，但可能降低多样性、搜索行为覆盖或答案鲁棒性。
   只有在相同 eval 设置下击败 soft SFT，才能把它作为主 checkpoint。

2. Dynamic LoRA 和 merged full model 默认不能互换。
   旧 bf16 merge 会显著改变 logits，所以继续训练前必须用 HF logits check
   验证 merged checkpoint。

3. 从坏 parent 继续训练会让下游结果失效。
   mix50 虽然完成了训练，但它从有损 bf16 merged parent 开始，
   所以不是有效的 continuation baseline。

4. 解释数字之前，必须先确认数据集指标是否适用。
   DeepFactCite strict47 没有 gold answer，所以该集合上的 answer_subem
   没有意义。答案/搜索防护评测必须使用带 gold answer 的 ShortQA 或
   Search-R1 NQ/HotpotQA 风格数据。

5. Serving path 是模型定义的一部分。
   即使 temperature=0，vLLM bf16 下的 `Base + LoRA`、bf16 merged full model
   和 fp32 merged full model 也可能产生不同的闭环搜索轨迹。
```

## 与 Search-R1 对齐的步骤

### 1. 数据准备

Search-R1 步骤：构建带 prompts 和规则奖励的 train/test parquet 文件。

DeepFactCite 步骤：

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite
python scripts/deepfactcite/prepare_data.py \
  --deepcitefact-dir /root/autodl-tmp/DeepCiteFact \
  --output-dir data/deepfactcite_strict \
  --min-sft-url-validity 1.0 \
  --min-sft-claim-support 0.35 \
  --max-sft-unsupported-rate 0.5 \
  --min-sft-citations 1

python scripts/deepfactcite/prepare_agentic_eval_data.py \
  --output-dir data/agentic_eval48
```

当前已准备数据：

```text
data/deepfactcite/sft/train.parquet 1140
data/deepfactcite/sft/test.parquet 60
data/deepfactcite/rl/train.parquet 1140
data/deepfactcite/rl/test.parquet 60
data/agentic_eval48/rl/test.parquet 10

strict regenerated data:
data/deepfactcite_strict/sft/train.parquet 900
data/deepfactcite_strict/sft/test.parquet 47
data/deepfactcite_strict/rl/train.parquet 1140
data/deepfactcite_strict/rl/test.parquet 60
```

strict SFT 过滤器只保留满足以下条件的轨迹：URL 从当前轨迹中复制而来，claim-level support >= 0.35，unsupported citation rate <= 0.5，并且至少有 1 条引用。2026-05-18 这一步从 2916 条源 SFT 轨迹中保留了 947 条。

### 2. 检索器

Search-R1 步骤：启动检索服务器，并把搜索结果注入轨迹。

DeepFactCite 步骤：使用相同形状的 retrieve API，但保留返回证据中的 URL，这样才能衡量引用真实性。

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite
CORPUS=data/deepfactcite/corpus.jsonl PORT=8000 \
  bash scripts/deepfactcite/start_lexical_retriever.sh
```

### 3. 基础模型评测

Search-R1 步骤：训练前先评测 base/search agent。

DeepFactCite 步骤：在 SFT 之前，让 base Qwen3-8B 跑短任务和长答案/引用任务，收集坏例并建立冷启动 baseline。

先前 `agentic-rl-searchqa` 仓库中的当前命令：

```bash
cd /root/autodl-tmp/agentic-rl-searchqa

PYTHON_BIN=/root/autodl-tmp/conda_envs/openrlhf_vllm085/bin/python \
BASE_MODEL_PATH=/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base \
MODEL_PATH=/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base \
ADAPTER_PATH=/root/autodl-tmp/agentic-rl-searchqa/.cache/no_lora_adapter_for_base_eval \
BASE_SERVED_MODEL_NAME=Qwen3-8B-Base \
SERVED_MODEL_NAME=Qwen3-8B-Base \
OUTPUT_PREFIX=base_qwen3_8b \
EVAL_TAG=eval48_ddgscorpus_top3_20260518 \
SEARCH_BACKEND=offline \
SEARCH_CORPUS=data/offline/eval_ckpt400_ddgs_corpus.jsonl \
SEARCH_TOP_K=3 \
MAX_SEARCHES=3 \
SHORTQA_DATASET=data/eval/ckpt400_shortqa_32.jsonl \
DEEPCITE_DATASET=data/eval/ckpt400_deepcite_16.jsonl \
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_TP_SIZE=2 \
PORT=8000 \
VLLM_LOG=$PWD/logs/vllm_base_qwen3_8b_eval48_ddgscorpus_top3_20260518_tp2.log \
bash scripts/eval_sft_checkpoint.sh
```

预期输出：

```text
data/mixed/base_qwen3_8b_short_rollouts_eval48_ddgscorpus_top3_20260518.jsonl
data/mixed/base_qwen3_8b_deepcite_rollouts_eval48_ddgscorpus_top3_20260518.jsonl
reports/eval_base_qwen3_8b_short_eval48_ddgscorpus_top3_20260518.json
reports/eval_base_qwen3_8b_deepcite_eval48_ddgscorpus_top3_20260518.json
reports/eval_base_qwen3_8b_citation_eval48_ddgscorpus_top3_20260518.json
reports/score_base_qwen3_8b_short_eval48_ddgscorpus_top3_20260518.json
reports/score_base_qwen3_8b_deepcite_eval48_ddgscorpus_top3_20260518.json
```

### 4. LoRA SFT 冷启动

Search-R1 步骤：训练 policy 遵循搜索轨迹格式。

DeepFactCite 步骤：在引用可信轨迹上 SFT Qwen3，让模型在学习 URL 引用格式的同时保留答案/搜索行为。

主要 8B 运行：

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/searchr1-qwen3-sft

WANDB_MODE=offline \
OUTPUT_ROOT=$PWD/outputs/deepfactcite \
DATA_DIR=data/deepfactcite_strict/sft \
EXPERIMENT_NAME=deepfactcite-sft-qwen3-8b-lora-strict-100 \
MODEL_SIZE=8B N_GPUS=2 TOTAL_STEPS=100 \
bash scripts/deepfactcite/train_sft_qwen3.sh
```

当前 SFT 候选运行：

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/searchr1-qwen3-sft

WANDB_MODE=offline \
bash scripts/deepfactcite/launch_mix_clean_sft_200.sh
```

这次训练直接从 Base Qwen3-8B 开始，而不是从旧 merged Soft parent 开始。
得到的 adapter 是：

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200/global_step_200
```

如果 8B 运行被阻塞超过 1 小时：

```bash
WANDB_MODE=offline \
OUTPUT_ROOT=$PWD/outputs/deepfactcite \
MODEL_SIZE=4B N_GPUS=1 TOTAL_STEPS=100 \
bash scripts/deepfactcite/train_sft_qwen3.sh
```

### 5. Base vs SFT 评测

Search-R1 步骤：训练后评测答案/搜索性能。

DeepFactCite 步骤：同时比较 base 和 SFT 的答案指标与引用指标。

本地评测 runner：

```bash
PYTHON_BIN=/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python
BASE_MODEL=/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base
STRICT_SFT=$PWD/outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-strict-100/global_step_100

$PYTHON_BIN scripts/deepfactcite/eval_citation_agent.py \
  --model "$BASE_MODEL" \
  --data data/deepfactcite_strict/sft/test.parquet \
  --corpus data/deepfactcite_strict/corpus.jsonl \
  --output-jsonl outputs/eval/base_qwen3_8b_deepfactcite_strict_test.jsonl \
  --report-json reports/base_qwen3_8b_deepfactcite_strict_test.json \
  --temperature 0.0 --max-turns 4 --topk 3

$PYTHON_BIN scripts/deepfactcite/eval_citation_agent.py \
  --model "$BASE_MODEL" \
  --adapter "$STRICT_SFT" \
  --data data/deepfactcite_strict/sft/test.parquet \
  --corpus data/deepfactcite_strict/corpus.jsonl \
  --output-jsonl outputs/eval/sft_qwen3_8b_deepfactcite_strict_test.jsonl \
  --report-json reports/sft_qwen3_8b_deepfactcite_strict_test.json \
  --temperature 0.0 --max-turns 4 --topk 3
```

主要指标：

```text
answer_subem / answer_exact
citation_presence
url_validity
citation_precision
claim_support
unsupported_citation_rate
fake_url_rate
avg_search_turns
avg_response_tokens
```

表格：

```text
Model | Answer | URL Validity | Citation Precision | Claim Support | Unsupported Rate | Avg Search Turns
Base Qwen3-8B
DeepFactCite SFT Qwen3-8B
```

当前固定 vLLM SFT 评测，2026-05-18：

```text
Serving:
  vLLM dynamic LoRA, bf16, tensor parallel 2
  base model: /root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base
  adapters: soft100 and mixclean200

Eval:
  temperature 0.0
  max_turns 4
  topk 3
```

ShortQA32：

| Model | Answer | Total | URL Validity | Citation Precision | Claim Support | Unsupported Rate | Search Turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B | 0.219 | 0.189 | 0.031 | 0.038 | 0.031 | 0.906 | 2.625 |
| Soft SFT 100 | 0.438 | 0.391 | 0.812 | 0.292 | 0.292 | 0.557 | 1.156 |
| MixClean SFT 200 | 0.500 | 0.427 | 0.969 | 0.333 | 0.333 | 0.495 | 1.125 |

Strict47 引用评测：

| Model | Total | Cite Presence | URL Validity | Citation Precision | Claim Support | Unsupported Rate | Fake URL | Search Turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B | 0.120 | 0.064 | 0.064 | 0.062 | 0.021 | 0.574 | 0.000 | 1.745 |
| Soft SFT 100 | 0.137 | 0.383 | 0.319 | 0.070 | 0.064 | 0.851 | 0.064 | 1.000 |
| MixClean SFT 200 | 0.142 | 0.489 | 0.418 | 0.074 | 0.072 | 0.879 | 0.071 | 1.234 |

解释：

- 在固定 dynamic LoRA serving path 下，MixClean SFT 200 是当前最好的 SFT 候选。
- 它在 ShortQA32 上优于 Soft100，并且在 Strict47 上改善了 URL/support。
- 它还没有解决 unsupported/fake citation。Strict47 上 unsupported/fake rate 仍然很高，所以下一阶段必须在 reward 阶段加入引用真实性和声明支持约束。
- Strict47 没有 gold answer；这个集合上的 answer_subem 没有意义。

Search-R1 core BM25 防护评测，2026-05-18：

```text
Dataset: data/searchr1_core_guardrail/test.parquet
Rows:    200 = NQ 100 + HotpotQA 100
Retrieval: wiki-18 BM25 + extracted wiki_dump.jsonl
Serving: same vLLM dynamic-LoRA bf16 path
```

| Model | Answer subEM | Exact | Search Success | Search Turns | Empty Answer | Budget Fail | NQ subEM | HotpotQA subEM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B | 0.290 | 0.135 | 0.960 | 1.720 | 0.350 | 0.170 | 0.330 | 0.250 |
| Soft SFT 100 | 0.450 | 0.060 | 0.990 | 1.195 | 0.075 | 0.045 | 0.490 | 0.410 |
| MixClean SFT 200 | 0.490 | 0.030 | 1.000 | 1.160 | 0.045 | 0.025 | 0.500 | 0.480 |

结论：

- MixClean SFT 200 通过了本地 Search-R1 风格答案/搜索防护评测。
- 它是当前 SFT baseline 候选，也很可能作为 GRPO 初始化。
- 这些数字是 BM25 控制条件下的本地防护评测数字，不是官方 Search-R1 E5 复现数字。

### 6. GRPO 消融

Search-R1 步骤：用 outcome-only search QA reward 训练。

DeepFactCite 步骤：用相同 backbone/data/backend 跑两组消融。

```text
Outcome-only GRPO:
  answer reward on
  search/format reward on
  citation/support reward off

Citation-aware GRPO:
  answer reward on
  search/format reward on
  citation authenticity reward on
  claim-support reward on
```

citation-aware GRPO 的目标权重：

```text
answer_weight=0.25
citation_weight=0.35
support_weight=0.25
format_weight=0.10
search_weight=0.05
cost_weight=0.05
```

### 7. Search-R1 Benchmark 防护评测

这一步是必要的，但它回答的问题和 DeepFactCite 引用 benchmark 不一样。

Search-R1 benchmark 的问题是：

```text
DeepFactCite 是否保持了原始 search-QA 能力？
```

DeepFactCite benchmark 的问题是：

```text
DeepFactCite 是否提升了 URL 真实性和声明级引用支持？
```

不要把这两个问题合并成一个指标。Search-R1 benchmark 用作答案/搜索行为的防护评测；DeepFactCite 则作为引用可信行为的主要证据。

#### 官方 Search-R1 设置

仓库的 NQ/HotpotQA 脚本使用：

```text
train: nq, hotpotqa
test:  nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle
retriever: wiki-18 corpus + e5 index
protocol: <think>, <search>, <information>, <answer>
metric: exact-match reward over <answer>
```

本地入口：

```text
scripts/nq_hotpotqa/README.md
scripts/nq_hotpotqa/data_process.sh
scripts/nq_hotpotqa/evaluate.sh
scripts/data_process/qa_search_train_merge.py
scripts/data_process/qa_search_test_merge.py
verl/trainer/main_ppo.py
verl/utils/reward_score/qa_em.py
```

2026-05-18 的当前本地状态：

```text
data/nq_hotpotqa_train is not present.
data/ only contains DeepFactCite, agentic_eval48, and shortqa_guardrail.
/root/autodl-tmp has about 17G free, so do not download wiki-18/e5 blindly.
```

#### 最小公平对比矩阵

所有行都必须在相同 prompt、相同检索器、相同 decoding、相同 max turns 和相同 evaluator 下运行。

```text
Model/Policy                              Purpose
Base Qwen3-8B                             same-backbone base guardrail
DeepFactCite Soft SFT 100 dynamic LoRA    current valid SFT baseline
Future valid DeepFactCite continuation    only if trained from a reproducible parent
Outcome-only GRPO Qwen3-8B                Search-R1-style ablation, if backend is ready
Citation-aware GRPO Qwen3-8B              final DeepFactCite ablation, if backend is ready
```

只有在使用原生 Qwen2.5/Llama backbone 和兼容 runtime 评测时，原始 Search-R1 发布 checkpoint 才能作为外部参考报告。它们不能作为 Qwen3-8B 的主要因果 baseline，因为 backbone、tokenizer、checkpoint family，有时甚至 runtime 都不同。

#### 数据集层级

Tier 0：本地免下载防护评测。

```text
data/shortqa_guardrail/rl/test.parquet
data/agentic_eval48/rl/test.parquet
```

只把它用于快速迭代。它不是官方 Search-R1 benchmark。

Tier 1：Search-R1 核心子集。

```text
NQ:       100 to 500 examples
HotpotQA: 100 to 500 examples
```

这是第一个真正与 Search-R1 对齐的 benchmark。NQ 检查单跳 search-QA，HotpotQA 检查多跳答案/搜索行为。

Tier 2：Search-R1 完整 eval mix。

```text
nq
triviaqa
popqa
hotpotqa
2wikimultihopqa
musique
bamboogle
```

只有在 Tier 1 路径稳定后再使用它。它是更适合公开展示的表格，但需要更多数据管线、更长评测时间和可靠的检索器设置。

#### 指标

Search-R1 防护评测指标：

```text
answer_exact
answer_subem
format_success
search_success
avg_search_turns
avg_response_tokens
empty_answer_rate
budget_fail_rate
```

DeepFactCite 引用指标：

```text
citation_presence
url_validity
citation_precision
claim_support
unsupported_citation_rate
fake_url_rate
avg_search_turns
avg_response_tokens
```

验收目标：

```text
在 Search-R1 NQ/HotpotQA 上：
  answer_subem >= same-backbone base - 1 到 2 个点
  avg_search_turns 不暴涨
  empty/budget-fail rate 不明显升高

在 DeepFactCite 上：
  URL validity 明显提升
  citation precision 明显提升
  claim support 明显提升
  unsupported citation rate 明显下降
```

#### 立即下一步

使用 GPU 时间之前：

```text
1. 检查 data/nq_hotpotqa_train 能否在不超磁盘的情况下下载。
2. 如果不能，从 FlashRAG 数据集生成小型 NQ/HotpotQA parquet 子集。
3. 在同一个 Tier 1 子集上运行 Base Qwen3-8B、Soft SFT 100 和 MixClean SFT 200。
   BM25 控制的本地防护评测已完成。
4. 保存逐样本 rollouts 和聚合报告。已在 `reports/*searchr1_core_bm25_200*` 中完成。
5. 只有当 Search-R1 防护评测没有严重退化时，才把 MixClean SFT 200 提升为 SFT baseline。
   对 BM25 控制的防护评测来说，这一步已完成。

下一步：

使用 MixClean SFT 200 作为 GRPO 的 SFT 初始化候选。
在 Qwen3 兼容 backend 下运行 outcome-only vs citation-aware GRPO。
除非 E5 设置被完全复现，否则原始 Search-R1 论文数字只作为外部参考。
```

这样可以让简历 claim 更站得住：

```text
DeepFactCite 在 NQ/HotpotQA 上保持了 Search-R1 风格的答案/搜索行为，
同时在面向引用的评测上提升了引用真实性和支持性。
```
