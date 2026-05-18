# DeepFactCite Search-R1-Aligned Experiment Plan

This plan keeps the original Search-R1 workflow structure while extending the
objective from answer correctness to answer correctness plus credible citations.

## Project Success Criteria

The project is not successful merely because a training run finishes. It is
successful only if both conditions hold:

```text
1. Evaluation target is met:
   - Search-R1-style answer/search benchmark does not regress materially.
   - DeepFactCite citation metrics improve materially.

2. Engineering learning is auditable:
   - Every failed or non-winning SFT/merge/eval attempt is recorded.
   - Each record includes the hypothesis, exact artifact paths, eval setup,
     observed metrics, root-cause analysis, and the next guardrail.
```

This failure record is part of the final deliverable. It should be good enough
to answer a senior interview question such as:

```text
What failed during SFT, why did it fail, how did you prove the root cause,
and what did you change so the result became reproducible?
```

## Positioning

Do not claim that DeepFactCite "beats Search-R1" unless a strict original
benchmark baseline is reproduced. The intended claim is:

DeepFactCite extends the Search-R1-style search agent objective with URL
authenticity and claim-support rewards, improving citation validity/support
while preserving answer/search performance.

## External Search-R1 References

Use these only as reference points, not as the primary causal baseline:

```text
PeterJinGo/Search-R1 and Search-R1-v0.3 collections:
  original released checkpoints are Qwen2.5/Llama3.2 based, not Qwen3.

orbit-ai/searchr1-repro-4b:
  Qwen3-4B Search-R1-style GRPO reproduction on NQ/HotpotQA, 200 steps,
  <search>/<information> tool protocol.

jiulaikankan/Qwen3-4B-Thinking-Search-R1-baseline:
  small Qwen3-4B-Thinking adapter/checkpoint from OpenRLHF-Agent/Search-R1,
  useful for sanity checking but not enough as the main 8B baseline.
```

For a fair 8B comparison, the main baseline remains same backbone, same
retriever, same prompt/data, and only the reward objective changed:
outcome-only/Search-R1-style versus citation-aware DeepFactCite.

## Original Search-R1 Result Reference

The final report should include original Search-R1 numbers as a result-only
reference band. This is not the main causal comparison because the backbone is
Qwen2.5 rather than Qwen3, but it prevents us from accepting a DeepFactCite
model whose answer/search behavior is far below the known Search-R1 level.

Published Search-R1 Table 2 reports EM on seven QA datasets using the Search-R1
protocol, NQ+HotpotQA training, 2018 Wikipedia, E5 retriever, and top-3
retrieved passages.

Reference numbers to keep in the report:

| Method | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B RAG | 0.349 | 0.585 | 0.392 | 0.299 | 0.235 | 0.058 | 0.208 | 0.304 |
| Qwen2.5-7B Rejection Sampling | 0.360 | 0.592 | 0.380 | 0.331 | 0.296 | 0.123 | 0.355 | 0.348 |
| Qwen2.5-7B Search-R1-base | 0.480 | 0.638 | 0.457 | 0.433 | 0.382 | 0.196 | 0.432 | 0.431 |
| Qwen2.5-7B Search-R1-instruct | 0.393 | 0.610 | 0.397 | 0.370 | 0.414 | 0.146 | 0.368 | 0.385 |
| Qwen2.5-3B Search-R1-base | 0.406 | 0.587 | 0.435 | 0.284 | 0.273 | 0.049 | 0.088 | 0.303 |
| Qwen2.5-3B Search-R1-instruct | 0.341 | 0.545 | 0.378 | 0.324 | 0.319 | 0.103 | 0.264 | 0.325 |

Source references:

```text
Paper: https://arxiv.org/abs/2503.09516
HTML table: https://arxiv.org/html/2503.09516
HF Search-R1 collection: https://huggingface.co/collections/PeterJinGo/search-r1
HF Search-R1-v0.3 collection: https://huggingface.co/collections/PeterJinGo/search-r1-v03
HF nq_hotpotqa_train dataset: https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train
```

How to use this table:

```text
Use it as external context:
  "Original Search-R1 Qwen2.5-7B-base reports 0.431 average EM."

Do not use it as the main win/loss claim:
  "DeepFactCite beats Search-R1" is not valid unless we run the same model
  family, same retriever, same data, same prompt, same metric, and same runtime.

Use it as a sanity check:
  If DeepFactCite Qwen3-8B collapses on NQ/HotpotQA-style answer/search eval,
  citation gains alone are not enough for the project claim.
```

## SFT Failure Record Requirement

Every SFT-related failure or non-winning result must be logged in
`docs/deepfactcite_reproducibility_issue_log.md` before launching the next
expensive run.

Required fields:

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

Current SFT lessons that must remain visible in the final report:

```text
1. Strict SFT is not automatically the winner.
   High-quality filtering improves data cleanliness but can reduce diversity,
   search behavior coverage, or answer robustness. It must beat soft SFT under
   the same eval setup before being used as the main checkpoint.

2. Dynamic LoRA and merged full model are not interchangeable by default.
   The old bf16 merge changed logits materially, so merged checkpoints must be
   validated with HF logits checks before continuation training.

3. Continuation from a bad parent invalidates downstream results.
   mix50 completed training, but because it started from the lossy bf16 merged
   parent, it is not a valid continuation baseline.

4. Dataset metric applicability must be checked before interpreting numbers.
   DeepFactCite strict47 has no gold answer, so answer_subem from that set is
   not meaningful. Answer/search guardrail must use ShortQA or Search-R1
   NQ/HotpotQA-style data with gold answers.

5. Serving path is part of the model definition.
   `Base + LoRA` under vLLM bf16, bf16 merged full model, and fp32 merged full
   model can produce different closed-loop search trajectories even with
   temperature=0.
```

## Search-R1-Aligned Steps

### 1. Data Preparation

Search-R1 step: build train/test parquet files with prompts and rule rewards.

DeepFactCite step:

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

Current prepared data:

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

The strict SFT filter keeps only traces with copied in-trajectory URLs,
claim-level support >= 0.35, unsupported citation rate <= 0.5, and at least
one citation. On 2026-05-18 this kept 947 of 2916 source SFT traces.

### 2. Retriever

Search-R1 step: launch a retrieval server and inject search results into the
trajectory.

DeepFactCite step: use the same retrieve API shape, but preserve URLs in the
returned evidence so citation authenticity is measurable.

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite
CORPUS=data/deepfactcite/corpus.jsonl PORT=8000 \
  bash scripts/deepfactcite/start_lexical_retriever.sh
```

### 3. Base Evaluation

Search-R1 step: evaluate the base/search agent before training.

DeepFactCite step: run base Qwen3-8B on short and long/citation tasks before
SFT, to collect bad cases and establish a cold-start baseline.

Current command in the previous `agentic-rl-searchqa` repo:

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

Expected outputs:

```text
data/mixed/base_qwen3_8b_short_rollouts_eval48_ddgscorpus_top3_20260518.jsonl
data/mixed/base_qwen3_8b_deepcite_rollouts_eval48_ddgscorpus_top3_20260518.jsonl
reports/eval_base_qwen3_8b_short_eval48_ddgscorpus_top3_20260518.json
reports/eval_base_qwen3_8b_deepcite_eval48_ddgscorpus_top3_20260518.json
reports/eval_base_qwen3_8b_citation_eval48_ddgscorpus_top3_20260518.json
reports/score_base_qwen3_8b_short_eval48_ddgscorpus_top3_20260518.json
reports/score_base_qwen3_8b_deepcite_eval48_ddgscorpus_top3_20260518.json
```

### 4. LoRA SFT Cold Start

Search-R1 step: train the policy to follow the search trajectory format.

DeepFactCite step: SFT Qwen3 on citation-faithful trajectories, preserving
answer/search behavior while learning URL citation format.

Primary 8B run:

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

Current SFT candidate run:

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/searchr1-qwen3-sft

WANDB_MODE=offline \
bash scripts/deepfactcite/launch_mix_clean_sft_200.sh
```

This trains directly from Base Qwen3-8B, not from the old merged Soft parent.
The resulting adapter is:

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200/global_step_200
```

If the 8B run is blocked for more than one hour:

```bash
WANDB_MODE=offline \
OUTPUT_ROOT=$PWD/outputs/deepfactcite \
MODEL_SIZE=4B N_GPUS=1 TOTAL_STEPS=100 \
bash scripts/deepfactcite/train_sft_qwen3.sh
```

### 5. Base vs SFT Evaluation

Search-R1 step: evaluate answer/search performance after training.

DeepFactCite step: compare base and SFT on answer plus citation metrics.

Local eval runner:

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

Primary metrics:

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

Table:

```text
Model | Answer | URL Validity | Citation Precision | Claim Support | Unsupported Rate | Avg Search Turns
Base Qwen3-8B
DeepFactCite SFT Qwen3-8B
```

Current fixed-vLLM SFT evaluation, 2026-05-18:

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

ShortQA32:

| Model | Answer | Total | URL Validity | Citation Precision | Claim Support | Unsupported Rate | Search Turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B | 0.219 | 0.189 | 0.031 | 0.038 | 0.031 | 0.906 | 2.625 |
| Soft SFT 100 | 0.438 | 0.391 | 0.812 | 0.292 | 0.292 | 0.557 | 1.156 |
| MixClean SFT 200 | 0.500 | 0.427 | 0.969 | 0.333 | 0.333 | 0.495 | 1.125 |

Strict47 citation eval:

| Model | Total | Cite Presence | URL Validity | Citation Precision | Claim Support | Unsupported Rate | Fake URL | Search Turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B | 0.120 | 0.064 | 0.064 | 0.062 | 0.021 | 0.574 | 0.000 | 1.745 |
| Soft SFT 100 | 0.137 | 0.383 | 0.319 | 0.070 | 0.064 | 0.851 | 0.064 | 1.000 |
| MixClean SFT 200 | 0.142 | 0.489 | 0.418 | 0.074 | 0.072 | 0.879 | 0.071 | 1.234 |

Interpretation:

- MixClean SFT 200 is the current best SFT candidate under the fixed dynamic
  LoRA serving path.
- It passes ShortQA32 better than Soft100 and improves URL/support on Strict47.
- It does not solve unsupported/fake citation. Strict47 unsupported/fake rates
  remain high, so the next stage must use reward-time citation authenticity and
  claim-support constraints.
- Strict47 has no gold answer; answer_subem on that set is not meaningful.

Search-R1 core BM25 guardrail, 2026-05-18:

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

Decision:

- MixClean SFT 200 passes the local Search-R1-style answer/search guardrail.
- It is the current SFT baseline candidate and likely GRPO initialization.
- These numbers are BM25-controlled local guardrail numbers, not official
  Search-R1 E5 reproduction numbers.

### 6. GRPO Ablation

Search-R1 step: train with an outcome-only search QA reward.

DeepFactCite step: run two ablations with the same backbone/data/backend.

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

Target weights for citation-aware GRPO:

```text
answer_weight=0.25
citation_weight=0.35
support_weight=0.25
format_weight=0.10
search_weight=0.05
cost_weight=0.05
```

### 7. Search-R1 Benchmark Guardrail

This is necessary, but it answers a different question from the
DeepFactCite citation benchmark.

Search-R1 benchmark question:

```text
Did DeepFactCite preserve the original search-QA ability?
```

DeepFactCite benchmark question:

```text
Did DeepFactCite improve URL authenticity and claim-level citation support?
```

Do not merge these into one metric. Use the Search-R1 benchmark as a guardrail
for answer/search behavior, and use DeepFactCite as the primary evidence for
credible citation behavior.

#### Official Search-R1 Setup

The repository's NQ/HotpotQA script uses:

```text
train: nq, hotpotqa
test:  nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle
retriever: wiki-18 corpus + e5 index
protocol: <think>, <search>, <information>, <answer>
metric: exact-match reward over <answer>
```

Local entry points:

```text
scripts/nq_hotpotqa/README.md
scripts/nq_hotpotqa/data_process.sh
scripts/nq_hotpotqa/evaluate.sh
scripts/data_process/qa_search_train_merge.py
scripts/data_process/qa_search_test_merge.py
verl/trainer/main_ppo.py
verl/utils/reward_score/qa_em.py
```

Current local status on 2026-05-18:

```text
data/nq_hotpotqa_train is not present.
data/ only contains DeepFactCite, agentic_eval48, and shortqa_guardrail.
/root/autodl-tmp has about 17G free, so do not download wiki-18/e5 blindly.
```

#### Minimum Fair Comparison Matrix

Run all rows under the same prompt, same retriever, same decoding, same max
turns, and same evaluator.

```text
Model/Policy                              Purpose
Base Qwen3-8B                             same-backbone base guardrail
DeepFactCite Soft SFT 100 dynamic LoRA    current valid SFT baseline
Future valid DeepFactCite continuation    only if trained from a reproducible parent
Outcome-only GRPO Qwen3-8B                Search-R1-style ablation, if backend is ready
Citation-aware GRPO Qwen3-8B              final DeepFactCite ablation, if backend is ready
```

Original Search-R1 released checkpoints can be reported as an external reference
only when evaluated with their native Qwen2.5/Llama backbone and compatible
runtime. They are not the primary causal baseline for Qwen3-8B because the
backbone, tokenizer, checkpoint family, and sometimes runtime differ.

#### Dataset Tiers

Tier 0: local no-download guardrail.

```text
data/shortqa_guardrail/rl/test.parquet
data/agentic_eval48/rl/test.parquet
```

Use this for quick iteration only. It is not an official Search-R1 benchmark.

Tier 1: Search-R1 core subset.

```text
NQ:       100 to 500 examples
HotpotQA: 100 to 500 examples
```

This is the first real Search-R1-aligned benchmark. NQ checks single-hop
search-QA. HotpotQA checks multi-hop answer/search behavior.

Tier 2: Search-R1 full eval mix.

```text
nq
triviaqa
popqa
hotpotqa
2wikimultihopqa
musique
bamboogle
```

Use this only after the Tier 1 path is stable. It is the stronger public-facing
table, but it requires more data plumbing, more eval time, and a reliable
retriever setup.

#### Metrics

Search-R1 guardrail metrics:

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

DeepFactCite citation metrics:

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

Acceptance target:

```text
On Search-R1 NQ/HotpotQA:
  answer_subem >= same-backbone base - 1 to 2 points
  avg_search_turns does not spike
  empty/budget-fail rate does not increase materially

On DeepFactCite:
  URL validity improves materially
  citation precision improves materially
  claim support improves materially
  unsupported citation rate decreases materially
```

#### Immediate Next Step

Before using GPU time:

```text
1. Check whether data/nq_hotpotqa_train can be downloaded without exceeding disk.
2. If not, generate small NQ/HotpotQA parquet subsets from FlashRAG datasets.
3. Run Base Qwen3-8B, Soft SFT 100, and MixClean SFT 200 on the same Tier 1 subset. Done on BM25-controlled local guardrail.
4. Save per-example rollouts and aggregate reports. Done in `reports/*searchr1_core_bm25_200*`.
5. Promote MixClean SFT 200 as the SFT baseline only if the Search-R1 guardrail
   does not regress badly. Done for the BM25-controlled guardrail.

Next action:

```text
Use MixClean SFT 200 as the SFT initialization candidate for GRPO.
Run outcome-only vs citation-aware GRPO under the Qwen3-compatible backend.
Keep original Search-R1 paper numbers as external reference unless the E5 setup
is reproduced exactly.
```
```

This keeps the resume claim defensible:

```text
DeepFactCite preserves Search-R1-style answer/search behavior on NQ/HotpotQA
while improving citation authenticity/support on citation-focused evaluation.
```
