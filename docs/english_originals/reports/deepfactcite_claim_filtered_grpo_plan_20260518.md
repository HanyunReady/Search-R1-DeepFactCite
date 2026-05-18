# DeepFactCite Claim-Filtered GRPO Plan

Date: 2026-05-18

## Decision

Do not launch a long GRPO run from the retrieval-hit smoke result alone.

The next useful experiment is a controlled 2-GPU ablation on the same
claim-level support-filtered data:

1. outcome-only GRPO: answer/search/format reward only;
2. citation-aware GRPO: answer/search/format plus URL authenticity and
   claim-support reward.

This keeps GPU time tied to a falsifiable question: can citation-aware reward
improve citation validity/support without materially damaging answer/search
behavior?

## Data Built

Script:

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python \
  scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py \
  --rows 32 \
  --out-dir data/deepfactcite_sglang_grpo_claim_filtered
```

Artifacts:

```text
data/deepfactcite_sglang_grpo_claim_filtered/train.parquet
data/deepfactcite_sglang_grpo_claim_filtered/test.parquet
data/deepfactcite_sglang_grpo_claim_filtered/corpus.jsonl
data/deepfactcite_sglang_grpo_claim_filtered/summary.json
data/deepfactcite_sglang_grpo_claim_filtered/preview.jsonl
data/deepfactcite_sglang_grpo_claim_filtered/reject_samples.jsonl
```

Build summary:

| Item | Value |
|---|---:|
| Source | `data/deepfactcite_mix/sft/train.parquet` |
| Kept rows | 32 |
| Train rows | 28 |
| Test rows | 4 |
| Corpus docs | 42 |
| Avg selected citations | 1.3125 |
| Min support score | 1.0 |
| Avg support score | 1.0 |

Reject reasons while collecting the 32 rows:

| Reason | Count |
|---|---:|
| weak_claim_support | 54 |
| too_few_supported_claims | 5 |

Filters enforced:

```text
1-2 citations per row
retrieved URL only
no truncated/fake citation URL
claim support score >= 1.0
claim length <= 42 support tokens
low-information citation fragments filtered
answer prompt capped at 1-3 concise sentences
```

## Retriever Validation

Validation used the same offline retriever class as the SGLang rollout path:

```text
agentic_rl_searchqa.tools.offline_search.OfflineSearchTool
SEARCHQA_TOPK=2
```

Result:

| Split | Rows | top-2 URL hit rows |
|---|---:|---:|
| train | 28 | 28 |
| test | 4 | 4 |

This is stronger than the previous retrieval-hit smoke set: it checks not only
that snippets exist, but that the selected cited URL is retrievable for the
training query under the same corpus and top-k setting.

## 2-GPU Ablation Entrypoint

Script:

```bash
scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

The script defaults to `DRY_RUN=1`. It checks data, model, corpus, disk, and GPU
state, then prints the exact command. Training starts only with `DRY_RUN=0`.

Outcome-only run:

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite
MODE=outcome-only DRY_RUN=0 \
  bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

Citation-aware run:

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite
MODE=citation-aware DRY_RUN=0 \
  bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

Common settings:

```text
actor = outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16
rollout backend = SGLang
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

Reward weights:

| Mode | Answer | Citation | Support | Format | Search | Cost |
|---|---:|---:|---:|---:|---:|---:|
| outcome-only | 0.80 | 0.00 | 0.00 | 0.15 | 0.10 | 0.05 |
| citation-aware | 0.15 | 0.35 | 0.30 | 0.10 | 0.05 | 0.05 |

Disk note: `/root/autodl-tmp` has about 14G free, so checkpoint saving remains
disabled until cleanup is done.

## Success / Stop Criteria

Use the same eval and rollout parsing for both modes. The ablation is useful
only if it reports both wins and failures.

Primary citation metrics:

```text
url_validity
citation_precision
claim_support
unsupported_citation_rate
fake_url_rate
avg_citations
```

Guardrail metrics:

```text
answer_subem or target subEM on claim-filtered rows
search reward / search turns
format reward
response_length clip ratio
budget-fail / no-search cases
```

Promotion rule:

```text
Move to longer or 4-GPU runs only if citation-aware improves URL/support metrics
over outcome-only while answer/search behavior does not collapse.
```

Stop rule:

```text
If citation-aware mostly increases citation count without reducing unsupported
rate, or if response truncation remains high at 384 tokens, do not scale GPU.
Fix prompt/data/length first.
```

## External Positioning

This project should be framed as Search-R1-style RL with citation authenticity
and claim-support rewards, not as a direct claim of beating the original
Search-R1 paper unless the original setup is reproduced.

Useful external reference points:

- Search-R1 trains LLMs with RL to use search during reasoning and reports
  multi-dataset QA gains with outcome-style rewards:
  https://arxiv.org/abs/2503.09516
- ALCE established automatic evaluation for long-form answers with citations,
  including correctness and citation quality:
  https://arxiv.org/abs/2305.14627
- Recent attribution work argues that answer correctness and citation
  faithfulness are separable, which supports this project's outcome-only vs
  citation-aware ablation:
  https://arxiv.org/abs/2412.18004
- OpenAI's Deep Research system card is a product-level reference for why
  search, source interpretation, and evidence synthesis matter in practical
  agentic research systems:
  https://openai.com/index/deep-research-system-card/

The interview-grade claim is therefore:

```text
I reproduced a Search-R1-style 2-GPU SGLang/GRPO path, identified that retrieval
hit alone did not solve citation support, built claim-level support-filtered
RL data, and designed a controlled reward ablation to isolate whether explicit
citation authenticity/support rewards improve attribution without degrading
answer/search behavior.
```
