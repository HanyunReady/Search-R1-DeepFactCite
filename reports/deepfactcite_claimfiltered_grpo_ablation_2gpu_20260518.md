# DeepFactCite Claim-Filtered GRPO 2-GPU Ablation

Date: 2026-05-18

## Question

Does citation-aware reward improve citation authenticity and claim support over
outcome-only reward when model, data, retriever, rollout stack, and step budget
are held fixed?

This is the gate before any longer or 4-GPU run. The experiment is intentionally
small: 16 training steps, 32 sampled trajectories per mode, no checkpoint save.

## Fixed Setup

Model:

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16
```

Data:

```text
data/deepfactcite_sglang_grpo_claim_filtered/train.parquet
data/deepfactcite_sglang_grpo_claim_filtered/test.parquet
data/deepfactcite_sglang_grpo_claim_filtered/corpus.jsonl
```

Rollout stack:

```text
SGLang backend
2 x A800 80G
tensor parallel = 2
rollout.n = 2
train_batch_size = 1
total_training_steps = 16
SEARCHQA_TOPK = 2
max_tool_response_length = 768
max_response_length = 384
reward_manager = deepfactcite_custom
save_freq = 0
```

Reward weights:

| Mode | Answer | Citation | Support | Format | Search | Cost |
|---|---:|---:|---:|---:|---:|---:|
| outcome-only | 0.80 | 0.00 | 0.00 | 0.15 | 0.10 | 0.05 |
| citation-aware | 0.15 | 0.35 | 0.30 | 0.10 | 0.05 | 0.05 |

Commands:

```bash
RUN_TAG=20260518_claimfiltered_2gpu MODE=outcome-only DRY_RUN=0 GRPO_TOTAL_STEPS=16 \
  bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh

RUN_TAG=20260518_claimfiltered_2gpu MODE=citation-aware DRY_RUN=0 GRPO_TOTAL_STEPS=16 \
  bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

Artifacts:

```text
logs/dfc-mixclean200-claimfiltered-outcome-only-20260518_claimfiltered_2gpu.log
logs/dfc-mixclean200-claimfiltered-citation-aware-20260518_claimfiltered_2gpu.log
logs/grpo/rollouts/dfc-mixclean200-claimfiltered-outcome-only-20260518_claimfiltered_2gpu/
logs/grpo/rollouts/dfc-mixclean200-claimfiltered-citation-aware-20260518_claimfiltered_2gpu/
reports/dfc_mixclean200_claimfiltered_outcome_only_2gpu_rollout_summary.md
reports/dfc_mixclean200_claimfiltered_citation_aware_2gpu_rollout_summary.md
```

## Result

Important: `reward` is not directly comparable across modes because the reward
weights differ. The comparable metrics are the diagnostic rollout metrics below.

| Metric | Outcome-Only | Citation-Aware | Better |
|---|---:|---:|---|
| samples | 32 | 32 | same |
| search | 1.0000 | 1.0000 | same |
| format | 0.9812 | 0.9750 | outcome-only |
| answer_subem | 0.0000 | 0.0312 | citation-aware |
| URL validity | 0.7656 | 0.6979 | outcome-only |
| citation precision | 0.4219 | 0.3828 | outcome-only |
| claim support | 0.4219 | 0.3828 | outcome-only |
| unsupported citation rate | 0.4219 | 0.4844 | outcome-only |
| fake URL rate | 0.0469 | 0.0521 | outcome-only |
| citation count | 1.0625 | 1.0000 | neutral |
| mean response clip ratio across steps | 0.1563 | 0.2813 | outcome-only |

Failure reason counts:

| Reason | Outcome-Only | Citation-Aware |
|---|---:|---:|
| weak_claim_support | 9 | 7 |
| unsupported_citation | 9 | 8 |
| no_citation | 6 | 8 |
| invalid_or_fake_url | 3 | 2 |

## Interpretation

Citation-aware did not pass the promotion gate in this exact setup. It kept
search behavior intact, but it did not improve URL validity, citation precision,
claim support, unsupported citation rate, or fake URL rate over outcome-only.

The likely cause is not the 2-GPU engineering chain. Both runs completed, loaded
`deepfactcite_custom`, wrote rollout JSONL, searched successfully, and did not
save checkpoints. The more likely cause is a data/prompt mismatch:

```text
The filtered data guarantees that one selected claim is supported by a snippet,
but some original user questions are broad. The model tries to answer the broad
question, adds extra facts, dates, causes, or background, and then the local
claim around the citation becomes broader than the snippet supports.
```

Example failure pattern:

```text
Question asks for a player's full organizational progression.
Selected evidence supports one narrow fact about draft or junior career.
Model answers the broad progression question anyway.
URL may be valid, but the cited local claim is not fully supported.
```

A second issue is response length. The citation-aware run had higher step-level
clip ratio at `max_response_length=384`:

```text
outcome-only mean step clip ratio = 0.1563
citation-aware mean step clip ratio = 0.2813
```

This can hide the final answer or truncate citations, so it is a training-risk
metric even when reward looks non-zero.

## Decision

Do not switch to 4 GPUs from this result.

Do not launch a longer GRPO run on the same v1 claim-filtered data.

The next useful experiment is a stricter one-claim / one-citation dataset that
reduces broad-question over-answering:

```text
data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite
```

v2 changes:

```text
max_citations = 1
max_claim_tokens = 30
max_query_tokens = 18
max_answer_sentences = 1
strict one-citation prompt = true
same retriever top-k validation = 28/28 train, 4/4 test
```

The expected signal is narrower: first prove that citation-aware reward can
increase supported one-claim answers under a prompt that makes unsupported
expansion less attractive. Only after that should a fair v2 outcome-only vs
citation-aware ablation or longer run be considered.
