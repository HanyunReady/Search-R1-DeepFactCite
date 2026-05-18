# DeepFactCite Learning Index

Date: 2026-05-18

This is the recommended reading order for understanding the current
DeepFactCite/Search-R1 experiment from scratch. Read the files in this order
instead of jumping directly into scripts or raw logs.

## 1. Main Story

Start here:

```text
docs/deepfactcite_experiment_record_20260518.md
```

What it teaches:

- why the project exists;
- what SFT and GRPO are doing in this repo;
- why MixClean200 became the current SFT baseline;
- why the first GRPO goal is not a long run, but a controlled comparison;
- how v1, v2, parser-fix, and v3 prompt-fix changed the result.

Beginner checkpoint:

```text
If you can explain why "real URL" is not the same as "supported claim", you
understand the core problem.
```

## 2. Failure Ledger

Read next:

```text
docs/deepfactcite_reproducibility_issue_log.md
```

What it teaches:

- why completed training is not automatically a useful result;
- how a bad checkpoint parent can poison later experiments;
- why dynamic LoRA and merged checkpoints must be checked;
- how to record failed attempts so they still create project value.

Beginner checkpoint:

```text
If you can say why mix50 was not promoted even though it finished training, you
understand reproducibility discipline.
```

## 3. SFT Baseline Reports

Then read:

```text
reports/deepfactcite_mixclean200_eval_summary.md
reports/searchr1_core_bm25_eval_summary.md
```

What they teach:

- how Base, Soft100, and MixClean200 compare;
- why ShortQA32 and Search-R1 BM25 200 are guardrails;
- why a citation project still needs answer/search metrics.

Key result:

```text
MixClean200 is the current SFT baseline because it improves Search-R1 BM25 200
subEM to 0.490 while preserving strong search behavior.
```

## 4. GRPO Ablation Record

Read:

```text
reports/deepfactcite_v2_onecite_grpo_ablation_2gpu_20260518.md
```

What it teaches:

- why claim-filtered data was needed;
- why outcome-only and citation-aware must run on the same data;
- why v2 was useful but not enough;
- how the promotion gate for 4-card runs was derived.

Key result:

```text
Citation-aware reward improved URL validity, citation precision, claim support,
and unsupported citation rate over outcome-only reward, but v2 still had too
many no-citation failures.
```

## 5. Rollout Gallery

Then inspect:

```text
reports/deepfactcite_v3_promptfix_rollout_gallery_20260518.md
reports/deepfactcite_v3_promptfix_rollout_gallery_samples_20260518.jsonl
```

What it teaches:

- how to read individual model trajectories;
- what a fully supported answer looks like;
- what a partially supported answer looks like;
- why some outputs have a valid URL but still fail claim support;
- why v3 is a real positive signal but not a final product checkpoint.

Beginner checkpoint:

```text
Pick one failure sample and explain which exact phrase in the answer is broader
than the cited snippet. That is claim-level attribution debugging.
```

## 6. Raw Metric Summary

Use this as the compact result table:

```text
reports/dfc_mixclean200_v3_promptfix_onecite_2gpu_rollout_summary.md
```

Important numbers:

| Metric | v3 Prompt-Fix |
|---|---:|
| reward | 0.5120 |
| search | 0.9375 |
| URL validity | 0.9375 |
| citation precision | 0.6312 |
| claim support | 0.6250 |
| unsupported citation rate | 0.1250 |
| no citation failures | 2/32 |

How to use it:

```text
Use this file for headline metrics. Use the rollout gallery to understand why
the metrics moved.
```

## 7. Next-Run Plan

Before renting GPUs, read:

```text
reports/deepfactcite_cpu_mode_consistency_check_20260518.md
reports/storage_cleanup_candidates_current_20260518.md
docs/deepfactcite_4gpu_saved_run_plan_20260518.md
```

What it teaches:

- whether the v3 parquet/jsonl/rollout/report artifacts agree;
- which large directories can be cleaned before migration or checkpoint saving;
- why 4 cards are justified only after v3;
- what command should be run;
- what disk and checkpoint constraints matter;
- what metrics must be checked before extending the run.

## 8. Code Entry Points

Only after the docs above, read the code:

```text
scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py
scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
scripts/deepfactcite/verl_deepfactcite_reward.py
deepfactcite/reward.py
scripts/deepfactcite/summarize_grpo_rollouts.py
```

What each file does:

| File | Role |
|---|---|
| `prepare_sglang_grpo_claim_filtered.py` | builds narrow claim-level GRPO data |
| `run_sglang_grpo_ablation_2gpu.sh` | launches Search-R1/SGLang/GRPO with DeepFactCite reward |
| `verl_deepfactcite_reward.py` | connects verl reward manager to project reward code |
| `deepfactcite/reward.py` | parses answer/search/citation behavior and computes reward details |
| `summarize_grpo_rollouts.py` | turns rollout JSONL into metric reports |

## Mental Model

The current workflow is:

```text
1. Build narrow claim-level data.
2. Verify retriever can return the target URL.
3. Run tiny GRPO ablations without checkpoint saving.
4. Inspect rollout-level failures.
5. Fix data/prompt/reward alignment.
6. Only then run a saved medium checkpoint.
7. Evaluate the checkpoint against answer/search and citation metrics.
8. Record both wins and failures.
```

The project is valuable because it treats citation quality as an engineering
system, not a single reward number.
