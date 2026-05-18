# DeepFactCite 4-GPU Saved Checkpoint Plan

Date: 2026-05-18

## Decision

Use 4 GPUs only for a saved medium run after the v3 prompt-fix result. Do not
rent 8 GPUs for the next step. The current bottleneck is experiment control,
checkpoint/eval discipline, and claim-support behavior, not raw parallelism.

Why 4 GPUs are now justified:

| Gate | v3 result | Decision |
|---|---:|---|
| no-citation failures | 2/32 | passed |
| claim support | 0.6250 | passed |
| unsupported citation rate | 0.1250 | passed |
| search | 0.9375 | watch closely |
| response clipping | 0.0000 | passed |
| fake URL rate | 0.0000 | passed |

The goal of the next run is not another smoke test. The goal is to save a
checkpoint that can be evaluated against answer/search and citation metrics.

## Prepared Entrypoints

Parameterized base launcher:

```text
scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

This script still defaults to 2 GPUs, but now accepts:

```text
N_GPUS_PER_NODE
TENSOR_MODEL_PARALLEL_SIZE
CUDA_VISIBLE_DEVICES
```

4-GPU v3 saved wrapper:

```text
scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

2-GPU fallback wrapper for 3-card availability:

```text
scripts/deepfactcite/run_sglang_grpo_v3_promptfix_2gpu_saved.sh
```

Default settings:

```text
DATA_DIR=data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite
MODE=citation-aware
CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS_PER_NODE=4
TENSOR_MODEL_PARALLEL_SIZE=4
GRPO_TOTAL_STEPS=64
GRPO_SAVE_FREQ=16
GRPO_MAX_ACTOR_CKPT_TO_KEEP=1
GRPO_ACTOR_CKPT_SAVE_CONTENTS=[model]
SEARCHQA_TOPK=2
rollout.n=2
train_batch_size=1
```

The wrapper defaults to `DRY_RUN=1`, so it is safe to inspect on CPU-only
machines.

## 3-GPU Availability

Do not run the 4-GPU config on 3 GPUs.

Reason:

```text
The prepared 4-GPU run uses tensor_model_parallel_size=4. A 3-GPU variant would
require TP=3 or a different actor/rollout placement. TP=3 is not a standard
validated split for this Qwen3-8B/SGLang setup and may be incompatible with
attention-head partitioning. Debugging that would spend GPU time on
infrastructure rather than citation learning.
```

If only 3 cards are available, use 2 cards with the stable TP=2 path:

```bash
DRY_RUN=0 \
GRPO_TOTAL_STEPS=8 \
GRPO_SAVE_FREQ=4 \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu_save_sanity \
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_2gpu_saved.sh
```

If the 8-step saved sanity is clean, continue:

```bash
DRY_RUN=0 \
GRPO_TOTAL_STEPS=64 \
GRPO_SAVE_FREQ=16 \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu_saved \
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_2gpu_saved.sh
```

This is slower than 4 GPUs but scientifically cleaner than inventing an
untested 3-GPU topology.

## Disk Requirement

Current `/root/autodl-tmp` free space was about 14G before cleanup. That is too
tight for saved checkpoint work.

Minimum recommendation before starting:

```text
At least 35G free for model-only checkpoint saving.
At least 50G free for resume-capable checkpoint saving with extra state.
```

The prepared wrapper defaults to model-only save:

```text
GRPO_ACTOR_CKPT_SAVE_CONTENTS=[model]
```

Use this first. Only switch to `[model,extra]` if resume is more important than
disk pressure.

## Commands

CPU/no-GPU dry-run check:

```bash
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

First 4-GPU checkpoint-write sanity run:

```bash
DRY_RUN=0 \
GRPO_TOTAL_STEPS=8 \
GRPO_SAVE_FREQ=4 \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260518_4gpu_save_sanity \
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

If that run starts cleanly, writes a checkpoint, and keeps rollout metrics sane,
run the medium saved experiment:

```bash
DRY_RUN=0 \
GRPO_TOTAL_STEPS=64 \
GRPO_SAVE_FREQ=16 \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260518_4gpu_saved \
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

Expected output locations:

```text
outputs/deepfactcite/grpo/<EXPERIMENT_NAME>/
logs/grpo/rollouts/<EXPERIMENT_NAME>/
logs/<EXPERIMENT_NAME>.log
tensorboard_log/DeepFactCite-GRPO/<EXPERIMENT_NAME>/
```

## Stop Rules

Stop the 4-GPU run early if any of these appear in the first 8 to 16 steps:

| Signal | Stop condition |
|---|---|
| search | drops materially below the v3 0.9375 diagnostic level |
| no citation | rises back toward v2 behavior, especially above 6/32 |
| claim support | falls below 0.50 on comparable samples |
| unsupported citation rate | rises above 0.25 |
| fake URL rate | becomes non-zero in repeated samples |
| checkpoint | save fails or disk free space becomes dangerously low |
| infrastructure | SGLang/Ray repeatedly restarts, hangs, or leaks memory |

## Post-Run Checklist

After the run finishes:

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python \
  scripts/deepfactcite/summarize_grpo_rollouts.py \
  --rollout-dir logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260518_4gpu_saved \
  --out reports/dfc_mixclean200_v3_promptfix_onecite_4gpu_saved_rollout_summary.md \
  --recompute-details
```

Then record:

```text
1. exact command;
2. free disk before/after;
3. checkpoint path and checkpoint size;
4. rollout metrics;
5. 5 good samples and 5 failure samples;
6. whether answer/search behavior stayed intact;
7. whether citation quality improved beyond the 16-step diagnostic.
```

Do not report the run as a win until the saved checkpoint is evaluated, not just
trained.
