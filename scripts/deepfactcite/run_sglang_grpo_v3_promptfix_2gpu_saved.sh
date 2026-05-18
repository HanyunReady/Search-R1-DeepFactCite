#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export RUN_TAG="${RUN_TAG:-20260518_v3_promptfix_2gpu_saved}"
export MODE="${MODE:-citation-aware}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu_saved}"
export DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"

export GRPO_TOTAL_STEPS="${GRPO_TOTAL_STEPS:-64}"
export GRPO_SAVE_FREQ="${GRPO_SAVE_FREQ:-16}"
export GRPO_MAX_ACTOR_CKPT_TO_KEEP="${GRPO_MAX_ACTOR_CKPT_TO_KEEP:-1}"
export GRPO_ACTOR_CKPT_SAVE_CONTENTS="${GRPO_ACTOR_CKPT_SAVE_CONTENTS:-[model]}"

export GRPO_N="${GRPO_N:-2}"
export GRPO_TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-1}"
export GRPO_MINI_BATCH_SIZE="${GRPO_MINI_BATCH_SIZE:-1}"
export GRPO_MAX_RESPONSE_LENGTH="${GRPO_MAX_RESPONSE_LENGTH:-384}"
export GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-2048}"
export GRPO_AGENT_WORKERS="${GRPO_AGENT_WORKERS:-2}"
export GRPO_GPU_MEMORY_UTILIZATION="${GRPO_GPU_MEMORY_UTILIZATION:-0.15}"
export SEARCHQA_TOPK="${SEARCHQA_TOPK:-2}"

# Default to dry-run so this wrapper is safe to inspect on CPU-only machines.
export DRY_RUN="${DRY_RUN:-1}"

exec bash "$REPO_ROOT/scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh"
