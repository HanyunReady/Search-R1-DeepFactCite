#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Search-R1-DeepFactCite
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/searchr1-qwen3-sft

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHON_BIN=/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python

export MODEL_SIZE=8B
export N_GPUS=2
export DATA_DIR=data/deepfactcite_mix/sft
export TOTAL_STEPS=200
export TRAIN_BATCH_SIZE=4
export MICRO_BATCH_SIZE=2
export EXPERIMENT_NAME=deepfactcite-sft-qwen3-8b-lora-mix-clean-200
export OUTPUT_ROOT=/root/autodl-tmp/Search-R1-DeepFactCite/outputs/deepfactcite

bash scripts/deepfactcite/train_sft_qwen3.sh
