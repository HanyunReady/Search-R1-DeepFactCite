#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Search-R1-DeepFactCite

BASE_MODEL=/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base
SOFT_ADAPTER=outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100
MIX_ADAPTER=outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200/global_step_200
VLLM_BIN=/root/autodl-tmp/conda_envs/openrlhf_vllm085/bin/vllm

test -d "$BASE_MODEL"
test -f "$SOFT_ADAPTER/adapter_config.json"
test -f "$MIX_ADAPTER/adapter_config.json"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "$VLLM_BIN" serve "$BASE_MODEL" \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name base \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --trust-remote-code \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 32 \
  --lora-modules "soft=$SOFT_ADAPTER" "mixclean=$MIX_ADAPTER"
