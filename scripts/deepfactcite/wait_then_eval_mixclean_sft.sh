#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Search-R1-DeepFactCite

ADAPTER=outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200/global_step_200
SFT_PATTERN=deepfactcite-sft-qwen3-8b-lora-mix-clean-200
VLLM_SCREEN=dfc_vllm_base_soft_mixclean

echo "[$(date -u '+%F %T UTC')] waiting for SFT process to finish"
while pgrep -af "$SFT_PATTERN" | grep -v wait_then_eval_mixclean_sft.sh >/dev/null; do
  tail -5 logs/deepfactcite-sft-qwen3-8b-lora-mix-clean-200.log 2>/dev/null || true
  sleep 30
done

echo "[$(date -u '+%F %T UTC')] SFT process finished; checking adapter"
test -f "$ADAPTER/adapter_config.json"
test -f "$ADAPTER/adapter_model.safetensors"

echo "[$(date -u '+%F %T UTC')] launching vLLM"
screen -dmS "$VLLM_SCREEN" bash scripts/deepfactcite/launch_vllm_base_soft_mixclean.sh

echo "[$(date -u '+%F %T UTC')] running eval suite"
BASE_URL=http://127.0.0.1:8001/v1 \
  bash scripts/deepfactcite/run_mixclean_sft_eval_suite.sh

echo "[$(date -u '+%F %T UTC')] eval suite finished"
