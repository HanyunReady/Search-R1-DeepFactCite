#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Search-R1-DeepFactCite

PYTHON_BIN=/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python
BASE_URL="${BASE_URL:-http://127.0.0.1:8001/v1}"
TOKENIZER=/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base

wait_for_vllm() {
  for _ in $(seq 1 180); do
    if curl -fsS "$BASE_URL/models" >/dev/null; then
      return 0
    fi
    sleep 5
  done
  echo "vLLM server did not become ready: $BASE_URL" >&2
  return 1
}

run_eval() {
  local model="$1"
  local tag="$2"

  "$PYTHON_BIN" scripts/deepfactcite/eval_citation_agent_vllm.py \
    --base-url "$BASE_URL" \
    --model "$model" \
    --tokenizer "$TOKENIZER" \
    --data data/shortqa_guardrail/rl/test.parquet \
    --corpus data/shortqa_guardrail/corpus.jsonl \
    --output-jsonl "reports/${tag}_vllm_shortqa_guardrail32.jsonl" \
    --report-json "reports/${tag}_vllm_shortqa_guardrail32.json" \
    --temperature 0.0 \
    --max-turns 4 \
    --topk 3

  "$PYTHON_BIN" scripts/deepfactcite/eval_citation_agent_vllm.py \
    --base-url "$BASE_URL" \
    --model "$model" \
    --tokenizer "$TOKENIZER" \
    --data data/deepfactcite_strict/sft/test.parquet \
    --corpus data/deepfactcite_strict/corpus.jsonl \
    --output-jsonl "reports/${tag}_vllm_deepfactcite_strict_sft_test47.jsonl" \
    --report-json "reports/${tag}_vllm_deepfactcite_strict_sft_test47.json" \
    --temperature 0.0 \
    --max-turns 4 \
    --topk 3
}

wait_for_vllm
curl -fsS "$BASE_URL/models" > reports/vllm_base_soft_mixclean_models.json

run_eval base base_qwen3_8b_mixclean_rerun
run_eval soft soft100_qwen3_8b_mixclean_rerun
run_eval mixclean mixclean200_qwen3_8b
