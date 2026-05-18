#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Search-R1-DeepFactCite

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8001/v1}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
DATA="${DATA:-data/searchr1_core_guardrail/test.parquet}"
LIMIT="${LIMIT:-0}"

mkdir -p reports logs

curl -fsS "${BASE_URL%/}/models" > reports/vllm_searchr1_core_bm25_models.json
curl -fsS "$RETRIEVER_URL" \
  -H 'Content-Type: application/json' \
  -d '{"queries":["who got the first nobel prize in physics"],"topk":1,"return_scores":true}' \
  > reports/searchr1_core_bm25_retriever_smoke.json

run_eval() {
  local model="$1"
  local prefix="$2"
  echo "[$(date -u '+%F %T UTC')] running $model on Search-R1 core BM25"
  "$PYTHON_BIN" scripts/deepfactcite/eval_searchr1_agent_vllm.py \
    --base-url "$BASE_URL" \
    --model "$model" \
    --tokenizer "$BASE_MODEL" \
    --data "$DATA" \
    --retriever-url "$RETRIEVER_URL" \
    --output-jsonl "reports/${prefix}_searchr1_core_bm25_200.jsonl" \
    --report-json "reports/${prefix}_searchr1_core_bm25_200.json" \
    --temperature 0.0 \
    --max-turns 4 \
    --topk 3 \
    --resume \
    ${LIMIT:+--limit "$LIMIT"}
}

run_eval base base_qwen3_8b
run_eval soft soft100_qwen3_8b
run_eval mixclean mixclean200_qwen3_8b

echo "[$(date -u '+%F %T UTC')] Search-R1 core BM25 eval suite finished"
