#!/usr/bin/env bash
set -euo pipefail

CORPUS="${CORPUS:-data/deepfactcite/corpus.jsonl}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

python -m deepfactcite.retriever_server \
  --corpus "$CORPUS" \
  --host "$HOST" \
  --port "$PORT"

