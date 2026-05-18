#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${MODEL_SIZE:-4B}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
LOCAL_DIR_ROOT="${LOCAL_DIR_ROOT:-/root/autodl-tmp/models}"
ARIA2_CONNECTIONS="${ARIA2_CONNECTIONS:-16}"
ARIA2_SPLITS="${ARIA2_SPLITS:-16}"
ARIA2_CHUNK_SIZE="${ARIA2_CHUNK_SIZE:-1M}"

case "$MODEL_SIZE" in
  4B|4b)
    MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Base}"
    LOCAL_DIR="${LOCAL_DIR:-$LOCAL_DIR_ROOT/Qwen3-4B-Base}"
    FILES=(
      .gitattributes LICENSE README.md added_tokens.json chat_template.jinja
      config.json generation_config.json merges.txt
      model-00001-of-00003.safetensors
      model-00002-of-00003.safetensors
      model-00003-of-00003.safetensors
      model.safetensors.index.json special_tokens_map.json
      tokenizer.json tokenizer_config.json vocab.json
    )
    ;;
  8B|8b)
    MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B-Base}"
    LOCAL_DIR="${LOCAL_DIR:-$LOCAL_DIR_ROOT/Qwen3-8B-Base}"
    FILES=(
      .gitattributes LICENSE README.md added_tokens.json chat_template.jinja
      config.json generation_config.json merges.txt
      model-00001-of-00005.safetensors
      model-00002-of-00005.safetensors
      model-00003-of-00005.safetensors
      model-00004-of-00005.safetensors
      model-00005-of-00005.safetensors
      model.safetensors.index.json special_tokens_map.json
      tokenizer.json tokenizer_config.json vocab.json
    )
    ;;
  *)
    echo "Unsupported MODEL_SIZE=$MODEL_SIZE. Use 4B or 8B." >&2
    exit 1
    ;;
esac

mkdir -p "$LOCAL_DIR"
BASE_URL="${HF_ENDPOINT%/}/${MODEL_ID}/resolve/main"

download_one() {
  local file="$1"
  local url="$BASE_URL/$file"
  if [ -s "$LOCAL_DIR/$file" ]; then
    echo "skip existing $LOCAL_DIR/$file"
    return
  fi
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --continue=true \
      --max-connection-per-server="$ARIA2_CONNECTIONS" \
      --split="$ARIA2_SPLITS" \
      --min-split-size="$ARIA2_CHUNK_SIZE" \
      --retry-wait=5 \
      --max-tries=20 \
      --dir="$LOCAL_DIR" \
      --out="$file" \
      "$url"
  else
    curl --location --continue-at - --fail --retry 20 --retry-delay 5 \
      --output "$LOCAL_DIR/$file" \
      "$url"
  fi
}

for file in "${FILES[@]}"; do
  download_one "$file"
done

echo "$LOCAL_DIR"
