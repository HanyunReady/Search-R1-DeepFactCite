#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${MODEL_SIZE:-4B}"
DATA_DIR="${DATA_DIR:-data/deepfactcite/sft}"
N_GPUS="${N_GPUS:-}"
LORA_RANK="${LORA_RANK:-32}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-}"
LR="${LR:-1e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/output/Search-R1-DeepFactCite}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$MODEL_SIZE" in
  4B|4b)
    BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/LLM-qwen3_posttrain/.cache/models/Qwen_Qwen3-4B-Base}"
    N_GPUS="${N_GPUS:-1}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-$N_GPUS}"
    ;;
  8B|8b)
    BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base}"
    N_GPUS="${N_GPUS:-2}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-$N_GPUS}"
    ;;
  *)
    echo "Unsupported MODEL_SIZE=$MODEL_SIZE. Use 4B or 8B." >&2
    exit 1
    ;;
esac

if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
  echo "Missing SFT parquet files under $DATA_DIR. Run scripts/deepfactcite/prepare_data.py first." >&2
  exit 1
fi

"$PYTHON_BIN" - "$BASE_MODEL" <<'PY'
import json
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version

model_path = sys.argv[1]
config_path = os.path.join(model_path, "config.json")
if not os.path.exists(config_path):
    raise SystemExit(f"Model config not found: {config_path}")

with open(config_path, "r", encoding="utf-8") as f:
    model_type = json.load(f).get("model_type")

try:
    transformers_version = version("transformers")
except PackageNotFoundError as exc:
    raise SystemExit("transformers is not installed. Activate the qwen3-sft env first.") from exc

match = re.match(r"^(\d+)\.(\d+)", transformers_version)
major_minor = tuple(map(int, match.groups())) if match else (0, 0)

if model_type == "qwen3" and major_minor < (4, 51):
    raise SystemExit(
        "Qwen3 SFT needs transformers>=4.51. Install with: "
        "PROFILE=qwen3-sft bash scripts/deepfactcite/install_searchr1_env.sh"
    )
PY

EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepfactcite-sft-qwen3-${MODEL_SIZE,,}-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR" logs

EXTRA_TOTAL_STEPS=()
if [ -n "$TOTAL_STEPS" ]; then
  EXTRA_TOTAL_STEPS=(trainer.total_training_steps="$TOTAL_STEPS")
fi

CMD=("$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$N_GPUS" -m verl.trainer.fsdp_sft_trainer
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.prompt_key=prompt \
  data.response_key=answer \
  data.max_length="$MAX_LENGTH" \
  data.truncation=right \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size="$MICRO_BATCH_SIZE" \
  model.partial_pretrain="$BASE_MODEL" \
  model.trust_remote_code=True \
  model.attn_implementation="$ATTN_IMPLEMENTATION" \
  model.enable_gradient_checkpointing=True \
  model.lora_rank="$LORA_RANK" \
  model.lora_alpha="$((LORA_RANK * 2))" \
  model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj] \
  optim.lr="$LR" \
  trainer.project_name=DeepFactCite-GRPO \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.logger=[console,wandb] \
  "${EXTRA_TOTAL_STEPS[@]}")

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'DRY_RUN command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}" 2>&1 | tee "logs/${EXPERIMENT_NAME}.log"
