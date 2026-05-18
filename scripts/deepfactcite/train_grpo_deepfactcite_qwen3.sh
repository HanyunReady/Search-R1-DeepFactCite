#!/usr/bin/env bash
set -euo pipefail

MODEL_SIZE="${MODEL_SIZE:-4B}"
DATA_DIR="${DATA_DIR:-data/deepfactcite/rl}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
N_GPUS="${N_GPUS:-}"
MAX_TURNS="${MAX_TURNS:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/output/Search-R1-DeepFactCite}"

case "$MODEL_SIZE" in
  4B|4b)
    BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/LLM-qwen3_posttrain/.cache/models/Qwen_Qwen3-4B-Base}"
    N_GPUS="${N_GPUS:-2}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
    MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
    LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-8}"
    TP_SIZE="${TP_SIZE:-1}"
    N_AGENT="${N_AGENT:-4}"
    ;;
  8B|8b)
    BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base}"
    N_GPUS="${N_GPUS:-4}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
    MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
    LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-4}"
    TP_SIZE="${TP_SIZE:-2}"
    N_AGENT="${N_AGENT:-4}"
    ;;
  *)
    echo "Unsupported MODEL_SIZE=$MODEL_SIZE. Use 4B or 8B." >&2
    exit 1
    ;;
esac

if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
  echo "Missing RL parquet files under $DATA_DIR. Run scripts/deepfactcite/prepare_data.py first." >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate the Search-R1 training environment first." >&2
  exit 1
fi

if [ -f "$BASE_MODEL/config.json" ]; then
  python - "$BASE_MODEL" <<'PY'
import json
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version

model_path = sys.argv[1]
with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
    model_type = json.load(f).get("model_type")

if model_type != "qwen3":
    raise SystemExit(0)

try:
    transformers_version = version("transformers")
except PackageNotFoundError as exc:
    raise SystemExit("transformers is not installed. Activate the training env first.") from exc

match = re.match(r"^(\d+)\.(\d+)", transformers_version)
major_minor = tuple(map(int, match.groups())) if match else (0, 0)
if major_minor < (4, 51):
    raise SystemExit("Qwen3 GRPO needs transformers>=4.51.")

if os.getenv("ALLOW_EXPERIMENTAL_QWEN3_GRPO") != "1":
    raise SystemExit(
        "Qwen3 GRPO is not one-command runnable on vanilla Search-R1: the original rollout path "
        "pins vLLM<=0.6.3 and the local vLLM adapter only supports 0.3.1/0.4.2/0.5.4/0.6.3. "
        "Run Qwen3 SFT first, or set ALLOW_EXPERIMENTAL_QWEN3_GRPO=1 only after porting rollout "
        "to a newer vLLM/veRL/SGLang stack."
    )
PY
else
  echo "[WARN] Skipping local model config preflight because BASE_MODEL is not a local directory: $BASE_MODEL" >&2
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-deepfactcite-grpo-qwen3-${MODEL_SIZE,,}}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR" logs

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export PYTHONUNBUFFERED=1

python -m verl.trainer.main_ppo_deepfactcite \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-4096}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-1024}" \
  data.max_start_length="${MAX_START_LENGTH:-2048}" \
  data.max_obs_length="${MAX_OBS_LENGTH:-1200}" \
  data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=grpo \
  algorithm.no_think_rl=false \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr="${LR:-5e-7}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size="$MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.state_masking=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="$LOGPROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.55}" \
  actor_rollout_ref.rollout.n_agent="$N_AGENT" \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE:-1.0}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size="$LOGPROB_MICRO_BATCH_SIZE" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.deepfactcite_answer_weight="${ANSWER_WEIGHT:-0.25}" \
  reward_model.deepfactcite_citation_weight="${CITATION_WEIGHT:-0.35}" \
  reward_model.deepfactcite_support_weight="${SUPPORT_WEIGHT:-0.25}" \
  reward_model.deepfactcite_format_weight="${FORMAT_WEIGHT:-0.10}" \
  reward_model.deepfactcite_search_weight="${SEARCH_WEIGHT:-0.05}" \
  reward_model.deepfactcite_cost_weight="${COST_WEIGHT:-0.05}" \
  reward_model.deepfactcite_max_searches="$MAX_TURNS" \
  reward_model.deepfactcite_use_judge="${USE_JUDGE:-False}" \
  trainer.logger="${LOGGER:-[console,wandb]}" \
  +trainer.val_only=false \
  +trainer.val_before_train=true \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.project_name=DeepFactCite-GRPO \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.total_epochs="${TOTAL_EPOCHS:-3}" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  max_turns="$MAX_TURNS" \
  retriever.url="$RETRIEVER_URL" \
  retriever.topk="${TOPK:-3}" \
  2>&1 | tee "logs/${EXPERIMENT_NAME}.log"
