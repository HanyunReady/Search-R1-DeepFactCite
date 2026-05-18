#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERL_DIR="${VERL_DIR:-/root/autodl-tmp/SearchShortQA/verl}"
AGENTIC_ROOT="${AGENTIC_ROOT:-/root/autodl-tmp/agentic-rl-searchqa}"
VERL_PYTHON="${VERL_PYTHON:-/root/autodl-tmp/conda_envs/grpo-sglang/bin/python}"

MODE="${MODE:-citation-aware}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/deepfactcite_sglang_grpo_claim_filtered}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/test.parquet}"
CORPUS="${CORPUS:-$DATA_DIR/corpus.jsonl}"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$REPO_ROOT/outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

case "$MODE" in
  outcome-only)
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-dfc-mixclean200-claimfiltered-outcome-only-$RUN_TAG}"
    export DFC_ANSWER_WEIGHT="${DFC_ANSWER_WEIGHT:-0.80}"
    export DFC_CITATION_WEIGHT="${DFC_CITATION_WEIGHT:-0.00}"
    export DFC_SUPPORT_WEIGHT="${DFC_SUPPORT_WEIGHT:-0.00}"
    export DFC_FORMAT_WEIGHT="${DFC_FORMAT_WEIGHT:-0.15}"
    export DFC_SEARCH_WEIGHT="${DFC_SEARCH_WEIGHT:-0.10}"
    export DFC_COST_WEIGHT="${DFC_COST_WEIGHT:-0.05}"
    ;;
  citation-aware)
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-dfc-mixclean200-claimfiltered-citation-aware-$RUN_TAG}"
    export DFC_ANSWER_WEIGHT="${DFC_ANSWER_WEIGHT:-0.15}"
    export DFC_CITATION_WEIGHT="${DFC_CITATION_WEIGHT:-0.35}"
    export DFC_SUPPORT_WEIGHT="${DFC_SUPPORT_WEIGHT:-0.30}"
    export DFC_FORMAT_WEIGHT="${DFC_FORMAT_WEIGHT:-0.10}"
    export DFC_SEARCH_WEIGHT="${DFC_SEARCH_WEIGHT:-0.05}"
    export DFC_COST_WEIGHT="${DFC_COST_WEIGHT:-0.05}"
    ;;
  *)
    echo "Unsupported MODE=$MODE. Use outcome-only or citation-aware." >&2
    exit 1
    ;;
esac

SAVE_PATH="${SAVE_PATH:-$REPO_ROOT/outputs/deepfactcite/grpo/$EXPERIMENT_NAME}"
ROLLOUT_DIR="${ROLLOUT_DIR:-$REPO_ROOT/logs/grpo/rollouts/$EXPERIMENT_NAME}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/logs/${EXPERIMENT_NAME}.log}"

for path in "$VERL_DIR" "$AGENTIC_ROOT/src" "$TRAIN_FILE" "$TEST_FILE" "$CORPUS" "$ACTOR_MODEL_PATH/config.json"; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

mkdir -p "$SAVE_PATH" "$ROLLOUT_DIR" "$REPO_ROOT/logs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="$REPO_ROOT:$AGENTIC_ROOT/src:$VERL_DIR:${PYTHONPATH:-}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29631}"
export DIST_INIT_METHOD="tcp://$MASTER_ADDR:$MASTER_PORT"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VERL_ATTN_IMPLEMENTATION="${VERL_ATTN_IMPLEMENTATION:-sdpa}"
export SEARCHQA_SEARCH_BACKEND="${SEARCHQA_SEARCH_BACKEND:-offline}"
export SEARCHQA_CORPUS="$CORPUS"
export SEARCHQA_TOPK="${SEARCHQA_TOPK:-2}"
export AGENTIC_SEARCHQA_ROLLOUT_DIR="$ROLLOUT_DIR"
export VERL_SGLANG_HTTP_MAX_ATTEMPTS="${VERL_SGLANG_HTTP_MAX_ATTEMPTS:-60}"
export VERL_SGLANG_HTTP_RETRY_DELAY="${VERL_SGLANG_HTTP_RETRY_DELAY:-1}"
export VERL_SGLANG_HTTP_MAX_RETRY_SLEEP="${VERL_SGLANG_HTTP_MAX_RETRY_SLEEP:-5}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DFC_MAX_SEARCHES="${DFC_MAX_SEARCHES:-3}"
export DFC_USE_JUDGE="${DFC_USE_JUDGE:-false}"

CMD=(
  "$VERL_PYTHON" -m verl.trainer.main_ppo
  --config-path="$VERL_DIR/examples/sglang_multiturn/config"
  --config-name=search_grpo
  algorithm.adv_estimator=grpo
  data.train_files="$TRAIN_FILE"
  data.val_files="$TEST_FILE"
  data.train_batch_size="${GRPO_TRAIN_BATCH_SIZE:-1}"
  data.max_prompt_length="${GRPO_MAX_PROMPT_LENGTH:-2048}"
  data.max_response_length="${GRPO_MAX_RESPONSE_LENGTH:-384}"
  data.prompt_key=prompt
  data.filter_overlong_prompts=True
  data.dataloader_num_workers="${GRPO_DATALOADER_NUM_WORKERS:-0}"
  data.truncation=right
  data.return_raw_chat=True
  actor_rollout_ref.model.path="$ACTOR_MODEL_PATH"
  actor_rollout_ref.actor.optim.lr="${GRPO_LR:-1e-6}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${GRPO_MINI_BATCH_SIZE:-1}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_kl_loss="${GRPO_USE_KL_LOSS:-False}"
  actor_rollout_ref.actor.kl_loss_coef="${GRPO_KL_LOSS_COEF:-0.001}"
  actor_rollout_ref.actor.use_torch_compile=False
  actor_rollout_ref.actor.checkpoint.save_contents="${GRPO_ACTOR_CKPT_SAVE_CONTENTS:-[model,extra]}"
  actor_rollout_ref.actor.checkpoint.load_contents="${GRPO_ACTOR_CKPT_LOAD_CONTENTS:-[model]}"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.rollout.tensor_model_parallel_size=2
  actor_rollout_ref.rollout.name=sglang
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.n="${GRPO_N:-2}"
  actor_rollout_ref.rollout.temperature="${GRPO_TEMPERATURE:-0.3}"
  actor_rollout_ref.rollout.top_p="${GRPO_TOP_P:-0.9}"
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENTIC_ROOT/configs/verl_agent_loop_config.yaml"
  actor_rollout_ref.rollout.agent.num_workers="${GRPO_AGENT_WORKERS:-2}"
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${GRPO_MAX_ASSISTANT_TURNS:-3}"
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${GRPO_MAX_USER_TURNS:-3}"
  actor_rollout_ref.rollout.multi_turn.format=custom
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${GRPO_MAX_TOOL_RESPONSE_LENGTH:-768}"
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$AGENTIC_ROOT/configs/verl_tool_config.yaml"
  actor_rollout_ref.rollout.response_length="${GRPO_MAX_RESPONSE_LENGTH:-384}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${GRPO_GPU_MEMORY_UTILIZATION:-0.15}"
  actor_rollout_ref.rollout.enforce_eager="${GRPO_ENFORCE_EAGER:-True}"
  actor_rollout_ref.rollout.max_num_seqs="${GRPO_MAX_NUM_SEQS:-1}"
  actor_rollout_ref.rollout.free_cache_engine="${GRPO_FREE_CACHE_ENGINE:-False}"
  +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="${SGLANG_ATTENTION_BACKEND:-triton}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.use_torch_compile=False
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  reward_model.enable=False
  reward_model.reward_manager=deepfactcite_custom
  custom_reward_function.path="$REPO_ROOT/scripts/deepfactcite/verl_deepfactcite_reward.py"
  custom_reward_function.name=compute_score
  algorithm.use_kl_in_reward=False
  trainer.critic_warmup=0
  trainer.logger='["console","tensorboard"]'
  trainer.project_name=DeepFactCite-GRPO
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.n_gpus_per_node=2
  trainer.nnodes=1
  trainer.default_local_dir="$SAVE_PATH"
  trainer.save_freq="${GRPO_SAVE_FREQ:-0}"
  trainer.max_actor_ckpt_to_keep="${GRPO_MAX_ACTOR_CKPT_TO_KEEP:-1}"
  trainer.test_freq=-1
  trainer.val_before_train=False
  trainer.total_epochs="${GRPO_TOTAL_EPOCHS:-1}"
  trainer.total_training_steps="${GRPO_TOTAL_STEPS:-16}"
)

echo "mode=$MODE"
echo "experiment=$EXPERIMENT_NAME"
echo "train=$TRAIN_FILE"
echo "test=$TEST_FILE"
echo "corpus=$CORPUS"
echo "save_path=$SAVE_PATH"
echo "rollout_dir=$ROLLOUT_DIR"
echo "log=$LOG_FILE"
echo "weights answer=$DFC_ANSWER_WEIGHT citation=$DFC_CITATION_WEIGHT support=$DFC_SUPPORT_WEIGHT format=$DFC_FORMAT_WEIGHT search=$DFC_SEARCH_WEIGHT cost=$DFC_COST_WEIGHT"
df -h /root/autodl-tmp
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

if [ "${DRY_RUN:-1}" != "0" ]; then
  printf 'DRY_RUN=1; command not executed. To run: DRY_RUN=0 MODE=%q bash %q\n' "$MODE" "$0"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

cd "$VERL_DIR"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
