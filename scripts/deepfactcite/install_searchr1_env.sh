#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-legacy}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

case "$PROFILE" in
  legacy)
    ENV_NAME="${ENV_NAME:-searchr1}"
    PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
    TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
    TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers<4.48}"
    VLLM_VERSION="${VLLM_VERSION:-0.6.3}"
    INSTALL_VLLM=1
    INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
    ;;
  qwen3-sft)
    ENV_NAME="${ENV_NAME:-searchr1-qwen3-sft}"
    PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
    TORCH_VERSION="${TORCH_VERSION:-2.4.0}"
    TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.51,<5}"
    INSTALL_VLLM=0
    INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
    ;;
  qwen3-grpo)
    cat >&2 <<'EOF'
PROFILE=qwen3-grpo is intentionally not installed by this script.

Vanilla Search-R1 pins vLLM <= 0.6.3 and its local vLLM adapter only supports
0.3.1, 0.4.2, 0.5.4, and 0.6.3. Qwen3 checkpoints use model_type=qwen3, which
requires a newer model stack than the original Search-R1 rollout path.

Use PROFILE=legacy for Search-R1-compatible GRPO with Qwen2.5/Llama baselines,
or PROFILE=qwen3-sft for Qwen3 LoRA SFT. Qwen3 GRPO needs a rollout port to a
newer vLLM/veRL/SGLang stack before it should be installed as a one-command env.
EOF
    exit 2
    ;;
  *)
    echo "Unsupported PROFILE=$PROFILE. Use legacy, qwen3-sft, or qwen3-grpo." >&2
    exit 1
    ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  if [ -f /root/autodl-tmp/miniconda3/bin/conda ]; then
    # shellcheck source=/dev/null
    source /root/autodl-tmp/miniconda3/bin/activate
  elif [ -f /root/miniconda3/bin/conda ]; then
    # shellcheck source=/dev/null
    source /root/miniconda3/bin/activate
  else
    echo "conda not found. Install or activate conda first." >&2
    exit 1
  fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m pip install -U pip setuptools wheel
python -m pip install "torch==$TORCH_VERSION" --index-url "$CUDA_INDEX_URL"

REQ_FILE="$(mktemp)"
grep -Ev '^(transformers|vllm|flash-attn)([<=>!~].*)?$' requirements.txt > "$REQ_FILE"
python -m pip install -r "$REQ_FILE"
rm -f "$REQ_FILE"

python -m pip install "$TRANSFORMERS_SPEC"
if [ "$INSTALL_VLLM" = "1" ]; then
  python -m pip install "vllm==$VLLM_VERSION"
fi
python -m pip install -e . --no-deps
python -m pip install peft fastapi uvicorn pyarrow

if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    python -m pip install flash-attn --no-build-isolation
  else
    echo "nvcc not found; skipping flash-attn. Set INSTALL_FLASH_ATTN=0 to silence this warning." >&2
  fi
fi

if command -v apt-get >/dev/null 2>&1 && ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c is not installed. Install it with apt-get if your image permits package installation."
fi

python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import torch
import transformers
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
try:
    print("vllm", version("vllm"))
except PackageNotFoundError:
    print("vllm", "not installed")
PY
