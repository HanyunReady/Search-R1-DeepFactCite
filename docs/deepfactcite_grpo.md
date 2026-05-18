# DeepFactCite-GRPO on Search-R1

This fork adds a DeepFactCite layer on top of Search-R1:

- Search-R1 remains the base training framework.
- DeepFactCite adds long-form citation prompts, SFT data conversion, a lightweight lexical retriever, and a citation-faithful GRPO reward.
- Qwen3-4B is the fast MVP target. Qwen3-8B is the stronger interview target.

## Resource Plan

Recommended stable allocation:

| Model | LoRA SFT | GRPO |
| --- | ---: | ---: |
| Qwen3-4B | 1x A800 80G | 2x A800 80G |
| Qwen3-8B | 2x A800 80G | 4x A800 80G |

The 8B GRPO run may be squeezed onto fewer GPUs with smaller batches and stronger offload, but 4x A800 is the practical target because Search-R1 keeps rollout/ref/critic components and a vLLM engine in the loop.

## Files Added

```text
deepfactcite/prompts.py
deepfactcite/reward.py
deepfactcite/retriever_server.py
verl/utils/reward_score/deepfactcite.py
verl/trainer/main_ppo_deepfactcite.py
verl/utils/dataset/sft_dataset.py
scripts/deepfactcite/prepare_data.py
scripts/deepfactcite/prepare_agentic_eval_data.py
scripts/deepfactcite/install_searchr1_env.sh
scripts/deepfactcite/download_qwen3.sh
scripts/deepfactcite/start_lexical_retriever.sh
scripts/deepfactcite/train_sft_qwen3.sh
scripts/deepfactcite/train_grpo_deepfactcite_qwen3.sh
```

Search-R1's retrieval display was also patched to include `URL:` lines in `<information>` blocks when the corpus provides URLs. This is required for citation-faithful training because the model must cite URLs returned by the current search trajectory.

## Environment

There are two useful environment profiles:

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite

# Download large model files into /root/autodl-tmp with aria2c/resume support.
MODEL_SIZE=4B LOCAL_DIR=/root/autodl-tmp/LLM-qwen3_posttrain/.cache/models/Qwen_Qwen3-4B-Base \
  bash scripts/deepfactcite/download_qwen3.sh

# Original Search-R1 stack: suitable for Qwen2.5/Llama GRPO reproduction.
PROFILE=legacy bash scripts/deepfactcite/install_searchr1_env.sh

# Qwen3 SFT stack: modern transformers without the old vLLM rollout pin.
PROFILE=qwen3-sft bash scripts/deepfactcite/install_searchr1_env.sh
```

The legacy profile follows Search-R1 defaults: Python 3.9, torch 2.4.0 CUDA 12.1, vLLM 0.6.3, editable install, flash-attn, PEFT, FastAPI, uvicorn, and pyarrow.

The Qwen3 SFT profile uses Python 3.10 and `transformers>=4.51`. It intentionally does not install vLLM because vanilla Search-R1 pins vLLM <= 0.6.3 and the local rollout adapter only supports vLLM 0.3.1/0.4.2/0.5.4/0.6.3. Qwen3 GRPO therefore needs a rollout port to a newer vLLM/veRL/SGLang stack before it should be treated as one-command runnable.

## Data Preparation

Prepare DeepCiteFact-derived SFT/RL data:

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite
python scripts/deepfactcite/prepare_data.py \
  --deepcitefact-dir /root/autodl-tmp/DeepCiteFact \
  --output-dir data/deepfactcite
```

Outputs:

```text
data/deepfactcite/sft/train.parquet
data/deepfactcite/sft/test.parquet
data/deepfactcite/rl/train.parquet
data/deepfactcite/rl/test.parquet
data/deepfactcite/corpus.jsonl
```

Optional small eval conversion from the existing `agentic-rl-searchqa` repo:

```bash
python scripts/deepfactcite/prepare_agentic_eval_data.py \
  --output-dir data/agentic_eval48
```

## Retriever

For the small DeepFactCite corpus, use the lightweight lexical server:

```bash
conda activate searchr1
cd /root/autodl-tmp/Search-R1-DeepFactCite
CORPUS=data/deepfactcite/corpus.jsonl PORT=8000 \
  bash scripts/deepfactcite/start_lexical_retriever.sh
```

This server implements Search-R1's `/retrieve` API and returns documents with `contents`, `url`, and `score`.

## SFT

Fast MVP:

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite
conda activate /root/autodl-tmp/conda_envs/searchr1-qwen3-sft

DRY_RUN=1 MODEL_SIZE=4B N_GPUS=1 TOTAL_STEPS=2 \
  bash scripts/deepfactcite/train_sft_qwen3.sh

MODEL_SIZE=4B N_GPUS=1 TOTAL_STEPS=200 \
  bash scripts/deepfactcite/train_sft_qwen3.sh
```

The Qwen3 SFT script defaults to `ATTN_IMPLEMENTATION=sdpa`, which works without compiling flash-attn. On a CUDA image with a working flash-attn install, override `ATTN_IMPLEMENTATION=flash_attention_2`.

Stronger run:

```bash
MODEL_SIZE=8B N_GPUS=2 TOTAL_STEPS=300 \
  bash scripts/deepfactcite/train_sft_qwen3.sh
```

If local model paths differ, override `BASE_MODEL`.

## GRPO

The reward, data, and config path for Qwen3 GRPO are prepared in this fork, but the original Search-R1 rollout stack is not Qwen3-compatible out of the box. The script has a preflight guard that stops local Qwen3 checkpoints unless `ALLOW_EXPERIMENTAL_QWEN3_GRPO=1` is set after the rollout stack has been ported.

After that port, start the retriever first, then run:

```bash
MODEL_SIZE=4B N_GPUS=2 TOTAL_STEPS=200 \
  bash scripts/deepfactcite/train_grpo_deepfactcite_qwen3.sh
```

For 8B:

```bash
MODEL_SIZE=8B N_GPUS=4 TOTAL_STEPS=200 \
  bash scripts/deepfactcite/train_grpo_deepfactcite_qwen3.sh
```

Outcome-only ablation can be approximated by disabling the citation/support weights:

```bash
CITATION_WEIGHT=0 SUPPORT_WEIGHT=0 ANSWER_WEIGHT=0.7 FORMAT_WEIGHT=0.2 SEARCH_WEIGHT=0.1 \
  MODEL_SIZE=4B N_GPUS=2 bash scripts/deepfactcite/train_grpo_deepfactcite_qwen3.sh
```

## Reward

The DeepFactCite reward combines:

```text
answer exact/substr match
citation precision
citation support
format validity
search validity
length/cost penalty
```

It includes hard caps for protocol violations and unsupported citations. By default, citation support uses a deterministic lexical entailment proxy so GRPO does not depend on an external judge server. To enable an OpenAI-compatible judge:

```bash
export DEEPFACTCITE_JUDGE_BASE_URL=http://127.0.0.1:8001
export DEEPFACTCITE_JUDGE_MODEL=Qwen2.5-32B-Instruct
USE_JUDGE=True bash scripts/deepfactcite/train_grpo_deepfactcite_qwen3.sh
```

## Interview Story

The short version:

```text
I built DeepFactCite-GRPO on top of Search-R1. Search-R1 optimizes search-augmented answer correctness, but final-answer rewards can allow answer-correct, citation-unsupported trajectories. I added a citation-faithful reward that checks whether inline markdown citations are copied from retrieved evidence and whether the cited text supports the claim. The training pipeline has SFT cold-start, outcome-only GRPO, and citation-aware GRPO ablations.
```
