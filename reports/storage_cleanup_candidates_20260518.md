# Storage Cleanup Candidates

Date: 2026-05-18

Scope: `/root/autodl-tmp/` only. No files have been deleted.

Current disk state:

```text
/root/autodl-tmp: 186G total, 170G used, 17G free, 92% used
```

## Recommended Cleanup Sets

### Set A: Old `agentic-rl-searchqa` merged full checkpoints

Potential free space: about 48G.

```text
/root/autodl-tmp/output/agentic-rl-searchqa/sft-final-ckpt471-merged        16G
/root/autodl-tmp/output/agentic-rl-searchqa/grpo-ckpt10-merged-hf          16G
/root/autodl-tmp/output/agentic-rl-searchqa/grpo-stage2-en30-ckpt30-merged-hf 16G
```

Assessment:

```text
Best cleanup target if these old full merged models are no longer needed for
direct re-eval. The old project still has adapter checkpoints under:

/root/autodl-tmp/output/agentic-rl-searchqa/verified_checkpoints
/root/autodl-tmp/output/agentic-rl-searchqa/sft-qwen3-8b-lora-bs4x4-20260517_105357

Reports and rollout JSONL are also under /root/autodl-tmp/agentic-rl-searchqa.
```

Risk:

```text
Medium. Deleting these removes convenient full-model serving artifacts. They
may be recreatable from adapters/base if lineage is intact, but recreation costs
time and should not be assumed unless tested.
```

Recommendation:

```text
Delete only if we do not need to rerun old SFT/GRPO merged-model eval.
Keep if the old project needs direct reproduction from merged HF models.
```

### Set B: DeepFactCite fp32 merged Soft parent

Potential free space: about 31G.

```text
/root/autodl-tmp/Search-R1-DeepFactCite/outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged-fp32
```

Assessment:

```text
This model was created to fix the bf16 merge problem and is a valid fp32 merged
parent. The canonical SFT baseline is still dynamic LoRA:

Base Qwen3-8B + outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100

The fp32 merged parent can be recreated from the base model and Soft SFT LoRA
using scripts/deepfactcite/merge_lora_adapter.py.
```

Risk:

```text
Medium. Safe for final eval if we standardize on dynamic LoRA. Less safe if we
plan immediate continuation SFT from this fp32 merged parent.
```

Recommendation:

```text
Keep until we decide whether the next SFT continuation needs a full parent.
Delete only if disk pressure blocks Search-R1 benchmark data/eval.
```

### Set C: Rebuildable caches under `/root/autodl-tmp`

Potential free space: about 4.3G.

```text
/root/autodl-tmp/.cache/wheelhouse                         3.3G
/root/autodl-tmp/wheelhouse                                763M
/root/autodl-tmp/.cache/pip                                303M
```

Assessment:

```text
These are installer/cache artifacts, not experiment outputs.
```

Risk:

```text
Low to medium. They are usually rebuildable, but deleting wheelhouses can slow
future environment setup or offline reinstall.
```

Recommendation:

```text
Good first cleanup if we need a few GB without touching model checkpoints.
Not enough alone for full wiki-18/e5 data.
```

### Set D: Old 4B post-training outputs

Potential free space: up to about 21G, depending on what is still needed.

```text
/root/autodl-tmp/LLM-qwen3_posttrain/outputs/checkpoints/qwen3_4b_aime_rlvr_dense_smoke_60_g8_ep1_offload_seed1 8.3G
/root/autodl-tmp/LLM-qwen3_posttrain/outputs/checkpoints/qwen3_4b_math_sft_v6_native_eos_10k_merged_hf          7.6G
/root/autodl-tmp/LLM-qwen3_posttrain/outputs/qwen3_4b_math_sft_v6_native_eos_10k_seed1                         2.6G
/root/autodl-tmp/LLM-qwen3_posttrain/outputs/qwen3_4b_math_sft_v6_native_eos_probe2k_seed1                     1.8G
```

Assessment:

```text
These appear unrelated to the current DeepFactCite 8B path. However, the
Qwen3-4B base model in this repo is still a fallback asset:

/root/autodl-tmp/LLM-qwen3_posttrain/.cache/models/Qwen_Qwen3-4B-Base 7.6G
```

Risk:

```text
Medium to high unless the old 4B post-training project is no longer needed.
Do not delete the Qwen3-4B base model if we still want a 4B fallback.
```

Recommendation:

```text
Only delete old outputs/checkpoints after confirming they are not needed.
Keep the 4B base cache.
```

## Do Not Delete Without Strong Reason

```text
/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base 16G
```

Reason:

```text
This is the main Qwen3-8B base model used by current DeepFactCite eval/SFT.
```

```text
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft
/root/autodl-tmp/conda_envs/openrlhf_vllm085
```

Reason:

```text
These are the currently useful SFT and vLLM eval environments.
```

```text
/root/autodl-tmp/LLM-qwen3_posttrain/.cache/models/Qwen_Qwen3-4B-Base
```

Reason:

```text
This is the 4B fallback model.
```

## Practical Options

Option 1: conservative cleanup.

```text
Delete Set C only.
Expected free space: about 4.3G.
```

Option 2: free enough for benchmark work while keeping current DeepFactCite fp32 parent.

```text
Delete Set A.
Expected free space: about 48G.
```

Option 3: aggressive current-project cleanup.

```text
Delete Set A + Set B + Set C.
Expected free space: about 83G.
Tradeoff: must recreate the DeepFactCite fp32 merged parent if needed.
```

Option 4: old-project cleanup.

```text
Delete selected Set D outputs after confirmation.
Expected free space: up to about 21G.
Tradeoff: affects old Qwen3-4B post-training artifacts.
```
