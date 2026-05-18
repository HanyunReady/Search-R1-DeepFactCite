# Storage Cleanup Candidates - Current State

Date: 2026-05-18

Current disk:

```text
/root/autodl-tmp: 186G total, 122G used, 65G free, 66% used
/tmp overlay:     30G total,  26G used, 4.1G free, 87% used
```

Note:

```text
Already removed:
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged-fp32

Freed about 30G. This was the Soft100 fp32 merged full model, not the current
MixClean200 GRPO baseline. The Soft100 LoRA adapter and merge script remain.

/root/autodl-tmp/LLM-qwen3_posttrain/outputs

Freed about 21G. This was the old Qwen3-4B post-training outputs/checkpoints
directory, not the current DeepFactCite/Search-R1 8B path.
```

No further large directories should be deleted without explicit confirmation.

## Priority Cleanup Candidates

| Priority | Path | Size | Risk | Recommendation |
|---|---:|---:|---|---|
| A | `/root/autodl-tmp/.cache/wheelhouse` | 3.3G | low/medium | Good first cleanup; installer cache, not experiment output |
| A | `/root/autodl-tmp/wheelhouse` | 763M | low/medium | Good first cleanup if offline reinstall is not needed |
| A | `/root/autodl-tmp/.cache/pip` | 303M | low | Safe cache cleanup |
| C | `/root/autodl-tmp/Search-R1-DeepFactCite/data/.cache/hf_datasets` | 13G | medium/high | Rebuildable cache, but deleting may slow/rebreak dataset loading |
| C | `/root/autodl-tmp/Search-R1-DeepFactCite/data/wiki-18-corpus` | 19G | high | Needed for Search-R1 BM25/corpus reproducibility; do not delete unless migration requires it |
| C | `/root/autodl-tmp/Search-R1-DeepFactCite/data/wiki-18-bm25-index` | 2.2G | high | Needed for BM25 eval speed/reproducibility |

## Do Not Delete For Current DeepFactCite Path

| Path | Size | Why keep |
|---|---:|---|
| `/root/autodl-tmp/Search-R1-DeepFactCite/outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16` | 16G | current GRPO actor model |
| `/root/autodl-tmp/Search-R1-DeepFactCite/outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200` | 349M | current SFT adapter lineage |
| `/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base` | 16G | base model used by current eval/SFT lineage |
| `/root/autodl-tmp/conda_envs/grpo-sglang` | 9.6G | current GRPO/SGLang environment |
| `/root/autodl-tmp/conda_envs/openrlhf_vllm085` | 11G | current vLLM eval environment |
| `/root/autodl-tmp/Search-R1-DeepFactCite/logs/grpo/rollouts` | 1.2M | small and valuable rollout evidence |
| `/root/autodl-tmp/Search-R1-DeepFactCite/reports` | 6.1M | small and valuable experiment record |

## Suggested Confirmation Sets

Conservative cache cleanup:

```text
/root/autodl-tmp/.cache/wheelhouse
/root/autodl-tmp/wheelhouse
/root/autodl-tmp/.cache/pip
```

Expected gain: about 4.4G.

Duplicate audit:

```text
Checked 15 files larger than 50MB under:
  /root/autodl-tmp/.cache/wheelhouse
  /root/autodl-tmp/wheelhouse
  /root/autodl-tmp/.cache/pip

sha256 duplicate groups: 0
```

Decision:

```text
Do not delete these cache/wheelhouse directories unless extra space is needed.
They are cache artifacts, but the large files are not exact duplicates.
```

Old 4B output cleanup: completed.

Dataset-cache cleanup:

```text
/root/autodl-tmp/Search-R1-DeepFactCite/data/.cache/hf_datasets
```

Expected gain: about 13G. Use only if the dataset cache is rebuildable on the
target machine.

## Migration Note

If the goal is to migrate this project to another GPU machine, the heavy
minimum set to preserve for the current DeepFactCite path is:

```text
Search-R1-DeepFactCite repo code/docs/reports/logs
data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200
/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base
/root/autodl-tmp/conda_envs/grpo-sglang or a reproducible env spec
```

The wiki corpus/index are needed for full Search-R1 BM25 evaluation, but not
for the v3 GRPO data itself because that run uses its own small offline corpus.
