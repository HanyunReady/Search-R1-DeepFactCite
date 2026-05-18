# DeepFactCite mix50 SFT Eval Summary

Date: 2026-05-18

## Run

- Parent: `outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100`
- Merged parent: `outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged`
- New continuation adapter: `outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged-mix-50/global_step_50`
- Train data: `data/deepfactcite_mix/sft`
- Train steps: 50
- Final val loss: 1.016
- Train exit code: 0

## Fixed Eval Sets

- ShortQA guardrail: `data/shortqa_guardrail/rl/test.parquet`, 32 rows, has gold answers.
- DeepFactCite strict: `data/deepfactcite_strict/sft/test.parquet`, 47 rows, no gold answer; answer_subem is not meaningful here.

## ShortQA Guardrail

| Model | Answer subEM | Total | URL Validity | Citation Precision | Claim Support | Unsupported Rate | Search Turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B | 0.219 | 0.189 | 0.031 | 0.041 | 0.031 | 0.875 | 2.500 |
| Soft SFT 100 | 0.469 | 0.368 | 0.750 | 0.232 | 0.232 | 0.661 | 1.125 |
| Strict SFT 100 | 0.469 | 0.370 | 0.781 | 0.237 | 0.234 | 0.635 | 1.000 |
| Soft merged parent | 0.438 | 0.444 | 0.844 | 0.375 | 0.375 | 0.479 | 1.125 |
| Soft merged + mix50 | 0.375 | 0.324 | 0.625 | 0.190 | 0.190 | 0.750 | 1.156 |

## DeepFactCite Strict Citation Eval

| Model | Total | Cite Presence | URL Validity | Citation Precision | Claim Support | Unsupported Rate | Fake URL | Search Turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B | 0.124 | 0.064 | 0.064 | 0.082 | 0.028 | 0.433 | 0.000 | 1.298 |
| Soft SFT 100 | 0.158 | 0.532 | 0.461 | 0.092 | 0.088 | 0.835 | 0.071 | 0.894 |
| Strict SFT 100 | 0.137 | 0.383 | 0.282 | 0.071 | 0.063 | 0.797 | 0.101 | 0.894 |
| Soft merged parent | 0.138 | 0.277 | 0.234 | 0.070 | 0.055 | 0.762 | 0.043 | 0.979 |
| Soft merged + mix50 | 0.150 | 0.340 | 0.312 | 0.095 | 0.087 | 0.778 | 0.028 | 0.894 |

## Decision

`mix50` is not a winner. It improves some citation metrics over the `soft_merged_parent` on the strict citation eval, but it is still worse than the original `Soft SFT 100` aggregate report on URL validity and cite presence, and it substantially regresses ShortQA answer_subEM and citation support compared with both `Soft SFT 100` and the merged parent.

Do not continue training from `mix50` by simply adding more steps.

Additional root-cause check:

- The original `soft` adapter now has reproducible current-code JSONL reports:
  - `reports/soft_current_qwen3_8b_vllm_shortqa_guardrail32.jsonl`
  - `reports/soft_current_qwen3_8b_vllm_deepfactcite_strict_sft_test47.jsonl`
- Current `Base + soft LoRA` remains stronger than the bf16 merged parent on the strict citation eval.
- HF logits check showed `PEFT forward(Base+LoRA)` vs `merge_and_unload()` in bf16 is not numerically equivalent: max absolute logit differences were 3.328 and 8.547 on two prompts.
- The same check in fp32 is effectively equivalent: max absolute logit differences were below 0.0002.
- Therefore `outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged` is a lossy bf16 merge and should not be used as a training parent.

Recommended next step:

1. Treat `Soft SFT 100` as the current best SFT baseline until proven otherwise.
2. If a full-model parent is needed, remake the merge in fp32; do not use the bf16 merged parent.
3. If continuing SFT, avoid a blind strict-heavy continuation. Use a smaller learning rate and a much lighter strict ratio, or train/evaluate direct from the original soft adapter path if the trainer supports adapter resume without full merge.
4. Keep the fixed guardrails: ShortQA 32 for answer/search and DeepFactCite strict 47 for citation behavior.
