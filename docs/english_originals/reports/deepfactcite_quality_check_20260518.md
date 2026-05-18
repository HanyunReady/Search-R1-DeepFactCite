# DeepFactCite Quality Check

Date: 2026-05-18

## Artifact Checks

- MixClean SFT adapter exists:
  `outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200/global_step_200/adapter_model.safetensors`
- Adapter size: 334MB
- wiki-18 corpus extracted:
  `data/wiki-18-corpus/wiki_dump.jsonl`
- Extracted corpus size: 14GB
- BM25/Datasets cache stored inside repo:
  `data/.cache`

## Eval Completeness Checks

ShortQA guardrail:

- `reports/base_qwen3_8b_mixclean_rerun_vllm_shortqa_guardrail32.jsonl`: 32 rows
- `reports/soft100_qwen3_8b_mixclean_rerun_vllm_shortqa_guardrail32.jsonl`: 32 rows
- `reports/mixclean200_qwen3_8b_vllm_shortqa_guardrail32.jsonl`: 32 rows

Strict citation eval:

- `reports/base_qwen3_8b_mixclean_rerun_vllm_deepfactcite_strict_sft_test47.jsonl`: 47 rows
- `reports/soft100_qwen3_8b_mixclean_rerun_vllm_deepfactcite_strict_sft_test47.jsonl`: 47 rows
- `reports/mixclean200_qwen3_8b_vllm_deepfactcite_strict_sft_test47.jsonl`: 47 rows

Search-R1 core BM25 guardrail:

- `reports/base_qwen3_8b_searchr1_core_bm25_200.jsonl`: 200 rows, NQ 100 + HotpotQA 100, indices 0-199
- `reports/soft100_qwen3_8b_searchr1_core_bm25_200.jsonl`: 200 rows, NQ 100 + HotpotQA 100, indices 0-199
- `reports/mixclean200_qwen3_8b_searchr1_core_bm25_200.jsonl`: 200 rows, NQ 100 + HotpotQA 100, indices 0-199

## Fairness Checks

- Base, Soft100, and MixClean200 were served through the same vLLM instance.
- Serving mode was dynamic LoRA bf16 with tensor parallel 2.
- Decoding was fixed: `temperature=0.0`, `max_turns=4`, `topk=3`.
- Search-R1 core guardrail used one retriever for all models:
  `http://127.0.0.1:8000/retrieve`, wiki-18 BM25.
- The Search-R1 core result is labeled BM25-controlled and is not presented as
  official Search-R1 E5 reproduction.

## Result Checks

MixClean200 improves over Base and Soft100 on the local Search-R1 BM25 answer
guardrail:

- Base subEM: 0.290
- Soft100 subEM: 0.450
- MixClean200 subEM: 0.490

MixClean200 improves over Soft100 on ShortQA32:

- answer_subem: 0.500 vs 0.438
- URL validity: 0.969 vs 0.812
- claim support: 0.333 vs 0.292
- unsupported rate: 0.495 vs 0.557

MixClean200 improves over Soft100 on Strict47 URL/support:

- URL validity: 0.418 vs 0.319
- citation precision: 0.074 vs 0.070
- claim support: 0.072 vs 0.064

Remaining weakness:

- Strict47 unsupported rate is still high: 0.879 for MixClean200.
- Strict47 fake URL rate is still non-zero: 0.071 for MixClean200.
- Exact match on Search-R1 core is lower for SFT models than Base, likely
  because SFT answers are longer and more formatted. Use subEM as the primary
  guardrail and keep exact as a secondary warning metric.

## Decision

MixClean200 is good enough to promote from "candidate run" to the current SFT
baseline candidate. It passes the local Search-R1 answer/search guardrail and
improves citation behavior, but SFT alone does not solve credible citations.

Next action should be GRPO, not more SFT:

1. Use MixClean200 as initialization.
2. Run outcome-only GRPO as the Search-R1-style ablation.
3. Run citation-aware GRPO with hard fake-URL and unsupported-citation caps.
4. Keep ShortQA/Search-R1 guardrails in the eval suite to catch answer/search
   regressions.
