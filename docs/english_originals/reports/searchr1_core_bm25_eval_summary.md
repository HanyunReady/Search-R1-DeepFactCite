# Search-R1 Core BM25 Guardrail Summary

Date: 2026-05-18

## Setup

- Dataset: `data/searchr1_core_guardrail/test.parquet`
- Size: 200 rows, NQ 100 + HotpotQA 100
- Retriever: wiki-18 BM25, `data/wiki-18-bm25-index/bm25`
- Corpus: `data/wiki-18-corpus/wiki_dump.jsonl`
- Serving: vLLM dynamic LoRA bf16, tensor parallel 2
- Models: `base`, `soft`, `mixclean`
- Decoding: `temperature=0.0`, `max_turns=4`, `topk=3`
- Runner: `scripts/deepfactcite/run_searchr1_core_bm25_eval_suite.sh`

This is a same-environment answer/search guardrail. It is not the official
Search-R1 paper E5-index reproduction.

## Results

| Model | Answer subEM | Exact | Answer Presence | Search Success | Search Turns | Empty Answer | Budget Fail | No Action |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B | 0.290 | 0.135 | 0.650 | 0.960 | 1.720 | 0.350 | 0.170 | 0.015 |
| Soft SFT 100 | 0.450 | 0.060 | 0.925 | 0.990 | 1.195 | 0.075 | 0.045 | 0.010 |
| MixClean SFT 200 | 0.490 | 0.030 | 0.955 | 1.000 | 1.160 | 0.045 | 0.025 | 0.000 |

By source:

| Model | NQ subEM | NQ Search | NQ Budget Fail | HotpotQA subEM | HotpotQA Search | HotpotQA Budget Fail |
|---|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B | 0.330 | 0.970 | 0.190 | 0.250 | 0.950 | 0.150 |
| Soft SFT 100 | 0.490 | 1.000 | 0.040 | 0.410 | 0.980 | 0.050 |
| MixClean SFT 200 | 0.500 | 1.000 | 0.030 | 0.480 | 1.000 | 0.020 |

## Interpretation

MixClean SFT 200 passes the Search-R1-style answer/search guardrail in this
BM25-controlled setup. It improves answer subEM and completion reliability over
both Base and Soft SFT 100. Average search turns do not spike.

Do not compare these numbers directly to the Search-R1 paper table as a model
quality claim, because the retriever is BM25 rather than the paper's E5 setup.
Use them as a local controlled guardrail: same backbone, same retriever, same
prompt/evaluator, only model weights differ.

## Decision

Use MixClean SFT 200 as the current SFT baseline candidate and likely GRPO
initialization. The remaining weakness is not answer/search behavior; it is
credible citation enforcement on Strict47, especially unsupported and fake URL
rates.
