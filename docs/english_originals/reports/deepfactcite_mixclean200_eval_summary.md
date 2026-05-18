# DeepFactCite MixClean200 SFT Eval Summary

Date: 2026-05-18

## Run

- Parent: Base Qwen3-8B, `/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base`
- Adapter: `outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200/global_step_200`
- Train script: `scripts/deepfactcite/launch_mix_clean_sft_200.sh`
- Train data: `data/deepfactcite_mix/sft`
- Train steps: 200
- Final validation loss: 0.992

## Fixed Serving Path

- vLLM dynamic LoRA, bf16, tensor parallel 2
- Served models: `base`, `soft`, `mixclean`
- Eval script: `scripts/deepfactcite/run_mixclean_sft_eval_suite.sh`
- Decoding: `temperature=0.0`, `max_turns=4`, `topk=3`

This report intentionally compares only models served through the same vLLM
dynamic-LoRA path. Older merged-model results are not mixed into the main table.

## ShortQA Guardrail

| Model | Answer subEM | Total | URL Validity | Citation Precision | Claim Support | Unsupported Rate | Search Turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B | 0.219 | 0.189 | 0.031 | 0.038 | 0.031 | 0.906 | 2.625 |
| Soft SFT 100 | 0.438 | 0.391 | 0.812 | 0.292 | 0.292 | 0.557 | 1.156 |
| MixClean SFT 200 | 0.500 | 0.427 | 0.969 | 0.333 | 0.333 | 0.495 | 1.125 |

## DeepFactCite Strict Citation Eval

This set has no gold answer. `answer_subem=0` is not meaningful here.

| Model | Total | Cite Presence | URL Validity | Citation Precision | Claim Support | Unsupported Rate | Fake URL | Search Turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B | 0.120 | 0.064 | 0.064 | 0.062 | 0.021 | 0.574 | 0.000 | 1.745 |
| Soft SFT 100 | 0.137 | 0.383 | 0.319 | 0.070 | 0.064 | 0.851 | 0.064 | 1.000 |
| MixClean SFT 200 | 0.142 | 0.489 | 0.418 | 0.074 | 0.072 | 0.879 | 0.071 | 1.234 |

## Decision

MixClean SFT 200 is the current SFT candidate. It improves ShortQA answer,
URL validity, citation precision, claim support, unsupported rate, and search
turns over Soft SFT 100 under the same vLLM dynamic-LoRA path. It also improves
Strict47 URL validity and claim support.

It is not the final credible-citation result. On Strict47, unsupported and fake
URL rates remain high because the model cites more aggressively. This is exactly
why the next stage needs citation-authenticity and claim-support rewards, with
hard penalties for fake or unsupported citations.

## Search-R1 Core BM25 Guardrail

This is not the paper's official E5 setup. It is a controlled same-environment
guardrail using:

- `data/searchr1_core_guardrail/test.parquet`, 200 rows
- NQ 100 + HotpotQA 100
- wiki-18 BM25 index
- extracted wiki corpus: `data/wiki-18-corpus/wiki_dump.jsonl`
- same vLLM dynamic-LoRA bf16 serving path

| Model | Answer subEM | Exact | Search Success | Search Turns | Empty Answer | Budget Fail | NQ subEM | HotpotQA subEM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B | 0.290 | 0.135 | 0.960 | 1.720 | 0.350 | 0.170 | 0.330 | 0.250 |
| Soft SFT 100 | 0.450 | 0.060 | 0.990 | 1.195 | 0.075 | 0.045 | 0.490 | 0.410 |
| MixClean SFT 200 | 0.490 | 0.030 | 1.000 | 1.160 | 0.045 | 0.025 | 0.500 | 0.480 |

Guardrail decision:

- MixClean SFT 200 does not regress answer/search behavior versus Base or
  Soft100 on this BM25-controlled Search-R1 subset.
- It improves subEM versus Base by +20.0 points and versus Soft100 by +4.0
  points.
- It reduces budget-fail rate versus Base by 14.5 points and versus Soft100 by
  2.0 points.
- Exact match is lower for SFT models because answers are longer and more
  citation/format-oriented. Use subEM as the primary guardrail metric here, and
  keep exact as a secondary warning metric.

## Next Checks

1. Promote MixClean SFT 200 as the current SFT baseline candidate.
2. Use MixClean SFT 200 as the likely initialization for outcome-only vs
   citation-aware GRPO.
3. In GRPO, explicitly cap or hard-penalize fake URLs and unsupported
   citations, because SFT alone still produces high Strict47 unsupported/fake
   rates.
4. Keep the BM25 guardrail as a reproducible local check, but label it clearly
   as BM25-controlled rather than official Search-R1 E5 reproduction.
