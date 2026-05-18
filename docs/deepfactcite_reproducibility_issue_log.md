# DeepFactCite Reproducibility Issue Log

Date: 2026-05-18

This note tracks issues found while evaluating Qwen3-8B Soft SFT, merged checkpoints, and continuation runs. The goal is to prevent mixing incompatible evaluation paths and to make every reported number reproducible.

## Success Standard

The final project deliverable must include both:

1. A result table that meets the intended target: preserve Search-R1-style
   answer/search behavior while improving DeepFactCite citation authenticity
   and claim support.
2. A failure ledger that explains every failed or non-winning SFT attempt well
   enough for external review.

Training completion is not enough. A run only counts as useful evidence if its
artifact path, serving path, dataset, metrics, and failure analysis are
recoverable from this log or the linked report.

## Current Baselines

Fixed eval sets:

- ShortQA guardrail: `data/shortqa_guardrail/rl/test.parquet`, 32 rows, has gold answers.
- DeepFactCite strict citation eval: `data/deepfactcite_strict/sft/test.parquet`, 47 rows, no gold answer; `answer_subem` is not meaningful.

Current reports:

| Model / Serving Path | Eval | Answer | Total | URL Validity | Citation Precision | Claim Support | Unsupported | Search Turns |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B, vLLM bf16 | ShortQA32 | 0.219 | 0.189 | 0.031 | 0.038 | 0.031 | 0.906 | 2.625 |
| `Base + soft LoRA`, vLLM bf16 dynamic LoRA | ShortQA32 | 0.438 | 0.391 | 0.812 | 0.292 | 0.292 | 0.557 | 1.156 |
| `Base + mixclean200 LoRA`, vLLM bf16 dynamic LoRA | ShortQA32 | 0.500 | 0.427 | 0.969 | 0.333 | 0.333 | 0.495 | 1.125 |
| `soft` bf16 merged full model | ShortQA32 | 0.438 | 0.444 | 0.844 | 0.375 | 0.375 | 0.479 | 1.125 |
| `soft` fp32 merged full model, vLLM fp32 | ShortQA32 | 0.406 | 0.361 | 0.688 | 0.240 | 0.237 | 0.615 | 1.000 |
| `soft` bf16 merged + mix50 LoRA | ShortQA32 | 0.375 | 0.324 | 0.625 | 0.190 | 0.190 | 0.750 | 1.156 |
| Base Qwen3-8B, vLLM bf16 | Strict47 | N/A | 0.120 | 0.064 | 0.062 | 0.021 | 0.574 | 1.745 |
| `Base + soft LoRA`, vLLM bf16 dynamic LoRA | Strict47 | N/A | 0.137 | 0.319 | 0.070 | 0.064 | 0.851 | 1.000 |
| `Base + mixclean200 LoRA`, vLLM bf16 dynamic LoRA | Strict47 | N/A | 0.142 | 0.418 | 0.074 | 0.072 | 0.879 | 1.234 |
| `soft` bf16 merged full model | Strict47 | N/A | 0.138 | 0.234 | 0.070 | 0.055 | 0.762 | 0.979 |
| `soft` bf16 merged + mix50 LoRA | Strict47 | N/A | 0.150 | 0.312 | 0.095 | 0.087 | 0.778 | 0.894 |

Decision so far:

- The current best SFT candidate is `Base + mixclean200 LoRA` served as a dynamic LoRA adapter in vLLM bf16.
- `mixclean200` is better than `soft100` on ShortQA32 answer, URL validity, citation precision, claim support, unsupported rate, and search turns under the same serving path.
- `mixclean200` is also better than `soft100` on Strict47 URL validity, citation precision, and claim support, but its Strict47 unsupported and fake URL rates are still high. This is a candidate SFT initialization, not the final credible-citation claim.
- `mixclean200` passed the Search-R1 core BM25 guardrail: 200-row NQ/HotpotQA subEM 0.490 versus `soft100` 0.450 and Base 0.290, with search success 1.000 and no search-turn spike.
- `mix50` is not a winner and should not be continued.
- The old bf16 merged parent must not be used as a continuation parent.

## SFT Failure Ledger

### F-SFT-001: Strict SFT did not automatically beat Soft SFT

Hypothesis:

```text
Filtering SFT traces for URL validity and claim support should improve citation
behavior and become the default SFT winner.
```

Artifacts:

```text
Soft adapter:
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100

Strict adapter:
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-strict-100/global_step_100
```

Observed result:

```text
ShortQA32:
  Soft dynamic LoRA:   answer 0.469, URL validity 0.812, claim support 0.240
  Strict dynamic LoRA: answer 0.469, URL validity 0.781, claim support 0.234

Strict47:
  Soft dynamic LoRA:   total 0.156, URL validity 0.346, claim support 0.103
  Strict dynamic LoRA: total 0.137, URL validity 0.282, claim support 0.063
```

Analysis:

Strict filtering improved the intended data cleanliness constraint, but it also
reduced training diversity and may have narrowed answer/search behavior coverage.
For this stage, strict data is a useful ablation, not the default winner.

Decision:

```text
Do not promote strict SFT unless it wins under the same serving path and eval.
Keep Soft SFT 100 dynamic LoRA as the current SFT baseline.
```

Guardrail:

```text
Any future "higher quality" SFT dataset must be evaluated against Soft SFT 100
on both answer/search guardrail data and DeepFactCite citation data before more
training is launched.
```

### F-SFT-002: bf16 LoRA merge was not equivalent to dynamic LoRA

Hypothesis:

```text
Base Qwen3-8B + Soft LoRA dynamic serving should be equivalent to a merged Soft
full model, so the merged model can be used as a continuation parent.
```

Artifacts:

```text
Dynamic LoRA:
Base model + outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100

Old bf16 merged full model:
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged
```

Observed result:

```text
HF logits:
  PEFT forward vs in-memory bf16 merge:
    max abs logit diff = 3.328 / 8.547 on two prompts
  in-memory bf16 merge vs saved bf16 merge:
    max abs logit diff = 0.0
  PEFT forward vs in-memory fp32 merge:
    max abs logit diff < 0.0002
```

Analysis:

The saved bf16 merged model was internally consistent, but it was not equivalent
to PEFT dynamic LoRA forward. The failure came from adding fp32 LoRA deltas into
bf16 base weights during `merge_and_unload()`.

Decision:

```text
Delete/retire the old bf16 merged parent.
Patch merge_lora_adapter.py so float32 is the default merge dtype.
Use HF logits equivalence checks before trusting merged parents.
```

Guardrail:

```text
Merged checkpoints are not valid continuation parents until PEFT-vs-merged HF
logits are checked and documented.
```

### F-SFT-003: mix50 continuation trained from an invalid parent

Hypothesis:

```text
Continue SFT for 50 steps from the merged Soft parent on mix data to improve
or stabilize citation behavior.
```

Artifacts:

```text
Original mix50 adapter:
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged-mix-50/global_step_50

Parent:
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged
```

Observed result:

```text
Training completed:
  train steps = 50
  val/loss = 1.016
  exit code = 0

ShortQA32:
  answer 0.375, total 0.324, URL validity 0.625, claim support 0.190

Strict47:
  total 0.150, URL validity 0.312, claim support 0.087
```

Analysis:

The training run itself did not crash, but its parent checkpoint was the lossy
bf16 merged model from F-SFT-002. Therefore the result is not a clean answer to
whether mix-data continuation helps.

Decision:

```text
Do not continue from mix50.
Do not report mix50 as a candidate winner.
Keep only as a failure/ablation record.
```

Guardrail:

```text
Before continuation SFT, verify the parent model and record the exact lineage:
base -> adapter or fp32 merged parent -> continuation adapter.
```

### S-SFT-001: mix-clean 200 became the current SFT candidate, but not the final answer

Hypothesis:

```text
Train a clean mixed SFT adapter directly from Base Qwen3-8B for 200 steps,
instead of continuing from a merged Soft parent, so answer/search guardrail
behavior is preserved while citation behavior improves.
```

Artifacts:

```text
Train script:
scripts/deepfactcite/launch_mix_clean_sft_200.sh

Adapter:
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200/global_step_200

Eval launcher:
scripts/deepfactcite/wait_then_eval_mixclean_sft.sh
scripts/deepfactcite/launch_vllm_base_soft_mixclean.sh
scripts/deepfactcite/run_mixclean_sft_eval_suite.sh

Reports:
reports/base_qwen3_8b_mixclean_rerun_vllm_shortqa_guardrail32.json
reports/base_qwen3_8b_mixclean_rerun_vllm_deepfactcite_strict_sft_test47.json
reports/soft100_qwen3_8b_mixclean_rerun_vllm_shortqa_guardrail32.json
reports/soft100_qwen3_8b_mixclean_rerun_vllm_deepfactcite_strict_sft_test47.json
reports/mixclean200_qwen3_8b_vllm_shortqa_guardrail32.json
reports/mixclean200_qwen3_8b_vllm_deepfactcite_strict_sft_test47.json
```

Observed result:

```text
ShortQA32:
  Base:     answer 0.219, total 0.189, URL 0.031, support 0.031, unsupported 0.906
  Soft100:  answer 0.438, total 0.391, URL 0.812, support 0.292, unsupported 0.557
  Mix200:   answer 0.500, total 0.427, URL 0.969, support 0.333, unsupported 0.495

Strict47:
  Base:     total 0.120, URL 0.064, precision 0.062, support 0.021, unsupported 0.574, fake 0.000
  Soft100:  total 0.137, URL 0.319, precision 0.070, support 0.064, unsupported 0.851, fake 0.064
  Mix200:   total 0.142, URL 0.418, precision 0.074, support 0.072, unsupported 0.879, fake 0.071
```

Analysis:

The run is a meaningful SFT improvement under a controlled dynamic-LoRA bf16
serving path. It improves ShortQA answer/search guardrail metrics and improves
URL validity/support on both eval sets. However, Strict47 unsupported and fake
URL rates remain high because the model cites more aggressively. This supports
the original project thesis: SFT can teach citation behavior, but credible
citation requires reward-time authenticity and claim-support constraints.

Decision:

```text
Use mixclean200 as the current SFT initialization candidate for the next stage.
Do not claim final credible-citation success from SFT alone.
Next stage must add hard fake/unsupported citation penalties and run
outcome-only vs citation-aware GRPO.
```

Search-R1 answer/search guardrail:

```text
Dataset:
  data/searchr1_core_guardrail/test.parquet
  200 rows = NQ 100 + HotpotQA 100

Retriever:
  wiki-18 BM25 + extracted wiki_dump.jsonl

Results:
  Base:     subEM 0.290, search 0.960, budget fail 0.170
  Soft100:  subEM 0.450, search 0.990, budget fail 0.045
  Mix200:   subEM 0.490, search 1.000, budget fail 0.025
```

This guardrail is BM25-controlled, not the paper's official E5 reproduction.
It is valid for same-backbone local comparison because only the served model
changes.

### F-SFT-004: Answer metrics were initially interpreted on a no-gold citation set

Hypothesis:

```text
DeepFactCite strict47 can be used for both answer/search and citation metrics.
```

Observed result:

```text
data/deepfactcite_strict/sft/test.parquet has no gold answer target.
answer_subem from this set is not meaningful.
```

Analysis:

DeepFactCite strict47 is a citation-behavior eval set. It can measure citation
presence, URL validity, precision, claim support, unsupported citation rate,
fake URL rate, search turns, and response length. It cannot support a reliable
answer_subem conclusion without gold answers.

Decision:

```text
Use ShortQA32 and Search-R1 NQ/HotpotQA-style data for answer/search guardrails.
Use strict47 for citation behavior only.
```

Guardrail:

```text
Every eval table must label whether answer_subem is applicable for that dataset.
```

### F-SFT-005: Serving path changes closed-loop search-agent metrics

Hypothesis:

```text
If two checkpoints are close at the logits level, greedy vLLM eval should give
the same aggregate metrics.
```

Observed result:

```text
Soft dynamic LoRA bf16, soft bf16 merged full model, and soft fp32 merged full
model produced different aggregate metrics under closed-loop search eval.
```

Analysis:

Search-agent eval amplifies small generation differences:

```text
early token difference -> different search query -> different retrieved snippets
-> different answer/citation trajectory -> different metrics
```

Greedy decoding does not remove differences caused by dynamic LoRA vs merged
weights, dtype, attention backend, or vLLM loading path.

Decision:

```text
Treat serving path as part of the model identity.
Do not mix dynamic-LoRA results with merged-model results in the same baseline.
```

Guardrail:

```text
Every result table must include serving path labels such as:
base_bf16, soft_dynamic_lora_bf16, soft_merged_bf16, soft_merged_fp32.
```

## Confirmed Issues

### 1. bf16 LoRA merge is not numerically equivalent

The old merge script loaded the base model with `torch_dtype=torch.bfloat16`, then called `merge_and_unload()`.

Direct HF logits checks showed:

| Comparison | Max Abs Logit Diff | Top1 |
|---|---:|---|
| `PEFT forward(Base+LoRA)` vs in-memory bf16 merge | 3.328 / 8.547 on two prompts | Same in tested prompts, but logits differ heavily |
| in-memory bf16 merge vs saved bf16 merged model | 0.0 | Same |
| `PEFT forward(Base+LoRA)` vs in-memory fp32 merge | < 0.0002 | Same |

Conclusion:

- Saved bf16 merge was internally consistent, but it was a lossy merge.
- The loss comes from adding the fp32 LoRA delta into bf16 base weights.
- The merge script now defaults to `--torch-dtype float32`.

### 2. Training dtype and merge dtype are different concepts

The SFT trainer loads `partial_pretrain` with `torch_dtype=torch.float32` and then uses FSDP mixed precision:

- parameters loaded in fp32,
- compute/mixed precision in bf16,
- LoRA adapter tensors saved as fp32.

Therefore fp32 merge for a training parent is not a contradiction. It prevents losing the LoRA delta during the merge step. Training can still use bf16 mixed precision.

### 3. vLLM dynamic LoRA does not support float32 LoRA kernels

Attempting to run `Base + soft LoRA` in vLLM with `--dtype float32` failed during LoRA graph profiling:

```text
assert weight.dtype in [torch.float16, torch.bfloat16]
AssertionError
```

So we cannot use vLLM to compare:

- `Base + soft LoRA` in float32 dynamic LoRA mode
- vs `soft` fp32 merged full model

The closest exact equivalence check is HF logits, not vLLM generation.

### 4. vLLM serving path changes output even under greedy decoding

Search-agent evaluation is a closed loop:

1. model emits a search query,
2. retriever returns snippets,
3. model continues from retrieved snippets,
4. answer/citation metrics depend on all previous text.

Small early-token differences can change the search query and therefore the whole trajectory. Greedy decoding does not guarantee identical rollouts across:

- dynamic LoRA vs merged full model,
- bf16 vs fp32,
- FlashAttention vs Triton attention,
- different vLLM model loading paths.

This is why aggregate eval can move by several points even when direct single-step logits look close.

## Variables That Must Be Recorded

For every run, record all of the following:

### Model Identity

- Base model path.
- Adapter path, if any.
- Merged full-model path, if any.
- Whether model is dynamic LoRA or merged.
- Whether merge was bf16 or fp32.
- Adapter `base_model_name_or_path`.
- Model commit/checksum if available.

### Serving Backend

- vLLM version.
- `--dtype`.
- `--tensor-parallel-size`.
- `--max-model-len`.
- `--gpu-memory-utilization`.
- Whether `--enable-lora` is used.
- Attention backend from log: FlashAttention, Triton, etc.
- Whether model was served as base model, LoRA module, or full merged model.
- The exact vLLM startup log path.

### Prompt and Tokenizer

- Tokenizer path passed to eval script.
- Chat template source.
- Prompt construction function/version.
- Stop markers.
- `max_new_tokens`, `temperature`, `top_p` if used.

### Retrieval and Agent Loop

- Eval parquet path.
- Corpus JSONL path.
- Retriever implementation/version.
- `topk`.
- `max_turns`.
- Whether results are resumed from JSONL.
- Search query normalization, if changed.

### Reward and Metrics

- Reward code version.
- Whether prompt tokens are included or response-only is scored.
- Ground-truth schema.
- Whether `answer_subem` is meaningful for that dataset.
- Aggregation script/version.

### Environment

- Conda env.
- torch, transformers, peft, vLLM, verl versions.
- CUDA visible devices.
- GPU type.
- Relevant env vars such as `CUDA_VISIBLE_DEVICES`, `TOKENIZERS_PARALLELISM`, `WANDB_MODE`.

## Non-Mixable Result Types

Do not compare these as if they are the same model:

- `Base + soft LoRA` dynamic vLLM bf16 vs `soft` bf16 merged full model.
- `Base + soft LoRA` dynamic vLLM bf16 vs `soft` fp32 merged full model served as vLLM fp32.
- Old aggregate-only reports vs current JSONL-backed reports.
- Strict47 citation eval vs ShortQA answer guardrail.
- DeepFactCite strict test `answer_subem` vs ShortQA `answer_subem`; strict47 has no gold answer.

## Required Reproducibility Protocol

Every eval must produce:

1. JSONL rollouts.
2. Aggregate JSON.
3. vLLM startup log.
4. Command line used.
5. Git status/diff summary.
6. Dataset/corpus path.
7. Serving path label:
   - `base_bf16`
   - `soft_dynamic_lora_bf16`
   - `soft_merged_bf16`
   - `soft_merged_fp32`
   - etc.

Every comparison table must say which serving path was used.

## Current Safe Operating Rules

1. For current SFT baseline reporting, use `Base + soft LoRA` as dynamic LoRA under vLLM bf16.
2. Do not use the bf16 merged parent as a training parent.
3. If a full parent is needed for continuation SFT, create it with fp32 merge.
4. Do not claim an improvement unless it beats the same baseline under the same serving path and eval script.
5. Keep ShortQA32 as answer/search guardrail and Strict47 as citation-behavior eval.
6. If a result changes unexpectedly, first run a one-prompt HF logits comparison before spending GPU time on full eval.
7. For large Hugging Face artifacts on this machine, run `unvpn` first and prefer domestic mirrors plus `aria2c`; direct HF/Xet transfer was observed to be much slower.
8. Treat `wiki-18.jsonl.gz` as a compressed archive until validated; the downloaded artifact was a gzip-compressed tar payload, not a naked JSONL gzip.

## Infrastructure Notes

### INF-001: Large HF artifact download path matters

Date: 2026-05-18

Context:

- Needed `PeterJinGo/wiki-18-corpus/wiki-18.jsonl.gz` for Search-R1 BM25 docid-to-text lookup.
- The official BM25 index stores only `id`, not raw document contents.
- Direct Hugging Face/Xet download through `aria2c` was initially slow, around sub-MB/s to about 0.7 MB/s.

Resolution:

- User enabled command-line `unvpn`.
- Download was retried through `hf-mirror.com` with `aria2c`.
- Observed speed increased to around 12 MiB/s.

Preferred command pattern:

```bash
unvpn
aria2c -d data/wiki-18-corpus -o wiki-18.jsonl.gz -x 8 -s 8 -j 4 -k 1M \
  --continue=true \
  --file-allocation=none \
  --auto-file-renaming=false \
  --allow-overwrite=true \
  --max-tries=0 \
  --retry-wait=5 \
  --connect-timeout=30 \
  --timeout=60 \
  --summary-interval=10 \
  https://hf-mirror.com/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz
```

Lesson:

- Before starting multi-GB downloads, test the network path and mirror.
- Do not assume `aria2c` alone is enough; route selection dominated throughput in this environment.

### INF-002: wiki-18 corpus file is a tar payload despite the `.jsonl.gz` name

Date: 2026-05-18

Observed:

```text
gzip -t data/wiki-18-corpus/wiki-18.jsonl.gz
  passed

Reading as UTF-8 JSONL:
  UnicodeDecodeError at byte 0x80

First decompressed bytes:
  tar header with ustar marker
```

Actual payload:

```text
data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl
uncompressed size: 14393573105 bytes
record shape: {"id": "...", "contents": "..."}
```

Resolution:

```bash
tar -xOzf data/wiki-18-corpus/wiki-18.jsonl.gz \
  data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl \
  > data/wiki-18-corpus/wiki_dump.jsonl.tmp
mv data/wiki-18-corpus/wiki_dump.jsonl.tmp data/wiki-18-corpus/wiki_dump.jsonl
```

Lesson:

- Validate both compression and logical file format before wiring the corpus
  into a retriever.
- The Search-R1 BM25 server should use
  `data/wiki-18-corpus/wiki_dump.jsonl` as `--corpus-path`, not the downloaded
  `wiki-18.jsonl.gz` archive.

## Open Questions

### GRPO-001: old SGLang backend `custom` reward manager does not use `custom_reward_function.path`

Date: 2026-05-18

Observed:

```text
scripts/train_grpo_2xa800.sh sets:
  reward_model.reward_manager=custom
  custom_reward_function.path=...

But /root/autodl-tmp/SearchShortQA/verl/verl/workers/reward_manager/custom.py
calls agentic_rl_searchqa.rewards.reward_manager.RewardManager directly and
does not call the loaded compute_score function.
```

Risk:

- A run can look like it is using DeepFactCite reward while actually using the
  old project reward.
- This would invalidate outcome-only vs citation-aware GRPO comparisons.

Resolution:

- Keep old backend code unchanged.
- Register `deepfactcite_custom` from
  `scripts/deepfactcite/verl_deepfactcite_reward.py`.
- Launch GRPO with:

```text
reward_model.reward_manager=deepfactcite_custom
custom_reward_function.path=/root/autodl-tmp/Search-R1-DeepFactCite/scripts/deepfactcite/verl_deepfactcite_reward.py
custom_reward_function.name=compute_score
```

Verification from the 2-GPU smoke run:

```text
reward_manager: deepfactcite_custom
using customized reward function 'compute_score' from .../verl_deepfactcite_reward.py
RewardManagerWorker loaded the same path
rollout_data_step_1.jsonl contains DeepFactCite details:
  url_validity, citation_precision, claim_support, unsupported_citation_rate
```

Lesson:

- Never trust Hydra `custom_reward_function.path` alone; inspect the selected
  reward manager implementation and prove the reward source in logs.

### GRPO-002: GRPO smoke rows must be retrieval-hit filtered

Date: 2026-05-18

Observed in the first 2-GPU smoke rollout:

```text
Query: Bach BWV 171 structure/scoring/features
Rollout 1: No relevant search results were found.
Rollout 2: retrieved irrelevant Magnificat / acoustics snippets.
Rewards: 0.08 to 0.10, citation_count=0, claim_support=0.
```

Interpretation:

- The GRPO mechanism is working: weak/no evidence receives low reward.
- As training data, blind `head(24)` long questions are inefficient because the
  offline smoke corpus may not contain or retrieve supporting evidence.

Resolution for the next run:

- Build a `retrieval-hit` GRPO subset before spending longer GPU time.
- Require at least one top-k retrieved URL/text with meaningful lexical overlap
  against the query and expected evidence.
- Keep short QA guardrail rows, but avoid long rows whose retrieved evidence is
  empty or obviously off-topic.

Lesson:

- For citation-faithful RL, data quality means prompt quality plus retriever
  hit quality. A correct reward cannot learn useful citation behavior if the
  environment rarely supplies valid evidence.

### GRPO-003: short tool responses can truncate URLs and break URL validity

Date: 2026-05-18

Observed in the first 2-GPU smoke run:

```text
max_tool_response_length=256
tool response URL: https://pixe...(truncated)...
```

Risk:

- The model cannot copy the exact retrieved URL.
- URL validity can be scored as fake/invalid even when the retriever found a
  relevant page.

Resolution:

- Use fewer results and longer tool responses for citation training:

```text
SEARCHQA_TOPK=2
GRPO_MAX_TOOL_RESPONSE_LENGTH=768
```

Lesson:

- In citation RL, tool-response truncation is not just a context-length issue;
  it changes the label by destroying the URL string.

### GRPO-004: SGLang memory fraction can be too low as well as too high

Date: 2026-05-18

Observed:

```text
GRPO_GPU_MEMORY_UTILIZATION=0.12
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

Interpretation:

- Lowering SGLang memory fraction too far can make the static memory pool too
  small for the 8B TP=2 rollout server.
- The prior value `0.15` initialized successfully on 2 A800 GPUs.

Resolution:

```text
GRPO_GPU_MEMORY_UTILIZATION=0.15
```

Lesson:

- Treat SGLang memory fraction as a required static-pool sizing parameter, not
  only as a knob for reducing memory pressure.

### GRPO-005: retrieval-hit improves URL validity but not enough claim support

Date: 2026-05-18

Retrieval-hit smoke result over 32 sampled trajectories:

```text
format = 1.000
search = 0.938
url_validity = 0.594
citation_precision = 0.126
claim_support = 0.122
unsupported_citation_rate = 0.755
```

Best observed supported case:

```text
reward ~= 0.625
url_validity = 1.0
claim_support = 0.75
unsupported_citation_rate = 0.0
```

Interpretation:

- The reward and SGLang backend are working.
- The model can produce valid, supported citations when the snippet is direct.
- Most sampled claims remain broader than the retrieved evidence, so support
  stays low.

Next fix:

- Construct the next GRPO dataset at the claim level, not just the query level:
  choose SFT traces where final cited claims are directly supported by compact
  retrieved snippets.
  Keep answers short enough that citations appear before response truncation.

### GRPO-006: claim-level filtering must reject low-information supported fragments

Date: 2026-05-18

Observed:

```text
The first claim-level support filter produced rows whose cited URL was valid and
whose overlap support was 1.0, but a small number of selected claims were too
low-information for RL, for example heading-like fragments rather than useful
answer claims.
```

Risk:

- A lexical support filter can over-credit title/heading fragments.
- GRPO could learn to cite narrow but unhelpful fragments instead of concise,
  answer-bearing claims.

Resolution:

```text
scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py
```

now filters:

```text
1-2 citations
retrieved URL only
no truncated/fake citation URL
claim support score >= 1.0
claim length cap
low-information citation fragments
```

Generated artifacts:

```text
data/deepfactcite_sglang_grpo_claim_filtered/train.parquet
data/deepfactcite_sglang_grpo_claim_filtered/test.parquet
data/deepfactcite_sglang_grpo_claim_filtered/corpus.jsonl
data/deepfactcite_sglang_grpo_claim_filtered/summary.json
data/deepfactcite_sglang_grpo_claim_filtered/preview.jsonl
data/deepfactcite_sglang_grpo_claim_filtered/reject_samples.jsonl
```

Validation:

```text
kept rows: 32
train/test: 28/4
corpus docs: 42
train top-2 retriever URL hit: 28/28
test top-2 retriever URL hit: 4/4
reject reasons while collecting: weak_claim_support=54, too_few_supported_claims=5
```

Decision:

- Use this dataset for the first outcome-only vs citation-aware 2-GPU ablation.
- Keep `save_freq=0` until disk cleanup; `/root/autodl-tmp` has about 14G free.
- Do not scale to 4 GPUs until the 2-GPU ablation shows a real citation metric
  gain without answer/search collapse.

## Open Questions

1. Can we add a trainer path that resumes from `Base + existing LoRA` without full merge?
2. Can we evaluate continuation adapters without changing the serving path from the baseline?
3. Should we use HF generation for small deterministic sanity evals where exact LoRA/merge equivalence matters more than speed?
4. Should vLLM eval standardize on dynamic LoRA bf16 only, and keep merged fp32 for training initialization only?

## Immediate Next Step

Do not continue from `mix50`.

`mixclean200` has passed the local Search-R1 BM25 answer/search guardrail and is
the current SFT baseline candidate.

Immediate next step:

1. Use `mixclean200` as the likely initialization checkpoint.
2. Port or reuse the Qwen3/SGLang GRPO backend.
3. Run outcome-only GRPO and citation-aware GRPO with the same backbone/data.
4. Keep fake/unsupported citation hard penalties in the citation-aware reward.
5. Keep reporting original Search-R1 paper numbers only as external reference
   unless the official E5 setup is reproduced.
