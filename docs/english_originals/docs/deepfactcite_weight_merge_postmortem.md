# DeepFactCite LoRA Merge and Eval Reproducibility Postmortem

Date: 2026-05-18

This document reconstructs the issue around `Soft SFT 100`, merged checkpoints, `mix50`, and inconsistent eval numbers. The purpose is to make the process auditable at an engineering-review level: what weights existed, how they were loaded, how they were merged, which serving path produced which number, what failed, and what we should do next.

## Executive Summary

The best reproducible SFT baseline is:

```text
Base Qwen3-8B + Soft SFT LoRA adapter, served by vLLM as dynamic LoRA in bf16.
```

It is reproducible under the current code path:

| Eval | Current rerun | Earlier aggregate |
|---|---:|---:|
| ShortQA32 answer_subEM | 0.469 | 0.469 |
| ShortQA32 URL validity | 0.812 | 0.750 |
| ShortQA32 claim support | 0.240 | 0.232 |
| Strict47 total | 0.156 | 0.158 |
| Strict47 claim support | 0.103 | 0.088 |

The problematic assumption was:

```text
Base + Soft LoRA dynamic serving == merged Soft full model
```

That assumption is not safe for this project. The old merged model was produced by loading the base model in bf16 and then merging a fp32 LoRA delta into bf16 weights. Direct HF logits checks showed this bf16 merge was not numerically equivalent to PEFT dynamic LoRA forward.

The downstream `mix50` continuation was trained from that lossy bf16 merged parent, so it is not a valid continuation result and should not be used as a winner.

## Artifacts and Weight Locations

### Base Model

```text
/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base
```

This is the base Qwen3-8B checkpoint used by the soft LoRA adapter.

### Soft SFT 100 Adapter

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100
```

Important files:

```text
adapter_config.json
adapter_model.safetensors
tokenizer_config.json
chat_template.jinja
tokenizer.json
```

Adapter config:

```text
base_model_name_or_path = /root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base
peft_type = LORA
r = 32
lora_alpha = 64
lora_dropout = 0.0
target_modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
task_type = CAUSAL_LM
```

The adapter tensors are fp32:

```text
adapter_model.safetensors: {'torch.float32': 504}
```

### Strict SFT 100 Adapter

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-strict-100/global_step_100
```

This was trained separately and is useful as an ablation. It is not currently the winner.

### Old bf16 Soft Merged Parent

Original output path:

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged
```

This directory was deleted after diagnosis to recover disk space. Logs and eval reports remain:

```text
logs/merge_soft100_lora.screen.log
logs/merge_soft100_lora.exit
reports/soft_merged_base_qwen3_8b_vllm_shortqa_guardrail32.json
reports/soft_merged_base_qwen3_8b_vllm_shortqa_guardrail32.jsonl
reports/soft_merged_base_qwen3_8b_vllm_deepfactcite_strict_sft_test47.json
reports/soft_merged_base_qwen3_8b_vllm_deepfactcite_strict_sft_test47.jsonl
```

This old merge loaded the base with `torch_dtype=torch.bfloat16`, then called `PeftModel.from_pretrained(...).merge_and_unload()`.

Result:

- internally self-consistent after saving,
- not equivalent to dynamic PEFT LoRA forward,
- not safe as a continuation parent.

### fp32 Soft Merged Parent

Current output path:

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged-fp32
```

Size:

```text
31G
```

Important files:

```text
config.json
generation_config.json
model-00001-of-00009.safetensors
...
model-00009-of-00009.safetensors
model.safetensors.index.json
tokenizer_config.json
chat_template.jinja
tokenizer.json
```

Merge command shape:

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python \
  scripts/deepfactcite/merge_lora_adapter.py \
  --base-model /root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base \
  --adapter outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100 \
  --output-dir outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged-fp32 \
  --torch-dtype float32
```

Logs:

```text
logs/merge_soft100_fp32_lora.screen.log
logs/merge_soft100_fp32_lora.exit
```

Exit code:

```text
0
```

### mix50 Continuation

Original output path:

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged-mix-50/global_step_50
```

This directory was deleted after diagnosis because it was trained from the lossy bf16 merged parent and should not be continued. Logs and reports remain:

```text
logs/deepfactcite-sft-qwen3-8b-soft-merged-mix-50.screen.log
logs/deepfactcite-sft-qwen3-8b-soft-merged-mix-50.exit
reports/mix50_qwen3_8b_vllm_shortqa_guardrail32.json
reports/mix50_qwen3_8b_vllm_shortqa_guardrail32.jsonl
reports/mix50_qwen3_8b_vllm_deepfactcite_strict_sft_test47.json
reports/mix50_qwen3_8b_vllm_deepfactcite_strict_sft_test47.jsonl
```

Training did complete normally:

```text
train steps = 50
val/loss = 1.016
exit code = 0
```

But because the parent was wrong, the result is not a valid continuation baseline.

## Timeline

### Step 1: Soft SFT 100 existed as a LoRA adapter

The soft SFT checkpoint was an adapter, not a full model:

```text
Base Qwen3-8B + LoRA delta
```

The correct serving path was:

```bash
vllm serve /root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base \
  --served-model-name base \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --trust-remote-code \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules soft=outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100
```

vLLM defaulted to bf16:

```text
dtype=torch.bfloat16
checkpoint size: 15.26 GiB
```

This path produced the current reproducible soft baseline:

```text
reports/soft_current_qwen3_8b_vllm_shortqa_guardrail32.json
reports/soft_current_qwen3_8b_vllm_shortqa_guardrail32.jsonl
reports/soft_current_qwen3_8b_vllm_deepfactcite_strict_sft_test47.json
reports/soft_current_qwen3_8b_vllm_deepfactcite_strict_sft_test47.jsonl
```

### Step 2: We needed a full parent to continue SFT

The original trainer takes `model.partial_pretrain` as a full HF model path and creates a new LoRA adapter. It did not directly support:

```text
Base full model + existing LoRA adapter as parent
```

So we tried to merge the soft adapter into a full model and then train a new LoRA on top.

### Step 3: The first merge was done in bf16

The old merge script loaded the base as bf16:

```python
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16,
    device_map=args.device_map,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, args.adapter)
model = model.merge_and_unload()
model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
```

This produced:

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged
```

That model looked valid on disk and could be served by vLLM, but its behavior was not equivalent to dynamic LoRA serving.

### Step 4: mix50 was trained from the lossy bf16 merged parent

Command shape:

```bash
BASE_MODEL=outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged \
DATA_DIR=data/deepfactcite_mix/sft \
EXPERIMENT_NAME=deepfactcite-sft-qwen3-8b-soft-merged-mix-50 \
MODEL_SIZE=8B N_GPUS=2 TOTAL_STEPS=50 \
bash scripts/deepfactcite/train_sft_qwen3.sh
```

Training completed:

```text
step:50 - val/loss:1.016
exit code: 0
```

Eval showed it was not a winner:

| Model | ShortQA answer | ShortQA URL | ShortQA support | Strict47 URL | Strict47 support |
|---|---:|---:|---:|---:|---:|
| Soft dynamic LoRA bf16 | 0.469 | 0.812 | 0.240 | 0.346 | 0.103 |
| mix50 | 0.375 | 0.625 | 0.190 | 0.312 | 0.087 |

At this point, the right engineering response was not to train more, but to investigate the parent equivalence.

## Root Cause Investigation

### Hypothesis 1: tokenizer or chat template mismatch

Checked:

- soft adapter tokenizer,
- bf16 merged tokenizer,
- mix50 tokenizer,
- base tokenizer.

The chat templates were effectively consistent. This was not the main root cause.

### Hypothesis 2: config mismatch

The merged config differed from base config in fields such as:

```text
transformers_version
dtype vs torch_dtype
layer_types
generation_config do_sample field
```

Core architecture fields matched:

```text
model_type = qwen3
hidden_size = 4096
num_hidden_layers = 36
num_attention_heads = 32
num_key_value_heads = 8
vocab_size = 151936
rope_theta = 1000000
```

Config differences were not sufficient to explain the major behavior shift.

### Hypothesis 3: bf16 merge lost LoRA delta fidelity

This was confirmed.

Direct HF logits test:

```text
Prompt 0 length: 94 tokens
Prompt 1 length: 98 tokens
```

Comparison results:

| Comparison | Prompt | Max Abs Diff | Mean Abs Diff | Top1 Equal | Top10 Overlap |
|---|---:|---:|---:|---:|---:|
| PEFT forward vs bf16 in-memory merge | 0 | 3.328125 | 1.3463 | yes | 8/10 |
| PEFT forward vs bf16 in-memory merge | 1 | 8.546875 | 4.3523 | yes | 9/10 |
| bf16 in-memory merge vs saved bf16 merge | 0 | 0.0 | 0.0 | yes | 10/10 |
| bf16 in-memory merge vs saved bf16 merge | 1 | 0.0 | 0.0 | yes | 10/10 |
| PEFT forward vs fp32 in-memory merge | 0 | 0.000081 | 0.000013 | yes | 10/10 |
| PEFT forward vs fp32 in-memory merge | 1 | 0.000177 | 0.000077 | yes | 10/10 |

Interpretation:

- Saving the merged model was not corrupting it.
- The bad behavior came from doing the merge into bf16 weights.
- fp32 merge is numerically equivalent to dynamic PEFT forward at the HF logits level.

### Why bf16 merge is different from bf16 training/inference

This point is easy to misunderstand.

The SFT trainer loads the model in fp32:

```python
AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32)
```

Then it wraps the model in FSDP mixed precision:

```python
MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)
```

So training can compute in bf16 while retaining a higher-precision master/optimizer pathway.

The problematic operation was different:

```text
base_weight_bf16 += lora_delta_fp32
```

If the base weight tensor is already bf16 at merge time, small LoRA deltas can be rounded away or distorted when absorbed into the base weight. Dynamic LoRA serving avoids this because the LoRA delta remains a separate adapter path.

### Hypothesis 4: fp32 merged full model should match dynamic LoRA vLLM eval

This is not testable with the same vLLM dynamic LoRA path because vLLM's LoRA kernels do not support float32 LoRA weights.

Attempted command shape:

```bash
vllm serve /root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base \
  --dtype float32 \
  --enable-lora \
  --lora-modules soft=outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100
```

It failed during LoRA graph profiling:

```text
assert weight.dtype in [torch.float16, torch.bfloat16]
AssertionError
```

So we cannot directly compare:

```text
vLLM float32 dynamic LoRA
vs
vLLM float32 merged full model
```

The only exact equivalence check available here is the HF logits comparison.

## Why Small Differences Can Move Eval Metrics

This is not ordinary one-shot QA. It is a search-agent loop:

1. Model emits `<search>query</search>`.
2. Local lexical retriever returns top-k evidence.
3. Model continues with retrieved snippets.
4. It may search again or answer with citations.
5. Reward checks answer, citation URLs, URL authenticity, and claim support.

If the first generated query changes slightly, the retrieved snippets can change. Once the snippets change, the whole later trajectory can diverge. Therefore a small model-side change can lead to large metric movement.

Examples observed in ShortQA32:

- Dynamic soft LoRA and fp32 merged often started similarly, but search queries differed.
- Some changed queries returned weaker snippets.
- Citation validity/support then changed even when the final answer was similar.

This is why deterministic `temperature=0` is necessary but not sufficient for cross-serving-path reproducibility.

## Eval Results by Serving Path

### ShortQA32

| Serving Path | Answer | Total | Citation Presence | URL Validity | Citation Precision | Claim Support | Unsupported | Search Turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B bf16 | 0.219 | 0.189 | 0.031 | 0.031 | 0.041 | 0.031 | 0.875 | 2.500 |
| Soft dynamic LoRA bf16, current rerun | 0.469 | 0.375 | 0.844 | 0.812 | 0.240 | 0.240 | 0.625 | 1.219 |
| Strict dynamic LoRA bf16 | 0.469 | 0.370 | 0.781 | 0.781 | 0.237 | 0.234 | 0.635 | 1.000 |
| Soft bf16 merged full model | 0.438 | 0.444 | 0.844 | 0.844 | 0.375 | 0.375 | 0.479 | 1.125 |
| Soft fp32 merged full model served as vLLM fp32 | 0.406 | 0.361 | 0.750 | 0.688 | 0.240 | 0.237 | 0.615 | 1.000 |
| mix50 from bf16 merged parent | 0.375 | 0.324 | 0.625 | 0.625 | 0.190 | 0.190 | 0.750 | 1.156 |

### Strict47

Strict47 has no gold answer, so `answer_subem` is not meaningful.

| Serving Path | Total | Citation Presence | URL Validity | Citation Precision | Claim Support | Unsupported | Search Turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B bf16 | 0.124 | 0.064 | 0.064 | 0.082 | 0.028 | 0.433 | 1.298 |
| Soft dynamic LoRA bf16, current rerun | 0.156 | 0.426 | 0.346 | 0.113 | 0.103 | 0.768 | 0.894 |
| Strict dynamic LoRA bf16 | 0.137 | 0.383 | 0.282 | 0.071 | 0.063 | 0.797 | 0.894 |
| Soft bf16 merged full model | 0.138 | 0.277 | 0.234 | 0.070 | 0.055 | 0.762 | 0.979 |
| mix50 from bf16 merged parent | 0.150 | 0.340 | 0.312 | 0.095 | 0.087 | 0.778 | 0.894 |

## What Is Reproducible

Reproducible:

```text
Base + soft LoRA, vLLM bf16 dynamic LoRA
```

with:

```text
reports/soft_current_qwen3_8b_vllm_shortqa_guardrail32.jsonl
reports/soft_current_qwen3_8b_vllm_deepfactcite_strict_sft_test47.jsonl
logs/vllm_origbase_soft.screen.log
```

Not reproducible as equivalent:

```text
Soft LoRA dynamic serving == merged full model serving
```

The merged full model is a different serving path and must be labeled separately.

## Engineering Fixes Applied

### Merge Script

File:

```text
scripts/deepfactcite/merge_lora_adapter.py
```

Changes:

- Added `--torch-dtype {float32,bfloat16}`.
- Default changed to `float32`.
- Tokenizer is loaded from adapter path if adapter contains tokenizer files.
- Help text documents that bf16 merge is not numerically equivalent to PEFT forward.

Validation:

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python -m py_compile \
  scripts/deepfactcite/merge_lora_adapter.py
```

### Reproducibility Log

File:

```text
docs/deepfactcite_reproducibility_issue_log.md
```

It records:

- confirmed issues,
- variables to record,
- non-mixable result types,
- current safe operating rules.

## Interview-Quality Explanation

If asked by an interviewer why the merged checkpoint behaved differently:

> The original SFT checkpoint was a LoRA adapter on top of Qwen3-8B. For faster continuation, I initially merged the adapter into a full model, but the merge script loaded the base in bf16. The adapter tensors were fp32, and merging fp32 LoRA deltas into bf16 base weights caused a non-equivalent model. I verified this by comparing HF logits: PEFT dynamic forward vs bf16 merged had max logit differences up to 8.5, while PEFT dynamic forward vs fp32 merged differed by less than 2e-4. The saved model itself was not corrupted; the issue was the dtype at merge time. I patched the merge script to default to fp32 and updated eval protocol so dynamic-LoRA, bf16-merged, and fp32-merged serving paths are never mixed in one comparison.

If asked why eval moved a lot:

> This is a search-agent loop, not single-turn classification. A small difference in early generated tokens can change the search query, which changes retrieved evidence, which changes citations and answer support. So even greedy decoding can diverge across serving paths. We now save JSONL rollouts, vLLM logs, model paths, dtype, tokenizer, corpus, and exact serving path for every result.

If asked what claims are safe:

> It is safe to claim that we built a Search-R1-style DeepFactCite pipeline and that the Soft SFT adapter improves citation behavior under a fixed dynamic-LoRA bf16 serving path. It is not safe to claim that merged checkpoints are interchangeable with dynamic LoRA serving, nor that mix50 improved the model, because mix50 was trained from a lossy bf16 merged parent.

## Resume Readiness Gate

Ready to say:

- Implemented DeepFactCite SFT/eval pipeline on top of Search-R1-style search-agent trajectories.
- Added URL-authenticity and claim-support metrics/reward components.
- Built deterministic vLLM evaluation with JSONL rollouts and fixed guardrail/citation eval sets.
- Debugged and fixed a LoRA merge reproducibility issue by proving bf16 merge was not equivalent and switching merge to fp32.

Not ready to say yet:

- Citation-aware GRPO beats outcome-only GRPO.
- The merged full model is equivalent to the LoRA adapter.
- mix50 improved the model.
- The project beats Search-R1.

For a high-standard resume result, we still need:

1. One fixed serving path for baseline and trained model.
2. A valid continuation training path that does not rely on lossy bf16 merge.
3. Base/Soft/Next-model tables using the same eval backend.
4. Eventually outcome-only GRPO vs citation-aware GRPO under the same backend.

## Next Step

The next engineering step should be one of these:

### Preferred

Implement continuation from:

```text
Base model + existing LoRA adapter
```

without first merging the adapter into a full model. This preserves the dynamic-LoRA serving path and avoids comparing incompatible checkpoint forms.

### Acceptable

Use the fp32 merged parent only as a training initialization path, but label all evals as merged-full-model evals and do not compare them as identical to dynamic LoRA baseline.

### Do Not Do

Do not continue training from:

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged
```

or from `mix50`. Those were tied to the lossy bf16 merged parent.

## Current Operational State

At the end of this postmortem:

- No screen jobs are running.
- GPUs are idle.
- `outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged-fp32` exists and is the only current merged full parent.
- Old bf16 merged parent and mix50 adapter directories were deleted to recover disk space.
- Logs and JSON/JSONL reports were retained for audit.
