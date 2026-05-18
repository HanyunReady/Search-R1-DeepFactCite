# DeepFactCite 2-GPU Retrieval-Hit GRPO Smoke Summary

Date: 2026-05-18

## Run

Purpose: validate a cost-efficient citation-aware GRPO path on only 2 A800 GPUs.

Initialization:

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16
```

Data:

```text
data/deepfactcite_sglang_grpo_retrieval_hit/train.parquet
16 rows, built from SFT traces with retrieved snippets
```

Corpus:

```text
data/deepfactcite_sglang_grpo_retrieval_hit/corpus.jsonl
80 snippet docs, query terms added as retrieval keywords
```

Log and rollouts:

```text
logs/grpo_mixclean200_2gpu_hit_smoke_retry.log
logs/grpo/rollouts/mixclean200_2gpu_hit_smoke/
```

Settings:

```text
reward_manager=deepfactcite_custom
rollout backend=SGLang
TP=2
rollout.n=2
train_batch_size=1
total_steps=16
topk=2
max_tool_response_length=768
max_response_length=384
save_freq=0
```

## Result

The run completed all 16 steps in about 10 minutes 44 seconds after startup.
No checkpoint was saved because disk space is tight and `save_freq=0`.

Aggregated over 32 sampled trajectories:

| Metric | Value |
|---|---:|
| Reward | 0.218 |
| Format | 1.000 |
| Search | 0.938 |
| URL Validity | 0.594 |
| Citation Precision | 0.126 |
| Claim Support | 0.122 |
| Unsupported Citation Rate | 0.755 |
| Avg Citations | 1.47 |

Best observed step:

```text
step 15
reward ~= 0.625
url_validity = 1.0
citation_precision = 0.75
claim_support = 0.75
unsupported_citation_rate = 0.0
```

This proves the reward path can distinguish:

- valid URL + supported local claim;
- valid URL but unsupported claim;
- malformed/truncated/fake URL;
- missing search or missing citation.

## Interpretation

This is not yet a final GRPO result. It is a successful backend and reward
validation run.

The main bottleneck is now evidence/support quality, not plumbing:

- URL validity improved versus the first blind smoke because retrieval-hit data
  and longer tool responses let the model see complete URLs.
- Claim support is still low because many snippets are too short or truncated,
  and the model often writes broader claims than the evidence supports.
- `max_response_length=384` still clips some responses; concise prompting helps
  but does not fully solve it.

## Next Action

Before spending longer GPU time:

1. Build a stronger retrieval-hit set from SFT traces where the snippet text
   directly supports the final cited claims.
2. Prefer claims that can be answered in 1 to 3 sentences with 1 to 2 citations.
3. Keep `topk=2`, `max_tool_response_length>=768`, and `gpu_memory_utilization=0.15`.
4. Run a saved checkpoint only after freeing disk; current run intentionally
   did not save model weights.
5. Then run the outcome-only vs citation-aware ablation on the same filtered set.
