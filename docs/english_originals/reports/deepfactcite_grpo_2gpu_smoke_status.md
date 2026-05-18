# DeepFactCite 2-GPU GRPO Smoke Status

Date: 2026-05-18

## Goal

Validate the shortest practical Qwen3-8B GRPO path on only 2 A800 GPUs before
spending longer GPU time:

- reuse the old Qwen3/SGLang backend from `agentic-rl-searchqa`;
- initialize from `mixclean200` SFT merged bf16 full model;
- use the current project DeepFactCite reward, not the old SearchQA reward;
- keep checkpoint writing disabled while disk has only about 14G free.

## Launched Run

Screen:

```text
dfc_grpo_smoke_2gpu
```

Log:

```text
logs/grpo_mixclean200_2gpu_smoke.log
```

Rollouts:

```text
logs/grpo/rollouts/mixclean200_2gpu_smoke/
```

Data:

```text
data/deepfactcite_sglang_grpo_smoke/train.parquet
32 rows = 24 long_fact_qa + 8 short_qa guardrail
```

Actor:

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16
Qwen3ForCausalLM, 8.19B params, bf16
```

Important launch settings:

```text
TP=2
train_batch_size=1
rollout.n=2
max_response_length=384
max_assistant_turns=4
max_tool_response_length=256
save_freq=0
reward_model.reward_manager=deepfactcite_custom
```

Reward weights:

```text
answer=0.15
citation=0.35
support=0.30
format=0.10
search=0.05
cost=0.05
```

## Verified

- Hydra config selected `reward_manager: deepfactcite_custom`.
- Reward hook loaded from
  `scripts/deepfactcite/verl_deepfactcite_reward.py`.
- SGLang started with Qwen3-8B and 2-GPU tensor parallelism.
- Search tool initialized as offline with current project corpus.
- First rollout file was written and includes DeepFactCite metrics:
  `url_validity`, `citation_precision`, `claim_support`,
  `unsupported_citation_rate`.
- Disk remained stable because checkpoint saving is disabled.

## First-Rollout Quality Finding

Step 1 completed in about 67 seconds. It produced two sampled trajectories for
the same long Bach BWV 171 query. Retrieval quality was poor:

- one sample got no relevant result;
- one sample got irrelevant snippets;
- both rewards were low, about `0.08` to `0.10`;
- `claim_support=0`, no useful markdown citations.

This is a useful smoke result: reward plumbing works and bad evidence is
penalized. It is not a good longer training set. The next run should use a
retrieval-hit filtered subset.

## Next Decision

If this smoke run finishes cleanly, do not treat it as the final GRPO result.
Use it as backend validation, then run a smaller but higher-quality
retrieval-hit GRPO set. With only 2 GPUs, keep runs short and checkpoint only
after freeing enough disk.
