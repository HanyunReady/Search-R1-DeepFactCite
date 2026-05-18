# DeepFactCite Experiment Record

Date: 2026-05-18

## Project Goal

Build a Search-R1-style search agent that preserves answer/search behavior
while improving citation authenticity and claim-level support.

The project is evaluated on two axes:

1. answer/search guardrail: the model must not collapse on ShortQA or
   NQ/HotpotQA-style Search-R1 BM25 evaluation;
2. citation quality: cited URLs must come from retrieved evidence, and the
   cited local claim must be supported by the corresponding snippet.

The core experimental question for GRPO is:

```text
Does citation-aware reward improve URL validity, citation precision, claim
support, and unsupported citation rate compared with outcome-only reward on the
same data, model, retriever, and rollout stack?
```

## Current Constraints

Hardware:

```text
2 x A800 80G currently allocated in this workspace
```

Storage:

```text
/root/autodl-tmp has about 14G free
```

Operational constraints:

```text
Do not save full checkpoints until disk cleanup is done.
Use save_freq=0 for smoke/ablation runs.
Do not start long GRPO until short ablation produces useful signal.
```

## External Positioning

This is not framed as "beating Search-R1" unless the original model/retriever
setup is reproduced. The defensible positioning is:

```text
DeepFactCite extends a Search-R1-style RL search agent with citation
authenticity and claim-support rewards, then tests those rewards against an
outcome-only baseline under the same backbone, data, and retriever.
```

Reference anchors:

```text
Search-R1: https://arxiv.org/abs/2503.09516
ALCE citation evaluation: https://arxiv.org/abs/2305.14627
Correctness vs attribution faithfulness: https://arxiv.org/abs/2412.18004
OpenAI Deep Research system card: https://openai.com/index/deep-research-system-card/
```

## How to Read This Record

This project has three layers. A beginner-friendly way to read every experiment
is:

```text
1. Can the model answer and search correctly?
2. If it cites sources, are the URLs real retrieved URLs?
3. Does each cited local claim actually follow from the cited snippet?
```

Important terms:

| Term | Meaning in this project | Why it matters |
|---|---|---|
| SFT | Supervised fine-tuning on example search/citation traces | Teaches the model the behavior pattern before RL |
| GRPO | RL training where several sampled answers are compared by reward | Lets reward push behavior that SFT alone does not reliably learn |
| Rollout | One complete agent trajectory: prompt, search calls, snippets, final answer | The unit we inspect when debugging |
| Retriever | The offline search component that returns snippets/URLs | Citation can only be trusted if it cites retrieved evidence |
| URL validity | Whether cited URLs come from retrieved evidence, not hallucinated links | Prevents fake or unrelated citations |
| Citation precision | Fraction of citations that are valid and locally useful | Penalizes excessive or low-quality citation |
| Claim support | Whether the cited snippet clearly supports the nearby claim | Measures attribution faithfulness, not just link existence |
| Unsupported rate | Fraction of cited claims not supported by their cited snippets | The main failure mode this project tries to reduce |

The most important engineering principle is:

```text
Do not scale GPU because a run completed. Scale only when the run isolates a
question, records failure modes, and improves the metric it was designed to
improve without breaking the guardrail metrics.
```

## Stage 1: Data and SFT Baseline

Prepared DeepFactCite-style SFT/RL data from DeepCiteFact traces:

```text
data/deepfactcite/sft/train.parquet
data/deepfactcite/sft/test.parquet
data/deepfactcite/rl/train.parquet
data/deepfactcite/rl/test.parquet
```

Strict SFT filtering kept only traces with retrieved URLs and non-trivial
claim-support:

```text
data/deepfactcite_strict/sft/train.parquet
data/deepfactcite_strict/sft/test.parquet
```

Key lesson:

```text
Strict filtering alone did not automatically beat Soft SFT. Cleaner citation
data can reduce diversity and must be evaluated against answer/search
guardrails before promotion.
```

Beginner explanation:

```text
"Cleaner data" is not automatically better training data. If filtering removes
many varied examples, the model may see fewer ways to answer/search. That can
hurt general behavior even if the remaining examples have nicer citations.
The right test is therefore not "does the dataset look clean", but "does a
model trained on it beat the old model on the same evaluation path".
```

The current SFT baseline is MixClean200:

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200/global_step_200
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16
```

Training script:

```text
scripts/deepfactcite/launch_mix_clean_sft_200.sh
```

Primary report:

```text
reports/deepfactcite_mixclean200_eval_summary.md
```

## Stage 2: Baseline Evaluation

MixClean200 was evaluated against Base and Soft100 under the same vLLM
dynamic-LoRA serving path.

ShortQA32:

| Model | Answer subEM | URL Validity | Claim Support | Unsupported |
|---|---:|---:|---:|---:|
| Base | 0.219 | 0.031 | 0.031 | 0.906 |
| Soft100 | 0.438 | 0.812 | 0.292 | 0.557 |
| MixClean200 | 0.500 | 0.969 | 0.333 | 0.495 |

Search-R1 BM25 200:

| Model | subEM | Search Success | Search Turns | Budget Fail |
|---|---:|---:|---:|---:|
| Base | 0.290 | 0.960 | 1.720 | 0.170 |
| Soft100 | 0.450 | 0.990 | 1.195 | 0.045 |
| MixClean200 | 0.490 | 1.000 | 1.160 | 0.025 |

Decision:

```text
Use MixClean200 as the SFT initialization for GRPO.
Do not claim final credible-citation success from SFT alone.
```

Why this matters:

```text
SFT successfully taught the model to search and cite more often, but it did not
fully teach the stricter behavior "only cite claims supported by retrieved
snippets". This is exactly the gap GRPO reward should target.
```

## Stage 3: GRPO Backend and Reward Plumbing

Problem found:

```text
The old SGLang backend's reward_manager=custom did not necessarily call the
new DeepFactCite reward function even when custom_reward_function.path was set.
```

Fix:

```text
Register reward_manager=deepfactcite_custom via:
scripts/deepfactcite/verl_deepfactcite_reward.py
```

Verified in logs:

```text
reward_model.reward_manager = deepfactcite_custom
custom_reward_function.path = scripts/deepfactcite/verl_deepfactcite_reward.py
rollout JSONL includes url_validity, citation_precision, claim_support,
unsupported_citation_rate
```

## Stage 4: 2-GPU Retrieval-Hit GRPO Smoke

Purpose:

```text
Validate 2-GPU Qwen3-8B SGLang GRPO, DeepFactCite reward, offline retrieval,
and rollout logging before spending longer GPU time.
```

Run:

```text
data/deepfactcite_sglang_grpo_retrieval_hit/train.parquet
logs/grpo_mixclean200_2gpu_hit_smoke_retry.log
logs/grpo/rollouts/mixclean200_2gpu_hit_smoke/
```

Settings:

```text
TP=2
rollout.n=2
train_batch_size=1
total_steps=16
topk=2
max_tool_response_length=768
max_response_length=384
save_freq=0
```

Result over 32 sampled trajectories:

| Metric | Value |
|---|---:|
| reward | 0.218 |
| search | 0.938 |
| URL validity | 0.594 |
| citation precision | 0.126 |
| claim support | 0.122 |
| unsupported citation rate | 0.755 |

Interpretation:

```text
The engineering chain works. The bottleneck is not SGLang plumbing; it is
evidence/claim quality. Retrieval-hit at query level is insufficient because
models still write broader claims than the snippets support.
```

Failure analysis:

```text
This was a successful systems smoke test but a failed citation-quality result.
The model usually searched, and many cited URLs were real retrieved URLs, but
the answer sentences often made claims broader than the snippet. Example pattern:
a snippet supports "X happened in 2012", while the answer says "X changed the
whole organization from 2010 to 2013". The URL is not fake, but the cited claim
is still not supported.
```

Lesson:

```text
Query-level retrieval hit is too weak for citation RL. The data must be filtered
at claim level: short answer, few citations, retrieved URLs only, and cited
claim explicitly supported by the snippet.
```

Decision:

```text
Build claim-level support-filtered GRPO data before longer runs.
```

## Failure Ledger in Plain Language

This section explains the earlier failed or non-winning experiments in a form
that can be checked by someone new to the project.

### Failure 1: Strict SFT was cleaner but not better

What we tried:

```text
Train on stricter SFT examples with better URL/support properties.
```

What happened:

```text
Strict SFT did not beat Soft SFT on the same eval path. It was not promoted.
```

Why it likely failed:

```text
The filter improved data cleanliness but reduced diversity. A search agent needs
to learn not only citation formatting, but also when to search, how to phrase
queries, and how to answer varied questions. Over-filtering can make that
behavior narrower.
```

Technical value:

```text
It established that dataset quality must be measured by downstream guardrail
metrics, not by filter strictness alone.
```

### Failure 2: bf16 LoRA merge changed the model

What we tried:

```text
Merge a LoRA adapter into the base model and use the merged full model as if it
were equivalent to dynamic LoRA serving.
```

What happened:

```text
HF logits showed large differences between PEFT dynamic LoRA and the saved bf16
merged model, even though the saved model was internally self-consistent.
```

Why it failed:

```text
LoRA deltas were effectively added into bf16 base weights during merge. That
can lose numerical detail. In a closed-loop search agent, small token changes
can cause different search queries, which then change retrieved evidence and
final metrics.
```

Technical value:

```text
Serving path is part of model identity. Dynamic LoRA, bf16 merged, and fp32
merged results must not be mixed in one baseline table unless explicitly labeled.
```

### Failure 3: mix50 continuation used an invalid parent

What we tried:

```text
Continue SFT for 50 steps from the old merged Soft parent.
```

What happened:

```text
The run completed, but the parent was the lossy bf16 merged model from the
previous failure. The result was worse and not a clean test of mix-data
continuation.
```

Why it failed:

```text
The training data was not the only variable. The parent checkpoint had already
changed behavior, so the experiment could not answer whether the continuation
recipe itself was good.
```

Technical value:

```text
Every training run needs explicit lineage: base model, adapter, merge dtype,
serving path, and eval path.
```

### Failure 4: Strict47 answer metric was not meaningful

What we tried:

```text
Use DeepFactCite strict47 as a general answer-and-citation evaluation set.
```

What happened:

```text
The set has no gold answer field suitable for answer_subem. Its citation
metrics are useful; its answer_subem is not.
```

Why it matters:

```text
An impressive-looking metric is useless if the dataset schema does not support
that metric. This is why the project separates answer/search guardrails
ShortQA/Search-R1 BM25 from citation-behavior evaluation strict47.
```

### Failure 5: Retrieval-hit GRPO did not solve support

What we tried:

```text
Give GRPO examples where retrieval could find the intended URL.
```

What happened:

```text
The 2-GPU stack ran, but claim support remained low and unsupported citation
rate remained high.
```

Why it failed:

```text
Finding the right URL is not the same as making a supported claim. The model can
cite a real URL while writing a sentence that the snippet does not prove.
```

Technical value:

```text
This failure justified the current claim-level support-filtered data build and
the outcome-only vs citation-aware ablation.
```

## Reusable Experiment Flow

The process that should be reused for future experiments is:

```text
1. Fix the question: what single hypothesis is this run testing?
2. Fix the model identity: base, adapter, merge dtype, serving path.
3. Fix the data identity: train/test parquet, corpus path, retriever top-k.
4. Run the smallest smoke that validates the full engineering path.
5. Parse rollouts, not just stdout reward.
6. Record both wins and failures.
7. Scale only if the failure analysis says the next larger run is justified.
```


## Stage 5: Claim-Level Support-Filtered GRPO Data

Script:

```text
scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py
```

Command:

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python \
  scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py \
  --rows 32 \
  --out-dir data/deepfactcite_sglang_grpo_claim_filtered
```

Artifacts:

```text
data/deepfactcite_sglang_grpo_claim_filtered/train.parquet
data/deepfactcite_sglang_grpo_claim_filtered/test.parquet
data/deepfactcite_sglang_grpo_claim_filtered/corpus.jsonl
data/deepfactcite_sglang_grpo_claim_filtered/summary.json
data/deepfactcite_sglang_grpo_claim_filtered/preview.jsonl
data/deepfactcite_sglang_grpo_claim_filtered/reject_samples.jsonl
```

Build result:

| Item | Value |
|---|---:|
| kept rows | 32 |
| train/test | 28/4 |
| corpus docs | 42 |
| avg selected citations | 1.3125 |
| min support score | 1.0 |
| avg support score | 1.0 |

Reject reasons while collecting:

| Reason | Count |
|---|---:|
| weak_claim_support | 54 |
| too_few_supported_claims | 5 |

Retriever validation:

```text
OfflineSearchTool, topk=2
train top-2 URL hit: 28/28
test top-2 URL hit: 4/4
```

Data filter rules:

```text
1-2 citations
retrieved URL only
no truncated/fake citation URL
claim support score >= 1.0
claim length cap
low-information citation fragments filtered
prompt asks for 1-3 concise sentences
```

## Stage 6: Outcome-Only vs Citation-Aware GRPO

Entrypoint:

```text
scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

Common settings:

```text
actor = outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16
data = data/deepfactcite_sglang_grpo_claim_filtered
rollout backend = SGLang
tensor parallel = 2
rollout.n = 2
train_batch_size = 1
total_training_steps = 16
topk = 2
max_tool_response_length = 768
max_response_length = 384
save_freq = 0
reward_manager = deepfactcite_custom
```

Outcome-only reward:

| Component | Weight |
|---|---:|
| answer | 0.80 |
| citation | 0.00 |
| support | 0.00 |
| format | 0.15 |
| search | 0.10 |
| cost | 0.05 |

Citation-aware reward:

| Component | Weight |
|---|---:|
| answer | 0.15 |
| citation | 0.35 |
| support | 0.30 |
| format | 0.10 |
| search | 0.05 |
| cost | 0.05 |

Commands:

```bash
MODE=outcome-only DRY_RUN=0 GRPO_TOTAL_STEPS=16 \
  bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh

MODE=citation-aware DRY_RUN=0 GRPO_TOTAL_STEPS=16 \
  bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

Completed runs:

```text
MODE=outcome-only
RUN_TAG=20260518_claimfiltered_2gpu
log=logs/dfc-mixclean200-claimfiltered-outcome-only-20260518_claimfiltered_2gpu.log
rollouts=logs/grpo/rollouts/dfc-mixclean200-claimfiltered-outcome-only-20260518_claimfiltered_2gpu/

MODE=citation-aware
RUN_TAG=20260518_claimfiltered_2gpu
log=logs/dfc-mixclean200-claimfiltered-citation-aware-20260518_claimfiltered_2gpu.log
rollouts=logs/grpo/rollouts/dfc-mixclean200-claimfiltered-citation-aware-20260518_claimfiltered_2gpu/
```

Startup verification:

```text
Hydra config passed validation.
reward_manager=deepfactcite_custom.
Reward function loaded from scripts/deepfactcite/verl_deepfactcite_reward.py.
Dataset len = 28 train, 4 val.
Total training steps = 16.
Offline retriever uses data/deepfactcite_sglang_grpo_claim_filtered/corpus.jsonl.
SGLang backend initialized on CUDA_VISIBLE_DEVICES=0,1.
Checkpoint saving disabled.
```

Result:

| Metric | Outcome-Only | Citation-Aware | Readout |
|---|---:|---:|---|
| samples | 32 | 32 | same budget |
| search | 1.0000 | 1.0000 | search did not collapse |
| format | 0.9812 | 0.9750 | both high |
| answer_subem | 0.0000 | 0.0312 | citation-aware slightly higher, but tiny sample |
| URL validity | 0.7656 | 0.6979 | citation-aware worse |
| citation precision | 0.4219 | 0.3828 | citation-aware worse |
| claim support | 0.4219 | 0.3828 | citation-aware worse |
| unsupported citation rate | 0.4219 | 0.4844 | citation-aware worse |
| fake URL rate | 0.0469 | 0.0521 | citation-aware worse |
| citation count | 1.0625 | 1.0000 | similar |
| mean step response clip ratio | 0.1563 | 0.2813 | citation-aware worse |

Read this table carefully:

```text
The citation-aware reward number itself is higher, but reward numbers are not
directly comparable across modes because the weights are different. The fair
comparison is URL validity, citation precision, claim support, unsupported rate,
search, format, and answer guardrails. On those metrics, v1 citation-aware did
not win.
```

Failure analysis:

```text
The v1 claim-filtered data guarantees that one selected claim is supported by a
snippet, but some original user questions are broader than the selected claim.
The model often answers the broad question anyway, adds extra background, then
places the citation after a sentence that the snippet does not fully prove.
This makes the URL real but the cited local claim unsupported.
```

Decision:

```text
Do not switch to 4 GPUs from this result.
Do not run a longer GRPO on the same v1 data.
Build and test a stricter one-claim / one-citation dataset first.
```

Follow-up report:

```text
reports/deepfactcite_claimfiltered_grpo_ablation_2gpu_20260518.md
```

## Stage 7: One-Claim / One-Citation Repair

Reason:

```text
The v1 failure is not "citation-aware reward is useless"; it is "the current
data and prompt still let the model over-answer broad questions." The repair is
to make the next dataset narrower so the supported behavior is easier to learn
and easier to measure.
```

Script updated:

```text
scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py
```

New filters/options:

```text
max_citations = 1
max_claim_tokens = 30
max_query_tokens = 18
max_answer_sentences = 1
strict_one_citation_prompt = true
```

Command:

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python \
  scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py \
  --rows 32 \
  --out-dir data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite \
  --max-citations 1 \
  --max-claim-tokens 30 \
  --max-query-tokens 18 \
  --max-answer-sentences 1 \
  --strict-one-citation-prompt
```

Build result:

| Item | Value |
|---|---:|
| kept rows | 32 |
| train/test | 28/4 |
| corpus docs | 32 |
| avg selected citations | 1.0000 |
| min support score | 1.0000 |
| avg support score | 1.0000 |

Reject reasons:

| Reason | Count |
|---|---:|
| weak_claim_support | 43 |
| query_too_broad | 19 |
| too_few_supported_claims | 3 |
| claim_too_broad | 1 |

Retriever validation:

```text
OfflineSearchTool, topk=2
train top-2 URL hit: 28/28
test top-2 URL hit: 4/4
```

Next small run:

```bash
RUN_TAG=20260518_v2_onecite_2gpu \
MODE=citation-aware \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite \
DRY_RUN=0 \
GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

Success standard for this repair run:

```text
It does not need to prove final superiority. It should show whether stricter
one-claim data reduces no-citation, fake URL, response clipping, and unsupported
claim behavior enough to justify a fair v2 outcome-only vs citation-aware
ablation.
```

Completed v2 runs:

```text
v2 citation-aware:
logs/grpo/rollouts/dfc-mixclean200-claimfiltered-citation-aware-20260518_v2_onecite_2gpu/
reports/dfc_mixclean200_claimfiltered_v2_onecite_citation_aware_2gpu_rollout_summary.md

v2 outcome-only:
logs/grpo/rollouts/dfc-mixclean200-claimfiltered-outcome-only-20260518_v2_onecite_2gpu/
reports/dfc_mixclean200_claimfiltered_v2_onecite_outcome_only_2gpu_rollout_summary.md

v2 citation-strong:
logs/grpo/rollouts/dfc-mixclean200-v2-onecite-citation-strong-20260518_2gpu/
reports/dfc_mixclean200_v2_onecite_citation_strong_2gpu_rollout_summary.md
```

Fair v2 ablation result:

| Metric | Outcome-Only | Citation-Aware | Readout |
|---|---:|---:|---|
| search | 0.9688 | 1.0000 | citation-aware preserved search |
| format | 0.9812 | 0.9625 | both usable |
| URL validity | 0.1562 | 0.7188 | citation-aware much better |
| citation precision | 0.1125 | 0.3937 | citation-aware much better |
| claim support | 0.1094 | 0.3906 | citation-aware much better |
| unsupported citation rate | 0.8438 | 0.3750 | citation-aware much better |
| fake URL rate | 0.0000 | 0.0000 | v2 fixed fake URLs in both |
| citation count | 0.1562 | 0.7188 | citation-aware cites far more |
| response clip ratio | 0.0000 | 0.0000 | v2 fixed truncation |
| no_citation failures | 27 | 9 | citation-aware much better |

This is the first clean positive GRPO signal in this stage:

```text
On the same v2 one-citation data, explicit citation/support reward greatly
improves citation behavior over outcome-only reward while preserving search and
removing response clipping.
```

Citation-strong follow-up:

| Metric | v2 Citation-Aware | v2 Citation-Strong |
|---|---:|---:|
| search | 1.0000 | 1.0000 |
| format | 0.9625 | 0.9812 |
| URL validity | 0.7188 | 0.7188 |
| citation precision | 0.3937 | 0.4219 |
| claim support | 0.3906 | 0.4219 |
| unsupported citation rate | 0.3750 | 0.4375 |
| no_citation failures | 9 | 9 |
| response clip ratio | 0.0000 | 0.0000 |

Interpretation:

```text
Stronger citation/support weights improved average support but did not reduce
no_citation and worsened unsupported rate. Scalar weight tuning alone is not the
clean next fix.
```

Updated decision:

```text
Do not move to 4 GPUs yet.
Do not run another longer blind GRPO.
The next engineering task is to patch reward/prompt handling for exact markdown
URL citation presence: penalize no citation, bare [S_xxx], bare [1], and raw
URLs that are not markdown links.
```

Detailed report:

```text
reports/deepfactcite_v2_onecite_grpo_ablation_2gpu_20260518.md
```

## Metrics to Report

Primary citation metrics:

```text
url_validity
citation_precision
claim_support
unsupported_citation_rate
fake_url_rate
citation_count
```

Guardrail metrics:

```text
answer_subem / target subEM
format
search
search turns
response_length clip ratio
no-search and no-citation failures
```

Summarizer:

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python \
  scripts/deepfactcite/summarize_grpo_rollouts.py \
  logs/grpo/rollouts/<run_name> \
  --out reports/<run_name>_rollout_summary.md
```

## Promotion Rules

Move to longer or 4-GPU runs only if:

```text
1. citation-aware improves URL validity / claim support / unsupported rate
   over outcome-only;
2. answer/search behavior does not materially collapse;
3. response truncation is not the dominant failure mode;
4. disk cleanup is done before enabling checkpoint saves.
```

Stop or revise if:

```text
1. citation-aware only increases citation count but unsupported rate stays high;
2. outcome-only wins answer/search while citation-aware collapses;
3. rollout is dominated by no-search/no-citation format behavior;
4. max_response_length=384 clips too many answers.
```

## Interview-Grade Narrative

The strongest project narrative is:

```text
I first established a reproducible SFT baseline and answer/search guardrail,
then validated the 2-GPU SGLang GRPO engineering path. The first retrieval-hit
smoke showed that backend plumbing worked but citation support stayed weak.
Instead of scaling GPU blindly, I built claim-level support-filtered RL data,
validated retriever top-k hits under the exact rollout retriever, and set up a
controlled outcome-only vs citation-aware reward ablation. The result is a
clean test of whether explicit citation authenticity and claim-support rewards
improve attribution without sacrificing the Search-R1-style answer/search
objective.
```

## Stage 8: Reward Parser and Markdown Citation Repair

Why this stage exists:

```text
The fair v2 ablation showed a useful result: citation-aware GRPO improved URL
validity, citation precision, claim support, unsupported rate, and no-citation
count compared with outcome-only. But the result was not yet good enough to
scale because citation-aware still had 9 no-citation failures out of 32 samples.
```

The naive next move would be to run longer or use more GPUs. We did not do that
because the failure analysis pointed to a narrower issue:

```text
The model often knows the answer and performs search, but it sometimes fails to
emit an exact markdown URL citation in the final answer.
```

That is a reward-specification problem, not just a capacity problem.

### What Counts as a Valid Citation Here

For this project, a citation is not merely a source-looking token. It must pass
three checks:

```text
1. Markdown form: [some label](https://retrieved-url)
2. URL provenance: the URL must appear in the retrieved evidence for that rollout
3. Claim support: the local claim near the citation must be supported by the
   text snippet behind that URL
```

Examples:

```text
Valid shape:
The cave contains preserved figurative paintings [Chauvet Cave](https://example.org/chauvet).

Invalid shape:
The cave contains preserved figurative paintings [S_abc123].

Invalid shape:
The cave contains preserved figurative paintings [1].

Invalid shape:
The cave contains preserved figurative paintings https://example.org/chauvet.
```

The invalid shapes are common in model rollouts because the model sees snippet
IDs and URLs in the tool output. If the reward accepts those loose forms, the
model can appear to "cite" without learning the target attribution behavior.

### Reward/Pipeline Patch

Patched files:

```text
deepfactcite/reward.py
scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py
scripts/deepfactcite/summarize_grpo_rollouts.py
```

Changes:

```text
1. citation-aware mode now sets DFC_REQUIRE_MARKDOWN_CITATION=true.
2. outcome-only mode sets DFC_REQUIRE_MARKDOWN_CITATION=false, preserving the
   ablation boundary.
3. If retrieved evidence exists but the final answer has no markdown URL
   citation, reward is capped at 0.08.
4. If the answer uses a bare bracket citation or raw URL outside markdown,
   reward is capped at 0.12.
5. The v2 prompt now asks for exactly one short sentence and exactly one
   markdown URL citation copied from retrieved evidence.
6. The rollout summarizer can recompute diagnostic details with the current
   reward code using --recompute-details while preserving the logged reward.
```

### Parser Bug Found

During the markdown-cap smoke, one failure sample looked like this:

```text
[NCDAS: Substance Abuse and Addiction Statistics [2025]](https://drugabusestatistics.org)
```

This is markdown. The old regex failed to count it because the label contained
an inner bracketed year, `[2025]`. That made the metrics say
`citation_count=0`, even though the model had produced a markdown link.

Why this matters:

```text
If the parser undercounts citations, the reward and reports can punish or
diagnose the wrong behavior. That contaminates the interpretation of GRPO runs.
```

Fix:

```text
deepfactcite/reward.py now scans markdown links with a small parser that allows
bracketed text inside the link label. It also strips legal markdown links before
checking for bare brackets or raw URLs, so `[2025]` inside a valid citation
label is not treated as a bad bare citation.
```

Sanity check passed:

```text
Nested-label markdown link:
citation_count=1, raw_url_count=0, bare_citation_count=0

Bare [S_1] plus raw URL:
citation_count=0, raw_url_count=1, bare_citation_count=1
```

### Incomplete Markdown-Cap Run

Command:

```bash
RUN_TAG=20260518_v2_markdowncap_2gpu \
MODE=citation-aware \
EXPERIMENT_NAME=dfc-mixclean200-v2-onecite-markdowncap-20260518_2gpu \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite \
DRY_RUN=0 \
GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

Observed:

```text
Only 6 of 16 steps were written.
No traceback, OOM, NCCL failure, or Python exception appeared in the log tail.
After the stop, GPU was idle.
```

Partial result before parser recomputation:

```text
samples = 12
steps = 6
reward = 0.2319
search = 0.7500
url_validity = 0.2500
claim_support = 0.2083
unsupported_citation_rate = 0.5833
citation_count = 0.4167
no_citation failures = 7
```

Diagnostic recomputation after parser fix:

```text
total = 0.2492
url_validity = 0.3333
citation_count = 0.5000
no_citation failures = 6
```

Interpretation:

```text
This partial run is useful as debugging evidence, not as a final experiment.
It exposed a parser issue and confirmed that the new no-citation cap is active,
but it cannot be compared fairly to completed 16-step runs.
```

### Current Parser-Fix Rerun

Command:

```bash
RUN_TAG=20260518_v2_markdowncap_parserfix2_2gpu \
MODE=citation-aware \
EXPERIMENT_NAME=dfc-mixclean200-v2-onecite-markdowncap-parserfix2-20260518_2gpu \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite \
DRY_RUN=0 \
GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

Launch status:

```text
Training entered async rollout.
SearchQAVerlTool initialized with the v2 one-citation offline corpus.
rollout_data_step_1.jsonl was created.
GPU memory was occupied by the 2-card GRPO/SGLang stack.
save_freq=0, so this run is still a smoke/ablation and will not consume large
checkpoint disk.
```

What to check when it completes:

```text
1. no_citation failures: target is below 9/32.
2. claim_support: target is above 0.3906.
3. unsupported_citation_rate: target is at or below 0.3750.
4. search: should stay near 1.0000.
5. response clip ratio: should stay 0.0000.
```

If those pass, the next step is a saved 2-GPU run after disk cleanup. If they do
not pass, do not scale to 4 GPUs; fix data/prompt/reward again first.

## Stage 9: Small-Scale vs Scale-Up Rule

Important distinction:

```text
Small runs are not used as final effect estimates. They are used as mechanism
tests.
```

At 32 rollout samples, one sample changes an aggregate rate by 3.125 points.
Therefore:

```text
no_citation 9 vs 10 is not statistically decisive.
unsupported 0.3750 vs 0.4062 is also only about one sample of movement.
```

But small runs can still show whether the experiment is pointed in the right
direction. The v2 result was large enough to count as a mechanism signal:

```text
outcome-only -> citation-aware
claim_support: 0.1094 -> 0.3906
unsupported_citation_rate: 0.8438 -> 0.3750
no_citation: 27 -> 9
```

That is why v2 is worth continuing. The parser-fix run had mixed evidence:

```text
claim_support improved to 0.4375
citation_precision improved to 0.4406
format/search stayed at 1.0000
but no_citation was 10 and unsupported was 0.4062
```

Interpretation:

```text
This does not prove the fix is bad, but it also does not justify spending 4/8
GPUs yet. It says the next cheap step should target the exact residual failure:
the model still ignores markdown citation formatting in some cases.
```

Scale-up policy:

```text
1. 2-GPU tiny run: verify engineering and mechanism.
2. 2-GPU or 4-GPU medium run: verify trend with saved checkpoint after the
   mechanism works.
3. 8-GPU run: reserve for final throughput or larger confirmed training, not
   for debugging reward/data definitions.
```

The promotion gate is based on the best completed v2 citation-aware baseline:

```text
no_citation < 9/32
claim_support > 0.3906
unsupported_citation_rate <= 0.3750
search near 1.0
response clip ratio 0.0
disk cleaned before checkpoint save
```

These are not universal scientific thresholds. They are local engineering
gates: a more expensive run should beat the current best cheap baseline on the
failure it is supposed to fix, while preserving Search-R1 behavior.

## Stage 10: v3 Prompt-Fix Data

Reason:

```text
The v2 parquet was generated before the prompt was patched. That means the
parser-fix run used the new reward cap but the old prompt did not explicitly
forbid bare [S_xxx], bare [1], and raw URLs.
```

Fix:

```text
Regenerate the same 32-row one-citation data with the current stricter prompt.
Keep rows, retriever corpus, support thresholds, model, reward, and 2-GPU setup
otherwise unchanged.
```

Command:

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python \
  scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py \
  --rows 32 \
  --out-dir data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite \
  --max-citations 1 \
  --max-claim-tokens 30 \
  --max-query-tokens 18 \
  --max-answer-sentences 1 \
  --strict-one-citation-prompt \
  --citation-format-template
```

Data summary:

```text
kept_rows = 32
train_rows = 28
test_rows = 4
corpus_docs = 32
avg_selected_citations = 1.0
min_support_score = 1.0
avg_support_score = 1.0
```

Retriever validation:

```text
train top-2 URL hit: 28/28
test top-2 URL hit: 4/4
```

Current run:

```bash
RUN_TAG=20260518_v3_promptfix_2gpu \
MODE=citation-aware \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite \
DRY_RUN=0 \
GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

Launch validation:

```text
trainer accepted train=28 / val=4
SearchQAVerlTool initialized with v3 prompt-fix corpus
reward_manager=deepfactcite_custom
save_freq=0
```

Result:

```text
reports/dfc_mixclean200_v3_promptfix_onecite_2gpu_rollout_summary.md
logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu/
```

| Metric | v2 Citation-Aware | v3 Prompt-Fix | Readout |
|---|---:|---:|---|
| search | 1.0000 | 0.9375 | v3 has 2 no-search failures |
| format | 0.9625 | 0.9875 | v3 better |
| URL validity | 0.7188 | 0.9375 | v3 much better |
| citation precision | 0.3937 | 0.6312 | v3 much better |
| claim support | 0.3906 | 0.6250 | v3 much better |
| unsupported citation rate | 0.3750 | 0.1250 | v3 much better |
| fake URL rate | 0.0000 | 0.0000 | clean |
| citation count | 0.7188 | 0.9375 | v3 cites more reliably |
| no_citation failures | 9 | 2 | v3 fixes main failure |
| response clip ratio | 0.0000 | 0.0000 | still clean |

Interpretation:

```text
This is the first result that clears the local scale-up gate. It shows that the
main bottleneck was not simply small data or insufficient GPU scale. The missing
piece was alignment between the training data prompt and the reward parser/caps.
```

Remaining caveat:

```text
The two no-search/no-answer failures mean search behavior must be watched in the
next run. The model did not collapse, but the next medium run should preserve
search above this level while saving a checkpoint.
```

Next decision:

```text
Do not use 8 GPUs for debugging. Use a saved medium run after disk cleanup:
4 GPUs if available, otherwise 2 GPUs. The purpose is to produce a checkpoint
for evaluation, not another no-checkpoint smoke.
```

Audit:

```text
Rows checked: 32
Steps checked: 16
Training reward recomputation: exact match, max diff = 0.0
Diagnostic metric recomputation: exact match against logged details
Prompt target leakage: 0/28 train, 0/4 test
Offline retriever target URL top-1 hit: 28/28 train, 4/4 test
bare_citation_count mean: 0.0
raw_url_count mean: 0.0
```

Failure audit:

```text
2 no_search/no_citation rows:
  Boredoms
  Chauvet-Pont d'Arc Cave

4 unsupported citation rows:
  climate-policy question, two samples
  Jim Umbricht question, two samples
```

Citation label caveat:

```text
The model mostly learned to attach real URLs, but many labels are still snippet
IDs. Out of 30 markdown citations, 22 labels were S_xxx-style and 8 were
human-readable. This is acceptable for the current URL-provenance/support
objective, but a final product-quality run should add a label-quality metric if
human-readable citation labels matter.
```

## Stage 7: CPU-Mode Consolidation and 4-GPU Prep

After the v3 prompt-fix result, GPU work can pause safely. Useful CPU-only work:

```text
1. preserve rollout examples as a readable gallery;
2. write the beginner-friendly reading path;
3. prepare the 4-GPU saved-checkpoint command;
4. check data/report consistency before renting GPUs again;
5. clean only safe temporary or empty training artifacts.
```

New artifacts:

```text
reports/deepfactcite_v3_promptfix_rollout_gallery_20260518.md
reports/deepfactcite_v3_promptfix_rollout_gallery_samples_20260518.jsonl
docs/deepfactcite_learning_index_20260518.md
docs/deepfactcite_4gpu_saved_run_plan_20260518.md
scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

The 4-GPU wrapper defaults to `DRY_RUN=1` and model-only checkpoint saving:

```text
GRPO_ACTOR_CKPT_SAVE_CONTENTS=[model]
GRPO_SAVE_FREQ=16
GRPO_MAX_ACTOR_CKPT_TO_KEEP=1
```

Reason:

```text
The disk is still tight. Model-only checkpoints are enough for evaluation and
use much less disk than resume-capable checkpoints with extra state.
```

Next GPU action:

```text
Run a short 4-GPU checkpoint-write sanity first. If the run starts cleanly,
writes a checkpoint, and keeps search/citation metrics sane, continue to the
64-step saved medium run.
```
