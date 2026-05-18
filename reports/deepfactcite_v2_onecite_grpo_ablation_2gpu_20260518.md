# DeepFactCite v2 One-Citation GRPO Ablation

Date: 2026-05-18

## Why v2 Was Needed

The first claim-filtered dataset improved retriever control but still allowed a
failure mode:

```text
The original question could be broad, while the selected supported claim was
narrow. The model answered the broad question, added unsupported background,
and attached a real URL to a claim that the snippet did not fully support.
```

v2 narrows the training problem so the reward can isolate citation behavior:

```text
one selected citation per row
one-sentence prompt
max query tokens = 18
max claim tokens = 30
retrieved URL only
top-2 retriever hit verified on train/test
```

Data:

```text
data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite/train.parquet
data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite/test.parquet
data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite/corpus.jsonl
```

Build result:

| Item | Value |
|---|---:|
| kept rows | 32 |
| train/test | 28/4 |
| corpus docs | 32 |
| avg selected citations | 1.0000 |
| min support score | 1.0000 |
| train top-2 URL hit | 28/28 |
| test top-2 URL hit | 4/4 |

Reject reasons:

| Reason | Count |
|---|---:|
| weak_claim_support | 43 |
| query_too_broad | 19 |
| too_few_supported_claims | 3 |
| claim_too_broad | 1 |

## Runs

Both runs used:

```text
actor = outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-mix-clean-200-merged-bf16
2 x A800 80G
SGLang rollout
reward_manager = deepfactcite_custom
rollout.n = 2
train_batch_size = 1
total_training_steps = 16
SEARCHQA_TOPK = 2
max_response_length = 384
save_freq = 0
```

Commands:

```bash
RUN_TAG=20260518_v2_onecite_2gpu MODE=citation-aware \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite \
DRY_RUN=0 GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh

RUN_TAG=20260518_v2_onecite_2gpu MODE=outcome-only \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite \
DRY_RUN=0 GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

Artifacts:

```text
reports/dfc_mixclean200_claimfiltered_v2_onecite_citation_aware_2gpu_rollout_summary.md
reports/dfc_mixclean200_claimfiltered_v2_onecite_outcome_only_2gpu_rollout_summary.md
logs/grpo/rollouts/dfc-mixclean200-claimfiltered-citation-aware-20260518_v2_onecite_2gpu/
logs/grpo/rollouts/dfc-mixclean200-claimfiltered-outcome-only-20260518_v2_onecite_2gpu/
```

## Result

Again, raw `reward` is not directly comparable across modes because reward
weights differ. Use the diagnostic metrics.

| Metric | v2 Outcome-Only | v2 Citation-Aware | Readout |
|---|---:|---:|---|
| samples | 32 | 32 | same |
| search | 0.9688 | 1.0000 | citation-aware slightly better |
| format | 0.9812 | 0.9625 | outcome-only slightly better |
| answer_subem | 0.0000 | 0.0000 | no useful answer EM signal on this data |
| URL validity | 0.1562 | 0.7188 | citation-aware much better |
| citation precision | 0.1125 | 0.3937 | citation-aware much better |
| claim support | 0.1094 | 0.3906 | citation-aware much better |
| unsupported citation rate | 0.8438 | 0.3750 | citation-aware much better |
| fake URL rate | 0.0000 | 0.0000 | both clean on fake URLs |
| citation count | 0.1562 | 0.7188 | citation-aware cites far more often |
| response clip ratio | 0.0000 | 0.0000 | v2 fixed clipping |

Failure counts:

| Reason | v2 Outcome-Only | v2 Citation-Aware |
|---|---:|---:|
| no_citation | 27 | 9 |
| no_search | 1 | 0 |
| weak_claim_support | 1 | 4 |
| unsupported_citation | 1 | 4 |

## Interpretation

This is the first clean positive GRPO signal in the current stage:

```text
On the same one-citation data and same rollout stack, explicit citation/support
reward strongly improves citation behavior over outcome-only reward while
preserving search and eliminating response clipping.
```

The result is still not a final model result because:

```text
1. It is only 16 steps / 32 sampled trajectories.
2. save_freq=0, so no trained checkpoint was kept.
3. citation-aware still has 9 no-citation failures.
4. claim support is improved but still only 0.3906.
```

The remaining dominant failure is not fake URLs; it is citation omission or
bare snippet-label behavior. Examples include answers that use no markdown URL,
or cite `[S_xxx]` without `[label](URL)`, which the reward correctly treats as
no citation.

## Decision

Do not jump straight to 4 GPUs yet.

The next justified experiment is not a longer blind run. It is a small reward
weight ablation on the same v2 data:

```text
answer = 0.05
citation = 0.45
support = 0.35
format = 0.10
search = 0.05
cost = 0.05
```

Purpose:

```text
Test whether stronger citation/support reward reduces the remaining no-citation
failures without damaging search/format.
```

Promotion rule:

```text
Only consider saved checkpoints or 4-GPU scale after a small run keeps search
near 1.0, keeps clip ratio near 0, reduces no_citation below 9/32, and improves
claim_support over 0.3906.
```

## Citation-Strong Follow-Up

Run:

```bash
RUN_TAG=20260518_v2_onecite_strong_2gpu \
MODE=citation-aware \
EXPERIMENT_NAME=dfc-mixclean200-v2-onecite-citation-strong-20260518_2gpu \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite \
DFC_ANSWER_WEIGHT=0.05 \
DFC_CITATION_WEIGHT=0.45 \
DFC_SUPPORT_WEIGHT=0.35 \
DFC_FORMAT_WEIGHT=0.10 \
DFC_SEARCH_WEIGHT=0.05 \
DFC_COST_WEIGHT=0.05 \
DRY_RUN=0 \
GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

Artifact:

```text
reports/dfc_mixclean200_v2_onecite_citation_strong_2gpu_rollout_summary.md
logs/grpo/rollouts/dfc-mixclean200-v2-onecite-citation-strong-20260518_2gpu/
```

Comparison against default v2 citation-aware:

| Metric | v2 Citation-Aware | v2 Citation-Strong | Readout |
|---|---:|---:|---|
| search | 1.0000 | 1.0000 | same |
| format | 0.9625 | 0.9812 | strong better |
| URL validity | 0.7188 | 0.7188 | same |
| citation precision | 0.3937 | 0.4219 | strong better |
| claim support | 0.3906 | 0.4219 | strong better |
| unsupported citation rate | 0.3750 | 0.4375 | strong worse |
| fake URL rate | 0.0000 | 0.0000 | same |
| citation count | 0.7188 | 0.7188 | same |
| no_citation failures | 9 | 9 | not fixed |
| response clip ratio | 0.0000 | 0.0000 | same |

Interpretation:

```text
Increasing citation/support weights helped average claim_support, but it did not
reduce no_citation and it made unsupported rate worse. This means the remaining
problem is not solved by scalar weight tuning alone.
```

The next technical fix should target citation formatting and omission directly:

```text
1. Add an explicit citation_presence/no_citation penalty or reward component.
2. Penalize bare labels such as [S_xxx] and [1] when no URL is present.
3. Add a prompt/data rule that the final answer must contain exactly one
   markdown URL citation copied from the snippet.
4. Keep v2 one-citation filtering and rerun a 16-step smoke before scaling.
```

Updated decision:

```text
v2 data is a real improvement and citation-aware beats outcome-only on the same
data. However, do not move to 4 GPUs until citation omission is directly fixed
and a saved checkpoint can be evaluated. The right next engineering task is a
reward/prompt patch for exact markdown citation presence, not a larger run.
```

## Markdown-Cap / Parser-Fix Follow-Up

Why this follow-up was needed:

```text
The v2 default run proved that citation-aware reward helps, but it still had
9/32 no-citation failures. The citation-strong run did not reduce that count,
so the next controlled change should target citation presence and formatting
directly instead of only increasing scalar weights.
```

Code changes:

```text
deepfactcite/reward.py
scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py
scripts/deepfactcite/summarize_grpo_rollouts.py
```

Reward/prompt behavior after the patch:

```text
1. citation-aware mode exports DFC_REQUIRE_MARKDOWN_CITATION=true.
2. outcome-only mode exports DFC_REQUIRE_MARKDOWN_CITATION=false so the ablation
   remains fair.
3. If retrieved evidence exists but the answer has no markdown URL citation,
   citation-aware reward is capped at DFC_NO_CITATION_CAP=0.08.
4. Bare labels such as [S_xxx], bare [1], or raw URLs outside markdown links are
   capped at DFC_BAD_CITATION_FORMAT_CAP=0.12.
5. The prompt now explicitly asks for exactly one markdown URL citation copied
   from retrieved evidence and forbids bare snippet labels/raw URLs.
```

Parser bug found during this follow-up:

```text
The original citation regex did not count markdown links whose label contained
bracketed text, for example:

[NCDAS: Substance Abuse and Addiction Statistics [2025]](https://drugabusestatistics.org)

That is a normal-looking model output, but the old regex treated it as
citation_count=0 and could also count the inner [2025] as a bare bracket.
```

Fix:

```text
deepfactcite/reward.py now uses a small markdown-link scanner instead of the old
flat regex. It counts bracketed link labels, strips legal markdown links before
bare-bracket/raw-URL diagnostics, and keeps the same retrieved-URL validity and
claim-support checks.
```

Sanity check:

```text
Nested-label markdown citation: citation_count=1, raw_url_count=0, bare_citation_count=0.
Bare [S_1] plus raw URL: citation_count=0, bare_citation_count=1, raw_url_count=1.
```

The first markdown-cap run was interrupted before completion:

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
Only rollout_data_step_1.jsonl through rollout_data_step_6.jsonl were written.
No Python traceback/OOM/NCCL error appeared in the trainer log. GPU was idle
after the interruption. Treat this as an incomplete smoke, not as a valid
16-step result.
```

Partial metrics before parser recomputation:

| Metric | Partial 6-Step |
|---|---:|
| samples | 12 |
| reward | 0.2319 |
| total | 0.2214 |
| search | 0.7500 |
| URL validity | 0.2500 |
| claim support | 0.2083 |
| unsupported citation rate | 0.5833 |
| citation count | 0.4167 |
| no_citation failures | 7 |

Diagnostic recomputation with the fixed parser:

| Metric | Partial 6-Step Recomputed |
|---|---:|
| total | 0.2492 |
| URL validity | 0.3333 |
| citation count | 0.5000 |
| no_citation failures | 6 |

Interpretation:

```text
The parser fix matters for diagnostics, but the partial run is too short and was
not completed. Do not compare it against the completed v2 runs as a training
result.
```

Current rerun:

```bash
RUN_TAG=20260518_v2_markdowncap_parserfix2_2gpu \
MODE=citation-aware \
EXPERIMENT_NAME=dfc-mixclean200-v2-onecite-markdowncap-parserfix2-20260518_2gpu \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v2_onecite \
DRY_RUN=0 \
GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

Run status at launch:

```text
SGLang/GRPO entered async rollout.
SearchQAVerlTool initialized with v2 one-cite offline corpus.
rollout_data_step_1.jsonl was created.
save_freq=0, so no checkpoint is being written.
```

Decision rule for this rerun:

```text
If no_citation drops below 9/32 while search stays near 1.0, response clip ratio
stays 0, and claim_support beats 0.3906, keep this reward/prompt direction.
If no_citation stays near 9/32 or support worsens, do not scale. The next fix
should be data/prompt supervision or an explicit positive citation-presence
reward component, not 4 GPUs.
```

Completed parser-fix rerun:

```text
reports/dfc_mixclean200_v2_onecite_markdowncap_parserfix2_2gpu_rollout_summary.md
logs/grpo/rollouts/dfc-mixclean200-v2-onecite-markdowncap-parserfix2-20260518_2gpu/
```

Comparison:

| Metric | v2 Citation-Aware | Parser-Fix Markdown-Cap | Promotion Check |
|---|---:|---:|---|
| search | 1.0000 | 1.0000 | pass |
| format | 0.9625 | 1.0000 | pass |
| URL validity | 0.7188 | 0.6875 | worse |
| citation precision | 0.3937 | 0.4406 | better |
| claim support | 0.3906 | 0.4375 | pass |
| unsupported citation rate | 0.3750 | 0.4062 | fail |
| fake URL rate | 0.0000 | 0.0000 | pass |
| citation count | 0.7188 | 0.6875 | slightly worse |
| no_citation failures | 9 | 10 | fail |
| response clip ratio | 0.0000 | 0.0000 | pass |

Decision:

```text
Do not scale to 4 or 8 GPUs from this result. The parser/prompt/cap patch
improved support and formatting but did not reduce no_citation below the prior
9/32 baseline and worsened unsupported rate relative to 0.3750. The next
iteration should remain a cheap 2-GPU controlled fix.
```

Where the 4-GPU gate came from:

```text
no_citation < 9/32:
  9/32 was the best completed v2 citation-aware baseline. A larger run should
  beat the known remaining failure count, not merely reproduce it.

claim_support > 0.3906:
  0.3906 was the completed v2 citation-aware baseline. Scaling is justified only
  if the support signal improves.

unsupported_citation_rate <= 0.3750:
  0.3750 was the completed v2 citation-aware baseline. A higher unsupported
  rate means the model may be citing more confidently but less reliably.

search ~= 1.0 and response clip ratio = 0:
  These are guardrails. Citation reward is not useful if it damages Search-R1
  search behavior or reintroduces truncation.

clear disk before checkpoint:
  /root/autodl-tmp had about 14G free. The short ablations used save_freq=0.
  Any promotion run must save a checkpoint for downstream eval, so disk cleanup
  is required before spending more GPU time.
```

## v3 Prompt-Fix Follow-Up

Why this exists:

```text
The parser-fix run used the new reward code but still used the already-generated
v2 parquet. That parquet did not yet contain the stricter prompt text forbidding
bare [S_xxx], bare [1], and raw URLs. Therefore the next controlled variable is
the data prompt, not GPU scale.
```

Data generation:

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

Validation:

```text
kept rows: 32
train/test: 28/4
corpus docs: 32
train top-2 target URL hit: 28/28
test top-2 target URL hit: 4/4
```

Run:

```bash
RUN_TAG=20260518_v3_promptfix_2gpu \
MODE=citation-aware \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu \
DATA_DIR=/root/autodl-tmp/Search-R1-DeepFactCite/data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite \
DRY_RUN=0 \
GRPO_TOTAL_STEPS=16 \
bash scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

Status:

```text
Launched on 2 GPUs with save_freq=0.
The trainer accepted train=28 / val=4.
SearchQAVerlTool initialized against the v3 prompt-fix corpus.
```

Completed result:

```text
reports/dfc_mixclean200_v3_promptfix_onecite_2gpu_rollout_summary.md
logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu/
```

Comparison:

| Metric | v2 Citation-Aware | Parser-Fix Markdown-Cap | v3 Prompt-Fix |
|---|---:|---:|---:|
| search | 1.0000 | 1.0000 | 0.9375 |
| format | 0.9625 | 1.0000 | 0.9875 |
| URL validity | 0.7188 | 0.6875 | 0.9375 |
| citation precision | 0.3937 | 0.4406 | 0.6312 |
| claim support | 0.3906 | 0.4375 | 0.6250 |
| unsupported citation rate | 0.3750 | 0.4062 | 0.1250 |
| fake URL rate | 0.0000 | 0.0000 | 0.0000 |
| citation count | 0.7188 | 0.6875 | 0.9375 |
| no_citation failures | 9 | 10 | 2 |
| no_search failures | 0 | 0 | 2 |
| response clip ratio | 0.0000 | 0.0000 | 0.0000 |

Interpretation:

```text
This is the strongest current 2-GPU result. The key missing piece was not GPU
scale; it was that the old v2 parquet did not contain the stricter citation
format prompt. Once the stricter prompt and reward parser/caps were aligned,
no_citation dropped from 9/32 to 2/32 and claim_support rose from 0.3906 to
0.6250.
```

Residual risk:

```text
search dropped from 1.0000 to 0.9375 because two samples failed to search and
returned no answer. This is acceptable for a medium-scale validation, but it is
the guardrail to watch before any final-scale run.
```

Decision:

```text
The v3 prompt-fix mechanism passes the local 2-GPU gate. The next justified step
is not 8 GPUs. It is a saved medium run after disk cleanup: preferably 4 GPUs if
available, or 2 GPUs if this machine remains limited to 2 visible GPUs. The run
must save a checkpoint and then be evaluated, otherwise it is only another smoke.
```

Audit checks:

```text
Independent recomputation over 32 saved rollout rows:
  reward recomputed with training weights exactly matched logged reward
  max reward diff = 0.0
  logged diagnostic details matched current parser recomputation
  metric changes vs logged details = 0

Prompt leakage check:
  train prompt target-prefix leaks = 0/28
  test prompt target-prefix leaks = 0/4

Retriever check with OfflineSearchTool and v3 corpus:
  train target URL top-1 hit = 28/28
  test target URL top-1 hit = 4/4

Citation parser diagnostics:
  bare_citation_count mean = 0.0
  raw_url_count mean = 0.0
  no_search/no_citation rows = 2, both had answer=None
  unsupported rows = 4, concentrated in climate-policy and Jim Umbricht examples
```

Label-quality caveat:

```text
30 markdown citations were produced. 22 used labels like [S_xxx](URL), and 8
used human-readable labels. This is not the old failure mode because the URL is
present and provenance can be checked. However, if the final product requires
human-readable citation labels rather than snippet IDs, add a separate
label-quality metric/reward before final scale.
```
