# DeepFactCite CPU-Mode Consistency Check

Date: 2026-05-18

Scope:

```text
data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite/
logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu/
reports/dfc_mixclean200_v3_promptfix_onecite_2gpu_rollout_summary.md
```

## Data Counts

| Item | Count |
|---|---:|
| `train.parquet` rows | 28 |
| `train.jsonl` rows | 28 |
| `test.parquet` rows | 4 |
| `test.jsonl` rows | 4 |
| `corpus.jsonl` docs | 32 |
| `preview.jsonl` rows | 32 |
| `reject_samples.jsonl` rows | 66 |

These match `summary.json`:

```text
kept_rows=32
train_rows=28
test_rows=4
corpus_docs=32
```

## Prompt Checks

All 32 train/test prompts contain the v3 prompt-fix constraints:

| Check | Result |
|---|---|
| exactly 1 concise sentence | pass |
| exactly 1 markdown citation | pass |
| URL copied exactly from tool response | pass |
| final answer shape template | pass |
| train prompt target URL leakage | 0/28 |
| test prompt target URL leakage | 0/4 |

## Rollout Counts

| Item | Count |
|---|---:|
| rollout JSONL files | 16 |
| total rollout samples | 32 |
| samples per step | 2 |

Every step file has exactly 2 samples.

## Metric Recompute

Aggregate metrics recomputed from rollout JSONL match the saved v3 summary at
4 decimal places:

| Metric | Recomputed | Summary |
|---|---:|---:|
| reward | 0.5120 | 0.5120 |
| total | 0.4807 | 0.4807 |
| format | 0.9875 | 0.9875 |
| search | 0.9375 | 0.9375 |
| URL validity | 0.9375 | 0.9375 |
| citation precision | 0.6312 | 0.6312 |
| claim support | 0.6250 | 0.6250 |
| unsupported citation rate | 0.1250 | 0.1250 |
| fake URL rate | 0.0000 | 0.0000 |
| citation count | 0.9375 | 0.9375 |

## Known Reporting Note

The rollout gallery uses stricter error-analysis tags than the compact summary.
In the gallery, every `claim_support=0.5` sample is tagged as
`weak_claim_support`, so it shows 12 weak-support samples. The compact v3
rollout summary only highlights the hard failure rows. This is a reporting
granularity difference, not a data mismatch.

## Result

The v3 data, rollout files, and summary report are internally consistent. The
next GPU run can use the v3 data path without rebuilding it.
