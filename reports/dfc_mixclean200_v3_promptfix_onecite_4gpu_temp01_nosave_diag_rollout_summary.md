# GRPO Rollout Summary: `logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260518_4gpu_temp01_nosave_diag`

- Samples: 32
- Steps: 8
- Diagnostic details recomputed with the current reward code; logged `reward` is preserved.

## Aggregate Metrics

| Metric | Mean |
|---|---:|
| reward | 0.2974 |
| total | 0.2825 |
| answer_subem | 0.0000 |
| format | 0.9000 |
| search | 0.4688 |
| url_validity | 0.4688 |
| citation_precision | 0.3469 |
| claim_support | 0.2969 |
| unsupported_citation_rate | 0.1562 |
| fake_url_rate | 0.0312 |
| citation_count | 0.5000 |

## Step Metrics

| Step | Samples | Reward | URL | Support | Unsupported | Citations |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.1955 | 0.5000 | 0.1250 | 0.5000 | 0.7500 |
| 2 | 4 | 0.5885 | 1.0000 | 0.7500 | 0.0000 | 1.0000 |
| 3 | 4 | 0.1113 | 0.2500 | 0.0000 | 0.2500 | 0.2500 |
| 4 | 4 | 0.4198 | 0.5000 | 0.5000 | 0.0000 | 0.5000 |
| 5 | 4 | 0.4205 | 0.5000 | 0.5000 | 0.0000 | 0.5000 |
| 6 | 4 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 7 | 4 | 0.1231 | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| 8 | 4 | 0.4202 | 0.5000 | 0.5000 | 0.0000 | 0.5000 |

## Failure Reasons

| Reason | Count |
|---|---:|
| no_search | 17 |
| no_citation | 16 |
| weak_claim_support | 5 |
| unsupported_citation | 5 |
| invalid_or_fake_url | 1 |

## Failure Samples

- step=1 reward=0.0982 reasons=no_search,invalid_or_fake_url,weak_claim_support,unsupported_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.09818333333333334, "answer_subem": 0.0, "format": 1.0, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 1.0, "citation_count": 1.0}`
  answer: The Japanese avant-garde band "Boredoms" is known for their experimental and noise-based music, often incorporating unconventional instruments and techniques. [Boredoms](https://en.wikipedia.org/wiki/Boredoms)
- step=4 reward=0.0997 reasons=no_search,no_citation
  query: What is the ozone hole and why does it matter that it's so small?
  details: `{"total": 0.09973333333333334, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=5 reward=0.0998 reasons=no_search,no_citation
  query: Why are witches of the occult often portrayed as green-skinned?
  details: `{"total": 0.09978333333333333, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=4 reward=0.0999 reasons=no_search,no_citation
  query: What is the ozone hole and why does it matter that it's so small?
  details: `{"total": 0.09990833333333334, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=3 reward=0.0999 reasons=no_search,no_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.0999125, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=3 reward=0.0999 reasons=no_search,no_citation
  query: What is the Chauvet-Pont d'Arc Cave?
  details: `{"total": 0.099925, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=6 reward=0.1000 reasons=no_search,no_citation
  query: How does marriage equality undermine the US constitution?
  details: `{"total": 0.09995000000000001, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=6 reward=0.1000 reasons=no_search,no_citation
  query: How does marriage equality undermine the US constitution?
  details: `{"total": 0.09995000000000001, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=6 reward=0.1001 reasons=no_search,no_citation
  query: Tell me about the John Radcliffe Hospital.
  details: `{"total": 0.10009166666666668, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=8 reward=0.1001 reasons=no_search,no_citation
  query: Why is there so little trade between Russia and the United States?
  details: `{"total": 0.10009166666666668, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=1 reward=0.1001 reasons=no_search,no_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.10013333333333334, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=5 reward=0.1001 reasons=no_search,no_citation
  query: Why are witches of the occult often portrayed as green-skinned?
  details: `{"total": 0.10013333333333334, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
