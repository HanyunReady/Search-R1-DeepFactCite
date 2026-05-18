# GRPO Rollout Summary: `logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260518_4gpu_save_sanity`

- Samples: 32
- Steps: 8
- Diagnostic details recomputed with the current reward code; logged `reward` is preserved.

## Aggregate Metrics

| Metric | Mean |
|---|---:|
| reward | 0.4498 |
| total | 0.4233 |
| answer_subem | 0.0000 |
| format | 0.9563 |
| search | 0.7812 |
| url_validity | 0.7812 |
| citation_precision | 0.5531 |
| claim_support | 0.5312 |
| unsupported_citation_rate | 0.1250 |
| fake_url_rate | 0.0000 |
| citation_count | 0.7812 |

## Step Metrics

| Step | Samples | Reward | URL | Support | Unsupported | Citations |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.2798 | 0.7500 | 0.2500 | 0.2500 | 0.7500 |
| 2 | 4 | 0.5886 | 1.0000 | 0.7500 | 0.0000 | 1.0000 |
| 3 | 4 | 0.1839 | 0.2500 | 0.1250 | 0.0000 | 0.2500 |
| 4 | 4 | 0.4198 | 0.5000 | 0.5000 | 0.0000 | 0.5000 |
| 5 | 4 | 0.6643 | 1.0000 | 0.8750 | 0.0000 | 1.0000 |
| 6 | 4 | 0.5039 | 0.7500 | 0.6250 | 0.0000 | 0.7500 |
| 7 | 4 | 0.4432 | 1.0000 | 0.5000 | 0.5000 | 1.0000 |
| 8 | 4 | 0.5153 | 1.0000 | 0.6250 | 0.2500 | 1.0000 |

## Failure Reasons

| Reason | Count |
|---|---:|
| no_search | 7 |
| no_citation | 7 |
| weak_claim_support | 4 |
| unsupported_citation | 4 |

## Failure Samples

- step=6 reward=0.0997 reasons=no_search,no_citation
  query: How does marriage equality undermine the US constitution?
  details: `{"total": 0.09966666666666667, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=4 reward=0.0999 reasons=no_search,no_citation
  query: What is the ozone hole and why does it matter that it's so small?
  details: `{"total": 0.09988333333333334, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=3 reward=0.0999 reasons=no_search,no_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.099925, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=4 reward=0.1001 reasons=no_search,no_citation
  query: What is the ozone hole and why does it matter that it's so small?
  details: `{"total": 0.10005, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=1 reward=0.1001 reasons=no_search,no_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.10009583333333334, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=3 reward=0.1002 reasons=no_search,no_citation
  query: What is the Chauvet-Pont d'Arc Cave?
  details: `{"total": 0.10019166666666668, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=3 reward=0.1003 reasons=no_search,no_citation
  query: What is the Chauvet-Pont d'Arc Cave?
  details: `{"total": 0.10034166666666668, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=8 reward=0.1452 reasons=weak_claim_support,unsupported_citation
  query: Why is there so little trade between Russia and the United States?
  details: `{"total": 0.14516250000000003, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: There is so little trade between Russia and the United States due to geopolitical tensions and sanctions, which have significantly reduced economic interactions between the two countries [S_d1f79694](https://usafacts.org/answers/what-is-the-value-of-us-trade/countries/russia).
- step=7 reward=0.1457 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.1456791666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s evolved from a starting pitcher to a reliever, with his career totals and statistics reflecting his transition and contributions to both teams [S_f452ae32](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=7 reward=0.1457 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.1457166666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s changed from a starting pitcher to a relief pitcher, with his career totals reflecting his transition and contributions to both teams [S_f452ae32](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=1 reward=0.1469 reasons=weak_claim_support,unsupported_citation
  query: How is there a "heroin epidemic", if some news sources say that drug use is at the lowest it's been in years?
  details: `{"total": 0.1468666666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Drug-related deaths have increased, particularly among older adults, despite overall drug use being at its lowest in years, highlighting a specific epidemic in certain demographics [S_5566a0e2](https://drugabusestatistics.org).
