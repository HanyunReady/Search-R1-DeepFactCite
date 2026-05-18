# GRPO Rollout Summary: `logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu`

- Samples: 32
- Steps: 16

## Aggregate Metrics

| Metric | Mean |
|---|---:|
| reward | 0.5120 |
| total | 0.4807 |
| answer_subem | 0.0000 |
| format | 0.9875 |
| search | 0.9375 |
| url_validity | 0.9375 |
| citation_precision | 0.6312 |
| claim_support | 0.6250 |
| unsupported_citation_rate | 0.1250 |
| fake_url_rate | 0.0000 |
| citation_count | 0.9375 |

## Step Metrics

| Step | Samples | Reward | URL | Support | Unsupported | Citations |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.2678 | 0.5000 | 0.2500 | 0.0000 | 0.5000 |
| 2 | 2 | 0.4367 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 3 | 2 | 0.7412 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 4 | 2 | 0.4361 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 5 | 2 | 0.4206 | 0.5000 | 0.5000 | 0.0000 | 0.5000 |
| 6 | 2 | 0.1451 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 7 | 2 | 0.4362 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 8 | 2 | 0.7397 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 9 | 2 | 0.7411 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 10 | 2 | 0.5877 | 1.0000 | 0.7500 | 0.0000 | 1.0000 |
| 11 | 2 | 0.4365 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 12 | 2 | 0.7408 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 13 | 2 | 0.7407 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 14 | 2 | 0.1460 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 15 | 2 | 0.7406 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 16 | 2 | 0.4352 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |

## Failure Reasons

| Reason | Count |
|---|---:|
| weak_claim_support | 4 |
| unsupported_citation | 4 |
| no_search | 2 |
| no_citation | 2 |

## Failure Samples

- step=1 reward=0.1003 reasons=no_search,no_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.10029166666666667, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=5 reward=0.1003 reasons=no_search,no_citation
  query: What is the Chauvet-Pont d'Arc Cave?
  details: `{"total": 0.10034166666666668, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=6 reward=0.1450 reasons=weak_claim_support,unsupported_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.14502083333333335, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Climate policies are crucial for mitigating the impacts of climate change, even if the situation seems dire, as they can prevent further damage and protect vulnerable populations [S_16db65fc](https://www.pewresearch.org/science/2024/12/09/how-americans-view-climate-change-and-policies-to-address-the-issue).
- step=6 reward=0.1453 reasons=weak_claim_support,unsupported_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.14527083333333335, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Climate change policies are still crucial for mitigating future impacts and protecting vulnerable populations, even if the situation is dire [S_142e6eff](https://www.pewresearch.org/science/2024/12/09/how-americans-view-climate-change-and-policies-to-address-the-issue).
- step=14 reward=0.1459 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.14588750000000003, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s evolved from a starting pitcher to a reliever, with his career spanning from 1959 to 1966, including stints with both teams [S_4a62ea91](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=14 reward=0.1461 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.14607083333333334, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's career with the Pittsburgh Pirates and Houston Colt .45s saw him transition from a starting pitcher to a relief pitcher, with his final season in 1965 [S_40cb031b](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
