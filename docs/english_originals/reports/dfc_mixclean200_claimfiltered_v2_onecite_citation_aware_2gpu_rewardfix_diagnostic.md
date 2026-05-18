# GRPO Rollout Summary: `logs/grpo/rollouts/dfc-mixclean200-claimfiltered-citation-aware-20260518_v2_onecite_2gpu`

- Samples: 32
- Steps: 16
- Diagnostic details recomputed with the current reward code; logged `reward` is preserved.

## Aggregate Metrics

| Metric | Mean |
|---|---:|
| reward | 0.3722 |
| total | 0.3526 |
| answer_subem | 0.0000 |
| format | 0.9625 |
| search | 1.0000 |
| url_validity | 0.7188 |
| citation_precision | 0.3937 |
| claim_support | 0.3906 |
| unsupported_citation_rate | 0.3750 |
| fake_url_rate | 0.0000 |
| citation_count | 0.7188 |

## Step Metrics

| Step | Samples | Reward | URL | Support | Unsupported | Citations |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.2905 | 0.5000 | 0.2500 | 0.5000 | 0.5000 |
| 2 | 2 | 0.4365 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 3 | 2 | 0.4369 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 4 | 2 | 0.2911 | 1.0000 | 0.2500 | 0.5000 | 1.0000 |
| 5 | 2 | 0.4436 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 6 | 2 | 0.1454 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 7 | 2 | 0.2913 | 0.5000 | 0.2500 | 0.5000 | 0.5000 |
| 8 | 2 | 0.2605 | 0.5000 | 0.2500 | 0.5000 | 0.5000 |
| 9 | 2 | 0.4436 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 10 | 2 | 0.4355 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 11 | 2 | 0.4362 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 12 | 2 | 0.7406 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 13 | 2 | 0.1281 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 14 | 2 | 0.1452 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 15 | 2 | 0.7400 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 16 | 2 | 0.2900 | 1.0000 | 0.2500 | 0.5000 | 1.0000 |

## Failure Reasons

| Reason | Count |
|---|---:|
| no_citation | 9 |
| weak_claim_support | 4 |
| unsupported_citation | 4 |

## Failure Samples

- step=8 reward=0.0855 reasons=no_citation
  query: is petting a cat actually good for you and if so why?
  details: `{"total": 0.08552083333333334, "answer_subem": 0.0, "format": 0.4, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=13 reward=0.1096 reasons=no_citation
  query: Who is Marianne Weber?
  details: `{"total": 0.10959166666666668, "answer_subem": 0.0, "format": 0.4, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=16 reward=0.1449 reasons=weak_claim_support,unsupported_citation
  query: Why is there so little trade between Russia and the United States?
  details: `{"total": 0.1448666666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: The low trade volume between Russia and the United States is largely due to geopolitical tensions and sanctions, as evidenced by the significant reduction in trade from $5.26 billion in 2024, reflecting strained diplomatic relations and economic isolation [S_d1f79694](https://usafacts.org/answers/what-is-the-value-of-us-trade/countries/russia).
- step=14 reward=0.1450 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.14495833333333336, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's role and performance evolved from a starting pitcher to a relief pitcher, with his ERA fluctuating and his role becoming more specialized in the latter part of his career [S_1d6c5baf](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=6 reward=0.1450 reasons=no_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.14501666666666668, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The point of climate change policies is to mitigate future impacts and prevent further damage, even if some harm is already inevitable, as evidenced by the global efforts and strengthened policies aimed at reducing greenhouse gas emissions and promoting sustainable development [Pew Research Center, 2024].
- step=14 reward=0.1455 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.14548333333333335, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's role and performance with the Pittsburgh Pirates and Houston Colt .45s evolved from a starting pitcher to a relief pitcher, with his career statistics and contributions varying across these teams [Jim Umbricht Stats, Height, Weight, Research & History - Baseball Almanac](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=1 reward=0.1457 reasons=no_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.14566250000000003, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Boredoms is a Japanese avant-garde noise rock band known for their experimental sound and extensive discography, including seven full-length studio albums and various EPs.
- step=6 reward=0.1457 reasons=no_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.14570000000000002, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Climate change policies are essential to mitigate future impacts and prevent further degradation, even if some damage is already irreversible.
- step=4 reward=0.1462 reasons=weak_claim_support,unsupported_citation
  query: What happens to my $15/hr job if the minimum wage is increased to $15?
  details: `{"total": 0.14620000000000002, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: If the minimum wage is increased to $15, your $15/hr job may remain unchanged, as it already meets the new minimum wage threshold [S_90c5bf9d](https://www.minneapolisfed.org/article/2023/better-together-pairing-smaller-minimum-wage-increases-with-tax-policy-to-reduce-inequality).
- step=9 reward=0.1462 reasons=no_citation
  query: What is density
  details: `{"total": 0.14620000000000002, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Density is the ratio of a substance's mass to its volume, expressed as ρ = m/V, where ρ is the density, m is the mass, and V is the volume.
- step=7 reward=0.1464 reasons=no_citation
  query: What is the ozone hole and why does it matter that it's so small?
  details: `{"total": 0.14637083333333337, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The ozone hole is a seasonal depletion of ozone in the stratosphere over Antarctica, and its small size in 2025 is significant because it reflects the success of the Montreal Protocol in reducing ozone-depleting substances, which has led to a gradual recovery of the ozone layer.
- step=13 reward=0.1466 reasons=no_citation
  query: Who is Marianne Weber?
  details: `{"total": 0.14655, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Marianne Weber, born Marianne Schnitger on 2 August 1870 in Oerlinghausen, Germany, was the daughter of medical doctor Eduard Schnitger and Anna Weber, the daughter of a prominent businessman Karl Weber [S_2a49155d].
