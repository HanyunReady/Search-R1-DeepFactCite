# GRPO Rollout Summary: `logs/grpo/rollouts/dfc-mixclean200-v2-onecite-citation-strong-20260518_2gpu`

- Samples: 32
- Steps: 16

## Aggregate Metrics

| Metric | Mean |
|---|---:|
| reward | 0.4487 |
| total | 0.3716 |
| answer_subem | 0.0000 |
| format | 0.9812 |
| search | 1.0000 |
| url_validity | 0.7188 |
| citation_precision | 0.4219 |
| claim_support | 0.4219 |
| unsupported_citation_rate | 0.4375 |
| fake_url_rate | 0.0000 |
| citation_count | 0.7188 |

## Step Metrics

| Step | Samples | Reward | URL | Support | Unsupported | Citations |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.1154 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 2 | 2 | 0.1463 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 3 | 2 | 0.3247 | 0.5000 | 0.2500 | 0.5000 | 0.5000 |
| 4 | 2 | 0.1458 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 5 | 2 | 0.1467 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 6 | 2 | 0.3226 | 0.5000 | 0.2500 | 0.5000 | 0.5000 |
| 7 | 2 | 0.5011 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 8 | 2 | 0.8736 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 9 | 2 | 0.8750 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 10 | 2 | 0.5004 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 11 | 2 | 0.5012 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 12 | 2 | 0.8737 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 13 | 2 | 0.5106 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 14 | 2 | 0.1449 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 15 | 2 | 0.8740 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 16 | 2 | 0.3226 | 1.0000 | 0.2500 | 0.5000 | 1.0000 |

## Failure Reasons

| Reason | Count |
|---|---:|
| no_citation | 9 |
| weak_claim_support | 5 |
| unsupported_citation | 5 |

## Failure Samples

- step=1 reward=0.0853 reasons=no_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.08527083333333334, "answer_subem": 0.0, "format": 0.4, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=14 reward=0.1444 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.14438333333333336, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s changed throughout his Major League Baseball career as he transitioned from a starting pitcher to a relief pitcher, with his ERA and win-loss record fluctuating based on his role and team needs [Jim Umbricht Stats, Height, Weight, Research & History - Baseball Almanac](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=16 reward=0.1453 reasons=weak_claim_support,unsupported_citation
  query: Why is there so little trade between Russia and the United States?
  details: `{"total": 0.14527916666666668, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: The trade volume between Russia and the United States is low due to geopolitical tensions and sanctions, which have significantly reduced economic cooperation [USAFacts](https://usafacts.org/answers/what-is-the-value-of-us-trade/countries/russia).
- step=14 reward=0.1454 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.1454166666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s evolved from a starting pitcher to a relief pitcher, with his career statistics and contributions varying across different seasons and teams [Jim Umbricht Stats, Height, Weight, Research & History - Baseball Almanac](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=6 reward=0.1454 reasons=no_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.1454416666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Climate change policies are crucial to mitigate the worst impacts of global warming, even if the situation is dire, as they can still prevent further catastrophic damage and protect vulnerable populations.
- step=4 reward=0.1455 reasons=weak_claim_support,unsupported_citation
  query: What happens to my $15/hr job if the minimum wage is increased to $15?
  details: `{"total": 0.14551666666666668, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: If the minimum wage is increased to $15, your $15/hr job may see no immediate change in pay, but it could lead to job losses or reduced hours for lower-wage workers, as employers may adjust to higher labor costs [Better together: Pairing smaller minimum wage increases with tax policy to reduce inequality](https://www.minneapolisfed.org/article/2023/better-together-pairing-smaller-minimum-wage-increases-with-tax-policy-to-reduce-inequality).
- step=1 reward=0.1456 reasons=no_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.14556666666666668, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Boredoms is a Japanese experimental noise rock band known for their avant-garde sound and extensive discography, including seven full-length studio albums, eleven EPs, and various side projects.
- step=4 reward=0.1461 reasons=weak_claim_support,unsupported_citation
  query: What happens to my $15/hr job if the minimum wage is increased to $15?
  details: `{"total": 0.14608333333333334, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: If the minimum wage is increased to $15, your $15/hr job may see a reduction in hours or potential job loss, as employers may adjust to the higher wage costs [S_90c5bf9d](https://www.minneapolisfed.org/article/2023/better-together-pairing-smaller-minimum-wage-increases-with-tax-policy-to-reduce-inequality).
- step=2 reward=0.1463 reasons=no_citation
  query: How is there a "heroin epidemic", if some news sources say that drug use is at the lowest it's been in years?
  details: `{"total": 0.14630833333333335, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The "heroin epidemic" persists despite overall drug use being at its lowest in years because the rate of drug use among older adults, particularly those aged 55-64 and 65+, is increasing significantly, as evidenced by a 2.3% and 11.4% rise in drug-related deaths in these age groups between 2022 and 2023 [NCDAS: Substance Abuse and Addiction Statistics [2025]].
- step=2 reward=0.1464 reasons=no_citation
  query: How is there a "heroin epidemic", if some news sources say that drug use is at the lowest it's been in years?
  details: `{"total": 0.1463541666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The "heroin epidemic" persists despite overall drug use being at its lowest in years because the rate of heroin-related deaths among older adults (55-64 and 65+) has increased significantly, indicating a specific and concerning trend within this demographic [NCDAS: Substance Abuse and Addiction Statistics [2025]](https://drugabusestatistics.org).
- step=13 reward=0.1466 reasons=no_citation
  query: Who is Marianne Weber?
  details: `{"total": 0.14662083333333337, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Marianne Weber, born Marianne Schnitger on 2 August 1870 in Oerlinghausen, Germany, was the daughter of medical doctor Eduard Schnitger and Anna Weber, the daughter of a prominent businessman Karl Weber [1].
- step=5 reward=0.1467 reasons=no_citation
  query: What is the Chauvet-Pont d'Arc Cave?
  details: `{"total": 0.14673333333333335, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The Chauvet-Pont d'Arc Cave is a cave in southeastern France with some of the best-preserved figurative cave paintings in the world. [https://en.wikipedia.org/wiki/Chauvet_Cave]
