# GRPO Rollout Summary: `logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260518_4gpu_n4_tb1_nosave_diag`

- Samples: 64
- Steps: 16
- Unique queries: 16
- Configuration: 4GPU no-save diagnostic, `GRPO_N=4`, `train_batch_size=1`, `ppo_mini_batch_size=1`, `temperature=0.3`.
- Diagnostic details are read from logged reward details; no checkpoint was written.

## Aggregate Metrics

| Metric | Mean |
|---|---:|
| reward | 0.5162 |
| total | 0.4845 |
| answer_subem | 0.0000 |
| format | 0.9938 |
| search | 0.9688 |
| url_validity | 0.9375 |
| citation_precision | 0.6359 |
| claim_support | 0.6328 |
| unsupported_citation_rate | 0.1719 |
| fake_url_rate | 0.0000 |
| citation_count | 0.9375 |

## Step Metrics

| Step | Samples | Reward | URL | Support | Unsupported | Citations |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.5039 | 0.7500 | 0.6250 | 0.0000 | 0.7500 |
| 2 | 4 | 0.3642 | 1.0000 | 0.3750 | 0.2500 | 1.0000 |
| 3 | 4 | 0.7411 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 4 | 4 | 0.2911 | 1.0000 | 0.2500 | 0.5000 | 1.0000 |
| 5 | 4 | 0.5804 | 0.7500 | 0.7500 | 0.0000 | 0.7500 |
| 6 | 4 | 0.2011 | 0.7500 | 0.1250 | 0.7500 | 0.7500 |
| 7 | 4 | 0.4361 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 8 | 4 | 0.7397 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 9 | 4 | 0.7410 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 10 | 4 | 0.4355 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 11 | 4 | 0.4234 | 0.7500 | 0.5000 | 0.2500 | 0.7500 |
| 12 | 4 | 0.7394 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 13 | 4 | 0.7406 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 14 | 4 | 0.1458 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 15 | 4 | 0.7406 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 16 | 4 | 0.4352 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |

## N=4 Group Signal

- Groups with non-zero reward sample std: 16/16
- Full-search groups: 14/16
- Partial-search groups: 2/16
- Zero-search groups: 0/16

| Step | N | Reward Mean | Reward Std | Search | Support | Unsupported | Query |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4 | 0.5039 | 0.3051 | 0.7500 | 0.6250 | 0.0000 | What is the Japanese avant-garde band "Boredoms"? |
| 2 | 4 | 0.3642 | 0.1449 | 1.0000 | 0.3750 | 0.2500 | How is there a "heroin epidemic", if some news sources say that drug use is a... |
| 3 | 4 | 0.7411 | 0.0002 | 1.0000 | 1.0000 | 0.0000 | What is the indie game "Hyper Light Drifter"? |
| 4 | 4 | 0.2911 | 0.1674 | 1.0000 | 0.2500 | 0.5000 | What happens to my $15/hr job if the minimum wage is increased to $15? |
| 5 | 4 | 0.5804 | 0.3205 | 0.7500 | 0.7500 | 0.0000 | What is the Chauvet-Pont d'Arc Cave? |
| 6 | 4 | 0.2011 | 0.1585 | 1.0000 | 0.1250 | 0.7500 | What's the point in climate change policies if we're screwed already? |
| 7 | 4 | 0.4361 | 0.0001 | 1.0000 | 0.5000 | 0.0000 | What is the ozone hole and why does it matter that it's so small? |
| 8 | 4 | 0.7397 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | is petting a cat actually good for you and if so why? |
| 9 | 4 | 0.7410 | 0.0002 | 1.0000 | 1.0000 | 0.0000 | What is density |
| 10 | 4 | 0.4355 | 0.0003 | 1.0000 | 0.5000 | 0.0000 | Why are witches of the occult often portrayed as green-skinned? |
| 11 | 4 | 0.4234 | 0.2700 | 1.0000 | 0.5000 | 0.2500 | How does marriage equality undermine the US constitution? |
| 12 | 4 | 0.7394 | 0.0009 | 1.0000 | 1.0000 | 0.0000 | Tell me about the John Radcliffe Hospital. |
| 13 | 4 | 0.7406 | 0.0001 | 1.0000 | 1.0000 | 0.0000 | Who is Marianne Weber? |
| 14 | 4 | 0.1458 | 0.0001 | 1.0000 | 0.0000 | 1.0000 | How did Jim Umbricht's performance and role with the Pittsburgh Pirates and H... |
| 15 | 4 | 0.7406 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | Tell me about the Hugging Face Hub in machine learning. |
| 16 | 4 | 0.4352 | 0.0001 | 1.0000 | 0.5000 | 0.0000 | Why is there so little trade between Russia and the United States? |

## Failure Reasons

| Reason | Count |
|---|---:|
| weak_claim_support | 9 |
| unsupported_citation | 9 |
| no_citation | 4 |
| no_search | 2 |

## Failure Samples

- step=6 reward=0.0800 reasons=no_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.08, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Climate change policies are essential to mitigate the impacts of global warming and prevent further environmental degradation, even if the situation is dire.
- step=11 reward=0.0800 reasons=no_citation
  query: How does marriage equality undermine the US constitution?
  details: `{"total": 0.08, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Marriage equality is upheld by the Supreme Court as a fundamental right under the Fourteenth Amendment's Due Process and Equal Protection Clauses, not undermining the Constitution.
- step=5 reward=0.0996 reasons=no_search,no_citation
  query: What is the Chauvet-Pont d'Arc Cave?
  details: `{"total": 0.0996375, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=1 reward=0.1001 reasons=no_search,no_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.10009166666666668, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=6 reward=0.1450 reasons=weak_claim_support,unsupported_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.1450375, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Climate change policies are essential for mitigating the worst impacts of global warming, even if the situation is dire, as they can still prevent further damage and protect vulnerable populations [S_142e6eff](https://www.pewresearch.org/science/2024/12/09/how-americans-view-climate-change-and-policies-to-address-the-issue).
- step=6 reward=0.1451 reasons=weak_claim_support,unsupported_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.1451041666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Climate change policies are crucial for mitigating future impacts and preventing further degradation of the environment, even if the current situation is dire [S_16db65fc](https://www.pewresearch.org/science/2024/12/09/how-americans-view-climate-change-and-policies-to-address-the-issue).
- step=14 reward=0.1457 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.14566250000000003, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's career transitioned from the Pittsburgh Pirates to the Houston Colt .45s (now the Houston Astros) after the 1960 season, where he continued to play as a utility infielder and outfielder [Jim Umbricht Stats, Height, Weight, Research & History - Baseball Almanac](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=14 reward=0.1457 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.14574166666666669, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's performance and role with thePittsburghPirates andHoustonColt .45s changed throughout his career, with his debut in 1959 and subsequent moves to the Colt .45s in 1962 [Jim Umbricht Stats, Height, Weight, Research & History - Baseball Almanac](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=14 reward=0.1457 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.14574583333333335, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's career transitioned from thePittsburghPirates to theHoustonColt .45s, with his performance and role evolving as he moved between teams, though specific details on his performance changes are not provided in the available sources [S_4a62ea91](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=14 reward=0.1460 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.1459791666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's career with thePittsburghPirates andHoustonColt .45s saw him transition from a starting pitcher to a relief pitcher, with his final season in 1964 as a relief pitcher for theColt .45s[1](https://www.baseball-almanac.com/players/player.php?p=umbriji01)
- step=4 reward=0.1461 reasons=weak_claim_support,unsupported_citation
  query: What happens to my $15/hr job if the minimum wage is increased to $15?
  details: `{"total": 0.14605416666666668, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Increasing the minimum wage to $15 would likely lead to higher pay for workers earning $15/hr, but could also result in job losses or reduced hours for some employees [S_90c5bf9d](https://www.minneapolisfed.org/article/2023/better-together-pairing-smaller-minimum-wage-increases-with-tax-policy-to-reduce-inequality).
- step=4 reward=0.1461 reasons=weak_claim_support,unsupported_citation
  query: What happens to my $15/hr job if the minimum wage is increased to $15?
  details: `{"total": 0.14607083333333334, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Increasing the minimum wage to $15 could lead to higher pay for workers earning $15/hr, but it might also result in reduced hours or job losses for some employees [S_71879d83](https://www.minneapolisfed.org/article/2023/better-together-pairing-smaller-minimum-wage-increases-with-tax-policy-to-reduce-inequality).
