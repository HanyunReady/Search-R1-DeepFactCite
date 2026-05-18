# GRPO Rollout Summary: `logs/grpo/rollouts/dfc-mixclean200-v2-onecite-markdowncap-parserfix2-20260518_2gpu`

- Samples: 32
- Steps: 16

## Aggregate Metrics

| Metric | Mean |
|---|---:|
| reward | 0.3862 |
| total | 0.3643 |
| answer_subem | 0.0000 |
| format | 1.0000 |
| search | 1.0000 |
| url_validity | 0.6875 |
| citation_precision | 0.4406 |
| claim_support | 0.4375 |
| unsupported_citation_rate | 0.4062 |
| fake_url_rate | 0.0000 |
| citation_count | 0.6875 |

## Step Metrics

| Step | Samples | Reward | URL | Support | Unsupported | Citations |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.0800 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 2 | 2 | 0.4365 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 3 | 2 | 0.2584 | 0.5000 | 0.2500 | 0.5000 | 0.5000 |
| 4 | 2 | 0.1460 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 5 | 2 | 0.0800 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 6 | 2 | 0.0800 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 7 | 2 | 0.4101 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 8 | 2 | 0.7395 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 9 | 2 | 0.7409 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 10 | 2 | 0.4354 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 11 | 2 | 0.4362 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 12 | 2 | 0.7389 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 13 | 2 | 0.1247 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 14 | 2 | 0.1449 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 15 | 2 | 0.7399 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 16 | 2 | 0.5872 | 1.0000 | 0.7500 | 0.0000 | 1.0000 |

## Failure Reasons

| Reason | Count |
|---|---:|
| no_citation | 10 |
| weak_claim_support | 4 |
| unsupported_citation | 4 |

## Failure Samples

- step=1 reward=0.0800 reasons=no_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.08, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Boredoms is a Japanese experimental noise rock band known for their avant-garde style and extensive discography, including seven full-length studio albums, eleven EPs, and various side projects.
- step=1 reward=0.0800 reasons=no_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.08, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Boredoms is a Japanese experimental noise rock band known for their avant-garde style and extensive discography, including seven full-length studio albums, eleven EPs, and various side projects.
- step=3 reward=0.0800 reasons=no_citation
  query: What is the indie game "Hyper Light Drifter"?
  details: `{"total": 0.08, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: "Hyper Light Drifter" is an action role-playing game developed and published by Heart Machine, released for Microsoft Windows, Linux, OS X, PlayStation 4, and Xbox One in 2016.
- step=5 reward=0.0800 reasons=no_citation
  query: What is the Chauvet-Pont d'Arc Cave?
  details: `{"total": 0.08, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The Chauvet-Pont d'Arc Cave is a cave in southeastern France with some of the best-preserved figurative cave paintings in the world. [https://en.wikipedia.org/wiki/Chauvet_Cave]
- step=5 reward=0.0800 reasons=no_citation
  query: What is the Chauvet-Pont d'Arc Cave?
  details: `{"total": 0.08, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The Chauvet-Pont d'Arc Cave is a cave in southeastern France with some of the best-preserved figurative cave paintings in the world. [https://en.wikipedia.org/wiki/Chauvet_Cave]
- step=6 reward=0.0800 reasons=no_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.08, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The point in climate change policies is to mitigate further damage and prevent the worst-case scenarios, even if some impacts are already unavoidable.
- step=6 reward=0.0800 reasons=no_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.08, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Climate change policies are crucial for mitigating future impacts and preventing the worst-case scenarios, even if some damage is already inevitable.
- step=7 reward=0.0800 reasons=no_citation
  query: What is the ozone hole and why does it matter that it's so small?
  details: `{"total": 0.08, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The ozone hole is a seasonal depletion of ozone in the stratosphere over Antarctica, and its small size in 2025 indicates progress in reducing ozone-depleting substances, thanks to the Montreal Protocol.
- step=13 reward=0.0800 reasons=no_citation
  query: Who is Marianne Weber?
  details: `{"total": 0.08, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Marianne Weber was born on August 2, 1870, in Oerlinghausen, Germany, to Eduard Schnitger and Anna Weber, who was the daughter of a prominent businessman, Karl Weber. [1]
- step=14 reward=0.1445 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.14452083333333335, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s changed throughout his Major League Baseball career as he transitioned from a starting pitcher to a relief pitcher, with his ERA and win-loss record reflecting these shifts in roles and teams [JimUmbrichtBaseball Stats | Baseball Almanac](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=14 reward=0.1454 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.14537083333333337, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s changed from a starting pitcher to a reliever, with his career totals reflecting this shift [Jim Umbricht Stats, Height, Weight, Research & History - Baseball Almanac](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=4 reward=0.1460 reasons=weak_claim_support,unsupported_citation
  query: What happens to my $15/hr job if the minimum wage is increased to $15?
  details: `{"total": 0.14600000000000002, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: If the minimum wage is increased to $15, your $15/hr job may be at risk of being eliminated or replaced by automation or outsourcing, as employers may seek to reduce labor costs [S_90c5bf9d](https://www.minneapolisfed.org/article/2023/better-together-pairing-smaller-minimum-wage-increases-with-tax-policy-to-reduce-inequality).
