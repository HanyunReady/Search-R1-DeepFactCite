# GRPO Rollout Summary: `logs/grpo/rollouts/dfc-mixclean200-v2-onecite-markdowncap-20260518_2gpu`

- Samples: 12
- Steps: 6
- Diagnostic details recomputed with the current reward code; logged `reward` is preserved.

## Aggregate Metrics

| Metric | Mean |
|---|---:|
| reward | 0.2319 |
| total | 0.2492 |
| answer_subem | 0.0000 |
| format | 0.9833 |
| search | 0.7500 |
| url_validity | 0.3333 |
| citation_precision | 0.2250 |
| claim_support | 0.2083 |
| unsupported_citation_rate | 0.5833 |
| fake_url_rate | 0.1667 |
| citation_count | 0.5000 |

## Step Metrics

| Step | Samples | Reward | URL | Support | Unsupported | Citations |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.0998 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 2 | 2 | 0.0800 | 0.5000 | 0.0000 | 1.0000 | 0.5000 |
| 3 | 2 | 0.4105 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 4 | 2 | 0.0989 | 0.0000 | 0.0000 | 0.5000 | 0.5000 |
| 5 | 2 | 0.4103 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 6 | 2 | 0.2917 | 0.5000 | 0.2500 | 0.5000 | 1.0000 |

## Failure Reasons

| Reason | Count |
|---|---:|
| no_citation | 6 |
| no_search | 3 |
| weak_claim_support | 3 |
| unsupported_citation | 3 |
| invalid_or_fake_url | 2 |

## Failure Samples

- step=1 reward=0.0800 reasons=no_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.14566250000000003, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Boredoms is a Japanese avant-garde noise rock band known for their experimental music and extensive discography, including seven full-length studio albums and various EPs.
- step=2 reward=0.0800 reasons=weak_claim_support,unsupported_citation
  query: How is there a "heroin epidemic", if some news sources say that drug use is at the lowest it's been in years?
  details: `{"total": 0.14652500000000002, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: The "heroin epidemic" persists despite overall drug use being at its lowest in years because specific age groups, particularly those over 40, are experiencing a significant increase in drug-related deaths and heroin use [NCDAS: Substance Abuse and Addiction Statistics [2025]](https://drugabusestatistics.org).
- step=2 reward=0.0800 reasons=no_citation
  query: How is there a "heroin epidemic", if some news sources say that drug use is at the lowest it's been in years?
  details: `{"total": 0.1469291666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The "heroin epidemic" persists despite overall drug use being at its lowest in years because specific demographics, particularly older adults, are experiencing a significant increase in heroin-related deaths.
- step=3 reward=0.0800 reasons=no_citation
  query: What is the indie game "Hyper Light Drifter"?
  details: `{"total": 0.14723333333333335, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: "Hyper Light Drifter" is an action role-playing game developed and published by Heart Machine, released for multiple platforms including Windows, Linux, OS X, PlayStation 4, and Xbox One in 2016.
- step=5 reward=0.0800 reasons=no_citation
  query: What is the Chauvet-Pont d'Arc Cave?
  details: `{"total": 0.14682083333333334, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: TheChauvet-Pont-d'ArcCaveis acavethat contains some of the best-preserved figurativecavepaintings in the world. [https://en.wikipedia.org/wiki/Chauvet_Cave]
- step=4 reward=0.0981 reasons=no_search,invalid_or_fake_url,weak_claim_support,unsupported_citation
  query: What happens to my $15/hr job if the minimum wage is increased to $15?
  details: `{"total": 0.09806666666666668, "answer_subem": 0.0, "format": 1.0, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 1.0, "citation_count": 1.0}`
  answer: The minimum wage increase to $15/hr could lead to job losses or reduced hours for some workers earning $15/hr, as employers may adjust to the higher cost [minimum wage impact](https://www.bls.gov/news.release/pdf/mwr.pdf).
- step=4 reward=0.0997 reasons=no_search,no_citation
  query: What happens to my $15/hr job if the minimum wage is increased to $15?
  details: `{"total": 0.0997375, "answer_subem": 0.0, "format": 0.8, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=1 reward=0.1196 reasons=no_search,no_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.11956666666666667, "answer_subem": 0.0, "format": 1.0, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The Japanese avant-garde band "Boredoms" is known for their experimental and improvisational music, blending elements of rock, jazz, and noise.
- step=6 reward=0.1485 reasons=invalid_or_fake_url,weak_claim_support,unsupported_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.14852083333333335, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 1.0, "citation_count": 1.0}`
  answer: Climate change policies are essential to mitigate future impacts and prevent further degradation of the environment, even if some damage is already irreversible [climate change policies](https://www.britannica.com/science/climate-change-policy).
