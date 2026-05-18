# GRPO Rollout Summary: `logs/grpo/rollouts/dfc-mixclean200-claimfiltered-outcome-only-20260518_v2_onecite_2gpu`

- Samples: 32
- Steps: 16

## Aggregate Metrics

| Metric | Mean |
|---|---:|
| reward | 0.2405 |
| total | 0.2029 |
| answer_subem | 0.0000 |
| format | 0.9812 |
| search | 0.9688 |
| url_validity | 0.1562 |
| citation_precision | 0.1125 |
| claim_support | 0.1094 |
| unsupported_citation_rate | 0.8438 |
| fake_url_rate | 0.0000 |
| citation_count | 0.1562 |

## Step Metrics

| Step | Samples | Reward | URL | Support | Unsupported | Citations |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.1969 | 0.0000 | 0.0000 | 0.5000 | 0.0000 |
| 2 | 2 | 0.2469 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 3 | 2 | 0.2471 | 0.5000 | 0.2500 | 0.5000 | 0.5000 |
| 4 | 2 | 0.2464 | 0.5000 | 0.0000 | 1.0000 | 0.5000 |
| 5 | 2 | 0.2467 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 6 | 2 | 0.2456 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 7 | 2 | 0.2465 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 8 | 2 | 0.2005 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 9 | 2 | 0.2470 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 10 | 2 | 0.2462 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 11 | 2 | 0.2467 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 12 | 2 | 0.2461 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 13 | 2 | 0.2468 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 14 | 2 | 0.2463 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 15 | 2 | 0.2466 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 16 | 2 | 0.2455 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |

## Failure Reasons

| Reason | Count |
|---|---:|
| no_citation | 27 |
| no_search | 1 |
| weak_claim_support | 1 |
| unsupported_citation | 1 |

## Failure Samples

- step=1 reward=0.1483 reasons=no_search,no_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.11929999999999999, "answer_subem": 0.0, "format": 1.0, "search": 0.0, "url_validity": 0.0, "citation_precision": 0.1, "claim_support": 0.0, "unsupported_citation_rate": 0.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The Japanese avant-garde band "Boredoms" is known for their experimental and improvisational music, often incorporating unconventional instruments and techniques.
- step=8 reward=0.1555 reasons=no_citation
  query: is petting a cat actually good for you and if so why?
  details: `{"total": 0.08548750000000001, "answer_subem": 0.0, "format": 0.4, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=12 reward=0.2453 reasons=no_citation
  query: Tell me about the John Radcliffe Hospital.
  details: `{"total": 0.14527500000000002, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The John Radcliffe Hospital is a part of the Oxford University Hospitals NHS Foundation Trust and is named after John Radcliffe, an 18th-century physician and Oxford University graduate, who endowed the Radcliffe Infirmary, the main hospital for Oxford from 1735 to 1972.
- step=16 reward=0.2454 reasons=no_citation
  query: Why is there so little trade between Russia and the United States?
  details: `{"total": 0.14539583333333336, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Trade between Russia and the United States is minimal due to economic sanctions and geopolitical tensions, as evidenced by the low value of goods and services exchanged and the limited promotion of Russia in Western media.
- step=6 reward=0.2455 reasons=no_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.14545000000000002, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The point in climate change policies is to mitigate further damage and protect vulnerable populations, even if the situation is dire, as evidenced by global efforts and agreements despite the challenges.
- step=1 reward=0.2455 reasons=no_citation
  query: What is the Japanese avant-garde band "Boredoms"?
  details: `{"total": 0.14555, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Boredoms is a Japanese experimental noise rock band known for their avant-garde style and extensive discography, including seven full-length studio albums and numerous EPs, singles, and live albums.
- step=16 reward=0.2456 reasons=no_citation
  query: Why is there so little trade between Russia and the United States?
  details: `{"total": 0.14557083333333334, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Trade between Russia and the United States is limited due to economic sanctions and geopolitical tensions, with the U.S. ranking Russia as its 61st top trading partner in 2024.
- step=6 reward=0.2457 reasons=no_citation
  query: What's the point in climate change policies if we're screwed already?
  details: `{"total": 0.14568750000000003, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Climate change policies are crucial for mitigating future impacts and preventing further degradation, even if some damage is already irreversible.
- step=4 reward=0.2460 reasons=weak_claim_support,unsupported_citation
  query: What happens to my $15/hr job if the minimum wage is increased to $15?
  details: `{"total": 0.14600000000000002, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: If the minimum wage is increased to $15, your $15/hr job may be at risk of being eliminated or replaced by automation or outsourcing, as employers may seek to reduce labor costs [S_90c5bf9d](https://www.minneapolisfed.org/article/2023/better-together-pairing-smaller-minimum-wage-increases-with-tax-policy-to-reduce-inequality).
- step=10 reward=0.2460 reasons=no_citation
  query: Why are witches of the occult often portrayed as green-skinned?
  details: `{"total": 0.14601666666666668, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Witches of the occult are often portrayed as green-skinned due to their association with nature and herbs in pre-modern European folklore, where they were depicted as knowledgeable in folk remedies and living on society's fringes.
- step=14 reward=0.2463 reasons=no_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.14626250000000002, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Jim Umbricht's career with thePittsburghPirates and Houston Colt .45s saw him transition from a starting pitcher to a relief pitcher, with his performance declining in the latter part of his career.
- step=15 reward=0.2463 reasons=no_citation
  query: Tell me about the Hugging Face Hub in machine learning.
  details: `{"total": 0.14627916666666668, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The Hugging Face Model Hub is an open-source platform where users can upload, share, and download pre-trained machine learning models, serving as a central repository for AI practitioners to reuse existing models and reduce computational costs and time-to-market.
