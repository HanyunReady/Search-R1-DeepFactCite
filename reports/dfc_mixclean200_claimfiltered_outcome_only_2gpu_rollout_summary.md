# GRPO Rollout Summary: `logs/grpo/rollouts/dfc-mixclean200-claimfiltered-outcome-only-20260518_claimfiltered_2gpu`

- Samples: 32
- Steps: 16

## Aggregate Metrics

| Metric | Mean |
|---|---:|
| reward | 0.2420 |
| total | 0.3713 |
| answer_subem | 0.0000 |
| format | 0.9812 |
| search | 1.0000 |
| url_validity | 0.7656 |
| citation_precision | 0.4219 |
| claim_support | 0.4219 |
| unsupported_citation_rate | 0.4219 |
| fake_url_rate | 0.0469 |
| citation_count | 1.0625 |

## Step Metrics

| Step | Samples | Reward | URL | Support | Unsupported | Citations |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.2469 | 1.0000 | 0.7500 | 0.0000 | 1.0000 |
| 2 | 2 | 0.2456 | 1.0000 | 0.8750 | 0.0000 | 1.5000 |
| 3 | 2 | 0.2459 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 4 | 2 | 0.2453 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 5 | 2 | 0.2441 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 6 | 2 | 0.2443 | 0.7500 | 0.0000 | 1.0000 | 1.5000 |
| 7 | 2 | 0.2433 | 0.5000 | 0.2500 | 0.5000 | 2.0000 |
| 8 | 2 | 0.2445 | 1.0000 | 0.5000 | 0.2500 | 2.0000 |
| 9 | 2 | 0.2445 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 10 | 2 | 0.2463 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 11 | 2 | 0.2444 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 12 | 2 | 0.2447 | 1.0000 | 0.7500 | 0.0000 | 1.5000 |
| 13 | 2 | 0.2450 | 1.0000 | 0.7500 | 0.0000 | 1.0000 |
| 14 | 2 | 0.2448 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 15 | 2 | 0.2133 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 16 | 2 | 0.2284 | 0.5000 | 0.3750 | 0.5000 | 1.0000 |

## Failure Reasons

| Reason | Count |
|---|---:|
| weak_claim_support | 9 |
| unsupported_citation | 9 |
| no_citation | 6 |
| invalid_or_fake_url | 3 |

## Failure Samples

- step=16 reward=0.2130 reasons=no_citation
  query: I've seen crabs living under water and also living out of water in the sand on the beach .How can they breath in both environments?
  details: `{"total": 0.12297916666666667, "answer_subem": 0.0, "format": 0.8, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=15 reward=0.2133 reasons=no_citation
  query: What are the defining mechanistic features and electronic considerations of the Burgess reagent-mediated dehydration of secondary and tertiary alcohols, and how do structural variations in alcohol substrates, solvent effects, and reagent stoichiometry influence reaction selectivity, byproduct formation, and synthetic utility in the preparation of alkenes and heterocycles?
  details: `{"total": 0.1232625, "answer_subem": 0.0, "format": 0.8, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=15 reward=0.2133 reasons=no_citation
  query: What are the defining mechanistic features and electronic considerations of the Burgess reagent-mediated dehydration of secondary and tertiary alcohols, and how do structural variations in alcohol substrates, solvent effects, and reagent stoichiometry influence reaction selectivity, byproduct formation, and synthetic utility in the preparation of alkenes and heterocycles?
  details: `{"total": 0.1232625, "answer_subem": 0.0, "format": 0.8, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=7 reward=0.2431 reasons=invalid_or_fake_url,weak_claim_support,unsupported_citation
  query: What is the ozone hole and why does it matter that it's so small?
  details: `{"total": 0.2755958333333334, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.5, "citation_precision": 0.25, "claim_support": 0.25, "unsupported_citation_rate": 0.5, "fake_url_rate": 0.5, "citation_count": 2.0}`
  answer: The ozone hole, a region of depleted ozone in the stratosphere over Antarctica, has been relatively small and short-lived in recent years, particularly in 2025, marking a positive sign of recovery [small and short-lived 2025 ozone hole](https://wmo.int/media/news/small-and-short-lived-2025-ozone-hole-confirms-long-term-recovery-trend). This trend is attributed to the success of the Montreal Protocol, which has led to a significant reduction in ozone-depleting substances [NOAA and NASA 2025 ozone hole](https://www.noaa.gov/news-release/noaa-nasa-2025-ozone-hole-is-5th-smallest-since-1992). The shrinking ozone hole is crucial because it helps protect life on Earth from harmful ultraviolet radiation, which can cause skin cancer, cataracts, and damage to ecosystems.
- step=9 reward=0.2432 reasons=no_citation
  query: Who is Marianne Weber?
  details: `{"total": 0.14316250000000003, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: and
- step=7 reward=0.2435 reasons=invalid_or_fake_url,weak_claim_support,unsupported_citation
  query: What is the ozone hole and why does it matter that it's so small?
  details: `{"total": 0.27603333333333335, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.5, "citation_precision": 0.25, "claim_support": 0.25, "unsupported_citation_rate": 0.5, "fake_url_rate": 0.5, "citation_count": 2.0}`
  answer: The ozone hole is a seasonal depletion of ozone in the stratosphere over Antarctica, primarily caused by ozone-depleting chemicals like CFCs. In 2025, it was the fifth smallest since 1992, a positive sign of recovery due to the Montreal Protocol [small and short-lived 2025 ozone hole](https://wmo.int/media/news/small-and-short-lived-2025-ozone-hole-confirms-long-term-recovery-trend). This recovery is crucial because the ozone layer protects Earth from harmful ultraviolet radiation, which can cause skin cancer, cataracts, and harm ecosystems [NOAA and NASA 2025 ozone hole](https://www.noaa.gov/news-release/noaa-nasa-2025-ozone-hole-is-5th-smallest-since-1992).
- step=8 reward=0.2436 reasons=weak_claim_support,unsupported_citation
  query: How did Ryan Ellis progress through the Nashville Predators organization from the end of the 2010–11 OHL season until the start of the 2012–13 NHL season, including his AHL and NHL performances and notable achievements?
  details: `{"total": 0.2760541666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.25, "claim_support": 0.25, "unsupported_citation_rate": 0.5, "fake_url_rate": 0.0, "citation_count": 2.0}`
  answer: Ryan Ellis was selected 11th overall by the Nashville Predators in the 2009 NHL entry draft after a successful junior career in the Ontario Hockey League (OHL). He began his professional career with the Predators' AHL affiliate, the Milwaukee Admirals, where he played from 2011 to 2012, showcasing his defensive skills and leadership [Ryan Ellis progression](https://en.wikipedia.org/wiki/Ryan_Ellis). In the 2012–13 NHL season, Ellis made his debut with the Predators, contributing to their playoff run and establishing himself as a key defenseman [NHL debut](https://en.wikipedia.org/wiki/Ryan_Ellis).
- step=6 reward=0.2442 reasons=invalid_or_fake_url,weak_claim_support,unsupported_citation
  query: What is the experimental metal band "Kayo Dot"?
  details: `{"total": 0.14418750000000002, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.5, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.5, "citation_count": 2.0}`
  answer: Kayo Dot is an American avant-garde metal band formed in 2002 by Toby Driver after the breakup of Maudlin of the Well. They released their debut album, *Choirs of the Eye*, on John Zorn's Tzadik Records in the same year. The band is known for its experimental and progressive sound, with their latest album, *Reason*, celebrating the 20th anniversary of their debut and featuring the original lineup [1](https://en.wikipedia.org/wiki/Kayo_Dot), [2](https://www.rollingstone.com/music/music-news/kayo-dot-reason-album-1148552/).
- step=11 reward=0.2444 reasons=no_citation
  query: How is there a "heroin epidemic", if some news sources say that drug use is at the lowest it's been in years?
  details: `{"total": 0.1443666666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The term "heroin epidemic" refers to the significant rise in heroin-related deaths and overdoses in the early 2010s, which peaked around 2017. However, recent data shows a decline in overall drug use, particularly among younger populations, with the rate of drug use among people over 40 increasing faster than younger age groups. This suggests that while the heroin epidemic may have peaked, the broader issue of drug use and addiction remains a concern, especially among older adults.
- step=14 reward=0.2444 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.1443791666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s evolved significantly throughout his career. Initially with the Pirates, he was a versatile player, primarily as a catcher, but his role shifted to a part-time player and eventually to a utility infielder as he aged. After the Pirates' move to Houston in 1962, he continued to adapt, playing both as a catcher and an infielder, though his playing time decreased as he approached retirement [Jim Umbricht Stats, Height, Weight, Research & History - Baseball Almanac](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=6 reward=0.2444 reasons=weak_claim_support,unsupported_citation
  query: What is the experimental metal band "Kayo Dot"?
  details: `{"total": 0.14443750000000002, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Kayo Dot is an American avant-garde metal band formed in 2002 by Toby Driver after the breakup of Maudlin of the Well. They released their debut album, *Choirs of the Eye*, on John Zorn's Tzadik Records in the same year. Their latest album, *Reason*, marks the 20th anniversary of their debut and reunites the original lineup, showcasing a blend of their early compositional practices and new musical directions [seminal debut](https://en.wikipedia.org/wiki/Kayo_Dot).
- step=11 reward=0.2445 reasons=no_citation
  query: How is there a "heroin epidemic", if some news sources say that drug use is at the lowest it's been in years?
  details: `{"total": 0.14452083333333335, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The term "heroin epidemic" refers to the significant rise in heroin use and related deaths in the early 2000s, which peaked around 2017. However, recent data shows that drug use, including heroin, has been declining in recent years, with overall substance abuse rates reaching their lowest levels in decades. This discrepancy highlights the cyclical nature of drug epidemics and the importance of ongoing public health efforts to address substance abuse.
