# GRPO Rollout Summary: `logs/grpo/rollouts/dfc-mixclean200-claimfiltered-citation-aware-20260518_claimfiltered_2gpu`

- Samples: 32
- Steps: 16

## Aggregate Metrics

| Metric | Mean |
|---|---:|
| reward | 0.3731 |
| total | 0.3557 |
| answer_subem | 0.0312 |
| format | 0.9750 |
| search | 1.0000 |
| url_validity | 0.6979 |
| citation_precision | 0.3828 |
| claim_support | 0.3828 |
| unsupported_citation_rate | 0.4844 |
| fake_url_rate | 0.0521 |
| citation_count | 1.0000 |

## Step Metrics

| Step | Samples | Reward | URL | Support | Unsupported | Citations |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 0.7409 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 2 | 2 | 0.5146 | 1.0000 | 0.6250 | 0.0000 | 2.0000 |
| 3 | 2 | 0.7398 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 4 | 2 | 0.2901 | 1.0000 | 0.2500 | 0.5000 | 1.5000 |
| 5 | 2 | 0.5866 | 1.0000 | 0.7500 | 0.0000 | 1.0000 |
| 6 | 2 | 0.1244 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 7 | 2 | 0.4330 | 1.0000 | 0.5000 | 0.0000 | 1.0000 |
| 8 | 2 | 0.1333 | 0.5000 | 0.0000 | 1.0000 | 0.5000 |
| 9 | 2 | 0.1451 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| 10 | 2 | 0.2907 | 1.0000 | 0.2500 | 0.5000 | 1.0000 |
| 11 | 2 | 0.2185 | 0.1667 | 0.0000 | 1.0000 | 2.0000 |
| 12 | 2 | 0.2895 | 0.5000 | 0.2500 | 0.5000 | 0.5000 |
| 13 | 2 | 0.4414 | 1.0000 | 0.5000 | 0.2500 | 1.5000 |
| 14 | 2 | 0.1439 | 1.0000 | 0.0000 | 1.0000 | 1.5000 |
| 15 | 2 | 0.4410 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 16 | 2 | 0.4369 | 0.5000 | 0.5000 | 0.5000 | 1.0000 |

## Failure Reasons

| Reason | Count |
|---|---:|
| unsupported_citation | 8 |
| no_citation | 8 |
| weak_claim_support | 7 |
| invalid_or_fake_url | 2 |

## Failure Samples

- step=16 reward=0.1230 reasons=no_citation
  query: I've seen crabs living under water and also living out of water in the sand on the beach .How can they breath in both environments?
  details: `{"total": 0.12297916666666667, "answer_subem": 0.0, "format": 0.8, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=8 reward=0.1235 reasons=no_citation
  query: How did Ryan Ellis progress through the Nashville Predators organization from the end of the 2010–11 OHL season until the start of the 2012–13 NHL season, including his AHL and NHL performances and notable achievements?
  details: `{"total": 0.1234875, "answer_subem": 0.0, "format": 0.8, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=6 reward=0.1243 reasons=no_citation
  query: What is the experimental metal band "Kayo Dot"?
  details: `{"total": 0.12433333333333334, "answer_subem": 0.0, "format": 0.8, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=6 reward=0.1244 reasons=no_citation
  query: What is the experimental metal band "Kayo Dot"?
  details: `{"total": 0.12440416666666668, "answer_subem": 0.0, "format": 0.8, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: None
- step=8 reward=0.1431 reasons=weak_claim_support,unsupported_citation
  query: How did Ryan Ellis progress through the Nashville Predators organization from the end of the 2010–11 OHL season until the start of the 2012–13 NHL season, including his AHL and NHL performances and notable achievements?
  details: `{"total": 0.1430791666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Ryan Ellis was selected 11th overall by the Nashville Predators in the 2009 NHL entry draft after a successful junior career in the Ontario Hockey League (OHL). He began his professional career with the Milwaukee Admirals of the American Hockey League (AHL) in the 2010–11 season, where he played 41 games and recorded 11 goals and 21 assists. Ellis then joined the Predators for the 2011–12 season, making his NHL debut and contributing 12 goals and 25 assists in 73 games. By the start of the 2012–13 season, Ellis had established himself as a key defenseman for the Predators, earning recognition for his defensive skills and leadership on the ice [Ryan Ellis progression](https://en.wikipedia.org/wiki/Ryan_Ellis).
- step=14 reward=0.1433 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.1433, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 2.0}`
  answer: Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s evolved significantly throughout his career. Initially, he played for the Pittsburgh Pirates, where he was a versatile player, primarily as a catcher and first baseman, from 1959 to 1960 [Jim Umbricht Stats, Height, Weight, Research & History - Baseball Almanac](https://www.baseball-almanac.com/players/player.php?p=umbriji01). After the team was relocated to Houston and renamed the Colt .45s in 1962, Umbricht continued to play as a catcher, though his role became more specialized, focusing on defensive duties and occasional pinch-hitting [Jim Umbricht Stats, Height, Weight, Research & History - Baseball Almanac](https://www.baseball-almanac.com/players/player.php?p=umbriji01). His career spanned from 1959 to 1966, with a notable decline in offensive production in his later years.
- step=11 reward=0.1435 reasons=invalid_or_fake_url,weak_claim_support,unsupported_citation
  query: How is there a "heroin epidemic", if some news sources say that drug use is at the lowest it's been in years?
  details: `{"total": 0.14354583333333334, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.3333333333333333, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.6666666666666667, "citation_count": 3.0}`
  answer: The "heroin epidemic" refers to the significant rise in heroin-related deaths and addiction rates in the early 2000s, particularly in the United States, which peaked in the mid-2010s [heroin epidemic timeline](https://www.cdc.gov/drugoverdose/data/epidemic.html). While overall drug use has decreased in recent years, the impact of the epidemic is still evident in the lingering effects of addiction and overdose rates [drug use trends](https://www.addictiongroup.org/resources/addiction-statistics). The increase in drug-related deaths among older adults also highlights the ongoing challenges posed by the epidemic [aging population and drug use](https://drugabusestatistics.org).
- step=14 reward=0.1444 reasons=weak_claim_support,unsupported_citation
  query: How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?
  details: `{"total": 0.14444583333333336, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 1.0}`
  answer: Jim Umbricht's career with the Pittsburgh Pirates and Houston Colt .45s (now the Houston Astros) saw him transition from a promising rookie to a solid utility player. He debuted with the Pirates in 1959 and played primarily as a catcher and outfielder, contributing to the team's success during his time there. After the Pirates' move to the National League, he joined the Colt .45s in 1962, where he continued to play a versatile role, often filling in as a backup catcher and outfielder [Jim Umbricht Stats, Height, Weight, Research & History - Baseball Almanac](https://www.baseball-almanac.com/players/player.php?p=umbriji01).
- step=9 reward=0.1447 reasons=no_citation
  query: Who is Marianne Weber?
  details: `{"total": 0.1446791666666667, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Marianne Weber was born on August 2, 1870, in Oerlinghausen, Germany, to Eduard Schnitger, a medical doctor, and Anna Weber, the daughter of a prominent businessman. [1] Her early life was marked by poverty and hardship, particularly after her mother's death in 1873, which led her to be raised by her grandmother and aunt in Lemgo. [1]
- step=12 reward=0.1448 reasons=no_citation
  query: What is the Chauvet-Pont d'Arc Cave?
  details: `{"total": 0.14477916666666668, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: The Chauvet-Pont d'Arc Cave, located near Vallon-Pont-d'Arc in southeastern France, is one of the oldest decorated Paleolithic caves, containing some of the best-preserved figurative cave paintings in the world. [GrotteChauvet-Pontd'Arc] Discovered in 1994 by Jean-Marie Chauvet, Éliette Brunel, and Christian Hillaire, the cave provides significant evidence of Upper Paleolithic life and art. [The Discovery]
- step=4 reward=0.1449 reasons=weak_claim_support,unsupported_citation
  query: What happens to my $15/hr job if the minimum wage is increased to $15?
  details: `{"total": 0.14494583333333336, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 1.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 2.0}`
  answer: If the minimum wage is increased to $15, your $15/hr job would likely remain unchanged, as your current wage already meets the new minimum [S_90c5bf9d](https://www.minneapolisfed.org/article/2023/better-together-pairing-smaller-minimum-wage-increases-with-tax-policy-to-reduce-inequality). However, the broader economic impact of such an increase could vary, potentially affecting job availability and wages for lower-paid workers [S_90c5bf9d](https://www.minneapolisfed.org/article/2023/better-together-pairing-smaller-minimum-wage-increases-with-tax-policy-to-reduce-inequality).
- step=15 reward=0.1452 reasons=no_citation
  query: What are the defining mechanistic features and electronic considerations of the Burgess reagent-mediated dehydration of secondary and tertiary alcohols, and how do structural variations in alcohol substrates, solvent effects, and reagent stoichiometry influence reaction selectivity, byproduct formation, and synthetic utility in the preparation of alkenes and heterocycles?
  details: `{"total": 0.14518750000000002, "answer_subem": 0.0, "format": 1.0, "search": 1.0, "url_validity": 0.0, "citation_precision": 0.0, "claim_support": 0.0, "unsupported_citation_rate": 1.0, "fake_url_rate": 0.0, "citation_count": 0.0}`
  answer: Cannot determine an answer based on the available information.
