# DeepFactCite v3 Prompt-Fix GRPO Rollout Gallery (2026-05-18)

这份文档把 `dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu` 的 32 条 rollout 做成人能读的样例集。它不是新的实验结果，而是把已经落盘的 JSONL rollout、reward 细项和证据片段整理出来，方便复盘模型到底学会了什么、还错在哪里。

## Files

- Raw rollout: `logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu/rollout_data_step_*.jsonl`
- Metric summary: `reports/dfc_mixclean200_v3_promptfix_onecite_2gpu_rollout_summary.md`
- Curated machine-readable samples: `reports/deepfactcite_v3_promptfix_rollout_gallery_samples_20260518.jsonl`

## Aggregate View

| Metric | Mean | How to read |
|---|---:|---|
| `reward` | 0.5120 | training reward after current DeepFactCite weights |
| `total` | 0.4807 | unshaped reward components before extra shaping term |
| `format` | 0.9875 | whether output follows search + answer format |
| `search` | 0.9375 | whether the model actually called search |
| `url_validity` | 0.9375 | whether cited URL appears in retrieved evidence |
| `citation_precision` | 0.6312 | whether cited claims align with evidence at citation level |
| `claim_support` | 0.6250 | whether the answer claim is supported by the cited snippet |
| `unsupported_citation_rate` | 0.1250 | lower is better; citations that do not support the attached claim |
| `fake_url_rate` | 0.0000 | lower is better; cited URLs not copied from retrieval |
| `citation_count` | 0.9375 | whether there is at least one markdown citation |

Gallery issue counts:

These counts are intentionally stricter than the compact rollout summary: every `claim_support=0.5` sample is tagged as `weak_claim_support`, because this gallery is meant for error analysis rather than headline reporting. The generated answers are preserved verbatim, including spacing artifacts from the model output.

| Reason | Count |
|---|---:|
| `weak_claim_support` | 12 |
| `unsupported_citation` | 4 |
| `no_search` | 2 |
| `no_citation` | 2 |

Citation label audit:

| Label type | Count | Note |
|---|---:|---|
| `S_xxx_label` | 22 | technically valid markdown links, but poor human-facing labels |
| `numeric_label` | 1 | valid URL but discouraged by the prompt |
| `human_or_symbol_label` | 7 | closer to readable citation labels |

## What Improved

- Compared with the earlier v2 one-citation run, v3 primarily fixed instruction following around citation syntax: no-citation failures dropped from 9/32 to 2/32, URL validity rose to 0.9375, and unsupported citation rate fell to 0.1250.
- The result is not a final model checkpoint. This was a 16-step, 32-sample diagnostic run. It proves the data/prompt/reward direction is worth scaling, while still exposing failure modes before spending 4-card training time.
- The biggest remaining issue is not fake URLs. It is claim granularity: the model still sometimes writes a broader explanatory claim than the retrieved snippet strictly supports.

## Full-Support Examples

这些样例的 URL 来自检索结果，并且 `claim_support=1.0`。它们说明 v3 prompt + markdown parser + citation-aware reward 已经能产生可用的 citation behavior。

### 1. step=3 line=2 reward=0.7414

**Question:** What is the indie game "Hyper Light Drifter"?

**Answer:** HyperLightDrifteris an action role-playinggamedeveloped and published by Heart Machine.[S_cd9490c8](https://en.wikipedia.org/wiki/Hyper_Light_Drifter)

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 1.0000 |
| `claim_support` | 1.0000 |
| `unsupported_citation_rate` | 0.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Citations used:**

- label=`S_cd9490c8` URL=`https://en.wikipedia.org/wiki/Hyper_Light_Drifter`

**Retrieved evidence excerpt:**

- `S_cd9490c8` HyperLightDrifter- Wikipedia / `https://en.wikipedia.org/wiki/Hyper_Light_Drifter`
  - excerpt: HyperLightDrifteris an action role-playinggamedeveloped and published by Heart Machine. The Microsoft Windows, Linux and OS X versions were released in March 2016, and the PlayStation 4 and Xbox One versions in July 2016.

### 2. step=9 line=2 reward=0.7411

**Question:** What is density

**Answer:** Density is the ratio of a substance's mass to its volume, symbolized by ρ [ρ](https://en.wikipedia.org/wiki/Density).

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 1.0000 |
| `claim_support` | 1.0000 |
| `unsupported_citation_rate` | 0.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Citations used:**

- label=`ρ` URL=`https://en.wikipedia.org/wiki/Density`

**Retrieved evidence excerpt:**

- `S_d37fbabc` Density - Wikipedia / `https://en.wikipedia.org/wiki/Density`
  - excerpt: Density(volumetric massdensityor specific mass) is the ratio of a substance's mass to its volume. The symbol most often used fordensityis ρ (the lower case Greek letter rho), although the Latin letter D (or d) can also be used: [1] where ρ is thedensity, m...

### 3. step=12 line=2 reward=0.7410

**Question:** Tell me about the John Radcliffe Hospital.

**Answer:** The John Radcliffe Hospital is a part of the Oxford University Hospitals NHS Foundation Trust, named after John Radcliffe, who endowed the Radcliffe Infirmary in 1770 [John Radcliffe Hospital](https://en.wikipedia.org/wiki/John_Radcliffe_Hospital).

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 1.0000 |
| `claim_support` | 1.0000 |
| `unsupported_citation_rate` | 0.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Citations used:**

- label=`John Radcliffe Hospital` URL=`https://en.wikipedia.org/wiki/John_Radcliffe_Hospital`

**Retrieved evidence excerpt:**

- `S_c6351a24` John Radcliffe Hospital - Wikipedia / `https://en.wikipedia.org/wiki/John_Radcliffe_Hospital`
  - excerpt: It is a part of the Oxford University Hospitals NHS Foundation Trust and is named after JohnRadcliffe, an 18th-century physician and Oxford University graduate, who endowed the Radcliffe Infirmary, the main hospital for Oxford from 1770 until 2007.

### 4. step=5 line=1 reward=0.7409

**Question:** What is the Chauvet-Pont d'Arc Cave?

**Answer:** TheChauvet-Pont-d'ArcCaveis acavethat contains some of the best-preserved figurativecavepaintings in the world, [1](https://en.wikipedia.org/wiki/Chauvet_Cave)

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 1.0000 |
| `claim_support` | 1.0000 |
| `unsupported_citation_rate` | 0.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Citations used:**

- label=`1` URL=`https://en.wikipedia.org/wiki/Chauvet_Cave`

**Retrieved evidence excerpt:**

- `S_c52fa014` Chauvet Cave - Wikipedia / `https://en.wikipedia.org/wiki/Chauvet_Cave`
  - excerpt: TheChauvet-Pont-d'ArcCave(French: GrotteChauvet-Pontd'Arc[ɡʁɔt ʃovɛ pɔ̃ daʁk]) in the Ardèche department of southeastern France is acavethat contains some of the best-preserved figurativecavepaintings in the world, [1] as well as other evidence of Upper Pal...

### 5. step=13 line=1 reward=0.7407

**Question:** Who is Marianne Weber?

**Answer:** MarianneWeberwas born 2 August 1870, in Oerlinghausen, Germany, to medical doctor Eduard Schnitger and his wife AnnaWeber [S_a89ec78f](https://en.wikipedia.org/wiki/Marianne_Weber).

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 1.0000 |
| `claim_support` | 1.0000 |
| `unsupported_citation_rate` | 0.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Citations used:**

- label=`S_a89ec78f` URL=`https://en.wikipedia.org/wiki/Marianne_Weber`

**Retrieved evidence excerpt:**

- `S_a89ec78f` Marianne Weber - Wikipedia / `https://en.wikipedia.org/wiki/Marianne_Weber`
  - excerpt: MarianneSchnitger was born 2 August 1870, in Oerlinghausen, Germany, to medical doctor, Eduard Schnitger, and his wife, AnnaWeber. Her mother was the daughter of a prominent Oerlinghausen businessman, KarlWeber. [1] Much ofMarianne'schildhood was characteri...

### 6. step=15 line=1 reward=0.7406

**Question:** Tell me about the Hugging Face Hub in machine learning.

**Answer:** the Hugging Face Model Hub is an open-source platform for uploading, sharing, and downloading pre-trained machine learning models [S_8d8e746c](https://mljourney.com/how-does-the-hugging-face-model-hub-work)

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 1.0000 |
| `claim_support` | 1.0000 |
| `unsupported_citation_rate` | 0.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Citations used:**

- label=`S_8d8e746c` URL=`https://mljourney.com/how-does-the-hugging-face-model-hub-work`

**Retrieved evidence excerpt:**

- `S_8d8e746c` How Does the Hugging Face Model Hub Work? - ML Journey / `https://mljourney.com/how-does-the-hugging-face-model-hub-work`
  - excerpt: Mar 19, 2025 ·The Hugging Face Model Hub is anopen-source platform where users can upload, share, and download pre-trained machine learning models. It serves as a central repository for AI practitioners who want to reuse existing models instead of training...

## Partial-Support Examples

这些样例有正确 URL 和 citation，但 `claim_support=0.5`。常见问题是答案多写了一个证据没有直接覆盖的解释、趋势或结论。

### 1. step=2 line=1 reward=0.4368

**Question:** How is there a "heroin epidemic", if some news sources say that drug use is at the lowest it's been in years?

**Answer:** Drug-related deaths among older adults have increased significantly, with a 11.4% rise in the 65 and older age group between 2022 and 2023, despite overall drug use being at its lowest in years [S_4e37067b](https://drugabusestatistics.org).

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 0.5000 |
| `claim_support` | 0.5000 |
| `unsupported_citation_rate` | 0.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Failure tags:** `weak_claim_support`

**Citations used:**

- label=`S_4e37067b` URL=`https://drugabusestatistics.org`

**Retrieved evidence excerpt:**

- `S_4e37067b` NCDAS: Substance Abuse and Addiction Statistics [2025] / `https://drugabusestatistics.org`
  - excerpt: While younger people are more likely to usedrugs, the rate ofdruguseamong people over 40 is increasing faster than it is among younger age groups. Between 2022 and 2023, thedrugrelated death rate increased by 2.3% for adults aged 55-64 and by 11.4% for adul...

### 2. step=11 line=1 reward=0.4365

**Question:** How does marriage equality undermine the US constitution?

**Answer:** Marriage equality does not undermine the US Constitution; instead, it upholds the principles of due process and equal protection as established in Obergefell v. Hodges [Obergefell v. Hodges](https://en.wikipedia.org/wiki/Obergefell_v._Hodges).

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 0.5000 |
| `claim_support` | 0.5000 |
| `unsupported_citation_rate` | 0.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Failure tags:** `weak_claim_support`

**Citations used:**

- label=`Obergefell v. Hodges` URL=`https://en.wikipedia.org/wiki/Obergefell_v._Hodges`

**Retrieved evidence excerpt:**

- `S_c000953b` Obergefell v. Hodges - Wikipedia / `https://en.wikipedia.org/wiki/Obergefell_v._Hodges`
  - excerpt: Obergefell v. Hodges, 576U.S. 644 (2015) (/ ˈoʊbərɡəfɛl / OH-bər-gə-fel), is a landmark decision of the United States Supreme Court which ruled that the fundamental right to marry is guaranteed to same-sex couples by both the Due Process Clause and the Equa...

### 3. step=7 line=1 reward=0.4362

**Question:** What is the ozone hole and why does it matter that it's so small?

**Answer:** The ozone hole is a seasonal depletion of ozone in the stratosphere over Antarctica, and its small size in 2025 is significant because it indicates progress in reducing ozone-depleting substances, thanks to the Montreal Protocol.[S_a563c239](https://www.noaa.gov/news-release/noaa-nasa-2025-ozone-hole-is-5th-smallest-since-1992)

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 0.5000 |
| `claim_support` | 0.5000 |
| `unsupported_citation_rate` | 0.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Failure tags:** `weak_claim_support`

**Citations used:**

- label=`S_a563c239` URL=`https://www.noaa.gov/news-release/noaa-nasa-2025-ozone-hole-is-5th-smallest-since-1992`

**Retrieved evidence excerpt:**

- `S_a563c239` NOAA, NASA: 2025 ozone hole is 5th smallest since 1992 / `https://www.noaa.gov/news-release/noaa-nasa-2025-ozone-hole-is-5th-smallest-since-1992`
  - excerpt: Nov 24, 2025 ·Scientists with NOAA and NASA have ranked this year’sozoneholeover the Antarctic as the fifth smallest since 1992 — the year that the Montreal Protocol, a landmark international agreement to phase outozone-depleting chemicals began to take eff...

### 4. step=4 line=1 reward=0.4361

**Question:** What happens to my $15/hr job if the minimum wage is increased to $15?

**Answer:** Increasing the minimum wage to $15 could lead to job losses for workers earning $15/hr, as employers may reduce hiring or hours to offset higher labor costs [S_71879d83](https://www.minneapolisfed.org/article/2023/better-together-pairing-smaller-minimum-wage-increases-with-tax-policy-to-reduce-inequality).

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 0.5000 |
| `claim_support` | 0.5000 |
| `unsupported_citation_rate` | 0.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Failure tags:** `weak_claim_support`

**Citations used:**

- label=`S_71879d83` URL=`https://www.minneapolisfed.org/article/2023/better-together-pairing-smaller-minimum-wage-increases-with-tax-policy-to-reduce-inequality`

**Retrieved evidence excerpt:**

- `S_71879d83` Better together: Pairing smaller minimum wage increases with ... / `https://www.minneapolisfed.org/article/2023/better-together-pairing-smaller-minimum-wage-increases-with-tax-policy-to-reduce-inequality`
  - excerpt: Dec 12, 2023 ·Specifically, the economists find thatincreasingtheminimumwageto around $9.25 is enough to effectively erode the labor-market power of employers, thus raising pay andincreasingjobs, especially for workers earning low wages.

### 5. step=10 line=2 reward=0.4359

**Question:** Why are witches of the occult often portrayed as green-skinned?

**Answer:** Witches in the occult are often portrayed as green-skinned due to their association with nature and herbal knowledge in pre-modern European folklore [S_8c7b4bf3](https://www.alibaba.com/product-insights/why-are-witches-green-origins-of-the-green-skinned-witch.html).

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 0.5000 |
| `claim_support` | 0.5000 |
| `unsupported_citation_rate` | 0.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Failure tags:** `weak_claim_support`

**Citations used:**

- label=`S_8c7b4bf3` URL=`https://www.alibaba.com/product-insights/why-are-witches-green-origins-of-the-green-skinned-witch.html`

**Retrieved evidence excerpt:**

- `S_8c7b4bf3` Why Are Witches Green Origins Of The Green Skinned Witch / `https://www.alibaba.com/product-insights/why-are-witches-green-origins-of-the-green-skinned-witch.html`
  - excerpt: The Witch BeforeGreen: Folklore Roots In pre-modern European traditions,witcheswere not inherentlygreen. Instead, they wereoftenportrayedasold women living on society's fringes—knowledgeable in herbs, midwifery, and folk remedies. Their appearance varied by...

## Failure Examples

这些样例不能忽略，因为它们决定下一轮扩大训练时应该盯哪些 guardrail。

### 1. step=1 line=2 reward=0.1003

**Question:** What is the Japanese avant-garde band "Boredoms"?

**Answer:** None

| detail | value |
|---|---:|
| `format` | 0.8000 |
| `search` | 0.0000 |
| `url_validity` | 0.0000 |
| `citation_precision` | 0.1000 |
| `claim_support` | 0.0000 |
| `unsupported_citation_rate` | 0.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 0.0000 |

**Failure tags:** `no_search, no_citation`

### 2. step=5 line=2 reward=0.1003

**Question:** What is the Chauvet-Pont d'Arc Cave?

**Answer:** None

| detail | value |
|---|---:|
| `format` | 0.8000 |
| `search` | 0.0000 |
| `url_validity` | 0.0000 |
| `citation_precision` | 0.1000 |
| `claim_support` | 0.0000 |
| `unsupported_citation_rate` | 0.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 0.0000 |

**Failure tags:** `no_search, no_citation`

### 3. step=6 line=1 reward=0.1450

**Question:** What's the point in climate change policies if we're screwed already?

**Answer:** Climate policies are crucial for mitigating the impacts of climate change, even if the situation seems dire, as they can prevent further damage and protect vulnerable populations [S_16db65fc](https://www.pewresearch.org/science/2024/12/09/how-americans-view-climate-change-and-policies-to-address-the-issue).

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 0.0000 |
| `claim_support` | 0.0000 |
| `unsupported_citation_rate` | 1.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Failure tags:** `unsupported_citation`

**Citations used:**

- label=`S_16db65fc` URL=`https://www.pewresearch.org/science/2024/12/09/how-americans-view-climate-change-and-policies-to-address-the-issue`

**Retrieved evidence excerpt:**

- `S_16db65fc` How Americans View Climate Change and Policies in 20249. CLIMATE CHANGE POLICIES AND GLOBAL AGREEMENTSClimate policy strengthens globally, despite unprecedented / `https://www.pewresearch.org/science/2024/12/09/how-americans-view-climate-change-and-policies-to-address-the-issue`
  - excerpt: Dec 9, 2024 ·A Pew Research Center survey of 9,5...(truncated).... These agreements involve international cooperation, binding targets, and innovativepolicymeasures designed to reduce greenhouse gas (GHG) emissions, promote sustainable development, and prot...

### 4. step=6 line=2 reward=0.1453

**Question:** What's the point in climate change policies if we're screwed already?

**Answer:** Climate change policies are still crucial for mitigating future impacts and protecting vulnerable populations, even if the situation is dire [S_142e6eff](https://www.pewresearch.org/science/2024/12/09/how-americans-view-climate-change-and-policies-to-address-the-issue).

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 0.0000 |
| `claim_support` | 0.0000 |
| `unsupported_citation_rate` | 1.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Failure tags:** `unsupported_citation`

**Citations used:**

- label=`S_142e6eff` URL=`https://www.pewresearch.org/science/2024/12/09/how-americans-view-climate-change-and-policies-to-address-the-issue`

**Retrieved evidence excerpt:**

- `S_142e6eff` How Americans View Climate Change and Policies in 20249. CLIMATE CHANGE POLICIES AND GLOBAL AGREEMENTSClimate policy strengthens globally, despite unprecedented / `https://www.pewresearch.org/science/2024/12/09/how-americans-view-climate-change-and-policies-to-address-the-issue`
  - excerpt: Dec 9, 2024 ·A Pew Research Center survey of 9,5...(truncated).... These agreements involve international cooperation, binding targets, and innovativepolicymeasures designed to reduce greenhouse gas (GHG) emissions, promote sustainable development, and prot...

### 5. step=14 line=1 reward=0.1459

**Question:** How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?

**Answer:** Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s evolved from a starting pitcher to a reliever, with his career spanning from 1959 to 1966, including stints with both teams [S_4a62ea91](https://www.baseball-almanac.com/players/player.php?p=umbriji01).

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 0.0000 |
| `claim_support` | 0.0000 |
| `unsupported_citation_rate` | 1.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Failure tags:** `unsupported_citation`

**Citations used:**

- label=`S_4a62ea91` URL=`https://www.baseball-almanac.com/players/player.php?p=umbriji01`

**Retrieved evidence excerpt:**

- `S_4a62ea91` Jim Umbricht Stats, Height, Weight, Research & History - Baseball Almanac / `https://www.baseball-almanac.com/players/player.php?p=umbriji01`
  - excerpt: JimUmbrichtBaseball Stats | Baseball AlmanacJimUmbricht, the Major League Baseball player, was born on Wednesday, September 17, 1930, in Chicago, Illinois.Umbrichtwas 29 years old when he broke into the major leagues on September 26, 1959, with thePittsburg...

### 6. step=14 line=2 reward=0.1461

**Question:** How did Jim Umbricht's performance and role with the Pittsburgh Pirates and Houston Colt .45s change throughout his Major League Baseball career?

**Answer:** Jim Umbricht's career with the Pittsburgh Pirates and Houston Colt .45s saw him transition from a starting pitcher to a relief pitcher, with his final season in 1965 [S_40cb031b](https://www.baseball-almanac.com/players/player.php?p=umbriji01).

| detail | value |
|---|---:|
| `format` | 1.0000 |
| `search` | 1.0000 |
| `url_validity` | 1.0000 |
| `citation_precision` | 0.0000 |
| `claim_support` | 0.0000 |
| `unsupported_citation_rate` | 1.0000 |
| `fake_url_rate` | 0.0000 |
| `citation_count` | 1.0000 |

**Failure tags:** `unsupported_citation`

**Citations used:**

- label=`S_40cb031b` URL=`https://www.baseball-almanac.com/players/player.php?p=umbriji01`

**Retrieved evidence excerpt:**

- `S_40cb031b` Jim Umbricht Stats, Height, Weight, Research & History - Baseball Almanac / `https://www.baseball-almanac.com/players/player.php?p=umbriji01`
  - excerpt: JimUmbrichtBaseball Stats | Baseball AlmanacJimUmbricht, the Major League Baseball player, was born on Wednesday, September 17, 1930, in Chicago, Illinois.Umbrichtwas 29 years old when he broke into the major leagues on September 26, 1959, with thePittsburg...

## Lessons For The Next 4-Card Run

1. 4 卡只适合做 saved medium run，不适合盲目长跑。v3 已经过了扩大门槛，但还要保持短周期保存和可回滚。
2. 继续保留 `DFC_REQUIRE_MARKDOWN_CITATION=true`，因为之前 parser/reward 没严格约束 markdown citation 时，模型容易给出裸 `[S_xxx]` 或不出 citation。
3. 下一轮要新增 label-quality 观测项。当前 reward 认为 `[S_xxx](URL)` 是合法 markdown citation，但给用户看的 citation label 仍不够好。
4. 对 partial/failure 样例，优先改数据和 prompt 中的“claim 只能写证据直接支持的窄事实”，再考虑加大训练步数。
5. 每次训练后都要重新跑这个 gallery 抽样流程，不能只看 reward 均值。
