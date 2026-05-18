# DeepFactCite 学习索引

日期：2026-05-18

这是从零理解当前 DeepFactCite/Search-R1 实验的推荐阅读顺序。建议按这个顺序读文件，不要一开始就直接跳到脚本或原始日志。

## 1. 主线故事

从这里开始：

```text
docs/deepfactcite_experiment_record_20260518.md
```

它会讲清楚：

- 这个项目为什么存在；
- 这个仓库里的 SFT 和 GRPO 分别在做什么；
- 为什么 MixClean200 成为当前 SFT 基线；
- 为什么第一阶段 GRPO 的目标不是长跑训练，而是受控对比；
- v1、v2、parser-fix 和 v3 prompt-fix 分别如何改变结果。

入门检查点：

```text
如果你能解释为什么“真实 URL”不等于“被支持的声明”，你就理解了这个项目的核心问题。
```

## 2. 失败台账

接着读：

```text
docs/deepfactcite_reproducibility_issue_log.md
```

它会讲清楚：

- 为什么“训练跑完了”并不自动等于“结果有价值”；
- 一个坏的父 checkpoint 如何污染后续实验；
- 为什么 dynamic LoRA 和 merged checkpoint 必须做一致性检查；
- 如何记录失败尝试，让失败本身也能沉淀项目价值。

入门检查点：

```text
如果你能说清楚为什么 mix50 虽然完成训练却没有被提升为主模型，你就理解了复现纪律。
```

## 3. SFT 基线报告

然后读：

```text
reports/deepfactcite_mixclean200_eval_summary.md
reports/searchr1_core_bm25_eval_summary.md
```

它们会讲清楚：

- Base、Soft100 和 MixClean200 如何对比；
- 为什么 ShortQA32 和 Search-R1 BM25 200 是防护评测；
- 为什么一个引用质量项目仍然需要答案/搜索指标。

关键结果：

```text
MixClean200 是当前 SFT 基线，因为它把 Search-R1 BM25 200 的 subEM 提升到 0.490，同时保持了较强的搜索行为。
```

## 4. GRPO 消融记录

阅读：

```text
reports/deepfactcite_v2_onecite_grpo_ablation_2gpu_20260518.md
```

它会讲清楚：

- 为什么需要 claim-filtered 数据；
- 为什么 outcome-only 和 citation-aware 必须在同一份数据上运行；
- 为什么 v2 有价值但还不够；
- 4 卡训练的晋升门槛是如何推导出来的。

关键结果：

```text
相较 outcome-only reward，citation-aware reward 改善了 URL validity、citation precision、claim support 和 unsupported citation rate，但 v2 仍然有太多 no-citation 失败。
```

## 5. Rollout 样例集

然后查看：

```text
reports/deepfactcite_v3_promptfix_rollout_gallery_20260518.md
reports/deepfactcite_v3_promptfix_rollout_gallery_samples_20260518.jsonl
```

它会讲清楚：

- 如何阅读单条模型轨迹；
- 完全被支持的答案是什么样；
- 部分被支持的答案是什么样；
- 为什么有些输出 URL 是有效的，却仍然在 claim support 上失败；
- 为什么 v3 是一个真实的正向信号，但还不是最终产品级 checkpoint。

入门检查点：

```text
选一个失败样例，解释答案里的哪一个具体短语比被引用片段表达得更宽。那就是声明级归因调试。
```

## 6. 原始指标摘要

把这个文件当作紧凑结果表使用：

```text
reports/dfc_mixclean200_v3_promptfix_onecite_2gpu_rollout_summary.md
```

重要数字：

| Metric | v3 Prompt-Fix |
|---|---:|
| reward | 0.5120 |
| search | 0.9375 |
| URL validity | 0.9375 |
| citation precision | 0.6312 |
| claim support | 0.6250 |
| unsupported citation rate | 0.1250 |
| no citation failures | 2/32 |

使用方式：

```text
用这个文件看核心指标；用 rollout gallery 理解指标为什么变化。
```

## 7. 下一轮运行计划

租 GPU 之前，先读：

```text
reports/deepfactcite_cpu_mode_consistency_check_20260518.md
reports/storage_cleanup_candidates_current_20260518.md
docs/deepfactcite_4gpu_saved_run_plan_20260518.md
```

它会讲清楚：

- v3 的 parquet/jsonl/rollout/report artifact 是否一致；
- 迁移或保存 checkpoint 之前，哪些大目录可以清理；
- 为什么只有 v3 之后才值得上 4 卡；
- 应该运行哪条命令；
- 哪些磁盘和 checkpoint 约束很重要；
- 扩展训练前必须检查哪些指标。

## 8. 代码入口

读完上面的文档后，再读代码：

```text
scripts/deepfactcite/prepare_sglang_grpo_claim_filtered.py
scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
scripts/deepfactcite/verl_deepfactcite_reward.py
deepfactcite/reward.py
scripts/deepfactcite/summarize_grpo_rollouts.py
```

每个文件的作用：

| File | Role |
|---|---|
| `prepare_sglang_grpo_claim_filtered.py` | 构建窄范围的声明级 GRPO 数据 |
| `run_sglang_grpo_ablation_2gpu.sh` | 使用 DeepFactCite reward 启动 Search-R1/SGLang/GRPO |
| `verl_deepfactcite_reward.py` | 把 verl reward manager 连接到项目 reward 代码 |
| `deepfactcite/reward.py` | 解析答案/搜索/引用行为，并计算 reward 细项 |
| `summarize_grpo_rollouts.py` | 把 rollout JSONL 转成指标报告 |

## 心智模型

当前工作流是：

```text
1. 构建窄范围的声明级数据。
2. 验证检索器能返回目标 URL。
3. 在不保存 checkpoint 的情况下运行小型 GRPO 消融。
4. 检查 rollout 级失败。
5. 修正数据、prompt 和 reward 的对齐问题。
6. 然后才运行一次保存 checkpoint 的中等规模训练。
7. 用答案/搜索指标和引用指标一起评测 checkpoint。
8. 同时记录成功和失败。
```

这个项目的价值在于：它把引用质量当成一个工程系统来处理，而不是只看某一个 reward 数字。
