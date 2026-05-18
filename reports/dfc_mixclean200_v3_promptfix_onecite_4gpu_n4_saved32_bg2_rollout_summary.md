# DeepFactCite 4GPU N=4 saved32 后台 GRPO 运行总结（2026-05-19）

## 结论

本次 `4GPU / GRPO_N=4 / train_batch_size=1 / 32 steps / model-only checkpoint` 运行已完成，并成功保存 checkpoint。训练进程使用 `nohup + setsid` 后台运行，未依赖交互会话。

- 实验名：`dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2`
- checkpoint：`outputs/deepfactcite/grpo/dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2/global_step_32/actor`
- tracker：`latest_checkpointed_iteration.txt = 32`
- checkpoint 内容：4 个 model shard，每个约 `8.19G`，总目录约 `31G`
- 训练耗时：进度条记录约 `22m17s`，step 32 保存耗时约 `13.6s`
- 结束状态：训练进程退出，4 张 GPU 空闲
- 磁盘状态：`/root/autodl-tmp` 剩余约 `16G`

## 本次配置

```bash
DRY_RUN=0
GRPO_TOTAL_STEPS=32
GRPO_SAVE_FREQ=32
GRPO_N=4
GRPO_TRAIN_BATCH_SIZE=1
GRPO_MINI_BATCH_SIZE=1
GRPO_TEMPERATURE=0.3
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

运行前修复了 saved run 容错：

- 按 train parquet 行数预检查 `GRPO_TOTAL_STEPS / GRPO_TOTAL_EPOCHS / GRPO_SAVE_FREQ`，避免“训练自然结束但没到保存点”。
- saved wrapper 默认启用 `GRPO_REQUIRE_FINAL_CHECKPOINT=1`，训练结束后强制检查 tracker 和 actor checkpoint。
- Ray 临时目录不能设为 `tmp/ray`，因为仓库路径过长会触发 Unix socket 107 字节限制；本次改为仓库根目录下的 `ray/`，仍不占 `/tmp` 根盘。

## Rollout 指标

本次共 `32` 个 step、`128` 个样本，`28` 个 unique query；每个 step 固定 `4` 个 rollout。

| 指标 | 全部 128 样本 | 前 16 step | 后 16 step |
|---|---:|---:|---:|
| reward | 0.5182 | 0.4952 | 0.5412 |
| total | 0.4863 | 0.4651 | 0.5076 |
| search | 0.9688 | 0.9375 | 1.0000 |
| format | 0.9922 | 0.9844 | 1.0000 |
| url_validity | 0.9453 | 0.8906 | 1.0000 |
| citation_precision | 0.6398 | 0.6078 | 0.6719 |
| claim_support | 0.6367 | 0.6016 | 0.6719 |
| unsupported | 0.1016 | 0.1719 | 0.0312 |
| citation_count | 0.9453 | 0.8906 | 1.0000 |

其他计数：

- `no_search`: `4 / 128`
- `no_citation`: `7 / 128`
- `claim_support < 0.5`: `17 / 128`
- `unsupported > 0`: `13 / 128`
- `fake_url_rate`: `0.0000`
- `bare_citation_count`: `0.0000`
- `raw_url_count`: `0.0000`

GRPO group 方差信号：

- `32 / 32` 个 group 都有非零 reward std
- mean group reward std：`0.0799`
- mean group reward range：`0.1848`

这说明 `GRPO_N=4` 下每个 query 的 4 个 rollout 之间确实存在可用于相对优势估计的 reward 差异，比 `N=2` 更有训练信号。

## 与前序诊断对比

| 运行 | 样本数 | reward | search/url | claim_support | unsupported | no_search / no_citation |
|---|---:|---:|---:|---:|---:|---:|
| v3 2GPU diagnostic, `N=2` | 32 | 0.5120 | URL 0.9375 | 0.6250 | 0.1250 | no_citation 2/32 |
| 4GPU N=4 no-save diagnostic | 64 | 0.5162 | search 0.9688 / URL 0.9375 | 0.6328 | 0.1719 | 2 / 4 |
| 4GPU N=4 saved32 本次 | 128 | 0.5182 | search 0.9688 / URL 0.9453 | 0.6367 | 0.1016 | 4 / 7 |
| 4GPU N=4 saved32 后半程 | 64 | 0.5412 | search 1.0000 / URL 1.0000 | 0.6719 | 0.0312 | 0 / 0 |

这些是训练 rollout 指标，不等价于 held-out eval。可以作为训练信号质量和策略轨迹改善的证据，但简历或面试里不能把它直接说成最终泛化指标。

## 观察

1. 保存链路已经打通：`global_step_32/actor`、4 个模型 shard、HF tokenizer/config、`data.pt` 和 tracker 都存在。
2. 后半程相较前半程明显改善，尤其是 `search/url_validity/citation_count` 全部达到 `1.0000`，`unsupported` 从 `0.1719` 降到 `0.0312`。
3. 仍然存在 claim support 上限问题：全局 `claim_support=0.6367`，后半程也只有 `0.6719`。下一阶段重点应是 evidence-claim 对齐，而不是继续只优化工具调用。
4. 保存 checkpoint 后磁盘只剩约 `16G`，继续 saved GRPO 前必须先规划磁盘；不能直接再跑一个会写 31G checkpoint 的实验。

## 建议下一步

1. 先对 `global_step_32/actor` 做同口径 eval，对比 MixClean200 SFT baseline 的 Search-R1 BM25 200 结果。
2. 如果 eval 有提升，再决定是否清理无用输出、转存 checkpoint 或继续更长训练。
3. 如果 eval 没提升，优先分析低分样本和 citation support reward，而不是盲目加步数。
