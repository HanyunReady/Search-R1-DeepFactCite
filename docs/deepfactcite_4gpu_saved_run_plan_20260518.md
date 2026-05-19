# DeepFactCite 4-GPU 保存 Checkpoint 计划

日期：2026-05-18

## 结论

在 v3 prompt-fix 结果出来之后，只使用 4 张 GPU 做一次会保存 checkpoint 的中等规模训练。下一步不要租 8 张 GPU。当前瓶颈不是原始并行度，而是实验控制、checkpoint/评测纪律，以及 claim-support 行为。

现在值得使用 4 张 GPU 的原因：

| Gate | v3 result | Decision |
|---|---:|---|
| no-citation failures | 2/32 | passed |
| claim support | 0.6250 | passed |
| unsupported citation rate | 0.1250 | passed |
| search | 0.9375 | watch closely |
| response clipping | 0.0000 | passed |
| fake URL rate | 0.0000 | passed |

下一轮训练的目标不是再做一次 smoke test，而是保存一个可以用答案/搜索指标和引用指标共同评测的 checkpoint。

这次 4-GPU 计划默认使用 `Qwen3-8B-Base` 和 v3 one-citation DeepFactCite 受控语料。它不是在 4 张卡上复刻原始 Search-R1 的 `Qwen2.5/Llama3.2 + wiki-18/E5` 大检索栈；后者本地 `wiki-18` 语料约 `21,015,324` 条、`19G`，BM25 索引约 `2.2G`，变量更多。这里先用 32 条带 URL 的窄 claim corpus 验证 citation-aware reward 是否真的改善引用行为。

## 已准备的入口

参数化基础启动脚本：

```text
scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

这个脚本默认仍然是 2 GPU，但现在接受这些参数：

```text
N_GPUS_PER_NODE
TENSOR_MODEL_PARALLEL_SIZE
CUDA_VISIBLE_DEVICES
```

4-GPU v3 保存版 wrapper：

```text
scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

当只拿到 3 张卡时使用的 2-GPU fallback wrapper：

```text
scripts/deepfactcite/run_sglang_grpo_v3_promptfix_2gpu_saved.sh
```

默认设置：

```text
DATA_DIR=data/deepfactcite_sglang_grpo_claim_filtered_v3_promptfix_onecite
MODE=citation-aware
CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS_PER_NODE=4
TENSOR_MODEL_PARALLEL_SIZE=4
GRPO_TOTAL_STEPS=64
GRPO_SAVE_FREQ=16
GRPO_MAX_ACTOR_CKPT_TO_KEEP=1
GRPO_ACTOR_CKPT_SAVE_CONTENTS=[model]
SEARCHQA_TOPK=2
rollout.n=2
train_batch_size=1
```

wrapper 默认 `DRY_RUN=1`，所以可以在没有 GPU 的机器上安全检查。

## 3-GPU 可用时的处理

不要在 3 张 GPU 上运行 4-GPU 配置。

原因：

```text
准备好的 4-GPU 运行使用 tensor_model_parallel_size=4。3-GPU 版本需要
TP=3 或不同的 actor/rollout 放置方式。TP=3 不是这个 Qwen3-8B/SGLang
设置下验证过的标准切分方式，并且可能与 attention head 分片不兼容。
调这个问题会把 GPU 时间花在基础设施上，而不是花在引用学习上。
```

如果只有 3 张卡可用，使用稳定 TP=2 路径的 2 卡配置：

```bash
DRY_RUN=0 \
GRPO_TOTAL_STEPS=8 \
GRPO_SAVE_FREQ=4 \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu_save_sanity \
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_2gpu_saved.sh
```

如果 8-step 保存 sanity run 是干净的，继续：

```bash
DRY_RUN=0 \
GRPO_TOTAL_STEPS=64 \
GRPO_SAVE_FREQ=16 \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260518_2gpu_saved \
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_2gpu_saved.sh
```

这会比 4 GPU 慢，但比临时发明一个未验证的 3-GPU 拓扑更干净、更有科学性。

## 磁盘要求

清理前 `/root/autodl-tmp` 的剩余空间大约只有 14G。这对保存 checkpoint 的工作来说太紧。

启动前的最低建议：

```text
保存 model-only checkpoint 至少需要 35G 可用空间。
保存带 extra state、可恢复训练的 checkpoint 至少需要 50G 可用空间。
```

准备好的 wrapper 默认只保存模型：

```text
GRPO_ACTOR_CKPT_SAVE_CONTENTS=[model]
```

先使用这个设置。只有在恢复训练比磁盘压力更重要时，才切换到 `[model,extra]`。

## 命令

CPU/无 GPU dry-run 检查：

```bash
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

第一次 4-GPU checkpoint 写入 sanity run：

```bash
DRY_RUN=0 \
GRPO_TOTAL_STEPS=8 \
GRPO_SAVE_FREQ=4 \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260518_4gpu_save_sanity \
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

如果这次运行能干净启动、写出 checkpoint，并且 rollout 指标保持正常，再运行中等规模保存实验：

```bash
DRY_RUN=0 \
GRPO_TOTAL_STEPS=64 \
GRPO_SAVE_FREQ=16 \
EXPERIMENT_NAME=dfc-mixclean200-v3-promptfix-onecite-20260518_4gpu_saved \
bash scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
```

预期输出位置：

```text
outputs/deepfactcite/grpo/<EXPERIMENT_NAME>/
logs/grpo/rollouts/<EXPERIMENT_NAME>/
logs/<EXPERIMENT_NAME>.log
tensorboard_log/DeepFactCite-GRPO/<EXPERIMENT_NAME>/
```

## 停止规则

如果前 8 到 16 步出现以下任一信号，应尽早停止 4-GPU 运行：

| Signal | Stop condition |
|---|---|
| search | 明显低于 v3 诊断运行的 0.9375 水平 |
| no citation | 回升到接近 v2 的行为，尤其是超过 6/32 |
| claim support | 在可比样本上低于 0.50 |
| unsupported citation rate | 高于 0.25 |
| fake URL rate | 在重复样本中变为非零 |
| checkpoint | 保存失败，或磁盘可用空间降到危险水平 |
| infrastructure | SGLang/Ray 反复重启、卡死或内存泄漏 |

## 运行后检查清单

运行结束后：

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python \
  scripts/deepfactcite/summarize_grpo_rollouts.py \
  --rollout-dir logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260518_4gpu_saved \
  --out reports/dfc_mixclean200_v3_promptfix_onecite_4gpu_saved_rollout_summary.md \
  --recompute-details
```

然后记录：

```text
1. 精确命令；
2. 运行前/运行后的可用磁盘；
3. checkpoint 路径和 checkpoint 大小；
4. rollout 指标；
5. 5 个好样例和 5 个失败样例；
6. 答案/搜索行为是否保持完整；
7. 引用质量是否超过 16-step 诊断运行。
```

不要只因为训练跑完就把它报告为胜利。必须等保存下来的 checkpoint 完成评测后，才能下结论。
