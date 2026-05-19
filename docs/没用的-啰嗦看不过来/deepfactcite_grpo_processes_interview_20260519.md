# DeepFactCite GRPO 训练进程与框架原理面试笔记

日期：2026-05-19

本文用于解释一次 4 卡 DeepFactCite-GRPO/SGLang 训练里 `nvidia-smi` 看到的进程、框架调用方式和底层原理。目标不是复述配置，而是回答面试里常见的几个问题：

- 这些 GPU 进程分别在干什么；
- 为什么同一张卡上会同时有 `ray::WorkerDict` 和 `sglang::scheduler_TP*`；
- GRPO 一步训练到底发生了什么；
- Ray、FSDP、SGLang、多轮 search agent 和 reward 是怎么串起来的；
- 如何用通俗但技术可信的方式讲清楚这套系统。

先说明实验口径：这次训练不是复刻原始 Search-R1 的重型 `wiki-18` / E5 检索栈。原始 Search-R1 常见基模是 `Qwen2.5-3B/7B`、`Llama3.2-3B`，`wiki-18` 本地语料约 `21,015,324` 条、`19G`；本项目使用 `Qwen3-8B-Base`，主线训练用带 URL 的 DeepFactCite citation corpus。这样做是为了把变量收窄到“搜索后引用是否可信”，而不是把大索引复现、模型差异和 citation reward 混在一起。

## 1. 结论先行

这不是 8 个彼此独立的训练进程，而是 4 个训练 worker 加 1 个 4-way tensor parallel 的 SGLang rollout 服务。

当前进程快照可以这样读：

| GPU | `ray::WorkerDict` | 显存 | `sglang::scheduler_TP*` | 显存 | 含义 |
|---:|---:|---:|---:|---:|---|
| 0 | 100417 | 8962 MiB | 102799 / TP0 | 13838 MiB | FSDP actor rank + SGLang TP rank 0 |
| 1 | 100418 | 13766 MiB | 102800 / TP1 | 14030 MiB | FSDP actor rank + SGLang TP rank 1 |
| 2 | 100419 | 13766 MiB | 102801 / TP2 | 14030 MiB | FSDP actor rank + SGLang TP rank 2 |
| 3 | 100420 | 13670 MiB | 102802 / TP3 | 13838 MiB | FSDP actor rank + SGLang TP rank 3 |

一句话解释：

```text
Ray 负责组织训练和分布式 actor 更新；SGLang 负责高吞吐多轮 rollout 生成。训练 actor 和推理 engine 共用同一组 GPU，所以每张卡上会看到两类主要 CUDA 进程。
```

## 2. 本次运行的关键配置

这次运行来自 4 卡 SGLang GRPO 脚本：

```text
scripts/deepfactcite/run_sglang_grpo_v3_promptfix_4gpu_saved.sh
scripts/deepfactcite/run_sglang_grpo_ablation_2gpu.sh
```

实际入口是：

```bash
/root/autodl-tmp/conda_envs/grpo-sglang/bin/python -m verl.trainer.main_ppo \
  --config-path=/root/autodl-tmp/SearchShortQA/verl/examples/sglang_multiturn/config \
  --config-name=search_grpo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  trainer.n_gpus_per_node=4
```

需要注意：

- 本次 SGLang GRPO 使用的 `VERL_DIR` 是 `/root/autodl-tmp/SearchShortQA/verl`，不是当前仓库内较旧的 `verl/` 目录。
- 当前模型是 Qwen3，日志显示 `Qwen3ForCausalLM contains 8.19B parameters`。
- 当前运行配置里 `rollout.n=4`，即每个 prompt 采样 4 条回答，用于 GRPO 组内相对比较。
- `reward_model.enable=False`，但 `reward_model.reward_manager=deepfactcite_custom`，说明不用单独训练 reward model，而是调用自定义规则 reward。
- `algorithm.adv_estimator=grpo` 会关闭 value critic。日志里出现 `critic/score/*` 只是 veRL 继承下来的指标命名，表示序列 reward/advantage 统计，不代表真的启用了 critic 模型。

主要产物路径：

```text
logs/dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2.log
logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2/
outputs/deepfactcite/grpo/dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2/
tensorboard_log/DeepFactCite-GRPO/dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2/
```

## 3. 这些进程分别是谁

### 3.1 Ray head/raylet

本次 Ray 本地集群由 `raylet` 管理：

```text
PID 96059  raylet
GPU resources: 4
CPU resources: 64
Ray session: ray/session_2026-05-19_00-58-26_470309_94851/
```

Raylet 不直接做大模型训练，它像一个调度器：

- 管理哪些 Ray actor 放到哪张 GPU；
- 给 worker 注入 `RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT` 等分布式环境变量；
- 维护对象存储、日志和 actor 生命周期。

### 3.2 `ray::WorkerDict`

`ray::WorkerDict` 是 veRL 动态生成的 Ray actor 类名。它不是普通 Python 字典，而是一个“角色容器”：

```text
create_colocated_worker_cls(class_dict=...) -> WorkerDict
```

在这次配置里，它主要承载 `actor_rollout` 角色：

- 加载 Qwen3 actor；
- 用 FSDP 把模型参数分片到 4 张 GPU；
- 计算生成 token 的 log probability；
- 执行 GRPO/PPO-style actor update；
- 保存 actor checkpoint；
- 在 rollout 前把最新 actor 权重同步给 SGLang 服务。

为什么显存占用不完全一致：

- FSDP 参数、梯度、优化器状态会按 rank 分片；
- optimizer offload、临时通信 buffer、CUDA allocator cache 会导致各 rank 显存有差异；
- GPU0 还常承担一些 rank0 逻辑，但实际显存高低不一定固定由 rank0 决定。

### 3.3 `SGLangHttpServer`

`SGLangHttpServer` 是 Ray actor，日志里显示：

```text
SGLang http server: rollout_mode=<RolloutMode.HYBRID: 'hybrid'>,
replica_rank=0, node_rank=0, nnodes=1, cuda_visible_devices='0,1,2,3'
```

它的作用是启动并管理一个 OpenAI-compatible/SGLang HTTP 推理服务。训练主循环中的多轮 agent 不直接调用 FSDP actor 生成，而是通过这个服务发起生成请求。

### 3.4 `sglang::scheduler_TP0` 到 `TP3`

这些是真正执行 SGLang 推理调度和 token 生成的进程：

```text
PID 102799  sglang::scheduler_TP0
PID 102800  sglang::scheduler_TP1
PID 102801  sglang::scheduler_TP2
PID 102802  sglang::scheduler_TP3
```

`TP0` 到 `TP3` 对应 `tensor_model_parallel_size=4`：

- 一个 Qwen3-8B 推理模型被切成 4 个 tensor parallel 分片；
- 每张 GPU 持有一部分权重和 KV cache；
- 每次 forward 需要 TP rank 间用 NCCL 做通信；
- SGLang scheduler 负责请求排队、动态 batching、prefix/KV cache 管理、采样和返回 token。

通俗说：

```text
FSDP actor 是训练用的模型副本，SGLang TP engine 是推理用的模型副本。
训练副本负责学习，推理副本负责快速生成样本。每轮 rollout 前，训练副本会把最新权重同步给推理副本。
```

### 3.5 没出现在 `nvidia-smi` 里的进程

`nvidia-smi` 只列使用 GPU 显存的进程。训练里还有一些重要但偏 CPU/控制面的 Ray actor：

| 进程 | 作用 |
|---|---|
| `TaskRunner` | 主训练控制器，打印配置、跑训练循环、记录指标、保存 checkpoint |
| `AgentLoopWorker` | 多轮 agent 执行器，负责把 prompt 变成一次或多次 LLM/tool 交互 |
| `RewardManagerWorker` | 在 agent loop 中可异步计算 reward；本项目最终训练 reward 也会在主训练 reward 阶段再算一次 |
| `BatchExecutor` | reward/model 批处理辅助执行器 |

## 4. 一步 GRPO 训练怎么走

可以把一步训练想象成 9 个阶段。

### 4.1 取一个 prompt

Dataloader 从 parquet 里取训练样本，包含：

- `prompt`
- `query`
- `answer`
- `task_type`
- `reward_model.ground_truth`
- `extra_info`

本次小规模保存训练里 `data.train_batch_size=1`，也就是每个 step 原始 prompt 数是 1。

### 4.2 把 prompt 复制 `n` 份

训练主循环会执行：

```text
gen_batch = gen_batch.repeat(repeat_times=rollout.n, interleave=True)
```

当前 `rollout.n=4`，所以一个问题会变成 4 条候选轨迹。GRPO 的核心就是比较同一个 prompt 下多条回答谁更好。

### 4.3 进入 async multi-turn rollout

日志里反复出现：

```text
async rollout...
```

这里的 async 主要指 rollout 阶段内部异步：

- 多个 `AgentLoopWorker` 并发处理样本；
- 每条样本可以多轮调用 LLM；
- LLM 需要搜索时，agent loop 调用 `google_search` 工具；
- 搜索工具当前是 offline backend，从本地 corpus 检索；
- 生成请求通过 HTTP/Ray RPC 发给 SGLang server。

这不表示训练更新和生成完全并行。主循环仍然是：

```text
生成 rollout -> 算 reward -> 算 log_prob/advantage -> 更新 actor
```

### 4.4 SGLang 生成 token

SGLang 的优势在 rollout：

- 支持高吞吐生成；
- 支持 prefix/KV cache；
- 能把多个请求动态调度到同一个 engine；
- 用 TP=4 分摊 Qwen3-8B 的推理权重和计算；
- 对多轮 agent 来说，比直接用训练 FSDP actor 生成更合适。

本项目设置了：

```text
actor_rollout_ref.rollout.name=sglang
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.agent.num_workers=2
actor_rollout_ref.rollout.max_num_seqs=1
```

`max_num_seqs=1` 比较保守，牺牲吞吐换稳定性和显存可控性。

### 4.5 工具调用和轨迹拼接

模型不是只输出最终答案，而是可能走这样的轨迹：

```text
用户问题
-> 模型生成 search/tool call
-> google_search 返回 tool response
-> 模型继续读证据
-> 模型输出带 markdown citation 的 <answer>
```

Agent loop 会把 LLM token 和 tool response 拼成一个 response 序列，同时构造 `response_mask`：

- LLM 自己生成的 token：`response_mask=1`；
- 工具返回的 observation：`response_mask=0`。

这很关键。训练时只应该优化模型生成的 token，不应该让模型为检索器返回的文本背梯度。

### 4.6 DeepFactCite reward 打分

本项目 reward 接在：

```text
scripts/deepfactcite/verl_deepfactcite_reward.py
deepfactcite/reward.py
```

`DeepFactCiteRewardManager` 做几件事：

- 解码 prompt 和 response；
- 如果有完整 `trajectory`，优先用轨迹文本打分；
- 调用 `deepfactcite.reward.compute_score`；
- 把标量 reward 放在最后一个有效 response token 上；
- 记录 rollout JSONL，方便后处理。

reward 由多个部分组成：

| 组件 | 作用 |
|---|---|
| `answer_subem` | 最终答案是否覆盖 gold answer |
| `url_validity` / `citation_precision` | markdown citation 是否来自检索证据中的 URL |
| `claim_support` | citation 附近声明是否被所引文档支持 |
| `format` | `<answer>`、搜索/信息标签、结尾结构是否合规 |
| `search` | 是否进行了合理搜索 |
| `cost` | 搜索次数和冗余行为惩罚 |

`citation-aware` 模式下权重大致是：

```text
answer=0.15
citation=0.35
support=0.30
format=0.10
search=0.05
cost=0.05
```

因此模型不会只因“答案看起来对”就拿高分，还必须引用真实检索到的 URL，并让引用文本支持对应声明。

### 4.7 重新计算 old log prob

生成完以后，actor worker 会对这些 response 重新计算 log probability：

```text
old_log_prob = actor_rollout_wg.compute_log_prob(batch)
```

这一步用训练 actor/FSDP 模型完成，不用 SGLang。原因是 actor update 需要知道“采样这些 token 时的策略概率”，并和更新后的概率做 PPO-style ratio。

### 4.8 计算 GRPO advantage

GRPO 的直觉非常简单：

```text
同一个问题生成 4 个答案。
把 4 个答案分别打分。
比组内平均分高的答案，鼓励；
比组内平均分低的答案，抑制。
```

代码里的公式是：

```text
score_i = sum(token_level_rewards_i)
adv_i = (score_i - mean(score_group)) / (std(score_group) + epsilon)
```

然后把这个标量 advantage broadcast 到该回答的所有模型生成 token 上：

```text
advantages_i = adv_i * response_mask_i
```

这就是 GRPO 相比 PPO 省资源的地方：

- PPO 需要训练 value critic 来估计 baseline；
- GRPO 直接用同一 prompt 下多条采样的组内均值/方差做 baseline；
- 因此当前配置可以关闭 critic worker，少一个大模型副本。

### 4.9 更新 actor

actor 更新仍然是 PPO-style policy gradient：

```text
ratio = exp(new_log_prob - old_log_prob)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

如果某条回答 reward 高于组内平均，advantage 为正，训练会提高这些 token 的概率；如果 reward 低于组内平均，advantage 为负，训练会降低这些 token 的概率。

当前运行里：

- `actor.use_kl_loss=False`
- `algorithm.use_kl_in_reward=False`

所以没有启用 reference policy KL 约束。若打开 KL，系统还会增加 ref policy logprob 计算或在 actor 内计算 reference logprob，用于限制模型偏离初始策略太远。

## 5. 为什么需要 Ray

Ray 在这里不是算法本身，而是分布式执行层。

它负责：

- 拉起本地或多节点 worker；
- 给每个 worker 分配 GPU；
- 用 placement group 固定资源放置；
- 把 actor、rollout、ref、reward 等角色映射到资源池；
- 提供 Ray RPC，让 `TaskRunner` 能调用远端 worker 的 `generate_sequences`、`compute_log_prob`、`update_actor` 等方法；
- 管理 SGLang server 和 agent loop worker 这些异步组件。

面试里可以这样说：

```text
Ray 解决的是“把训练系统拆成多个长期运行的角色，并稳定调度到 GPU/CPU 上”的问题。GRPO 算法不依赖 Ray，但没有 Ray，这套 actor、SGLang rollout server、多轮 agent worker 和 reward worker 的编排会非常麻烦。
```

## 6. 为什么需要 FSDP

FSDP 负责训练副本的显存控制。

Qwen3-8B 即使 bf16 权重只有十几 GB，训练时还会有：

- 参数；
- 梯度；
- optimizer state；
- activation；
- logprob/update 临时张量；
- CUDA/NCCL buffer。

FSDP 把模型参数和优化状态切到多个 rank：

```text
4 个 WorkerDict rank 共同持有一个 actor。
每个 rank 只常驻一部分参数/optimizer state。
需要 forward/backward 时再按层 all-gather 或 reduce-scatter。
```

当前配置还用了：

```text
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

含义：

- 参数主要留在 GPU 上，避免频繁 CPU/GPU 拷贝；
- optimizer state 可以 offload，降低训练显存；
- gradient checkpointing 用额外计算换 activation 显存。

## 7. 为什么还要单独 SGLang

训练 actor 已经有模型，为什么还要 SGLang？

原因是训练和推理的优化目标不同：

| 组件 | 优化目标 | 典型技术 |
|---|---|---|
| FSDP actor | 反向传播、参数更新、optimizer 管理 | FSDP、gradient checkpointing、optimizer offload |
| SGLang engine | 快速生成大量 rollout | KV cache、prefix cache、dynamic batching、tensor parallel |

如果直接用 FSDP actor 做多轮生成：

- 生成吞吐差；
- KV cache 管理不如推理引擎；
- 多轮 tool call 的请求调度复杂；
- 训练/推理模式频繁切换更重。

所以本项目采用 hybrid engine：

```text
训练时：FSDP actor 更新权重。
rollout 前：wake_up，把 actor 权重同步到 SGLang。
rollout 中：SGLang 负责生成，agent loop 负责工具交互。
rollout 后：sleep，切回训练状态，然后 actor 更新。
```

这也是为什么每张 GPU 上同时有两类显存占用。

## 8. 指标怎么读

日志里的 step 指标可以分成几类。

### 8.1 行为指标

```text
acc/rewards/*
search/rewards/*
search/nums/*
format/rewards/*
```

这些来自 `DeepFactCiteRewardManager` 返回的辅助统计。它们告诉我们：

- 答案是否命中；
- 是否搜索；
- 搜索次数是否合理；
- 输出格式是否合规。

### 8.2 序列 reward/advantage 指标

```text
critic/score/*
critic/rewards/*
critic/advantages/*
critic/returns/*
```

虽然前缀叫 `critic`，但当前 GRPO 没有启用 critic。这里是 veRL 的通用 metric 命名：

- `score` 是 `token_level_scores` 按序列求和；
- `rewards` 是 KL 后的 reward，当前未启用 KL 时等于 score；
- `advantages` 是 GRPO 组内标准化后的 advantage；
- `returns` 在 GRPO 下通常与 advantage 同形。

### 8.3 性能指标

```text
timing_s/gen
timing_s/reward
timing_s/old_log_prob
timing_s/update_actor
timing_s/step
timing_per_token_ms/gen
perf/throughput
perf/max_memory_allocated_gb
```

本次日志里 rollout 通常是最耗时部分，例如：

```text
timing_s/gen: 20s-50s+
timing_s/update_actor: 3s 左右
```

这说明当前瓶颈主要在多轮生成和搜索轨迹，而不是 actor backward。

## 9. 常见异常怎么判断

| 现象 | 可能原因 | 优先检查 |
|---|---|---|
| `scheduler_TP*` 高显存/OOM | SGLang KV cache 或 TP 分片显存过高 | 降 `GRPO_MAX_RESPONSE_LENGTH`、`GRPO_MAX_NUM_SEQS`、`GRPO_GPU_MEMORY_UTILIZATION` |
| `WorkerDict` 高显存/OOM | actor update/logprob 显存过高 | 降 micro batch、开更多 offload、缩 response length |
| rollout 很慢 | SGLang 生成慢、多轮太多、工具调用等待 | 看 `timing_s/gen`、`num_turns/*`、`agent_loop/slowest/*` |
| reward 很慢 | reward 解析或 judge 太重 | 看 `timing_s/reward`，默认规则 reward 应该很快 |
| `acc/rewards/mean=0` 但 reward 不低 | citation/format/search 给了分 | 看 `critic/score/mean` 和 rollout JSONL |
| 有 `critic/*` 但配置是 GRPO | 指标历史命名 | 确认日志中有 `Disabled critic as algorithm.adv_estimator != gae` |
| 每步 advantage 很小 | 组内 4 条回答分数太接近 | 提高采样温度、增加 `rollout.n`、改 reward 区分度 |

## 10. 面试讲法

### 10.1 30 秒版本

```text
我做的是一个基于 Search-R1/veRL 的 citation-aware GRPO 训练。系统里 Ray 负责任务调度和分布式 worker 管理，FSDP actor 负责训练，SGLang 负责多轮搜索轨迹的高吞吐 rollout。每个问题会采样多条回答，DeepFactCite reward 同时检查答案、搜索、URL 引用和声明支持性。GRPO 不训练 critic，而是用同一问题下多条回答的组内均值做 baseline：比组内平均好的轨迹被增强，比组内平均差的轨迹被抑制。
```

### 10.2 2 分钟版本

```text
这个训练系统可以拆成三层。第一层是 Ray/veRL 的训练编排层，它把 actor、rollout server、agent loop 和 reward manager 都变成 Ray actor，并把 GPU 资源固定分配好。第二层是模型执行层：训练模型用 FSDP 分片，负责 logprob 和 actor update；推理生成用 SGLang，通过 4 路 tensor parallel 启动一个 rollout server。第三层是任务逻辑层：每条样本不是单轮回答，而是 search agent 轨迹，模型可以调用 google_search 工具，读取返回证据，再输出带 markdown citation 的答案。

算法上我用 GRPO 而不是 PPO critic。每个 prompt 采样 n 条轨迹，规则 reward 给每条轨迹一个标量分数，然后在同一个 prompt 的组内做标准化 advantage。这样省掉 value model，资源更可控，也更适合答案正确性和引用可信性这种结果型 reward。关键 reward 不只看最终答案，还检查引用 URL 是否来自检索证据，以及 citation 附近的声明是否真的被文档支持。这个设计能避免模型只学会“答案对但乱引用”。
```

### 10.3 解释进程快照

如果面试官指着 `nvidia-smi` 问，可以这样讲：

```text
这里每张 GPU 上有两个主要进程。`ray::WorkerDict` 是 veRL 通过 Ray 拉起的训练 worker，里面是 FSDP actor rank，负责训练和 logprob。`sglang::scheduler_TP0-TP3` 是 SGLang 推理服务的 4 个 tensor parallel rank，负责 rollout 生成。它们共用 4 张卡：actor 学完以后把权重同步到 SGLang，SGLang 生成多条搜索轨迹，reward 打分后再回到 actor 做 GRPO 更新。
```

### 10.4 为什么不用普通 SFT

```text
SFT 可以教模型模仿已有搜索和引用格式，但很难直接标注“这个问题最应该搜什么 query、哪个证据最值得引用、引用是否足够支持声明”。GRPO 把这些难以逐步标注的行为变成结果奖励：答案是否正确、URL 是否真实、声明是否被证据支持。模型通过多条采样之间的相对好坏，间接学会更好的查询、证据选择和引用行为。
```

### 10.5 为什么 GRPO 适合这里

```text
这个任务的 reward 是结果型的，而且可以规则化计算。GRPO 对每个 prompt 生成多条候选，用组内平均作为 baseline，不需要额外训练 critic。对 8B 模型来说，这能少维护一个大 value model，工程上更省显存，也减少 critic 估计不准带来的不稳定。
```

### 10.6 为什么还要 citation-aware reward

```text
只看答案正确会奖励一种坏行为：模型可能知道答案，或者碰巧答对，但 citation 并不支持对应声明。DeepFactCite 的目标不是短答案问答，而是可信引用回答。所以 reward 里必须把答案、URL validity、citation precision 和 claim support 分开看。这样模型才会学到“引用必须来自检索证据，并且引用文本要支撑当前句子”。
```

## 11. 可以直接引用的技术要点

- `WorkerDict` 是 Ray 动态生成的 colocated worker 类名，用来把一个或多个 veRL 角色放进同一组 Ray actors。
- 当前 GRPO 配置禁用了 critic；日志中的 `critic/score` 是通用 metric 前缀，不是 value critic。
- `SGLangHttpServer` 是父 Ray actor，`sglang::scheduler_TP*` 是实际推理调度进程。
- `TP0-TP3` 来自 `actor_rollout_ref.rollout.tensor_model_parallel_size=4`。
- `rollout.n=4` 表示每个 prompt 生成 4 条候选轨迹，GRPO 在这 4 条里做相对优势估计。
- `response_mask` 把 LLM token 和 tool observation 分开，避免对检索返回文本做策略梯度。
- 自定义 reward 在 `scripts/deepfactcite/verl_deepfactcite_reward.py`，核心解析和打分在 `deepfactcite/reward.py`。
- 当前训练瓶颈主要看 `timing_s/gen`；如果它远大于 `update_actor`，优化重点是 rollout/SGLang/agent loop，而不是 backward。

## 12. 推荐排查命令

查看 GPU 进程：

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader
```

查看父子进程：

```bash
ps -fp 96059,102403
ps -fp 100417,100418,100419,100420,102799,102800,102801,102802
```

查看训练日志：

```bash
tail -f logs/dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2.log
```

查看 rollout 样本：

```bash
ls logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2/
```

生成 rollout 摘要：

```bash
python scripts/deepfactcite/summarize_grpo_rollouts.py \
  --rollout-dir logs/grpo/rollouts/dfc-mixclean200-v3-promptfix-onecite-20260519_4gpu_n4_saved32_bg2 \
  --out reports/dfc_mixclean200_v3_promptfix_onecite_4gpu_n4_saved32_bg2_rollout_summary.md
```

## 13. 一句话心智模型

```text
Ray 是调度层，FSDP actor 是学习层，SGLang 是生成层，AgentLoop 是搜索交互层，DeepFactCite reward 是裁判；GRPO 用同题多答案的相对分数，把“答案正确且引用可信”的行为变成可优化信号。
```
