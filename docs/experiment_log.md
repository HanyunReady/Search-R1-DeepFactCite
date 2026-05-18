
## 实验日志

### 初步结果

资源：[wandb](https://wandb.ai/peterjin/Search-R1-open)


初步实验只在 Natural Questions（NQ）数据集上进行，训练方法为 PPO，训练步数较少。


### v0.1

资源：[wandb](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train)、[docs](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa)、[scripts](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.1)


我们把实验从 NQ 扩展到 7 个数据集，并同时使用 PPO 和 GRPO 方法。这个阶段的训练步数仍然较少，并且学习率 warmup 比例较大。


### v0.2

资源：[wandb](https://wandb.ai/peterjin/Search-R1-v0.2)、[docs](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa)、[scripts](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.2)、[paper](https://arxiv.org/abs/2503.09516)


我们修复了几个 bug，包括 [retrieved token masking](https://github.com/PeterGriffinJin/Search-R1/pull/21) 和 [GRPO sample indexing](https://github.com/PeterGriffinJin/Search-R1/commit/9ec2fa9892fbf0315d0c67b4dc08ae8f6cf5f378)。
前者可以显著提升 RL 训练的稳定性。
随后，我们调整了训练脚本：增加训练步数、降低学习率 warmup 比例，以获得更好的性能；同时在不同规模的 LLM（3B、7B、14B）上开展实验。


### v0.3

资源：[wandb](https://wandb.ai/peterjin/Search-R1-v0.3)、[docs](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa)、[scripts](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.3)、[paper](https://arxiv.org/abs/2505.15117)

我们研究了：（1）奖励设计；（2）LLM backbone；（3）搜索引擎。

- 奖励设计
  - 格式奖励
  - 中间检索奖励
- LLM backbone
  - LLM 类型（例如通用 LLM 或推理型 LLM）
  - LLM 规模（3B/7B/14B/32B）
- 搜索引擎
  - RL 训练动态
  - 推理阶段泛化
- 数据规模扩展

详细内容见这篇 [paper](https://arxiv.org/abs/2505.15117)。
