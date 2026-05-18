# DeepFactCite LoRA 合并与评测可复现性复盘

日期：2026-05-18

本文档复盘 `Soft SFT 100`、merged checkpoint、`mix50` 以及评测数字不一致的问题。目标是让整个过程达到工程评审可审计的程度：有哪些权重，它们如何加载，如何合并，哪个 serving path 产生了哪个数字，哪里失败了，以及下一步应该怎么做。

## 执行摘要

当前最好的可复现 SFT baseline 是：

```text
Base Qwen3-8B + Soft SFT LoRA adapter, served by vLLM as dynamic LoRA in bf16.
```

它在当前代码路径下可复现：

| Eval | Current rerun | Earlier aggregate |
|---|---:|---:|
| ShortQA32 answer_subEM | 0.469 | 0.469 |
| ShortQA32 URL validity | 0.812 | 0.750 |
| ShortQA32 claim support | 0.240 | 0.232 |
| Strict47 total | 0.156 | 0.158 |
| Strict47 claim support | 0.103 | 0.088 |

有问题的假设是：

```text
Base + Soft LoRA dynamic serving == merged Soft full model
```

这个假设对本项目并不安全。旧 merged model 是通过 bf16 加载 base model，然后把 fp32 LoRA delta 合并进 bf16 权重得到的。直接的 HF logits 检查显示，这个 bf16 merge 与 PEFT dynamic LoRA forward 在数值上不等价。

下游 `mix50` continuation 是从这个有损 bf16 merged parent 继续训练的，所以它不是有效的 continuation 结果，也不应该被当成获胜模型。

## Artifact 与权重位置

### Base Model

```text
/root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base
```

这是 soft LoRA adapter 使用的 base Qwen3-8B checkpoint。

### Soft SFT 100 Adapter

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100
```

重要文件：

```text
adapter_config.json
adapter_model.safetensors
tokenizer_config.json
chat_template.jinja
tokenizer.json
```

Adapter 配置：

```text
base_model_name_or_path = /root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base
peft_type = LORA
r = 32
lora_alpha = 64
lora_dropout = 0.0
target_modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
task_type = CAUSAL_LM
```

Adapter tensor 是 fp32：

```text
adapter_model.safetensors: {'torch.float32': 504}
```

### Strict SFT 100 Adapter

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora-strict-100/global_step_100
```

这是单独训练的 adapter，适合作为消融项。目前它不是赢家。

### 旧 bf16 Soft Merged Parent

原始输出路径：

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged
```

诊断后这个目录已删除，用于回收磁盘空间。日志和评测报告仍然保留：

```text
logs/merge_soft100_lora.screen.log
logs/merge_soft100_lora.exit
reports/soft_merged_base_qwen3_8b_vllm_shortqa_guardrail32.json
reports/soft_merged_base_qwen3_8b_vllm_shortqa_guardrail32.jsonl
reports/soft_merged_base_qwen3_8b_vllm_deepfactcite_strict_sft_test47.json
reports/soft_merged_base_qwen3_8b_vllm_deepfactcite_strict_sft_test47.jsonl
```

旧 merge 过程使用 `torch_dtype=torch.bfloat16` 加载 base，然后调用 `PeftModel.from_pretrained(...).merge_and_unload()`。

结果：

- 保存后内部自洽；
- 但不等价于 dynamic PEFT LoRA forward；
- 不能安全地作为 continuation parent。

### fp32 Soft Merged Parent

当前输出路径：

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged-fp32
```

大小：

```text
31G
```

重要文件：

```text
config.json
generation_config.json
model-00001-of-00009.safetensors
...
model-00009-of-00009.safetensors
model.safetensors.index.json
tokenizer_config.json
chat_template.jinja
tokenizer.json
```

合并命令形态：

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python \
  scripts/deepfactcite/merge_lora_adapter.py \
  --base-model /root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base \
  --adapter outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100 \
  --output-dir outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged-fp32 \
  --torch-dtype float32
```

日志：

```text
logs/merge_soft100_fp32_lora.screen.log
logs/merge_soft100_fp32_lora.exit
```

退出码：

```text
0
```

### mix50 Continuation

原始输出路径：

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged-mix-50/global_step_50
```

诊断后这个目录已删除，因为它是从有损 bf16 merged parent 训练出来的，不应该继续使用。日志和报告仍然保留：

```text
logs/deepfactcite-sft-qwen3-8b-soft-merged-mix-50.screen.log
logs/deepfactcite-sft-qwen3-8b-soft-merged-mix-50.exit
reports/mix50_qwen3_8b_vllm_shortqa_guardrail32.json
reports/mix50_qwen3_8b_vllm_shortqa_guardrail32.jsonl
reports/mix50_qwen3_8b_vllm_deepfactcite_strict_sft_test47.json
reports/mix50_qwen3_8b_vllm_deepfactcite_strict_sft_test47.jsonl
```

训练确实正常完成：

```text
train steps = 50
val/loss = 1.016
exit code = 0
```

但由于 parent 是错的，这个结果不是有效的 continuation baseline。

## 时间线

### Step 1：Soft SFT 100 以 LoRA adapter 形式存在

soft SFT checkpoint 是 adapter，不是完整模型：

```text
Base Qwen3-8B + LoRA delta
```

正确的 serving path 是：

```bash
vllm serve /root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base \
  --served-model-name base \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --trust-remote-code \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules soft=outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100
```

vLLM 默认使用 bf16：

```text
dtype=torch.bfloat16
checkpoint size: 15.26 GiB
```

这条路径产生了当前可复现的 soft baseline：

```text
reports/soft_current_qwen3_8b_vllm_shortqa_guardrail32.json
reports/soft_current_qwen3_8b_vllm_shortqa_guardrail32.jsonl
reports/soft_current_qwen3_8b_vllm_deepfactcite_strict_sft_test47.json
reports/soft_current_qwen3_8b_vllm_deepfactcite_strict_sft_test47.jsonl
```

### Step 2：继续 SFT 需要完整 parent

原始 trainer 把 `model.partial_pretrain` 当成完整 HF model 路径，并在其上创建新的 LoRA adapter。它不直接支持：

```text
Base full model + existing LoRA adapter as parent
```

因此我们尝试先把 soft adapter 合并成完整模型，再在上面训练新的 LoRA。

### Step 3：第一次 merge 使用了 bf16

旧 merge 脚本用 bf16 加载 base：

```python
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16,
    device_map=args.device_map,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, args.adapter)
model = model.merge_and_unload()
model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
```

这生成了：

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged
```

这个模型在磁盘上看起来有效，也可以由 vLLM serve，但行为不等价于 dynamic LoRA serving。

### Step 4：mix50 从有损 bf16 merged parent 继续训练

命令形态：

```bash
BASE_MODEL=outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged \
DATA_DIR=data/deepfactcite_mix/sft \
EXPERIMENT_NAME=deepfactcite-sft-qwen3-8b-soft-merged-mix-50 \
MODEL_SIZE=8B N_GPUS=2 TOTAL_STEPS=50 \
bash scripts/deepfactcite/train_sft_qwen3.sh
```

训练完成：

```text
step:50 - val/loss:1.016
exit code: 0
```

评测显示它不是赢家：

| Model | ShortQA answer | ShortQA URL | ShortQA support | Strict47 URL | Strict47 support |
|---|---:|---:|---:|---:|---:|
| Soft dynamic LoRA bf16 | 0.469 | 0.812 | 0.240 | 0.346 | 0.103 |
| mix50 | 0.375 | 0.625 | 0.190 | 0.312 | 0.087 |

此时正确的工程响应不是继续训练更多，而是调查 parent 是否等价。

## 根因调查

### 假设 1：tokenizer 或 chat template 不匹配

检查了：

- soft adapter tokenizer；
- bf16 merged tokenizer；
- mix50 tokenizer；
- base tokenizer。

chat template 实际上是一致的。这不是主要根因。

### 假设 2：config 不匹配

merged config 与 base config 在这些字段上不同：

```text
transformers_version
dtype vs torch_dtype
layer_types
generation_config do_sample field
```

核心架构字段一致：

```text
model_type = qwen3
hidden_size = 4096
num_hidden_layers = 36
num_attention_heads = 32
num_key_value_heads = 8
vocab_size = 151936
rope_theta = 1000000
```

config 差异不足以解释明显的行为变化。

### 假设 3：bf16 merge 损失了 LoRA delta 保真度

这个假设被证实了。

直接 HF logits 测试：

```text
Prompt 0 length: 94 tokens
Prompt 1 length: 98 tokens
```

对比结果：

| Comparison | Prompt | Max Abs Diff | Mean Abs Diff | Top1 Equal | Top10 Overlap |
|---|---:|---:|---:|---:|---:|
| PEFT forward vs bf16 in-memory merge | 0 | 3.328125 | 1.3463 | yes | 8/10 |
| PEFT forward vs bf16 in-memory merge | 1 | 8.546875 | 4.3523 | yes | 9/10 |
| bf16 in-memory merge vs saved bf16 merge | 0 | 0.0 | 0.0 | yes | 10/10 |
| bf16 in-memory merge vs saved bf16 merge | 1 | 0.0 | 0.0 | yes | 10/10 |
| PEFT forward vs fp32 in-memory merge | 0 | 0.000081 | 0.000013 | yes | 10/10 |
| PEFT forward vs fp32 in-memory merge | 1 | 0.000177 | 0.000077 | yes | 10/10 |

解释：

- 保存 merged model 的过程没有损坏模型。
- 坏行为来自把 LoRA delta 合并进 bf16 权重这一步。
- 在 HF logits 层面，fp32 merge 与 dynamic PEFT forward 数值等价。

### 为什么 bf16 merge 不同于 bf16 训练/推理

这一点很容易误解。

SFT trainer 以 fp32 加载模型：

```python
AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32)
```

然后用 FSDP mixed precision 包装模型：

```python
MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)
```

所以训练可以用 bf16 计算，同时保留更高精度的 master/optimizer 路径。

出问题的是另一个操作：

```text
base_weight_bf16 += lora_delta_fp32
```

如果 merge 时 base weight tensor 已经是 bf16，小的 LoRA delta 在吸收到 base weight 时可能被舍入掉或扭曲。Dynamic LoRA serving 避免了这个问题，因为 LoRA delta 仍然保留为独立 adapter 路径。

### 假设 4：fp32 merged full model 应该匹配 dynamic LoRA vLLM eval

这个假设无法用同一条 vLLM dynamic LoRA 路径测试，因为 vLLM 的 LoRA kernel 不支持 float32 LoRA 权重。

尝试过的命令形态：

```bash
vllm serve /root/autodl-tmp/agentic-rl-searchqa/.cache/models/Qwen3-8B-Base \
  --dtype float32 \
  --enable-lora \
  --lora-modules soft=outputs/deepfactcite/deepfactcite-sft-qwen3-8b-lora/global_step_100
```

它在 LoRA graph profiling 阶段失败：

```text
assert weight.dtype in [torch.float16, torch.bfloat16]
AssertionError
```

所以我们无法直接比较：

```text
vLLM float32 dynamic LoRA
vs
vLLM float32 merged full model
```

这里唯一可用的精确等价性检查是 HF logits 对比。

## 为什么小差异会放大成评测指标变化

这不是普通的单轮 QA，而是一个搜索智能体循环：

1. 模型输出 `<search>query</search>`。
2. 本地词汇检索器返回 top-k 证据。
3. 模型基于检索片段继续生成。
4. 它可能继续搜索，也可能带引用作答。
5. Reward 检查答案、引用 URL、URL 真实性和声明支持。

如果第一条生成查询有细微变化，检索到的片段就可能变化。一旦片段变化，后续整个轨迹都可能分叉。因此，一个很小的模型侧变化也可能导致很大的指标变化。

ShortQA32 中观察到的例子：

- Dynamic soft LoRA 和 fp32 merged 往往开头相似，但搜索查询不同。
- 有些变化后的查询返回了更弱的片段。
- 即使最终答案相似，引用有效性和支持性也会变化。

这就是为什么 `temperature=0` 是跨 serving path 可复现的必要条件，但不是充分条件。

## 不同 Serving Path 的评测结果

### ShortQA32

| Serving Path | Answer | Total | Citation Presence | URL Validity | Citation Precision | Claim Support | Unsupported | Search Turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B bf16 | 0.219 | 0.189 | 0.031 | 0.031 | 0.041 | 0.031 | 0.875 | 2.500 |
| Soft dynamic LoRA bf16, current rerun | 0.469 | 0.375 | 0.844 | 0.812 | 0.240 | 0.240 | 0.625 | 1.219 |
| Strict dynamic LoRA bf16 | 0.469 | 0.370 | 0.781 | 0.781 | 0.237 | 0.234 | 0.635 | 1.000 |
| Soft bf16 merged full model | 0.438 | 0.444 | 0.844 | 0.844 | 0.375 | 0.375 | 0.479 | 1.125 |
| Soft fp32 merged full model served as vLLM fp32 | 0.406 | 0.361 | 0.750 | 0.688 | 0.240 | 0.237 | 0.615 | 1.000 |
| mix50 from bf16 merged parent | 0.375 | 0.324 | 0.625 | 0.625 | 0.190 | 0.190 | 0.750 | 1.156 |

### Strict47

Strict47 没有 gold answer，所以 `answer_subem` 没有意义。

| Serving Path | Total | Citation Presence | URL Validity | Citation Precision | Claim Support | Unsupported | Search Turns |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-8B bf16 | 0.124 | 0.064 | 0.064 | 0.082 | 0.028 | 0.433 | 1.298 |
| Soft dynamic LoRA bf16, current rerun | 0.156 | 0.426 | 0.346 | 0.113 | 0.103 | 0.768 | 0.894 |
| Strict dynamic LoRA bf16 | 0.137 | 0.383 | 0.282 | 0.071 | 0.063 | 0.797 | 0.894 |
| Soft bf16 merged full model | 0.138 | 0.277 | 0.234 | 0.070 | 0.055 | 0.762 | 0.979 |
| mix50 from bf16 merged parent | 0.150 | 0.340 | 0.312 | 0.095 | 0.087 | 0.778 | 0.894 |

## 什么是可复现的

可复现的是：

```text
Base + soft LoRA, vLLM bf16 dynamic LoRA
```

对应 artifact：

```text
reports/soft_current_qwen3_8b_vllm_shortqa_guardrail32.jsonl
reports/soft_current_qwen3_8b_vllm_deepfactcite_strict_sft_test47.jsonl
logs/vllm_origbase_soft.screen.log
```

不能当作等价项复现的是：

```text
Soft LoRA dynamic serving == merged full model serving
```

merged full model 是不同 serving path，必须单独标注。

## 已应用的工程修复

### Merge Script

文件：

```text
scripts/deepfactcite/merge_lora_adapter.py
```

变更：

- 增加 `--torch-dtype {float32,bfloat16}`。
- 默认值改为 `float32`。
- 如果 adapter 路径包含 tokenizer 文件，则从 adapter 路径加载 tokenizer。
- help 文本说明 bf16 merge 与 PEFT forward 在数值上不等价。

验证：

```bash
/root/autodl-tmp/conda_envs/searchr1-qwen3-sft/bin/python -m py_compile \
  scripts/deepfactcite/merge_lora_adapter.py
```

### Reproducibility Log

文件：

```text
docs/deepfactcite_reproducibility_issue_log.md
```

它记录：

- 已确认的问题；
- 必须记录的变量；
- 不能混在一起比较的结果类型；
- 当前安全操作规则。

## 面试级解释

如果面试官问为什么 merged checkpoint 表现不同：

> 原始 SFT checkpoint 是 Qwen3-8B 上的一个 LoRA adapter。为了更快继续训练，我一开始把 adapter 合并成完整模型，但 merge 脚本用 bf16 加载 base。adapter tensor 是 fp32，把 fp32 LoRA delta 合并进 bf16 base weight 会产生一个不等价的模型。我用 HF logits 对比验证了这一点：PEFT dynamic forward vs bf16 merged 的最大 logit 差异最高到 8.5，而 PEFT dynamic forward vs fp32 merged 的差异小于 2e-4。保存模型本身没有损坏，问题出在 merge 时的 dtype。我把 merge 脚本默认值改成 fp32，并更新评测协议，确保 dynamic-LoRA、bf16-merged 和 fp32-merged serving path 不再混在同一张对比表里。

如果被问到为什么 eval 数字变化很大：

> 这是搜索智能体循环，不是单轮分类。早期生成 token 的细微差异会改变搜索查询，查询变化会改变检索证据，证据变化会改变引用和答案支持。因此，即使是 greedy decoding，不同 serving path 之间也可能分叉。现在我们为每个结果保存 JSONL rollouts、vLLM 日志、模型路径、dtype、tokenizer、语料库和精确 serving path。

如果被问到哪些 claim 是安全的：

> 可以安全地说：我们构建了 Search-R1 风格的 DeepFactCite pipeline，并且 Soft SFT adapter 在固定 dynamic-LoRA bf16 serving path 下改善了引用行为。不应该说 merged checkpoint 与 dynamic LoRA serving 可互换，也不应该说 mix50 改善了模型，因为 mix50 是从有损 bf16 merged parent 训练出来的。

## 简历就绪门槛

现在可以说：

- 在 Search-R1 风格搜索智能体轨迹之上，实现了 DeepFactCite SFT/eval pipeline。
- 增加了 URL 真实性和声明支持指标/reward 组件。
- 构建了确定性的 vLLM 评测，保存 JSONL rollouts，并固定答案防护评测集和引用评测集。
- 通过证明 bf16 merge 不等价并切换到 fp32 merge，定位并修复了 LoRA merge 可复现性问题。

还不能说：

- Citation-aware GRPO 击败 outcome-only GRPO。
- Merged full model 等价于 LoRA adapter。
- mix50 改善了模型。
- 项目击败了 Search-R1。

对于高标准简历结果，还需要：

1. baseline 和训练后模型使用同一个固定 serving path。
2. 一条不依赖有损 bf16 merge 的有效 continuation 训练路径。
3. Base/Soft/Next-model 在同一 eval backend 下的表格。
4. 最终在同一 backend 下完成 outcome-only GRPO vs citation-aware GRPO。

## 下一步

下一步工程工作应该从下面选一种：

### 首选

实现从以下形式继续训练：

```text
Base model + existing LoRA adapter
```

不要先把 adapter 合并成完整模型。这样可以保留 dynamic-LoRA serving path，并避免比较不兼容的 checkpoint 形式。

### 可接受

只把 fp32 merged parent 当作训练初始化路径，但所有 eval 都要标成 merged-full-model eval，不要把它们和 dynamic LoRA baseline 当作同一个东西比较。

### 不要做

不要从下面这个路径继续训练：

```text
outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged
```

也不要从 `mix50` 继续。它们都和有损 bf16 merged parent 绑定。

## 当前操作状态

本复盘结束时：

- 没有正在运行的 screen job。
- GPU 空闲。
- `outputs/deepfactcite/deepfactcite-sft-qwen3-8b-soft-merged-fp32` 存在，是当前唯一的 merged full parent。
- 旧 bf16 merged parent 和 mix50 adapter 目录已经删除，用于回收磁盘。
- 日志和 JSON/JSONL 报告保留下来用于审计。
