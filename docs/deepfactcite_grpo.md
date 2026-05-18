# Search-R1 上的 DeepFactCite-GRPO

这个分支在 Search-R1 之上增加了一层 DeepFactCite 能力：

- Search-R1 仍然是底层训练框架。
- DeepFactCite 增加了长答案引用提示、SFT 数据转换、轻量级词汇检索器，以及面向“引用可信”的 GRPO 奖励。
- Qwen3-4B 是快速 MVP 目标；Qwen3-8B 是更适合面试展示的强版本目标。

## 资源规划

推荐的稳定资源配置：

| 模型 | LoRA SFT | GRPO |
| --- | ---: | ---: |
| Qwen3-4B | 1x A800 80G | 2x A800 80G |
| Qwen3-8B | 2x A800 80G | 4x A800 80G |

8B 的 GRPO 也许可以通过更小 batch 和更强 offload 压到更少 GPU 上运行，但更现实的目标是 4x A800，因为 Search-R1 的训练环路里同时保留了 rollout/ref/critic 组件和 vLLM 引擎。

## 新增文件

```text
deepfactcite/prompts.py
deepfactcite/reward.py
deepfactcite/retriever_server.py
verl/utils/reward_score/deepfactcite.py
verl/trainer/main_ppo_deepfactcite.py
verl/utils/dataset/sft_dataset.py
scripts/deepfactcite/prepare_data.py
scripts/deepfactcite/prepare_agentic_eval_data.py
scripts/deepfactcite/install_searchr1_env.sh
scripts/deepfactcite/download_qwen3.sh
scripts/deepfactcite/start_lexical_retriever.sh
scripts/deepfactcite/train_sft_qwen3.sh
scripts/deepfactcite/train_grpo_deepfactcite_qwen3.sh
```

此外，Search-R1 的检索结果展示也做了补丁：当语料库提供 URL 时，在 `<information>` 块中加入 `URL:` 行。引用可信训练必须这样做，因为模型只能引用当前搜索轨迹实际返回过的 URL。

## 环境

当前有两个有用的环境配置：

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite

# Download large model files into /root/autodl-tmp with aria2c/resume support.
MODEL_SIZE=4B LOCAL_DIR=/root/autodl-tmp/LLM-qwen3_posttrain/.cache/models/Qwen_Qwen3-4B-Base \
  bash scripts/deepfactcite/download_qwen3.sh

# Original Search-R1 stack: suitable for Qwen2.5/Llama GRPO reproduction.
PROFILE=legacy bash scripts/deepfactcite/install_searchr1_env.sh

# Qwen3 SFT stack: modern transformers without the old vLLM rollout pin.
PROFILE=qwen3-sft bash scripts/deepfactcite/install_searchr1_env.sh
```

`legacy` 配置沿用 Search-R1 默认栈：Python 3.9、torch 2.4.0 CUDA 12.1、vLLM 0.6.3、editable install、flash-attn、PEFT、FastAPI、uvicorn 和 pyarrow。

`qwen3-sft` 配置使用 Python 3.10 和 `transformers>=4.51`。它故意不安装 vLLM，因为原版 Search-R1 固定在 vLLM <= 0.6.3，本地 rollout 适配器也只支持 vLLM 0.3.1/0.4.2/0.5.4/0.6.3。因此，在把 rollout 栈移植到更新的 vLLM/veRL/SGLang 之前，Qwen3 GRPO 不能当成“一条命令即可稳定运行”的路径。

## 数据准备

准备由 DeepCiteFact 转换出的 SFT/RL 数据：

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite
python scripts/deepfactcite/prepare_data.py \
  --deepcitefact-dir /root/autodl-tmp/DeepCiteFact \
  --output-dir data/deepfactcite
```

输出文件：

```text
data/deepfactcite/sft/train.parquet
data/deepfactcite/sft/test.parquet
data/deepfactcite/rl/train.parquet
data/deepfactcite/rl/test.parquet
data/deepfactcite/corpus.jsonl
```

也可以从现有的 `agentic-rl-searchqa` 仓库转换一个小型评测集：

```bash
python scripts/deepfactcite/prepare_agentic_eval_data.py \
  --output-dir data/agentic_eval48
```

## 检索器

对于小型 DeepFactCite 语料库，使用轻量级词汇检索服务：

```bash
conda activate searchr1
cd /root/autodl-tmp/Search-R1-DeepFactCite
CORPUS=data/deepfactcite/corpus.jsonl PORT=8000 \
  bash scripts/deepfactcite/start_lexical_retriever.sh
```

这个服务实现了 Search-R1 的 `/retrieve` API，并返回包含 `contents`、`url` 和 `score` 的文档。

## SFT

快速 MVP：

```bash
cd /root/autodl-tmp/Search-R1-DeepFactCite
conda activate /root/autodl-tmp/conda_envs/searchr1-qwen3-sft

DRY_RUN=1 MODEL_SIZE=4B N_GPUS=1 TOTAL_STEPS=2 \
  bash scripts/deepfactcite/train_sft_qwen3.sh

MODEL_SIZE=4B N_GPUS=1 TOTAL_STEPS=200 \
  bash scripts/deepfactcite/train_sft_qwen3.sh
```

Qwen3 SFT 脚本默认使用 `ATTN_IMPLEMENTATION=sdpa`，不需要编译 flash-attn。若 CUDA 镜像里已经有可用的 flash-attn，可以覆盖为 `ATTN_IMPLEMENTATION=flash_attention_2`。

更强的训练配置：

```bash
MODEL_SIZE=8B N_GPUS=2 TOTAL_STEPS=300 \
  bash scripts/deepfactcite/train_sft_qwen3.sh
```

如果本地模型路径不同，覆盖 `BASE_MODEL` 即可。

## GRPO

这个分支已经准备好了 Qwen3 GRPO 所需的奖励、数据和配置路径，但原始 Search-R1 的 rollout 栈并不天然兼容 Qwen3。脚本里有预检保护：除非 rollout 栈已经移植并显式设置 `ALLOW_EXPERIMENTAL_QWEN3_GRPO=1`，否则会阻止直接用本地 Qwen3 检查点启动。

移植完成后，先启动检索器，再运行：

```bash
MODEL_SIZE=4B N_GPUS=2 TOTAL_STEPS=200 \
  bash scripts/deepfactcite/train_grpo_deepfactcite_qwen3.sh
```

8B 版本：

```bash
MODEL_SIZE=8B N_GPUS=4 TOTAL_STEPS=200 \
  bash scripts/deepfactcite/train_grpo_deepfactcite_qwen3.sh
```

仅结果奖励的消融实验，可以通过关闭引用/支持权重来近似：

```bash
CITATION_WEIGHT=0 SUPPORT_WEIGHT=0 ANSWER_WEIGHT=0.7 FORMAT_WEIGHT=0.2 SEARCH_WEIGHT=0.1 \
  MODEL_SIZE=4B N_GPUS=2 bash scripts/deepfactcite/train_grpo_deepfactcite_qwen3.sh
```

## 奖励

DeepFactCite 奖励由以下部分组成：

```text
answer exact/substr match
citation precision
citation support
format validity
search validity
length/cost penalty
```

它会对协议违规和不受支持的引用设置硬上限。默认情况下，引用支持使用确定性的词汇蕴含代理，因此 GRPO 不依赖外部评判服务。若要启用 OpenAI 兼容评判器：

```bash
export DEEPFACTCITE_JUDGE_BASE_URL=http://127.0.0.1:8001
export DEEPFACTCITE_JUDGE_MODEL=Qwen2.5-32B-Instruct
USE_JUDGE=True bash scripts/deepfactcite/train_grpo_deepfactcite_qwen3.sh
```

## 面试讲法

简短版本：

```text
我在 Search-R1 之上构建了 DeepFactCite-GRPO。Search-R1 优化的是搜索增强后的答案正确性，但只看最终答案的 reward 可能允许“答案正确、引用不支持”的轨迹通过。我增加了一个引用可信 reward：检查行内 markdown 引用是否来自检索证据，以及被引用文本是否真的支持对应声明。训练流程包含 SFT 冷启动、仅结果 GRPO 和引用感知 GRPO 三类消融。
```

更能体现难点的说法：

> 模型出错不只是因为它没学会“搜、选、引用”，更关键的是：到底该用哪些关键词去搜才能命中证据，本身就很难用传统监督信号直接教会模型。
>
> 现实任务里，问题表达千变万化，文档空间巨大，同一个答案也可能对应多种检索表达方式。因此，“生成什么查询词能找到正确证据”是一个隐性的、可优化但不易直接标注的目标。RL/GRPO 的价值就在这里：把最终答案正确性、URL 真实性和引用支持性变成 reward，让模型从结果回溯到搜索行为，间接学会更有效的查询生成和证据选择策略。

面试中可以这样压缩表达：

> 这也是短事实问答和引用问答里最难优化的部分。就像教学生查文献，老师很难提前标注“每个问题到底该搜哪几个关键词”。所以我没有只做监督学习，而是把答案正确性和引用有效性作为奖励，让模型通过 GRPO 从结果反推哪些搜索行为更有价值。
