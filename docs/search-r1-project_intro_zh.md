# Search-R1 项目介绍文档

## 1. 项目概述

### 1.1 项目目标与推理检索交替任务定义

本项目命名为 Search-R1，核心目标是利用强化学习（Reinforcement Learning, RL）训练大语言模型，使其在回答开放域问题时能够自主完成“思考（Think）-搜索（Search）-观察证据（Information）-继续思考（Think）-回答（Answer）”的多轮交互过程。

不同于传统的静态 RAG 系统，Search-R1 并不是在推理前一次性检索固定上下文，而是将搜索引擎调用纳入模型的行动空间。模型在每一步生成中首先需要在 `<think>` 标签内进行推理，如果判断当前知识不足，则通过 `<search> query </search>` 主动构造检索词；系统调用外部检索服务后，将返回结果写入 `<information> ... </information>`；模型再基于新信息继续推理，直到最终在 `<answer> ... </answer>` 中给出短答案。

本项目可以被理解为对 DeepSeek-R1/TinyZero 式推理强化学习的工具增强扩展：模型不仅要学会内部推理，还要学会什么时候搜索、如何搜索、如何利用搜索结果，并在终局答案上接受基于标准答案的规则奖励。项目底层基于 veRL 实现，支持 PPO、GRPO、REINFORCE 等强化学习算法，支持 Llama、Qwen2.5、DeepSeek-R1-Distill 等不同模型系列，并提供本地稀疏检索、本地稠密检索、在线搜索 API 等多种搜索后端。

### 1.2 现状与挑战

开放域问答任务的关键难点在于，模型需要在内部知识和外部知识之间做动态分配。对于常见问题，模型可以直接回答；对于长尾事实、多跳问题或容易混淆的实体关系，模型必须主动获取外部证据。传统提示工程或固定 RAG 流程往往难以让模型形成稳定的搜索策略：一方面，模型可能在不需要搜索的问题上过度调用工具，导致成本上升；另一方面，模型也可能在证据不足时直接猜测答案，造成事实错误。

Search-R1 面临的核心挑战主要有三点。第一，搜索动作是离散且稀疏的，终局 Exact Match 奖励很难直接告诉模型哪一次检索词构造是有效的。第二，检索结果属于环境状态，不应被当作模型自己生成的文本参与策略梯度更新，否则会导致训练目标污染。第三，模型需要严格遵守结构化交互格式，否则搜索服务无法稳定解析动作，训练环境也无法区分“继续搜索”和“最终回答”。

因此，Search-R1 的重点不是单纯提升答案命中率，而是建立一个可复现、可扩展的“LLM + 搜索工具 + RL”训练框架，使模型在强化学习过程中逐步形成可用的工具调用行为。

### 1.3 项目技术路线与创新点

Search-R1 采用 veRL 作为强化学习训练底座，通过 Ray 组织 Actor、Rollout、Reference、Critic 等训练角色，并使用 vLLM 加速在线 rollout。训练时，模型面对带有搜索协议说明的用户问题，多轮生成 `<think>`、`<search>` 或 `<answer>`。当模型输出搜索动作时，`search_r1/llm_agent/generation.py` 会调用检索服务的 `/retrieve` 接口，将返回文档转成 `Doc 1(Title: ...) ...` 格式后重新拼入上下文。

项目的主要创新点包括：

- 将搜索引擎作为 RL 环境的一部分，而不是作为预处理模块静态拼接上下文。
- 使用 `<think>`、`<search>`、`<information>`、`<answer>` 标签约束智能体轨迹，便于动作解析、环境反馈与奖励计算。
- 引入 retrieved token masking / state masking，对 `<information>` 检索结果进行 loss mask，避免环境状态 token 被策略模型学习为自回归目标。
- 支持 PPO 与 GRPO 两类训练方式，其中 GRPO 通过 `n_agent` 多样本采样实现组内相对优势估计，适合无 critic 或弱 critic 的推理检索训练。
- 支持本地 BM25、本地稠密向量检索（Flat / ANN）以及 Google、SerpAPI 等在线搜索后端，便于在公开知识库、私有语料和真实互联网环境之间切换。

## 2. 数据处理

### 2.1 数据集概述

项目默认使用 RUC-NLPIR/FlashRAG_datasets 中的开放域 QA 数据，并在快速启动路径中以 Natural Questions（NQ）为主要示例。复现实验脚本进一步构建了 `nq_hotpotqa_train` 数据集，其中训练集由 NQ 与 HotpotQA 合并而成，测试集覆盖 NQ、TriviaQA、PopQA、HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle 等七类数据源。

相关数据处理脚本主要位于：

- `scripts/data_process/nq_search.py`：构建单一 NQ 搜索式训练/测试数据。
- `scripts/data_process/qa_search_train_merge.py`：合并多个 QA 训练数据源。
- `scripts/data_process/qa_search_test_merge.py`：合并多个 QA 测试数据源。
- `scripts/nq_hotpotqa/data_process.sh`：复现实验数据处理入口。

每条样本会被转换为 veRL 兼容的 parquet 格式，核心字段包括 `prompt`、`data_source`、`ability`、`reward_model.ground_truth` 和 `extra_info`。其中 `reward_model.ground_truth.target` 保存标准答案列表，用于后续 Exact Match 奖励计算。

### 2.2 搜索式 Prompt 构建

Search-R1 的搜索式 prompt 直接定义了模型与检索环境交互的协议。项目使用的基础模板如下：

```text
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
```

该模板强调三点：

- 每次获得新信息后必须先进入 `<think>` 推理。
- 搜索动作必须写成 `<search> query </search>`，并由环境返回 `<information>`。
- 最终答案必须写入 `<answer>`，且只要求短答案，不要求长篇解释。

这与普通 Chain-of-Thought prompt 的区别在于，它显式暴露了搜索动作，并且训练框架会真正执行该动作，而不只是让模型“假装检索”。

### 2.3 Parquet 样本格式

项目中的 QA 样本会被整理成如下结构：

```python
data = {
    "data_source": data_source,
    "prompt": [{
        "role": "user",
        "content": question,
    }],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {
            "target": example["golden_answers"],
        },
    },
    "extra_info": {
        "split": split,
        "index": idx,
    },
}
```

这种格式直接对齐 veRL 的 `DataProto` 数据流。训练时，`prompt` 用于生成输入，`ground_truth` 用于奖励函数，`data_source` 用于选择对应的评分逻辑。

### 2.4 语料与索引构建

项目默认示例使用 `wiki-18` 作为检索语料。语料文件推荐采用 jsonl 格式，每行包含 `id` 和 `contents` 两个字段，其中 `contents` 通常由标题和正文拼接而成：

```json
{"id": "0", "contents": "\"Title\"\npassage text ..."}
```

如果使用本地检索器，需要提前构建索引。相关代码位于 `search_r1/search/index_builder.py` 和 `search_r1/search/build_index.sh`。其中：

- BM25 索引依赖 Pyserini / Lucene，适合无 GPU 或领域内缺少高质量 embedding 模型的场景。
- Dense Flat 索引使用 FAISS 精确向量匹配，检索质量较高，但建议开启 GPU。
- Dense ANN 索引支持 HNSW 等近似检索结构，适合 CPU 环境或大规模语料。

快速启动中，官方提供了 `wiki-18` 语料和 e5 Flat 索引的下载脚本：

```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

## 3. 检索系统设计

### 3.1 检索服务架构

Search-R1 将检索服务与 RL 训练主进程解耦。训练脚本只需要知道检索服务的 HTTP 地址，例如：

```bash
retriever.url="http://127.0.0.1:8000/retrieve"
retriever.topk=3
```

当模型生成 `<search> query </search>` 后，环境会向 `/retrieve` 发送请求：

```json
{
  "queries": ["query text"],
  "topk": 3,
  "return_scores": true
}
```

检索服务返回每个 query 对应的文档列表，训练环境再将其格式化为：

```text
Doc 1(Title: "title") passage text
Doc 2(Title: "title") passage text
Doc 3(Title: "title") passage text
```

随后这些内容被包装到 `<information>` 标签中，重新进入模型上下文。

### 3.2 本地稀疏检索

本地 BM25 检索服务位于 `search_r1/search/retrieval_server.py`，启动示例如下：

```bash
index_file=/path/to/bm25
corpus_file=/path/to/wiki-18.jsonl
retriever_name=bm25

python search_r1/search/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name
```

BM25 的优势是部署简单、无需 GPU、查询速度稳定；不足是对语义匹配和复杂多跳问题的召回能力通常弱于稠密检索。

### 3.3 本地稠密检索

稠密检索默认以 `intfloat/e5-base-v2` 为示例模型。Flat 索引适合 GPU 资源充足且追求精确匹配的场景：

```bash
index_file=/path/to/e5_Flat.index
corpus_file=/path/to/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu
```

如果 GPU 资源不足，可以使用 HNSW64 等 ANN 索引。该方式牺牲少量精确性换取 CPU 上更高的检索效率。

### 3.4 在线搜索引擎

除本地语料外，项目也支持在线搜索服务。SerpAPI 服务位于 `search_r1/search/serp_search_server.py`：

```bash
search_url=https://serpapi.com/search
serp_api_key="your_api_key"

python search_r1/search/serp_search_server.py \
  --search_url $search_url \
  --topk 3 \
  --serp_api_key $serp_api_key
```

Google Custom Search 服务位于 `search_r1/search/google_search_server.py`：

```bash
api_key="your_google_api_key"
cse_id="your_cse_id"

python search_r1/search/google_search_server.py \
  --api_key $api_key \
  --topk 5 \
  --cse_id $cse_id \
  --snippet_only
```

在线搜索更适合训练通用搜索智能体，但需要考虑 API 成本、限额、延迟和返回结果稳定性。

## 4. 强化学习与奖励机制设计

### 4.1 多轮 Rollout 交互流程

Search-R1 的多轮交互逻辑由 `search_r1/llm_agent/generation.py` 实现。每轮 rollout 中，模型生成一段文本后，系统会截断到 `</search>` 或 `</answer>`：

- 如果输出 `<answer>`，当前样本结束。
- 如果输出 `<search>`，系统调用检索服务并返回 `<information>`，样本继续。
- 如果既不是合法搜索也不是合法回答，环境返回一段错误提示，要求模型重新尝试。

训练配置中通过 `max_turns` 控制最大交互轮数。根目录脚本通常设置 `max_turns=2`，复现实验脚本通常设置 `max_turns=4`，默认配置文件中最大可设为 10。

### 4.2 Answer Reward：基于 Exact Match 的终局奖励

基础奖励函数位于 `verl/utils/reward_score/qa_em.py`。它首先从模型完整输出中提取最后一个 `<answer>...</answer>`，然后对答案和标准答案进行归一化处理，包括小写化、去标点、去冠词和空白规整。最终使用 Exact Match 判断是否命中任一标准答案。

基础奖励逻辑如下：

- 若无法提取合法 `<answer>`，奖励为 0。
- 若提取出的答案与标准答案 Exact Match，奖励为 1。
- 若答案格式存在但答案错误，则返回 `format_score`，默认通常为 0。

这一设计非常直接，适合短答案开放域 QA。它不会显式奖励冗长推理，而是要求模型通过搜索和推理最终给出可验证的短答案。

### 4.3 Format Reward：结构化轨迹约束

v0.3 实验中加入了结构格式奖励，相关代码位于 `verl/utils/reward_score/qa_em_format.py` 与 `verl/trainer/main_ppo_format.py`。

格式校验器要求 assistant 侧输出满足如下状态机：

```text
<think>...</think>
(<search>...</search><information>...</information><think>...</think>)*
<answer>...</answer>
```

校验逻辑会检查 `<think>`、`<search>`、`<information>`、`<answer>` 的开闭标签数量是否匹配，并禁止标签之间出现非空的游离文本。该机制可以显著减少模型输出无法被环境解析的问题。

在 `train_grpo_format.sh` 中，典型奖励配置为：

```bash
reward_model.structure_format_score=0.2
reward_model.final_format_score=0.1
reward_model.retrieval_score=0
```

对应规则可以概括为：

- 答案正确且结构合法：奖励 1.0。
- 答案正确但结构不完整：奖励 `1.0 - structure_format_score`，默认 0.8。
- 答案错误但结构合法：奖励 `structure_format_score`，默认 0.2。
- 答案错误且结构不合法，但存在可提取的最终答案：奖励 `final_format_score`，默认 0.1。
- 无法提取答案且结构不合法：奖励 0。

这种奖励设计让模型在尚未答对时也能先学会合法工具调用轨迹，从而降低纯终局奖励过于稀疏的问题。

### 4.4 Retrieval Reward：可选的中间检索奖励

`qa_em_format.py` 还提供了 `retrieval_score` 参数。其判断逻辑是提取所有 `<information>` 块，如果任一检索结果中包含标准答案的归一化文本，则认为检索命中。

当样本尚未给出正确最终答案，但结构合法且检索结果包含答案时，可以额外获得 `retrieval_score`。该机制用于鼓励模型先学会搜到正确证据，再进一步学会从证据中抽取最终答案。当前 v0.3 示例脚本中 `retrieval_score=0`，说明论文实验里可以单独控制该项进行消融。

### 4.5 State Masking：检索结果 Token 屏蔽

Search-R1 的一个重要工程细节是对 `<information>` 检索结果进行 token mask。检索结果来自环境，不是模型策略生成的动作。如果直接把这些 token 纳入 actor loss 或 KL 计算，训练会错误地要求模型“生成检索结果正文”，破坏策略学习。

项目在 `generation.py` 中构建 `info_mask`，将检索结果对应位置替换为 pad mask；在 `verl/trainer/ppo/ray_trainer.py` 中，如果 `actor_rollout_ref.actor.state_masking=true`，训练会使用该 mask 创建 `loss_mask`。同时，KL 计算也优先使用 `info_mask`。这使策略更新主要作用在模型真正生成的 `<think>`、`<search>` 和 `<answer>` token 上。

### 4.6 行为统计指标

训练过程中，环境会记录多类行为指标：

- `env/number_of_actions/mean`：平均交互轮数。
- `env/finish_ratio`：样本最终完成回答的比例。
- `env/number_of_valid_action`：合法动作数量。
- `env/ratio_of_valid_action`：合法动作占比。
- `env/number_of_valid_search`：有效搜索次数。

这些指标能够帮助定位训练问题。例如，若 valid action 很低，说明格式学习不足；若 valid search 很低，说明模型没有学会调用检索；若 finish ratio 低，说明模型在最大轮数内无法收敛到最终回答。

## 5. 训练

### 5.1 环境安装

Search-R1 训练环境建议使用 Python 3.9：

```bash
conda create -n searchr1 python=3.9
conda activate searchr1

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb
```

本地检索服务建议使用独立环境，避免与训练侧依赖冲突：

```bash
conda create -n retriever python=3.10
conda activate retriever

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

仓库 `requirements.txt` 中的关键依赖包括 `accelerate`、`datasets`、`hydra-core`、`ray`、`transformers<4.48`、`vllm<=0.6.3`、`wandb`、`flash-attn` 等。

### 5.2 Quick Start 训练流程

以 NQ + wiki-18 + e5 检索器为例，完整流程如下。

步骤 1：下载索引和语料：

```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

步骤 2：处理 NQ 数据：

```bash
python scripts/data_process/nq_search.py
```

步骤 3：启动本地检索服务：

```bash
conda activate retriever
bash retrieval_launch.sh
```

步骤 4：启动强化学习训练：

```bash
conda activate searchr1
bash train_ppo.sh
# 或
bash train_grpo.sh
```

### 5.3 PPO 训练实践

根目录 `train_ppo.sh` 使用 `verl.trainer.main_ppo` 启动 PPO 训练，默认示例模型为 `meta-llama/Llama-3.2-3B`，也预留了 Qwen2.5-3B/7B 及 Instruct 版本配置。

关键训练参数包括：

- `data.train_batch_size=512`
- `data.val_batch_size=256`
- `data.max_prompt_length=4096`
- `data.max_response_length=500`
- `actor_rollout_ref.actor.optim.lr=1e-6`
- `critic.optim.lr=1e-5`
- `algorithm.adv_estimator=gae`
- `actor_rollout_ref.rollout.name=vllm`
- `actor_rollout_ref.actor.state_masking=true`
- `trainer.total_training_steps=1005`
- `retriever.topk=3`

PPO 路径包含 critic 更新，适合稳定的 actor-critic 训练场景。

### 5.4 GRPO 训练实践

根目录 `train_grpo.sh` 同样使用 `verl.trainer.main_ppo` 入口，但将 `algorithm.adv_estimator=grpo`，并设置 `actor_rollout_ref.rollout.n_agent=5`。这意味着每个问题会采样多个回答轨迹，通过组内相对表现估计优势。

GRPO 典型配置包括：

- `actor_rollout_ref.actor.use_kl_loss=true`
- `actor_rollout_ref.actor.kl_loss_coef=0.001`
- `actor_rollout_ref.actor.kl_loss_type=low_var_kl`
- `actor_rollout_ref.rollout.n_agent=5`
- `actor_rollout_ref.rollout.temperature=1`
- `actor_rollout_ref.actor.state_masking=true`

GRPO 更贴近 R1 类训练范式，尤其适合规则奖励明确、但 token 级价值函数难以学习的搜索推理任务。

### 5.5 v0.1 / v0.2 / v0.3 实验版本

项目在 `docs/experiment_log.md` 和 `scripts/nq_hotpotqa/` 中记录了多个实验版本：

- Preliminary：仅在 NQ 上用 PPO 做少量训练步验证，证明 base model 可以通过 RL 学会调用搜索。
- v0.1：扩展到 NQ 与 HotpotQA 训练，并在七个测试集上评测，同时支持 PPO 和 GRPO。
- v0.2：修复 retrieved token masking 与 GRPO sample indexing 等问题，增加训练步数，降低 warmup ratio，并扩展到 3B、7B、14B 等模型规模。
- v0.3：系统研究 reward design、LLM backbone、search engine、data scaling 等因素，加入结构格式奖励和中间检索奖励等设计。

复现实验入口为：

```bash
huggingface-cli download --repo-type dataset PeterJinGo/nq_hotpotqa_train --local-dir $WORK_DIR/data/nq_hotpotqa_train
bash retrieval_launch.sh
bash scripts/nq_hotpotqa/v0.2/train_ppo.sh
bash scripts/nq_hotpotqa/v0.2/train_grpo.sh
bash scripts/nq_hotpotqa/evaluate.sh
```

### 5.6 多机训练

Search-R1 支持基于 Ray 的多节点训练，相关文档位于 `docs/multinode.md`，示例脚本位于 `example/multinode/`。多机训练中存在 head node 和 worker node 两类节点：

```bash
# head node
ray start --head --dashboard-host=0.0.0.0

# worker node
ray start --address=<head_gcs_address>
```

检索服务建议在每个节点都启动一份，保证 rollout 阶段访问本地 `127.0.0.1:8000/retrieve` 时行为一致。随后只需要在 head node 上通过 `ray job submit` 提交训练任务。仓库提供了 Qwen2.5-32B / 72B 的 GRPO 多机脚本，以及 Qwen2.5-32B 的 PPO 多机脚本。

## 6. 测评与推理

### 6.1 评测流程

评测入口为 `scripts/nq_hotpotqa/evaluate.sh`。该脚本将 `+trainer.val_only=true`，通过验证集 dataloader 生成模型输出，并使用同一套规则奖励函数计算评测结果。评测前需要：

- 下载或准备 `data/nq_hotpotqa_train/test.parquet`。
- 启动检索服务。
- 将 `BASE_MODEL` 设置为待评测模型路径。

示例命令：

```bash
data_name=nq_hotpotqa_train
export DATA_DIR=data/${data_name}
export BASE_MODEL=/path/to/model

bash scripts/nq_hotpotqa/evaluate.sh
```

评测指标本质上与训练奖励一致，主要反映各数据集上的 Exact Match 表现，同时训练日志还会记录有效搜索次数、合法动作比例、完成率等行为指标。

### 6.2 推理体验

项目提供 `infer.py` 用于单问题交互推理。默认模型为 HuggingFace 上的 `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo`。使用前需要先启动检索服务，然后运行：

```bash
python infer.py
```

脚本会在模型生成 `</search>` 后暂停，解析最近一次 `<search>` 查询，调用本地 `/retrieve`，再把搜索结果拼回 prompt，直到模型生成最终 `<answer>` 或触发 EOS。

### 6.3 实验结果概述

根据项目 README 与实验日志，Search-R1 的初步结论包括：

- Llama-3.2-3B base model 可以通过 RL 学会调用搜索引擎，并获得性能提升。
- Qwen2.5-7B base model 可以学习多轮搜索与推理交替行为。
- v0.2 修复 retrieved token masking 后，训练稳定性显著改善。
- v0.3 进一步验证了奖励设计、模型规模、基础模型类型、搜索后端与数据规模对训练效果的影响。

仓库中的可视化材料包括 `public/llama32-3b.png`、`public/multi-turn.png`、`public/main.png` 等，可用于展示训练曲线和整体框架。

## 7. 维护支持与二次开发建议

### 7.1 自定义数据集接入

如果需要接入私有 QA 数据，只需保证每条样本包含问题和标准答案，并转换为项目要求的 parquet schema。可以直接参考 `scripts/data_process/qa_search_train_merge.py` 和 `scripts/data_process/qa_search_test_merge.py`。

需要注意的是，当前奖励函数主要面向短答案 Exact Match。如果业务目标是长答案、摘要、引用或开放式生成，需要额外设计事实性、引用、覆盖率或人工偏好奖励，不能直接沿用现有 EM 奖励。

### 7.2 自定义检索后端接入

项目对检索服务的要求很轻量：只要提供兼容 `/retrieve` 的 HTTP 接口，并返回 `{"result": ...}` 结构即可。对于企业私有知识库，可以实现自己的检索服务，然后将训练脚本中的 `retriever.url` 指向该服务。

建议优先保证以下几点：

- 返回结果必须稳定包含 `document.contents` 字段。
- `contents` 中建议保留标题和正文，便于模型构造答案。
- 检索延迟要可控，否则 RL rollout 会被检索服务拖慢。
- 多机训练时，每个节点都应能访问同等质量的检索服务。

### 7.3 常见问题定位

如果模型训练中无法稳定搜索，优先检查：

- Prompt 是否仍包含完整的 `<think>`、`<search>`、`<information>`、`<answer>` 协议。
- 检索服务 `/retrieve` 是否可用，返回结构是否符合预期。
- `actor_rollout_ref.actor.state_masking` 是否开启。
- `max_turns` 是否过小，导致模型来不及搜索后回答。
- `reward_model.structure_format_score` 是否过低，导致模型缺乏格式学习信号。

如果答案 EM 提升有限，优先检查：

- 检索语料是否覆盖标准答案。
- `retriever.topk` 是否过小。
- 检索器类型是否适合当前领域。
- 标准答案是否存在别名、缩写、大小写或粒度差异，导致 EM 低估真实正确率。

### 7.4 项目支持范围

本项目适合作为搜索增强推理智能体的训练底座，重点支持：

1. 技术咨询：解释 Search-R1 数据格式、训练脚本、奖励函数和检索服务接入方式。
2. 环境配置：协助配置 searchr1 训练环境和 retriever 检索环境。
3. 数据适配：将用户自有 QA 数据转换为 veRL parquet schema。
4. 检索适配：对接本地知识库、BM25、向量检索或在线搜索 API。
5. 训练调参：根据模型规模、GPU 数量、检索延迟和任务难度调整 PPO/GRPO 参数。
6. 评测扩展：在 EM 之外增加 F1、Rouge、LLM judge、事实性验证或引用质量评估。
