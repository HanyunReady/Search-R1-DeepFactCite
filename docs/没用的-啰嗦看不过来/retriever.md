
## 搜索引擎

本文档给出几种检索器的启动示例，包括本地稀疏检索器（例如 BM25）、本地稠密检索器（例如 e5）以及在线搜索引擎。
对于本地检索器，本文以 [wiki-18](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus) 语料库为例；对应的语料索引可以在 [bm25](https://huggingface.co/datasets/PeterJinGo/wiki-18-bm25-index)、[e5-flat](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index)、[e5-HNSW64](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index-HNSW64) 中找到。

### 如何选择检索器？

- 如果你有私有语料库或特定领域语料库，选择**本地检索器**。

    - 如果你的领域里没有高质量的 embedding 稠密检索器，选择**本地稀疏检索器**（例如 BM25）。

    - 否则选择**本地稠密检索器**。
    
        - 如果没有足够 GPU 做精确的稠密向量匹配，选择 CPU 上的 **ANN 索引**。

        - 如果 GPU 资源足够，选择 GPU 上的**平面索引**。


- 如果你想训练通用 LLM 搜索智能体，并且预算充足，选择**在线搜索引擎**（例如 [SerpAPI](https://serpapi.com/)）。


- 如果你有某个领域专用的在线搜索引擎（例如 PubMed 搜索），可以参考这个 [link](https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/serp_search_server.py)，自行把它集成到 Search-R1 中。

搜索引擎启动脚本见这个 [link](https://github.com/PeterGriffinJin/Search-R1/tree/main/example/retriever)。

### 本地稀疏检索器

稀疏检索器（例如 bm25）是一类传统方法。它的检索过程很高效，而且不需要 GPU。不过，在某些特定领域里，它的准确性可能不如稠密检索器。

(1) 下载索引。
```bash
save_path=/your/path/to/save
huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir $save_path
```

(2) 启动本地 BM25 检索服务器。
```bash
conda activate retriever

index_file=$save_path/bm25
corpus_file=$save_path/wiki-18.jsonl
retriever_name=bm25

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name
```


### 本地稠密检索器

你也可以使用现成的稠密检索器，例如 e5。这些模型在某些特定领域里通常比稀疏检索器强得多。
如果 GPU 资源充足，建议使用下面的平面索引版本；否则可以使用 ANN 版本。

#### 平面索引

平面索引会做精确的 embedding 匹配，速度较慢但准确性高。为了让它足够高效地支持在线 RL，建议通过 ```--faiss_gpu``` 启用 **GPU**。

(1) 下载索引和语料库。
```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) 启动本地 flat e5 检索服务器。

```bash
conda activate retriever

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path --faiss_gpu

```


#### ANN 索引（HNSW64）

如果只有 **CPU**，可以使用近似最近邻（ANN）索引来提高搜索效率，例如 HNSW64。
它非常高效，但准确性可能不如平面索引，尤其是在检索段落数量较少时。

(1) 下载索引。
```bash
save_path=/the/path/to/save
huggingface-cli download PeterJinGo/wiki-18-e5-index-HNSW64 --repo-type dataset --local-dir $save_path
cat $save_path/part_* > $save_path/e5_HNSW64.index
```


(2) 启动本地 ANN 稠密检索服务器。
```bash
conda activate retriever

index_file=$save_path/e5_HNSW64.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name --retriever_model $retriever_path
```


### 在线搜索引擎

我们同时支持 [Google Search API](https://developers.google.com/custom-search/v1/overview) 和 [SerpAPI](https://serpapi.com/)。更推荐 [SerpAPI](https://serpapi.com/)，因为它集成了多个在线搜索引擎 API（包括 Google、Bing、Baidu 等），并且没有月度配额限制；而 [Google Search API](https://developers.google.com/custom-search/v1/overview) 有每月 10k 的硬配额，不足以支撑在线 LLM RL 训练。

#### SerpAPI 在线搜索服务器

```bash
search_url=https://serpapi.com/search
serp_api_key="" # put your serp api key here (https://serpapi.com/)

python search_r1/search/serp_search_server.py --search_url $search_url --topk 3 --serp_api_key $serp_api_key
```

#### Google 在线搜索服务器

```bash
api_key="" # put your google custom API key here (https://developers.google.com/custom-search/v1/overview)
cse_id="" # put your google cse API key here (https://developers.google.com/custom-search/v1/overview)

python search_r1/search/google_search_server.py --api_key $api_key --topk 5 --cse_id $cse_id --snippet_only
```

