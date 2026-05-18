#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SNIPPET_RE = re.compile(r"<snippet\s+id\s*=?\s*[\"']?([^\"'>\s]+)[\"']?\s*>(.*?)</snippet>", re.S | re.I)
MD_CITE_RE = re.compile(r"\[[^\[\]\n]+\]\(https?://[^)\s]+\)")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a retrieval-hit DeepFactCite GRPO subset for SGLang.")
    parser.add_argument("--sft-train", default="data/deepfactcite/sft/train.parquet")
    parser.add_argument("--out-dir", default="data/deepfactcite_sglang_grpo_retrieval_hit")
    parser.add_argument("--rows", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, Any]] = []
    corpus_docs: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    df = pd.read_parquet(args.sft_train)
    for idx, row in df.iterrows():
        if len(train_rows) >= args.rows:
            break
        query = str(row.get("query") or "").strip()
        answer_trace = str(row.get("answer") or "")
        snippets = parse_snippets(answer_trace)
        if not query or len(snippets) < 2 or not MD_CITE_RE.search(answer_trace):
            continue
        train_rows.append(make_row(query, idx))
        for snip in snippets:
            url = snip["url"]
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            corpus_docs.append(
                {
                    "title": snip["title"],
                    "url": url,
                    "text": snip["text"],
                    "keywords": keywords(query, snip["title"], snip["text"]),
                }
            )

    if not train_rows:
        raise RuntimeError("No retrieval-hit rows were produced.")

    train_path = out_dir / "train.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)

    corpus_path = out_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as fout:
        for doc in corpus_docs:
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(train_rows),
        "corpus_docs": len(corpus_docs),
        "train_file": str(train_path),
        "corpus_file": str(corpus_path),
        "source_sft": args.sft_train,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_row(query: str, source_index: int) -> dict[str, Any]:
    prompt = [
        {
            "role": "user",
            "content": (
                "Answer the question in 2 to 4 concise sentences. You must first search using "
                "<google_search> query </google_search>. Use markdown citations [short label](URL) "
                "immediately after factual claims, and every URL must come from the tool response. "
                "Finish with the final answer inside <answer> and </answer>.\n"
                f"Question: {query}\n"
            ),
        }
    ]
    return {
        "agent_name": "agentic_rl_searchqa",
        "data_source": "deepfactcite_retrieval_hit",
        "task_type": "long_fact_qa",
        "query": query,
        "answer": "",
        "prompt": prompt,
        "reward_model": {"style": "rule", "ground_truth": {"query": query, "target": [], "task_type": "long_fact_qa"}},
        "extra_info": {
            "task_type": "long_fact_qa",
            "query": query,
            "source": "deepfactcite_sft_retrieval_hit",
            "source_index": int(source_index),
        },
    }


def parse_snippets(text: str) -> list[dict[str, str]]:
    snippets = []
    for _, body in SNIPPET_RE.findall(text or ""):
        title = field(body, "Title")
        url = field(body, "URL")
        snippet_text = field(body, "Text", multiline=True)
        if url and snippet_text:
            snippets.append({"title": title, "url": url, "text": snippet_text})
    return snippets


def field(text: str, name: str, multiline: bool = False) -> str:
    if multiline:
        match = re.search(rf"{re.escape(name)}:\s*(.*)", text or "", flags=re.S)
    else:
        match = re.search(rf"{re.escape(name)}:\s*(.*)", text or "")
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def keywords(*texts: str) -> list[str]:
    seen = []
    for tok in TOKEN_RE.findall(" ".join(texts).lower()):
        if len(tok) <= 2 and tok.isascii():
            continue
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= 48:
            break
    return seen


if __name__ == "__main__":
    main()
