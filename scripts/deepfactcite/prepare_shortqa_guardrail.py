#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepfactcite.prompts import make_searchr1_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare short QA answer guardrail data in DeepFactCite format.")
    parser.add_argument("--shortqa-jsonl", default="/root/autodl-tmp/agentic-rl-searchqa/data/eval/ckpt400_shortqa_32.jsonl")
    parser.add_argument("--corpus-jsonl", default="/root/autodl-tmp/agentic-rl-searchqa/data/offline/eval_ckpt400_ddgs_corpus.jsonl")
    parser.add_argument("--output-dir", default="data/shortqa_guardrail")
    args = parser.parse_args()

    output = Path(args.output_dir)
    (output / "rl").mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, item in enumerate(read_jsonl(Path(args.shortqa_jsonl))):
        query = item.get("query", "").strip()
        answer = item.get("answer")
        if not query or answer is None:
            continue
        targets = [answer] if isinstance(answer, str) else list(answer)
        rows.append(
            {
                "data_source": "deepfactcite",
                "prompt": [{"role": "user", "content": make_searchr1_prompt(query)}],
                "ability": "short_qa",
                "reward_model": {"style": "rule", "ground_truth": {"target": targets, "query": query}},
                "extra_info": {"split": "test", "index": idx, "query": query, "task_type": item.get("task_type")},
            }
        )

    pd.DataFrame(rows).to_parquet(output / "rl" / "test.parquet")
    write_jsonl(output / "rl" / "test.jsonl", rows)
    corpus = convert_corpus(Path(args.corpus_jsonl))
    write_jsonl(output / "corpus.jsonl", corpus)
    print(json.dumps({"test_rows": len(rows), "corpus_rows": len(corpus), "output_dir": str(output)}, indent=2))


def convert_corpus(path: Path) -> list[dict[str, Any]]:
    rows = []
    for item in read_jsonl(path):
        title = item.get("title") or "Untitled"
        url = item.get("url") or f"doc://{len(rows)}"
        text = item.get("text") or ""
        doc_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        rows.append({"id": doc_id, "url": url, "contents": f"\"{title}\"\nURL: {url}\nText: {text}"})
    return rows


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
