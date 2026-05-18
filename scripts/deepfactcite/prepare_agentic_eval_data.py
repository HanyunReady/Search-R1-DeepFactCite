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
    parser = argparse.ArgumentParser(description="Convert local agentic-rl-searchqa eval data into Search-R1 format.")
    parser.add_argument("--eval-jsonl", default="/root/autodl-tmp/agentic-rl-searchqa/data/eval/ckpt400_mixed_48.jsonl")
    parser.add_argument("--corpus-jsonl", default="/root/autodl-tmp/agentic-rl-searchqa/data/offline/eval_ckpt400_ddgs_corpus.jsonl")
    parser.add_argument("--output-dir", default="data/agentic_eval48")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, item in enumerate(read_jsonl(Path(args.eval_jsonl))):
        query = item.get("query", "").strip()
        target = item.get("answer")
        target_list = [] if target is None else ([target] if isinstance(target, str) else list(target))
        if not query:
            continue
        rows.append(
            {
                "data_source": "deepfactcite",
                "prompt": [{"role": "user", "content": make_searchr1_prompt(query)}],
                "ability": "deepfactcite",
                "reward_model": {"style": "rule", "ground_truth": {"target": target_list, "query": query}},
                "extra_info": {"split": "train", "index": idx, "query": query, "task_type": item.get("task_type")},
            }
        )

    split_at = max(1, int(len(rows) * (1 - args.val_ratio)))
    train, test = rows[:split_at], rows[split_at:] or rows[:1]
    write_dataset(output / "rl", train, test)
    corpus = convert_corpus(Path(args.corpus_jsonl))
    write_jsonl(output / "corpus.jsonl", corpus)
    print(json.dumps({"train": len(train), "test": len(test), "corpus": len(corpus)}, indent=2))


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


def write_dataset(out_dir: Path, train: list[dict[str, Any]], test: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(out_dir / "train.parquet")
    pd.DataFrame(test).to_parquet(out_dir / "test.parquet")
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "test.jsonl", test)


if __name__ == "__main__":
    main()
