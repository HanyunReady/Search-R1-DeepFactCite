#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small DeepFactCite GRPO smoke set for the SGLang backend.")
    parser.add_argument("--deepfactcite-rl", default="data/deepfactcite/rl/train.parquet")
    parser.add_argument("--guardrail", default="data/searchr1_core_guardrail/test.parquet")
    parser.add_argument("--corpus", default="data/deepfactcite/corpus.jsonl")
    parser.add_argument("--out-dir", default="data/deepfactcite_sglang_grpo_smoke")
    parser.add_argument("--deepfactcite-rows", type=int, default=24)
    parser.add_argument("--guardrail-rows", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows.extend(convert_deepfactcite(args.deepfactcite_rl, args.deepfactcite_rows))
    rows.extend(convert_guardrail(args.guardrail, args.guardrail_rows))

    df = pd.DataFrame(rows)
    train_path = out_dir / "train.parquet"
    df.to_parquet(train_path, index=False)

    corpus_path = out_dir / "corpus.jsonl"
    convert_corpus(args.corpus, corpus_path)

    summary = {
        "rows": len(df),
        "task_type_counts": df["task_type"].value_counts().to_dict(),
        "train_file": str(train_path),
        "corpus_file": str(corpus_path),
        "source_deepfactcite": args.deepfactcite_rl,
        "source_guardrail": args.guardrail,
        "source_corpus": args.corpus,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def convert_deepfactcite(path: str, limit: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(path).head(limit)
    out = []
    for i, row in df.iterrows():
        query = str(row.get("extra_info", {}).get("query") or row.get("reward_model", {}).get("ground_truth", {}).get("query") or "")
        prompt = convert_prompt(row["prompt"])
        ground_truth = to_jsonable(row.get("reward_model", {}).get("ground_truth", {}))
        ground_truth.setdefault("query", query)
        ground_truth.setdefault("target", [])
        out.append(
            {
                "agent_name": "agentic_rl_searchqa",
                "data_source": "deepfactcite",
                "task_type": "long_fact_qa",
                "query": query,
                "answer": "",
                "prompt": prompt,
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {"task_type": "long_fact_qa", "query": query, "source": "deepfactcite", "source_index": int(i)},
            }
        )
    return out


def convert_guardrail(path: str, limit: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(path).head(limit)
    out = []
    for i, row in df.iterrows():
        query = str(row.get("question", ""))
        targets = to_jsonable(row.get("reward_model", {}).get("ground_truth", {}).get("target", []))
        answer = primary_answer(targets)
        prompt = convert_prompt(row["prompt"])
        out.append(
            {
                "agent_name": "agentic_rl_searchqa",
                "data_source": str(row.get("data_source", "searchr1_guardrail")),
                "task_type": "short_qa",
                "query": query,
                "answer": answer,
                "prompt": prompt,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"query": query, "target": targets, "answer": answer, "task_type": "short_qa"},
                },
                "extra_info": {
                    "task_type": "short_qa",
                    "query": query,
                    "answer": answer,
                    "target": targets,
                    "source": "searchr1_core",
                    "source_index": int(i),
                },
            }
        )
    return out


def convert_prompt(value: Any) -> list[dict[str, str]]:
    value = to_jsonable(value)
    if not isinstance(value, list):
        value = [{"role": "user", "content": str(value)}]
    out = []
    for item in value:
        role = str(item.get("role", "user")) if isinstance(item, dict) else "user"
        content = str(item.get("content", item)) if isinstance(item, dict) else str(item)
        content = re.sub(r"<\s*/?\s*search\s*>", lambda m: m.group(0).lower().replace("search", "google_search"), content, flags=re.I)
        content = re.sub(r"<\s*information\s*>", "<tool_response>", content, flags=re.I)
        content = re.sub(r"<\s*/\s*information\s*>", "</tool_response>", content, flags=re.I)
        content = content.replace("<search> query </search>", "<google_search> query </google_search>")
        content = content.replace("<information> and </information>", "<tool_response> and </tool_response>")
        out.append({"role": role, "content": content})
    return out


def convert_corpus(src: str, dst: Path) -> None:
    with Path(src).open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            contents = str(obj.get("contents") or obj.get("text") or "")
            title = ""
            text = contents
            parts = contents.splitlines()
            if parts:
                title = parts[0].strip().strip('"')
            url = str(obj.get("url") or "")
            text_match = re.search(r"Text:\s*(.*)", contents, flags=re.S)
            if text_match:
                text = text_match.group(1).strip()
            url_match = re.search(r"URL:\s*(\S+)", contents)
            if not url and url_match:
                url = url_match.group(1).strip()
            out.write(json.dumps({"title": title, "url": url, "text": text, "keywords": keywords(title, text)}, ensure_ascii=False) + "\n")


def keywords(*texts: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", " ".join(texts).lower())
    seen = []
    for tok in toks:
        if len(tok) <= 2 and tok.isascii():
            continue
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= 32:
            break
    return seen


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    return value


def primary_answer(value: Any) -> str:
    value = to_jsonable(value)
    if isinstance(value, list):
        return " / ".join(str(item) for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    main()
