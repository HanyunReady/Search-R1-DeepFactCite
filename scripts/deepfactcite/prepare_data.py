#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepfactcite.prompts import make_searchr1_prompt
from deepfactcite.reward import convert_deepcite_tags, explain_score


SNIPPET_RE = re.compile(r'<snippet[^>]*>(.*?)</snippet>', re.S | re.I)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DeepFactCite data for Search-R1 SFT and GRPO.")
    parser.add_argument("--deepcitefact-dir", default="/root/autodl-tmp/DeepCiteFact")
    parser.add_argument("--output-dir", default="data/deepfactcite")
    parser.add_argument("--sft-input", default="data/sft_trace_filter.jsonl")
    parser.add_argument("--rl-input", default="data/rl_train_data_filter_grpo.jsonl")
    parser.add_argument("--max-sft", type=int, default=1200)
    parser.add_argument("--max-rl", type=int, default=1200)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-sft-url-validity", type=float, default=1.0)
    parser.add_argument("--min-sft-claim-support", type=float, default=0.35)
    parser.add_argument("--max-sft-unsupported-rate", type=float, default=0.5)
    parser.add_argument("--min-sft-citations", type=int, default=1)
    args = parser.parse_args()

    src_root = Path(args.deepcitefact_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    sft_rows, corpus_rows, sft_filter_summary = build_sft(src_root / args.sft_input, args.max_sft, args)
    rl_rows = build_rl(src_root / args.rl_input, args.max_rl)
    if not rl_rows:
        rl_rows = build_rl_from_sft(sft_rows)

    write_split(sft_rows, output / "sft", args.val_ratio)
    write_split(rl_rows, output / "rl", args.val_ratio)
    write_jsonl(output / "corpus.jsonl", corpus_rows)
    summary = {
        "sft_rows": len(sft_rows),
        "rl_rows": len(rl_rows),
        "corpus_rows": len(corpus_rows),
        "sft_filter": sft_filter_summary,
        "output_dir": str(output),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_sft(path: Path, limit: int, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = []
    corpus = []
    seen_docs = set()
    seen = 0
    skipped = 0
    skip_reasons: dict[str, int] = {}
    for item in read_jsonl(path):
        seen += 1
        query = item.get("query", "").strip()
        full_response = item.get("full_response") or item.get("response") or ""
        if not query or not full_response:
            skipped += 1
            skip_reasons["missing_query_or_response"] = skip_reasons.get("missing_query_or_response", 0) + 1
            continue
        response = convert_deepcite_tags(full_response)
        metrics = explain_score(response, {"target": []})
        reason = sft_reject_reason(metrics, args)
        if reason:
            skipped += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        rows.append(
            {
                "prompt": [{"role": "user", "content": make_searchr1_prompt(query)}],
                "answer": response,
                "query": query,
            }
        )
        for doc in extract_corpus_docs(full_response):
            if doc["id"] in seen_docs:
                continue
            seen_docs.add(doc["id"])
            corpus.append(doc)
        if len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError(f"No SFT rows survived filtering from {path}. Relax thresholds or inspect source data.")
    summary = {
        "seen": seen,
        "kept": len(rows),
        "skipped": skipped,
        "skip_reasons": skip_reasons,
        "min_url_validity": args.min_sft_url_validity,
        "min_claim_support": args.min_sft_claim_support,
        "max_unsupported_rate": args.max_sft_unsupported_rate,
        "min_citations": args.min_sft_citations,
    }
    return rows, corpus, summary


def build_rl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for item in read_jsonl(path):
        query = item.get("query", "").strip()
        if not query:
            continue
        rows.append(make_rl_row(query, len(rows), extract_targets(item)))
        if len(rows) >= limit:
            break
    return rows


def build_rl_from_sft(sft_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [make_rl_row(row["query"], idx, []) for idx, row in enumerate(sft_rows)]


def make_rl_row(query: str, idx: int, targets: list[str] | None = None) -> dict[str, Any]:
    targets = targets or []
    return {
        "data_source": "deepfactcite",
        "prompt": [{"role": "user", "content": make_searchr1_prompt(query)}],
        "ability": "deepfactcite",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": targets, "query": query, "answer_scored": bool(targets)},
        },
        "extra_info": {"split": "train", "index": idx, "query": query},
    }


def extract_targets(item: dict[str, Any]) -> list[str]:
    for key in ["target", "targets", "answer", "answers", "gold", "labels"]:
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
    return []


def sft_reject_reason(metrics: dict[str, Any], args: argparse.Namespace) -> str | None:
    if metrics["format"] < 1.0:
        return "bad_format"
    if metrics["search"] <= 0.0:
        return "no_search"
    if metrics["citation_count"] < args.min_sft_citations:
        return "too_few_citations"
    if metrics["url_validity"] < args.min_sft_url_validity:
        return "invalid_url"
    if metrics["claim_support"] < args.min_sft_claim_support:
        return "weak_claim_support"
    if metrics["unsupported_citation_rate"] > args.max_sft_unsupported_rate:
        return "too_many_unsupported_citations"
    return None


def extract_corpus_docs(text: str) -> list[dict[str, str]]:
    docs = []
    for snippet in SNIPPET_RE.findall(text or ""):
        title = extract_line(snippet, "Title") or "Untitled"
        url = extract_line(snippet, "URL")
        body = extract_text(snippet)
        if not url or not body:
            continue
        doc_id = stable_id(url)
        docs.append({"id": doc_id, "url": url, "contents": f"\"{title}\"\nURL: {url}\nText: {body}"})
    return docs


def extract_line(text: str, name: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}:\s*(.*?)\s*$", text, re.M)
    return match.group(1).strip() if match else None


def extract_text(text: str) -> str | None:
    match = re.search(r"^Text:\s*(.*)", text, re.S | re.M)
    return match.group(1).strip() if match else None


def stable_id(value: str) -> str:
    import hashlib

    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


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


def write_split(rows: list[dict[str, Any]], out_dir: Path, val_ratio: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise RuntimeError(f"No rows to write for {out_dir}")
    random.shuffle(rows)
    val_size = max(1, int(len(rows) * val_ratio)) if len(rows) > 1 else 0
    val = rows[:val_size]
    train = rows[val_size:]
    pd.DataFrame(train).to_parquet(out_dir / "train.parquet")
    pd.DataFrame(val or train[:1]).to_parquet(out_dir / "test.parquet")
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "test.jsonl", val or train[:1])


if __name__ == "__main__":
    main()
