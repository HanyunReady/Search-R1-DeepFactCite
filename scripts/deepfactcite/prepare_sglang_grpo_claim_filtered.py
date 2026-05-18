#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepfactcite.reward import (  # noqa: E402
    _canonical_url,
    _citation_claims,
    _overlap_support_score,
    _tokenize,
    convert_deepcite_tags,
    extract_answer,
)


SNIPPET_RE = re.compile(r"<snippet\s+id\s*=?\s*[\"']?([^\"'>\s]+)[\"']?\s*>(.*?)</snippet>", re.S | re.I)
URL_RE = re.compile(r"^https?://", re.I)
LOW_INFORMATION_CLAIM_RE = re.compile(
    r"\b("
    r"real examples?|learn how|this article|this page|the source|overview|details?|"
    r"click here|read more|examples? of|guide to|introduction to"
    r")\b",
    re.I,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build claim-level support-filtered DeepFactCite GRPO data for the SGLang backend."
    )
    parser.add_argument("--sft-train", default="data/deepfactcite_mix/sft/train.parquet")
    parser.add_argument("--out-dir", default="data/deepfactcite_sglang_grpo_claim_filtered")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-citations", type=int, default=2)
    parser.add_argument("--min-citations", type=int, default=1)
    parser.add_argument("--min-support", type=float, default=1.0)
    parser.add_argument("--min-claim-tokens", type=int, default=4)
    parser.add_argument("--max-claim-tokens", type=int, default=42)
    parser.add_argument("--max-tool-text-chars", type=int, default=700)
    parser.add_argument("--test-rows", type=int, default=4)
    parser.add_argument("--reject-preview", type=int, default=80)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.sft_train)
    order = list(range(len(df)))
    random.shuffle(order)

    rows: list[dict[str, Any]] = []
    corpus_docs: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    reject_rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    reasons: Counter[str] = Counter()
    seen = 0

    for source_index in order:
        if len(rows) >= args.rows:
            break
        seen += 1
        row = df.iloc[source_index]
        query = str(row.get("query") or "").strip()
        response = convert_deepcite_tags(str(row.get("answer") or ""))
        candidate, reason = build_candidate(query, response, int(source_index), args)
        if candidate is None:
            reasons[reason] += 1
            if len(reject_rows) < args.reject_preview:
                reject_rows.append({"source_index": int(source_index), "query": query, "reason": reason})
            continue

        rows.append(candidate["row"])
        preview_rows.append(candidate["preview"])
        for doc in candidate["docs"]:
            canonical = _canonical_url(doc["url"])
            if canonical in seen_urls:
                continue
            seen_urls.add(canonical)
            corpus_docs.append(doc)

    if not rows:
        raise RuntimeError("No claim-filtered rows were produced. Relax thresholds or inspect reject_samples.jsonl.")

    train_rows, test_rows = split_rows(rows, args.test_rows)
    write_table(out_dir / "train.parquet", train_rows)
    write_table(out_dir / "test.parquet", test_rows or train_rows[:1])
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows or train_rows[:1])
    write_jsonl(out_dir / "corpus.jsonl", corpus_docs)
    write_jsonl(out_dir / "preview.jsonl", preview_rows)
    write_jsonl(out_dir / "reject_samples.jsonl", reject_rows)

    citation_counts = [len(row["extra_info"]["selected_claims"]) for row in rows]
    support_scores = [
        claim["support_score"] for row in rows for claim in row["extra_info"]["selected_claims"]
    ]
    summary = {
        "source_sft": args.sft_train,
        "out_dir": str(out_dir),
        "seen_until_kept": seen,
        "kept_rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows or train_rows[:1]),
        "corpus_docs": len(corpus_docs),
        "reject_reasons": dict(reasons.most_common()),
        "filters": {
            "min_citations": args.min_citations,
            "max_citations": args.max_citations,
            "min_support": args.min_support,
            "min_claim_tokens": args.min_claim_tokens,
            "max_claim_tokens": args.max_claim_tokens,
            "max_tool_text_chars": args.max_tool_text_chars,
        },
        "metrics": {
            "avg_selected_citations": sum(citation_counts) / len(citation_counts),
            "min_support_score": min(support_scores),
            "avg_support_score": sum(support_scores) / len(support_scores),
        },
        "files": {
            "train": str(out_dir / "train.parquet"),
            "test": str(out_dir / "test.parquet"),
            "corpus": str(out_dir / "corpus.jsonl"),
            "preview": str(out_dir / "preview.jsonl"),
            "reject_samples": str(out_dir / "reject_samples.jsonl"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_candidate(query: str, response: str, source_index: int, args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    if not query:
        return None, "missing_query"
    answer = extract_answer(response) or ""
    if not answer:
        return None, "missing_answer"

    snippets = parse_snippets(response, args.max_tool_text_chars)
    by_url = {_canonical_url(snip["url"]): snip for snip in snippets if snip["url"]}
    if not by_url:
        return None, "no_retrieved_snippets"

    citations = _citation_claims(answer)
    if not citations:
        return None, "no_citations"

    selected: list[dict[str, Any]] = []
    bad_missing_url = False
    bad_weak = False
    bad_broad = False
    used_urls: set[str] = set()
    for citation in citations:
        url = _canonical_url(citation["url"])
        claim = clean_claim(citation["claim"])
        if not URL_RE.search(url) or "(truncated)" in url or "..." in url:
            bad_missing_url = True
            continue
        doc = by_url.get(url)
        if not doc:
            bad_missing_url = True
            continue
        claim_tokens = _tokenize(claim)
        if len(claim_tokens) < args.min_claim_tokens or is_low_information_claim(claim):
            continue
        if len(claim_tokens) > args.max_claim_tokens:
            bad_broad = True
            continue
        support_score = _overlap_support_score(claim, f"{doc['title']} {doc['text']}")
        if support_score < args.min_support:
            bad_weak = True
            continue
        if url in used_urls:
            continue
        used_urls.add(url)
        selected.append(
            {
                "label": citation["label"],
                "url": url,
                "claim": claim,
                "support_score": float(support_score),
                "title": doc["title"],
                "evidence": doc["text"],
            }
        )
        if len(selected) >= args.max_citations:
            break

    if len(selected) < args.min_citations:
        if bad_missing_url:
            return None, "citation_url_not_from_evidence_or_truncated"
        if bad_broad:
            return None, "claim_too_broad"
        if bad_weak:
            return None, "weak_claim_support"
        return None, "too_few_supported_claims"

    reference_answer = " ".join(f"{claim['claim']} [{claim['label']}]({claim['url']})." for claim in selected)
    target_claims = [claim["claim"] for claim in selected]
    row = {
        "agent_name": "agentic_rl_searchqa",
        "data_source": "deepfactcite_claim_filtered",
        "task_type": "long_fact_qa",
        "query": query,
        "answer": " ".join(target_claims),
        "prompt": [make_prompt(query)],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "query": query,
                "target": target_claims,
                "reference_answer": reference_answer,
                "task_type": "long_fact_qa",
            },
        },
        "extra_info": {
            "task_type": "long_fact_qa",
            "query": query,
            "answer": " ".join(target_claims),
            "target": target_claims,
            "source": "deepfactcite_claim_filtered",
            "source_index": source_index,
            "selected_claims": [
                {
                    "claim": claim["claim"],
                    "url": claim["url"],
                    "title": claim["title"],
                    "support_score": claim["support_score"],
                }
                for claim in selected
            ],
            "reference_answer": reference_answer,
        },
    }
    docs = [
        {
            "title": claim["title"],
            "url": claim["url"],
            "text": claim["evidence"],
            "keywords": keywords(query, claim["claim"], claim["title"], claim["evidence"]),
        }
        for claim in selected
    ]
    preview = {
        "source_index": source_index,
        "query": query,
        "reference_answer": reference_answer,
        "selected_claims": row["extra_info"]["selected_claims"],
        "docs": docs,
    }
    return {"row": row, "docs": docs, "preview": preview}, "kept"


def make_prompt(query: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Answer the question in 1 to 3 concise sentences. You must first search using "
            "<google_search> query </google_search>. Use 1 to 2 markdown citations "
            "[short label](URL), placed immediately after the claim they support. Every URL "
            "must be copied exactly from the tool response, and every cited claim must be "
            "directly supported by the cited snippet. If the evidence is not enough, say so "
            "instead of broadening the claim. Finish with the final answer inside <answer> "
            "and </answer>.\n"
            f"Question: {query}\n"
        ),
    }


def parse_snippets(text: str, max_text_chars: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for _, body in SNIPPET_RE.findall(text or ""):
        title = field(body, "Title")
        url = field(body, "URL")
        snippet_text = field(body, "Text", multiline=True)
        if not url or not snippet_text:
            continue
        out.append(
            {
                "title": compact(title, 240),
                "url": compact(url, 500),
                "text": compact(snippet_text, max_text_chars),
            }
        )
    return out


def field(text: str, name: str, multiline: bool = False) -> str:
    flags = re.S if multiline else 0
    match = re.search(rf"{re.escape(name)}:\s*(.*)", text or "", flags=flags)
    return compact(match.group(1)) if match else ""


def clean_claim(text: str) -> str:
    text = re.sub(r"\[[^\[\]\n]+\]\([^)\s]+\)", "", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n-:;,")


def is_low_information_claim(claim: str) -> bool:
    if LOW_INFORMATION_CLAIM_RE.search(claim or ""):
        return True
    tokens = _tokenize(claim)
    if len(tokens) < 6:
        return True
    has_predicate = bool(
        re.search(
            r"\b("
            r"is|are|was|were|be|being|been|has|have|had|can|could|may|might|"
            r"will|would|does|do|did|suggests?|shows?|finds?|found|includes?|"
            r"uses?|used|requires?|required|causes?|caused|increases?|increased|"
            r"reduces?|reduced|became|becomes?|won|served|published|began|ended"
            r")\b",
            claim,
            re.I,
        )
    )
    has_number = bool(re.search(r"\b\d{2,}\b", claim))
    return not (has_predicate or has_number)


def compact(text: str, max_chars: int | None = None) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if max_chars is not None:
        text = text[:max_chars].rstrip()
    return text


def keywords(*texts: str) -> list[str]:
    seen: list[str] = []
    for tok in _tokenize(" ".join(texts)):
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= 64:
            break
    return seen


def split_rows(rows: list[dict[str, Any]], test_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if test_rows <= 0 or len(rows) <= 1:
        return rows, rows[:1]
    n_test = min(test_rows, max(1, len(rows) // 5), len(rows) - 1)
    return rows[:-n_test], rows[-n_test:]


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
