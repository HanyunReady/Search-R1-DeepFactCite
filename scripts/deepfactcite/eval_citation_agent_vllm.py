#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepfactcite.retriever_server import LexicalRetriever
from deepfactcite.reward import explain_score

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.S | re.I)
ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.S | re.I)
STOP_MARKERS = ["</search>", "</answer>"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vLLM/OpenAI Search-R1-style citation-agent evaluation.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    retriever = LexicalRetriever(args.corpus)
    dataset = pd.read_parquet(args.data)
    if args.limit:
        dataset = dataset.head(args.limit)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_indices(output_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    metrics: list[dict[str, Any]] = load_existing_metrics(output_path) if args.resume else []

    with output_path.open(mode, encoding="utf-8") as f:
        for idx, row in dataset.iterrows():
            if int(idx) in done:
                continue
            prompt = build_prompt_text(tokenizer, row["prompt"])
            ground_truth = to_jsonable(row.get("reward_model", {}).get("ground_truth", {}))
            rollout, search_turns = run_rollout(
                args=args,
                retriever=retriever,
                prompt=prompt,
            )
            score = explain_score(rollout, ground_truth)
            score["search_turns"] = search_turns
            score["response_tokens"] = len(tokenizer(rollout, add_special_tokens=False)["input_ids"])
            metrics.append(score)
            f.write(
                json.dumps(
                    {
                        "index": int(idx),
                        "query": row.get("query") or row.get("extra_info", {}).get("query"),
                        "prompt": prompt,
                        "rollout": rollout,
                        "score": score,
                        "ground_truth": ground_truth,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()

    report = aggregate(metrics)
    report["rows"] = len(metrics)
    report["model"] = args.model
    report["data"] = args.data
    report["corpus"] = args.corpus
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_done_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(int(json.loads(line)["index"]))
    return done


def load_existing_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    metrics = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line)["score"])
    return metrics


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


def build_prompt_text(tokenizer, value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(value, add_generation_prompt=True, tokenize=False)
        return "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in value)
    return str(value)


def run_rollout(args: argparse.Namespace, retriever: LexicalRetriever, prompt: str) -> tuple[str, int]:
    rollout = ""
    handled_searches = 0
    for _ in range(args.max_turns):
        generated = complete(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt + rollout,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        rollout += generated
        if ANSWER_RE.search(rollout):
            break
        searches = SEARCH_RE.findall(rollout)
        if len(searches) <= handled_searches:
            break
        query = searches[-1].strip()
        handled_searches = len(searches)
        rollout += format_observation(retriever.search(query, args.topk))
    return rollout, handled_searches


def complete(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": STOP_MARKERS,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    last_error = None
    for _ in range(3):
        try:
            resp = requests.post(f"{base_url.rstrip('/')}/completions", headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            text = choice.get("text") or ""
            stop_reason = choice.get("stop_reason")
            if stop_reason in STOP_MARKERS and not text.endswith(stop_reason):
                text += stop_reason
            return truncate_to_action(text)
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"vLLM completion failed for model={model}: {last_error}")


def truncate_to_action(text: str) -> str:
    lower = text.lower()
    candidates = []
    for marker in STOP_MARKERS:
        pos = lower.find(marker)
        if pos >= 0:
            candidates.append(pos + len(marker))
    if not candidates:
        return text
    return text[: min(candidates)]


def format_observation(results: list[dict[str, Any]]) -> str:
    lines = ["\n<information>"]
    for item in results:
        doc = item["document"]
        lines.append(f"<snippet id={doc.get('id', '')}>")
        lines.append(doc.get("contents", ""))
        lines.append("</snippet>")
    lines.append("</information>\n")
    return "\n".join(lines)


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "total",
        "answer_subem",
        "citation_presence",
        "url_validity",
        "citation_precision",
        "claim_support",
        "unsupported_citation_rate",
        "fake_url_rate",
        "search_turns",
        "response_tokens",
    ]
    out = {}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in metrics]
        out[key] = sum(values) / len(values) if values else 0.0
    return out


if __name__ == "__main__":
    main()
