#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from transformers import AutoTokenizer


SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.S | re.I)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
STOP_MARKERS = ["</search>", "</answer>"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Search-R1-style answer/search guardrail eval through vLLM.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--retriever-url", default="")
    parser.add_argument("--no-search", action="store_true", help="Do not call a retriever; append empty observations.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    if not args.no_search and not args.retriever_url:
        raise ValueError("Set --retriever-url for Search-R1 eval, or pass --no-search for a backend smoke check.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
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
            rollout, runtime = run_rollout(args, prompt)
            score = score_rollout(rollout, ground_truth, runtime, args.max_turns)
            score["response_tokens"] = len(tokenizer(rollout, add_special_tokens=False)["input_ids"])
            score["data_source"] = str(row.get("data_source", "unknown"))
            metrics.append(score)
            f.write(
                json.dumps(
                    {
                        "index": int(idx),
                        "id": to_jsonable(row.get("id")),
                        "data_source": row.get("data_source"),
                        "question": to_jsonable(row.get("question")),
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
    report["retriever_url"] = args.retriever_url
    report["no_search"] = bool(args.no_search)
    report["topk"] = args.topk
    report["max_turns"] = args.max_turns
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


def build_prompt_text(tokenizer, value: Any) -> str:
    value = to_jsonable(value)
    if isinstance(value, list):
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(value, add_generation_prompt=True, tokenize=False)
        return "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in value)
    return str(value)


def run_rollout(args: argparse.Namespace, prompt: str) -> tuple[str, dict[str, Any]]:
    rollout = ""
    handled_searches = 0
    invalid_turns = 0
    retrieved_docs = 0
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
            invalid_turns += 1
            break
        query = searches[-1].strip()
        handled_searches = len(searches)
        if args.no_search:
            observation = ""
        else:
            docs = retrieve(args.retriever_url, query, args.topk, args.timeout)
            retrieved_docs += len(docs)
            observation = passages_to_string(docs)
        rollout += f"\n\n<information>{observation.strip()}</information>\n\n"
    return rollout, {"search_turns": handled_searches, "invalid_turns": invalid_turns, "retrieved_docs": retrieved_docs}


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
            choice = resp.json()["choices"][0]
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


def retrieve(url: str, query: str, topk: int, timeout: float) -> list[dict[str, Any]]:
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("result", [[]])[0]


def passages_to_string(retrieval_result: list[dict[str, Any]]) -> str:
    formatted = ""
    for idx, doc_item in enumerate(retrieval_result):
        document = doc_item.get("document", doc_item)
        content = str(document.get("contents") or document.get("text") or "")
        lines = content.split("\n")
        title = lines[0] if lines else document.get("title", "")
        body_lines = lines[1:] if len(lines) > 1 else []
        url = document.get("url") or document.get("source_url") or document.get("id") or f"doc://{idx + 1}"
        text_lines = []
        for line in body_lines:
            stripped = line.strip()
            if stripped.lower().startswith("url:"):
                url = stripped.split(":", 1)[1].strip() or url
            elif stripped.lower().startswith("text:"):
                text_lines.append(stripped.split(":", 1)[1].strip())
            else:
                text_lines.append(line)
        text = "\n".join(text_lines).strip() or content
        formatted += f"Doc {idx + 1}(Title: {title})\nURL: {url}\nText: {text}\n"
    return formatted


def score_rollout(rollout: str, ground_truth: dict[str, Any], runtime: dict[str, Any], max_turns: int) -> dict[str, Any]:
    answer = extract_answer(rollout)
    targets = ground_truth.get("target")
    search_turns = int(runtime.get("search_turns", 0))
    answer_present = answer is not None and bool(answer.strip())
    return {
        "answer": answer,
        "answer_exact": exact_match(answer or "", targets),
        "answer_subem": subem_match(answer or "", targets),
        "answer_presence": 1.0 if answer_present else 0.0,
        "search_success": 1.0 if search_turns > 0 else 0.0,
        "search_turns": float(search_turns),
        "invalid_turns": float(runtime.get("invalid_turns", 0)),
        "retrieved_docs": float(runtime.get("retrieved_docs", 0)),
        "empty_answer_rate": 0.0 if answer_present else 1.0,
        "budget_fail_rate": 1.0 if (not answer_present and search_turns >= max_turns) else 0.0,
        "no_action_rate": 1.0 if (not answer_present and search_turns == 0) else 0.0,
    }


def extract_answer(text: str) -> str | None:
    matches = list(ANSWER_RE.finditer(text or ""))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def exact_match(prediction: str, targets: Any) -> float:
    pred = normalize_answer(prediction)
    return 1.0 if any(pred == normalize_answer(target) for target in iter_targets(targets)) else 0.0


def subem_match(prediction: str, targets: Any) -> float:
    pred = normalize_answer(prediction)
    return 1.0 if any(normalize_answer(target) in pred for target in iter_targets(targets)) else 0.0


def iter_targets(targets: Any) -> list[str]:
    targets = to_jsonable(targets)
    if targets is None:
        return []
    if isinstance(targets, str):
        return [targets]
    if isinstance(targets, list):
        return [str(item) for item in targets]
    return [str(targets)]


def normalize_answer(text: Any) -> str:
    exclude = set(string.punctuation)
    text = "".join(ch for ch in str(text or "").lower() if ch not in exclude)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "answer_exact",
        "answer_subem",
        "answer_presence",
        "search_success",
        "search_turns",
        "invalid_turns",
        "retrieved_docs",
        "empty_answer_rate",
        "budget_fail_rate",
        "no_action_rate",
        "response_tokens",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in metrics]
        out[key] = sum(values) / len(values) if values else 0.0
    by_source: dict[str, Any] = {}
    for source in sorted({str(row.get("data_source", "unknown")) for row in metrics}):
        subset = [row for row in metrics if str(row.get("data_source", "unknown")) == source]
        by_source[source] = {key: sum(float(row.get(key, 0.0)) for row in subset) / len(subset) for key in keys}
        by_source[source]["rows"] = len(subset)
    out["by_source"] = by_source
    return out


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
