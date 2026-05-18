#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_KEYS = [
    "total",
    "answer_subem",
    "format",
    "search",
    "url_validity",
    "citation_precision",
    "claim_support",
    "unsupported_citation_rate",
    "fake_url_rate",
    "citation_count",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DeepFactCite GRPO rollout JSONL files.")
    parser.add_argument("rollout_dir")
    parser.add_argument("--out", default="")
    parser.add_argument("--failure-samples", type=int, default=12)
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    rows = read_rollouts(rollout_dir)
    if not rows:
        raise SystemExit(f"No rollout rows found under {rollout_dir}")

    summary = summarize(rows, args.failure_samples)
    text = render_markdown(rollout_dir, summary)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def read_rollouts(rollout_dir: Path) -> list[dict[str, Any]]:
    files = sorted(rollout_dir.glob("rollout_data_step_*.jsonl"), key=step_number)
    rows: list[dict[str, Any]] = []
    for file in files:
        step = step_number(file)
        with file.open(encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_step"] = step
                rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], failure_samples: int) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    for key in METRIC_KEYS:
        values = [float(row.get("details", {}).get(key, row.get("reward", 0.0))) for row in rows if has_metric(row, key)]
        metrics[key] = mean(values) if values else 0.0

    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["_step"])].append(row)

    step_metrics = []
    for step, step_rows in sorted(by_step.items()):
        step_metrics.append(
            {
                "step": step,
                "samples": len(step_rows),
                "reward": mean(float(row.get("reward", 0.0)) for row in step_rows),
                "url_validity": metric_mean(step_rows, "url_validity"),
                "claim_support": metric_mean(step_rows, "claim_support"),
                "unsupported_citation_rate": metric_mean(step_rows, "unsupported_citation_rate"),
                "citation_count": metric_mean(step_rows, "citation_count"),
            }
        )

    failures = []
    for row in rows:
        details = row.get("details", {})
        reasons = []
        if float(details.get("search", 0.0)) <= 0:
            reasons.append("no_search")
        if float(details.get("citation_count", 0.0)) <= 0:
            reasons.append("no_citation")
        if float(details.get("url_validity", 0.0)) < 1.0 and float(details.get("citation_count", 0.0)) > 0:
            reasons.append("invalid_or_fake_url")
        if float(details.get("claim_support", 0.0)) < 0.5 and float(details.get("citation_count", 0.0)) > 0:
            reasons.append("weak_claim_support")
        if float(details.get("unsupported_citation_rate", 0.0)) >= 0.5 and float(details.get("citation_count", 0.0)) > 0:
            reasons.append("unsupported_citation")
        if not reasons:
            continue
        failures.append(
            {
                "step": row["_step"],
                "query": row.get("query", ""),
                "reward": float(row.get("reward", 0.0)),
                "reasons": reasons,
                "details": {key: details.get(key) for key in METRIC_KEYS if key in details},
                "answer": str(details.get("answer", ""))[:900],
            }
        )

    reason_counts = Counter(reason for item in failures for reason in item["reasons"])
    failures.sort(key=lambda item: (item["reward"], item["step"]))
    return {
        "samples": len(rows),
        "steps": len(by_step),
        "metrics": metrics,
        "step_metrics": step_metrics,
        "failure_reason_counts": dict(reason_counts.most_common()),
        "failure_samples": failures[:failure_samples],
    }


def render_markdown(rollout_dir: Path, summary: dict[str, Any]) -> str:
    lines = [
        f"# GRPO Rollout Summary: `{rollout_dir}`",
        "",
        f"- Samples: {summary['samples']}",
        f"- Steps: {summary['steps']}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Mean |",
        "|---|---:|",
    ]
    for key in METRIC_KEYS:
        lines.append(f"| {key} | {summary['metrics'].get(key, 0.0):.4f} |")

    lines.extend(["", "## Step Metrics", "", "| Step | Samples | Reward | URL | Support | Unsupported | Citations |", "|---:|---:|---:|---:|---:|---:|---:|"])
    for item in summary["step_metrics"]:
        lines.append(
            "| {step} | {samples} | {reward:.4f} | {url_validity:.4f} | {claim_support:.4f} | "
            "{unsupported_citation_rate:.4f} | {citation_count:.4f} |".format(**item)
        )

    lines.extend(["", "## Failure Reasons", "", "| Reason | Count |", "|---|---:|"])
    for reason, count in summary["failure_reason_counts"].items():
        lines.append(f"| {reason} | {count} |")

    lines.extend(["", "## Failure Samples", ""])
    for item in summary["failure_samples"]:
        lines.append(f"- step={item['step']} reward={item['reward']:.4f} reasons={','.join(item['reasons'])}")
        lines.append(f"  query: {item['query']}")
        lines.append(f"  details: `{json.dumps(item['details'], ensure_ascii=False)}`")
        answer = re.sub(r"\s+", " ", item["answer"]).strip()
        if answer:
            lines.append(f"  answer: {answer}")
    lines.append("")
    return "\n".join(lines)


def has_metric(row: dict[str, Any], key: str) -> bool:
    if key == "total":
        return "reward" in row or "total" in row.get("details", {})
    return key in row.get("details", {})


def metric_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get("details", {}).get(key, 0.0)) for row in rows if key in row.get("details", {})]
    return mean(values) if values else 0.0


def step_number(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


if __name__ == "__main__":
    main()
