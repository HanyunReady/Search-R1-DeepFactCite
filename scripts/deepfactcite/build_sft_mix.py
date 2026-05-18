#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepfactcite.reward import explain_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mixed soft+strict DeepFactCite SFT data.")
    parser.add_argument("--soft-dir", default="data/deepfactcite/sft")
    parser.add_argument("--strict-dir", default="data/deepfactcite_strict/sft")
    parser.add_argument("--output-dir", default="data/deepfactcite_mix/sft")
    parser.add_argument("--strict-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    soft_train = read_rows(Path(args.soft_dir) / "train.parquet", "soft")
    strict_train = read_rows(Path(args.strict_dir) / "train.parquet", "strict")
    soft_test = read_rows(Path(args.soft_dir) / "test.parquet", "soft")
    strict_test = read_rows(Path(args.strict_dir) / "test.parquet", "strict")

    train = dedupe_rows(soft_train + strict_train * args.strict_repeat)
    test = dedupe_rows(soft_test + strict_test)
    random.shuffle(train)
    random.shuffle(test)

    pd.DataFrame(train).to_parquet(output / "train.parquet")
    pd.DataFrame(test).to_parquet(output / "test.parquet")
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "test.jsonl", test)

    summary = {
        "output_dir": str(output),
        "soft_train": len(soft_train),
        "strict_train": len(strict_train),
        "strict_repeat": args.strict_repeat,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_metrics": summarize(train),
        "test_metrics": summarize(test),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_rows(path: Path, source: str) -> list[dict[str, Any]]:
    rows = pd.read_parquet(path).to_dict("records")
    for row in rows:
        row["mix_source"] = source
    return rows


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = {}
    for row in rows:
        key = (row.get("query", ""), row.get("answer", ""))
        if key not in deduped or row.get("mix_source") == "strict":
            deduped[key] = row
    return list(deduped.values())


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = [explain_score(row["answer"], {"target": []}) for row in rows]
    summary: dict[str, float] = {"rows": float(len(rows))}
    for key in [
        "format",
        "search",
        "url_validity",
        "citation_presence",
        "claim_support",
        "unsupported_citation_rate",
        "fake_url_rate",
        "citation_count",
    ]:
        values = [float(item.get(key, 0.0)) for item in metrics]
        summary[key] = sum(values) / len(values) if values else 0.0
    summary["strict_rows"] = float(sum(1 for row in rows if row.get("mix_source") == "strict"))
    summary["soft_rows"] = float(sum(1 for row in rows if row.get("mix_source") == "soft"))
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


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


if __name__ == "__main__":
    main()
