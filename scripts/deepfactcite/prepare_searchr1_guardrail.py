#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic Search-R1 core guardrail subsets from the official nq_hotpotqa_train test parquet."
    )
    parser.add_argument("--input", default="data/nq_hotpotqa_train/test.parquet")
    parser.add_argument("--output-dir", default="data/searchr1_core_guardrail")
    parser.add_argument("--data-sources", default="nq,hotpotqa")
    parser.add_argument("--per-source", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--sample-mode", choices=["random", "first"], default="random")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    sources = [item.strip() for item in args.data_sources.split(",") if item.strip()]
    if not sources:
        raise ValueError("--data-sources must contain at least one source")

    selected_frames = []
    source_stats: dict[str, dict[str, Any]] = {}
    for source in sources:
        part = df[df["data_source"] == source].copy()
        available = len(part)
        if available == 0:
            raise ValueError(f"No rows found for data_source={source!r}")
        if args.per_source > 0 and available > args.per_source:
            if args.sample_mode == "random":
                part = part.sample(n=args.per_source, random_state=args.seed).sort_index()
            else:
                part = part.head(args.per_source)
        source_stats[source] = {"available": available, "selected": len(part)}
        selected_frames.append(part)

    out = pd.concat(selected_frames, ignore_index=True)
    out["extra_info"] = out.apply(lambda row: enrich_extra_info(row, args.seed, args.sample_mode), axis=1)

    parquet_path = output_dir / "test.parquet"
    jsonl_path = output_dir / "test.jsonl"
    summary_path = output_dir / "summary.json"
    out.to_parquet(parquet_path)
    write_jsonl(jsonl_path, out)

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "rows": len(out),
        "data_sources": sources,
        "per_source": args.per_source,
        "seed": args.seed,
        "sample_mode": args.sample_mode,
        "source_stats": source_stats,
        "columns": list(out.columns),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def enrich_extra_info(row: pd.Series, seed: int, sample_mode: str) -> dict[str, Any]:
    extra = to_jsonable(row.get("extra_info") or {})
    if not isinstance(extra, dict):
        extra = {"raw_extra_info": extra}
    extra.update(
        {
            "searchr1_guardrail": True,
            "source_id": to_jsonable(row.get("id")),
            "source_data_source": to_jsonable(row.get("data_source")),
            "sample_seed": seed,
            "sample_mode": sample_mode,
        }
    )
    return extra


def write_jsonl(path: Path, df: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


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
