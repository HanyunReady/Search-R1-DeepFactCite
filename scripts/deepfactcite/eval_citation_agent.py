#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepfactcite.retriever_server import LexicalRetriever
from deepfactcite.reward import explain_score

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.S | re.I)
ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.S | re.I)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Search-R1-style citation-agent evaluation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    if args.adapter:
        disable_peft_torchao_dispatcher()
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    retriever = LexicalRetriever(args.corpus)
    dataset = pd.read_parquet(args.data)
    if args.limit:
        dataset = dataset.head(args.limit)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in dataset.iterrows():
            prompt = build_prompt_text(tokenizer, row["prompt"])
            ground_truth = row.get("reward_model", {}).get("ground_truth", {})
            rollout, search_turns = run_rollout(
                model=model,
                tokenizer=tokenizer,
                retriever=retriever,
                prompt=prompt,
                topk=args.topk,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
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
    report["adapter"] = args.adapter
    report["data"] = args.data
    report["corpus"] = args.corpus
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_prompt_text(tokenizer, value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(value, add_generation_prompt=True, tokenize=False)
        return "\n".join(str(item.get("content", item)) if isinstance(item, dict) else str(item) for item in value)
    return str(value)


def disable_peft_torchao_dispatcher() -> None:
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.torchao as peft_lora_torchao

        peft_import_utils.is_torchao_available = lambda: False
        peft_lora_torchao.is_torchao_available = lambda: False
    except Exception:
        pass


@torch.inference_mode()
def run_rollout(
    model,
    tokenizer,
    retriever: LexicalRetriever,
    prompt: str,
    topk: int,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, int]:
    rollout = ""
    handled_searches = 0
    for _ in range(max_turns):
        current = prompt + rollout
        inputs = tokenizer(current, return_tensors="pt", add_special_tokens=False).to(model.device)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.95})
        else:
            generate_kwargs.update({"do_sample": False})
        output = model.generate(**inputs, **generate_kwargs)
        new_text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        new_text = truncate_to_action(new_text)
        rollout += new_text
        if ANSWER_RE.search(rollout):
            break
        searches = SEARCH_RE.findall(rollout)
        if len(searches) <= handled_searches:
            break
        query = searches[-1].strip()
        handled_searches = len(searches)
        rollout += format_observation(retriever.search(query, topk))
    return rollout, handled_searches


def truncate_to_action(text: str) -> str:
    lower = text.lower()
    candidates = []
    for marker in ["</search>", "</answer>"]:
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
