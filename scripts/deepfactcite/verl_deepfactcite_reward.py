from __future__ import annotations

import os
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from deepfactcite.reward import DeepFactCiteWeights, compute_score as _compute_score, explain_score

try:
    from verl import DataProto
    from verl.workers.reward_manager import register
    from verl.workers.reward_manager.abstract import AbstractRewardManager
except Exception:  # pragma: no cover - allows importing this file outside the GRPO env.
    DataProto = Any  # type: ignore
    AbstractRewardManager = object  # type: ignore
    register = None  # type: ignore


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any | None = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    solution = solution_str or kwargs.get("response") or kwargs.get("solution_str") or ""
    gt = {}
    if isinstance(ground_truth, dict):
        gt.update(ground_truth)
    elif ground_truth is not None:
        gt["target"] = ground_truth
    if extra_info:
        if "query" in extra_info:
            gt.setdefault("query", extra_info["query"])
        if "answer" in extra_info:
            gt.setdefault("target", extra_info["answer"])
    if "query" in kwargs:
        gt.setdefault("query", kwargs["query"])
    if "answer" in kwargs:
        gt.setdefault("target", kwargs["answer"])

    weights = DeepFactCiteWeights(
        answer=float(os.getenv("DFC_ANSWER_WEIGHT", "0.25")),
        citation=float(os.getenv("DFC_CITATION_WEIGHT", "0.35")),
        support=float(os.getenv("DFC_SUPPORT_WEIGHT", "0.25")),
        format=float(os.getenv("DFC_FORMAT_WEIGHT", "0.10")),
        search=float(os.getenv("DFC_SEARCH_WEIGHT", "0.05")),
        cost=float(os.getenv("DFC_COST_WEIGHT", "0.05")),
    )
    return float(
        _compute_score(
            solution_str=solution,
            ground_truth=gt,
            weights=weights,
            max_searches=int(os.getenv("DFC_MAX_SEARCHES", "4")),
            use_judge=os.getenv("DFC_USE_JUDGE", "false").lower() in {"1", "true", "yes"},
        )
    )


if register is not None:

    @register("deepfactcite_custom")
    class DeepFactCiteRewardManager(AbstractRewardManager):
        """Reward manager with the tuple shape expected by the local SGLang trainer."""

        def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
            self.tokenizer = tokenizer
            self.num_examine = num_examine
            self.compute_score = compute_score or globals()["compute_score"]
            self.reward_fn_key = reward_fn_key

        def __call__(self, data: DataProto, global_steps: int = 0, return_dict: bool = False):
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_extra_info = defaultdict(list)

            primary_rewards: list[float] = []
            format_rewards: list[float] = []
            search_rewards: list[float] = []
            search_nums: list[int] = []
            rollout_rows: list[dict[str, Any]] = []

            for i in range(len(data)):
                data_item = data[i]
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                meta = _extract_meta(data_item)
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                if "trajectory" in data_item.non_tensor_batch:
                    response_str = str(_to_python(data_item.non_tensor_batch["trajectory"]) or "")
                else:
                    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                score = float(
                    self.compute_score(
                        data_source=_to_python(data_item.non_tensor_batch.get(self.reward_fn_key, "")),
                        solution_str=response_str,
                        ground_truth=meta.get("ground_truth", {}),
                        extra_info=meta,
                    )
                )
                details = explain_score(response_str, meta.get("ground_truth", {}))
                primary_rewards.append(float(details.get("answer_subem", 0.0)))
                format_rewards.append(float(details.get("format", 0.0)))
                search_rewards.append(float(details.get("search", 0.0)))
                search_nums.append(_count_searches(response_str))

                valid_len = int(valid_response_length.item() if hasattr(valid_response_length, "item") else valid_response_length)
                if valid_len > 0:
                    reward_tensor[i, valid_len - 1] = score

                reward_extra_info["answer_subem"].append(float(details.get("answer_subem", 0.0)))
                reward_extra_info["url_validity"].append(float(details.get("url_validity", 0.0)))
                reward_extra_info["citation_precision"].append(float(details.get("citation_precision", 0.0)))
                reward_extra_info["claim_support"].append(float(details.get("claim_support", 0.0)))
                reward_extra_info["unsupported_citation_rate"].append(float(details.get("unsupported_citation_rate", 0.0)))

                rollout_rows.append(
                    {
                        "query": meta.get("query", ""),
                        "task_type": meta.get("task_type", ""),
                        "prompt": prompt_str,
                        "response": response_str,
                        "reward": score,
                        "details": details,
                    }
                )

            _dump_rollouts(rollout_rows, global_steps)
            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            return reward_tensor, primary_rewards, format_rewards, search_rewards, search_nums


def _extract_meta(data_item: Any) -> dict[str, Any]:
    non_tensor = data_item.non_tensor_batch
    meta: dict[str, Any] = {}
    for key in ("query", "answer", "task_type"):
        if key in non_tensor:
            meta[key] = _to_python(non_tensor[key])
    extra = _to_python(non_tensor.get("extra_info", {})) if hasattr(non_tensor, "get") else {}
    if isinstance(extra, dict):
        meta.update(extra)
    reward_model = _to_python(non_tensor.get("reward_model", {})) if hasattr(non_tensor, "get") else {}
    ground_truth = reward_model.get("ground_truth", {}) if isinstance(reward_model, dict) else {}
    if isinstance(ground_truth, dict):
        meta.setdefault("ground_truth", ground_truth)
        meta.update({k: v for k, v in ground_truth.items() if k not in meta})
    else:
        meta.setdefault("ground_truth", {"target": ground_truth})
    meta.setdefault("ground_truth", {})
    if "query" in meta:
        meta["ground_truth"].setdefault("query", meta["query"])
    if "target" in meta:
        meta["ground_truth"].setdefault("target", meta["target"])
    if "answer" in meta:
        meta["ground_truth"].setdefault("target", meta["answer"])
    meta.setdefault("task_type", "short_qa")
    return meta


def _to_python(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return _to_python(value.item())
        except Exception:
            return value
    if hasattr(value, "tolist"):
        try:
            return _to_python(value.tolist())
        except Exception:
            return value
    if isinstance(value, dict):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    return value


def _count_searches(text: str) -> int:
    lowered = (text or "").lower()
    return lowered.count("<google_search>") + lowered.count("<search>")


def _dump_rollouts(rows: list[dict[str, Any]], global_steps: int) -> None:
    out_dir = Path(os.getenv("AGENTIC_SEARCHQA_ROLLOUT_DIR", "./rollout_data"))
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"rollout_data_step_{global_steps}.jsonl"
        with path.open("w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return
