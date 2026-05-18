from __future__ import annotations

from deepfactcite.reward import DeepFactCiteWeights, compute_score as _compute_score


def compute_score(
    solution_str,
    ground_truth,
    answer_weight=0.25,
    citation_weight=0.35,
    support_weight=0.25,
    format_weight=0.10,
    search_weight=0.05,
    cost_weight=0.05,
    max_searches=4,
    use_judge=False,
):
    weights = DeepFactCiteWeights(
        answer=answer_weight,
        citation=citation_weight,
        support=support_weight,
        format=format_weight,
        search=search_weight,
        cost=cost_weight,
    )
    return _compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        weights=weights,
        max_searches=max_searches,
        use_judge=use_judge,
    )
