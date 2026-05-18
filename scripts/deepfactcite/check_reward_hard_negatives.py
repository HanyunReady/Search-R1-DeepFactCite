#!/usr/bin/env python
from __future__ import annotations

import json

from deepfactcite.reward import explain_score


EVIDENCE = """<think>Need one lookup.</think>
<search>capital of France</search>
<information>
Doc 1(Title: France capital) URL: https://example.org/france
Text: Paris is the capital and largest city of France.
</information>
"""


def main() -> None:
    cases = {
        "supported": EVIDENCE + "<answer>Paris is the capital of France [France capital](https://example.org/france).</answer>",
        "fake_url": EVIDENCE + "<answer>Paris is the capital of France [France capital](https://fake.example/france).</answer>",
        "unsupported": EVIDENCE
        + "<answer>The Moon is made of green cheese [France capital](https://example.org/france).</answer>",
    }
    scores = {name: explain_score(text, {"target": ["Paris"]}) for name, text in cases.items()}
    print(json.dumps(scores, ensure_ascii=False, indent=2))

    assert scores["supported"]["url_validity"] == 1.0
    assert scores["supported"]["claim_support"] >= 0.5
    assert scores["fake_url"]["fake_url_rate"] > 0.0
    assert scores["fake_url"]["total"] <= 0.35
    assert scores["unsupported"]["unsupported_citation_rate"] >= 1.0
    assert scores["unsupported"]["total"] < scores["supported"]["total"]


if __name__ == "__main__":
    main()
