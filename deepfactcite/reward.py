from __future__ import annotations

import json
import os
import re
import string
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
INFO_RE = re.compile(r"<information>(.*?)</information>", re.S | re.I)
TOOL_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S | re.I)
SNIPPET_RE = re.compile(r'<snippet[^>]*>(.*?)</snippet>', re.S | re.I)
CITATION_RE = re.compile(r"\[([^\[\]]+)\]\(([^()\s]+)\)")
INDEX_CITATION_RE = re.compile(r"\[([1-9]\d*)\]\(([^()\s]+)\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\[\]]+)\]\(([^()\s]+)\)")
BARE_BRACKET_CITATION_RE = re.compile(r"\[([^\[\]\n]+)\](?!\()")
RAW_URL_RE = re.compile(r"(?<!\()https?://[^\s)>\]]+")
DOC_RE = re.compile(
    r"Doc\s+(?P<idx>\d+)\(Title:\s*(?P<title>.*?)\)\s*"
    r"(?:URL:\s*(?P<url>\S+)\s*)?"
    r"(?:Text:\s*)?(?P<text>.*?)(?=\nDoc\s+\d+\(Title:|\Z)",
    re.S,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
    "evidence",
    "source",
    "sources",
    "summary",
    "overview",
    "details",
}


@dataclass
class DeepFactCiteWeights:
    answer: float = 0.25
    citation: float = 0.35
    support: float = 0.25
    format: float = 0.10
    search: float = 0.05
    cost: float = 0.05


@dataclass(frozen=True)
class MarkdownLink:
    label: str
    url: str
    start: int
    end: int


def normalize_answer(text: str) -> str:
    exclude = set(string.punctuation)
    text = "".join(ch for ch in (text or "").lower() if ch not in exclude)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def em_or_subem(answer: str, targets: Any) -> float:
    if not targets:
        return 0.0
    if isinstance(targets, str):
        targets = [targets]
    pred = normalize_answer(answer)
    for target in targets:
        gold = normalize_answer(str(target))
        if gold and (pred == gold or gold in pred):
            return 1.0
    return 0.0


def extract_answer(text: str) -> str | None:
    matches = list(ANSWER_RE.finditer(text or ""))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def convert_deepcite_tags(text: str) -> str:
    replacements = {
        "<google_search>": "<search>",
        "</google_search>": "</search>",
        "<tool_response>": "<information>",
        "</tool_response>": "</information>",
    }
    out = text or ""
    for old, new in replacements.items():
        out = re.sub(re.escape(old), new, out, flags=re.I)
    return out


def validate_structure(text: str) -> tuple[float, list[str]]:
    errors: list[str] = []
    lowered = text.lower()
    if not text.strip():
        return 0.0, ["empty"]
    if "<answer>" not in lowered or "</answer>" not in lowered:
        errors.append("missing answer")
    if lowered.rfind("</answer>") >= 0 and text[lowered.rfind("</answer>") + len("</answer>") :].strip():
        errors.append("content after answer")
    for tag in ["think", "search", "information", "answer"]:
        if lowered.count(f"<{tag}>") != lowered.count(f"</{tag}>"):
            errors.append(f"unbalanced {tag}")
    answer_pos = lowered.rfind("<answer>")
    if answer_pos >= 0 and "<search>" in lowered[answer_pos:]:
        errors.append("search after answer")
    if "<information>" in lowered:
        for match in re.finditer(r"<information>", lowered):
            before = lowered[: match.start()]
            if before.rfind("<search>") < before.rfind("<answer>"):
                errors.append("information not after search")
                break
    if not errors:
        return 1.0, []
    if any(err.startswith("unbalanced") or err in {"content after answer", "search after answer"} for err in errors):
        return 0.4, errors
    return max(0.0, 1.0 - 0.2 * len(set(errors))), errors


def _canonical_url(url: str) -> str:
    url = (url or "").strip().rstrip(".,;")
    parsed = urlparse(url)
    if not parsed.scheme:
        return url
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return parsed._replace(netloc=netloc, path=path, query=parsed.query, fragment="").geturl()


def _tokenize(text: str) -> list[str]:
    return [
        tok.lower()
        for tok in TOKEN_RE.findall(text or "")
        if tok.lower() not in STOPWORDS and (len(tok) > 1 or not tok.isascii())
    ]


def _overlap_support_score(claim: str, document: str) -> float:
    claim_tokens = set(_tokenize(claim))
    if not claim_tokens:
        return 0.0
    doc_tokens = set(_tokenize(document))
    overlap = len(claim_tokens & doc_tokens) / max(len(claim_tokens), 1)
    if overlap >= 0.65:
        return 1.0
    if overlap >= 0.25:
        return 0.5
    return 0.0


def _iter_markdown_links(text: str) -> list[MarkdownLink]:
    """Extract markdown links while tolerating bracketed text inside labels."""
    text = text or ""
    links: list[MarkdownLink] = []
    i = 0
    while i < len(text):
        start = text.find("[", i)
        if start < 0:
            break
        cursor = start + 1
        matched = False
        while cursor < len(text):
            close = text.find("]", cursor)
            if close < 0:
                break
            if close + 1 < len(text) and text[close + 1] == "(":
                end = text.find(")", close + 2)
                if end < 0:
                    break
                label = text[start + 1 : close].strip()
                url = text[close + 2 : end].strip()
                if label and url and not re.search(r"\s", url) and "(" not in url:
                    links.append(MarkdownLink(label=label, url=url, start=start, end=end + 1))
                    i = end + 1
                    matched = True
                    break
            cursor = close + 1
        if not matched:
            i = start + 1
    return links


def _strip_markdown_links(text: str) -> str:
    text = text or ""
    chunks: list[str] = []
    last = 0
    for link in _iter_markdown_links(text):
        chunks.append(text[last : link.start])
        chunks.append(" ")
        last = link.end
    chunks.append(text[last:])
    return "".join(chunks)


def _clean_claim(text: str) -> str:
    text = _strip_markdown_links(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"^#+\s*", " ", text, flags=re.M)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n-:;,.")


def _last_boundary(text: str) -> int:
    boundary = -1
    for match in re.finditer(r"(?<=[.!?])\s+|\n{2,}|(?:^|\n)#{1,6}\s+", text):
        boundary = max(boundary, match.end())
    return boundary


def _first_boundary(text: str) -> int:
    match = re.search(r"(?<=[.!?])\s+|\n{2,}|(?:^|\n)#{1,6}\s+", text)
    return match.start() if match else len(text)


def _claim_near_citation(answer: str, start: int, end: int) -> str:
    """Return the local claim that a citation is expected to support."""
    before = answer[:start]
    after = answer[end:]
    left = max(_last_boundary(before), 0)
    right = end + _first_boundary(after)
    claim = _clean_claim(answer[left:right])

    # If the citation is attached to a very short fragment, pull in one more
    # preceding sentence. This keeps support scoring claim-level rather than
    # citation-label-level while avoiding whole-paragraph credit.
    if len(_tokenize(claim)) < 6 and left > 0:
        prev_left = max(_last_boundary(answer[: max(left - 1, 0)]), 0)
        claim = _clean_claim(answer[prev_left:right])

    tokens = claim.split()
    if len(tokens) > 80:
        claim = " ".join(tokens[-80:])
    return claim


def _citation_claims(answer: str) -> list[dict[str, str]]:
    claims = []
    for link in _iter_markdown_links(answer or ""):
        label = link.label.strip()
        url = _canonical_url(link.url)
        claim = _claim_near_citation(answer or "", link.start, link.end)
        if not claim:
            claim = label
        claims.append({"label": label, "url": url, "claim": claim})
    return claims


def extract_documents(text: str) -> dict[str, str]:
    docs: dict[str, str] = {}
    blocks = INFO_RE.findall(text or "") + TOOL_RE.findall(text or "")
    for block in blocks:
        for snippet in SNIPPET_RE.findall(block):
            url_match = re.search(r"URL:\s*(\S+)", snippet)
            text_match = re.search(r"Text:\s*(.*)", snippet, re.S)
            title_match = re.search(r"Title:\s*(.*)", snippet)
            if url_match:
                body = " ".join(
                    part.strip()
                    for part in [
                        title_match.group(1) if title_match else "",
                        text_match.group(1) if text_match else snippet,
                    ]
                    if part
                )
                docs[_canonical_url(url_match.group(1))] = body
        for match in DOC_RE.finditer(block.strip()):
            title = match.group("title").strip().strip('"')
            url = match.group("url") or f"doc://{match.group('idx')}"
            body = f"{title} {match.group('text').strip()}"
            docs[_canonical_url(url)] = body
    return docs


def citation_metrics(text: str, use_judge: bool | None = None) -> dict[str, float]:
    answer = extract_answer(text) or ""
    docs = extract_documents(text)
    citations = _citation_claims(answer)
    index_citations = [citation for citation in citations if re.fullmatch(r"[1-9]\d*", citation["label"])]
    answer_without_markdown_links = _strip_markdown_links(answer)
    bare_citations = BARE_BRACKET_CITATION_RE.findall(answer_without_markdown_links)
    raw_urls = RAW_URL_RE.findall(answer_without_markdown_links)
    if not citations:
        return {
            "citation_presence": 0.0,
            "url_validity": 0.0,
            "citation_precision": 0.0 if docs else 0.1,
            "citation_recall": 0.0,
            "citation_format": 0.0,
            "claim_support": 0.0,
            "support": 0.0,
            "unsupported": 1.0 if docs else 0.0,
            "unsupported_citation_rate": 1.0 if docs else 0.0,
            "fake_url_rate": 0.0,
            "citation_count": 0.0,
            "bare_citation_count": float(len(bare_citations)),
            "raw_url_count": float(len(raw_urls)),
        }

    valid = [citation for citation in citations if citation["url"] in docs]
    url_validity = len(valid) / len(citations)
    citation_format = max(0.0, url_validity - min(0.1 * len(index_citations), 0.5))
    support_scores = []
    unsupported = len(citations) - len(valid)
    for citation in valid:
        url = citation["url"]
        claim = citation["claim"]
        score = _judge_support(claim, docs[url]) if use_judge else _overlap_support_score(claim, docs[url])
        support_scores.append(score)
        if score < 0.5:
            unsupported += 1
    claim_support = sum(support_scores) / len(citations) if citations else 0.0
    citation_precision = claim_support
    fully_supported = sum(1 for score in support_scores if score >= 1.0)
    citation_recall = min(fully_supported * 0.1, 0.5) + claim_support * 0.5
    unsupported_citation_rate = unsupported / len(citations)
    return {
        "citation_presence": 1.0,
        "url_validity": url_validity,
        "citation_precision": citation_precision,
        "citation_recall": min(citation_recall, 1.0),
        "citation_format": citation_format,
        "claim_support": claim_support,
        "support": claim_support,
        "unsupported": unsupported_citation_rate,
        "unsupported_citation_rate": unsupported_citation_rate,
        "fake_url_rate": 1.0 - url_validity,
        "citation_count": float(len(citations)),
        "bare_citation_count": float(len(bare_citations)),
        "raw_url_count": float(len(raw_urls)),
    }


def _judge_support(claim: str, document: str) -> float:
    base_url = os.getenv("DEEPFACTCITE_JUDGE_BASE_URL")
    model = os.getenv("DEEPFACTCITE_JUDGE_MODEL")
    if not base_url or not model:
        return _overlap_support_score(claim, document)
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url.rstrip("/") + "/v1", api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
        prompt = (
            "Judge whether the document supports the claim. Output exactly one label: "
            "Fully supported / Partially supported / No support.\n\n"
            f"Claim: {claim}\n\nDocument: {document[:3000]}"
        )
        for _ in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=8,
                )
                label = resp.choices[0].message.content.strip()
                if label.startswith("Fully"):
                    return 1.0
                if label.startswith("Partially"):
                    return 0.5
                return 0.0
            except Exception:
                time.sleep(1)
    except Exception:
        pass
    return _overlap_support_score(claim, document)


def search_score(text: str, max_searches: int = 4) -> float:
    searches = len(re.findall(r"<search>.*?</search>", text or "", re.S | re.I))
    answer_pos = (text or "").lower().rfind("<answer>")
    searched_before_answer = "<search>" in (text or "").lower()[:answer_pos] if answer_pos >= 0 else searches > 0
    if searches == 0:
        return 0.0
    score = 1.0 if searched_before_answer else 0.4
    if searches > max_searches:
        score -= min(0.5, 0.1 * (searches - max_searches))
    return max(0.0, score)


def compute_score(
    solution_str: str,
    ground_truth: dict[str, Any] | None = None,
    weights: DeepFactCiteWeights | None = None,
    max_searches: int = 4,
    use_judge: bool | None = None,
) -> float:
    ground_truth = ground_truth or {}
    weights = weights or DeepFactCiteWeights()
    text = convert_deepcite_tags(solution_str)
    answer = extract_answer(text)
    format_value, _ = validate_structure(text)
    answer_targets = ground_truth.get("target")
    answer_value = em_or_subem(answer or "", answer_targets)
    has_evidence = bool(extract_documents(text))
    cite = citation_metrics(text, use_judge=use_judge)
    search_value = search_score(text, max_searches=max_searches)
    cost_penalty = min(len(text) / 12000.0, 1.0)
    citation_value = 0.6 * cite["citation_precision"] + 0.4 * cite["citation_recall"]
    total = (
        weights.answer * answer_value
        + weights.citation * citation_value
        + weights.support * cite["support"]
        + weights.format * format_value
        + weights.search * search_value
        - weights.cost * cost_penalty
    )
    total = max(0.0, min(1.0, total))
    if format_value < 1.0:
        total = min(total, _format_cap(format_value))
    if cite["citation_count"] > 0 and cite["url_validity"] < 1.0:
        total = min(total, 0.35)
    if os.getenv("DFC_REQUIRE_MARKDOWN_CITATION", "false").lower() in {"1", "true", "yes"}:
        if has_evidence and cite["citation_count"] <= 0:
            total = min(total, float(os.getenv("DFC_NO_CITATION_CAP", "0.08")))
        if cite.get("bare_citation_count", 0.0) > 0 or cite.get("raw_url_count", 0.0) > 0:
            total = min(total, float(os.getenv("DFC_BAD_CITATION_FORMAT_CAP", "0.12")))
    if answer_value >= 1.0 and cite["citation_precision"] < 0.5:
        total = min(total, 0.5)
    if cite["unsupported_citation_rate"] >= 0.5 and cite["citation_count"] > 0:
        total = min(total, 0.55)
    if cite["support"] <= 0.0 and cite["citation_count"] > 0:
        total = min(total, 0.4)
    return total


def explain_score(solution_str: str, ground_truth: dict[str, Any] | None = None) -> dict[str, Any]:
    text = convert_deepcite_tags(solution_str)
    fmt, errors = validate_structure(text)
    cite = citation_metrics(text)
    answer = extract_answer(text)
    ground_truth = ground_truth or {}
    answer_targets = ground_truth.get("target")
    return {
        "total": compute_score(text, ground_truth),
        "answer": answer,
        "answer_applicable": 1.0 if answer_targets else 0.0,
        "answer_subem": em_or_subem(answer or "", answer_targets),
        "format": fmt,
        "format_errors": errors,
        "search": search_score(text),
        **cite,
    }


def _format_cap(format_value: float) -> float:
    if format_value <= 0.0:
        return 0.15
    if format_value <= 0.4:
        return 0.4
    if format_value <= 0.6:
        return 0.6
    if format_value <= 0.8:
        return 0.8
    return 0.9


def dumps_score(solution_str: str, ground_truth: dict[str, Any] | None = None) -> str:
    return json.dumps(explain_score(solution_str, ground_truth), ensure_ascii=False, indent=2)
