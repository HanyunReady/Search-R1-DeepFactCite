from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text or "")]


class RetrieveRequest(BaseModel):
    queries: list[str]
    topk: int = 3
    return_scores: bool = True


class LexicalRetriever:
    def __init__(self, corpus_path: str | Path):
        self.docs = []
        self.doc_tokens = []
        with Path(corpus_path).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                contents = row.get("contents") or ""
                doc = {
                    "id": str(row.get("id", len(self.docs))),
                    "contents": contents,
                    "url": row.get("url") or row.get("source_url"),
                }
                self.docs.append(doc)
                self.doc_tokens.append(tokenize(contents))

    def search(self, query: str, topk: int) -> list[dict[str, Any]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        scored = []
        for doc, tokens in zip(self.docs, self.doc_tokens):
            if not tokens:
                continue
            score = self._score(q_tokens, tokens)
            if score <= 0:
                continue
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{"document": doc, "score": score} for score, doc in scored[:topk]]

    @staticmethod
    def _score(query_tokens: list[str], doc_tokens: list[str]) -> float:
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0
        tf = {}
        for tok in doc_tokens:
            tf[tok] = tf.get(tok, 0) + 1
        score = 0.0
        for tok in query_tokens:
            freq = tf.get(tok, 0)
            if freq:
                score += math.log1p(freq) / math.sqrt(doc_len)
        return score


def build_app(corpus_path: str | Path) -> FastAPI:
    retriever = LexicalRetriever(corpus_path)
    app = FastAPI()

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest):
        return {"result": [retriever.search(query, req.topk) for query in req.queries]}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(build_app(args.corpus), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
