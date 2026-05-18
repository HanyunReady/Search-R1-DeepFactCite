#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel


class RetrieveRequest(BaseModel):
    queries: list[str]
    topk: int = 3
    return_scores: bool = True


class SearchR1BM25:
    def __init__(self, index_dir: str | Path, corpus_path: str | Path | None = None):
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(str(index_dir))
        self.corpus = None
        if self._doc_raw("0") is None and corpus_path:
            import datasets

            self.corpus = datasets.load_dataset(
                "json",
                data_files=str(corpus_path),
                split="train",
                num_proc=4,
                cache_dir=os.environ.get("HF_DATASETS_CACHE"),
            )

    def search(self, query: str, topk: int, return_scores: bool) -> list[dict[str, Any]]:
        hits = self.searcher.search(query, topk)
        rows = []
        for hit in hits:
            raw = self._doc_raw(hit.docid)
            if raw is None and self.corpus is not None:
                document = normalize_corpus_doc(self.corpus[int(hit.docid)], hit.docid)
            else:
                document = parse_raw_doc(raw, hit.docid)
            row = {"document": document}
            if return_scores:
                row["score"] = float(hit.score)
            rows.append(row)
        return rows

    def _doc_raw(self, docid: str) -> str | None:
        doc = self.searcher.doc(docid)
        return doc.raw() if doc is not None else None


def parse_raw_doc(raw: str | None, docid: str) -> dict[str, Any]:
    if not raw:
        return {"id": docid, "contents": ""}
    try:
        doc = json.loads(raw)
        if "id" not in doc:
            doc["id"] = docid
        if "contents" not in doc:
            title = doc.get("title") or doc.get("name") or docid
            text = doc.get("text") or doc.get("contents") or ""
            doc["contents"] = f"{title}\n{text}"
        return doc
    except Exception:
        return {"id": docid, "contents": raw}


def normalize_corpus_doc(doc: dict[str, Any], docid: str) -> dict[str, Any]:
    out = dict(doc)
    out.setdefault("id", docid)
    if "contents" not in out:
        title = out.get("title") or out.get("name") or docid
        text = out.get("text") or ""
        out["contents"] = f"{title}\n{text}".strip()
    return out


def build_app(index_dir: str | Path, corpus_path: str | Path | None = None) -> FastAPI:
    retriever = SearchR1BM25(index_dir, corpus_path=corpus_path)
    app = FastAPI()

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest):
        return {"result": [retriever.search(query, req.topk, req.return_scores) for query in req.queries]}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Search-R1 wiki-18 BM25 index with the /retrieve API.")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--corpus-path", default="")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(build_app(args.index_dir, corpus_path=args.corpus_path or None), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
