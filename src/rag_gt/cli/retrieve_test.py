"""Simple retrieval test that produces retrieval_logs.jsonl.

Reuses the GT pipeline's chunker so chunk IDs match the GT spans. The
previous version minted ad-hoc IDs that never aligned with the GT, so
`fact_span_recall` was always 0 with no diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_gt.chunking.strategies import chunk_document
from rag_gt.ingestion import ingest_document
from rag_gt.profiling.profiler import profile_document


def _doc_chunks(input_dir: str) -> dict[str, list[dict]]:
    """Ingest every PDF/DOCX in `input_dir` using the canonical pipeline chunker."""
    out: dict[str, list[dict]] = {}
    base = Path(input_dir)
    for path in sorted(base.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".pdf", ".docx"):
            continue
        if path.suffix.lower() == ".docx":
            # Skip silently if DOCX wasn't enabled — ingest_document raises.
            try:
                doc = ingest_document(str(path))
            except ValueError as e:
                logger.warning(f"[retrieve_test] skipping {path.name}: {e}")
                continue
        else:
            try:
                doc = ingest_document(str(path))
            except Exception as e:
                logger.warning(f"[retrieve_test] failed to ingest {path.name}: {e}")
                continue
        profile = profile_document(doc, path=path)
        chunks = chunk_document(doc, profile)
        out[doc.doc_id] = chunks
    return out


def _load_gt(gt_path: str) -> list[dict]:
    with open(gt_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _retrieve_top_k(
    query: str,
    chunks: list[str],
    chunk_ids: list[str],
    k: int = 5,
) -> list[str]:
    if not chunks:
        return []
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform(chunks + [query])
    except ValueError:
        return chunk_ids[:k]
    similarities = cosine_similarity(tfidf[-1:], tfidf[:-1])[0]
    top_indices = similarities.argsort()[-k:][::-1]
    return [chunk_ids[i] for i in top_indices if similarities[i] > 0]


def _gather_chunks(doc_chunks: dict[str, list[dict]], doc_ids: Iterable[str]):
    texts: list[str] = []
    ids: list[str] = []
    for did in doc_ids:
        if did not in doc_chunks:
            continue
        for c in doc_chunks[did]:
            texts.append(c["text"])
            ids.append(c["chunk_id"])
    return texts, ids


def main() -> None:
    p = argparse.ArgumentParser(description="Generate retrieval logs for evaluation")
    p.add_argument("--input_dir", required=True, help="Directory with PDF/DOCX files")
    p.add_argument("--gt", required=True, help="GT JSONL file")
    p.add_argument(
        "--output", default="retrieval_logs.jsonl", help="Output retrieval log file"
    )
    p.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    args = p.parse_args()

    print(f"Loading documents from {args.input_dir}...")
    doc_chunks = _doc_chunks(args.input_dir)
    print(f"Loaded {len(doc_chunks)} documents (chunked via pipeline)")

    print(f"Loading GT from {args.gt}...")
    gt_questions = _load_gt(args.gt)
    print(f"Loaded {len(gt_questions)} questions")

    results = []
    for q in gt_questions:
        q_id = q["q_id"]
        doc_ids = q.get("doc_ids") or []
        question = q["question"]

        if not doc_ids:
            print(f"Warning: q_id={q_id} has no doc_ids; skipping")
            continue

        texts, ids = _gather_chunks(doc_chunks, doc_ids)
        if not texts:
            print(f"Warning: no chunks loaded for q_id={q_id} (doc_ids={doc_ids})")
            continue

        retrieved = _retrieve_top_k(question, texts, ids, args.top_k)
        results.append({"q_id": q_id, "retrieved_chunk_ids": retrieved})

    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Written {len(results)} retrieval logs to {args.output}")


if __name__ == "__main__":
    main()
