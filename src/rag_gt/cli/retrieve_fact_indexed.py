"""`python -m rag_gt.cli.retrieve_fact_indexed` -- index facts directly,
return their source chunks (de-duplicated).

Pipeline:
  1. Load facts.jsonl. Each fact has text + supporting_spans[].chunk_id.
  2. Embed every fact's text with BGE/e5.
  3. For each question, compute cosine similarity with every fact for that doc.
  4. Take top_k_facts highest-scoring facts, collect their source chunk_ids,
     keep insertion order (highest-scoring fact first), de-duplicate, take the
     first `top_k` chunks.

This treats RAG_GT's fact-extraction pipeline as the *index*. It is the most
RAG_GT-native retrieval design — but partially circular if all gold facts come
from this same fact set (which is the case here). Reported alongside
fact-anchored as the "RAG_GT-native upper-bound" reference point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.core.types import Span


def _load_jsonl(p: Path) -> List[dict]:
    out: List[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def main() -> None:
    p = argparse.ArgumentParser(prog="rag-gt-retrieve-fact-indexed",
                                description=__doc__)
    p.add_argument("--gt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--facts", default="data/cache/facts.jsonl")
    p.add_argument("--top_k", type=int, default=5,
                   help="distinct chunks to return per question")
    p.add_argument("--top_k_facts", type=int, default=20,
                   help="facts to retrieve before de-duplicating chunks")
    p.add_argument("--embedder", default="BAAI/bge-base-en-v1.5")
    p.add_argument(
        "--chunks-cache",
        default=None,
        help=(
            "Optional active chunk cache. When set, fact source spans are "
            "projected to these chunks, enabling user-specific chunking."
        ),
    )
    args = p.parse_args()

    is_e5 = "e5" in args.embedder.lower()
    Q_PREFIX = "query: " if is_e5 else ""
    P_PREFIX = "passage: " if is_e5 else ""

    facts = _load_jsonl(Path(args.facts))
    print(f"loaded {len(facts)} facts from {args.facts}")
    resolver = ChunkResolver.from_cache(args.chunks_cache) if args.chunks_cache else None
    if resolver is not None:
        print(f"using active chunks cache: {args.chunks_cache} ({len(resolver)} chunks)")

    # Bucket facts by doc_id and capture (text, source_chunk_id) per fact.
    by_doc: Dict[str, List[dict]] = {}
    for f in facts:
        did = f.get("doc_id")
        if not did or not (f.get("text") or "").strip():
            continue
        spans = f.get("supporting_spans") or []
        chunk_ids: List[str] = []
        if resolver is not None:
            for raw_span in spans:
                span = Span.from_any(raw_span)
                chunk_ids.extend(resolver.chunks_for_span(span))
        if not chunk_ids and spans and spans[0].get("chunk_id"):
            chunk_ids = [spans[0]["chunk_id"]]
        if not chunk_ids:
            continue
        by_doc.setdefault(did, []).append({
            "fact_id": f.get("fact_id"),
            "text": f["text"].strip(),
            "chunk_ids": _dedupe(chunk_ids),
        })
    print(f"  bucketed by {len(by_doc)} docs")

    # Embed each doc's facts once.
    if is_e5 or args.embedder != "BAAI/bge-base-en-v1.5":
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedder: {args.embedder}")
        embedder = SentenceTransformer(args.embedder)
    else:
        from rag_gt.core.models import MM
        embedder = MM.get_embedding()

    fact_vecs_by_doc: Dict[str, np.ndarray] = {}
    fact_meta_by_doc: Dict[str, List[dict]] = {}
    for did, items in by_doc.items():
        print(f"encoding {len(items)} facts for {did} ...")
        vecs = np.asarray(embedder.encode(
            [P_PREFIX + it["text"] for it in items],
            normalize_embeddings=True, show_progress_bar=False,
        ))
        fact_vecs_by_doc[did] = vecs
        fact_meta_by_doc[did] = items

    gt = _load_jsonl(Path(args.gt))
    print(f"retrieving over {len(gt)} questions ...")
    results: List[dict] = []
    for q in gt:
        qid = q["q_id"]
        question = q["question"]
        doc_ids = q.get("doc_ids") or []
        # Pool facts from all docs for the question.
        cand: List[tuple] = []          # (sim, fact_meta)
        qv = np.asarray(embedder.encode([Q_PREFIX + question], normalize_embeddings=True,
                                          show_progress_bar=False))[0]
        for did in doc_ids:
            if did not in fact_vecs_by_doc:
                continue
            sims = fact_vecs_by_doc[did] @ qv
            for i, s in enumerate(sims):
                cand.append((float(s), fact_meta_by_doc[did][i]))
        cand.sort(key=lambda t: -t[0])
        # de-duplicate chunks in insertion order
        seen = set(); ordered: List[str] = []
        for _, meta in cand[: args.top_k_facts]:
            for cid in meta["chunk_ids"]:
                if cid in seen:
                    continue
                seen.add(cid); ordered.append(cid)
                if len(ordered) >= args.top_k:
                    break
            if len(ordered) >= args.top_k:
                break
        results.append({"q_id": qid, "retrieved_chunk_ids": ordered})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"written {len(results)} retrieval logs to {args.output}")


if __name__ == "__main__":
    main()
