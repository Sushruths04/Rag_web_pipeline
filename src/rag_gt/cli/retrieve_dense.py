"""`python -m rag_gt.cli.retrieve_dense` -- BGE dense retriever.

Drop-in replacement for `cli/retrieve_test.py` that uses BGE embeddings
(via `core/models.MM.get_embedding()`) instead of TF-IDF. Emits the same
`retrieval_logs.jsonl` schema, so all downstream evaluators
(`retrieval_eval.py`, `evaluate.py`, `compare.py`) work without changes.

Restricts retrieval to chunks of the documents listed in each question's
``doc_ids`` (matches `retrieve_test.py` semantics). For full-corpus dense
retrieval, drop the doc_ids filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from loguru import logger

from rag_gt.chunking.strategies import chunk_document
from rag_gt.ingestion import ingest_document
from rag_gt.profiling.profiler import profile_document


def _doc_chunks(input_dir: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    base = Path(input_dir)
    for path in sorted(base.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".pdf", ".docx"):
            continue
        try:
            doc = ingest_document(str(path))
        except Exception as e:
            logger.warning(f"[retrieve_dense] skip {path.name}: {e}")
            continue
        profile = profile_document(doc, path=path)
        out[doc.doc_id] = chunk_document(doc, profile)
    return out


def _load_gt(gt_path: str) -> List[dict]:
    with open(gt_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _gather_chunks(doc_chunks: Dict[str, List[dict]], doc_ids: Iterable[str]):
    texts: List[str] = []
    ids: List[str] = []
    for did in doc_ids:
        if did not in doc_chunks:
            continue
        for c in doc_chunks[did]:
            texts.append(c["text"])
            ids.append(c["chunk_id"])
    return texts, ids


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag-gt-retrieve-dense",
        description="BGE dense retriever; emits retrieval_logs.jsonl.",
    )
    p.add_argument("--input_dir", required=True, help="Directory with PDF/DOCX files.")
    p.add_argument("--gt", required=True, help="GT JSONL file.")
    p.add_argument(
        "--output", default="retrieval_logs_dense.jsonl",
        help="Output retrieval log file (default: retrieval_logs_dense.jsonl).",
    )
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()

    from rag_gt.core.models import MM

    print(f"Loading documents from {args.input_dir}...")
    doc_chunks = _doc_chunks(args.input_dir)
    print(f"Loaded {len(doc_chunks)} documents (chunked via pipeline)")

    print(f"Loading GT from {args.gt}...")
    gt = _load_gt(args.gt)
    print(f"Loaded {len(gt)} questions")

    embedder = MM.get_embedding()

    # Encode every chunk of every doc once and cache by chunk_id.
    print("Encoding chunks (BGE)...")
    all_texts: List[str] = []
    all_ids: List[str] = []
    for did, chunks in doc_chunks.items():
        for c in chunks:
            all_texts.append(c["text"])
            all_ids.append(c["chunk_id"])
    if not all_texts:
        print("No chunks to encode; aborting.")
        return
    chunk_vecs = embedder.encode(
        all_texts, normalize_embeddings=True, show_progress_bar=True
    )
    chunk_vecs = np.asarray(chunk_vecs)
    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}

    print("Retrieving...")
    results: List[dict] = []
    for q in gt:
        q_id = q["q_id"]
        doc_ids = q.get("doc_ids") or []
        if not doc_ids:
            print(f"Warning: q_id={q_id} has no doc_ids; skipping")
            continue
        texts, ids = _gather_chunks(doc_chunks, doc_ids)
        if not texts:
            print(f"Warning: no chunks for q_id={q_id} (doc_ids={doc_ids})")
            continue
        # Slice the precomputed chunk vectors by ids in this question's docs.
        idxs = np.array([id_to_idx[i] for i in ids if i in id_to_idx])
        sub_vecs = chunk_vecs[idxs]
        qv = embedder.encode(
            [q["question"]], normalize_embeddings=True, show_progress_bar=False
        )
        qv = np.asarray(qv)[0]
        sims = sub_vecs @ qv
        top = np.argsort(-sims)[: args.top_k]
        retrieved = [ids[int(i)] for i in top if sims[int(i)] > 0]
        results.append({"q_id": q_id, "retrieved_chunk_ids": retrieved})

    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written {len(results)} retrieval logs to {args.output}")


if __name__ == "__main__":
    main()
