"""`python -m rag_gt.cli.retrieve_hybrid` -- BM25 + BGE hybrid retriever
with optional cross-encoder reranker.

Pipeline (in order):
  1.  BM25 over raw chunk text -> top_k_lex candidates per question
  2.  BGE-base dense cosine    -> top_k_dense candidates per question
  3.  Reciprocal-Rank-Fusion   -> merged top_k_fused
  4.  Optional cross-encoder reranker (bge-reranker-base) over top_k_fused
                                -> final top_k

Emits the standard `retrieval_logs.jsonl` schema so all downstream evaluators
work unchanged. Restricts retrieval to chunks whose doc_id is in each
question's `doc_ids` (matches retrieve_dense semantics).

Examples:
  # Hybrid only (no rerank), top_k=5 to match the original RAGAS run
  python -m rag_gt.cli.retrieve_hybrid \
      --input_dir data/docs_rl --gt data/gt/reinforcement_qa.jsonl \
      --output data/eval_runs/reinforcement_qa_retrieval_hybrid_topk5.jsonl \
      --top_k 5

  # Hybrid + cross-encoder rerank, top_k=5
  python -m rag_gt.cli.retrieve_hybrid \
      --input_dir data/docs_rl --gt data/gt/reinforcement_qa.jsonl \
      --output data/eval_runs/reinforcement_qa_retrieval_hybrid_rerank_topk5.jsonl \
      --top_k 5 --rerank
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from rag_gt.chunking.strategies import chunk_document
from rag_gt.ingestion import ingest_document
from rag_gt.profiling.profiler import profile_document


_TOK = re.compile(r"\w+")


def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _TOK.findall(s)]


def _doc_chunks(input_dir: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    base = Path(input_dir)
    for path in sorted(base.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".pdf", ".docx"):
            continue
        try:
            doc = ingest_document(str(path))
        except Exception as e:
            logger.warning(f"[retrieve_hybrid] skip {path.name}: {e}")
            continue
        profile = profile_document(doc, path=path)
        out[doc.doc_id] = chunk_document(doc, profile)
    return out


def _load_gt(gt_path: str) -> List[dict]:
    with open(gt_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _rrf(rankings: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion. Each list is [(chunk_id, score), ...] sorted desc."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag-gt-retrieve-hybrid",
        description="Hybrid BM25 + BGE retriever with optional cross-encoder rerank.",
    )
    p.add_argument("--input_dir", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top_k", type=int, default=5, help="final top-k written to log")
    p.add_argument("--top_k_lex", type=int, default=50, help="BM25 candidates")
    p.add_argument("--top_k_dense", type=int, default=50, help="BGE candidates")
    p.add_argument("--top_k_fused", type=int, default=20, help="kept after RRF")
    p.add_argument("--rerank", action="store_true", help="apply cross-encoder reranker")
    p.add_argument(
        "--reranker_model", default="BAAI/bge-reranker-base",
        help="cross-encoder model id (HuggingFace)",
    )
    p.add_argument(
        "--embedder", default="BAAI/bge-base-en-v1.5",
        help="dense embedder model id; supports BGE and E5 families",
    )
    p.add_argument(
        "--query-rewrite", action="store_true",
        help="rewrite each question via the configured judge LLM before retrieval",
    )
    args = p.parse_args()

    # E5 family expects "query: " / "passage: " prefixes for best performance.
    is_e5 = "e5" in args.embedder.lower()
    Q_PREFIX = "query: " if is_e5 else ""
    P_PREFIX = "passage: " if is_e5 else ""

    print(f"Loading documents from {args.input_dir} ...")
    doc_chunks = _doc_chunks(args.input_dir)
    print(f"Loaded {len(doc_chunks)} documents")

    print(f"Loading GT from {args.gt} ...")
    gt = _load_gt(args.gt)
    print(f"Loaded {len(gt)} questions")

    # Encode chunks once. Use the configured embedder; E5 needs prefixes.
    if is_e5 or args.embedder != "BAAI/bge-base-en-v1.5":
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedder: {args.embedder}")
        embedder = SentenceTransformer(args.embedder)
    else:
        from rag_gt.core.models import MM
        embedder = MM.get_embedding()

    all_texts: List[str] = []
    all_ids: List[str] = []
    for did, chunks in doc_chunks.items():
        for c in chunks:
            all_texts.append(c["text"])
            all_ids.append(c["chunk_id"])
    if not all_texts:
        print("No chunks to encode; aborting.")
        return

    print(f"Encoding chunks ({args.embedder}) ...")
    chunk_vecs = np.asarray(embedder.encode(
        [P_PREFIX + t for t in all_texts],
        normalize_embeddings=True, show_progress_bar=True,
    ))
    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
    text_by_id = dict(zip(all_ids, all_texts))

    # Build BM25 indexes per doc_id (so the lexical search is also doc-restricted).
    print("Tokenizing for BM25 ...")
    bm25_by_doc: Dict[str, Tuple[BM25Okapi, List[str]]] = {}
    for did, chunks in doc_chunks.items():
        toks = [_tokenize(c["text"]) for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        bm25_by_doc[did] = (BM25Okapi(toks), ids)

    reranker = None
    if args.rerank:
        from sentence_transformers import CrossEncoder
        print(f"Loading cross-encoder reranker: {args.reranker_model} ...")
        reranker = CrossEncoder(args.reranker_model)

    rewriter_chat = None
    if args.query_rewrite:
        from dotenv import load_dotenv
        load_dotenv()
        from rag_gt.core.llm import get_llm
        rewriter_chat = get_llm("gt")
        print("Query rewriting enabled (judge LLM).")

    print(f"Retrieving with top_k={args.top_k} (lex={args.top_k_lex}, dense={args.top_k_dense}, "
          f"fused={args.top_k_fused}, rerank={args.rerank}, qr={args.query_rewrite}) ...")
    results: List[dict] = []
    for q in gt:
        qid = q["q_id"]
        question = q["question"]
        doc_ids = q.get("doc_ids") or []
        if not doc_ids:
            print(f"Warning: q_id={qid} has no doc_ids; skipping")
            continue
        if rewriter_chat is not None:
            try:
                prompt = (
                    "You are a search query writer. Rewrite the user's question into "
                    "the most effective short search query for a corpus retrieval "
                    "system (BM25 + dense). Keep technical terms verbatim. Return "
                    "only the rewritten query, no preamble.\n\n"
                    f"Question: {question}\nSearch query:"
                )
                rewritten = rewriter_chat.generate(prompt, temperature=0.0, max_tokens=64) or ""
                rewritten = rewritten.strip().strip('"').strip("'")
                if rewritten:
                    question = rewritten
            except Exception as e:
                print(f"  query-rewrite failed for {qid}: {e!r}")

        # 1. BM25 per doc, then concat
        lex_scores: Dict[str, float] = {}
        q_toks = _tokenize(question)
        for did in doc_ids:
            bm = bm25_by_doc.get(did)
            if not bm:
                continue
            bm25, ids = bm
            scores = bm25.get_scores(q_toks)
            top_idxs = np.argsort(-scores)[: args.top_k_lex]
            for i in top_idxs:
                if scores[int(i)] > 0:
                    lex_scores[ids[int(i)]] = max(lex_scores.get(ids[int(i)], -1e9), float(scores[int(i)]))
        lex_ranking = sorted(lex_scores.items(), key=lambda x: -x[1])[: args.top_k_lex]

        # 2. BGE dense over the same restricted chunks
        idxs = np.array([id_to_idx[i] for did in doc_ids for i in (bm25_by_doc[did][1] if did in bm25_by_doc else [])
                         if i in id_to_idx])
        if len(idxs) == 0:
            results.append({"q_id": qid, "retrieved_chunk_ids": []})
            continue
        sub_vecs = chunk_vecs[idxs]
        sub_ids = [all_ids[int(i)] for i in idxs]
        qv = np.asarray(embedder.encode(
            [Q_PREFIX + question], normalize_embeddings=True, show_progress_bar=False,
        ))[0]
        sims = sub_vecs @ qv
        order = np.argsort(-sims)[: args.top_k_dense]
        dense_ranking = [(sub_ids[int(i)], float(sims[int(i)])) for i in order if sims[int(i)] > 0]

        # 3. Reciprocal Rank Fusion
        fused = _rrf([lex_ranking, dense_ranking])[: args.top_k_fused]

        # 4. Optional rerank
        if reranker is not None and fused:
            cand_ids = [cid for cid, _ in fused]
            pairs = [(question, text_by_id[cid]) for cid in cand_ids]
            ce_scores = reranker.predict(pairs, show_progress_bar=False)
            order = np.argsort(-np.asarray(ce_scores))
            final = [cand_ids[int(i)] for i in order[: args.top_k]]
        else:
            final = [cid for cid, _ in fused[: args.top_k]]

        results.append({"q_id": qid, "retrieved_chunk_ids": final})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written {len(results)} retrieval logs to {args.output}")


if __name__ == "__main__":
    main()
