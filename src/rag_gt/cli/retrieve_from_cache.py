"""`python -m rag_gt.cli.retrieve_from_cache` -- universal hybrid retriever
that reads chunks from the precomputed cache instead of re-ingesting PDFs.

Use this when the chunks are already in `data/cache/chunks.jsonl` and you
just want to swap the retriever or the GT.

Same pipeline shape as `retrieve_hybrid` (BM25 + dense + RRF + optional rerank)
— only the chunk source differs.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


_TOK = re.compile(r"\w+")


def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _TOK.findall(s)]


def _load_jsonl(p: Path) -> List[dict]:
    out: List[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _rrf(rankings: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks", default="data/cache/chunks.jsonl",
                   help="path to chunks.jsonl (text + chunk_id + doc_id)")
    p.add_argument("--gt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--top_k_lex", type=int, default=50)
    p.add_argument("--top_k_dense", type=int, default=50)
    p.add_argument("--top_k_fused", type=int, default=20)
    p.add_argument("--mode", choices=["dense", "hybrid"], default="hybrid",
                   help="dense = BGE only; hybrid = BM25+BGE via RRF")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--reranker_model", default="BAAI/bge-reranker-base")
    p.add_argument("--embedder", default="BAAI/bge-base-en-v1.5")
    args = p.parse_args()

    is_e5 = "e5" in args.embedder.lower()
    Q_PREFIX = "query: " if is_e5 else ""
    P_PREFIX = "passage: " if is_e5 else ""

    print(f"Loading chunks from {args.chunks} ...")
    chunks = _load_jsonl(Path(args.chunks))
    by_doc: Dict[str, List[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c.get("doc_id", "_unknown"), []).append(c)
    print(f"  {len(chunks)} chunks across {len(by_doc)} docs")

    print(f"Loading GT from {args.gt} ...")
    gt = _load_jsonl(Path(args.gt))
    print(f"  {len(gt)} questions")

    needed_docs = set()
    for q in gt:
        for d in q.get("doc_ids") or []:
            needed_docs.add(d)
    chunks_for_query = {d: by_doc.get(d, []) for d in needed_docs}
    n_to_encode = sum(len(v) for v in chunks_for_query.values())
    print(f"  encoding only {n_to_encode} chunks for these docs")

    if is_e5 or args.embedder != "BAAI/bge-base-en-v1.5":
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(args.embedder)
    else:
        from rag_gt.core.models import MM
        embedder = MM.get_embedding()

    all_texts: List[str] = []
    all_ids: List[str] = []
    for d in sorted(needed_docs):
        for c in chunks_for_query[d]:
            all_texts.append(c["text"])
            all_ids.append(c["chunk_id"])
    print(f"Encoding {len(all_texts)} chunks ({args.embedder}) ...")
    chunk_vecs = np.asarray(embedder.encode(
        [P_PREFIX + t for t in all_texts],
        normalize_embeddings=True, show_progress_bar=True,
    ))
    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
    text_by_id = dict(zip(all_ids, all_texts))

    bm25_by_doc: Dict[str, Tuple[BM25Okapi, List[str]]] = {}
    if args.mode == "hybrid":
        print("Tokenizing for BM25 ...")
        for d in needed_docs:
            cs = chunks_for_query[d]
            ids = [c["chunk_id"] for c in cs]
            toks = [_tokenize(c["text"]) for c in cs]
            if toks:
                bm25_by_doc[d] = (BM25Okapi(toks), ids)

    reranker = None
    if args.rerank:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(args.reranker_model)

    print(f"Retrieving (mode={args.mode}, rerank={args.rerank}, top_k={args.top_k}) ...")
    results: List[dict] = []
    for q in gt:
        qid = q["q_id"]
        question = q["question"]
        doc_ids = q.get("doc_ids") or []
        if not doc_ids:
            results.append({"q_id": qid, "retrieved_chunk_ids": []}); continue

        idxs = np.array([id_to_idx[c["chunk_id"]] for d in doc_ids
                         for c in chunks_for_query.get(d, [])
                         if c["chunk_id"] in id_to_idx])
        if len(idxs) == 0:
            results.append({"q_id": qid, "retrieved_chunk_ids": []}); continue
        sub_vecs = chunk_vecs[idxs]
        sub_ids = [all_ids[int(i)] for i in idxs]
        qv = np.asarray(embedder.encode([Q_PREFIX + question], normalize_embeddings=True,
                                          show_progress_bar=False))[0]
        sims = sub_vecs @ qv
        order = np.argsort(-sims)[: args.top_k_dense]
        dense_ranking = [(sub_ids[int(i)], float(sims[int(i)])) for i in order if sims[int(i)] > 0]

        if args.mode == "dense":
            results.append({"q_id": qid,
                            "retrieved_chunk_ids": [c for c, _ in dense_ranking[: args.top_k]]})
            continue

        lex_scores: Dict[str, float] = {}
        q_toks = _tokenize(question)
        for d in doc_ids:
            bm = bm25_by_doc.get(d)
            if not bm: continue
            bm25, ids = bm
            scores = bm25.get_scores(q_toks)
            top = np.argsort(-scores)[: args.top_k_lex]
            for i in top:
                if scores[int(i)] > 0:
                    lex_scores[ids[int(i)]] = max(lex_scores.get(ids[int(i)], -1e9),
                                                   float(scores[int(i)]))
        lex_ranking = sorted(lex_scores.items(), key=lambda x: -x[1])[: args.top_k_lex]

        fused = _rrf([lex_ranking, dense_ranking])[: args.top_k_fused]

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
