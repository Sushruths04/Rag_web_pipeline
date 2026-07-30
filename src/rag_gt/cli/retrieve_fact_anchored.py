"""`python -m rag_gt.cli.retrieve_fact_anchored` -- hybrid retriever that
augments each chunk with its extracted facts before embedding.

The augmentation is concatenated as a "FACTS:" suffix to the chunk text. The
embedder thus sees both the raw chunk content AND the curated propositions,
which makes question-to-chunk matching tighter for chunks that have facts.

Pipeline:
  1. Load the chunk cache (text + chunk_id) and the facts cache.
  2. For each chunk, append the texts of its supporting facts ("FACTS: ...").
  3. BM25 over augmented text + BGE/e5 dense over augmented text.
  4. RRF merge → optional cross-encoder rerank → top_k.

This is RAG_GT-native: the index is built using the same fact extraction that
defines the gold standard, but the retriever still returns chunk_ids (so a real
RAG system can swap it in 1:1 for its current retriever).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

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
            logger.warning(f"[retrieve_fact_anchored] skip {path.name}: {e}")
            continue
        profile = profile_document(doc, path=path)
        out[doc.doc_id] = chunk_document(doc, profile)
    return out


def _load_jsonl(p: Path) -> List[dict]:
    out: List[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _build_facts_by_chunk(facts_path: Path) -> Dict[str, List[str]]:
    """chunk_id -> list of fact texts that come from that chunk."""
    out: Dict[str, List[str]] = {}
    for fact in _load_jsonl(facts_path):
        text = (fact.get("text") or "").strip()
        if not text:
            continue
        for sp in fact.get("supporting_spans") or []:
            cid = sp.get("chunk_id", "")
            if cid:
                out.setdefault(cid, []).append(text)
    return out


def _rrf(rankings: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def main() -> None:
    p = argparse.ArgumentParser(prog="rag-gt-retrieve-fact-anchored",
                                description=__doc__)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--facts", default="data/cache/facts.jsonl",
                   help="path to facts.jsonl (defines the augmentation)")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--top_k_lex", type=int, default=50)
    p.add_argument("--top_k_dense", type=int, default=50)
    p.add_argument("--top_k_fused", type=int, default=20)
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--reranker_model", default="BAAI/bge-reranker-base")
    p.add_argument("--embedder", default="BAAI/bge-base-en-v1.5")
    args = p.parse_args()

    is_e5 = "e5" in args.embedder.lower()
    Q_PREFIX = "query: " if is_e5 else ""
    P_PREFIX = "passage: " if is_e5 else ""

    print(f"Loading documents from {args.input_dir} ...")
    doc_chunks = _doc_chunks(args.input_dir)
    print(f"Loaded {len(doc_chunks)} documents")

    print(f"Loading facts from {args.facts} ...")
    facts_by_chunk = _build_facts_by_chunk(Path(args.facts))
    n_anchored = sum(1 for d in doc_chunks.values() for c in d if c["chunk_id"] in facts_by_chunk)
    n_total = sum(len(d) for d in doc_chunks.values())
    print(f"  {n_anchored}/{n_total} chunks will be fact-anchored "
          f"({100*n_anchored/max(1,n_total):.1f}%)")

    print(f"Loading GT from {args.gt} ...")
    gt = _load_jsonl(Path(args.gt))
    print(f"Loaded {len(gt)} questions")

    # Build augmented texts per chunk: chunk_text + " FACTS: " + concatenated fact texts.
    aug_texts: List[str] = []
    aug_ids: List[str] = []
    for did, chunks in doc_chunks.items():
        for c in chunks:
            cid = c["chunk_id"]
            facts = facts_by_chunk.get(cid, [])
            if facts:
                aug = c["text"] + "\n\nFACTS: " + " ".join(facts)
            else:
                aug = c["text"]
            aug_texts.append(aug)
            aug_ids.append(cid)
    if not aug_texts:
        print("No chunks to encode; aborting.")
        return

    # Encode with the chosen embedder.
    if is_e5 or args.embedder != "BAAI/bge-base-en-v1.5":
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedder: {args.embedder}")
        embedder = SentenceTransformer(args.embedder)
    else:
        from rag_gt.core.models import MM
        embedder = MM.get_embedding()
    print(f"Encoding {len(aug_texts)} fact-anchored chunks ({args.embedder}) ...")
    chunk_vecs = np.asarray(embedder.encode(
        [P_PREFIX + t for t in aug_texts],
        normalize_embeddings=True, show_progress_bar=True,
    ))
    id_to_idx = {cid: i for i, cid in enumerate(aug_ids)}
    text_by_id = dict(zip(aug_ids, aug_texts))

    # BM25 per doc on the augmented text.
    print("Tokenizing augmented chunks for BM25 ...")
    bm25_by_doc: Dict[str, Tuple[BM25Okapi, List[str]]] = {}
    chunks_by_doc: Dict[str, List[str]] = {}
    for did, chunks in doc_chunks.items():
        ids = [c["chunk_id"] for c in chunks]
        chunks_by_doc[did] = ids
        toks = [_tokenize(text_by_id[c]) for c in ids]
        bm25_by_doc[did] = (BM25Okapi(toks), ids)

    reranker = None
    if args.rerank:
        from sentence_transformers import CrossEncoder
        print(f"Loading reranker: {args.reranker_model} ...")
        reranker = CrossEncoder(args.reranker_model)

    print(f"Retrieving top_k={args.top_k} ...")
    results: List[dict] = []
    for q in gt:
        qid = q["q_id"]
        question = q["question"]
        doc_ids = q.get("doc_ids") or []
        if not doc_ids:
            continue

        # BM25 over augmented chunks per doc
        lex_scores: Dict[str, float] = {}
        q_toks = _tokenize(question)
        for did in doc_ids:
            bm = bm25_by_doc.get(did)
            if not bm: continue
            bm25, ids = bm
            scores = bm25.get_scores(q_toks)
            top = np.argsort(-scores)[: args.top_k_lex]
            for i in top:
                if scores[int(i)] > 0:
                    lex_scores[ids[int(i)]] = max(lex_scores.get(ids[int(i)], -1e9),
                                                   float(scores[int(i)]))
        lex_ranking = sorted(lex_scores.items(), key=lambda x: -x[1])[: args.top_k_lex]

        # Dense over augmented chunks per doc
        idxs = np.array([id_to_idx[i] for did in doc_ids for i in chunks_by_doc.get(did, []) if i in id_to_idx])
        if len(idxs) == 0:
            results.append({"q_id": qid, "retrieved_chunk_ids": []}); continue
        sub_vecs = chunk_vecs[idxs]
        sub_ids = [aug_ids[int(i)] for i in idxs]
        qv = np.asarray(embedder.encode([Q_PREFIX + question], normalize_embeddings=True,
                                         show_progress_bar=False))[0]
        sims = sub_vecs @ qv
        order = np.argsort(-sims)[: args.top_k_dense]
        dense_ranking = [(sub_ids[int(i)], float(sims[int(i)])) for i in order if sims[int(i)] > 0]

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
