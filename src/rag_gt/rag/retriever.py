"""CP2 — BM25 retriever + optional dense retriever.

BM25Retriever  — pure Python, no external dependency beyond stdlib + math.
                 Reuses _tokenize / _build_bm25 / _bm25_score from evaluate.py.

DenseRetriever — optional, uses sentence-transformers (local, no API key).
                 Silently falls back to None if the library is not installed.

HybridRetriever — RRF fusion of BM25 + Dense results.

All retrievers expose the same interface:
    retriever.retrieve(query: str, top_k: int) -> List[Tuple[str, float]]
    where each tuple is (chunk_id, score).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rag_gt.core.config import load_config

# ---------------------------------------------------------------------------
# Shared tokenizer (identical to evaluate.py for consistent overlap semantics)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# BM25 index (pure Python)
# ---------------------------------------------------------------------------

@dataclass
class _BM25Index:
    inverted: Dict[str, Dict[str, int]]   # token -> {chunk_id: tf}
    doc_lens: Dict[str, int]              # chunk_id -> doc length in tokens
    avgdl: float
    id_to_text: Dict[str, str]            # chunk_id -> text
    id_to_chunk: Dict[str, dict]          # chunk_id -> full chunk dict


def _build_index(chunks: List[dict]) -> _BM25Index:
    inverted: Dict[str, Dict[str, int]] = {}
    doc_lens: Dict[str, int] = {}
    id_to_text: Dict[str, str] = {}
    id_to_chunk: Dict[str, dict] = {}

    for chunk in chunks:
        cid = chunk.get("chunk_id", "")
        text = chunk.get("text", "")
        toks = _tokenize(text)
        doc_lens[cid] = len(toks)
        id_to_text[cid] = text
        id_to_chunk[cid] = chunk
        for tok in toks:
            inverted.setdefault(tok, {})
            inverted[tok][cid] = inverted[tok].get(cid, 0) + 1

    avgdl = sum(doc_lens.values()) / max(1, len(doc_lens))
    return _BM25Index(
        inverted=inverted,
        doc_lens=doc_lens,
        avgdl=avgdl,
        id_to_text=id_to_text,
        id_to_chunk=id_to_chunk,
    )


def _bm25_scores(
    query_tokens: List[str],
    idx: _BM25Index,
    k1: float = 1.5,
    b: float = 0.75,
) -> Dict[str, float]:
    N = len(idx.doc_lens)
    scores: Dict[str, float] = {}
    for tok in set(query_tokens):
        postings = idx.inverted.get(tok, {})
        df = len(postings)
        if df == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        for cid, tf in postings.items():
            dl = idx.doc_lens.get(cid, idx.avgdl)
            tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / idx.avgdl))
            scores[cid] = scores.get(cid, 0.0) + idf * tf_norm
    return scores


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """Build a BM25 index over a list of chunk dicts and retrieve by query."""

    def __init__(self, chunks: List[dict]) -> None:
        self._idx = _build_index(chunks)
        self.n_chunks = len(chunks)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return [(chunk_id, bm25_score)] sorted desc, length <= top_k."""
        toks = _tokenize(query)
        if not toks:
            return []
        scores = _bm25_scores(toks, self._idx)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def get_chunk_text(self, chunk_id: str) -> str:
        return self._idx.id_to_text.get(chunk_id, "")

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        return self._idx.id_to_chunk.get(chunk_id)

    def all_chunk_ids(self) -> List[str]:
        return list(self._idx.doc_lens.keys())


# ---------------------------------------------------------------------------
# DenseRetriever (optional)
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Dense retriever using the configured embedding adapter.

    With no explicit ``model``, this reuses ``MM.get_embedding()`` so the
    ACTIVE_EMBEDDING/API configuration remains the single source of truth.
    Passing a model name explicitly retains the local SentenceTransformer path.
    """

    def __init__(
        self,
        chunks: List[dict],
        model: Optional[str] = None,
        *,
        embedder: object | None = None,
        embed_source: str = "api",
    ) -> None:
        import numpy as np

        if embedder is not None:
            self._model = embedder
        elif model:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for an explicit local dense model"
                ) from exc
            self._model = SentenceTransformer(model)
        elif embed_source == "local":
            # Force a local SentenceTransformer regardless of ACTIVE_EMBEDDING —
            # guarantees an offline embedder even if the environment has an API
            # embedding endpoint configured (F9: hybrid eval silently calling a
            # paid embedding API).
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for embed_source='local'"
                ) from exc
            from rag_gt.core.models import EMBED_MODEL_NAME

            self._model = SentenceTransformer(EMBED_MODEL_NAME)
        else:
            from rag_gt.core.models import MM

            self._model = MM.get_embedding()
        self._chunk_ids = [c.get("chunk_id", "") for c in chunks]
        self._id_to_chunk = {c.get("chunk_id", ""): c for c in chunks}
        texts = [c.get("text", "") for c in chunks]
        batch_size = int(
            load_config()["multigold_evaluation"]["embedding_batch_size"]
        )
        self._embeddings = np.asarray(
            self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        self._np = np
        self._query_cache: Dict[str, object] = {}

    def prime_queries(self, queries: List[str]) -> None:
        """Embed uncached queries in one batched adapter call."""
        missing = list(dict.fromkeys(q for q in queries if q not in self._query_cache))
        if not missing:
            return
        batch_size = int(
            load_config()["multigold_evaluation"]["embedding_batch_size"]
        )
        vectors = self._model.encode(
            missing,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        for query, vector in zip(missing, vectors):
            self._query_cache[query] = self._np.asarray(vector, dtype=self._np.float32)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if query not in self._query_cache:
            self.prime_queries([query])
        q_emb = self._np.asarray(self._query_cache[query], dtype=self._np.float32).reshape(1, -1)
        # Cosine similarity
        norms = self._np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        q_norm = self._np.linalg.norm(q_emb, axis=1, keepdims=True)
        sims = (self._embeddings @ q_emb.T) / (norms * q_norm + 1e-10)
        sims = sims.flatten()
        ranked_idx = self._np.argsort(sims)[::-1][:top_k]
        return [
            (self._chunk_ids[i], float(sims[i]))
            for i in ranked_idx
        ]

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        return self._id_to_chunk.get(chunk_id)

    def get_chunk_text(self, chunk_id: str) -> str:
        chunk = self.get_chunk(chunk_id)
        return str(chunk.get("text", "")) if chunk else ""

    def all_chunk_ids(self) -> List[str]:
        return list(self._chunk_ids)


# ---------------------------------------------------------------------------
# HybridRetriever (RRF fusion)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """RRF fusion of BM25 + Dense retrievers.

    If dense is None, falls back to BM25 only.
    """

    def __init__(
        self,
        chunks: List[dict],
        dense_model: Optional[str] = None,
        rrf_k: Optional[int] = None,
        *,
        dense_retriever: Optional[DenseRetriever] = None,
        embed_source: str = "api",
    ) -> None:
        self.bm25 = BM25Retriever(chunks)
        configured_rrf = int(load_config()["multigold_evaluation"]["rrf_k"])
        self.rrf_k = configured_rrf if rrf_k is None else rrf_k
        self._dense = dense_retriever or DenseRetriever(
            chunks, model=dense_model, embed_source=embed_source)

    def prime_queries(self, queries: List[str]) -> None:
        self._dense.prime_queries(queries)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        bm25_results = self.bm25.retrieve(query, top_k=max(top_k * 3, 30))
        if self._dense is None:
            return bm25_results[:top_k]

        dense_results = self._dense.retrieve(query, top_k=max(top_k * 3, 30))

        # RRF
        scores: Dict[str, float] = {}
        for rank, (cid, _) in enumerate(bm25_results, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (cid, _) in enumerate(dense_results, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]

    def get_chunk_text(self, chunk_id: str) -> str:
        return self.bm25.get_chunk_text(chunk_id)

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        return self.bm25.get_chunk(chunk_id)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_retriever(
    chunks: List[dict],
    strategy: str = "bm25",
    dense_model: Optional[str] = None,
    *,
    dense_retriever: Optional[DenseRetriever] = None,
    embed_source: str = "api",
) -> BM25Retriever | DenseRetriever | HybridRetriever:
    """Build a retriever from chunks.

    strategy: "bm25" | "dense" | "hybrid"
    Dense and hybrid use the configured embedding adapter unless ``dense_model``
    explicitly selects a local SentenceTransformer model, or ``embed_source``
    ("local" | "api") is set — "local" forces an offline SentenceTransformer
    and ignores any ACTIVE_EMBEDDING API configuration; "api" (default,
    backward compatible) defers to MM.get_embedding().
    """
    if strategy == "bm25":
        return BM25Retriever(chunks)
    if strategy == "dense":
        return DenseRetriever(chunks, model=dense_model, embed_source=embed_source)
    if strategy == "hybrid":
        return HybridRetriever(
            chunks,
            dense_model=dense_model,
            dense_retriever=dense_retriever,
            embed_source=embed_source,
        )
    raise ValueError(f"Unknown retriever strategy: {strategy!r}. "
                     "Choose from: bm25, dense, hybrid")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CP2 Retriever Smoke Test ===")

    # Load ecma404_json chunks
    from rag_gt.rag.loader import load_chunks, load_gt_pairs

    doc_id = "ecma404_json"
    chunks = load_chunks(doc_id)
    pairs = load_gt_pairs(doc_id)

    print(f"\n[{doc_id}] {len(chunks)} chunks, {len(pairs)} pairs")

    retriever = BM25Retriever(chunks)
    q = pairs[0]["question"]
    results = retriever.retrieve(q, top_k=10)
    print(f"\nQuery: {q[:80]}")
    print("Top-5 results:")
    for cid, score in results[:5]:
        text = retriever.get_chunk_text(cid)
        print(f"  [{score:.3f}] {cid}: {text[:60]}...")
