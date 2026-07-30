"""Inference-safe cross-encoder and bridge-complementarity reranking.

The bridge-complementarity mode uses only the user question and retrieved
chunks. It never reads gold facts, answers, bridge labels, or evaluation hits.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np

from rag_gt.core.config import load_config

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "what", "when", "which", "with",
}


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.casefold()) if token not in _STOP}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


class CrossEncoderScorer:
    """Shared score cache so multiple reranking ablations pay once per pair."""

    def __init__(self, model: object | None = None) -> None:
        cfg = load_config()["multigold_evaluation"]
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(str(cfg["reranker_model"]))
        self.model = model
        self.batch_size = int(cfg["reranker_batch_size"])
        self._cache: dict[tuple[str, str], float] = {}

    def score(
        self,
        query: str,
        chunk_ids: Sequence[str],
        text_by_id: Mapping[str, str],
    ) -> list[float]:
        missing = [
            chunk_id
            for chunk_id in chunk_ids
            if (query, chunk_id) not in self._cache
        ]
        if missing:
            pairs = [(query, text_by_id[chunk_id]) for chunk_id in missing]
            values = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            for chunk_id, value in zip(missing, values):
                self._cache[(query, chunk_id)] = float(np.asarray(value).reshape(-1)[0])
        return [self._cache[(query, chunk_id)] for chunk_id in chunk_ids]


class CrossEncoderReranker:
    """Rerank a hybrid candidate pool by relevance or complementarity."""

    def __init__(
        self,
        base_retriever: object,
        scorer: CrossEncoderScorer,
        *,
        mode: str = "relevance",
    ) -> None:
        if mode not in {"relevance", "bridge_complementarity"}:
            raise ValueError(f"Unknown reranking mode: {mode}")
        cfg = load_config()["multigold_evaluation"]
        self.base = base_retriever
        self.scorer = scorer
        self.mode = mode
        self.candidate_k = int(cfg["reranker_candidate_k"])
        self.weights = {
            key: float(value)
            for key, value in cfg["bridge_reranker_weights"].items()
        }

    def prime_queries(self, queries: list[str]) -> None:
        prime = getattr(self.base, "prime_queries", None)
        if callable(prime):
            prime(queries)
        for query in queries:
            candidates = self.base.retrieve(query, top_k=self.candidate_k)
            ids = [chunk_id for chunk_id, _ in candidates]
            texts = {chunk_id: self.get_chunk_text(chunk_id) for chunk_id in ids}
            self.scorer.score(query, ids, texts)

    def get_chunk(self, chunk_id: str):
        return self.base.get_chunk(chunk_id)

    def get_chunk_text(self, chunk_id: str) -> str:
        return self.base.get_chunk_text(chunk_id)

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        candidates = self.base.retrieve(
            query, top_k=max(top_k, self.candidate_k)
        )
        if not candidates:
            return []
        ids = [chunk_id for chunk_id, _ in candidates]
        text_by_id = {chunk_id: self.get_chunk_text(chunk_id) for chunk_id in ids}
        scores = self.scorer.score(query, ids, text_by_id)
        if self.mode == "relevance":
            ranked = sorted(zip(ids, scores), key=lambda item: item[1], reverse=True)
            return ranked[:top_k]
        return self._complementarity_rank(query, ids, scores, text_by_id, top_k)

    def _complementarity_rank(
        self,
        query: str,
        ids: list[str],
        scores: list[float],
        text_by_id: Mapping[str, str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        # Rank-normalized relevance avoids assuming a calibration range for the
        # selected cross-encoder model.
        relevance_order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
        # Keep every cross-encoder candidate materially relevant; the lower
        # half of the objective is reserved for complementarity signals.
        denominator = max(1, 2 * len(ids))
        relevance = {
            ids[index]: 1.0 - rank / denominator
            for rank, index in enumerate(relevance_order)
        }
        query_tokens = _tokens(query)
        chunk_tokens = {chunk_id: _tokens(text_by_id[chunk_id]) for chunk_id in ids}
        remaining = set(ids)
        selected: list[str] = []
        covered_query: set[str] = set()
        selected_pages: set[int] = set()
        objective: dict[str, float] = {}

        while remaining and len(selected) < top_k:
            best_id = ""
            best_score = float("-inf")
            for chunk_id in remaining:
                tokens = chunk_tokens[chunk_id]
                redundancy = max(
                    (_jaccard(tokens, chunk_tokens[other]) for other in selected),
                    default=0.0,
                )
                new_query = len((tokens & query_tokens) - covered_query) / max(
                    1, len(query_tokens)
                )
                chunk = self.get_chunk(chunk_id) or {}
                page = int(chunk.get("page_start", chunk.get("page", 0)) or 0)
                page_gain = 1.0 if page and page not in selected_pages else 0.0
                score = (
                    self.weights["relevance"] * relevance[chunk_id]
                    + self.weights["novelty"] * (1.0 - redundancy)
                    + self.weights["query_coverage"] * new_query
                    + self.weights["page_diversity"] * page_gain
                )
                if score > best_score:
                    best_id, best_score = chunk_id, score
            selected.append(best_id)
            remaining.remove(best_id)
            objective[best_id] = best_score
            covered_query.update(chunk_tokens[best_id] & query_tokens)
            chunk = self.get_chunk(best_id) or {}
            page = int(chunk.get("page_start", chunk.get("page", 0)) or 0)
            if page:
                selected_pages.add(page)
        return [(chunk_id, objective[chunk_id]) for chunk_id in selected]
