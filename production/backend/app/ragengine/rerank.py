"""Cross-encoder reranking on top of any base retriever's candidates."""
from __future__ import annotations

from typing import Any, Callable


class Reranker:
    """Re-scores (query, chunk_text) candidate pairs with a cross-encoder.

    `model` may be a model name (loaded via sentence_transformers.CrossEncoder)
    or any object with `.predict(list[tuple[str, str]]) -> list[float]`
    (injection point for tests).
    """

    def __init__(self, model: str | Any = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        if isinstance(model, str):
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model)
            self.model_name = model
        else:
            self._model = model
            self.model_name = type(model).__name__

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        get_text: Callable[[str], str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Re-order candidate (chunk_id, score) pairs by cross-encoder score."""
        if not candidates:
            return []
        pairs = [(query, get_text(cid)) for cid, _ in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip((cid for cid, _ in candidates), (float(s) for s in scores)),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return ranked[:top_k]
