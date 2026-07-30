"""Fact embedding via the shared BGE encoder from core.models.MM."""

from __future__ import annotations

from typing import List

import numpy as np

from rag_gt.core.models import MM
from rag_gt.core.types import Fact


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row. Defensive — `model.encode(..., normalize_embeddings=True)`
    already normalizes, but callers that pass raw vectors should still come out clean."""
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Encode `texts`. Empty input returns a zero-row array of the encoder's
    actual dim — never a hard-coded 768."""
    dim = MM.embedding_dim()
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    model = MM.get_embedding()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = vectors.astype(np.float32)
    # Belt-and-braces: ensure unit norm even if the encoder skipped normalization.
    return _normalize(vectors)


def embed_facts(facts: List[Fact], batch_size: int = 32) -> np.ndarray:
    return embed_texts([f.text for f in facts], batch_size=batch_size)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
