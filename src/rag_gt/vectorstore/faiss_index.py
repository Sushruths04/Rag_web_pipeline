"""Fact similarity index. FAISS IndexFlatIP primary; sklearn fallback.

Inputs are L2-normalized defensively in `add()` and `search()` so the FAISS
(`IndexFlatIP`) and sklearn (`metric="cosine"`) backends produce numerically
identical scores on the same inputs.

`add()` *appends*. To re-build from scratch, call `reset()` first.

Save/load uses sibling paths derived by appending `.npy` and `.json` to the
input path's name — `pathlib.Path.with_suffix` was avoided because it
misinterprets directories with dots in the name (e.g. `D:\\v1.0\\index`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from rag_gt.core.config import load_config

_cfg = load_config()
_BACKEND = _cfg.get("vectorstore", {}).get("backend", "faiss").lower()


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except ImportError:
        return False


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return (matrix / norms).astype(np.float32)


def _sibling(path: Path, suffix: str) -> Path:
    """Return `path.parent / (path.name + suffix)`. Avoids `with_suffix` issues
    when the path's parent directory contains a dot."""
    return path.parent / (path.name + suffix)


class FactIndex:
    """Exact cosine search over L2-normalized fact embeddings."""

    def __init__(self, dim: int = 768, backend: Optional[str] = None) -> None:
        self.dim = int(dim)
        self.backend = (backend or _BACKEND).lower()
        if self.backend == "faiss" and not _faiss_available():
            self.backend = "sklearn"
        self.fact_ids: List[str] = []
        self._id_to_idx: dict[str, int] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._index = None
        self._nn = None

    # ---------- mutation ----------

    def reset(self) -> None:
        """Clear all state. Call before re-building from scratch."""
        self.fact_ids = []
        self._id_to_idx = {}
        self._embeddings = None
        self._index = None
        self._nn = None

    def add(self, embeddings: np.ndarray, fact_ids: List[str]) -> None:
        """Append `(embeddings, fact_ids)` to the index. Subsequent calls extend
        rather than replace state."""
        if embeddings.shape[0] != len(fact_ids):
            raise ValueError(
                f"shape mismatch: embeddings={embeddings.shape[0]} vs fact_ids={len(fact_ids)}"
            )
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"dim mismatch: index={self.dim} vs embeddings={embeddings.shape[1]}"
            )
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("embeddings contain non-finite values")
        embeddings = _l2_normalize(embeddings.astype(np.float32))

        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

        for i, fid in enumerate(fact_ids):
            if fid in self._id_to_idx:
                raise ValueError(f"duplicate fact_id: {fid!r}")
            self._id_to_idx[fid] = len(self.fact_ids) + i
        self.fact_ids.extend(fact_ids)

        # Rebuild backend index. (FAISS IndexFlatIP supports incremental .add,
        # but sklearn's NearestNeighbors does not — re-fitting on the full set
        # keeps both backends behaviorally identical.)
        if self.backend == "faiss":
            import faiss

            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(self._embeddings)
        else:
            from sklearn.neighbors import NearestNeighbors

            self._nn = NearestNeighbors(metric="cosine", algorithm="brute")
            self._nn.fit(self._embeddings)

    # ---------- query ----------

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if self._embeddings is None or not self.fact_ids:
            return []
        q = query.reshape(1, -1).astype(np.float32)
        q = _l2_normalize(q)
        k = max(1, min(k, len(self.fact_ids)))

        if self.backend == "faiss":
            scores, indices = self._index.search(q, k)
            return [
                (self.fact_ids[i], float(s))
                for s, i in zip(scores[0], indices[0])
                if i >= 0
            ]
        dists, indices = self._nn.kneighbors(q, n_neighbors=k)
        # `dists` from sklearn cosine = 1 - cos_sim. With normalized inputs, this
        # range is [0, 2]; we return the cosine similarity directly.
        return [
            (self.fact_ids[i], float(1.0 - d))
            for d, i in zip(dists[0], indices[0])
        ]

    def embedding_for(self, fact_id: str) -> np.ndarray:
        idx = self._id_to_idx.get(fact_id)
        if idx is None:
            raise KeyError(f"unknown fact_id: {fact_id!r}")
        if self._embeddings is None:
            raise RuntimeError("FactIndex.embedding_for called before add()")
        return self._embeddings[idx]

    # ---------- persistence ----------

    def save(self, path: Path | str) -> None:
        if self._embeddings is None:
            raise RuntimeError("FactIndex.save called before add()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(_sibling(path, ".npy"), self._embeddings)
        meta = {
            "fact_ids": self.fact_ids,
            "backend": self.backend,
            "dim": self.dim,
        }
        with open(_sibling(path, ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path | str) -> "FactIndex":
        path = Path(path)
        # numpy auto-appends .npy when saving stem paths; handle both shapes.
        npy_path = _sibling(path, ".npy")
        if not npy_path.exists() and path.suffix == ".npy":
            npy_path = path
        embeddings = np.load(npy_path)
        if embeddings.ndim != 2:
            raise ValueError(f"expected 2-D embedding array, got shape {embeddings.shape}")
        meta_path = _sibling(path, ".json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        saved_backend = meta.get("backend")
        if saved_backend and saved_backend != _BACKEND:
            logger.warning(
                f"[FactIndex] saved with backend={saved_backend!r}, loading with {_BACKEND!r}; "
                "scores may differ slightly across backends."
            )
        idx = cls(dim=int(meta.get("dim", embeddings.shape[1])), backend=saved_backend)
        idx.add(embeddings, meta["fact_ids"])
        return idx
