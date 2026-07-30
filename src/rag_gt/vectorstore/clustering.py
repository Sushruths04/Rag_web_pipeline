"""Agglomerative / k-means clustering over fact embeddings (cosine distance)."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def cluster_facts(
    embeddings: np.ndarray,
    fact_ids: List[str],
    method: str = "agglomerative",
    cosine_threshold: float = 0.55,
    k: Optional[int] = None,
) -> Dict[int, List[str]]:
    if len(fact_ids) == 0:
        return {}
    if embeddings.shape[0] != len(fact_ids):
        raise ValueError(
            f"shape mismatch: embeddings={embeddings.shape[0]} vs fact_ids={len(fact_ids)}"
        )
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("embeddings contain non-finite values")
    if np.any(np.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError("embeddings contain zero-norm rows; cosine distance is undefined")
    if len(fact_ids) == 1:
        return {0: [fact_ids[0]]}

    method = method.lower()
    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        distance_threshold = 1.0 - cosine_threshold
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels = clusterer.fit_predict(embeddings)
    elif method == "kmeans":
        from sklearn.cluster import KMeans

        n = k if k is not None else max(2, min(len(fact_ids) // 10, 20))
        n = max(1, min(n, len(fact_ids)))  # clamp so KMeans never gets n>n_samples
        clusterer = KMeans(n_clusters=n, n_init=10, random_state=42)
        labels = clusterer.fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    clusters: Dict[int, List[str]] = {}
    for label, fid in zip(labels, fact_ids):
        clusters.setdefault(int(label), []).append(fid)
    return clusters
