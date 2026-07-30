"""2+2 context clusters seeded from VERIFIED cross-page bridge pairs.

The cross-page link is never re-derived (novelty claim: the LLM never creates
the multi-hop link). A cluster = a bridge pair (fact A on page p, fact B on
page q) + one same-page reading-order neighbour per side (window <= 3, nearest
first, optional cosine guard). If either side has no qualifying neighbour the
original 2-fact bridge pair is returned as fallback — evidence is never
fabricated.

At 4 facts, leave-one-out necessity becomes a genuinely stronger check than
the pairwise single-sufficiency gate (GT_GENERATION_AUDIT_PLAN.md BUG-H / N1).
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from rag_gt.generation.neighbor_pairs import (
    cosine_matrix,
    fact_id_of,
    page_of,
    source_ordered,
)


def _nearest_neighbour(
    anchor_idx: int,
    ordered: list[dict],
    *,
    window: int,
    exclude: set[str],
    sims: Any,
    min_cosine: float,
    max_cosine: float | None = None,
) -> Optional[dict]:
    """Closest same-page fact within +-window source positions, nearest first;
    ties broken toward the earlier (lower-index) candidate. A candidate above
    ``max_cosine`` is a near-duplicate of the anchor and is skipped (the next
    in-band candidate is tried instead)."""
    page = page_of(ordered[anchor_idx])
    for dist in range(1, window + 1):
        for j in (anchor_idx - dist, anchor_idx + dist):
            if j < 0 or j >= len(ordered):
                continue
            candidate = ordered[j]
            if page_of(candidate) != page or fact_id_of(candidate) in exclude:
                continue
            if sims is not None:
                cosine = float(sims[anchor_idx, j])
                if cosine < min_cosine or (max_cosine is not None and cosine >= max_cosine):
                    continue
            return candidate
    return None


def build_clusters(
    bridge_pairs: Sequence[dict],
    facts: Sequence[dict],
    *,
    window: int = 3,
    min_cosine: float = 0.40,
    max_cosine: float | None = None,
    embed_fn: Callable[[list[str]], Any] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Returns (clusters, fallback_pairs). Fallback pairs are the original
    bridge-pair dicts, unmodified, for the plain 2-fact v1 path."""
    ordered = source_ordered(facts)
    index_of = {fact_id_of(f): i for i, f in enumerate(ordered)}
    sims = cosine_matrix(ordered, embed_fn) if (embed_fn is not None and ordered) else None

    clusters: list[dict] = []
    fallback: list[dict] = []
    for pair in bridge_pairs:
        fact_a, fact_b = str(pair.get("fact_a") or ""), str(pair.get("fact_b") or "")
        if fact_a not in index_of or fact_b not in index_of:
            fallback.append(dict(pair))
            continue
        exclude = {fact_a, fact_b}
        neighbour_a = _nearest_neighbour(
            index_of[fact_a], ordered, window=window, exclude=exclude,
            sims=sims, min_cosine=min_cosine, max_cosine=max_cosine,
        )
        neighbour_b = _nearest_neighbour(
            index_of[fact_b], ordered, window=window,
            exclude=exclude | ({fact_id_of(neighbour_a)} if neighbour_a else set()),
            sims=sims, min_cosine=min_cosine, max_cosine=max_cosine,
        )
        if neighbour_a is None or neighbour_b is None:
            fallback.append(dict(pair))
            continue
        clusters.append(
            {
                "kind": "cluster_2plus2",
                "doc": str(pair.get("doc") or ""),
                "fact_a": fact_a,
                "fact_a2": fact_id_of(neighbour_a),
                "fact_b": fact_b,
                "fact_b2": fact_id_of(neighbour_b),
                "bridge_entity": str(pair.get("bridge_entity") or ""),
                "bridge_norm": str(pair.get("bridge_norm") or ""),
                "pages": [page_of(ordered[index_of[fact_a]]), page_of(ordered[index_of[fact_b]])],
                "source_pair_id": str(pair.get("pair_id") or ""),
            }
        )
    return clusters, fallback
