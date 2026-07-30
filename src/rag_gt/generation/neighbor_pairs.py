"""Deterministic same-page neighbour-pair sampler for evidence-dense singles.

A v2 "single-hop" QA is grounded in TWO facts adjacent in reading order on the
same page (window <= 3 source positions — the same neighbourhood rule as
graph.pair_scheduler._collect_source_neighbours). The generated question must
need both facts, so each single-hop row carries two answer clauses = two
retrievable evidence units, lifting the raw precision@5 ceiling from 0.2 to
0.4 and making rank-weighted precision meaningful.

Greedy and disjoint: each fact joins at most one pair (nearest neighbour
first), so overlapping pairs cannot spawn near-duplicate questions. Zero LLM;
the optional ``embed_fn`` cosine guard keeps only pairs questionable as one
unit.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence


def page_of(record: dict) -> Optional[int]:
    value = record.get("page_start", record.get("page"))
    return int(value) if value is not None else None


def char_start_of(record: dict) -> int:
    value = record.get("char_start")
    return int(value) if value is not None else 0


def fact_id_of(record: dict) -> str:
    return str(record.get("fact_id") or record.get("id") or "")


def fact_text_of(record: dict) -> str:
    return str(record.get("canonical_form") or record.get("text") or "")


def source_ordered(facts: Sequence[dict]) -> list[dict]:
    """Reading order: (page, char offset, fact id) — the pair_scheduler rule."""
    return sorted(
        (f for f in facts if fact_id_of(f) and page_of(f) is not None),
        key=lambda f: (page_of(f), char_start_of(f), fact_id_of(f)),
    )


def cosine_matrix(ordered: Sequence[dict], embed_fn: Callable[[list[str]], Any]):
    import numpy as np

    vectors = np.asarray(embed_fn([fact_text_of(f) for f in ordered]), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    return unit @ unit.T


def sample_neighbor_pairs(
    facts: Sequence[dict],
    *,
    doc: str = "",
    window: int = 3,
    min_cosine: float = 0.40,
    max_cosine: float | None = None,
    embed_fn: Callable[[list[str]], Any] | None = None,
    max_pairs: int | None = None,
    max_uses_per_fact: int = 1,
) -> list[dict]:
    """``max_cosine`` (pilot-derived): a neighbour above the cap is a
    near-duplicate restatement — one fact would entail the whole answer and
    the pair would rightly die at the joint-necessity gate. The band keeps
    neighbours related enough to question together, distinct enough that
    both are needed.

    ``max_uses_per_fact`` relaxes the disjointness rule: at 1 (default) each
    fact joins at most one pair; at 2 a fact may join a second pair, roughly
    doubling single-hop volume on dense pages. The duplicate-question gate
    downstream still rejects near-identical questions."""
    ordered = source_ordered(facts)
    sims = cosine_matrix(ordered, embed_fn) if (embed_fn is not None and ordered) else None

    uses: dict[int, int] = {}
    pairs: list[dict] = []
    for i, fact_a in enumerate(ordered):
        if uses.get(i, 0) >= max_uses_per_fact:
            continue
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
        for j in range(i + 1, min(i + 1 + window, len(ordered))):
            if uses.get(j, 0) >= max_uses_per_fact or page_of(ordered[j]) != page_of(fact_a):
                continue
            cosine = float(sims[i, j]) if sims is not None else None
            if cosine is not None and (
                cosine < min_cosine or (max_cosine is not None and cosine >= max_cosine)
            ):
                continue
            fact_b = ordered[j]
            pairs.append(
                {
                    "kind": "neighbor_pair",
                    "doc": doc or str(fact_a.get("doc") or ""),
                    "fact_a": fact_id_of(fact_a),
                    "fact_b": fact_id_of(fact_b),
                    "page": page_of(fact_a),
                    "distance": j - i,
                    "cosine": round(cosine, 4) if cosine is not None else None,
                }
            )
            uses[i] = uses.get(i, 0) + 1
            uses[j] = uses.get(j, 0) + 1
            break
    return pairs
