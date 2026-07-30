"""Proportional single-hop budget allocation across documents, plus
assembly-time dedup of near-identical QA pairs.

Multi-hop pairs (clusters + 2-fact bridges) are always kept in full — they
are the scarce, expensive product. When a total-dataset cap is set, the
remaining budget goes to single-hop pairs proportionally to each document's
own single yield (largest-remainder rounding), so document size self-adjusts:
a 20-page standard contributes few singles, a 500-page book contributes many.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence


def allocate_singles(counts: dict[str, int], budget: int) -> dict[str, int]:
    """Split ``budget`` single-hop slots across docs proportionally to
    ``counts`` (each doc's available singles), never exceeding a doc's own
    supply. Largest-remainder rounding; deterministic."""
    total = sum(counts.values())
    if budget >= total:
        return dict(counts)
    if budget <= 0 or total == 0:
        return {doc: 0 for doc in counts}

    quotas = {doc: budget * n / total for doc, n in counts.items()}
    alloc = {doc: min(counts[doc], int(q)) for doc, q in quotas.items()}
    remainder = budget - sum(alloc.values())
    order = sorted(counts, key=lambda d: (quotas[d] - int(quotas[d]), counts[d], d),
                   reverse=True)
    while remainder > 0:
        progressed = False
        for doc in order:
            if remainder <= 0:
                break
            if alloc[doc] < counts[doc]:
                alloc[doc] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            break
    return alloc


# ---------------------------------------------------------------------------
# Assembly-time dedup: shared-evidence + near-dupe question (T4)
# ---------------------------------------------------------------------------

EmbedFn = Callable[[list], Sequence[Sequence[float]]]


def _evidence_key(qa: Mapping[str, object]) -> str:
    return "|".join(sorted(str(f) for f in qa.get("gold_fact_ids") or []))


def _evidence_unit_count(qa: Mapping[str, object]) -> int:
    clauses = qa.get("answer_clauses") or qa.get("gold_fact_ids") or []
    return len(clauses)


def _min_clause_nli(qa: Mapping[str, object]) -> float:
    clauses = qa.get("answer_clauses") or []
    scores = [float(c.get("nli", 0.0)) for c in clauses]
    return min(scores) if scores else 0.0


def _keep_rank(qa: Mapping[str, object]) -> tuple:
    """Sort key: HIGHER is better (more evidence units, then higher min
    clause NLI). Used both for shared-evidence collapse and for picking the
    near-dupe question survivor."""
    return (_evidence_unit_count(qa), _min_clause_nli(qa))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def dedup_pairs(
    pairs: Sequence[Mapping[str, object]],
    *,
    embed_fn: EmbedFn | None = None,
    cosine_threshold: float = 0.92,
) -> tuple[list[Mapping[str, object]], dict[str, int]]:
    """Drop near-duplicate QA pairs at final assembly, scoped per document.

    Two passes, both keeping the record with more evidence units (answer
    clauses), tie-broken by higher min clause NLI:

    (a) shared-evidence — records in the same doc whose sorted
        ``gold_fact_ids`` set is identical collapse to one survivor.
    (b) near-dupe question — among the (a) survivors, records in the same
        doc whose question embeddings have cosine similarity >=
        ``cosine_threshold`` collapse to one survivor. Requires ``embed_fn``
        (a callable mapping a list of question strings to a list of
        embedding vectors); if ``embed_fn`` is None this pass is skipped.

    Returns ``(kept_pairs, stats)`` with
    ``stats = {"n_dupes_dropped_evidence": int, "n_dupes_dropped_question": int}``.
    Original relative order is preserved among survivors.
    """
    stats = {"n_dupes_dropped_evidence": 0, "n_dupes_dropped_question": 0}
    if not pairs:
        return list(pairs), stats

    # ---- (a) shared-evidence collapse, grouped by (doc, evidence key) -----
    groups: dict[tuple, list[int]] = {}
    for i, qa in enumerate(pairs):
        key = qa.get("gold_fact_ids") or []
        doc = qa.get("doc", "")
        group_key = (doc, _evidence_key(qa)) if key else (doc, f"__nogold_{i}__")
        groups.setdefault(group_key, []).append(i)

    survivors_a: list[int] = []
    for idxs in groups.values():
        if len(idxs) == 1:
            survivors_a.append(idxs[0])
            continue
        winner = max(idxs, key=lambda i: (_keep_rank(pairs[i]), -i))
        stats["n_dupes_dropped_evidence"] += len(idxs) - 1
        survivors_a.append(winner)
    survivors_a.sort()
    stage_a = [pairs[i] for i in survivors_a]

    # ---- (b) near-dupe question collapse, grouped by doc ------------------
    if embed_fn is None or len(stage_a) < 2:
        return stage_a, stats

    by_doc: dict[object, list[int]] = {}
    for i, qa in enumerate(stage_a):
        by_doc.setdefault(qa.get("doc", ""), []).append(i)

    dropped_b: set[int] = set()
    for idxs in by_doc.values():
        if len(idxs) < 2:
            continue
        questions = [str(stage_a[i].get("question", "")) for i in idxs]
        vectors = list(embed_fn(questions))
        # process best-quality first so the highest-ranked record in a
        # near-dupe cluster is always the one that survives
        order = sorted(range(len(idxs)), key=lambda k: _keep_rank(stage_a[idxs[k]]),
                       reverse=True)
        kept_local: list[tuple[int, Sequence[float]]] = []
        for k in order:
            vec = vectors[k]
            is_dup = any(_cosine(vec, kv) >= cosine_threshold for _, kv in kept_local)
            if is_dup:
                dropped_b.add(idxs[k])
            else:
                kept_local.append((k, vec))

    stats["n_dupes_dropped_question"] = len(dropped_b)
    final = [qa for i, qa in enumerate(stage_a) if i not in dropped_b]
    return final, stats
