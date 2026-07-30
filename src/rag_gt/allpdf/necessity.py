"""Leave-one-out necessity scoring for multi-hop Q&A pairs (all-PDF, Phase 2).

A genuine multi-hop pair requires every fact to be load-bearing: removing any
single fact should make the answer no longer recoverable from the remaining
facts. This module measures that with a leave-one-out NLI check.

Uses the local, SQLite-cached cross-encoder NLI (`validation.nli_check.nli_batch`)
so it costs **no external API calls**. The model is loaded lazily on first use.

necessity_score = (#facts whose removal drops answer-entailment below threshold)
                  / (#facts)
A score of 1.0 means every fact is necessary (strict multi-hop). Values near 0
mean the answer survives removing facts — i.e. the pair is effectively
single-hop despite using multiple facts.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from rag_gt.core.types import Fact


def _fact_text(fact: Fact) -> str:
    return (fact.canonical_form.strip() or fact.text.strip())


def leave_one_out_necessity(
    answer: str,
    facts: List[Fact],
    *,
    threshold: Optional[float] = None,
) -> Optional[dict]:
    """Return necessity metrics for an (answer, facts) pair, or None if N/A.

    Returns None when the check does not apply (no answer, abstention answer,
    or fewer than 2 facts). Otherwise returns:
      {necessity_score, necessary_fact_ids, loo_entailment: [per-fact score]}
    where loo_entailment[i] is the entailment of `answer` by all facts EXCEPT
    fact i; a low value means fact i is load-bearing.
    """
    if not answer or "insufficient information" in answer.lower():
        return None
    if len(facts) < 2:
        return None

    from rag_gt.validation.nli_check import nli_batch, GT_CONSISTENCY_THRESHOLD

    thr = GT_CONSISTENCY_THRESHOLD if threshold is None else threshold

    pairs = []
    for i in range(len(facts)):
        others = " ".join(_fact_text(f) for j, f in enumerate(facts) if j != i)
        pairs.append((others, answer))

    scores = nli_batch(pairs)
    necessary_ids: List[str] = []
    for i, score in enumerate(scores):
        if score < thr:
            necessary_ids.append(facts[i].fact_id)

    necessity_score = len(necessary_ids) / len(facts)
    return {
        "necessity_score": round(necessity_score, 3),
        "necessary_fact_ids": necessary_ids,
        "loo_entailment": [round(float(s), 3) for s in scores],
    }


def leave_one_out_necessity_batch(
    items: Sequence[Tuple[str, Sequence[Fact]]],
    *,
    threshold: Optional[float] = None,
) -> List[Optional[dict]]:
    """Batched leave-one-out necessity across MULTIPLE (answer, facts)
    candidates (Phase 1 debug plan T8, fixes F8).

    ``leave_one_out_necessity`` already batches the N leave-one-out checks
    for a SINGLE candidate into one ``nli_batch`` call; scoring a whole
    group of candidates (e.g. every cluster in a doc) still meant one
    ``nli_batch`` round-trip PER candidate. This collects every eligible
    candidate's leave-one-out pairs first and scores the entire group in
    ONE ``nli_batch`` call, then slices the results back per candidate.

    Returns one entry per item, in the same order as ``items``: either the
    same dict shape as ``leave_one_out_necessity`` returns, or None for an
    item that does not apply (empty/abstention answer, or fewer than 2
    facts) -- identical semantics to calling ``leave_one_out_necessity``
    once per item, just without the extra round-trips.
    """
    from rag_gt.validation.nli_check import nli_batch, GT_CONSISTENCY_THRESHOLD

    thr = GT_CONSISTENCY_THRESHOLD if threshold is None else threshold

    eligible_indices: List[int] = []
    all_pairs: List[Tuple[str, str]] = []
    spans: List[Tuple[int, int]] = []
    for idx, (answer, facts) in enumerate(items):
        if not answer or "insufficient information" in answer.lower() or len(facts) < 2:
            continue
        start = len(all_pairs)
        for i in range(len(facts)):
            others = " ".join(_fact_text(f) for j, f in enumerate(facts) if j != i)
            all_pairs.append((others, answer))
        spans.append((start, len(all_pairs)))
        eligible_indices.append(idx)

    scores = nli_batch(all_pairs) if all_pairs else []

    results: List[Optional[dict]] = [None] * len(items)
    for pos, idx in enumerate(eligible_indices):
        start, end = spans[pos]
        item_scores = scores[start:end]
        facts = items[idx][1]
        necessary_ids = [f.fact_id for f, s in zip(facts, item_scores) if s < thr]
        results[idx] = {
            "necessity_score": round(len(necessary_ids) / len(facts), 3),
            "necessary_fact_ids": necessary_ids,
            "loo_entailment": [round(float(s), 3) for s in item_scores],
        }
    return results
