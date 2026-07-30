"""CP4 — Retrieval metrics: recall@k, precision@k, MRR, NDCG@k, aggregates.

fill_metrics(mr: MatchResult) -> MatchResult (in-place, returns self)
    Computes fact_recall_at_k, hit_at_k, mrr, ndcg_at_k, coverage
    for K = [1, 3, 5, 10].

aggregate(results: List[MatchResult], pair_type_filter=None)
    -> AggregateMetrics

AggregateMetrics can be broken down by pair_type or doc_id.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from rag_gt.rag.matcher import MatchResult

K_VALUES = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# Fill per-question metrics into a MatchResult
# ---------------------------------------------------------------------------

def fill_metrics(mr: MatchResult, k_values: List[int] = K_VALUES) -> MatchResult:
    """Populate fact_recall_at_k, hit_at_k, mrr, ndcg_at_k, coverage.

    Modifies mr in-place and returns it.
    """
    n_facts = len(mr.fact_matches)
    if n_facts == 0:
        mr.fact_recall_at_k = {k: 0.0 for k in k_values}
        mr.precision_at_k = {k: 0.0 for k in k_values}
        mr.precision_rw_at_k = {k: 0.0 for k in k_values}
        mr.hit_at_k = {k: 0 for k in k_values}
        mr.mrr = 0.0
        mr.ndcg_at_k = {k: 0.0 for k in k_values}
        mr.coverage = 0.0
        return mr

    for k in k_values:
        # fact_recall@k: fraction of facts covered in top-k
        n_hit_k = sum(
            1 for fm in mr.fact_matches
            if fm.first_hit_rank is not None and fm.first_hit_rank <= k
        )
        mr.fact_recall_at_k[k] = n_hit_k / n_facts

        # precision@k: fraction of the first k retrieved chunks that cover at
        # least one gold fact. relevant_ranks is the union across gold facts,
        # so one chunk covering two facts is counted once.
        relevant_k = sum(1 for rank in mr.relevant_ranks if rank <= k)
        mr.precision_at_k[k] = relevant_k / k

        # precision_rw@k: rank-weighted precision — mean of Precision@r over
        # the relevant ranks r <= k, so a question with fewer gold-evidence
        # units than k is not structurally punished (a lone hit at rank 1
        # scores 1.0, not 1/k).
        rel_ranks_k = sorted(r for r in mr.relevant_ranks if r <= k)
        if rel_ranks_k:
            mr.precision_rw_at_k[k] = sum(
                (i + 1) / rank for i, rank in enumerate(rel_ranks_k)
            ) / len(rel_ranks_k)
        else:
            mr.precision_rw_at_k[k] = 0.0

        # hit@k: 1 only if ALL facts are covered in top-k
        mr.hit_at_k[k] = 1 if n_hit_k == n_facts else 0

    # MRR: reciprocal rank of first chunk that covers ANY fact
    first_ranks = [
        fm.first_hit_rank
        for fm in mr.fact_matches
        if fm.first_hit_rank is not None
    ]
    if first_ranks:
        mr.mrr = 1.0 / min(first_ranks)
    else:
        mr.mrr = 0.0

    # NDCG@k: binary relevance per chunk position
    # rel[rank] = 1 if that ranked chunk covers at least 1 fact, else 0
    # Build per-rank relevance for each chunk in the ranked list
    for k in k_values:
        mr.ndcg_at_k[k] = _ndcg_at_k(mr, k)

    # Coverage = fact_recall@(max k) — mirrors evaluate.py "coverage" metric
    max_k = max(k_values)
    mr.coverage = mr.fact_recall_at_k.get(max_k, 0.0)

    return mr


def _ndcg_at_k(mr: MatchResult, k: int) -> float:
    """Compute NDCG@k with binary relevance."""
    ranked_ids = mr.retrieved_chunk_ids[:k]
    if not ranked_ids:
        return 0.0

    # A rank position is relevant if that chunk covers >= 1 fact
    fact_hit_ranks = {rank for rank in mr.relevant_ranks if rank <= k}

    # DCG
    dcg = 0.0
    for i, rank in enumerate(range(1, len(ranked_ids) + 1)):
        rel = 1.0 if rank in fact_hit_ranks else 0.0
        dcg += rel / math.log2(rank + 1)

    # Ideal DCG: put all hits at the top
    n_hits = len(fact_hit_ranks)
    idcg = 0.0
    for i in range(1, min(n_hits, k) + 1):
        idcg += 1.0 / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# AggregateMetrics
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    n_pairs: int
    mean_fact_recall_at_k: Dict[int, float] = field(default_factory=dict)
    mean_precision_at_k: Dict[int, float] = field(default_factory=dict)
    mean_precision_rw_at_k: Dict[int, float] = field(default_factory=dict)
    mean_hit_at_k: Dict[int, float] = field(default_factory=dict)
    mean_mrr: float = 0.0
    mean_ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mean_coverage: float = 0.0

    # Breakdowns
    by_pair_type: Dict[str, "AggregateMetrics"] = field(default_factory=dict)

    def summary(self) -> dict:
        out = {
            "n_pairs": self.n_pairs,
            "mrr": round(self.mean_mrr, 4),
            "coverage": round(self.mean_coverage, 4),
        }
        for k in sorted(self.mean_fact_recall_at_k):
            out[f"fact_recall@{k}"] = round(self.mean_fact_recall_at_k[k], 4)
        for k in sorted(self.mean_precision_at_k):
            out[f"precision@{k}"] = round(self.mean_precision_at_k[k], 4)
        for k in sorted(self.mean_precision_rw_at_k):
            out[f"precision_rw@{k}"] = round(self.mean_precision_rw_at_k[k], 4)
        for k in sorted(self.mean_hit_at_k):
            out[f"hit@{k}"] = round(self.mean_hit_at_k[k], 4)
        for k in sorted(self.mean_ndcg_at_k):
            out[f"ndcg@{k}"] = round(self.mean_ndcg_at_k[k], 4)
        return out


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def aggregate(
    results: List[MatchResult],
    k_values: List[int] = K_VALUES,
    _breakdown: bool = True,
) -> AggregateMetrics:
    """Aggregate metrics across a list of MatchResult objects.

    Results must have already had fill_metrics() called on them.
    """
    if not results:
        am = AggregateMetrics(n_pairs=0)
        am.mean_fact_recall_at_k = {k: 0.0 for k in k_values}
        am.mean_precision_at_k = {k: 0.0 for k in k_values}
        am.mean_precision_rw_at_k = {k: 0.0 for k in k_values}
        am.mean_hit_at_k = {k: 0.0 for k in k_values}
        am.mean_ndcg_at_k = {k: 0.0 for k in k_values}
        return am

    am = AggregateMetrics(n_pairs=len(results))
    am.mean_mrr = _mean([r.mrr for r in results])
    am.mean_coverage = _mean([r.coverage for r in results])

    for k in k_values:
        am.mean_fact_recall_at_k[k] = _mean([r.fact_recall_at_k.get(k, 0.0) for r in results])
        am.mean_precision_at_k[k] = _mean([r.precision_at_k.get(k, 0.0) for r in results])
        am.mean_precision_rw_at_k[k] = _mean([r.precision_rw_at_k.get(k, 0.0) for r in results])
        am.mean_hit_at_k[k] = _mean([float(r.hit_at_k.get(k, 0)) for r in results])
        am.mean_ndcg_at_k[k] = _mean([r.ndcg_at_k.get(k, 0.0) for r in results])

    # Breakdown by pair_type — one level deep only (no further breakdown)
    if _breakdown:
        pair_types = sorted({r.pair_type for r in results})
        for pt in pair_types:
            subset = [r for r in results if r.pair_type == pt]
            am.by_pair_type[pt] = aggregate(subset, k_values=k_values, _breakdown=False)

    return am


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CP4 Metrics Smoke Test ===")

    from rag_gt.rag.loader import load_chunks, load_gt_pairs
    from rag_gt.rag.retriever import BM25Retriever
    from rag_gt.rag.matcher import match_pair

    doc_id = "din_iso_15609_welding_procedure"
    chunks = load_chunks(doc_id)
    pairs = load_gt_pairs(doc_id)
    retriever = BM25Retriever(chunks)
    id_to_text = {c["chunk_id"]: c["text"] for c in chunks}

    results = []
    for pair in pairs:
        ranked = retriever.retrieve(pair["question"], top_k=10)
        mr = match_pair(pair, ranked, id_to_text)
        fill_metrics(mr)
        results.append(mr)

    agg = aggregate(results)
    print(f"\n[{doc_id}] {agg.n_pairs} pairs")
    s = agg.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")

    print("\n  By pair_type:")
    for pt, sub_agg in sorted(agg.by_pair_type.items()):
        ss = sub_agg.summary()
        print(f"    [{pt}] n={ss['n_pairs']} fr@5={ss.get('fact_recall@5',0):.3f} "
              f"mrr={ss['mrr']:.3f}")
