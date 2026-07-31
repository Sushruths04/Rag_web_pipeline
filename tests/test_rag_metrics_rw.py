"""Rank-weighted precision@k for the no-API eval pipeline.

Raw precision@k punishes questions whose gold evidence has fewer units than k
(a perfect 1-evidence retrieval caps at 1/k). precision_rw@k averages
Precision@r over the relevant ranks r <= k instead, so rank-1 perfection
scores 1.0 regardless of k. Same semantics as
production/backend/app/evaluation/matching.py::precision_rw restricted to the
top-k prefix.
"""
from rag_gt.rag.matcher import MatchResult
from rag_gt.rag.metrics import aggregate, fill_metrics


def _mr(relevant_ranks, n_facts=1, first_hit=None):
    return MatchResult(
        question="q", pair_type="single", depth=1, page_spread=0,
        necessity_score=1.0, n_facts=n_facts,
        retrieved_chunk_ids=[f"c{i}" for i in range(1, 11)],
        fact_matches=[_fm(first_hit)] * n_facts,
        relevant_ranks=list(relevant_ranks),
    )


def _fm(first_hit):
    from rag_gt.rag.matcher import FactMatch
    return FactMatch(fact_id="f1", fact_text="t", hit=first_hit is not None,
                     first_hit_rank=first_hit, best_overlap=1.0)


def test_precision_rw_single_relevant_at_rank_one_is_perfect():
    mr = fill_metrics(_mr([1], first_hit=1))
    assert mr.precision_rw_at_k[5] == 1.0
    assert mr.precision_at_k[5] == 1 / 5  # raw metric unchanged


def test_precision_rw_single_relevant_at_rank_three():
    mr = fill_metrics(_mr([3], first_hit=3))
    assert abs(mr.precision_rw_at_k[5] - 1 / 3) < 1e-9
    assert abs(mr.precision_at_k[5] - 0.2) < 1e-9


def test_precision_rw_relevant_at_one_and_four():
    mr = fill_metrics(_mr([1, 4], n_facts=2, first_hit=1))
    # P@1 = 1/1, P@4 = 2/4 -> mean = 0.75
    assert abs(mr.precision_rw_at_k[5] - 0.75) < 1e-9


def test_precision_rw_none_relevant_is_zero():
    mr = fill_metrics(_mr([], first_hit=None))
    assert mr.precision_rw_at_k[5] == 0.0


def test_aggregate_exposes_precision_rw():
    results = [fill_metrics(_mr([1], first_hit=1)), fill_metrics(_mr([3], first_hit=3))]
    summary = aggregate(results).summary()
    assert abs(summary["precision_rw@5"] - (1.0 + 1 / 3) / 2) < 1e-4
    assert "precision@5" in summary
