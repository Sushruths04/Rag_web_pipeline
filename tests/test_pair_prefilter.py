"""Tests for V16.2 L0 pair pre-filter."""

from __future__ import annotations

import pytest

from rag_gt.core.types import Fact, Span
from rag_gt.graph.pair_prefilter import (
    PairPrefilterConfig,
    prefilter_pair,
    prefilter_pairs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _span(page: int) -> Span:
    return Span(doc_id="d1", chunk_id="c1", start_token=0, end_token=10, page_start=page)


def _fact(
    fid: str,
    text: str,
    role: str = "definition",
    page: int | None = None,
    canonical_form: str = "",
) -> Fact:
    spans = [_span(page)] if page is not None else []
    return Fact(
        fact_id=fid,
        text=text,
        role=role,
        supporting_spans=spans,
        canonical_form=canonical_form,
    )


_LONG_A = "The transmission control protocol ensures reliable delivery of data packets."
_LONG_B = "Flow control mechanisms prevent buffer overflow at the receiver side."
_SHARED_B = "The transmission control protocol also provides flow control mechanisms."

_CFG = PairPrefilterConfig()


# ---------------------------------------------------------------------------
# Hard reject: short_fact
# ---------------------------------------------------------------------------

def test_short_fact_a_rejected() -> None:
    short = _fact("s1", "Short text.")  # < 30 chars but also < 10 fails Fact validation
    # Use exactly 10-char text but below min_chars=30
    short = Fact(fact_id="s1", text="Short text", role="definition", supporting_spans=[])
    normal = _fact("n1", _LONG_B)
    reason, _ = prefilter_pair(short, normal, cosine=0.7, cfg=_CFG)
    assert reason == "short_fact"


def test_short_fact_b_rejected() -> None:
    normal = _fact("n1", _LONG_A)
    short = Fact(fact_id="s2", text="Short text", role="definition", supporting_spans=[])
    reason, _ = prefilter_pair(normal, short, cosine=0.7, cfg=_CFG)
    assert reason == "short_fact"


# ---------------------------------------------------------------------------
# Hard reject: weak_bridge (combo of weak tokens AND weak cosine)
# ---------------------------------------------------------------------------

def test_weak_bridge_rejected_when_both_signals_weak() -> None:
    # No shared tokens AND cosine below threshold
    fact_a = _fact("a1", "Quantum entanglement describes the correlation between particles at a distance.")
    fact_b = _fact("b1", "Medieval architecture relied on flying buttresses for structural support.")
    reason, _ = prefilter_pair(fact_a, fact_b, cosine=0.20, cfg=_CFG)
    assert reason == "weak_bridge"


def test_strong_cosine_saves_weak_token_pair() -> None:
    fact_a = _fact("a1", "Quantum entanglement describes the correlation between particles at a distance.")
    fact_b = _fact("b1", "Medieval architecture relied on flying buttresses for structural support.")
    # Strong cosine compensates for low token overlap.
    reason, _ = prefilter_pair(fact_a, fact_b, cosine=0.70, cfg=_CFG)
    assert reason != "weak_bridge"


def test_shared_tokens_save_low_cosine_pair() -> None:
    fact_a = _fact("a1", _LONG_A)
    fact_b = _fact("b1", _SHARED_B)  # shares "transmission control protocol"
    reason, _ = prefilter_pair(fact_a, fact_b, cosine=0.20, cfg=_CFG)
    # bridge_tokens ≥ 2 (transmission, control, protocol) → not weak_bridge
    assert reason != "weak_bridge"


# ---------------------------------------------------------------------------
# Hard reject: page_gap_outlier_combined (gap + weak tokens + weak cosine)
# ---------------------------------------------------------------------------

def test_page_gap_outlier_rejected_when_all_weak() -> None:
    far_a = _fact("a1", "Quantum entanglement describes particle correlations at a distance.", page=1)
    far_b = _fact("b1", "Medieval buttresses provided lateral structural support in Gothic buildings.", page=80)
    reason, _ = prefilter_pair(far_a, far_b, cosine=0.20, cfg=_CFG)
    assert reason == "page_gap_outlier_combined"


def test_page_gap_alone_not_rejected_with_strong_bridge() -> None:
    # Large page gap but strong lexical bridge → penalty only, not hard reject.
    fact_a = _fact("a1", "The transmission control protocol handles reliable packet delivery.", page=1)
    fact_b = _fact("b1", "Transmission control protocol retransmission uses exponential backoff.", page=80)
    reason, penalty = prefilter_pair(fact_a, fact_b, cosine=0.65, cfg=_CFG)
    assert reason is None
    assert penalty > 0.0  # page-gap outlier penalty applied


def test_page_gap_below_threshold_no_rejection() -> None:
    close_a = _fact("a1", _LONG_A, page=10)
    close_b = _fact("b1", _LONG_B, page=12)
    reason, _ = prefilter_pair(close_a, close_b, cosine=0.50, cfg=_CFG)
    assert reason != "page_gap_outlier_combined"


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------

def test_same_page_same_role_high_overlap_penalised() -> None:
    text_a = "The TCP protocol ensures reliable delivery of packets across networks."
    text_b = "TCP protocol ensures reliable delivery of data packets over networks."  # high overlap
    fact_a = _fact("a1", text_a, role="definition", page=5)
    fact_b = _fact("b1", text_b, role="definition", page=5)
    cfg = PairPrefilterConfig(near_duplicate_overlap_cap=1.01)
    reason, penalty = prefilter_pair(fact_a, fact_b, cosine=0.95, cfg=cfg)
    assert reason is None
    assert penalty == pytest.approx(
        cfg.penalty_same_role_high_overlap + cfg.penalty_same_page_high_overlap
    )


def test_cross_page_same_role_overlap_is_downranked_not_rejected() -> None:
    fact_a = _fact(
        "a1",
        "Bootstrapping methods usually perform much better than nonbootstrapping methods.",
        role="definition",
        page=5,
    )
    fact_b = _fact(
        "b1",
        "It remains unclear why bootstrapping methods perform better than nonbootstrapping methods.",
        role="definition",
        page=7,
    )
    cfg = PairPrefilterConfig(near_duplicate_overlap_cap=1.01)
    reason, penalty = prefilter_pair(fact_a, fact_b, cosine=0.94, cfg=cfg)
    assert reason is None
    assert penalty >= cfg.penalty_same_role_high_overlap


def test_near_duplicate_fact_pair_rejected_before_classification() -> None:
    fact_a = _fact(
        "a1",
        "Updates for both accumulating and replacing traces are specified.",
        role="definition",
        page=5,
    )
    fact_b = _fact(
        "b1",
        "Updates for both accumulating and replacing traces are specified.",
        role="definition",
        page=6,
    )
    reason, penalty = prefilter_pair(fact_a, fact_b, cosine=0.99, cfg=_CFG)
    assert reason == "near_duplicate_fact_text"
    assert penalty == 0.0


def test_paraphrase_with_high_token_containment_is_rejected() -> None:
    fact_a = _fact(
        "a1",
        "Memory may be required to build accurate approximations of value functions and policies.",
    )
    fact_b = _fact(
        "b1",
        "A large amount of memory is required to build approximations of value functions and policies.",
    )
    reason, _ = prefilter_pair(fact_a, fact_b, cosine=0.98, cfg=_CFG)
    assert reason == "near_duplicate_fact_text"


def test_explicit_restatement_pair_is_rejected() -> None:
    fact_a = _fact(
        "a1",
        "Another way of saying this is that a greedy policy for the optimal evaluation is optimal.",
    )
    fact_b = _fact(
        "b1",
        "A policy greedy with respect to the optimal value is an optimal policy.",
    )
    reason, _ = prefilter_pair(fact_a, fact_b, cosine=0.98, cfg=_CFG)
    assert reason in {
        "fragment_pair",
        "near_duplicate_fact_text",
        "explicit_restatement_pair",
    }


def test_identical_prefix_penalised() -> None:
    text_a = "The network layer is responsible for packet forwarding and routing."
    text_b = "The network layer also handles congestion control and traffic management."
    fact_a = _fact("a1", text_a)
    fact_b = _fact("b1", text_b)
    reason, penalty = prefilter_pair(fact_a, fact_b, cosine=0.60, cfg=_CFG)
    assert reason is None
    assert penalty >= _CFG.penalty_identical_prefix


def test_good_pair_no_reject_no_penalty() -> None:
    fact_a = _fact("a1", "TCP uses sequence numbers to ensure in-order delivery of segments.", role="definition", page=3)
    fact_b = _fact("b1", "Flow control allows the receiver to limit the rate of incoming data.", role="condition", page=4)
    reason, penalty = prefilter_pair(fact_a, fact_b, cosine=0.60, cfg=_CFG)
    assert reason is None
    assert penalty == pytest.approx(0.0)


def test_definition_definition_without_shared_anchor_penalised() -> None:
    fact_a = _fact(
        "a1",
        "A policy specifies how an agent selects actions in each available state.",
        role="definition",
    )
    fact_b = _fact(
        "b1",
        "A value function estimates expected future return from a starting situation.",
        role="definition",
    )
    reason, penalty = prefilter_pair(fact_a, fact_b, cosine=0.70, cfg=_CFG)
    assert reason is None
    assert penalty >= _CFG.penalty_definition_definition_weak_anchor
    assert penalty >= (
        _CFG.penalty_definition_definition_weak_anchor
        + _CFG.penalty_weak_role_pair
    )


def test_definition_definition_with_shared_symbol_not_penalised_for_weak_anchor() -> None:
    fact_a = _fact(
        "a1",
        "The value function vπ(s) gives the expected return from state s.",
        role="definition",
    )
    fact_b = _fact(
        "b1",
        "The Bellman equation recursively defines vπ(s) using successor states.",
        role="definition",
    )
    reason, penalty = prefilter_pair(fact_a, fact_b, cosine=0.70, cfg=_CFG)
    assert reason is None
    assert penalty < _CFG.penalty_definition_definition_weak_anchor


def test_deictic_unresolved_pair_penalised() -> None:
    fact_a = _fact(
        "a1",
        "This method updates estimates after each observed transition in the task.",
        role="condition",
    )
    fact_b = _fact(
        "b1",
        "Eligibility traces assign temporary credit to recently visited states.",
        role="definition",
    )
    reason, penalty = prefilter_pair(fact_a, fact_b, cosine=0.70, cfg=_CFG)
    assert reason is None
    assert penalty >= _CFG.penalty_deictic_unresolved


def test_definition_example_without_shared_anchor_penalised() -> None:
    fact_a = _fact(
        "a1",
        "A policy specifies how an agent selects actions in each available state.",
        role="definition",
    )
    fact_b = _fact(
        "b1",
        "For example, a robot may move through a hallway while searching for cans.",
        role="example",
    )
    reason, penalty = prefilter_pair(fact_a, fact_b, cosine=0.70, cfg=_CFG)
    assert reason is None
    assert penalty >= _CFG.penalty_weak_role_pair


def test_math_fragment_pair_is_downranked_not_rejected() -> None:
    fact_a = _fact(
        "a1",
        "The value function vπ(s) gives the expected return from state s under policy π.",
        role="definition",
    )
    fact_b = _fact(
        "b1",
        "vπ(s) = Eπ[Gt | St = s] defines the state-value expectation for policy π.",
        role="definition",
    )
    reason, penalty = prefilter_pair(fact_a, fact_b, cosine=0.75, cfg=_CFG)
    assert reason is None
    assert penalty >= _CFG.penalty_math_fragment_pair


# ---------------------------------------------------------------------------
# prefilter_pairs (batch)
# ---------------------------------------------------------------------------

def test_batch_filters_hard_rejects() -> None:
    good_a = _fact("g1", "TCP sequence numbers ensure reliable in-order delivery of data.", page=1)
    good_b = _fact("g2", "Flow control prevents receiver buffer overflow in TCP connections.", page=2)
    short = Fact(fact_id="s1", text="Short text", role="definition", supporting_spans=[])
    pairs = [(good_a, good_b, 0.65), (short, good_b, 0.65)]
    kept, stats = prefilter_pairs(pairs, _CFG)
    assert len(kept) == 1
    assert kept[0][0].fact_id == "g1"
    assert stats["hard_rejected"]["short_fact"] == 1
    assert stats["input"] == 2
    assert stats["passed"] == 1


def test_batch_disabled_passes_everything() -> None:
    short = Fact(fact_id="s1", text="Short text", role="definition", supporting_spans=[])
    fact_b = _fact("b1", _LONG_B)
    cfg = PairPrefilterConfig(enabled=False)
    kept, stats = prefilter_pairs([(short, fact_b, 0.5)], cfg)
    assert len(kept) == 1
    assert stats["passed"] == 1


def test_batch_penalty_reduces_score() -> None:
    text_a = "The TCP protocol ensures reliable delivery of packets across networks."
    text_b = "TCP protocol ensures reliable delivery of data packets over networks."
    fact_a = _fact("a1", text_a, role="definition", page=5)
    fact_b = _fact("b1", text_b, role="definition", page=5)
    pairs = [(fact_a, fact_b, 0.95)]
    kept, stats = prefilter_pairs(
        pairs,
        PairPrefilterConfig(near_duplicate_overlap_cap=1.01),
    )
    assert len(kept) == 1
    # Score should be reduced by at least the same-page penalty.
    assert kept[0][2] < 0.95
    assert stats["penalized"] == 1


def test_batch_stats_structure() -> None:
    fact_a = _fact("a1", _LONG_A, page=1)
    fact_b = _fact("b1", _LONG_B, page=2)
    _, stats = prefilter_pairs([(fact_a, fact_b, 0.6)], _CFG)
    for key in ("input", "hard_rejected", "penalized", "passed"):
        assert key in stats
