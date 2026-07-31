"""Tests for V16.2 topology-aware intent gate."""

from __future__ import annotations

from rag_gt.core.types import Fact, FactChain, Span
from rag_gt.generation.topology_intent import (
    TopologyIntentConfig,
    allowed_intents_for_chain,
    pick_topology_intent,
)


def _fact(fid: str, text: str | None = None) -> Fact:
    text = text or f"Fact {fid} states a grounded networking property for testing."
    return Fact(
        fact_id=fid,
        text=text,
        role="definition",
        supporting_spans=[
            Span(
                doc_id="doc",
                chunk_id="c1",
                start_token=0,
                end_token=8,
                char_start=0,
                char_end=len(text),
                page_start=1,
                page_end=1,
            )
        ],
    )


def _chain(edge_types: list[str], *, nli: float = 0.80) -> FactChain:
    facts = [_fact(f"f{i}") for i in range(len(edge_types) + 1)]
    edges = [
        {
            "edge_id": f"e{i}",
            "src": facts[i].fact_id,
            "dst": facts[i + 1].fact_id,
            "type": edge_type,
            "nli_score": nli,
        }
        for i, edge_type in enumerate(edge_types)
    ]
    return FactChain(
        fact_ids=[f.fact_id for f in facts],
        anchor_id=facts[0].fact_id,
        role_path=[f.role for f in facts],
        chain_edges=edges,
    )


def test_every_canonical_label_maps_to_at_least_one_intent() -> None:
    for label in (
        "definition",
        "rule",
        "comparative",
        "causal",
        "quantitative",
        "procedural",
        "temporal",
        "intersection",
        "descriptive",
    ):
        intents = allowed_intents_for_chain(_chain([label]), typed_confidence=0.55)
        assert intents, label


def test_single_hop_fallback_is_factoid_only() -> None:
    chain = FactChain(fact_ids=["f1"], role_path=["definition"])
    assert allowed_intents_for_chain(chain, category="fb") == ("factoid",)


def test_untyped_mh_high_pass1_routes_to_factoid() -> None:
    assert allowed_intents_for_chain(
        _chain(["unknown"]),
        category="untyped_mh",
        pass1_score=0.70,
    ) == ("factoid",)


def test_untyped_mh_low_pass1_drops() -> None:
    assert allowed_intents_for_chain(
        _chain(["unknown"]),
        category="untyped_mh",
        pass1_score=0.50,
    ) == ()


def test_multi_edge_typed_chain_removes_factoid_when_possible() -> None:
    intents = allowed_intents_for_chain(
        _chain(["definition", "temporal"]),
        typed_confidence=0.55,
    )
    assert "inferential" in intents
    assert "factoid" not in intents


def test_counterfactual_only_above_confidence_threshold() -> None:
    high = allowed_intents_for_chain(_chain(["causal"], nli=0.90), typed_confidence=0.90)
    low = allowed_intents_for_chain(_chain(["causal"], nli=0.50), typed_confidence=0.50)
    assert "counterfactual" in high
    assert "counterfactual" not in low


def test_unanswerable_only_below_confidence_threshold() -> None:
    low = allowed_intents_for_chain(_chain(["causal"], nli=0.30), typed_confidence=0.30)
    high = allowed_intents_for_chain(_chain(["causal"], nli=0.80), typed_confidence=0.80)
    assert "unanswerable" in low
    assert "unanswerable" not in high


def test_pick_topology_intent_returns_drop_reason_when_no_intent() -> None:
    decision = pick_topology_intent(
        _chain(["unknown"]),
        current_counts={},
        category="untyped_mh",
        pass1_score=0.20,
    )
    assert decision.intent is None
    assert decision.reason == "no_compatible_intent"


def test_pick_topology_intent_renormalizes_to_allowed_subset() -> None:
    decision = pick_topology_intent(
        _chain(["comparative"]),
        current_counts={"comparative": 0, "factoid": 100},
        typed_confidence=0.55,
    )
    assert decision.intent == "comparative"
    assert decision.allowed_intents == ("comparative",)


def test_config_overrides_label_mapping() -> None:
    cfg = TopologyIntentConfig.from_dict({"map": {"definition": ["procedural"]}})
    assert allowed_intents_for_chain(
        _chain(["definition"]),
        cfg,
        typed_confidence=0.55,
    ) == ("procedural",)
