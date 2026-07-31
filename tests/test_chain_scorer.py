"""Tests for V16.2 L1 chain scorer."""

from __future__ import annotations

import pytest

from rag_gt.core.types import Fact, FactChain, Span
from rag_gt.graph.chain_scorer import (
    ChainScorerConfig,
    assign_category,
    compositional_actionability,
    edge_answer_contribution_score,
    edge_joint_only_margin,
    multi_hop_questionability,
    pass2_bound,
    rank_and_select,
    relation_type_strength,
    score_chains_pass1,
    source_text_quality,
    typed_edge_confidence,
)


def _fact(fid: str, text: str, *, role: str = "definition", page: int = 1) -> Fact:
    return Fact(
        fact_id=fid,
        text=text,
        role=role,  # type: ignore[arg-type]
        supporting_spans=[
            Span(
                doc_id="doc",
                chunk_id=f"c{page}",
                start_token=0,
                end_token=8,
                char_start=0,
                char_end=len(text),
                page_start=page,
                page_end=page,
            )
        ],
    )


def _chain(
    *facts: Fact,
    edge_type: str = "definition",
    nli: float = 0.80,
    margin: float = 0.50,
    contribution: float = 0.0,
) -> FactChain:
    fact_ids = [f.fact_id for f in facts]
    edges = [
        {
            "edge_id": f"e{i}",
            "src": a.fact_id,
            "dst": b.fact_id,
            "type": edge_type,
            "nli_score": nli,
            "joint_only_margin": margin,
            "answer_contribution_score": contribution,
        }
        for i, (a, b) in enumerate(zip(facts, facts[1:]), start=1)
    ]
    return FactChain(
        fact_ids=fact_ids,
        anchor_id=fact_ids[0],
        mean_cosine=0.7,
        role_path=[f.role for f in facts],
        chain_edges=edges,
    )


def _facts_by_id(facts: list[Fact]) -> dict[str, Fact]:
    return {f.fact_id: f for f in facts}


def test_assign_category_single_hop_is_fb() -> None:
    f1 = _fact("f1", "TCP provides reliable byte stream service to applications.")
    assert assign_category(_chain(f1)) == "fb"


def test_assign_category_typed_multihop_requires_all_canonical_edges() -> None:
    f1 = _fact("f1", "A protocol defines the format and order of exchanged messages.")
    f2 = _fact("f2", "TCP uses acknowledgements to provide reliable delivery.")
    f3 = _fact("f3", "Reliable delivery reduces data loss for applications.")
    assert assign_category(_chain(f1, f2, f3, edge_type="causal")) == "typed_mh"


def test_assign_category_unknown_or_missing_edge_is_untyped() -> None:
    f1 = _fact("f1", "A protocol defines the format and order of exchanged messages.")
    f2 = _fact("f2", "TCP uses acknowledgements to provide reliable delivery.")
    f3 = _fact("f3", "Reliable delivery reduces data loss for applications.")
    unknown = _chain(f1, f2, edge_type="none")
    missing = FactChain(
        fact_ids=[f1.fact_id, f2.fact_id, f3.fact_id],
        role_path=[f1.role, f2.role, f3.role],
        chain_edges=[unknown.chain_edges[0]],
    )
    assert assign_category(unknown) == "untyped_mh"
    assert assign_category(missing) == "untyped_mh"


def test_typed_edge_confidence_mean_counts_unknown_as_zero() -> None:
    f1 = _fact("f1", "A protocol defines the format and order of exchanged messages.")
    f2 = _fact("f2", "TCP uses acknowledgements to provide reliable delivery.")
    f3 = _fact("f3", "Reliable delivery reduces data loss for applications.")
    chain = _chain(f1, f2, f3, edge_type="definition", nli=0.80)
    chain.chain_edges[1]["type"] = "unknown"
    assert typed_edge_confidence(chain) == pytest.approx(0.40)


def test_pass1_scores_all_chains_without_nli(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_nli(_pairs):
        raise AssertionError("pass1 must not call NLI")

    monkeypatch.setattr("rag_gt.graph.chain_scorer.nli_check.nli_batch", fail_nli)
    f1 = _fact("f1", "A protocol defines the format and order of exchanged messages.")
    f2 = _fact("f2", "TCP uses acknowledgements to provide reliable delivery.")
    f3 = _fact("f3", "Reliable delivery reduces data loss for applications.")
    scored = score_chains_pass1(
        [_chain(f1, f2), _chain(f2, f3), _chain(f1)],
        _facts_by_id([f1, f2, f3]),
    )
    assert len(scored) == 3
    assert all(0.0 <= item.pass1_score <= 1.0 for item in scored)
    assert all("typed_edge_confidence" in item.signals for item in scored)
    assert all("edge_joint_only_margin" in item.signals for item in scored)
    assert all("answer_contribution_score" in item.signals for item in scored)
    assert all("relation_type_strength" in item.signals for item in scored)
    assert all("compositional_actionability" in item.signals for item in scored)


def test_relation_quality_signals_prefer_strong_joint_edges() -> None:
    f1 = _fact("f1", "Eligibility traces assign credit to recent states.")
    f2 = _fact("f2", "TD errors update value estimates after transitions.")
    f3 = _fact("f3", "A glossary defines a term used in the chapter.")
    strong = _chain(f1, f2, edge_type="mechanism", margin=0.92)
    weak = _chain(f1, f3, edge_type="definition", margin=0.08)

    assert edge_joint_only_margin(strong) > edge_joint_only_margin(weak)
    assert relation_type_strength(strong) > relation_type_strength(weak)

    cfg = ChainScorerConfig(
        nli_rerank_k=0,
        min_pass1_score=0.0,
        min_final_score_to_keep=0.0,
    )
    ranked, _stats = rank_and_select(
        [weak, strong],
        _facts_by_id([f1, f2, f3]),
        cfg,
        attempt_cap=10,
    )
    assert ranked[0].chain is strong


def test_answer_contribution_signal_prefers_distinct_grounded_pair() -> None:
    f1 = _fact("f1", "TD errors update value estimates after observed transitions.")
    f2 = _fact("f2", "Eligibility traces assign credit to recent states.")
    f3 = _fact("f3", "Eligibility traces are also described elsewhere.")
    distinct = _chain(f1, f2, edge_type="causal", contribution=0.88)
    recoverable = _chain(f1, f3, edge_type="causal", contribution=0.20)
    facts = _facts_by_id([f1, f2, f3])

    assert edge_answer_contribution_score(distinct) > edge_answer_contribution_score(recoverable)

    cfg = ChainScorerConfig(
        nli_rerank_k=0,
        min_pass1_score=0.0,
        min_final_score_to_keep=0.0,
        pass1_weights={"answer_contribution_score": 1.0},
    )
    ranked = score_chains_pass1([recoverable, distinct], facts, cfg)
    assert ranked[0].chain is distinct


def test_source_text_quality_downranks_math_fragments() -> None:
    clean_a = _fact("c1", "Eligibility traces assign credit to recently visited states.")
    clean_b = _fact("c2", "TD errors update value estimates after observed transitions.")
    math_a = _fact("m1", "vπ(s) = Eπ[Gt | St = s] defines a state-value expectation.")
    math_b = _fact("m2", "qπ(s, a) = Eπ[Gt | St = s, At = a] defines action value.")
    clean = _chain(clean_a, clean_b, edge_type="causal", nli=0.80, margin=0.50)
    mathy = _chain(math_a, math_b, edge_type="causal", nli=0.80, margin=0.50)
    facts = _facts_by_id([clean_a, clean_b, math_a, math_b])

    assert source_text_quality(clean, facts) > source_text_quality(mathy, facts)

    cfg = ChainScorerConfig(
        nli_rerank_k=0,
        min_pass1_score=0.0,
        min_final_score_to_keep=0.0,
    )
    ranked = score_chains_pass1([mathy, clean], facts, cfg)
    assert ranked[0].chain is clean
    assert "source_text_quality" in ranked[0].signals


def test_compositional_actionability_prefers_actionable_causal_chain() -> None:
    action_a = _fact(
        "a1",
        "The policy update changes the action selected in each state.",
        role="definition",
    )
    action_b = _fact(
        "a2",
        "Following a deterministic policy observes returns for only one action per state.",
        role="condition",
    )
    weak_a = _fact(
        "w1",
        "The Bellman equation for vπ defines a value relation for a policy.",
        role="definition",
    )
    weak_b = _fact(
        "w2",
        "The Bellman optimality equation is a system of equations over states.",
        role="definition",
    )
    actionable = _chain(action_a, action_b, edge_type="causal", nli=0.85, margin=0.90)
    weak = _chain(weak_a, weak_b, edge_type="definition", nli=0.95, margin=0.90)
    facts = _facts_by_id([action_a, action_b, weak_a, weak_b])

    assert compositional_actionability(actionable, facts) > compositional_actionability(weak, facts)

    cfg = ChainScorerConfig(
        nli_rerank_k=0,
        min_pass1_score=0.0,
        min_final_score_to_keep=0.0,
    )
    ranked = score_chains_pass1([weak, actionable], facts, cfg)
    assert ranked[0].chain is actionable
    assert "compositional_actionability" in ranked[0].signals


def test_multi_hop_questionability_prefers_setup_condition_pattern() -> None:
    setup = _fact(
        "s1",
        "The return is the function of future rewards that the agent seeks to maximize.",
        role="definition",
    )
    condition = _fact(
        "s2",
        "If each action influences only immediate reward, a myopic agent can maximize each reward separately.",
        role="condition",
    )
    generic_a = _fact(
        "g1",
        "The Bellman equation is a relation used in reinforcement learning.",
        role="definition",
    )
    generic_b = _fact(
        "g2",
        "The Bellman optimality equation is a system of equations.",
        role="definition",
    )
    good = _chain(setup, condition, edge_type="contrast", nli=0.82, margin=0.65)
    generic = _chain(generic_a, generic_b, edge_type="definition", nli=0.92, margin=0.65)
    facts = _facts_by_id([setup, condition, generic_a, generic_b])

    assert multi_hop_questionability(good, facts) > multi_hop_questionability(generic, facts)

    cfg = ChainScorerConfig(
        nli_rerank_k=0,
        min_pass1_score=0.0,
        min_final_score_to_keep=0.0,
        pass1_weights={"multi_hop_questionability": 1.0},
    )
    ranked = score_chains_pass1([generic, good], facts, cfg)
    assert ranked[0].chain is good
    assert "multi_hop_questionability" in ranked[0].signals


def test_pass2_bound_uses_attempt_cap() -> None:
    cfg = ChainScorerConfig(nli_rerank_k=80)
    assert pass2_bound(cfg, attempt_cap=2) == 3
    assert pass2_bound(cfg, attempt_cap=100) == 80


def test_rank_and_select_reranks_only_bounded_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_nli(pairs):
        calls.append(len(pairs))
        return [0.9 for _ in pairs]

    monkeypatch.setattr("rag_gt.graph.chain_scorer.nli_check.nli_batch", fake_nli)
    facts = [
        _fact(f"f{i}", f"Networking fact {i} describes reliable protocol behavior.", page=i)
        for i in range(1, 7)
    ]
    chains = [_chain(a, b) for a, b in zip(facts, facts[1:])]
    cfg = ChainScorerConfig(
        min_pass1_score=0.0,
        min_final_score_to_keep=0.0,
        nli_rerank_k=80,
    )
    ranked, stats = rank_and_select(
        chains,
        _facts_by_id(facts),
        cfg,
        attempt_cap=2,
    )
    assert len(ranked) == len(chains)
    assert stats["pass2_input"] == 3
    assert len(calls) == 3
    assert all(item.pass2_reranked for item in ranked[:3])


def test_rank_and_select_keeps_fallback_chains_for_yield_controller(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_nli(_pairs):
        raise AssertionError("single-hop fallback should not need pass2 NLI")

    monkeypatch.setattr("rag_gt.graph.chain_scorer.nli_check.nli_batch", fail_nli)
    fact = _fact("f1", "The agent learns from interaction with its environment.", page=1)
    cfg = ChainScorerConfig(
        min_pass1_score=0.99,
        min_final_score_to_keep=0.99,
        nli_rerank_k=80,
    )
    ranked, stats = rank_and_select(
        [FactChain(fact_ids=[fact.fact_id], role_path=[fact.role])],
        _facts_by_id([fact]),
        cfg,
        attempt_cap=4,
    )
    assert len(ranked) == 1
    assert ranked[0].category == "fb"
