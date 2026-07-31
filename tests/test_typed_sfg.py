"""Phase B tests — TF-SFG typed edge graph (P1).

All LLM and NLI calls are stubbed so tests are deterministic and offline.
"""

from __future__ import annotations

import random
import time

import numpy as np
import pytest

from rag_gt.budget.adaptive_budget import DocBudget
from rag_gt.core.llm import APIError
from rag_gt.core.types import Fact, FactChain, Span
from rag_gt.graph.edge_classifier import EdgeRecord, classify_edge
from rag_gt.graph.pair_scheduler import schedule_candidate_pairs
from rag_gt.graph.typed_sfg import TypedSFG, _role_bonus
from rag_gt.observability.cost_tracker import CostTracker, TrackedLLM
from rag_gt.vectorstore.faiss_index import FactIndex


# ---------- helpers ----------


def _span(doc_id: str = "d1") -> Span:
    return Span(doc_id=doc_id, chunk_id=f"{doc_id}_c000000", start_token=0, end_token=5)


def _fact(fid: str, text: str, role: str = "definition") -> Fact:
    return Fact(
        fact_id=fid,
        text=text,
        role=role,  # type: ignore[arg-type]
        supporting_spans=[_span()],
    )


def _page_fact(fid: str, text: str, *, page: int, role: str = "definition") -> Fact:
    return Fact(
        fact_id=fid,
        text=text,
        role=role,  # type: ignore[arg-type]
        supporting_spans=[
            Span(
                doc_id="d1",
                chunk_id=f"d1_c{page:03d}",
                start_token=0,
                end_token=8,
                page_start=page,
                page_end=page,
            )
        ],
    )


def test_pair_scheduler_prior_prefers_strict_survivor_role_shape() -> None:
    definition = _fact("D", "The return identifies the objective optimized by an agent.", "definition")
    condition = _fact(
        "C",
        "If actions change only immediate rewards, a myopic policy is sufficient.",
        "condition",
    )
    generic_definition = _fact("G", "A second term is described elsewhere.", "definition")

    assert _role_bonus(definition, condition) > _role_bonus(definition, generic_definition)
    assert _role_bonus(definition, condition) <= 0.08


def test_pair_scheduler_canonicalizes_candidates_to_source_order() -> None:
    earlier = _page_fact(
        "C",
        "If the step size decreases appropriately, convergence is guaranteed.",
        page=1,
        role="condition",
    )
    later = _page_fact(
        "D",
        "With this choice, the gradient method converges to a local optimum.",
        page=2,
        role="definition",
    )
    index = _MapIndex(
        {
            "C": [("D", 0.90)],
            "D": [("C", 0.90)],
        }
    )

    selected, _report = schedule_candidate_pairs(
        [later, earlier],
        index,
        min_cosine=0.0,
        raw_candidate_pairs=10,
        max_pairs=10,
    )

    assert len(selected) == 1
    assert (selected[0][0].fact_id, selected[0][1].fact_id) == ("C", "D")


def test_pair_scheduler_promotes_directional_condition_outcome_pair() -> None:
    condition = _page_fact(
        "C",
        "When rewards are delayed by many steps, eligibility traces assign credit backward.",
        page=1,
        role="condition",
    )
    outcome = _page_fact(
        "O",
        "Eligibility traces therefore produce faster learning in delayed-reward tasks.",
        page=2,
        role="consequence",
    )
    topic_a = _page_fact(
        "A",
        "Rewards in biological systems are analogous to pleasure or pain.",
        page=3,
        role="definition",
    )
    topic_b = _page_fact(
        "B",
        "Human pleasure is sometimes used as an analogy for reward signals.",
        page=4,
        role="condition",
    )
    index = _MapIndex(
        {
            "C": [("O", 0.80)],
            "O": [("C", 0.80)],
            "A": [("B", 0.80)],
            "B": [("A", 0.80)],
        }
    )

    selected, _report = schedule_candidate_pairs(
        [condition, outcome, topic_a, topic_b],
        index,
        min_cosine=0.0,
        raw_candidate_pairs=10,
        max_pairs=2,
    )

    assert selected[0][0].fact_id == "C"
    assert selected[0][1].fact_id == "O"
    assert selected[0][2] > selected[1][2]


def _make_index(facts, dim=8):
    rng = np.random.default_rng(42)
    embs = rng.random((len(facts), dim)).astype(np.float32)
    idx = FactIndex(dim=dim)
    idx.add(embs, [f.fact_id for f in facts])
    return idx, embs


class _MapIndex:
    def __init__(self, neighbours):
        self.neighbours = neighbours
        self.current = ""

    def embedding_for(self, fid):
        self.current = fid
        return [1.0]

    def search(self, emb, k=30):
        return self.neighbours.get(getattr(self, "current", ""), [])


class _FakeLLM:
    """Returns a canned JSON edge verdict."""
    model = "fake-edge-llm"

    def __init__(self, relation_type="causal", quote="leads to convergence"):
        self._relation_type = relation_type
        self._quote = quote

    def generate(self, prompt, temperature=0.0, max_tokens=512):
        return (
            '{"relation_type": "' + self._relation_type + '", '
            '"bridging_fact_id": "F001", '
            '"bridging_quote": "' + self._quote + '", '
            '"relation_claim": "F001 causes F002 to exhibit the described property.", '
            '"rationale": "A leads to B."}'
        )


class _ContributionLLM:
    model = "fake-edge-contribution-llm"

    def generate(self, prompt, temperature=0.0, max_tokens=512):
        return (
            '{"relation_type": "causal", '
            '"bridging_fact_id": "F001", '
            '"bridging_quote": "TD errors update value estimates", '
            '"relation_claim": "TD errors combine with eligibility traces to update recent states.", '
            '"source_contribution": "TD errors update value estimates.", '
            '"destination_contribution": "Eligibility traces assign credit to recent states.", '
            '"question_seed": "How do TD errors and eligibility traces combine during value updates?", '
            '"rationale": "Each mechanism supplies a different part of the update."}'
        )


class _FakeLLMNone:
    model = "fake-edge-llm-none"

    def generate(self, prompt, temperature=0.0, max_tokens=512):
        return '{"relation_type": "none", "bridging_fact_id": "", "bridging_quote": "", "relation_claim": "", "rationale": ""}'


class _FakeLLMApiError:
    model = "fake-edge-api-error"

    def generate(self, prompt, temperature=0.0, max_tokens=512):
        raise APIError("network outage")


# ---------- EdgeRecord ----------


def test_edge_record_to_dict():
    rec = EdgeRecord(
        edge_id="e00001",
        src="F001",
        dst="F002",
        type="causal",
        bridging_fact_id="F001",
        bridging_quote="leads to convergence",
        relation_claim="F001 causes F002 to converge.",
        nli_score=0.75,
    )
    d = rec.to_dict()
    assert d["edge_id"] == "e00001"
    assert d["type"] == "causal"
    assert d["nli_score"] == 0.75


# ---------- TypedSFG.build (stubbed NLI) ----------


def _stub_nli(monkeypatch, score: float):
    """Patch nli_entailment in edge_classifier to return a fixed score."""
    monkeypatch.setattr(
        "rag_gt.graph.edge_classifier.nli_entailment",
        lambda premise, hypothesis: score,
    )
    # Also patch the SQLite cache to avoid disk I/O.
    monkeypatch.setattr(
        "rag_gt.graph.edge_classifier._get_cached",
        lambda conn, key: None,
    )
    monkeypatch.setattr(
        "rag_gt.graph.edge_classifier._set_cached",
        lambda conn, key, result: None,
    )


def test_typed_sfg_build_no_facts(monkeypatch):
    _stub_nli(monkeypatch, 0.9)
    idx, _ = _make_index([])
    sfg = TypedSFG([], idx)
    sfg.build(_FakeLLM())
    assert sfg.edge_count == 0


def test_typed_sfg_build_accepts_high_nli(monkeypatch):
    _stub_nli(monkeypatch, 0.80)
    facts = [
        _fact("F001", "TD learning leads to convergence in policy evaluation."),
        _fact("F002", "Policy evaluation produces the value function under current policy."),
    ]
    idx, embs = _make_index(facts)
    llm = _FakeLLM(quote="leads to convergence")
    sfg = TypedSFG(facts, idx, {"tf_sfg": {"min_cosine": 0.0, "nli_edge_threshold": 0.55}})
    sfg.build(llm)
    assert sfg.classified_pairs >= 1
    assert sfg.edge_count >= 1
    key = ("F001", "F002")
    assert key in sfg.edge_map
    assert sfg.edge_map[key].type == "causal"


def test_typed_sfg_build_stops_at_graph_deadline(monkeypatch):
    _stub_nli(monkeypatch, 0.80)
    facts = [
        _fact("F001", "TD learning leads to convergence in policy evaluation."),
        _fact("F002", "Policy evaluation produces the value function under current policy."),
    ]
    idx, _ = _make_index(facts)
    sfg = TypedSFG(facts, idx, {"tf_sfg": {"min_cosine": 0.0}})

    sfg.build(_FakeLLM(), deadline=time.time() - 1.0)

    assert sfg.build_timed_out is True
    assert sfg.edge_count == 0
    assert sfg.classified_pairs == 0


def test_typed_sfg_build_stops_before_live_call_budget_overrun(monkeypatch):
    _stub_nli(monkeypatch, 0.80)
    facts = [
        _fact("F001", "TD learning leads to convergence in policy evaluation."),
        _fact("F002", "Policy evaluation produces the value function under current policy."),
    ]
    idx, _ = _make_index(facts)
    sfg = TypedSFG(facts, idx, {"tf_sfg": {"min_cosine": 0.0}})
    tracked = TrackedLLM(
        _FakeLLM(),
        CostTracker(max_live_api_calls=0),
        stage="tf_sfg_classify",
        doc_id="doc1",
    )

    sfg.build(tracked)

    assert sfg.build_budget_exhausted is True
    assert sfg.edge_count == 0
    assert sfg.classified_pairs == 0


def test_typed_sfg_skips_mutually_entailing_pairs_before_paid_classification(monkeypatch):
    monkeypatch.setattr(
        "rag_gt.graph.typed_sfg.nli_batch",
        lambda pairs: [0.96 for _ in pairs],
    )
    facts = [
        _fact("F001", "An optimal policy is greedy with respect to the optimal value function."),
        _fact("F002", "A policy greedy with respect to the optimal value function is optimal."),
    ]
    idx, _ = _make_index(facts)
    sfg = TypedSFG(
        facts,
        idx,
        {
            "tf_sfg": {
                "min_cosine": 0.0,
                "pair_prefilter": {"enabled": False},
                "pre_llm_redundancy": {
                    "enabled": True,
                    "bidirectional_entailment_threshold": 0.86,
                },
            }
        },
    )

    class _NoPaidCallLLM:
        model = "must-not-be-called"

        def generate(self, *args, **kwargs):
            raise AssertionError("relation LLM should not be called for redundant pair")

    sfg.build(_NoPaidCallLLM())

    assert sfg.redundant_pairs_skipped >= 1
    assert sfg.classified_pairs == 0
    assert sfg.edge_count == 0


def test_typed_sfg_skips_one_way_entailed_pairs_before_paid_classification(monkeypatch):
    monkeypatch.setattr(
        "rag_gt.graph.typed_sfg.nli_batch",
        lambda pairs: [0.96, 0.02],
    )
    facts = [
        _fact("F001", "If lambda equals zero, the lambda-return is the one-step return."),
        _fact("F002", "The lambda-return at lambda zero is the one-step return."),
    ]
    idx, _ = _make_index(facts)
    sfg = TypedSFG(
        facts,
        idx,
        {
            "tf_sfg": {
                "min_cosine": 0.0,
                "pair_prefilter": {"enabled": False},
                "pre_llm_redundancy": {
                    "enabled": True,
                    "bidirectional_entailment_threshold": 0.86,
                    "single_fact_entailment_threshold": 0.94,
                },
            }
        },
    )

    class _NoPaidCallLLM:
        model = "must-not-be-called"

        def generate(self, *args, **kwargs):
            raise AssertionError("relation LLM should not be called for entailed pair")

    sfg.build(_NoPaidCallLLM())

    assert sfg.single_fact_entailed_pairs_skipped >= 1
    assert sfg.classified_pairs == 0
    assert sfg.edge_count == 0


def test_edge_classifier_recovers_verbatim_quote_from_paraphrase(monkeypatch):
    _stub_nli(monkeypatch, 0.80)
    fact_a = _fact("F001", "TD learning leads to convergence in policy evaluation.")
    fact_b = _fact("F002", "Policy evaluation produces the value function under current policy.")
    rec = classify_edge(
        fact_a,
        fact_b,
        _FakeLLM(quote="causes stable value estimates"),
        edge_counter=[0],
        nli_threshold=0.55,
    )
    assert rec is not None
    assert rec.bridging_quote == "TD learning leads to convergence in policy evaluation."


def test_edge_classifier_scores_distinct_grounded_answer_contributions(monkeypatch):
    monkeypatch.setattr("rag_gt.graph.edge_classifier._get_cached", lambda conn, key: None)
    monkeypatch.setattr("rag_gt.graph.edge_classifier._set_cached", lambda conn, key, result: None)

    def contribution_nli(premise, hypothesis):
        if "combine with eligibility traces" in hypothesis:
            return 0.90 if "\n" in premise else 0.10
        if hypothesis == "TD errors update value estimates.":
            return 0.92 if "TD errors update" in premise else 0.05
        if hypothesis == "Eligibility traces assign credit to recent states.":
            return 0.91 if "Eligibility traces assign" in premise else 0.04
        return 0.0

    monkeypatch.setattr("rag_gt.graph.edge_classifier.nli_entailment", contribution_nli)
    fact_a = _fact("F001", "TD errors update value estimates after observed transitions.")
    fact_b = _fact("F002", "Eligibility traces assign credit to recent states.")
    rec = classify_edge(
        fact_a,
        fact_b,
        _ContributionLLM(),
        edge_counter=[0],
        nli_threshold=0.55,
        edge_minimality_enabled=True,
        answer_contribution_enabled=True,
        contribution_own_threshold=0.45,
    )
    assert rec is not None
    assert rec.source_contribution_score == pytest.approx(0.92)
    assert rec.destination_contribution_score == pytest.approx(0.91)
    assert rec.contribution_distinctness == pytest.approx(0.87)
    assert rec.question_seed_score == 1.0
    assert rec.answer_contribution_score > 0.90


def test_edge_classifier_requires_contributions_when_contract_enabled(monkeypatch):
    _stub_nli(monkeypatch, 0.90)
    fact_a = _fact("F001", "TD learning leads to convergence in policy evaluation.")
    fact_b = _fact("F002", "Policy evaluation produces the value function under current policy.")
    rec = classify_edge(
        fact_a,
        fact_b,
        _FakeLLM(quote="leads to convergence"),
        edge_counter=[0],
        nli_threshold=0.55,
        answer_contribution_enabled=True,
    )
    assert rec is None


def test_edge_classifier_downranks_cross_recoverable_contributions_without_rejecting(monkeypatch):
    monkeypatch.setattr("rag_gt.graph.edge_classifier._get_cached", lambda conn, key: None)
    monkeypatch.setattr("rag_gt.graph.edge_classifier._set_cached", lambda conn, key, result: None)

    def overlapping_nli(premise, hypothesis):
        if "combine with eligibility traces" in hypothesis:
            return 0.90 if "\n" in premise else 0.10
        return 0.90 if "update" in hypothesis or "assign credit" in hypothesis else 0.0

    monkeypatch.setattr("rag_gt.graph.edge_classifier.nli_entailment", overlapping_nli)
    rec = classify_edge(
        _fact("F001", "TD errors update value estimates after observed transitions."),
        _fact("F002", "Eligibility traces assign credit to recent states."),
        _ContributionLLM(),
        edge_counter=[0],
        nli_threshold=0.55,
        edge_minimality_enabled=True,
        answer_contribution_enabled=True,
    )
    assert rec is not None
    assert rec.contribution_distinctness == 0.0
    assert rec.question_seed_score == 1.0
    assert rec.answer_contribution_score == pytest.approx(0.71)


def test_typed_sfg_build_rejects_low_nli(monkeypatch):
    _stub_nli(monkeypatch, 0.20)
    facts = [
        _fact("F001", "TD learning leads to convergence in policy evaluation."),
        _fact("F002", "Policy evaluation produces the value function under current policy."),
    ]
    idx, _ = _make_index(facts)
    sfg = TypedSFG(facts, idx, {"tf_sfg": {"min_cosine": 0.0, "nli_edge_threshold": 0.55}})
    sfg.build(_FakeLLM())
    assert sfg.edge_count == 0


def test_edge_classifier_can_soft_accept_source_backed_low_nli(monkeypatch):
    _stub_nli(monkeypatch, 0.01)
    fact_a = _fact("F001", "TD learning leads to convergence in policy evaluation.")
    fact_b = _fact("F002", "Policy evaluation produces the value function under current policy.")
    rec = classify_edge(
        fact_a,
        fact_b,
        _FakeLLM(quote="leads to convergence"),
        edge_counter=[0],
        nli_threshold=0.55,
        allow_low_nli_with_quote=True,
        low_nli_floor=0.0,
    )
    assert rec is not None
    assert rec.nli_score == 0.01


def test_typed_sfg_build_skips_none_relation(monkeypatch):
    _stub_nli(monkeypatch, 0.90)
    facts = [
        _fact("F001", "TD learning leads to convergence."),
        _fact("F002", "Policy evaluation uses a value function."),
    ]
    idx, _ = _make_index(facts)
    sfg = TypedSFG(facts, idx, {"tf_sfg": {"min_cosine": 0.0}})
    sfg.build(_FakeLLMNone())
    assert sfg.edge_count == 0


def test_typed_sfg_build_skips_none_relation_with_canonicalization(monkeypatch):
    _stub_nli(monkeypatch, 0.90)
    facts = [
        _fact("F001", "TD learning leads to convergence."),
        _fact("F002", "Policy evaluation uses a value function."),
    ]
    idx, _ = _make_index(facts)
    sfg = TypedSFG(
        facts,
        idx,
        {
            "tf_sfg": {"min_cosine": 0.0},
            "v16_2": {"enabled": True, "pair_prefilter": {"enabled": False}},
        },
    )
    sfg.build(_FakeLLMNone())
    assert sfg.edge_count == 0


def test_typed_sfg_api_errors_fail_fast_without_caching(monkeypatch):
    writes = []
    monkeypatch.setattr(
        "rag_gt.graph.edge_classifier._get_cached",
        lambda conn, key: None,
    )
    monkeypatch.setattr(
        "rag_gt.graph.edge_classifier._set_cached",
        lambda conn, key, result: writes.append(result),
    )
    facts = [
        _fact("F001", "TD learning leads to convergence in policy evaluation."),
        _fact("F002", "Policy evaluation produces the value function under current policy."),
    ]
    idx, _ = _make_index(facts)
    sfg = TypedSFG(
        facts,
        idx,
        {
            "tf_sfg": {
                "min_cosine": 0.0,
                "nli_edge_threshold": 0.55,
                "max_consecutive_api_errors": 1,
            }
        },
    )
    with pytest.raises(RuntimeError, match="consecutive API errors"):
        sfg.build(_FakeLLMApiError())
    assert writes == []


# ---------- walk_typed_paths ----------


def _build_chain_graph(monkeypatch):
    """Build a 3-fact linear graph F001→F002→F003 with NLI=0.8."""
    _stub_nli(monkeypatch, 0.80)
    facts = [
        _fact("F001", "The agent updates its estimate after each step."),
        _fact("F002", "The TD error is the difference between successive estimates."),
        _fact("F003", "Convergence is guaranteed under standard conditions."),
    ]
    idx, _ = _make_index(facts)

    class _LinearLLM:
        model = "fake-linear"

        def generate(self, prompt, temperature=0.0, max_tokens=512):
            if "F001" in prompt and "F002" in prompt:
                return '{"relation_type":"causal","bridging_fact_id":"F001","bridging_quote":"updates its estimate","relation_claim":"F001 causes F002 to arise.","rationale":"A leads to B."}'
            if "F002" in prompt and "F003" in prompt:
                return '{"relation_type":"causal","bridging_fact_id":"F002","bridging_quote":"TD error","relation_claim":"F002 causes F003 to hold.","rationale":"B leads to C."}'
            return '{"relation_type":"none","bridging_fact_id":"","bridging_quote":"","relation_claim":"","rationale":""}'

    sfg = TypedSFG(facts, idx, {"tf_sfg": {"min_cosine": 0.0, "nli_edge_threshold": 0.55}})
    sfg.build(_LinearLLM())
    return sfg, facts


def test_walk_depth1_returns_single_fact_chains(monkeypatch):
    sfg, facts = _build_chain_graph(monkeypatch)
    rng = random.Random(42)
    chains = sfg.walk_typed_paths({1: 5}, rng)
    assert len(chains) >= 1
    for c in chains:
        assert c.depth == 1
        assert len(c.chain_edges) == 0


def test_walk_depth2_carries_edge(monkeypatch):
    sfg, facts = _build_chain_graph(monkeypatch)
    rng = random.Random(42)
    chains = sfg.walk_typed_paths({2: 10}, rng)
    depth2 = [c for c in chains if c.depth == 2]
    assert len(depth2) >= 1
    for c in depth2:
        assert len(c.chain_edges) == 1
        assert c.chain_edges[0]["type"] == "causal"


def test_walk_depth3_carries_two_edges(monkeypatch):
    sfg, facts = _build_chain_graph(monkeypatch)
    rng = random.Random(0)
    chains = sfg.walk_typed_paths({3: 10}, rng)
    depth3 = [c for c in chains if c.depth == 3]
    if depth3:
        for c in depth3:
            assert len(c.chain_edges) == 2


def test_walk_not_built_raises():
    facts = [_fact("F001", "Some fact text that is long enough.")]
    idx, _ = _make_index(facts)
    sfg = TypedSFG(facts, idx)
    with pytest.raises(RuntimeError, match="build\\(\\)"):
        sfg.walk_typed_paths({2: 1}, random.Random())


def test_edges_for_chain(monkeypatch):
    sfg, facts = _build_chain_graph(monkeypatch)
    edges = sfg.edges_for_chain(["F001", "F002"])
    assert len(edges) == 1
    assert edges[0]["src"] == "F001"
    assert edges[0]["dst"] == "F002"


def test_edges_for_chain_missing_edge(monkeypatch):
    sfg, facts = _build_chain_graph(monkeypatch)
    # F003→F001 has no edge
    edges = sfg.edges_for_chain(["F003", "F001"])
    assert edges == []


def test_fact_chain_v16_fields_default():
    """FactChain should default chain_edges and question_assumptions to []."""
    chain = FactChain(fact_ids=["F001", "F002"])
    assert chain.chain_edges == []
    assert chain.question_assumptions == []


def _budget(tf_sfg_pairs: int = 2) -> DocBudget:
    return DocBudget(
        fact_count=4,
        page_count=4,
        doc_type="TEXTBOOK",
        size_signal=10.0,
        tf_sfg_pairs=tf_sfg_pairs,
        fallback_singlehop=8,
        strict_target=4,
        attempt_cap=16,
        mh_floor_fraction=0.60,
        untyped_mh_cap_fraction=0.10,
        fb_cap_fraction=0.40,
    )


def test_v16_2_doc_budget_replaces_max_candidate_pairs():
    facts = [
        _page_fact("F001", "Reliable delivery uses acknowledgements to recover lost packets.", page=1),
        _page_fact("F002", "Acknowledgements allow senders to detect missing packets.", page=2),
        _page_fact("F003", "Retransmission sends packets again after loss is detected.", page=3),
        _page_fact("F004", "Congestion control limits sending rate during overload.", page=4),
    ]
    idx = _MapIndex({
        "F001": [("F002", 0.9), ("F003", 0.8), ("F004", 0.7)],
        "F002": [("F003", 0.9), ("F004", 0.8)],
        "F003": [("F004", 0.9)],
    })
    sfg = TypedSFG(
        facts,
        idx,  # type: ignore[arg-type]
        {
            "tf_sfg": {
                "min_cosine": 0.0,
                "max_candidate_pairs": 10,
                "raw_candidate_pairs": 20,
                "per_page_pair_cap": 10,
            },
            "v16_2": {
                "enabled": True,
                "pair_prefilter": {"enabled": False},
            },
        },
        doc_budget=_budget(tf_sfg_pairs=2),
    )
    assert len(sfg._candidate_pairs([f.fact_id for f in facts])) == 2


def test_v16_2_pair_prefilter_runs_before_selection():
    strong = _page_fact(
        "F001",
        "TCP reliable delivery uses acknowledgements and retransmission after packet loss.",
        page=1,
    )
    weak_far = _page_fact(
        "F002",
        "Medieval buttresses provided lateral structural support in Gothic buildings.",
        page=80,
    )
    bridged = _page_fact(
        "F003",
        "Retransmission after packet loss helps reliable delivery preserve byte streams.",
        page=2,
    )
    facts = [strong, weak_far, bridged]
    idx = _MapIndex({
        "F001": [("F002", 0.20), ("F003", 0.90)],
    })
    sfg = TypedSFG(
        facts,
        idx,  # type: ignore[arg-type]
        {
            "tf_sfg": {
                "min_cosine": 0.0,
                "max_candidate_pairs": 10,
                "raw_candidate_pairs": 10,
                "per_page_pair_cap": 10,
            },
            "v16_2": {
                "enabled": True,
                "pair_prefilter": {
                    "enabled": True,
                    "min_chars": 30,
                    "min_bridge_tokens": 2,
                    "bridge_cosine_min": 0.55,
                    "max_page_gap_unaided": 30,
                },
            },
        },
        doc_budget=_budget(tf_sfg_pairs=10),
    )
    pairs = sfg._candidate_pairs([f.fact_id for f in facts])
    assert [(a.fact_id, b.fact_id) for a, b in pairs] == [("F001", "F003")]
    assert sfg.l0_stats["hard_rejected"]["page_gap_outlier_combined"] == 1


def test_plain_v16_pair_prefilter_runs_before_selection():
    strong = _page_fact(
        "F001",
        "TCP reliable delivery uses acknowledgements and retransmission after packet loss.",
        page=1,
    )
    weak_far = _page_fact(
        "F002",
        "Medieval buttresses provided lateral structural support in Gothic buildings.",
        page=80,
    )
    bridged = _page_fact(
        "F003",
        "Retransmission after packet loss helps reliable delivery preserve byte streams.",
        page=2,
    )
    facts = [strong, weak_far, bridged]
    idx = _MapIndex({
        "F001": [("F002", 0.20), ("F003", 0.90)],
    })
    sfg = TypedSFG(
        facts,
        idx,  # type: ignore[arg-type]
        {
            "tf_sfg": {
                "min_cosine": 0.0,
                "max_candidate_pairs": 10,
                "raw_candidate_pairs": 10,
                "per_page_pair_cap": 10,
                "pair_prefilter": {
                    "enabled": True,
                    "min_chars": 30,
                    "min_bridge_tokens": 2,
                    "bridge_cosine_min": 0.55,
                    "max_page_gap_unaided": 30,
                },
            },
            "v16_2": {"enabled": False},
        },
    )
    pairs = sfg._candidate_pairs([f.fact_id for f in facts])
    assert [(a.fact_id, b.fact_id) for a, b in pairs] == [("F001", "F003")]
    assert sfg.l0_stats["hard_rejected"]["page_gap_outlier_combined"] == 1
