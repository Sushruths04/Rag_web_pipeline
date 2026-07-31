"""Tests for AT/CT twin derivation (P4, v16).

All LLM, NLI, and infill calls are stubbed — tests run offline.
"""

from __future__ import annotations

import pytest

from rag_gt.core.types import Fact, MSFS, QuestionGT, Span
from rag_gt.generation.answers import ABSTENTION_TEXT
from rag_gt.generation.counterfactual_infill import CounterfactualResult
from rag_gt.generation.twins import (
    _load_bearing_fact_id,
    _make_abstention_twin,
    _make_counterfactual_twin,
    derive_twins,
)
from rag_gt.observability.cost_tracker import LiveCallBudgetExceeded


# ---------- helpers ----------


def _span() -> Span:
    return Span(doc_id="d1", chunk_id="d1_c000000", start_token=0, end_token=5)


def _fact(fid: str, text: str) -> Fact:
    return Fact(
        fact_id=fid,
        text=text,
        role="definition",
        supporting_spans=[_span()],
    )


def _strict_row(q_id: str = "d1_q001", n_facts: int = 2) -> QuestionGT:
    facts = [_fact(f"F{i:03d}", f"Fact {i} text about the topic.") for i in range(n_facts)]
    fact_ids = [f.fact_id for f in facts]
    return QuestionGT(
        q_id=q_id,
        question="How does topic A relate to topic B?",
        gold_answer="Topic A relates via mechanism B.",
        msfs_list=[MSFS(msfs_id=f"{q_id}_msfs1", fact_ids=fact_ids)],
        doc_ids=["d1"],
        required_fact_ids=fact_ids,
        difficulty_reasoning_depth=n_facts,
        difficulty_semantic_distance="intra_doc",
        required_facts=facts,
        required_fact_groups=[[fid] for fid in fact_ids],
        hop_type="multi_fact",
        minimum_required_fact_count=n_facts,
    )


def _ct_infill(fact, llm, *, contradiction_threshold, max_retries):
    return CounterfactualResult(
        original_fact_text=fact.text,
        perturbed_fact_text=fact.text + " (perturbed)",
        original_span="text",
        perturbed_span="text (perturbed)",
        span_start=5,
        span_end=9,
        contradiction_score=0.75,
        is_valid=True,
    )


class _FakeLLM:
    model = "fake-twins"

    def generate(self, prompt, temperature=0.0, max_tokens=256):
        return '{"replacement": "999", "rationale": "different number"}'


# ---------- _load_bearing_fact_id ----------


def test_load_bearing_falls_back_to_last_fact():
    q = _strict_row(n_facts=3)
    result = _load_bearing_fact_id(q)
    assert result == q.required_fact_ids[-1]


def test_load_bearing_uses_weakest_rel_assumption():
    q = _strict_row(n_facts=3)
    q.question_assumptions = [
        {"type": "REL", "entails_from_sfu": 0, "nli_score": 0.90},
        {"type": "REL", "entails_from_sfu": 2, "nli_score": 0.20},  # weakest
        {"type": "REL", "entails_from_sfu": 1, "nli_score": 0.60},
    ]
    result = _load_bearing_fact_id(q)
    assert result == q.required_fact_ids[2]


def test_load_bearing_skips_non_rel_assumptions():
    q = _strict_row(n_facts=2)
    # Only EXIST assumptions; no REL — should fall back to last fact.
    q.question_assumptions = [
        {"type": "EXIST", "entails_from_sfu": 0, "nli_score": 0.10},
    ]
    result = _load_bearing_fact_id(q)
    assert result == q.required_fact_ids[-1]


def test_load_bearing_empty_facts_returns_empty():
    q = _strict_row(n_facts=2)
    q.required_fact_ids = []  # simulate no required facts after construction
    result = _load_bearing_fact_id(q)
    assert result == ""


# ---------- AT twin ----------


def test_at_twin_created_correctly():
    q = _strict_row(n_facts=2)
    twin = _make_abstention_twin(q)
    assert twin is not None
    assert twin.twin_of == q.q_id
    assert twin.twin_type == "abstention"
    assert twin.gold_answer == ABSTENTION_TEXT
    assert twin.q_id == f"{q.q_id}_at"
    assert twin.difficulty_adversarial == "abstention"
    # One fact removed from the chain.
    assert len(twin.required_fact_ids) == len(q.required_fact_ids) - 1


def test_at_twin_skipped_for_single_fact():
    q = _strict_row(n_facts=1)
    twin = _make_abstention_twin(q)
    assert twin is None


def test_at_twin_question_unchanged():
    q = _strict_row(n_facts=2)
    twin = _make_abstention_twin(q)
    assert twin is not None
    assert twin.question == q.question


def test_at_twin_metadata_has_removed_fact():
    q = _strict_row(n_facts=3)
    twin = _make_abstention_twin(q)
    assert twin is not None
    assert "removed_fact_id" in twin.twin_metadata
    assert twin.twin_metadata["removed_fact_id"] in q.required_fact_ids


def test_at_twin_intent_propagated():
    q = _strict_row(n_facts=2)
    q.intent = "inferential"
    twin = _make_abstention_twin(q)
    assert twin is not None
    assert twin.intent == "inferential"


# ---------- CT twin ----------


def test_ct_twin_created_correctly(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.twins.infill_counterfactual", _ct_infill)
    monkeypatch.setattr(
        "rag_gt.generation.twins.generate_answer",
        lambda q, facts, llm: "Perturbed answer.",
    )

    q = _strict_row(n_facts=2)
    twin = _make_counterfactual_twin(q, _FakeLLM())
    assert twin is not None
    assert twin.twin_type == "counterfactual"
    assert twin.twin_of == q.q_id
    assert twin.gold_answer == "Perturbed answer."
    assert twin.difficulty_adversarial == "counterfactual"
    assert twin.q_id == f"{q.q_id}_ct"


def test_ct_twin_does_not_swallow_live_call_cap(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.twins.infill_counterfactual", _ct_infill)
    monkeypatch.setattr(
        "rag_gt.generation.twins.generate_answer",
        lambda q, facts, llm: (_ for _ in ()).throw(
            LiveCallBudgetExceeded("cap reached")
        ),
    )

    with pytest.raises(LiveCallBudgetExceeded, match="cap reached"):
        _make_counterfactual_twin(_strict_row(n_facts=2), _FakeLLM())


def test_ct_twin_metadata_has_span_info(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.twins.infill_counterfactual", _ct_infill)
    monkeypatch.setattr(
        "rag_gt.generation.twins.generate_answer",
        lambda q, facts, llm: "Perturbed answer.",
    )
    q = _strict_row(n_facts=2)
    twin = _make_counterfactual_twin(q, _FakeLLM())
    assert twin is not None
    meta = twin.twin_metadata
    assert "perturbed_fact_id" in meta
    assert "original_span" in meta
    assert "perturbed_span" in meta
    assert "contradiction_score" in meta


def test_ct_twin_skipped_when_infill_returns_none(monkeypatch):
    monkeypatch.setattr(
        "rag_gt.generation.twins.infill_counterfactual", lambda *a, **k: None
    )
    q = _strict_row(n_facts=2)
    twin = _make_counterfactual_twin(q, _FakeLLM())
    assert twin is None


def test_ct_twin_skipped_when_answer_is_abstention(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.twins.infill_counterfactual", _ct_infill)
    monkeypatch.setattr(
        "rag_gt.generation.twins.generate_answer",
        lambda q, facts, llm: ABSTENTION_TEXT,
    )
    q = _strict_row(n_facts=2)
    twin = _make_counterfactual_twin(q, _FakeLLM())
    assert twin is None


def test_ct_twin_skipped_when_answer_unchanged(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.twins.infill_counterfactual", _ct_infill)
    q = _strict_row(n_facts=2)
    monkeypatch.setattr(
        "rag_gt.generation.twins.generate_answer",
        lambda question, facts, llm: q.gold_answer,
    )
    twin = _make_counterfactual_twin(q, _FakeLLM())
    assert twin is None


def test_ct_twin_can_skip_answer_change_requirement(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.twins.infill_counterfactual", _ct_infill)
    q = _strict_row(n_facts=2)
    monkeypatch.setattr(
        "rag_gt.generation.twins.generate_answer",
        lambda question, facts, llm: q.gold_answer,
    )
    twin = _make_counterfactual_twin(
        q,
        _FakeLLM(),
        require_answer_change=False,
    )
    assert twin is not None


def test_ct_twin_skipped_when_no_facts():
    q = _strict_row(n_facts=2)
    q.required_facts = []  # simulate no required facts after construction
    twin = _make_counterfactual_twin(q, _FakeLLM())
    assert twin is None


def test_ct_twin_perturbed_facts_in_required(monkeypatch):
    """CT twin's required_facts contains the perturbed version of the targeted fact."""
    monkeypatch.setattr("rag_gt.generation.twins.infill_counterfactual", _ct_infill)
    monkeypatch.setattr(
        "rag_gt.generation.twins.generate_answer",
        lambda q, facts, llm: "Perturbed answer.",
    )
    q = _strict_row(n_facts=2)
    twin = _make_counterfactual_twin(q, _FakeLLM())
    assert twin is not None
    perturbed_id = twin.twin_metadata["perturbed_fact_id"]
    perturbed_fact = next(f for f in twin.required_facts if f.fact_id == perturbed_id)
    assert "(perturbed)" in perturbed_fact.text


# ---------- derive_twins ----------


def test_derive_twins_disabled_returns_empty():
    rows = [_strict_row()]
    result = derive_twins(rows, _FakeLLM(), twins_enabled=False)
    assert result == []


def test_derive_twins_empty_input_returns_empty():
    result = derive_twins([], _FakeLLM())
    assert result == []


def test_derive_twins_returns_at_and_ct(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.twins.infill_counterfactual", _ct_infill)
    monkeypatch.setattr(
        "rag_gt.generation.twins.generate_answer",
        lambda q, facts, llm: "Perturbed answer.",
    )

    rows = [_strict_row(q_id="d1_q001", n_facts=2)]
    twins = derive_twins(rows, _FakeLLM())

    twin_types = {t.twin_type for t in twins}
    assert "abstention" in twin_types
    assert "counterfactual" in twin_types
    assert all(t.twin_of == "d1_q001" for t in twins)


def test_derive_twins_single_fact_no_at(monkeypatch):
    """Single-fact rows get CT but not AT twins."""
    monkeypatch.setattr("rag_gt.generation.twins.infill_counterfactual", _ct_infill)
    monkeypatch.setattr(
        "rag_gt.generation.twins.generate_answer",
        lambda q, facts, llm: "Perturbed answer.",
    )

    rows = [_strict_row(q_id="d1_q001", n_facts=1)]
    twins = derive_twins(rows, _FakeLLM())
    at_twins = [t for t in twins if t.twin_type == "abstention"]
    assert at_twins == []


def test_derive_twins_multiple_rows(monkeypatch):
    """Twins are derived independently for each input row."""
    monkeypatch.setattr("rag_gt.generation.twins.infill_counterfactual", _ct_infill)
    monkeypatch.setattr(
        "rag_gt.generation.twins.generate_answer",
        lambda q, facts, llm: "Perturbed answer.",
    )

    rows = [
        _strict_row(q_id="d1_q001", n_facts=2),
        _strict_row(q_id="d1_q002", n_facts=2),
    ]
    twins = derive_twins(rows, _FakeLLM())
    twin_ids = {t.q_id for t in twins}
    # Each row should produce an _at and a _ct twin.
    assert "d1_q001_at" in twin_ids
    assert "d1_q001_ct" in twin_ids
    assert "d1_q002_at" in twin_ids
    assert "d1_q002_ct" in twin_ids


def test_derive_twins_respects_ct_sample_rate(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.twins.infill_counterfactual", _ct_infill)
    monkeypatch.setattr(
        "rag_gt.generation.twins.generate_answer",
        lambda q, facts, llm: "Perturbed answer.",
    )

    rows = [_strict_row(q_id="d1_q001", n_facts=2)]
    twins = derive_twins(rows, _FakeLLM(), ct_sample_rate=0.0)

    assert any(t.twin_type == "abstention" for t in twins)
    assert not any(t.twin_type == "counterfactual" for t in twins)
