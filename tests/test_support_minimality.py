from __future__ import annotations

from rag_gt.core.types import Fact, Span
from rag_gt.validation.minimality import support_minimality_check


def _fact(fid: str, text: str) -> Fact:
    return Fact(
        fact_id=fid,
        text=text,
        role="definition",  # type: ignore[arg-type]
        supporting_spans=[
            Span(doc_id="d1", chunk_id="d1_c001", start_token=0, end_token=10)
        ],
    )


def test_support_minimality_rejects_answer_entailed_without_fact(monkeypatch):
    facts = [
        _fact("F1", "Bellman optimality relates a state value to successor states."),
        _fact("F2", "Dynamic programming is a class of algorithms."),
    ]

    # Removing F1 leaves only F2 -> answer not entailed.
    # Removing F2 leaves F1 -> answer still entailed, so F2 is redundant.
    monkeypatch.setattr(
        "rag_gt.validation.minimality.batch_answer_entailment",
        lambda pairs: [False, True],
    )

    verdict = support_minimality_check(
        "How does the Bellman optimality equation relate state values to successors?",
        facts,
        "Bellman optimality relates a state value to successor states.",
    )

    assert not verdict.passed
    assert verdict.reason == "answer_entailed_without_fact"
    assert verdict.redundant_fact_ids == ["F2"]


def test_support_minimality_rejects_noncontributing_question_fact(monkeypatch):
    facts = [
        _fact("F1", "Bellman optimality relates a state value to successor states."),
        _fact("F2", "Dynamic programming is a class of algorithms."),
    ]
    assumptions = [
        {
            "assumption_id": "a1",
            "type": "REL",
            "hypothesis": "Bellman optimality relates state values to successors.",
            "passes": True,
            "entails_from_sfu": 0,
        }
    ]

    monkeypatch.setattr(
        "rag_gt.validation.minimality.batch_answer_entailment",
        lambda pairs: [False, False],
    )
    # One assumption checked against one remaining fact for each leave-one-out:
    # without F1 -> unsupported by F2; without F2 -> supported by F1.
    scores = iter([[0.10], [0.90]])
    monkeypatch.setattr(
        "rag_gt.validation.minimality.nli_batch",
        lambda pairs: next(scores),
    )

    verdict = support_minimality_check(
        "How does Bellman optimality relate state values to successors?",
        facts,
        "Bellman optimality relates state values to successors.",
        question_assumptions=assumptions,
    )

    assert not verdict.passed
    assert verdict.reason == "question_supported_without_noncontributing_fact"
    assert verdict.redundant_fact_ids == ["F2"]


def test_support_minimality_passes_when_each_fact_is_needed(monkeypatch):
    facts = [
        _fact("F1", "Bellman optimality relates a state value to successor states."),
        _fact("F2", "The recursive equation backs up over successor values."),
    ]
    assumptions = [
        {
            "assumption_id": "a1",
            "type": "REL",
            "hypothesis": "Bellman optimality relates state values to successors.",
            "passes": True,
            "entails_from_sfu": 0,
        },
        {
            "assumption_id": "a2",
            "type": "QUAL",
            "hypothesis": "The equation backs up over successor values.",
            "passes": True,
            "entails_from_sfu": 1,
        },
    ]

    monkeypatch.setattr(
        "rag_gt.validation.minimality.batch_answer_entailment",
        lambda pairs: [False, False],
    )
    # Two assumptions x one remaining fact per LOO. Neither reduced set passes
    # the full QA-NLI failure-rate caps.
    reduced_scores = iter([[0.10, 0.90], [0.90, 0.10]])
    monkeypatch.setattr(
        "rag_gt.validation.minimality.nli_batch",
        lambda pairs: next(reduced_scores),
    )

    verdict = support_minimality_check(
        "How does Bellman optimality back up values through successors?",
        facts,
        "Bellman optimality relates state values to successor states and backs up over successor values.",
        question_assumptions=assumptions,
    )

    assert verdict.passed
    assert verdict.reason == "accepted"
    assert verdict.redundant_fact_ids == []
