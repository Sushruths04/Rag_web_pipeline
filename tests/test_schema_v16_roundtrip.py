"""Phase A — v16 schema round-trip tests.

Verifies that:
1. All new v16 fields on QuestionGT default correctly (back-compat).
2. v16-populated rows survive a _to_dict → _from_dict round-trip.
3. Legacy JSONL rows (no v16 keys) load with safe defaults.
4. The difficulty_adversarial "clean" default is NOT emitted (byte-compat).
"""

from __future__ import annotations

import json

import pytest

from rag_gt.core.types import Fact, MSFS, QuestionGT, Span
from rag_gt.storage.gt_io import _from_dict, _to_dict


# ---------- helpers ----------


def _make_span() -> Span:
    return Span(doc_id="d1", chunk_id="d1_c000000", start_token=0, end_token=5)


def _make_fact(fid: str) -> Fact:
    return Fact(
        fact_id=fid,
        text="TD learning updates value estimates incrementally.",
        role="definition",
        supporting_spans=[_make_span()],
    )


def _base_question(**overrides) -> QuestionGT:
    defaults = dict(
        q_id="d1_q001",
        question="What does TD learning update?",
        gold_answer="TD learning updates value estimates incrementally.",
        msfs_list=[MSFS(msfs_id="d1_q001_msfs1", fact_ids=["F001"])],
        doc_ids=["d1"],
        required_fact_ids=["F001"],
        difficulty_reasoning_depth=1,
        difficulty_semantic_distance="intra_doc",
        required_facts=[_make_fact("F001")],
    )
    defaults.update(overrides)
    return QuestionGT(**defaults)


# ---------- back-compat defaults ----------


def test_v16_fields_default_to_empty():
    q = _base_question()
    assert q.tf_sfg_edges == []
    assert q.question_assumptions == []
    assert q.distractor_spans == []
    assert q.twin_of is None
    assert q.twin_type is None
    assert q.twin_metadata is None
    assert q.reasoning_trace == []
    assert q.difficulty_adversarial == "clean"
    assert q.intent == ""


def test_clean_difficulty_adversarial_not_emitted():
    """'clean' is the default; emitting it would bloat legacy files."""
    q = _base_question()
    d = _to_dict(q)
    assert "difficulty_adversarial" not in d


def test_populated_difficulty_adversarial_is_emitted():
    q = _base_question(difficulty_adversarial="abstention")
    d = _to_dict(q)
    assert d["difficulty_adversarial"] == "abstention"


def test_empty_v16_lists_not_emitted():
    """Empty lists must NOT appear in the serialised output (back-compat)."""
    q = _base_question()
    d = _to_dict(q)
    for key in ("tf_sfg_edges", "question_assumptions", "distractor_spans", "reasoning_trace"):
        assert key not in d, f"'{key}' should be omitted when empty"


def test_none_twin_fields_not_emitted():
    q = _base_question()
    d = _to_dict(q)
    for key in ("twin_of", "twin_type", "twin_metadata"):
        assert key not in d, f"'{key}' should be omitted when None"


# ---------- round-trip with v16 data ----------


_SAMPLE_TF_SFG_EDGE = {
    "edge_id": "e1",
    "src": "F001",
    "dst": "F002",
    "type": "causal",
    "bridging_fact_id": "F001",
    "bridging_quote": "TD learning updates value estimates",
    "nli_score": 0.79,
}

_SAMPLE_ASSUMPTION = {
    "assumption_id": "a1",
    "type": "REL",
    "args": ["TD learning", "value estimates", "causal"],
    "entails_from": "F001",
    "nli_score": 0.82,
}

_SAMPLE_DISTRACTOR = {
    "distractor_id": "d1",
    "doc_id": "d1",
    "page": 42,
    "bbox": [10.0, 20.0, 100.0, 30.0],
    "char_span": [500, 600],
    "source_fact_id": "F001",
    "cosine_to_support": 0.61,
    "nli_to_question": 0.12,
}

_SAMPLE_REASONING_STEP = {
    "step_id": "s1",
    "claim": "TD learning incrementally refines estimates.",
    "fact_ids": ["F001"],
    "relation_type": "definition",
    "bbox_refs": [],
}


def test_v16_strict_row_roundtrip():
    q = _base_question(
        tf_sfg_edges=[_SAMPLE_TF_SFG_EDGE],
        question_assumptions=[_SAMPLE_ASSUMPTION],
        distractor_spans=[_SAMPLE_DISTRACTOR],
        reasoning_trace=[_SAMPLE_REASONING_STEP],
        difficulty_adversarial="distractor",
        intent="inferential",
    )
    d = _to_dict(q)
    q2 = _from_dict(d)

    assert q2.tf_sfg_edges == [_SAMPLE_TF_SFG_EDGE]
    assert q2.question_assumptions == [_SAMPLE_ASSUMPTION]
    assert q2.distractor_spans == [_SAMPLE_DISTRACTOR]
    assert q2.reasoning_trace == [_SAMPLE_REASONING_STEP]
    assert q2.difficulty_adversarial == "distractor"
    assert q2.intent == "inferential"


def test_abstention_twin_roundtrip():
    q = _base_question(
        twin_of="d1_q001",
        twin_type="abstention",
        twin_metadata={"removed_fact_id": "F001"},
        difficulty_adversarial="abstention",
        gold_answer="Insufficient information to answer from the provided context.",
    )
    d = _to_dict(q)
    assert d["twin_of"] == "d1_q001"
    assert d["twin_type"] == "abstention"
    assert d["twin_metadata"] == {"removed_fact_id": "F001"}

    q2 = _from_dict(d)
    assert q2.twin_of == "d1_q001"
    assert q2.twin_type == "abstention"
    assert q2.twin_metadata == {"removed_fact_id": "F001"}
    assert q2.difficulty_adversarial == "abstention"


def test_counterfactual_twin_roundtrip():
    q = _base_question(
        twin_of="d1_q001",
        twin_type="counterfactual",
        twin_metadata={
            "perturbed_span": "Q-learning",
            "original_value": "TD(lambda)",
            "perturbed_value": "Q-learning",
        },
        difficulty_adversarial="counterfactual",
    )
    d = _to_dict(q)
    q2 = _from_dict(d)
    assert q2.twin_type == "counterfactual"
    assert q2.twin_metadata["original_value"] == "TD(lambda)"


# ---------- legacy JSONL back-compat ----------


_LEGACY_ROW = {
    "q_id": "legacy_q001",
    "question": "What is temporal difference learning?",
    "gold_answer": "A method that updates value estimates from incomplete episodes.",
    "msfs_list": [{"msfs_id": "legacy_q001_msfs1", "fact_ids": ["F099"]}],
    "doc_ids": ["legacy_doc"],
    "required_fact_ids": ["F099"],
    "required_facts": [
        {
            "fact_id": "F099",
            "text": "Temporal difference learning updates value estimates from incomplete episodes.",
            "role": "definition",
            "supporting_spans": [
                {
                    "doc_id": "legacy_doc",
                    "chunk_id": "legacy_doc_c000000",
                    "start_token": 0,
                    "end_token": 5,
                }
            ],
        }
    ],
    "difficulty_reasoning_depth": 1,
    "difficulty_semantic_distance": "intra_doc",
}


def test_legacy_row_loads_with_v16_defaults():
    q = _from_dict(_LEGACY_ROW)
    assert q.tf_sfg_edges == []
    assert q.question_assumptions == []
    assert q.distractor_spans == []
    assert q.twin_of is None
    assert q.twin_type is None
    assert q.twin_metadata is None
    assert q.reasoning_trace == []
    assert q.difficulty_adversarial == "clean"
    assert q.intent == ""


def test_json_serialization_is_valid():
    """The dict produced by _to_dict must be JSON-serialisable (no sets, etc.)."""
    q = _base_question(
        tf_sfg_edges=[_SAMPLE_TF_SFG_EDGE],
        reasoning_trace=[_SAMPLE_REASONING_STEP],
        intent="factoid",
    )
    raw = json.dumps(_to_dict(q), ensure_ascii=False)
    assert json.loads(raw)["intent"] == "factoid"
