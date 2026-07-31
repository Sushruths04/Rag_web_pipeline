"""Phase B tests — QA-NLI question assumption gate (P2).

All LLM and NLI calls are stubbed so tests are deterministic and offline.
"""

from __future__ import annotations

import pytest

from rag_gt.core.types import Fact, Span
from rag_gt.observability.cost_tracker import LiveCallBudgetExceeded
from rag_gt.validation.question_assumption_gate import (
    QANLIGateVerdict,
    check_question_assumptions,
    verdicts_to_dicts,
)


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


_FACTS = [
    _fact("F001", "TD learning updates value estimates using temporal differences."),
    _fact("F002", "Value estimates converge to the true value function over time."),
]

_QUESTION = "How does TD learning cause value estimates to converge?"


class _FakeLLMOK:
    """Returns a valid JSON assumption decomposition."""
    model = "fake-qa-nli"

    def generate(self, prompt, temperature=0.0, max_tokens=512):
        return """{
  "assumptions": [
    {"id": "a1", "type": "EXIST", "args": ["TD learning"],
     "hypothesis": "The source mentions TD learning."},
    {"id": "a2", "type": "REL", "args": ["TD learning", "value estimates", "causal"],
     "hypothesis": "TD learning causes value estimates to update."},
    {"id": "a3", "type": "QUAL", "args": ["value estimates", "convergent"],
     "hypothesis": "Value estimates are convergent."}
  ]
}"""


class _FakeLLMEmpty:
    model = "fake-qa-nli-empty"

    def generate(self, prompt, temperature=0.0, max_tokens=512):
        return '{"assumptions": []}'


class _FakeLLMBroken:
    model = "fake-qa-nli-broken"

    def generate(self, prompt, temperature=0.0, max_tokens=512):
        raise RuntimeError("simulated LLM error")


class _FakeLLMBudgetExceeded:
    model = "paid-cap"

    def generate(self, prompt, temperature=0.0, max_tokens=512):
        raise LiveCallBudgetExceeded("cap reached")


# ---------- stub NLI ----------


def _stub_nli(monkeypatch, scores):
    """Replace nli_batch with a function that returns `scores` cyclically."""
    call_count = [0]

    def _fake_nli_batch(pairs):
        start = call_count[0]
        result = []
        for i in range(len(pairs)):
            result.append(scores[(start + i) % len(scores)])
        call_count[0] += len(pairs)
        return result

    monkeypatch.setattr(
        "rag_gt.validation.question_assumption_gate.nli_batch",
        _fake_nli_batch,
    )


# ---------- tests ----------


def test_gate_accepts_when_all_pass(monkeypatch):
    _stub_nli(monkeypatch, [0.80])  # all assumptions pass
    verdict = check_question_assumptions(_QUESTION, _FACTS, _FakeLLMOK())
    assert verdict.accepted is True
    assert len(verdict.assumption_verdicts) == 3
    assert all(v.passes for v in verdict.assumption_verdicts)
    assert verdict.rel_fail_rate == 0.0


def test_gate_rejects_when_rel_fails(monkeypatch):
    # Return low score for every NLI call — REL assumption will fail.
    _stub_nli(monkeypatch, [0.10])
    verdict = check_question_assumptions(
        _QUESTION,
        _FACTS,
        _FakeLLMOK(),
        max_rel_fail_rate=0.10,  # strict threshold
    )
    assert verdict.accepted is False
    assert verdict.rel_fail_rate > 0.0
    assert verdict.guidance  # guidance string is populated


def test_gate_passes_through_on_empty_assumptions(monkeypatch):
    _stub_nli(monkeypatch, [0.80])
    verdict = check_question_assumptions(_QUESTION, _FACTS, _FakeLLMEmpty())
    assert verdict.accepted is True
    assert verdict.assumption_verdicts == []
    assert "passthrough" in verdict.reason


def test_gate_passes_through_on_llm_error(monkeypatch):
    _stub_nli(monkeypatch, [0.80])
    verdict = check_question_assumptions(_QUESTION, _FACTS, _FakeLLMBroken())
    assert verdict.accepted is True
    assert "passthrough" in verdict.reason


def test_gate_does_not_fail_open_when_live_call_cap_is_reached(monkeypatch):
    _stub_nli(monkeypatch, [0.80])
    with pytest.raises(LiveCallBudgetExceeded, match="cap reached"):
        check_question_assumptions(_QUESTION, _FACTS, _FakeLLMBudgetExceeded())


def test_gate_rejects_on_empty_facts():
    verdict = check_question_assumptions(_QUESTION, [], _FakeLLMOK())
    assert verdict.accepted is False
    assert verdict.reason == "empty_chain_facts"


def test_verdicts_to_dicts(monkeypatch):
    _stub_nli(monkeypatch, [0.75])
    verdict = check_question_assumptions(_QUESTION, _FACTS, _FakeLLMOK())
    dicts = verdicts_to_dicts(verdict.assumption_verdicts)
    assert len(dicts) == 3
    assert all("assumption_id" in d for d in dicts)
    assert all("nli_score" in d for d in dicts)
    assert all("passes" in d for d in dicts)


def test_assumption_type_is_uppercase(monkeypatch):
    _stub_nli(monkeypatch, [0.80])
    verdict = check_question_assumptions(_QUESTION, _FACTS, _FakeLLMOK())
    for v in verdict.assumption_verdicts:
        assert v.assumption_type == v.assumption_type.upper()


def test_guidance_populated_on_rejection(monkeypatch):
    _stub_nli(monkeypatch, [0.10])
    verdict = check_question_assumptions(
        _QUESTION, _FACTS, _FakeLLMOK(), max_rel_fail_rate=0.05
    )
    if not verdict.accepted:
        assert len(verdict.guidance) > 20  # non-trivial guidance string


def test_cross_check_rel_edge_type_mismatch_does_not_hard_reject(monkeypatch):
    """REL-edge type mismatch is a soft check — gate should not reject solely on this."""
    _stub_nli(monkeypatch, [0.80])
    chain_edges = [{"type": "definition", "src": "F001", "dst": "F002"}]
    # The assumption claims "causal" but the edge is "definition" — soft mismatch only.
    verdict = check_question_assumptions(
        _QUESTION, _FACTS, _FakeLLMOK(), chain_edges=chain_edges
    )
    # Gate still accepts because NLI scores are high (the soft check only logs).
    assert verdict.accepted is True


# ---------- Synonym canonicalization tests ----------


class _FakeLLMRelSynonym:
    """Returns a REL assumption using a non-canonical synonym label."""

    def __init__(self, rel_label: str):
        self._rel = rel_label
        self.model = f"fake-qa-nli-{rel_label}"

    def generate(self, prompt, temperature=0.0, max_tokens=512):
        return (
            '{"assumptions": ['
            '{"id": "a1", "type": "REL",'
            f' "args": ["TD learning", "value estimates", "{self._rel}"],'
            ' "hypothesis": "TD learning causes value estimates to update."}'
            "]}"
        )


def _enforce_match_verdict(monkeypatch, rel_label: str, edge_type: str) -> bool:
    """Run the gate with enforce_edge_relation_match=True and return accepted."""
    _stub_nli(monkeypatch, [0.80])
    chain_edges = [{"type": edge_type, "src": "F001", "dst": "F002"}]
    verdict = check_question_assumptions(
        _QUESTION,
        _FACTS,
        _FakeLLMRelSynonym(rel_label),
        chain_edges=chain_edges,
        enforce_edge_relation_match=True,
        max_rel_mismatch_rate=0.0,  # strict: any mismatch fails
    )
    return verdict.accepted


def test_synonym_mechanism_matches_causal_edge(monkeypatch):
    """'mechanism' in assumption should canonicalize to 'causal' and match a causal edge."""
    assert _enforce_match_verdict(monkeypatch, "mechanism", "causal") is True


def test_synonym_comparison_matches_comparative_edge(monkeypatch):
    """'comparison' in assumption should canonicalize to 'comparative'."""
    assert _enforce_match_verdict(monkeypatch, "comparison", "comparative") is True


def test_synonym_contrast_matches_comparative_edge(monkeypatch):
    """'contrast' in assumption should canonicalize to 'comparative'."""
    assert _enforce_match_verdict(monkeypatch, "contrast", "comparative") is True


def test_synonym_list_matches_descriptive_edge(monkeypatch):
    """'list' in assumption should canonicalize to 'descriptive'."""
    assert _enforce_match_verdict(monkeypatch, "list", "descriptive") is True


def test_max_rel_mismatch_rate_threshold(monkeypatch):
    """Gate rejects only when mismatch fraction exceeds max_rel_mismatch_rate."""
    _stub_nli(monkeypatch, [0.80])
    chain_edges = [{"type": "definition", "src": "F001", "dst": "F002"}]
    # assumption claims "causal" — mismatches the "definition" edge (after canon both differ).
    # With max_rel_mismatch_rate=1.0 (allow all mismatches) the gate must accept.
    verdict_lenient = check_question_assumptions(
        _QUESTION,
        _FACTS,
        _FakeLLMRelSynonym("causal"),
        chain_edges=chain_edges,
        enforce_edge_relation_match=True,
        max_rel_mismatch_rate=1.0,
    )
    assert verdict_lenient.accepted is True

    # With max_rel_mismatch_rate=0.0 (zero tolerance) the gate must reject.
    verdict_strict = check_question_assumptions(
        _QUESTION,
        _FACTS,
        _FakeLLMRelSynonym("causal"),
        chain_edges=chain_edges,
        enforce_edge_relation_match=True,
        max_rel_mismatch_rate=0.0,
    )
    assert verdict_strict.accepted is False
    assert "relation_mismatch_with_tf_sfg" in verdict_strict.reason


def test_per_fact_coverage_rejects_uncovered_fact(monkeypatch):
    # All assumptions best-match SFU0; SFU1 never contributes.
    _stub_nli(monkeypatch, [0.95, 0.05])
    verdict = check_question_assumptions(
        _QUESTION,
        _FACTS,
        _FakeLLMOK(),
        require_per_fact_coverage=True,
    )
    assert verdict.accepted is False
    assert verdict.reason == "missing_required_fact_coverage:1"
    assert verdict.missing_fact_indices == [1]
    assert "uncovered fact indices" in verdict.guidance


def test_per_fact_coverage_accepts_when_each_fact_contributes(monkeypatch):
    # a1 -> SFU0, a2 -> SFU0, a3 -> SFU1.
    _stub_nli(monkeypatch, [0.95, 0.05, 0.95, 0.05, 0.05, 0.95])
    verdict = check_question_assumptions(
        _QUESTION,
        _FACTS,
        _FakeLLMOK(),
        require_per_fact_coverage=True,
    )
    assert verdict.accepted is True
    assert verdict.missing_fact_indices == []


def test_per_fact_coverage_rescues_explicit_question_anchor_with_strong_edge(monkeypatch):
    facts = [
        _fact(
            "F001",
            "For some state s we would like to know whether we should change the policy "
            "to deterministically choose an action.",
        ),
        _fact(
            "F002",
            "If π is a deterministic policy, then following π observes returns only "
            "for one action from each state.",
        ),
    ]
    question = (
        "Why does determining whether to change the policy to deterministically choose "
        "an action lead to observing returns only for one action from each state?"
    )
    # Decomposition assigns every assumption to SFU1, but the question directly
    # names concrete anchors from SFU0 and the TF-SFG edge is strong.
    _stub_nli(monkeypatch, [0.05, 0.95])
    verdict = check_question_assumptions(
        question,
        facts,
        _FakeLLMOK(),
        chain_edges=[
            {
                "src": "F001",
                "dst": "F002",
                "type": "causal",
                "nli_score": 0.99,
                "joint_only_margin": 0.90,
            }
        ],
        require_per_fact_coverage=True,
    )
    assert verdict.accepted is True
    assert verdict.missing_fact_indices == []
