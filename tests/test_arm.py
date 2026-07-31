"""Tests for P5 ARM (Abstractive Reasoning Map).

All LLM and NLI calls are stubbed — tests run offline.
"""

from __future__ import annotations

import pytest

from rag_gt.core.types import Fact, SourceBBox, Span
from rag_gt.generation.arm import ARMResult, ARMStep, build_abstractive_answer, ngram_overlap
from rag_gt.observability.cost_tracker import CostTracker, LiveCallBudgetExceeded, TrackedLLM


# ---------- helpers ----------


def _span() -> Span:
    return Span(
        doc_id="d1",
        chunk_id="d1_c000000",
        start_token=0,
        end_token=5,
        char_start=10,
        char_end=80,
        page_start=3,
        page_end=3,
        bboxes=[SourceBBox(page_no=3, l=1.0, t=2.0, r=3.0, b=4.0)],
        source_path="data/docs/example.pdf",
        source_sha1="abc123",
        source_text_sha1="def456",
        extractor="docling",
    )


def _fact(fid: str, text: str, canonical: str = "") -> Fact:
    f = Fact(
        fact_id=fid,
        text=text,
        role="definition",
        supporting_spans=[_span()],
    )
    f.canonical_form = canonical or text
    return f


class _LLMSynthesis:
    """Stub LLM that returns a valid synthesis JSON then a compose answer."""

    model = "fake-arm"
    _calls: int

    def __init__(self):
        self._calls = 0

    def generate(self, prompt: str, temperature=0.0, max_tokens=512) -> str:
        self._calls += 1
        if "Return JSON only" in prompt:
            return '{"claim": "Abbreviated claim here.", "rationale": "own phrasing"}'
        # Compose prompt
        return "Final composed answer from the steps."


class _LLMBadJSON:
    """Stub LLM that always returns invalid JSON for synthesis."""

    model = "fake-bad"

    def generate(self, prompt: str, temperature=0.0, max_tokens=512) -> str:
        if "Return JSON only" in prompt:
            return "not json at all"
        return "Fallback composed answer."


class _LLMOmitsOneStep:
    model = "fake-omit"

    def generate(self, prompt: str, temperature=0.0, max_tokens=512) -> str:
        if "Return JSON only" in prompt:
            if "First source fact" in prompt:
                return '{"claim": "First claim.", "rationale": "own phrasing"}'
            return '{"claim": "Second claim.", "rationale": "own phrasing"}'
        return "Second claim."


class _LLMAddsUnsupportedContent:
    model = "fake-unsupported-compose"

    def generate(self, prompt: str, temperature=0.0, max_tokens=512) -> str:
        if "Return JSON only" in prompt:
            if "First source fact" in prompt:
                return '{"claim": "First claim.", "rationale": "own phrasing"}'
            return '{"claim": "Second claim.", "rationale": "own phrasing"}'
        return "First claim. Second claim. Extra unsupported claim."


class _LLMCapturesComposeRelation:
    model = "fake-capture-relation"

    def __init__(self):
        self.compose_prompt = ""

    def generate(self, prompt: str, temperature=0.0, max_tokens=512) -> str:
        if "Return JSON only" in prompt:
            if "First source fact" in prompt:
                return '{"claim": "First claim.", "rationale": "own phrasing"}'
            return '{"claim": "Second claim.", "rationale": "own phrasing"}'
        self.compose_prompt = prompt
        return "First claim. Second claim."


# ---------- ngram_overlap ----------


def test_ngram_overlap_identical():
    text = "the quick brown fox jumps over the lazy dog"
    assert ngram_overlap(text, text) == pytest.approx(1.0)


def test_ngram_overlap_no_overlap():
    assert ngram_overlap("alpha beta gamma delta", "one two three four") == pytest.approx(0.0)


def test_ngram_overlap_partial():
    syn = "the quick brown fox leaps"
    src = "the quick brown fox jumps over the lazy dog"
    score = ngram_overlap(syn, src)
    assert 0.0 < score < 1.0


def test_ngram_overlap_short_synthesis():
    # Synthesis shorter than n (4) — overlap is 0.
    assert ngram_overlap("short", "short phrase here today") == pytest.approx(0.0)


def test_ngram_overlap_empty_source():
    assert ngram_overlap("hello world from me here", "") == pytest.approx(0.0)


# ---------- build_abstractive_answer ----------


def test_arm_returns_arm_result(monkeypatch):
    monkeypatch.setattr(
        "rag_gt.generation.arm.nli_entailment", lambda p, h: 0.80
    )
    facts = [_fact("F001", "The learning rate controls convergence speed in gradient descent.")]
    result = build_abstractive_answer("What controls convergence?", facts, _LLMSynthesis())
    assert isinstance(result, ARMResult)
    assert result.construction_method == "SFU+ARM"


def test_arm_reasoning_trace_populated(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.80)
    facts = [
        _fact("F001", "Fact one about learning rate and neural networks training."),
        _fact("F002", "Fact two about gradient descent convergence properties."),
    ]
    result = build_abstractive_answer("Q?", facts, _LLMSynthesis())
    assert len(result.reasoning_trace) == 2
    for step in result.reasoning_trace:
        assert "step_id" in step
        assert "claim" in step
        assert "fact_ids" in step
        assert "nli_score" in step
        assert "np_overlap" in step
        assert "passes" in step


def test_arm_reasoning_trace_keeps_bbox_provenance(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.80)
    facts = [_fact("F001", "Fact with page-level source provenance.")]
    result = build_abstractive_answer("Q?", facts, _LLMSynthesis())
    ref = result.reasoning_trace[0]["bbox_refs"][0]
    assert ref["doc_id"] == "d1"
    assert ref["chunk_id"] == "d1_c000000"
    assert ref["page_start"] == 3
    assert ref["char_start"] == 10
    assert ref["char_end"] == 80
    assert ref["source_path"] == "data/docs/example.pdf"
    assert ref["source_sha1"] == "abc123"
    assert ref["bboxes"][0]["page_no"] == 3


def test_arm_step_ids_sequential(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.75)
    facts = [
        _fact("F001", "First fact about temperature and entropy in thermodynamics."),
        _fact("F002", "Second fact about pressure and volume relations in gases."),
        _fact("F003", "Third fact about ideal gas law and its applications."),
    ]
    result = build_abstractive_answer("Q?", facts, _LLMSynthesis())
    step_ids = [s["step_id"] for s in result.reasoning_trace]
    assert step_ids == ["s1", "s2", "s3"]


def test_arm_all_steps_pass_when_nli_high(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.90)
    facts = [_fact("F001", "The policy gradient theorem relates value functions to rewards.")]
    result = build_abstractive_answer("Q?", facts, _LLMSynthesis())
    assert result.all_steps_pass is True


def test_arm_all_steps_fail_when_nli_low(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.10)
    facts = [_fact("F001", "Temporal difference learning bootstraps from estimated values.")]
    result = build_abstractive_answer("Q?", facts, _LLMSynthesis())
    assert result.all_steps_pass is False


def test_arm_gold_answer_non_empty(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.80)
    facts = [_fact("F001", "Backpropagation computes gradients via chain rule in neural networks.")]
    result = build_abstractive_answer("Q?", facts, _LLMSynthesis())
    assert len(result.gold_answer.strip()) > 0


def test_arm_falls_back_when_composed_answer_drops_step(monkeypatch):
    def _nli(premise, hypothesis):
        if premise == "Second claim." and hypothesis == "First claim.":
            return 0.10
        return 0.80

    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", _nli)
    facts = [
        _fact("F001", "First source fact about routing."),
        _fact("F002", "Second source fact about addressing."),
    ]
    result = build_abstractive_answer("Q?", facts, _LLMOmitsOneStep())
    assert result.gold_answer == "First claim. Second claim."


def test_arm_falls_back_when_composed_answer_adds_unsupported_content(monkeypatch):
    unsupported = "First claim. Second claim. Extra unsupported claim."

    def _nli(premise, hypothesis):
        if hypothesis == unsupported:
            return 0.10
        return 0.80

    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", _nli)
    facts = [
        _fact("F001", "First source fact about routing."),
        _fact("F002", "Second source fact about addressing."),
    ]
    result = build_abstractive_answer("Q?", facts, _LLMAddsUnsupportedContent())
    assert result.gold_answer == "First claim. Second claim."


def test_arm_compose_prompt_includes_verified_relation(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.80)
    llm = _LLMCapturesComposeRelation()
    facts = [
        _fact("F001", "First source fact about policy changes."),
        _fact("F002", "Second source fact about observed returns."),
    ]
    edges = [
        {
            "src": "F001",
            "dst": "F002",
            "type": "causal",
            "relation_claim": "Policy changes affect which action returns are observed.",
            "bridging_quote": "returns only for one of the actions",
        }
    ]
    build_abstractive_answer("Why?", facts, llm, tf_sfg_edges=edges)
    assert "Verified relation evidence" in llm.compose_prompt
    assert "type=causal" in llm.compose_prompt
    assert "Policy changes affect which action returns are observed." in llm.compose_prompt


def test_arm_fallback_preserves_verified_relation(monkeypatch):
    def _nli(premise, hypothesis):
        if premise == "Second claim." and hypothesis == "First claim.":
            return 0.10
        return 0.80

    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", _nli)
    facts = [
        _fact("F001", "First source fact about policy changes."),
        _fact("F002", "Second source fact about observed returns."),
    ]
    edges = [
        {
            "src": "F001",
            "dst": "F002",
            "type": "causal",
            "relation_claim": "Policy changes affect which action returns are observed.",
        }
    ]
    result = build_abstractive_answer("Why?", facts, _LLMOmitsOneStep(), tf_sfg_edges=edges)
    assert result.gold_answer == (
        "Policy changes affect which action returns are observed. "
        "First claim. Second claim."
    )


def test_arm_raises_on_empty_facts(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.80)
    with pytest.raises(ValueError, match="at least one chain fact"):
        build_abstractive_answer("Q?", [], _LLMSynthesis())


def test_arm_does_not_fallback_when_live_call_cap_is_reached(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.80)
    facts = [_fact("F001", "A source fact that would require paid synthesis.")]
    llm = TrackedLLM(
        _LLMSynthesis(),
        CostTracker(max_live_api_calls=0),
        stage="arm_step_synth",
        doc_id="d1",
    )

    with pytest.raises(LiveCallBudgetExceeded, match="cap reached"):
        build_abstractive_answer("Q?", facts, llm)


def test_arm_bad_json_falls_back_to_truncated_source(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.80)
    facts = [_fact("F001", "The reward function defines the objective of reinforcement learning.")]
    result = build_abstractive_answer("Q?", facts, _LLMBadJSON())
    assert len(result.reasoning_trace) == 1
    assert len(result.reasoning_trace[0]["claim"]) > 0


def test_arm_fact_ids_in_step(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.80)
    facts = [_fact("F042", "Q-learning is an off-policy temporal difference control algorithm.")]
    result = build_abstractive_answer("Q?", facts, _LLMSynthesis())
    assert result.reasoning_trace[0]["fact_ids"] == ["F042"]


def test_arm_tf_sfg_edges_annotate_relation_type(monkeypatch):
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.80)
    facts = [
        _fact("F001", "Markov decision processes model sequential decision problems."),
        _fact("F002", "Bellman equations describe the recursive structure of value functions."),
    ]
    edges = [{"src": "F001", "dst": "F002", "type": "causal", "nli_score": 0.78}]
    result = build_abstractive_answer("Q?", facts, _LLMSynthesis(), tf_sfg_edges=edges)
    step_by_id = {s["step_id"]: s for s in result.reasoning_trace}
    assert step_by_id["s2"]["relation_type"] == "causal"


# ---------- cga.py ARM delegation ----------


def test_cga_use_arm_returns_cga_result(monkeypatch):
    """build_constructive_gold_answer with use_arm=True wraps ARMResult as CGAResult."""
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.80)

    from rag_gt.generation.cga import CGAResult, build_constructive_gold_answer

    facts = [_fact("F001", "The value function estimates expected future rewards in RL.")]
    result = build_constructive_gold_answer(
        "What does the value function estimate?",
        facts,
        _LLMSynthesis(),
        use_arm=True,
    )
    assert isinstance(result, CGAResult)
    assert result.construction_method == "SFU+ARM"
    assert len(result.reasoning_trace) == 1


def test_max_np_overlap_from_cga_reaches_arm(monkeypatch):
    """max_np_overlap passed to build_constructive_gold_answer flows into ARM step evaluation."""
    monkeypatch.setattr("rag_gt.generation.arm.nli_entailment", lambda p, h: 0.90)
    monkeypatch.setattr("rag_gt.generation.arm.ngram_overlap", lambda syn, src, n=4: 0.3636)

    from rag_gt.generation.cga import build_constructive_gold_answer

    facts = [_fact("F001", "The learning rate controls convergence speed in gradient descent.")]

    result_low = build_constructive_gold_answer(
        "What controls convergence?",
        facts,
        _LLMSynthesis(),
        max_np_overlap=0.30,
        use_arm=True,
    )
    assert result_low.reasoning_trace[-1]["passes"] is False

    result_high = build_constructive_gold_answer(
        "What controls convergence?",
        facts,
        _LLMSynthesis(),
        max_np_overlap=0.40,
        use_arm=True,
    )
    assert result_high.reasoning_trace[-1]["passes"] is True


def test_cga_use_arm_false_uses_paraphrase(monkeypatch):
    """build_constructive_gold_answer with use_arm=False goes through CGA path."""
    from rag_gt.generation.cga import build_constructive_gold_answer
    import rag_gt.generation.cga as cga_mod

    called = []

    def _fake_decompose(text, **kw):
        called.append("decompose")
        return ["clause one.", "clause two."]

    monkeypatch.setattr("rag_gt.generation.cga.decompose_into_atomic_clauses", _fake_decompose)
    monkeypatch.setattr(
        "rag_gt.generation.cga.atomic_clause_entailment",
        lambda clauses, texts, **kw: [(True, 0.80, 0)] * len(clauses),
    )

    facts = [_fact("F001", "The Q-function maps state-action pairs to expected returns.")]
    build_constructive_gold_answer("Q?", facts, _LLMSynthesis(), use_arm=False)
    assert "decompose" in called
