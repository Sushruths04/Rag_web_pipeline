"""Tests for V16.2 CostTracker."""

from __future__ import annotations

import pytest

from rag_gt.observability.cost_tracker import (
    CostEvent,
    CostSummary,
    CostTracker,
    LiveCallBudgetExceeded,
    TrackedLLM,
)


def _ev(
    stage: str = "question_gen",
    doc_id: str = "d001",
    model: str = "gpt-4o-mini",
    prompt_chars: int = 1000,
    completion_chars: int = 200,
    cache_hit: bool = False,
    cache_source: str = "",
) -> CostEvent:
    return CostEvent(
        stage=stage,
        doc_id=doc_id,
        model=model,
        call_count=1,
        prompt_chars=prompt_chars,
        completion_chars=completion_chars,
        cache_hit=cache_hit,
        cache_source=cache_source,
    )


# ---------------------------------------------------------------------------
# Basic record + aggregate
# ---------------------------------------------------------------------------

def test_empty_tracker_returns_zero_summary() -> None:
    tracker = CostTracker()
    s = tracker.get_aggregate_summary()
    assert s.live_api_calls == 0
    assert s.cache_hit_calls == 0
    assert s.total_logical_calls == 0
    assert s.prompt_chars == 0
    assert s.completion_chars == 0


def test_single_live_event_counted() -> None:
    tracker = CostTracker()
    tracker.record(_ev(prompt_chars=500, completion_chars=100, cache_hit=False))
    s = tracker.get_aggregate_summary()
    assert s.live_api_calls == 1
    assert s.cache_hit_calls == 0
    assert s.total_logical_calls == 1
    assert s.prompt_chars == 500
    assert s.completion_chars == 100


def test_single_cache_hit_counted() -> None:
    tracker = CostTracker()
    tracker.record(_ev(cache_hit=True, cache_source="tfsfg_sqlite"))
    s = tracker.get_aggregate_summary()
    assert s.live_api_calls == 0
    assert s.cache_hit_calls == 1
    assert s.total_logical_calls == 1


def test_mixed_events_separated() -> None:
    tracker = CostTracker()
    tracker.record(_ev(cache_hit=False))
    tracker.record(_ev(cache_hit=True))
    tracker.record(_ev(cache_hit=False))
    s = tracker.get_aggregate_summary()
    assert s.live_api_calls == 2
    assert s.cache_hit_calls == 1
    assert s.total_logical_calls == 3


# ---------------------------------------------------------------------------
# Per-doc isolation
# ---------------------------------------------------------------------------

def test_per_doc_summary_isolates_correctly() -> None:
    tracker = CostTracker()
    tracker.record(_ev(doc_id="d001", prompt_chars=100))
    tracker.record(_ev(doc_id="d002", prompt_chars=200))
    tracker.record(_ev(doc_id="d001", prompt_chars=50))
    s1 = tracker.get_summary_for_doc("d001")
    s2 = tracker.get_summary_for_doc("d002")
    assert s1.prompt_chars == 150
    assert s2.prompt_chars == 200
    assert s1.total_logical_calls == 2
    assert s2.total_logical_calls == 1


def test_unknown_doc_returns_zero_summary() -> None:
    tracker = CostTracker()
    tracker.record(_ev(doc_id="d001"))
    s = tracker.get_summary_for_doc("nonexistent")
    assert s.total_logical_calls == 0


# ---------------------------------------------------------------------------
# Stage and model breakdowns
# ---------------------------------------------------------------------------

def test_by_stage_breakdown() -> None:
    tracker = CostTracker()
    tracker.record(_ev(stage="tf_sfg_classify", prompt_chars=800))
    tracker.record(_ev(stage="question_gen", prompt_chars=600))
    tracker.record(_ev(stage="question_gen", prompt_chars=600))
    s = tracker.get_aggregate_summary()
    assert "tf_sfg_classify" in s.by_stage
    assert "question_gen" in s.by_stage
    assert s.by_stage["tf_sfg_classify"].total_logical_calls == 1
    assert s.by_stage["question_gen"].total_logical_calls == 2
    assert s.by_stage["question_gen"].prompt_chars == 1200


def test_by_model_breakdown() -> None:
    tracker = CostTracker()
    tracker.record(_ev(model="gpt-4o-mini", completion_chars=100))
    tracker.record(_ev(model="gpt-4o", completion_chars=300))
    s = tracker.get_aggregate_summary()
    assert "gpt-4o-mini" in s.by_model
    assert "gpt-4o" in s.by_model
    assert s.by_model["gpt-4o"].completion_chars == 300


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def test_token_estimate_uses_chars_per_token() -> None:
    s = CostSummary(prompt_chars=350, completion_chars=70)
    assert s.prompt_tokens_est(chars_per_token=3.5) == pytest.approx(100.0)
    assert s.completion_tokens_est(chars_per_token=3.5) == pytest.approx(20.0)


def test_tracker_chars_per_token_overrideable() -> None:
    tracker = CostTracker(chars_per_token=4.0)
    tracker.record(_ev(prompt_chars=400, completion_chars=40))
    d = tracker.to_dict_aggregate()
    assert d["prompt_tokens_est"] == pytest.approx(100.0)
    assert d["completion_tokens_est"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# to_dict output shape
# ---------------------------------------------------------------------------

def test_to_dict_contains_required_keys() -> None:
    tracker = CostTracker()
    tracker.record(_ev())
    d = tracker.to_dict_aggregate()
    for key in (
        "live_api_calls", "cache_hit_calls", "total_logical_calls",
        "prompt_chars", "completion_chars",
        "prompt_tokens_est", "completion_tokens_est",
        "by_stage", "by_model",
    ):
        assert key in d, f"missing key: {key}"


def test_to_build_summary_dict_structure() -> None:
    tracker = CostTracker()
    tracker.record(_ev(doc_id="d001"))
    tracker.record(_ev(doc_id="d002"))
    result = tracker.to_build_summary_dict()
    assert "d001" in result
    assert "d002" in result
    assert "aggregate" in result
    assert result["aggregate"]["total_logical_calls"] == 2


# ---------------------------------------------------------------------------
# Summation correctness
# ---------------------------------------------------------------------------

def test_chars_accumulate_correctly() -> None:
    tracker = CostTracker()
    for _ in range(5):
        tracker.record(_ev(prompt_chars=100, completion_chars=20))
    s = tracker.get_aggregate_summary()
    assert s.prompt_chars == 500
    assert s.completion_chars == 100


def test_aggregate_equals_sum_of_per_doc() -> None:
    tracker = CostTracker()
    tracker.record(_ev(doc_id="a", prompt_chars=200, completion_chars=40))
    tracker.record(_ev(doc_id="b", prompt_chars=300, completion_chars=60))
    agg = tracker.get_aggregate_summary()
    sa = tracker.get_summary_for_doc("a")
    sb = tracker.get_summary_for_doc("b")
    assert agg.prompt_chars == sa.prompt_chars + sb.prompt_chars
    assert agg.live_api_calls == sa.live_api_calls + sb.live_api_calls


def test_tracked_llm_records_generate_call() -> None:
    class _LLM:
        model = "fake-model"

        def generate(self, prompt, temperature=0.0, max_tokens=512):
            return "completion"

    tracker = CostTracker()
    llm = TrackedLLM(_LLM(), tracker, stage="question_gen", doc_id="doc1")
    assert llm.generate("prompt text") == "completion"
    summary = tracker.to_dict_for_doc("doc1")
    assert summary["live_api_calls"] == 1
    assert summary["prompt_chars"] == len("prompt text")
    assert summary["completion_chars"] == len("completion")
    assert summary["by_stage"]["question_gen"]["live_api_calls"] == 1


def test_tracked_llm_stops_before_exceeding_live_call_cap() -> None:
    class _LLM:
        model = "paid-model"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt, temperature=0.0, max_tokens=512):
            self.calls += 1
            return "completion"

    base = _LLM()
    tracker = CostTracker(max_live_api_calls=1)
    llm = TrackedLLM(base, tracker, stage="question_gen", doc_id="doc1")

    assert llm.generate("first") == "completion"
    with pytest.raises(LiveCallBudgetExceeded):
        llm.generate("second")

    assert base.calls == 1
    assert tracker.to_dict_aggregate()["live_api_calls"] == 1
    assert tracker.budget_control_dict() == {
        "max_live_api_calls": 1,
        "reserved_live_calls": 1,
        "budget_exhausted": True,
    }


def test_cache_hit_wrapper_does_not_consume_live_call_cap() -> None:
    class _LLM:
        model = "cached-model"

        def generate(self, prompt, temperature=0.0, max_tokens=512):
            return "cached"

    tracker = CostTracker(max_live_api_calls=0)
    llm = TrackedLLM(
        _LLM(),
        tracker,
        stage="tf_sfg_classify",
        doc_id="doc1",
        cache_hit=True,
        cache_source="tfsfg_sqlite",
    )

    assert llm.generate("cached prompt") == "cached"
    assert tracker.to_dict_aggregate()["cache_hit_calls"] == 1
    assert tracker.budget_control_dict()["reserved_live_calls"] == 0
