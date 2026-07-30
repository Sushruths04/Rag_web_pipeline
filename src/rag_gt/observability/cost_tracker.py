"""CostTracker for V16.2 — real per-stage LLM call instrumentation.

Records every LLM.generate / LLM.classify call with stage, doc, model,
prompt/completion char counts, and cache state. Aggregates per-doc and globally.

Paper integrity rule: the cost-efficiency claim uses live_api_calls only.
cache_hit_calls are reported in a separate column, never merged into savings.

Usage::

    tracker = CostTracker()

    # Manual record (use where the LLM is called directly):
    tracker.record(CostEvent(
        stage="tf_sfg_classify",
        doc_id="d001",
        model="gpt-4o-mini",
        call_count=1,
        prompt_chars=len(prompt),
        completion_chars=len(raw_response),
        cache_hit=False,
        cache_source="",
    ))

    summary = tracker.get_summary_for_doc("d001")
    all_data = tracker.to_build_summary_dict()  # → build_summary.json:cost_tracker

See docs/V16_2_FILTER_PLAN_20260519.md §4.7.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from rag_gt.core.llm import APIError

# Default chars-per-token estimate for English prose. Applied at summary time
# only; raw char counts are always stored so the estimate can be recomputed.
_DEFAULT_CHARS_PER_TOKEN: float = 3.5


class LiveCallBudgetExceeded(APIError):
    """Raised before a live LLM call when a configured paid-run cap is reached."""


@dataclass
class CostEvent:
    stage: str           # "tf_sfg_classify" | "question_gen" | "qa_nli" | ...
    doc_id: str
    model: str           # e.g. "gpt-4o-mini"
    call_count: int      # 1 per logical LLM call
    prompt_chars: int
    completion_chars: int
    cache_hit: bool
    cache_source: str    # "tfsfg_sqlite" | "in_memory" | "" for live calls
    timestamp: float = field(default_factory=time.time)


@dataclass
class CostSummary:
    live_api_calls: int = 0
    cache_hit_calls: int = 0
    total_logical_calls: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    by_stage: dict[str, "CostSummary"] = field(default_factory=dict)
    by_model: dict[str, "CostSummary"] = field(default_factory=dict)

    def prompt_tokens_est(self, chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN) -> float:
        return self.prompt_chars / chars_per_token

    def completion_tokens_est(self, chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN) -> float:
        return self.completion_chars / chars_per_token

    def to_dict(self, chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN) -> dict:
        return {
            "live_api_calls": self.live_api_calls,
            "cache_hit_calls": self.cache_hit_calls,
            "total_logical_calls": self.total_logical_calls,
            "prompt_chars": self.prompt_chars,
            "completion_chars": self.completion_chars,
            "prompt_tokens_est": round(self.prompt_tokens_est(chars_per_token), 1),
            "completion_tokens_est": round(self.completion_tokens_est(chars_per_token), 1),
            "by_stage": {k: v.to_dict(chars_per_token) for k, v in self.by_stage.items()},
            "by_model": {k: v.to_dict(chars_per_token) for k, v in self.by_model.items()},
        }


def _merge_event_into(dst: CostSummary, ev: CostEvent) -> None:
    dst.total_logical_calls += ev.call_count
    dst.prompt_chars += ev.prompt_chars
    dst.completion_chars += ev.completion_chars
    if ev.cache_hit:
        dst.cache_hit_calls += ev.call_count
    else:
        dst.live_api_calls += ev.call_count


class CostTracker:
    """Thread-safe per-stage / per-doc / aggregate LLM cost accumulator.

    Live-call reservation and event recording share a lock so a concurrent paid
    run cannot exceed its configured uncached-call cap.
    """

    def __init__(
        self,
        chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN,
        max_live_api_calls: int | None = None,
    ) -> None:
        self.chars_per_token = chars_per_token
        self.max_live_api_calls = (
            max(0, int(max_live_api_calls))
            if max_live_api_calls is not None
            else None
        )
        self._events: list[CostEvent] = []
        self._reserved_live_calls = 0
        self._budget_exhausted = False
        self._lock = Lock()

    def record(self, event: CostEvent) -> None:
        """Append one CostEvent to the tracker."""
        with self._lock:
            self._events.append(event)

    def reserve_live_call(self) -> None:
        """Reserve one provider request before it can incur paid usage."""
        with self._lock:
            if (
                self.max_live_api_calls is not None
                and self._reserved_live_calls >= self.max_live_api_calls
            ):
                self._budget_exhausted = True
                raise LiveCallBudgetExceeded(
                    "Live API call cap reached "
                    f"({self.max_live_api_calls}); stopping before another paid request."
                )
            self._reserved_live_calls += 1

    def budget_control_dict(self) -> dict:
        with self._lock:
            return {
                "max_live_api_calls": self.max_live_api_calls,
                "reserved_live_calls": self._reserved_live_calls,
                "budget_exhausted": self._budget_exhausted,
            }

    def get_summary_for_doc(self, doc_id: str) -> CostSummary:
        """Aggregate all events for a single document."""
        return self._aggregate([e for e in self._events if e.doc_id == doc_id])

    def get_aggregate_summary(self) -> CostSummary:
        """Aggregate all events across all documents."""
        return self._aggregate(self._events)

    def to_dict_for_doc(self, doc_id: str) -> dict:
        return self.get_summary_for_doc(doc_id).to_dict(self.chars_per_token)

    def to_dict_aggregate(self) -> dict:
        return self.get_aggregate_summary().to_dict(self.chars_per_token)

    def to_build_summary_dict(self) -> dict:
        """Return the build_summary.json:cost_tracker block.

        Structure::
            {
                "<doc_id>": { live_api_calls, cache_hit_calls, ... },
                "aggregate": { ... }
            }
        """
        doc_ids = sorted({e.doc_id for e in self._events})
        result: dict = {did: self.to_dict_for_doc(did) for did in doc_ids}
        result["aggregate"] = self.to_dict_aggregate()
        result["aggregate"]["budget_control"] = self.budget_control_dict()
        return result

    @staticmethod
    def _aggregate(events: list[CostEvent]) -> CostSummary:
        summary = CostSummary()
        stage_map: dict[str, CostSummary] = {}
        model_map: dict[str, CostSummary] = {}

        for ev in events:
            _merge_event_into(summary, ev)

            if ev.stage not in stage_map:
                stage_map[ev.stage] = CostSummary()
            _merge_event_into(stage_map[ev.stage], ev)

            if ev.model not in model_map:
                model_map[ev.model] = CostSummary()
            _merge_event_into(model_map[ev.model], ev)

        summary.by_stage = stage_map
        summary.by_model = model_map
        return summary


class TrackedLLM:
    """Stage-specific wrapper that records every generate() call."""

    def __init__(
        self,
        llm: Any,
        tracker: CostTracker | None,
        *,
        stage: str,
        doc_id: str,
        cache_hit: bool = False,
        cache_source: str = "",
    ) -> None:
        self._llm = llm
        self._tracker = tracker
        self._stage = stage
        self._doc_id = doc_id
        self._cache_hit = cache_hit
        self._cache_source = cache_source
        self.model = getattr(llm, "model", type(llm).__name__)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        if self._tracker is not None and not self._cache_hit:
            self._tracker.reserve_live_call()
        raw = self._llm.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        if self._tracker is not None:
            self._tracker.record(
                CostEvent(
                    stage=self._stage,
                    doc_id=self._doc_id,
                    model=str(self.model),
                    call_count=1,
                    prompt_chars=len(prompt or ""),
                    completion_chars=len(raw or ""),
                    cache_hit=self._cache_hit,
                    cache_source=self._cache_source,
                )
            )
        return raw

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)
