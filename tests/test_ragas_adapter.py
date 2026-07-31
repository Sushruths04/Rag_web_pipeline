"""Tests for RagasAdapter — dry-run and mocked-real paths."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.comparison.ragas_adapter import (
    RAGAS_METRIC_NAMES,
    RagasAdapter,
    RagasConfig,
    RagasResult,
)
from rag_gt.core.types import AnswerLog, Fact, MSFS, QuestionGT, RetrievalLog, Span


def _make_resolver(tmp_path: Path) -> ChunkResolver:
    cache = tmp_path / "chunks.jsonl"
    cache.write_text(
        json.dumps({"chunk_id": "doc_c000000", "doc_id": "doc", "text": "alpha"}) + "\n"
        + json.dumps({"chunk_id": "doc_c000001", "doc_id": "doc", "text": "beta"}) + "\n",
        encoding="utf-8",
    )
    return ChunkResolver.from_cache(cache)


def _make_inputs():
    facts = [
        Fact(
            fact_id="F1",
            text="A long-enough fact about temperature.",
            role="rule",
            supporting_spans=[
                Span(doc_id="doc", chunk_id="doc_c000000", start_token=0, end_token=4)
            ],
        ),
    ]
    q = QuestionGT(
        q_id="doc_q001",
        question="What is the temperature?",
        gold_answer="The temperature is 100C.",
        msfs_list=[MSFS(msfs_id="m1", fact_ids=["F1"])],
        doc_ids=["doc"],
        required_fact_ids=["F1"],
        difficulty_reasoning_depth=1,
        difficulty_semantic_distance="local",
        required_facts=facts,
    )
    q_map = {q.q_id: q}
    ret = {q.q_id: RetrievalLog(q_id=q.q_id, retrieved_chunk_ids=["doc_c000000"])}
    ans = {q.q_id: AnswerLog(q_id=q.q_id, predicted_answer="100C.", abstained=False)}
    return q_map, ret, ans


def test_dry_run_path(tmp_path: Path):
    resolver = _make_resolver(tmp_path)
    cfg = RagasConfig(backend="dry_run", seed=42)
    adapter = RagasAdapter(cfg, resolver)
    q_map, ret, ans = _make_inputs()
    rag_gt = {"doc_q001": {"fact_span_recall": 0.8, "fact_span_precision": 0.6, "faithfulness": 0.7}}

    result = adapter.run(q_map, ret, ans, rag_gt)
    assert isinstance(result, RagasResult)
    assert result.backend_used == "dry_run"
    assert result.judge_calls == 0
    assert result.prompt_tokens == 0
    assert len(result.per_question) == 1
    row = result.per_question[0]
    for m in RAGAS_METRIC_NAMES:
        v = row[m]
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


def test_real_path_falls_back_when_ragas_missing(tmp_path: Path, monkeypatch):
    resolver = _make_resolver(tmp_path)
    cfg = RagasConfig(backend="api", judge_model="gpt-4o-mini", api_base_url="x", api_key="y")
    adapter = RagasAdapter(cfg, resolver)
    monkeypatch.setattr(RagasAdapter, "available", staticmethod(lambda: False))
    q_map, ret, ans = _make_inputs()
    result = adapter.run(q_map, ret, ans, {"doc_q001": {"fact_span_recall": 0.5}})
    # Falls back to dry_run cleanly.
    assert result.backend_used == "dry_run"


def test_real_path_mocked(tmp_path: Path, monkeypatch):
    resolver = _make_resolver(tmp_path)
    cfg = RagasConfig(backend="api", judge_model="gpt-4o-mini",
                      api_base_url="https://x", api_key="key", embed_model="m")
    adapter = RagasAdapter(cfg, resolver)
    q_map, ret, ans = _make_inputs()

    # Pretend ragas is importable (the available() check runs first).
    monkeypatch.setattr(RagasAdapter, "available", staticmethod(lambda: True))

    # Stub the import + judge wiring + evaluate call.
    fake_evaluate = lambda *a, **kw: SimpleNamespace(
        to_pandas=lambda: _DF([{
            "context_precision": 0.7,
            "context_recall": 0.8,
            "faithfulness": 0.9,
            "answer_relevancy": 0.6,
        }])
    )
    monkeypatch.setattr(
        adapter, "_import_ragas_metrics",
        lambda: (fake_evaluate, ["m1", "m2", "m3", "m4"]),
    )
    monkeypatch.setattr(
        adapter, "_build_judge", lambda: ("LLM", "EMB"),
    )
    # Build a fake datasets module so the local import succeeds.
    import types, sys
    fake_ds_module = types.ModuleType("datasets")
    fake_ds_module.Dataset = SimpleNamespace(from_list=lambda data: data)
    monkeypatch.setitem(sys.modules, "datasets", fake_ds_module)

    result = adapter.run(q_map, ret, ans, {"doc_q001": {"fact_span_recall": 0.5}})
    assert result.backend_used == "api"
    assert len(result.per_question) == 1
    assert result.per_question[0]["context_precision"] == pytest.approx(0.7)
    assert result.per_question[0]["faithfulness"] == pytest.approx(0.9)
    # Token counts default to zero in mocked path (no callback fired).
    assert result.prompt_tokens == 0


class _DF:
    """Minimal pandas-DataFrame stand-in for the mocked test."""

    def __init__(self, rows):
        self.rows = rows
        self.columns = list(rows[0].keys()) if rows else []

    def iloc(self):
        return self

    @property
    def loc(self):
        return self

    def __getitem__(self, key):
        return self

    class _Row:
        def __init__(self, d):
            self.d = d
        def __getitem__(self, k):
            return self.d[k]

    @property
    def iloc(self):
        outer = self
        class _IL:
            def __getitem__(self, idx):
                return outer.rows[idx]
        return _IL()
