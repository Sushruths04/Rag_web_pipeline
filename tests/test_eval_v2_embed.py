"""Local-embedding eval mode (T9): --embed-source local|api, default local.

F9 root cause: DenseRetriever's default embedder resolves through
``MM.get_embedding()``, which silently calls a paid API embedder when
``ACTIVE_EMBEDDING`` is set in the environment — so "no-API eval" was not
fully true for hybrid strategy. These tests verify the flag selects the
right embedder path via fakes/monkeypatches; no real model is loaded and no
full eval is run.
"""
import json

import numpy as np
import pytest

import rag_gt.rag.eval_v2 as eval_v2_mod
from rag_gt.rag.retriever import DenseRetriever, HybridRetriever, build_retriever


class _FakeSTModel:
    def __init__(self, name: str = "") -> None:
        self.name = name

    def encode(self, texts, **_kwargs):
        return np.zeros((len(texts), 2), dtype=np.float32)


# ---------------------------------------------------------------------------
# retriever.py plumbing
# ---------------------------------------------------------------------------

def test_dense_retriever_local_embed_source_never_touches_mm(monkeypatch):
    calls = {}

    def fake_st(name):
        calls["model_name"] = name
        return _FakeSTModel(name)

    def fail_mm():
        raise AssertionError("MM.get_embedding must not be called for embed_source='local'")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", fake_st)
    monkeypatch.setattr("rag_gt.core.models.MM.get_embedding", fail_mm)

    chunks = [{"chunk_id": "a", "text": "alpha"}]
    DenseRetriever(chunks, embed_source="local")

    assert "model_name" in calls


def test_dense_retriever_api_embed_source_uses_mm_get_embedding(monkeypatch):
    calls = {"count": 0}

    def fake_get_embedding():
        calls["count"] += 1
        return _FakeSTModel("api-model")

    def fail_st(name):
        raise AssertionError("SentenceTransformer must not load for embed_source='api'")

    monkeypatch.setattr("rag_gt.core.models.MM.get_embedding", fake_get_embedding)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", fail_st)

    chunks = [{"chunk_id": "a", "text": "alpha"}]
    DenseRetriever(chunks, embed_source="api")

    assert calls["count"] == 1


def test_dense_retriever_default_embed_source_preserves_api_backward_compat(monkeypatch):
    """Existing callers (pipeline.py, multigold_report.py) that don't pass
    embed_source at all must keep hitting MM.get_embedding(), unchanged."""
    calls = {"count": 0}

    def fake_get_embedding():
        calls["count"] += 1
        return _FakeSTModel("api-model")

    monkeypatch.setattr("rag_gt.core.models.MM.get_embedding", fake_get_embedding)

    chunks = [{"chunk_id": "a", "text": "alpha"}]
    DenseRetriever(chunks)

    assert calls["count"] == 1


def test_explicit_model_or_embedder_bypasses_embed_source(monkeypatch):
    """embed_source is only consulted when neither `model` nor `embedder`
    is given explicitly — explicit choices always win."""

    class _Embedder:
        def encode(self, texts, **_kwargs):
            return np.zeros((len(texts), 2), dtype=np.float32)

    def fail_mm():
        raise AssertionError("must not be called: explicit embedder given")

    monkeypatch.setattr("rag_gt.core.models.MM.get_embedding", fail_mm)

    chunks = [{"chunk_id": "a", "text": "alpha"}]
    DenseRetriever(chunks, embedder=_Embedder(), embed_source="api")


def test_hybrid_retriever_threads_embed_source_to_dense(monkeypatch):
    calls = {}

    def fake_st(name):
        calls["model_name"] = name
        return _FakeSTModel(name)

    def fail_mm():
        raise AssertionError("MM.get_embedding must not be called for embed_source='local'")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", fake_st)
    monkeypatch.setattr("rag_gt.core.models.MM.get_embedding", fail_mm)

    chunks = [{"chunk_id": "a", "text": "alpha rule"}]
    HybridRetriever(chunks, embed_source="local")

    assert "model_name" in calls


def test_build_retriever_threads_embed_source_for_hybrid_strategy(monkeypatch):
    calls = {}

    def fake_st(name):
        calls["model_name"] = name
        return _FakeSTModel(name)

    def fail_mm():
        raise AssertionError("MM.get_embedding must not be called for embed_source='local'")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", fake_st)
    monkeypatch.setattr("rag_gt.core.models.MM.get_embedding", fail_mm)

    chunks = [{"chunk_id": "a", "text": "alpha rule"}]
    build_retriever(chunks, strategy="hybrid", embed_source="local")

    assert "model_name" in calls


# ---------------------------------------------------------------------------
# eval_v2.py CLI + threading into evaluate_v2 -> build_retriever
# ---------------------------------------------------------------------------

def test_cli_default_embed_source_is_local(monkeypatch, tmp_path):
    captured = {}

    def fake_evaluate_v2(v2_dir, facts_dir, top_k=10, strategies=eval_v2_mod.STRATEGIES,
                          embed_source="local"):
        captured["embed_source"] = embed_source
        return {"top_k": top_k, "strategies": {}}

    monkeypatch.setattr(eval_v2_mod, "evaluate_v2", fake_evaluate_v2)

    v2_dir = tmp_path / "v2"
    v2_dir.mkdir()
    facts_dir = tmp_path / "facts"
    facts_dir.mkdir()
    out = tmp_path / "out" / "result"

    eval_v2_mod.main(["--v2-dir", str(v2_dir), "--facts-dir", str(facts_dir), "--out", str(out)])

    assert captured["embed_source"] == "local"


def test_cli_explicit_api_embed_source_threaded(monkeypatch, tmp_path):
    captured = {}

    def fake_evaluate_v2(v2_dir, facts_dir, top_k=10, strategies=eval_v2_mod.STRATEGIES,
                          embed_source="local"):
        captured["embed_source"] = embed_source
        return {"top_k": top_k, "strategies": {}}

    monkeypatch.setattr(eval_v2_mod, "evaluate_v2", fake_evaluate_v2)

    v2_dir = tmp_path / "v2"
    v2_dir.mkdir()
    facts_dir = tmp_path / "facts"
    facts_dir.mkdir()
    out = tmp_path / "out" / "result"

    eval_v2_mod.main([
        "--v2-dir", str(v2_dir), "--facts-dir", str(facts_dir), "--out", str(out),
        "--embed-source", "api",
    ])

    assert captured["embed_source"] == "api"


def test_cli_rejects_invalid_embed_source(tmp_path):
    v2_dir = tmp_path / "v2"
    v2_dir.mkdir()
    facts_dir = tmp_path / "facts"
    facts_dir.mkdir()
    out = tmp_path / "out" / "result"

    with pytest.raises(SystemExit):
        eval_v2_mod.main([
            "--v2-dir", str(v2_dir), "--facts-dir", str(facts_dir), "--out", str(out),
            "--embed-source", "bogus",
        ])


def test_evaluate_v2_threads_embed_source_into_build_retriever(monkeypatch, tmp_path):
    """The real threading path: evaluate_v2 -> build_retriever(embed_source=...)
    for every strategy, using stubbed chunks/pairs/retriever so no real model
    or data-root access happens."""
    v2_dir = tmp_path / "v2"
    v2_dir.mkdir()
    facts_dir = tmp_path / "facts"
    facts_dir.mkdir()

    (facts_dir / "facts_docA_full.json").write_text(
        json.dumps([{"fact_id": "docA_F1", "text": "fact one text"}]), encoding="utf-8")

    pair = {
        "qa_id": "V2Q00001", "doc": "docA_full", "evidence_strategy": "neighbor_pair_v2",
        "question": "q?", "gold_fact_ids": ["docA_F1"], "gold_pages": [1],
        "necessity": {"necessity_score": 1.0},
    }
    (v2_dir / "docA.json").write_text(
        json.dumps({"pairs": [pair], "stats": {"doc": "docA_full"}}), encoding="utf-8")

    captured_embed_sources = []

    class _StubRetriever:
        def retrieve(self, query, top_k=10):
            return []

    def fake_build_retriever(chunks, strategy="bm25", embed_source="api", **kwargs):
        captured_embed_sources.append(embed_source)
        return _StubRetriever()

    monkeypatch.setattr(eval_v2_mod, "build_retriever", fake_build_retriever)
    monkeypatch.setattr(eval_v2_mod, "load_chunks",
                         lambda doc_id: [{"chunk_id": "c1", "text": "fact one text"}])
    monkeypatch.setattr(eval_v2_mod, "load_gt_pairs", lambda doc_id: [])

    eval_v2_mod.evaluate_v2(v2_dir, facts_dir, top_k=5, strategies=("bm25", "hybrid"),
                            embed_source="local")

    assert captured_embed_sources == ["local", "local"]
