"""Tests for L3 semantic recall + RAGAS-style rank-weighted precision."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.comparison.retrieval_metrics import (
    _bootstrap_ci,
    _rank_weighted_precision,
    evaluate_question,
)
from rag_gt.core.types import Fact, MSFS, QuestionGT, RetrievalLog, Span


# ---------- rank-weighted precision ----------


def test_rank_weighted_precision_all_relevant():
    # All hits at top → MAP = 1.0
    assert _rank_weighted_precision([1, 1, 1]) == pytest.approx(1.0)


def test_rank_weighted_precision_none_relevant():
    assert _rank_weighted_precision([0, 0, 0]) == 0.0


def test_rank_weighted_precision_known_value():
    # rel = [1, 0, 1, 0] → P@1=1.0 (v=1), P@3=2/3 (v=1), sum/2 = 5/6
    assert _rank_weighted_precision([1, 0, 1, 0]) == pytest.approx(5 / 6)


def test_rank_weighted_precision_empty():
    assert _rank_weighted_precision([]) == 0.0


# ---------- bootstrap CI ----------


def test_bootstrap_ci_constant_vector():
    lo, hi = _bootstrap_ci([0.5] * 20, n_resamples=200, seed=1)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_bootstrap_ci_brackets_mean():
    lo, hi = _bootstrap_ci([0.0, 1.0] * 20, n_resamples=500, seed=1)
    assert lo < 0.5 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_bootstrap_ci_handles_singleton():
    lo, hi = _bootstrap_ci([0.7], n_resamples=10, seed=1)
    assert lo == 0.7 and hi == 0.7


def test_bootstrap_ci_handles_empty():
    lo, hi = _bootstrap_ci([], n_resamples=10, seed=1)
    assert math.isnan(lo) and math.isnan(hi)


# ---------- L3 semantic ----------


def _fake_embedder(direction: str = "match"):
    """Returns an embedder whose chunk vector points in a controllable
    direction relative to fact vectors. ``match``: cosine ≈ 0.95;
    ``orthogonal``: cosine ≈ 0; ``opposite``: cosine ≈ -0.9."""

    def _vec(text: str) -> np.ndarray:
        # Use a tiny deterministic pseudo-embedding: hash → 16-d unit vector.
        h = abs(hash(text)) % (2 ** 32)
        rng = np.random.default_rng(h)
        v = rng.standard_normal(16)
        return v / np.linalg.norm(v)

    class _E:
        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            base = np.stack([_vec(t) for t in texts])
            if direction == "match":
                # Re-emit fact vectors aligned with chunk vectors → high cosine.
                # Trick: encode any text to the same direction via deterministic seed.
                shared = _vec("__shared_anchor__")
                return np.tile(shared, (len(texts), 1))
            if direction == "orthogonal":
                return base
            if direction == "opposite":
                shared = _vec("__shared_anchor__")
                opp = -shared
                return np.tile(opp, (len(texts), 1))
            return base

    return _E()


def _resolver_with(tmp_path: Path, chunk_id: str, text: str) -> ChunkResolver:
    cache = tmp_path / "chunks.jsonl"
    cache.write_text(
        json.dumps({"chunk_id": chunk_id, "doc_id": "doc", "text": text}) + "\n",
        encoding="utf-8",
    )
    return ChunkResolver.from_cache(cache)


def _q(fact_text: str, chunk_id: str = "doc_c000000") -> QuestionGT:
    fact = Fact(
        fact_id="doc_F0001",
        text=fact_text,
        role="rule",
        supporting_spans=[Span(doc_id="doc", chunk_id=chunk_id, start_token=0, end_token=4)],
    )
    return QuestionGT(
        q_id="doc_q1",
        question="?",
        gold_answer="A.",
        msfs_list=[MSFS(msfs_id="m", fact_ids=["doc_F0001"])],
        doc_ids=["doc"],
        required_fact_ids=["doc_F0001"],
        difficulty_reasoning_depth=2,
        difficulty_semantic_distance="local",
        required_facts=[fact],
    )


def test_l3_match(tmp_path: Path):
    resolver = _resolver_with(tmp_path, "doc_c000000", "any text")
    q = _q("the temperature must not exceed 1500C")
    ret = RetrievalLog(q_id="doc_q1", retrieved_chunk_ids=["doc_c000000"])
    r = evaluate_question(
        q, ret, resolver=resolver,
        embedder=_fake_embedder("match"), l3_threshold=0.75,
    )
    assert r.text_recall_l3 == 1.0
    assert r.strict_recall_l13 == 1.0
    assert r.facts[0].best_cosine is not None
    assert r.facts[0].best_cosine >= 0.99


def test_l3_orthogonal_misses(tmp_path: Path):
    resolver = _resolver_with(tmp_path, "doc_c000000", "any text")
    q = _q("the temperature must not exceed 1500C")
    ret = RetrievalLog(q_id="doc_q1", retrieved_chunk_ids=["doc_c000000"])
    r = evaluate_question(
        q, ret, resolver=resolver,
        embedder=_fake_embedder("orthogonal"), l3_threshold=0.75,
    )
    assert r.text_recall_l3 == 0.0
    assert r.strict_recall_l13 == 0.0


def test_l3_skipped_without_embedder(tmp_path: Path):
    resolver = _resolver_with(tmp_path, "doc_c000000", "any text")
    q = _q("the temperature must not exceed 1500C")
    ret = RetrievalLog(q_id="doc_q1", retrieved_chunk_ids=["doc_c000000"])
    r = evaluate_question(q, ret, resolver=resolver, embedder=None)
    assert r.text_recall_l3 is None
    assert r.strict_recall_l13 is None


def test_difficulty_passed_through():
    q = _q("any reasonably long fact text.")
    ret = RetrievalLog(q_id="doc_q1", retrieved_chunk_ids=["doc_c000000"])
    r = evaluate_question(q, ret)
    assert r.reasoning_depth == 2
    assert r.semantic_distance == "local"


def test_fact_precision_rw_inside_evaluate_question():
    # Two retrieved chunks: one supporting (rank 1), one not (rank 2).
    # rel_at_k = [1, 0]; P@1·v1 = 1.0; sum/1 = 1.0
    fact = Fact(
        fact_id="F1", text="something long enough to be a fact",
        role="rule",
        supporting_spans=[Span(doc_id="doc", chunk_id="doc_c000000", start_token=0, end_token=4)],
    )
    q = QuestionGT(
        q_id="doc_q1", question="?", gold_answer="A.",
        msfs_list=[MSFS(msfs_id="m", fact_ids=["F1"])],
        doc_ids=["doc"], required_fact_ids=["F1"],
        difficulty_reasoning_depth=1, difficulty_semantic_distance="local",
        required_facts=[fact],
    )
    ret = RetrievalLog(q_id="doc_q1", retrieved_chunk_ids=["doc_c000000", "doc_c000099"])
    r = evaluate_question(q, ret)
    assert r.fact_precision_rw == pytest.approx(1.0)
