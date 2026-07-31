"""Tests for the L2 (lexical fact-text-in-chunk) layer."""

from __future__ import annotations

import json
from pathlib import Path

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.comparison.retrieval_metrics import evaluate_question
from rag_gt.core.types import Fact, MSFS, QuestionGT, RetrievalLog, Span


def _resolver(tmp_path: Path, records):
    p = tmp_path / "chunks.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return ChunkResolver.from_cache(p)


def _q(text: str, chunk_id: str = "doc_c000000") -> QuestionGT:
    fact = Fact(
        fact_id="doc_F0001",
        text=text,
        role="rule",
        supporting_spans=[
            Span(doc_id="doc", chunk_id=chunk_id, start_token=0, end_token=4)
        ],
    )
    return QuestionGT(
        q_id="doc_q1",
        question="?",
        gold_answer="A.",
        msfs_list=[MSFS(msfs_id="m", fact_ids=["doc_F0001"])],
        doc_ids=["doc"],
        required_fact_ids=["doc_F0001"],
        difficulty_reasoning_depth=1,
        difficulty_semantic_distance="local",
        required_facts=[fact],
    )


def test_l2_strict_hit_when_chunk_and_text_match(tmp_path: Path):
    resolver = _resolver(
        tmp_path,
        [{"chunk_id": "doc_c000000", "doc_id": "doc",
          "text": "Some preamble. The temperature must not exceed 1500C exactly. More text."}],
    )
    q = _q("The temperature must not exceed 1500C exactly.")
    ret = RetrievalLog(q_id="doc_q1", retrieved_chunk_ids=["doc_c000000"])
    r = evaluate_question(q, ret, resolver=resolver, l2_threshold=70.0)
    assert r.fact_recall == 1.0          # L1: chunk retrieved
    assert r.text_recall == 1.0          # L2: text found in chunk
    assert r.strict_recall == 1.0        # L1 ∧ L2
    assert r.facts[0].best_partial_ratio == 100.0


def test_l2_chunk_retrieved_but_text_absent(tmp_path: Path):
    """Chunk_id matches but the chunk text does NOT contain the fact —
    L1 hits, L2 misses, strict_recall = 0."""
    resolver = _resolver(
        tmp_path,
        [{"chunk_id": "doc_c000000", "doc_id": "doc",
          "text": "completely unrelated text about elephants and bicycles"}],
    )
    q = _q("The temperature must not exceed 1500C exactly.")
    ret = RetrievalLog(q_id="doc_q1", retrieved_chunk_ids=["doc_c000000"])
    r = evaluate_question(q, ret, resolver=resolver, l2_threshold=70.0)
    assert r.fact_recall == 1.0
    assert r.text_recall == 0.0
    assert r.strict_recall == 0.0
    assert r.facts[0].best_partial_ratio < 70.0


def test_l2_text_present_in_wrong_chunk(tmp_path: Path):
    """The fact text appears in a chunk the system retrieved, but it's a
    different chunk_id than the GT span. L1 misses, L2 hits, strict = 0."""
    resolver = _resolver(
        tmp_path,
        [
            {"chunk_id": "doc_c000000", "doc_id": "doc", "text": "expected GT chunk text"},
            {"chunk_id": "doc_c000099", "doc_id": "doc",
             "text": "Some context. The temperature must not exceed 1500C exactly. More."},
        ],
    )
    q = _q("The temperature must not exceed 1500C exactly.", chunk_id="doc_c000000")
    ret = RetrievalLog(q_id="doc_q1", retrieved_chunk_ids=["doc_c000099"])
    r = evaluate_question(q, ret, resolver=resolver, l2_threshold=70.0)
    assert r.fact_recall == 0.0          # L1: required chunk_id missing
    assert r.text_recall == 1.0          # L2: text appears in retrieved
    assert r.strict_recall == 0.0        # L1 ∧ L2 fails


def test_resolver_none_skips_l2():
    q = _q("any reasonably long fact text.")
    ret = RetrievalLog(q_id="doc_q1", retrieved_chunk_ids=["doc_c000000"])
    r = evaluate_question(q, ret, resolver=None)
    assert r.text_recall is None
    assert r.strict_recall is None
    for h in r.facts:
        assert h.text_present is None
        assert h.strict_hit is None
