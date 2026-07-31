"""Tests for the chunks-cache resolver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.core.types import Fact, MSFS, QuestionGT, Span


def _write_cache(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _q(qid: str, chunk_ids):
    facts = [
        Fact(
            fact_id=f"F_{i}",
            text="Some fact text long enough to pass the validator.",
            role="rule",
            supporting_spans=[
                Span(doc_id="doc", chunk_id=cid, start_token=0, end_token=4)
            ],
        )
        for i, cid in enumerate(chunk_ids)
    ]
    return QuestionGT(
        q_id=qid,
        question="Q?",
        gold_answer="A.",
        msfs_list=[MSFS(msfs_id=f"{qid}_msfs1", fact_ids=[f.fact_id for f in facts])],
        doc_ids=["doc"],
        required_fact_ids=[f.fact_id for f in facts],
        difficulty_reasoning_depth=1,
        difficulty_semantic_distance="local",
        required_facts=facts,
    )


def test_resolver_roundtrip(tmp_path: Path):
    cache = tmp_path / "chunks.jsonl"
    _write_cache(
        cache,
        [
            {"chunk_id": "doc_c000000", "doc_id": "doc", "text": "Hello world."},
            {"chunk_id": "doc_c000001", "doc_id": "doc", "text": "Second chunk."},
        ],
    )
    r = ChunkResolver.from_cache(cache)
    assert "doc_c000000" in r
    assert r.get("doc_c000000") == "Hello world."
    assert r.get_many(["doc_c000001", "missing", "doc_c000000"]) == [
        "Second chunk.",
        "Hello world.",
    ]
    with pytest.raises(KeyError):
        r.get("nope")
    assert len(r) == 2


def test_resolver_missing_cache(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ChunkResolver.from_cache(tmp_path / "absent.jsonl")


def test_verify_coverage(tmp_path: Path):
    cache = tmp_path / "chunks.jsonl"
    _write_cache(
        cache,
        [
            {"chunk_id": "doc_c000000", "doc_id": "doc", "text": "X"},
        ],
    )
    r = ChunkResolver.from_cache(cache)
    qs = [_q("q1", ["doc_c000000", "doc_c000001"])]
    cov = r.verify_coverage(qs)
    assert cov.requested == 2
    assert cov.found == 1
    assert cov.missing == ["doc_c000001"]
    assert not cov.is_complete
