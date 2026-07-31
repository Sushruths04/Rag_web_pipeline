"""Tests for PASD distractor mining (P3, v16).

All FAISS and NLI calls are stubbed so tests run offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag_gt.core.types import Fact, Span
from rag_gt.generation.distractors import mine_distractors


# ---------- helpers ----------


def _span(doc_id: str = "d1", page: int = 1) -> Span:
    return Span(
        doc_id=doc_id,
        chunk_id="d1_c000000",
        start_token=0,
        end_token=5,
        page_start=page,
    )


def _fact(fid: str, text: str, page: int = 1) -> Fact:
    return Fact(
        fact_id=fid,
        text=text,
        role="definition",
        supporting_spans=[_span(page=page)],
    )


def _index(neighbors: list) -> MagicMock:
    """Return a stub FactIndex whose search() always returns `neighbors`."""
    idx = MagicMock()
    idx.embedding_for.return_value = [0.1, 0.2]
    idx.search.return_value = neighbors
    return idx


# ---------- tests ----------


def test_mine_distractors_normal(monkeypatch):
    """Neighbor on a different page with low NLI to answer is accepted."""
    chain_fact = _fact("F001", "Chain fact text.", page=1)
    distractor_fact = _fact("D001", "Distractor text.", page=2)
    all_facts = {"F001": chain_fact, "D001": distractor_fact}

    monkeypatch.setattr(
        "rag_gt.generation.distractors.nli_entailment", lambda p, h: 0.05
    )

    result = mine_distractors(
        "Gold answer.",
        [chain_fact],
        all_facts,
        _index([("D001", 0.70)]),
        per_fact=1,
        min_cosine=0.55,
    )
    assert len(result) == 1
    assert result[0]["distractor_id"] == "d00001"
    assert result[0]["cosine_to_support"] == pytest.approx(0.70, abs=1e-4)
    assert result[0]["source_fact_id"] == "F001"


def test_mine_distractors_chain_fact_excluded(monkeypatch):
    """A neighbor whose fact_id is in the chain is never collected."""
    chain_fact = _fact("F001", "Chain fact.", page=1)
    all_facts = {"F001": chain_fact}

    monkeypatch.setattr(
        "rag_gt.generation.distractors.nli_entailment", lambda p, h: 0.05
    )

    result = mine_distractors(
        "Gold answer.",
        [chain_fact],
        all_facts,
        _index([("F001", 0.95)]),  # neighbor IS the chain fact
    )
    assert result == []


def test_mine_distractors_same_page_excluded(monkeypatch):
    """Neighbor on the same page as a chain fact is excluded."""
    chain_fact = _fact("F001", "Chain fact.", page=1)
    distractor_fact = _fact("D001", "Distractor.", page=1)  # same page!
    all_facts = {"F001": chain_fact, "D001": distractor_fact}

    monkeypatch.setattr(
        "rag_gt.generation.distractors.nli_entailment", lambda p, h: 0.05
    )

    result = mine_distractors(
        "Gold.",
        [chain_fact],
        all_facts,
        _index([("D001", 0.75)]),
        min_cosine=0.55,
    )
    assert result == []


def test_mine_distractors_high_nli_excluded(monkeypatch):
    """Neighbor whose text entails the gold answer is excluded."""
    chain_fact = _fact("F001", "Chain fact.", page=1)
    distractor_fact = _fact("D001", "Distractor.", page=2)
    all_facts = {"F001": chain_fact, "D001": distractor_fact}

    monkeypatch.setattr(
        "rag_gt.generation.distractors.nli_entailment", lambda p, h: 0.90  # high!
    )

    result = mine_distractors(
        "Gold.",
        [chain_fact],
        all_facts,
        _index([("D001", 0.75)]),
        min_cosine=0.55,
        max_nli_to_answer=0.30,
    )
    assert result == []


def test_mine_distractors_low_cosine_excluded(monkeypatch):
    """Neighbor below min_cosine is excluded."""
    chain_fact = _fact("F001", "Chain fact.", page=1)
    distractor_fact = _fact("D001", "Distractor.", page=2)
    all_facts = {"F001": chain_fact, "D001": distractor_fact}

    monkeypatch.setattr(
        "rag_gt.generation.distractors.nli_entailment", lambda p, h: 0.05
    )

    result = mine_distractors(
        "Gold.",
        [chain_fact],
        all_facts,
        _index([("D001", 0.30)]),  # below 0.55 threshold
        min_cosine=0.55,
    )
    assert result == []


def test_mine_distractors_missing_from_all_facts_excluded(monkeypatch):
    """Neighbor whose fact_id is not in all_facts_by_id is skipped."""
    chain_fact = _fact("F001", "Chain fact.", page=1)
    all_facts = {"F001": chain_fact}  # D001 absent

    monkeypatch.setattr(
        "rag_gt.generation.distractors.nli_entailment", lambda p, h: 0.05
    )

    result = mine_distractors(
        "Gold.",
        [chain_fact],
        all_facts,
        _index([("D001", 0.80)]),  # D001 not in all_facts
        min_cosine=0.55,
    )
    assert result == []


def test_mine_distractors_embedding_key_error_skips(monkeypatch):
    """KeyError from embedding_for skips that chain fact without crashing."""
    chain_fact = _fact("F001", "Chain fact.", page=1)
    all_facts = {"F001": chain_fact}

    idx = MagicMock()
    idx.embedding_for.side_effect = KeyError("F001")

    monkeypatch.setattr(
        "rag_gt.generation.distractors.nli_entailment", lambda p, h: 0.05
    )

    result = mine_distractors("Gold.", [chain_fact], all_facts, idx)
    assert result == []


def test_mine_distractors_deduplication(monkeypatch):
    """The same neighbor is collected at most once across all chain facts."""
    chain_f1 = _fact("F001", "Chain fact A about the topic.", page=1)
    chain_f2 = _fact("F002", "Chain fact B about the topic.", page=2)
    distractor = _fact("D001", "Distractor fact about the topic.", page=3)
    all_facts = {"F001": chain_f1, "F002": chain_f2, "D001": distractor}

    idx = MagicMock()
    idx.embedding_for.return_value = [0.1]
    idx.search.return_value = [("D001", 0.70)]

    monkeypatch.setattr(
        "rag_gt.generation.distractors.nli_entailment", lambda p, h: 0.05
    )

    result = mine_distractors(
        "Gold.",
        [chain_f1, chain_f2],
        all_facts,
        idx,
        per_fact=2,
        min_cosine=0.55,
    )
    distractor_ids = [r["distractor_id"] for r in result]
    assert len(distractor_ids) == len(set(distractor_ids)), "duplicate distractor IDs"


def test_mine_distractors_offset(monkeypatch):
    """distractor_id_offset shifts the numeric counter for cross-row uniqueness."""
    chain_fact = _fact("F001", "Chain fact.", page=1)
    distractor_fact = _fact("D001", "Distractor.", page=2)
    all_facts = {"F001": chain_fact, "D001": distractor_fact}

    monkeypatch.setattr(
        "rag_gt.generation.distractors.nli_entailment", lambda p, h: 0.05
    )

    result = mine_distractors(
        "Gold.",
        [chain_fact],
        all_facts,
        _index([("D001", 0.70)]),
        per_fact=1,
        min_cosine=0.55,
        distractor_id_offset=100,
    )
    assert result[0]["distractor_id"] == "d00101"


def test_mine_distractors_per_fact_cap(monkeypatch):
    """Collects at most `per_fact` distractors per chain fact."""
    chain_fact = _fact("F001", "Chain fact.", page=1)
    distractors = [_fact(f"D{i:03d}", f"Distractor {i}.", page=i + 2) for i in range(5)]
    all_facts = {"F001": chain_fact}
    all_facts.update({d.fact_id: d for d in distractors})

    idx = MagicMock()
    idx.embedding_for.return_value = [0.1]
    idx.search.return_value = [(d.fact_id, 0.70) for d in distractors]

    monkeypatch.setattr(
        "rag_gt.generation.distractors.nli_entailment", lambda p, h: 0.05
    )

    result = mine_distractors(
        "Gold.",
        [chain_fact],
        all_facts,
        idx,
        per_fact=2,  # cap at 2
        min_cosine=0.55,
    )
    assert len(result) <= 2
