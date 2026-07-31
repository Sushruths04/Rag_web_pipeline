"""Tests for FactStore: harvest from inline GT, save/load roundtrip, coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_gt.comparison.fact_store import FactStore, required_fact_ids_from_gt


def _gt_record(qid: str, fact_id: str, chunk_id: str, with_inline: bool):
    base = {
        "q_id": qid,
        "question": f"{qid}?",
        "gold_answer": f"{qid} answer text long enough.",
        "msfs_list": [{"msfs_id": f"{qid}_m1", "fact_ids": [fact_id]}],
        "doc_ids": ["doc"],
        "required_fact_ids": [fact_id],
        "difficulty_reasoning_depth": 1,
        "difficulty_semantic_distance": "local",
    }
    if with_inline:
        base["required_facts"] = [
            {
                "fact_id": fact_id,
                "text": f"Fact text long enough about {fact_id}.",
                "role": "rule",
                "supporting_spans": [
                    {"doc_id": "doc", "chunk_id": chunk_id, "start_token": 0, "end_token": 4}
                ],
            }
        ]
    return base


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_from_gt_inline_harvests_embedded_facts(tmp_path: Path):
    gt = tmp_path / "new.jsonl"
    _write_jsonl(
        gt,
        [
            _gt_record("doc_q1", "doc_F0001", "doc_c000000", with_inline=True),
            _gt_record("doc_q2", "doc_F0002", "doc_c000001", with_inline=True),
        ],
    )
    store = FactStore.from_gt_inline(gt)
    assert len(store) == 2
    f = store.get("doc_F0001")
    assert f.fact_id == "doc_F0001"
    assert f.text.startswith("Fact text long enough")
    assert f.role == "rule"
    assert f.supporting_spans[0].chunk_id == "doc_c000000"


def test_from_gt_inline_skips_old_format(tmp_path: Path):
    gt = tmp_path / "old.jsonl"
    _write_jsonl(
        gt,
        [
            _gt_record("doc_q1", "doc_F0001", "doc_c000000", with_inline=False),
        ],
    )
    store = FactStore.from_gt_inline(gt)
    # Old-format rows expose only `required_fact_ids`; nothing to inline-harvest.
    assert len(store) == 0


def test_save_load_roundtrip(tmp_path: Path):
    gt = tmp_path / "new.jsonl"
    _write_jsonl(gt, [_gt_record("doc_q1", "doc_F0001", "doc_c000000", with_inline=True)])
    store = FactStore.from_gt_inline(gt)
    out = tmp_path / "facts.jsonl"
    store.save(out)
    re = FactStore.from_cache(out)
    assert len(re) == 1
    assert re.get("doc_F0001").text.startswith("Fact text long enough")


def test_coverage(tmp_path: Path):
    gt = tmp_path / "new.jsonl"
    _write_jsonl(gt, [_gt_record("doc_q1", "doc_F0001", "doc_c000000", with_inline=True)])
    store = FactStore.from_gt_inline(gt)
    cov = store.coverage(["doc_F0001", "doc_F0002"])
    assert cov.requested == 2
    assert cov.found == 1
    assert cov.missing == ["doc_F0002"]
    assert not cov.is_complete


def test_required_fact_ids_from_gt(tmp_path: Path):
    gt = tmp_path / "g.jsonl"
    _write_jsonl(
        gt,
        [
            _gt_record("doc_q1", "doc_F0001", "doc_c000000", with_inline=False),
            _gt_record("doc_q2", "doc_F0002", "doc_c000001", with_inline=True),
        ],
    )
    ids = required_fact_ids_from_gt(gt)
    assert ids == {"doc_F0001", "doc_F0002"}


def test_get_unknown_raises(tmp_path: Path):
    store = FactStore()
    with pytest.raises(KeyError):
        store.get("missing")
    assert store.get_optional("missing") is None
