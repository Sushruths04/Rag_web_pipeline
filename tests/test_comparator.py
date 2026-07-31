"""End-to-end test of the Comparator on a tiny in-memory fixture."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.comparison.comparator import Comparator
from rag_gt.comparison.cost_tracker import CostTracker
from rag_gt.comparison.ragas_adapter import RagasConfig
from rag_gt.core import models as core_models


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _gt_record(qid: str, fact_id: str, chunk_id: str):
    return {
        "q_id": qid,
        "question": f"{qid} question?",
        "gold_answer": f"{qid} answer text long enough.",
        "msfs_list": [{"msfs_id": f"{qid}_m1", "fact_ids": [fact_id]}],
        "doc_ids": ["doc"],
        "required_fact_ids": [fact_id],
        "required_facts": [
            {
                "fact_id": fact_id,
                "text": f"Fact text long enough about {qid}.",
                "role": "rule",
                "supporting_spans": [
                    {
                        "doc_id": "doc",
                        "chunk_id": chunk_id,
                        "start_token": 0,
                        "end_token": 4,
                    }
                ],
            }
        ],
        "difficulty_reasoning_depth": 1,
        "difficulty_semantic_distance": "local",
    }


@pytest.fixture
def fixture_paths(tmp_path: Path):
    gt = tmp_path / "test_corpus.jsonl"
    ret = tmp_path / "retrieval.jsonl"
    ans = tmp_path / "answers.jsonl"
    chunks = tmp_path / "chunks.jsonl"

    _write_jsonl(
        gt,
        [
            _gt_record("doc_q001", "F1", "doc_c000000"),
            _gt_record("doc_q002", "F2", "doc_c000001"),
            _gt_record("doc_q003", "F3", "doc_c000002"),
            _gt_record("doc_q004", "F4", "doc_c000003"),
        ],
    )
    _write_jsonl(
        ret,
        [
            {"q_id": "doc_q001", "retrieved_chunk_ids": ["doc_c000000", "doc_c000001"]},
            {"q_id": "doc_q002", "retrieved_chunk_ids": ["doc_c000099"]},
            {"q_id": "doc_q003", "retrieved_chunk_ids": ["doc_c000002"]},
            {"q_id": "doc_q004", "retrieved_chunk_ids": ["doc_c000003", "doc_c000004"]},
        ],
    )
    _write_jsonl(
        ans,
        [
            {"q_id": "doc_q001", "predicted_answer": "doc_q001 answer text long enough.", "abstained": False},
            {"q_id": "doc_q002", "predicted_answer": "", "abstained": True},
            {"q_id": "doc_q003", "predicted_answer": "doc_q003 answer text long enough.", "abstained": False},
            {"q_id": "doc_q004", "predicted_answer": "Different text entirely.", "abstained": False},
        ],
    )
    _write_jsonl(
        chunks,
        [
            {"chunk_id": f"doc_c00000{i}", "doc_id": "doc", "text": f"chunk {i} text"}
            for i in range(5)
        ],
    )
    return gt, ret, ans, chunks


def test_comparator_dry_run(fixture_paths, monkeypatch):
    gt, ret, ans, chunks = fixture_paths

    # Stub NLI so RAG_GT side runs offline. Returns 0.9 entailment for any pair.
    fake_model = SimpleNamespace(
        predict=lambda pairs, apply_softmax=True: [[0.05, 0.9, 0.05] for _ in pairs],
        model=SimpleNamespace(
            config=SimpleNamespace(
                id2label={0: "contradiction", 1: "entailment", 2: "neutral"}
            )
        ),
    )
    monkeypatch.setattr(core_models.MM, "get_nli", lambda: fake_model)
    monkeypatch.setattr(core_models.MM, "load_nli", lambda: None)
    core_models.NLI_LABEL_INDEX.update({"contradiction": 0, "entailment": 1, "neutral": 2})

    from rag_gt.cli import evaluate as eval_mod

    # Vary scores by hash so faithfulness/fact_precision are not constant
    # across questions — otherwise correlation is undefined (constant input).
    def _vary(pairs):
        return [0.3 + 0.6 * ((hash(p[1]) & 0xFFFF) / 0xFFFF) for p in pairs]

    monkeypatch.setattr(eval_mod, "nli_batch", _vary)
    monkeypatch.setattr(eval_mod, "nli_entailment", lambda p, h: 0.7)

    resolver = ChunkResolver.from_cache(chunks)
    cfg = RagasConfig(backend="dry_run", seed=7)
    cost = CostTracker(judge_model="dry_run")

    cmp = Comparator(
        gt_path=gt,
        retrieval_path=ret,
        answers_path=ans,
        resolver=resolver,
        ragas_cfg=cfg,
        cost_tracker=cost,
    )
    report = cmp.run()
    assert len(report.rows) == 4
    assert len(report.correlations) == 3
    # FSR and FSP both vary across the four fixture rows; faithfulness varies
    # because the stub differs per (premise, hypothesis). Correlations should
    # therefore be finite for at least the FSR and FSP pairs.
    fsr_cell = next(c for c in report.correlations if c.rag_gt_metric == "fact_span_recall")
    fsp_cell = next(c for c in report.correlations if c.rag_gt_metric == "fact_span_precision")
    assert fsr_cell.n == 4
    assert fsp_cell.n == 4
    assert fsr_cell.pearson_r == fsr_cell.pearson_r
    assert fsp_cell.pearson_r == fsp_cell.pearson_r

    # No API calls in dry_run.
    assert report.cost.ragas_usd == 0.0
    assert report.cost.ragas_judge_calls == 0
    assert report.cost.rag_gt_seconds >= 0.0
    assert report.cost.ragas_seconds >= 0.0
    assert report.rank_agreement == report.rank_agreement
