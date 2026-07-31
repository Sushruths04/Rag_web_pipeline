"""CLI smoke test for `python -m rag_gt.cli.compare --ragas-llm dry_run --no-plots`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _gt_record(qid: str, fact_id: str, chunk_id: str):
    return {
        "q_id": qid,
        "question": f"{qid} question?",
        "gold_answer": f"{qid} long enough answer text.",
        "msfs_list": [{"msfs_id": f"{qid}_m1", "fact_ids": [fact_id]}],
        "doc_ids": ["doc"],
        "required_fact_ids": [fact_id],
        "required_facts": [
            {
                "fact_id": fact_id,
                "text": f"Fact text long enough about {qid}.",
                "role": "rule",
                "supporting_spans": [
                    {"doc_id": "doc", "chunk_id": chunk_id, "start_token": 0, "end_token": 4}
                ],
            }
        ],
        "difficulty_reasoning_depth": 1,
        "difficulty_semantic_distance": "local",
    }


def test_cli_compare_dry_run(tmp_path: Path, monkeypatch, capsys):
    gt = tmp_path / "mini.jsonl"
    ret = tmp_path / "retrieval.jsonl"
    ans = tmp_path / "answers.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    out_dir = tmp_path / "out"

    _write_jsonl(
        gt,
        [
            _gt_record(f"doc_q{i:03d}", f"F{i}", f"doc_c00000{i}")
            for i in range(4)
        ],
    )
    _write_jsonl(
        ret,
        [
            {"q_id": f"doc_q{i:03d}", "retrieved_chunk_ids": [f"doc_c00000{i}"]}
            for i in range(4)
        ],
    )
    _write_jsonl(
        ans,
        [
            {"q_id": f"doc_q{i:03d}", "predicted_answer": f"doc_q{i:03d} long enough answer text.", "abstained": False}
            for i in range(4)
        ],
    )
    _write_jsonl(
        chunks,
        [
            {"chunk_id": f"doc_c00000{i}", "doc_id": "doc", "text": f"chunk {i} content"}
            for i in range(4)
        ],
    )

    # Stub NLI so the RAG_GT side runs offline.
    from rag_gt.core import models as core_models
    fake = SimpleNamespace(
        predict=lambda pairs, apply_softmax=True: [[0.05, 0.9, 0.05] for _ in pairs],
        model=SimpleNamespace(config=SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})),
    )
    monkeypatch.setattr(core_models.MM, "get_nli", lambda: fake)
    monkeypatch.setattr(core_models.MM, "load_nli", lambda: None)
    core_models.NLI_LABEL_INDEX.update({"contradiction": 0, "entailment": 1, "neutral": 2})
    from rag_gt.cli import evaluate as eval_mod
    monkeypatch.setattr(eval_mod, "nli_batch", lambda pairs: [0.9 for _ in pairs])
    monkeypatch.setattr(eval_mod, "nli_entailment", lambda p, h: 0.9)

    from rag_gt.cli import compare as compare_cli

    argv = [
        "rag-gt-compare",
        "--gt", str(gt),
        "--retrieval", str(ret),
        "--answers", str(ans),
        "--chunks-cache", str(chunks),
        "--ragas-llm", "dry_run",
        "--output-dir", str(out_dir),
        "--no-plots",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc:
        compare_cli.main()
    assert exc.value.code == 0

    assert (out_dir / "comparison.json").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "cost_report.json").exists()
    md = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "RAG_GT vs RAGAS" in md
    assert "Cost & wall-time" in md
