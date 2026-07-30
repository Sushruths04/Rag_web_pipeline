import json
from pathlib import Path

from scripts.rescore_run import rescore


def _mk_run(tmp_path: Path) -> Path:
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "chunks_doc.json").write_text(json.dumps([
        {"chunk_id": "c1", "text": "alpha beta gamma delta context"},
        {"chunk_id": "c2", "text": "unrelated noise text here"},
    ]), encoding="utf-8")
    (art / "gt.jsonl").write_text(json.dumps({
        "qa_id": "Q1", "hop_type": "single", "answer": "alpha beta gamma delta",
        "answer_clauses": [{"text": "alpha beta gamma delta"}],
    }) + "\n", encoding="utf-8")
    (art / "eval_results.json").write_text(json.dumps({
        "winner": {"config": "bm25@2"},
        "aggregate": {"recall": 1.0, "precision": 0.5, "f1": 0.6667},
        "rows": [{"qa_id": "Q1", "hop_type": "single", "recall": 1.0,
                  "precision": 0.5, "hit": True, "retrieved": ["c1", "c2"]}],
    }), encoding="utf-8")
    return tmp_path


def test_rescore_recomputes_rw_metrics(tmp_path):
    out = rescore(_mk_run(tmp_path))
    assert out["n_rows"] == 1
    assert out["new"]["recall"] == 1.0
    assert out["new"]["precision_rw"] == 1.0   # evidence at rank 1
    assert out["old"]["precision"] == 0.5
    assert out["config"] == "bm25@2"
