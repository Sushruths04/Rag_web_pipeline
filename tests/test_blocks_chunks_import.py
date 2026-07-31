"""TDD: rag_gt.blocks.chunks_import wraps rag_gt.rag.loader.load_chunks(doc_id)
verbatim -- no reimplementation of chunk loading.

Uses the checked-in din_iso_6507_vickers chunk checkpoint, reachable via
RAG_GT_DATA_ROOT when this worktree doesn't carry the (large, untracked)
pipeline_run chunk files itself. See src/rag_gt/rag/loader.py's _DATA_ROOT.
"""
from __future__ import annotations

import json
import os

import pytest

from rag_gt.rag.loader import load_chunks

pytestmark = pytest.mark.skipif(
    not os.environ.get("RAG_GT_DATA_ROOT")
    and not (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "data" / "test_corpus_allpdf" / "pipeline_run"
    ).exists(),
    reason="chunk data not reachable: set RAG_GT_DATA_ROOT to a checkout that has "
           "data/test_corpus_allpdf/pipeline_run/<doc>_phase2/checkpoints/s2_chunks_full.json",
)

DOC_ID = "din_iso_6507_vickers"


def test_chunks_import_returns_chunks_artifact(tmp_path):
    from rag_gt.blocks.chunks_import import run

    out = run({}, {"doc_id": DOC_ID}, artifacts_dir=tmp_path)

    assert set(out.keys()) == {"chunks"}
    artifact = out["chunks"]
    assert artifact["type"] == "chunks"
    assert isinstance(artifact["ref"], str) and artifact["ref"]
    assert artifact["meta"]["count"] > 0
    assert artifact["meta"]["doc_id"] == DOC_ID


def test_chunks_import_writes_real_chunk_data_matching_loader(tmp_path):
    from rag_gt.blocks.chunks_import import run

    expected = load_chunks(DOC_ID)
    out = run({}, {"doc_id": DOC_ID}, artifacts_dir=tmp_path)

    written = json.loads(open(out["chunks"]["ref"], encoding="utf-8").read())
    assert written == expected
    assert out["chunks"]["meta"]["count"] == len(expected)


def test_chunks_import_unknown_doc_id_raises():
    from rag_gt.blocks.chunks_import import run

    with pytest.raises(ValueError):
        run({}, {"doc_id": "not_a_real_doc"})
