"""TDD: rag_gt.blocks.chunker wraps rag_gt.rag.loader.rechunk(chunks, strategy)
verbatim (05_BLOCK_CATALOG.md §3.6). Reads the upstream chunks artifact by
its ref path -- exercises the "read prior artifact, call engine fn, write new
artifact" pattern all downstream FREE-spine blocks share.
"""
from __future__ import annotations

import json

import pytest

from rag_gt.rag.loader import rechunk

SAMPLE_CHUNKS = [
    {"chunk_id": "d_c0", "doc_id": "d", "text": "First sentence here. Second sentence follows nicely.",
     "page_start": 1, "page_end": 1},
    {"chunk_id": "d_c1", "doc_id": "d", "text": "Third sentence over here. Fourth one too now.",
     "page_start": 2, "page_end": 2},
]


def _chunks_artifact(tmp_path):
    from rag_gt.blocks._common import artifact, write_json_artifact

    ref = write_json_artifact(tmp_path, "chunks_import", SAMPLE_CHUNKS)
    return {"chunks": artifact("chunks", str(ref), {"count": len(SAMPLE_CHUNKS), "doc_id": "d"})}


def test_chunker_original_strategy_passes_through(tmp_path):
    from rag_gt.blocks.chunker import run

    inputs = _chunks_artifact(tmp_path)
    out = run(inputs, {"strategy": "original"}, artifacts_dir=tmp_path)

    assert set(out.keys()) == {"chunks"}
    artifact = out["chunks"]
    assert artifact["meta"]["count"] == len(SAMPLE_CHUNKS)
    assert artifact["meta"]["strategy"] == "original"
    written = json.loads(open(artifact["ref"], encoding="utf-8").read())
    assert written == SAMPLE_CHUNKS


def test_chunker_matches_rechunk_directly_for_sentence_strategy(tmp_path):
    from rag_gt.blocks.chunker import run

    inputs = _chunks_artifact(tmp_path)
    out = run(inputs, {"strategy": "sentence"}, artifacts_dir=tmp_path)

    expected = rechunk(SAMPLE_CHUNKS, "sentence")
    written = json.loads(open(out["chunks"]["ref"], encoding="utf-8").read())
    assert written == expected
    assert out["chunks"]["meta"]["count"] == len(expected)


def test_chunker_unknown_strategy_raises(tmp_path):
    from rag_gt.blocks.chunker import run

    inputs = _chunks_artifact(tmp_path)
    with pytest.raises(ValueError):
        run(inputs, {"strategy": "bogus"}, artifacts_dir=tmp_path)
