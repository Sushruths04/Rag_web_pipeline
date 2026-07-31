"""TDD: rag_gt.blocks.index_builder wraps rag_gt.rag.retriever.build_retriever
verbatim (05_BLOCK_CATALOG.md §3.27).

A BM25/Dense/Hybrid retriever object is not JSON-serializable and the engine
has no save/load for it, so the index artifact's ref is a small manifest
{"chunks_ref": <upstream chunks artifact path>, "strategy":..., "embed_source":...}
rather than a serialized index. rag_gt.blocks.evaluator (the only consumer)
rebuilds the retriever from that manifest by calling build_retriever again on
the referenced chunks -- still zero reimplementation of retriever internals.
"""
from __future__ import annotations

import json

from rag_gt.rag.retriever import BM25Retriever, build_retriever


def _chunks_artifact(tmp_path):
    from rag_gt.blocks._common import artifact, write_json_artifact

    chunks = [
        {"chunk_id": "c1", "doc_id": "d", "text": "alpha beta gamma", "page_start": 1, "page_end": 1},
        {"chunk_id": "c2", "doc_id": "d", "text": "delta epsilon zeta", "page_start": 2, "page_end": 2},
    ]
    ref = write_json_artifact(tmp_path, "chunks_import", chunks)
    return {"chunks": artifact("chunks", str(ref), {"count": len(chunks), "doc_id": "d"})}, chunks


def test_index_builder_returns_index_artifact(tmp_path):
    from rag_gt.blocks.index_builder import run

    inputs, chunks = _chunks_artifact(tmp_path)
    out = run(inputs, {"strategy": "bm25"}, artifacts_dir=tmp_path)

    assert set(out.keys()) == {"index"}
    artifact = out["index"]
    assert artifact["type"] == "index"
    assert artifact["meta"]["strategy"] == "bm25"
    assert artifact["meta"]["docs"] == len(chunks)


def test_index_builder_manifest_reconstructs_working_bm25_retriever(tmp_path):
    from rag_gt.blocks.index_builder import run

    inputs, chunks = _chunks_artifact(tmp_path)
    out = run(inputs, {"strategy": "bm25"}, artifacts_dir=tmp_path)

    manifest = json.loads(open(out["index"]["ref"], encoding="utf-8").read())
    reconstructed_chunks = json.loads(open(manifest["chunks_ref"], encoding="utf-8").read())
    retriever = build_retriever(reconstructed_chunks, strategy=manifest["strategy"])

    assert isinstance(retriever, BM25Retriever)
    results = retriever.retrieve("alpha beta", top_k=2)
    expected = BM25Retriever(chunks).retrieve("alpha beta", top_k=2)
    assert results == expected
    assert results[0][0] == "c1"
