"""Block: chunks_import [FREE] -- (empty) -> chunks.

Wraps rag_gt.rag.loader.load_chunks(doc_id) verbatim (05_BLOCK_CATALOG.md
§3.2). Params: {"doc_id": str}.
"""
from __future__ import annotations

from pathlib import Path

from rag_gt.blocks._common import artifact, write_json_artifact
from rag_gt.rag.loader import load_chunks


def run(inputs: dict, params: dict, artifacts_dir: Path | str | None = None) -> dict:
    doc_id = str(params.get("doc_id") or "")
    chunks = load_chunks(doc_id)
    ref = write_json_artifact(artifacts_dir, "chunks_import", chunks)
    return {
        "chunks": artifact(
            "chunks", str(ref), {"count": len(chunks), "doc_id": doc_id}
        )
    }
