"""Block: chunker [FREE] -- chunks -> chunks.

Wraps rag_gt.rag.loader.rechunk(chunks, strategy) verbatim
(05_BLOCK_CATALOG.md §3.6). Params: {"strategy": "original"|"sentence"|
"sliding_256"|"paragraph"}.
"""
from __future__ import annotations

from pathlib import Path

from rag_gt.blocks._common import artifact, read_list_input, write_json_artifact
from rag_gt.rag.loader import rechunk


def run(inputs: dict, params: dict, artifacts_dir: Path | str | None = None) -> dict:
    chunks = read_list_input(inputs.get("chunks"))
    strategy = str(params.get("strategy") or "original")
    out_chunks = rechunk(chunks, strategy)
    ref = write_json_artifact(artifacts_dir, "chunker", out_chunks)
    return {
        "chunks": artifact(
            "chunks", str(ref), {"count": len(out_chunks), "strategy": strategy}
        )
    }
