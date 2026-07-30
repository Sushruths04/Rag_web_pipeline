"""Block: index_builder [FREE] -- chunks -> index.

Wraps rag_gt.rag.retriever.build_retriever(chunks, strategy=..., embed_source=...)
verbatim (05_BLOCK_CATALOG.md §3.27). The engine's retriever objects
(BM25Retriever/DenseRetriever/HybridRetriever) are not JSON-serializable and
have no save/load, so the index artifact's ref is a small manifest pointing
back at the upstream chunks artifact plus the build params; the retriever is
rebuilt on demand (see rag_gt.blocks.evaluator) by calling build_retriever
again -- no retriever internals are reimplemented here.
"""
from __future__ import annotations

from pathlib import Path

from rag_gt.blocks._common import artifact, read_list_input, write_json_artifact
from rag_gt.rag.retriever import build_retriever


def run(inputs: dict, params: dict, artifacts_dir: Path | str | None = None) -> dict:
    chunks_artifact = inputs.get("chunks")
    chunks = read_list_input(chunks_artifact)
    strategy = str(params.get("strategy") or "bm25")
    embed_source = str(params.get("embedding_source") or "local")

    # Build once here to fail fast on an invalid strategy/config, exactly
    # like the real engine call would -- the manifest is written only if
    # this succeeds.
    build_retriever(chunks, strategy=strategy, embed_source=embed_source)

    manifest = {
        "chunks_ref": chunks_artifact["ref"] if chunks_artifact else None,
        "strategy": strategy,
        "embed_source": embed_source,
    }
    ref = write_json_artifact(artifacts_dir, "index_builder", manifest)
    return {
        "index": artifact(
            "index", str(ref), {"strategy": strategy, "docs": len(chunks)}
        )
    }
