"""Block: facts_import [FREE] -- (empty) -> facts.

Wraps rag_gt.generation.answer_first._load_records(path) verbatim
(05_BLOCK_CATALOG.md §3.3). Params: {"path": str} pointing at a grounded
facts JSON file (a plain JSON list, e.g. data/eval_results/facts_v1_grounded/
facts_<doc>.json).
"""
from __future__ import annotations

from pathlib import Path

from rag_gt.blocks._common import artifact, write_json_artifact
from rag_gt.generation.answer_first import _load_records


def run(inputs: dict, params: dict, artifacts_dir: Path | str | None = None) -> dict:
    path = Path(str(params.get("path") or ""))
    facts = _load_records(path)
    ref = write_json_artifact(artifacts_dir, "facts_import", facts)
    return {
        "facts": artifact(
            "facts", str(ref), {"count": len(facts), "grounded": True}
        )
    }
