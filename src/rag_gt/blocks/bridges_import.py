"""Block: bridges_import [FREE] -- (empty) -> bridges.

Wraps rag_gt.generation.answer_first._load_records(path, "pairs") verbatim
(05_BLOCK_CATALOG.md §3.4). Params: {"path": str} pointing at a bridge-pairs
JSON file (either a plain JSON list, or a dict with a "pairs" list, e.g.
data/eval_results/allpdf_bridge_pairs_clean.json).
"""
from __future__ import annotations

from pathlib import Path

from rag_gt.blocks._common import artifact, write_json_artifact
from rag_gt.generation.answer_first import _load_records


def run(inputs: dict, params: dict, artifacts_dir: Path | str | None = None) -> dict:
    path = Path(str(params.get("path") or ""))
    pairs = _load_records(path, "pairs")
    ref = write_json_artifact(artifacts_dir, "bridges_import", pairs)
    return {
        "bridges": artifact("bridges", str(ref), {"count": len(pairs)})
    }
