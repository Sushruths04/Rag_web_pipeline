"""Block: neighbor_sampler [FREE] -- facts -> candidates.

Wraps rag_gt.generation.neighbor_pairs.sample_neighbor_pairs(...) verbatim
(05_BLOCK_CATALOG.md §3.13). No embed_fn is threaded through yet (no
embedding-provider block exists in this M0 slice), so the cosine band
params are accepted but have no effect until an embedder is wired in --
matching sample_neighbor_pairs' own behaviour when embed_fn=None.
"""
from __future__ import annotations

from pathlib import Path

from rag_gt.blocks._common import artifact, read_list_input, write_json_artifact
from rag_gt.generation.neighbor_pairs import sample_neighbor_pairs


def run(inputs: dict, params: dict, artifacts_dir: Path | str | None = None) -> dict:
    facts = read_list_input(inputs.get("facts"))
    doc = str(facts[0].get("doc") or "") if facts else ""

    pairs = sample_neighbor_pairs(
        facts,
        doc=doc,
        window=int(params.get("window", 3)),
        min_cosine=float(params.get("min_cosine", 0.40)),
        max_cosine=params.get("max_cosine", 0.95),
        max_uses_per_fact=int(params.get("max_uses_per_fact", 2)),
        max_pairs=params.get("max_pairs"),
    )
    ref = write_json_artifact(artifacts_dir, "neighbor_sampler", pairs)
    return {
        "candidates": artifact(
            "candidates", str(ref), {"pairs": len(pairs), "clusters": 0}
        )
    }
