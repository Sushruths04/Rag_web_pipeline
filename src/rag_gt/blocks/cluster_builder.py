"""Block: cluster_builder [FREE] -- bridges + facts -> candidates.

Wraps rag_gt.generation.cluster_bridge.build_clusters(bridge_pairs, facts,
...) verbatim (05_BLOCK_CATALOG.md §3.14). build_clusters returns
(clusters, fallback_pairs); both are written into one candidates artifact so
the 2-fact fallback path (a bridge pair whose fact(s) have no eligible
same-page neighbour) is preserved rather than dropped.
"""
from __future__ import annotations

from pathlib import Path

from rag_gt.blocks._common import artifact, read_list_input, write_json_artifact
from rag_gt.generation.cluster_bridge import build_clusters


def run(inputs: dict, params: dict, artifacts_dir: Path | str | None = None) -> dict:
    bridge_pairs = read_list_input(inputs.get("bridges"))
    facts = read_list_input(inputs.get("facts"))

    clusters, fallback_pairs = build_clusters(
        bridge_pairs,
        facts,
        window=int(params.get("window", 3)),
        min_cosine=float(params.get("min_cosine", 0.40)),
        max_cosine=params.get("max_cosine", 0.95),
    )
    payload = {"clusters": clusters, "fallback_pairs": fallback_pairs}
    ref = write_json_artifact(artifacts_dir, "cluster_builder", payload)
    return {
        "candidates": artifact(
            "candidates",
            str(ref),
            {"pairs": len(fallback_pairs), "clusters": len(clusters)},
        )
    }
