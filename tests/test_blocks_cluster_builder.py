"""TDD: rag_gt.blocks.cluster_builder wraps
rag_gt.generation.cluster_bridge.build_clusters(bridge_pairs, facts, ...)
verbatim (05_BLOCK_CATALOG.md §3.14). Output candidates meta carries both
"clusters" and leftover "pairs" (fallback 2-fact path), matching the
"52 pairs · 147 clusters" wire-badge convention.
"""
from __future__ import annotations

import json

from rag_gt.generation.cluster_bridge import build_clusters


def _fact(fid, page, char, text="alpha beta gamma"):
    return {"fact_id": fid, "doc": "d", "text": text, "canonical_form": text,
            "page_start": page, "char_start": char}


FACTS = [
    _fact("F1", 1, 0), _fact("F2", 1, 100), _fact("F3", 1, 200),
    _fact("F4", 1, 300), _fact("F5", 2, 0),
]

BRIDGE_PAIRS = [
    {"doc": "d", "fact_a": "F1", "fact_b": "F4", "bridge_entity": "X", "pages": [1, 1]},
    {"doc": "d", "fact_a": "F5", "fact_b": "F999", "bridge_entity": "Y", "pages": [2, 9]},
]


def _bridges_and_facts_artifacts(tmp_path):
    from rag_gt.blocks._common import artifact, write_json_artifact

    bref = write_json_artifact(tmp_path, "bridges_import", BRIDGE_PAIRS)
    fref = write_json_artifact(tmp_path, "facts_import", FACTS)
    return {
        "bridges": artifact("bridges", str(bref), {"count": len(BRIDGE_PAIRS)}),
        "facts": artifact("facts", str(fref), {"count": len(FACTS), "grounded": True}),
    }


def test_cluster_builder_returns_candidates_artifact(tmp_path):
    from rag_gt.blocks.cluster_builder import run

    inputs = _bridges_and_facts_artifacts(tmp_path)
    out = run(inputs, {"window": 3}, artifacts_dir=tmp_path)

    assert set(out.keys()) == {"candidates"}
    artifact = out["candidates"]
    assert artifact["type"] == "candidates"
    # F1/F4 have neighbours -> becomes a cluster; F5/F999 -> F999 not in facts -> fallback
    assert artifact["meta"]["clusters"] == 1
    assert artifact["meta"]["pairs"] == 1


def test_cluster_builder_matches_build_clusters_directly(tmp_path):
    from rag_gt.blocks.cluster_builder import run

    inputs = _bridges_and_facts_artifacts(tmp_path)
    out = run(inputs, {"window": 3, "min_cosine": 0.40, "max_cosine": 0.95}, artifacts_dir=tmp_path)

    expected_clusters, expected_fallback = build_clusters(
        BRIDGE_PAIRS, FACTS, window=3, min_cosine=0.40, max_cosine=0.95
    )
    written = json.loads(open(out["candidates"]["ref"], encoding="utf-8").read())
    assert written == {"clusters": expected_clusters, "fallback_pairs": expected_fallback}
