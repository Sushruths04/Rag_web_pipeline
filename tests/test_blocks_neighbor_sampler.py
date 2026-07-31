"""TDD: rag_gt.blocks.neighbor_sampler wraps
rag_gt.generation.neighbor_pairs.sample_neighbor_pairs(...) verbatim
(05_BLOCK_CATALOG.md §3.13).
"""
from __future__ import annotations

import json

from rag_gt.generation.neighbor_pairs import sample_neighbor_pairs


def _fact(fid, page, char, text="alpha beta gamma"):
    return {"fact_id": fid, "doc": "d", "text": text, "canonical_form": text,
            "page_start": page, "char_start": char}


FACTS = [
    _fact("F1", 1, 0), _fact("F2", 1, 100), _fact("F3", 1, 200),
    _fact("F4", 2, 0), _fact("F5", 2, 100),
]


def _facts_artifact(tmp_path):
    from rag_gt.blocks._common import artifact, write_json_artifact

    ref = write_json_artifact(tmp_path, "facts_import", FACTS)
    return {"facts": artifact("facts", str(ref), {"count": len(FACTS), "grounded": True})}


def test_neighbor_sampler_returns_candidates_artifact(tmp_path):
    from rag_gt.blocks.neighbor_sampler import run

    inputs = _facts_artifact(tmp_path)
    out = run(inputs, {"window": 3, "max_uses_per_fact": 2}, artifacts_dir=tmp_path)

    assert set(out.keys()) == {"candidates"}
    artifact = out["candidates"]
    assert artifact["type"] == "candidates"
    assert artifact["meta"]["clusters"] == 0
    assert artifact["meta"]["pairs"] > 0


def test_neighbor_sampler_matches_sample_neighbor_pairs_directly(tmp_path):
    from rag_gt.blocks.neighbor_sampler import run

    inputs = _facts_artifact(tmp_path)
    params = {"window": 3, "min_cosine": 0.40, "max_cosine": 0.95, "max_uses_per_fact": 2}
    out = run(inputs, params, artifacts_dir=tmp_path)

    expected = sample_neighbor_pairs(
        FACTS, doc="d", window=3, min_cosine=0.40, max_cosine=0.95, max_uses_per_fact=2
    )
    written = json.loads(open(out["candidates"]["ref"], encoding="utf-8").read())
    assert written == expected
    assert out["candidates"]["meta"]["pairs"] == len(expected)


def test_neighbor_sampler_respects_max_pairs(tmp_path):
    from rag_gt.blocks.neighbor_sampler import run

    inputs = _facts_artifact(tmp_path)
    out = run(inputs, {"max_pairs": 1}, artifacts_dir=tmp_path)
    assert out["candidates"]["meta"]["pairs"] == 1
