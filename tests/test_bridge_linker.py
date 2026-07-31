"""Stage B acceptance — verified multi-hop pairs ($0)."""
from rag_gt.graph.bridge_index import build_bridge_index
from rag_gt.graph.bridge_linker import build_bridge_pairs


def _facts():
    return [
        # genuine multi-hop: different things about 'essential variable', 2 pages
        {"doc": "d", "id": "d_F1", "text": "The preheating temperature is an essential variable.", "page": 13},
        {"doc": "d", "id": "d_F2", "text": "A change in an essential variable beyond its qualified range requires requalification.", "page": 16},
        # near-duplicate: same statement about the same bridge on two pages -> drop
        {"doc": "d", "id": "d_F3", "text": "The distance contact tube work piece is the distance from the nozzle to the work piece.", "page": 13},
        {"doc": "d", "id": "d_F4", "text": "The distance contact tube work piece is the distance from the nozzle to the work piece surface.", "page": 14},
    ]


def test_genuine_pair_kept_duplicate_dropped():
    facts = _facts()
    idx = build_bridge_index(facts)
    res = build_bridge_pairs(facts, idx)
    pairs = {(p["fact_a"], p["fact_b"]) for p in res["pairs"]}
    assert ("d_F1", "d_F2") in pairs, res["pairs"]          # genuine multi-hop kept
    assert ("d_F3", "d_F4") not in pairs                     # duplicate dropped
    assert res["stats"]["dropped_duplicate"] >= 1


def test_pairs_are_cross_page():
    facts = _facts()
    res = build_bridge_pairs(facts, build_bridge_index(facts))
    assert all(len(p["pages"]) == 2 for p in res["pairs"])
