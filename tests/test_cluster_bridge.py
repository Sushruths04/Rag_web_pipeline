from rag_gt.generation.cluster_bridge import build_clusters


def _fact(fid, page, char, text="alpha beta gamma"):
    return {"fact_id": fid, "text": text, "canonical_form": text,
            "page_start": page, "char_start": char, "chunk_id": f"ck_{fid}"}


FACTS = [
    _fact("A", 1, 0), _fact("A2", 1, 100),          # page 1: anchor + neighbour
    _fact("B", 3, 0), _fact("B2", 3, 100),          # page 3: anchor + neighbour
    _fact("C", 7, 0),                                # page 7: anchor, NO neighbour
]

BRIDGE = {"doc": "d", "fact_a": "A", "fact_b": "B", "bridge_entity": "ISO 15607",
          "bridge_norm": "iso 15607", "bridge_type": "STANDARD_REF",
          "pages": [1, 3], "pair_id": "BP0001"}
LONELY = {"doc": "d", "fact_a": "A", "fact_b": "C", "bridge_entity": "ISO 15607",
          "bridge_norm": "iso 15607", "bridge_type": "STANDARD_REF",
          "pages": [1, 7], "pair_id": "BP0002"}


def test_builds_cluster_with_both_neighbours():
    clusters, fallback = build_clusters([BRIDGE], FACTS)
    assert len(clusters) == 1 and not fallback
    c = clusters[0]
    assert (c["fact_a"], c["fact_a2"], c["fact_b"], c["fact_b2"]) == ("A", "A2", "B", "B2")
    assert c["kind"] == "cluster_2plus2" and c["pages"] == [1, 3]
    assert c["source_pair_id"] == "BP0001"


def test_falls_back_when_one_side_has_no_neighbour():
    clusters, fallback = build_clusters([LONELY], FACTS)
    assert not clusters and fallback == [LONELY]


def test_neighbour_never_reuses_the_other_anchor():
    # B sits within A's window on the same page: it must not be chosen as A2.
    facts = [_fact("A", 1, 0), _fact("B", 1, 50), _fact("A2", 1, 100)]
    pair = dict(BRIDGE, pages=[1, 1])
    clusters, fallback = build_clusters([pair], facts)
    if clusters:
        c = clusters[0]
        assert c["fact_a2"] not in (c["fact_a"], c["fact_b"])
        assert c["fact_b2"] not in (c["fact_a"], c["fact_b"])
        assert c["fact_a2"] != c["fact_b2"]


def test_deterministic():
    a = build_clusters([BRIDGE], FACTS)
    b = build_clusters([BRIDGE], list(reversed(FACTS)))
    assert a == b


def test_cosine_cap_skips_duplicate_neighbour_picks_next():
    import numpy as np

    def fake_embed(texts):
        table = {
            "alpha beta gamma": [1.0, 0.0],          # anchors + B2
            "alpha beta gamma copy": [0.999, 0.045],  # near-duplicate of anchor
            "alpha related claim": [0.7, 0.714],      # in-band alternative
        }
        return np.array([table[t] for t in texts])

    facts = [_fact("A", 1, 0), _fact("Adup", 1, 50, text="alpha beta gamma copy"),
             _fact("A3", 1, 100, text="alpha related claim"),
             _fact("B", 3, 0), _fact("B2", 3, 100, text="alpha related claim")]
    clusters, _ = build_clusters([BRIDGE], facts, embed_fn=fake_embed, max_cosine=0.95)
    assert len(clusters) == 1
    assert clusters[0]["fact_a2"] == "A3"  # duplicate skipped, next in-band neighbour chosen


def test_cosine_guard_drops_unrelated_neighbour():
    import numpy as np

    def fake_embed(texts):
        table = {"alpha beta gamma": [1.0, 0.0], "noise": [0.0, 1.0]}
        return np.array([table[t] for t in texts])

    facts = [_fact("A", 1, 0), _fact("A2", 1, 100, text="noise"),
             _fact("B", 3, 0), _fact("B2", 3, 100)]
    clusters, fallback = build_clusters([BRIDGE], facts, embed_fn=fake_embed)
    assert not clusters and fallback == [BRIDGE]
