from rag_gt.generation.neighbor_pairs import sample_neighbor_pairs


def _fact(fid, page, char, text="alpha beta gamma"):
    return {"fact_id": fid, "text": text, "canonical_form": text,
            "page_start": page, "char_start": char, "chunk_id": f"ck_{fid}"}


FACTS = [
    _fact("F1", 1, 0), _fact("F2", 1, 100), _fact("F3", 1, 200),
    _fact("F4", 2, 0), _fact("F5", 2, 100),
    _fact("F6", 5, 0),  # alone on its page -> never paired
]


def test_pairs_are_same_page_and_disjoint():
    pairs = sample_neighbor_pairs(FACTS, doc="d")
    ids = [p["fact_a"] for p in pairs] + [p["fact_b"] for p in pairs]
    assert len(ids) == len(set(ids))                     # each fact in <= 1 pair
    assert {("F1", "F2"), ("F4", "F5")} <= {(p["fact_a"], p["fact_b"]) for p in pairs}
    assert all(p["page"] in (1, 2) for p in pairs)       # F6 never paired


def test_window_respected():
    far = [_fact("A", 1, 0)] + [_fact(f"X{i}", 1, 100 * (i + 1)) for i in range(3)] + [_fact("B", 1, 999)]
    pairs = sample_neighbor_pairs(far, window=1)
    assert all(p["distance"] == 1 for p in pairs)


def test_deterministic():
    assert sample_neighbor_pairs(FACTS) == sample_neighbor_pairs(list(reversed(FACTS)))


def test_cosine_guard_filters_unrelated():
    import numpy as np

    def fake_embed(texts):
        # F1/F2 similar, F3 orthogonal to both
        table = {"alpha beta gamma": [1.0, 0.0], "unrelated topic entirely": [0.0, 1.0]}
        return np.array([table[t] for t in texts])

    facts = [_fact("F1", 1, 0), _fact("F2", 1, 50),
             _fact("F3", 1, 100, text="unrelated topic entirely")]
    pairs = sample_neighbor_pairs(facts, embed_fn=fake_embed, min_cosine=0.4)
    assert {(p["fact_a"], p["fact_b"]) for p in pairs} == {("F1", "F2")}
    assert pairs[0]["cosine"] == 1.0


def test_max_pairs_caps_output():
    assert len(sample_neighbor_pairs(FACTS, max_pairs=1)) == 1


def test_cosine_cap_rejects_near_duplicate_neighbour():
    import numpy as np

    def fake_embed(texts):
        # F1 and F2 are near-duplicates (cos ~1.0); F3 is related but distinct (~0.7)
        table = {
            "alpha beta gamma": [1.0, 0.0],
            "alpha beta gamma again": [0.999, 0.045],
            "alpha related claim": [0.7, 0.714],
        }
        return np.array([table[t] for t in texts])

    facts = [_fact("F1", 1, 0), _fact("F2", 1, 50, text="alpha beta gamma again"),
             _fact("F3", 1, 100, text="alpha related claim")]
    pairs = sample_neighbor_pairs(facts, embed_fn=fake_embed, min_cosine=0.4, max_cosine=0.95)
    # F1-F2 skipped (redundant); F1-F3 accepted (in band)
    assert {(p["fact_a"], p["fact_b"]) for p in pairs} == {("F1", "F3")}


def test_overlap_factor_allows_fact_reuse():
    # Three related same-page facts: disjoint pairing yields 1 pair; with
    # max_uses_per_fact=2 the middle facts can join a second pair.
    facts = [_fact("F1", 1, 0), _fact("F2", 1, 50), _fact("F3", 1, 100)]
    assert len(sample_neighbor_pairs(facts)) == 1
    pairs = sample_neighbor_pairs(facts, max_uses_per_fact=2)
    keys = {(p["fact_a"], p["fact_b"]) for p in pairs}
    assert len(pairs) == 2 and ("F1", "F2") in keys
