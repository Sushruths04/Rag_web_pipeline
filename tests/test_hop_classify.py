"""Unit tests for single-hop / multi-hop classification (all-PDF)."""

from rag_gt.allpdf.hop_classify import (
    MULTI_CROSS_NEAR,
    MULTI_CROSS_STRICT,
    MULTI_DEEP,
    MULTI_SAME_PAGE,
    SINGLE_HOP,
    classify_hop,
    classify_pair_dict,
    is_multi_hop,
)


def test_same_page_neighbours_are_single_hop():
    res = classify_hop([34, 34], [310, 312], window=3)
    assert res["pair_type"] == SINGLE_HOP
    assert res["page_spread"] == 0
    assert res["neighbor_distance"] == 2
    assert res["is_multi"] is False


def test_same_page_far_apart_is_multi_same_page():
    res = classify_hop([38, 38], [10, 50], window=3)
    assert res["pair_type"] == MULTI_SAME_PAGE
    assert res["page_spread"] == 0
    assert res["is_multi"] is True


def test_adjacent_page_is_cross_page_near():
    res = classify_hop([30, 31], [273, 275], window=3)
    assert res["pair_type"] == MULTI_CROSS_NEAR
    assert res["page_spread"] == 1


def test_far_apart_pages_is_cross_page_strict():
    res = classify_hop([11, 45], [21, 350], window=3)
    assert res["pair_type"] == MULTI_CROSS_STRICT
    assert res["page_spread"] == 34
    assert res["is_multi"] is True


def test_depth_three_is_deep_multi_hop():
    res = classify_hop([10, 10, 10], [1, 2, 3], window=3)
    assert res["pair_type"] == MULTI_DEEP
    assert res["depth"] == 3


def test_missing_pages_falls_back_to_neighbour_distance():
    res = classify_hop([None, None], [10, 11], window=3)
    assert res["pair_type"] == SINGLE_HOP
    assert res["page_spread"] is None
    res_far = classify_hop([None, None], [10, 80], window=3)
    assert res_far["pair_type"] == MULTI_SAME_PAGE


def test_classify_pair_dict_reads_saved_row():
    pair = {
        "chain_fact_ids": ["doc_F000310", "doc_F000312"],
        "facts": [
            {"fact_id": "doc_F000310", "page_start": 34},
            {"fact_id": "doc_F000312", "page_start": 34},
        ],
    }
    res = classify_pair_dict(pair, window=3)
    assert res["pair_type"] == SINGLE_HOP
    assert res["neighbor_distance"] == 2


def test_classify_pair_dict_cross_page():
    pair = {
        "chain_fact_ids": ["doc_F000859", "doc_F000897"],
        "facts": [
            {"fact_id": "doc_F000859", "page_start": 77},
            {"fact_id": "doc_F000897", "page_start": 79},
        ],
    }
    res = classify_pair_dict(pair, window=3)
    assert res["pair_type"] == MULTI_CROSS_NEAR
    assert res["page_spread"] == 2


def test_is_multi_hop_helper():
    assert is_multi_hop(MULTI_DEEP)
    assert is_multi_hop(MULTI_CROSS_STRICT)
    assert not is_multi_hop(SINGLE_HOP)
