"""Proportional single-hop budget allocation across documents.

Multi-hop pairs are always kept in full; the remaining budget is split across
documents proportionally to how many singles each produced (largest-remainder
rounding), so a 20-page standard is never asked to carry as many questions as
a 500-page book.
"""
from rag_gt.generation.dataset_budget import allocate_singles


def test_budget_larger_than_supply_keeps_everything():
    counts = {"a": 10, "b": 5}
    assert allocate_singles(counts, 50) == counts


def test_proportional_split_with_largest_remainder():
    counts = {"big": 90, "small": 10}
    alloc = allocate_singles(counts, 50)
    assert sum(alloc.values()) == 50
    assert alloc["big"] == 45 and alloc["small"] == 5


def test_allocation_never_exceeds_per_doc_supply():
    counts = {"tiny": 3, "big": 100}
    alloc = allocate_singles(counts, 60)
    assert alloc["tiny"] <= 3
    assert sum(alloc.values()) == 60


def test_zero_budget():
    assert allocate_singles({"a": 4}, 0) == {"a": 0}
