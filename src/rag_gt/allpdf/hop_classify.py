"""Single-hop / multi-hop classification for all-PDF Q&A pairs.

Classifies a fact chain by its source page-spread and source-order neighbour
distance. This separates honest *local-neighbor* (single / pair-hop) pairs from
genuine *cross-page* or *far-apart* multi-hop pairs.

The classification is intentionally cheap and deterministic: it reads only the
page of each fact and the source order of the facts. No model is loaded.

Two entry points share one core (`classify_hop`):
  - `classify_pair_dict(pair, window)`  — for saved s7_pairs.json rows.
  - `classify_pair_facts(facts, window)` — for live Fact objects in the pipeline.

Taxonomy (see docs/allpdf_singlehop_multihop_redesign_plan_20260622.md §3):
  single_hop                  depth==2, same page, neighbours (|idx_a-idx_b| <= W)
  multi_hop_same_page         depth==2, same page, far apart (> W)
  multi_hop_cross_page_near   depth==2, page gap in [1, 3]
  multi_hop_cross_page_strict depth==2, page gap >= 4
  multi_hop_deep              depth >= 3
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from rag_gt.core.types import Fact

DEFAULT_NEIGHBOR_WINDOW = 3
STRICT_CROSS_PAGE_GAP = 4

_FACT_NUM_RE = re.compile(r"(\d+)\s*$")

SINGLE_HOP = "single_hop"
MULTI_SAME_PAGE = "multi_hop_same_page"
MULTI_CROSS_NEAR = "multi_hop_cross_page_near"
MULTI_CROSS_STRICT = "multi_hop_cross_page_strict"
MULTI_DEEP = "multi_hop_deep"

_MULTI_TYPES = frozenset(
    {MULTI_SAME_PAGE, MULTI_CROSS_NEAR, MULTI_CROSS_STRICT, MULTI_DEEP}
)


def is_multi_hop(pair_type: str) -> bool:
    return pair_type in _MULTI_TYPES


def _fact_order_index(fact_id: str) -> Optional[int]:
    """Extraction-order proxy from the numeric suffix of a fact_id.

    `requirements_eng_F000310` -> 310. Extraction order tracks source order, so
    the suffix delta is a good proxy for source-order neighbour distance when the
    exact char_start is not available (it is not stored in s7_pairs.json).
    """
    m = _FACT_NUM_RE.search(fact_id or "")
    return int(m.group(1)) if m else None


def classify_hop(
    pages: Sequence[Optional[int]],
    order_indices: Sequence[Optional[int]],
    *,
    window: int = DEFAULT_NEIGHBOR_WINDOW,
) -> dict:
    """Core classifier over primitive page + order-index sequences.

    Returns a dict with: pair_type, depth, page_spread (or None when any page is
    missing), neighbor_distance (or None when any index is missing), is_multi.
    """
    depth = len(pages)
    known_pages = [int(p) for p in pages if p is not None]
    page_spread: Optional[int] = (
        max(known_pages) - min(known_pages) if len(known_pages) == depth and depth else None
    )
    known_idx = [int(i) for i in order_indices if i is not None]
    neighbor_distance: Optional[int] = None
    if len(known_idx) == depth and depth >= 2:
        neighbor_distance = max(
            abs(known_idx[i + 1] - known_idx[i]) for i in range(depth - 1)
        )

    if depth >= 3:
        pair_type = MULTI_DEEP
    elif depth <= 1:
        pair_type = SINGLE_HOP
    elif page_spread is None:
        # Pages unknown: fall back to neighbour distance only.
        if neighbor_distance is not None and neighbor_distance > window:
            pair_type = MULTI_SAME_PAGE
        else:
            pair_type = SINGLE_HOP
    elif page_spread == 0:
        if neighbor_distance is not None and neighbor_distance > window:
            pair_type = MULTI_SAME_PAGE
        else:
            pair_type = SINGLE_HOP
    elif page_spread >= STRICT_CROSS_PAGE_GAP:
        pair_type = MULTI_CROSS_STRICT
    else:
        pair_type = MULTI_CROSS_NEAR

    return {
        "pair_type": pair_type,
        "depth": depth,
        "page_spread": page_spread,
        "neighbor_distance": neighbor_distance,
        "is_multi": is_multi_hop(pair_type),
    }


def classify_pair_dict(pair: dict, *, window: int = DEFAULT_NEIGHBOR_WINDOW) -> dict:
    """Classify a saved s7_pairs.json row.

    Reads `facts[].page_start` for pages and `chain_fact_ids` for order indices.
    """
    facts = pair.get("facts") or []
    fact_ids = pair.get("chain_fact_ids") or [f.get("fact_id", "") for f in facts]
    pages = [f.get("page_start") for f in facts]
    if not pages and fact_ids:
        pages = [None] * len(fact_ids)
    order = [_fact_order_index(fid) for fid in fact_ids]
    # Align lengths defensively: depth is driven by fact_ids when facts[] missing.
    depth = max(len(pages), len(order))
    pages = (list(pages) + [None] * depth)[:depth]
    order = (list(order) + [None] * depth)[:depth]
    return classify_hop(pages, order, window=window)


def _fact_page(fact: Fact) -> Optional[int]:
    for span in fact.supporting_spans:
        if span.page_start is not None:
            return int(span.page_start)
    return None


def _fact_char_start(fact: Fact) -> Optional[int]:
    starts = [
        int(span.char_start)
        for span in fact.supporting_spans
        if span.char_start is not None
    ]
    return min(starts) if starts else None


def classify_pair_facts(
    facts: List[Fact],
    *,
    window: int = DEFAULT_NEIGHBOR_WINDOW,
    order_index_by_id: Optional[dict] = None,
) -> dict:
    """Classify a chain of live Fact objects.

    `order_index_by_id` maps fact_id -> exact source-order index (built once per
    document from the kept-facts list sorted by (page, char_start, fact_id)).
    When omitted, falls back to the fact_id numeric-suffix proxy.
    """
    pages = [_fact_page(f) for f in facts]
    if order_index_by_id is not None:
        order = [order_index_by_id.get(f.fact_id) for f in facts]
    else:
        order = [_fact_order_index(f.fact_id) for f in facts]
    return classify_hop(pages, order, window=window)


def build_source_order_index(facts: List[Fact]) -> dict:
    """Map fact_id -> exact source-order index, matching the pair scheduler.

    Order key is (page, char_start, fact_id), identical to
    pair_scheduler._collect_source_neighbours so neighbour distance is exact.
    """
    ordered = sorted(
        facts,
        key=lambda f: (
            _fact_page(f) if _fact_page(f) is not None else 10**12,
            _fact_char_start(f) if _fact_char_start(f) is not None else 10**12,
            f.fact_id,
        ),
    )
    return {f.fact_id: i for i, f in enumerate(ordered)}
