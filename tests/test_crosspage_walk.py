"""Phase 2 tests — cross-page multi-hop walk + leave-one-out necessity.

Builds a TypedSFG with hand-set edges (no LLM/NLI) and verifies the cross-page
controls on walk_typed_paths.
"""

from __future__ import annotations

import random

from collections import defaultdict

from rag_gt.core.types import Fact, Span
from rag_gt.graph.edge_classifier import EdgeRecord
from rag_gt.graph.typed_sfg import TypedSFG


def _page_fact(fid: str, page: int) -> Fact:
    return Fact(
        fact_id=fid,
        text=f"Fact {fid} describes a technical concept on page {page}.",
        role="definition",  # type: ignore[arg-type]
        supporting_spans=[
            Span(doc_id="d1", chunk_id=f"d1_c{page:03d}", start_token=0,
                 end_token=8, page_start=page, page_end=page)
        ],
    )


def _edge(src: str, dst: str) -> EdgeRecord:
    return EdgeRecord(
        edge_id=f"{src}->{dst}", src=src, dst=dst, type="causal",
        bridging_fact_id=src, bridging_quote="", relation_claim="x causes y",
        nli_score=0.9,
    )


def _make_sfg(facts, edges) -> TypedSFG:
    # FactIndex is unused by the walker; pass a stub via __new__ to avoid embedding.
    sfg = TypedSFG.__new__(TypedSFG)
    sfg.facts = {f.fact_id: f for f in facts}
    sfg.adj = defaultdict(list)
    sfg.edge_map = {}
    for e in edges:
        sfg.adj[e.src].append(e)
        sfg.edge_map[(e.src, e.dst)] = e
    sfg._built = True
    return sfg


def test_require_cross_page_skips_same_page_edges():
    # A->B same page (1), A->C cross page (1->5).
    facts = [_page_fact("A", 1), _page_fact("B", 1), _page_fact("C", 5)]
    edges = [_edge("A", "B"), _edge("A", "C")]
    sfg = _make_sfg(facts, edges)
    rng = random.Random(0)
    chains = sfg.walk_typed_paths(
        {2: 20}, rng, require_cross_page=True, min_page_gap=4
    )
    assert chains, "should produce at least one cross-page chain"
    for c in chains:
        # Every produced chain must be the cross-page one (A->C).
        assert c.fact_ids == ["A", "C"]


def test_require_cross_page_yields_nothing_when_only_same_page():
    facts = [_page_fact("A", 1), _page_fact("B", 1)]
    edges = [_edge("A", "B")]
    sfg = _make_sfg(facts, edges)
    rng = random.Random(0)
    chains = sfg.walk_typed_paths(
        {2: 10}, rng, require_cross_page=True, min_page_gap=1
    )
    assert chains == []


def test_default_walk_unconstrained_still_works():
    facts = [_page_fact("A", 1), _page_fact("B", 1)]
    edges = [_edge("A", "B")]
    sfg = _make_sfg(facts, edges)
    rng = random.Random(0)
    chains = sfg.walk_typed_paths({2: 5}, rng)
    assert chains and all(len(c.fact_ids) == 2 for c in chains)


def test_leave_one_out_necessity_skips_abstention():
    from rag_gt.allpdf.necessity import leave_one_out_necessity
    facts = [_page_fact("A", 1), _page_fact("B", 5)]
    assert leave_one_out_necessity("Insufficient information to answer.", facts) is None
    assert leave_one_out_necessity("real answer", [facts[0]]) is None
