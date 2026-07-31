import random

import numpy as np

from rag_gt.core.types import Fact, Span
from rag_gt.generation.multihop_sampler import (
    enumerate_candidate_chains,
    sample_chain,
)
from rag_gt.vectorstore.faiss_index import FactIndex


def _mk_fact(fid: str, text: str, role="definition", chunk="c1", weight=0.7) -> Fact:
    if len(text) < 10:
        text = text + " padding text to satisfy length"
    f = Fact(
        fact_id=fid,
        text=text,
        role=role,
        weight=weight,
        supporting_spans=[
            Span(doc_id="d1", chunk_id=chunk, start_token=0, end_token=5)
        ],
    )
    return f


def _mk_index(facts, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(len(facts), dim)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-9
    idx = FactIndex(dim=dim, backend="sklearn")
    idx.add(base, [f.fact_id for f in facts])
    return idx, base


def test_depth1_returns_single_fact():
    facts = [_mk_fact(f"F{i:03d}", f"fact number {i} about welding") for i in range(5)]
    idx, _ = _mk_index(facts)
    chain = sample_chain(facts, idx, depth=1, rng=random.Random(0))
    assert chain is not None
    assert chain.depth == 1
    assert len(chain.fact_ids) == 1


def test_depth3_succeeds_when_threshold_permissive():
    """With min_cosine=-1.0 (no filtering) and 12 random facts, the sampler
    must produce a chain of depth 3. The previous version of this test
    accepted `chain is None`, which made an always-None sampler pass."""
    facts = [_mk_fact(f"F{i:03d}", f"fact {i}") for i in range(12)]
    idx, _vecs = _mk_index(facts, seed=7)
    chain = sample_chain(
        facts, idx, depth=3, min_cosine=-1.0, role_bias=False, rng=random.Random(1)
    )
    assert chain is not None
    assert chain.depth == 3
    assert len(chain.fact_ids) == 3


def test_depth3_returns_none_when_threshold_excludes_all():
    """With a high min_cosine and only 3 facts, the chain may not satisfy the
    threshold; returning None is the correct outcome."""
    facts = [_mk_fact(f"F{i:03d}", f"fact {i}") for i in range(3)]
    idx, _ = _mk_index(facts, seed=11)
    chain = sample_chain(
        facts, idx, depth=3, min_cosine=0.99, role_bias=False, rng=random.Random(2)
    )
    assert chain is None


def test_chain_dedupe():
    facts = [_mk_fact(f"F{i:03d}", f"fact {i}") for i in range(6)]
    idx, _ = _mk_index(facts, seed=9)
    chains = enumerate_candidate_chains(
        facts, idx, n_chains_per_depth={1: 2, 2: 3}, min_cosine=-1.0,
        role_bias=False, rng=random.Random(0),
    )
    # The sampler dedupes by `tuple(fact_ids)` (order-preserving), so we test
    # that no two chains share the same ordered tuple.
    keys = [tuple(c.fact_ids) for c in chains]
    assert len(keys) == len(set(keys))


def test_empty_facts_returns_none():
    idx = FactIndex(dim=8, backend="sklearn")
    assert sample_chain([], idx, depth=1) is None
    assert sample_chain([], idx, depth=3) is None
