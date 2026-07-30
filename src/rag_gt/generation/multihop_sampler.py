"""Build semantically-coherent fact chains for genuine multi-hop questions.

Replaces the V8 `random.sample(facts, k)` that produced stitched non-multi-hop
questions. An anchor fact is chosen weighted by structural quality; neighbors
come from the FAISS fact index, filtered by cosine threshold and (optionally)
matched to role templates like definition->rule->consequence.

Dedup uses `tuple(chain.fact_ids)` (order-preserving) so chains that traverse
the same fact set in different orders — and therefore have different role paths
— are kept as distinct candidates.
"""

from __future__ import annotations

import random
from typing import Dict, List, NamedTuple, Optional

import numpy as np

from rag_gt.core.types import Fact, FactChain
from rag_gt.vectorstore.faiss_index import FactIndex

ROLE_TEMPLATES: List[List[str]] = [
    ["definition", "rule"],
    ["rule", "exception"],
    ["condition", "consequence"],
    ["definition", "rule", "consequence"],
    ["rule", "constraint", "exception"],
    ["definition", "condition", "rule"],
]


class _RankedCandidate(NamedTuple):
    fact_id: str
    cosine: float
    ranked_score: float


def _weighted_choice(
    facts: List[Fact], rng: random.Random, exclude: Optional[set[str]] = None
) -> Optional[Fact]:
    exclude = exclude or set()
    pool = [f for f in facts if f.fact_id not in exclude]
    if not pool:
        return None
    weights = [max(f.weight, 0.01) for f in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


def _fact_chunk_id(fact: Fact) -> str:
    if fact.supporting_spans:
        return fact.supporting_spans[0].chunk_id
    return ""


def _template_match_score(path_roles: List[str], candidate_role: str) -> int:
    for template in ROLE_TEMPLATES:
        if len(path_roles) >= len(template):
            continue
        if path_roles == template[: len(path_roles)]:
            if template[len(path_roles)] == candidate_role:
                return 2
    return 0


def sample_chain(
    facts: List[Fact],
    index: Optional[FactIndex],
    depth: int,
    min_cosine: float = 0.55,
    role_bias: bool = True,
    rng: Optional[random.Random] = None,
) -> Optional[FactChain]:
    if depth < 1 or not facts:
        return None
    rng = rng or random.Random()

    if depth == 1:
        anchor = _weighted_choice(facts, rng)
        if anchor is None:
            return None
        return FactChain(
            fact_ids=[anchor.fact_id],
            anchor_id=anchor.fact_id,
            mean_cosine=1.0,
            role_path=[anchor.role],
        )

    if index is None or not index.fact_ids:
        return None
    # O(1) membership test; the previous `in index.fact_ids` was O(n).
    index_id_set = set(index.fact_ids)

    fact_by_id: Dict[str, Fact] = {f.fact_id: f for f in facts}

    anchor = _weighted_choice(facts, rng)
    if anchor is None or anchor.fact_id not in index_id_set:
        return None

    chain_ids: List[str] = [anchor.fact_id]
    chain_roles: List[str] = [anchor.role]
    scores: List[float] = []

    current_id = anchor.fact_id
    while len(chain_ids) < depth:
        anchor_vec = index.embedding_for(current_id)
        k = max(depth * 4, 8)
        neighbors = index.search(anchor_vec, k=k)

        chain_set = set(chain_ids)
        best: Optional[_RankedCandidate] = None
        for fid, score in neighbors:
            if fid in chain_set:
                continue
            if score < min_cosine:
                continue
            cand = fact_by_id.get(fid)
            if cand is None:
                continue
            bias = (
                _template_match_score(chain_roles, cand.role) if role_bias else 0
            )
            cand_chunk = _fact_chunk_id(cand)
            cur_chunk = _fact_chunk_id(fact_by_id[current_id])
            bonus = 0.05 if cand_chunk and cand_chunk != cur_chunk else 0.0
            ranked = score + bonus + 0.01 * bias
            if best is None or ranked > best.ranked_score:
                best = _RankedCandidate(fact_id=fid, cosine=score, ranked_score=ranked)

        if best is None:
            break
        chain_ids.append(best.fact_id)
        chain_roles.append(fact_by_id[best.fact_id].role)
        scores.append(best.cosine)
        current_id = best.fact_id

    if len(chain_ids) < depth:
        return None

    return FactChain(
        fact_ids=chain_ids,
        anchor_id=anchor.fact_id,
        mean_cosine=float(np.mean(scores)) if scores else 1.0,
        role_path=chain_roles,
    )


def enumerate_candidate_chains(
    facts: List[Fact],
    index: Optional[FactIndex],
    n_chains_per_depth: Dict[int, int],
    min_cosine: float = 0.55,
    role_bias: bool = True,
    rng: Optional[random.Random] = None,
) -> List[FactChain]:
    rng = rng or random.Random()
    seen: set[tuple] = set()  # ordered dedup so role-distinct chains survive
    out: List[FactChain] = []
    for depth, n_wanted in n_chains_per_depth.items():
        attempts = 0
        max_attempts = n_wanted * 10
        accepted = 0
        while accepted < n_wanted and attempts < max_attempts:
            attempts += 1
            chain = sample_chain(
                facts, index, depth, min_cosine=min_cosine,
                role_bias=role_bias, rng=rng,
            )
            if chain is None:
                continue
            key = tuple(chain.fact_ids)
            if key in seen:
                continue
            seen.add(key)
            out.append(chain)
            accepted += 1
    rng.shuffle(out)
    return out
