"""CP3 — Fact containment checker (token overlap, >=60%).

fact_hit(fact, chunk_texts)  → bool
match_pair(pair, ranked_results, retriever, top_k) → MatchResult

Uses the same _tokenize and >=60% token overlap rule as evaluate.py
(_text_covered / _fact_chunk_ids) to ensure comparable semantics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rag_gt.core.config import load_config

# ---------------------------------------------------------------------------
# Tokenizer (identical to evaluate.py)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Core containment check
# ---------------------------------------------------------------------------

OVERLAP_THRESHOLD = float(
    load_config()["multigold_evaluation"]["match_overlap_min"]
)


def fact_hit(fact: dict, chunk_texts: List[str]) -> bool:
    """Return True if >=60% of the fact's tokens appear in any single chunk."""
    text = (fact.get("canonical_form") or fact.get("text") or "").strip()
    if not text:
        return False
    fact_toks = set(_tokenize(text))
    if not fact_toks:
        return False
    for ct in chunk_texts:
        chunk_toks = set(_tokenize(ct))
        if len(fact_toks & chunk_toks) / len(fact_toks) >= OVERLAP_THRESHOLD:
            return True
    return False


def fact_overlap(fact: dict, chunk_text: str) -> float:
    """Return the token overlap fraction (0..1) between fact and a single chunk."""
    text = (fact.get("canonical_form") or fact.get("text") or "").strip()
    if not text:
        return 0.0
    fact_toks = set(_tokenize(text))
    if not fact_toks:
        return 0.0
    chunk_toks = set(_tokenize(chunk_text))
    return len(fact_toks & chunk_toks) / len(fact_toks)


# ---------------------------------------------------------------------------
# MatchResult — output of match_pair
# ---------------------------------------------------------------------------

@dataclass
class FactMatch:
    fact_id: str
    fact_text: str
    hit: bool                         # covered in top_k
    first_hit_rank: Optional[int]     # 1-indexed rank of first covering chunk; None if miss
    best_overlap: float               # best overlap fraction across all retrieved chunks


@dataclass
class MatchResult:
    question: str
    pair_type: str
    depth: int
    page_spread: int
    necessity_score: float
    n_facts: int
    retrieved_chunk_ids: List[str]    # ordered by rank
    fact_matches: List[FactMatch]
    relevant_ranks: List[int] = field(default_factory=list)

    # Derived per-question metrics (filled by metrics.py)
    fact_recall_at_k: Dict[int, float] = field(default_factory=dict)  # k -> recall
    precision_at_k: Dict[int, float] = field(default_factory=dict)    # k -> relevant chunks / k
    precision_rw_at_k: Dict[int, float] = field(default_factory=dict) # k -> rank-weighted precision
    hit_at_k: Dict[int, int] = field(default_factory=dict)            # k -> 0/1
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)         # k -> ndcg
    coverage: float = 0.0  # fact_recall over all retrieved (like evaluate.py)


# ---------------------------------------------------------------------------
# match_pair
# ---------------------------------------------------------------------------

def match_pair(
    pair: dict,
    ranked_results: List[Tuple[str, float]],   # [(chunk_id, score)] from retriever
    id_to_text: Dict[str, str],                # chunk_id -> text
    top_k_values: List[int] = (1, 3, 5, 10),
) -> MatchResult:
    """Match a GT pair against retrieval results.

    ranked_results should be sorted by descending score (rank 1 = first entry).
    id_to_text must cover all chunk_ids in ranked_results.
    """
    facts = pair.get("facts", [])
    ranked_ids = [cid for cid, _ in ranked_results]

    fact_matches: List[FactMatch] = []
    relevant_ranks: set[int] = set()
    for fact in facts:
        first_hit_rank: Optional[int] = None
        best_overlap = 0.0
        for rank, cid in enumerate(ranked_ids, start=1):
            chunk_text = id_to_text.get(cid, "")
            ov = fact_overlap(fact, chunk_text)
            if ov > best_overlap:
                best_overlap = ov
            if ov >= OVERLAP_THRESHOLD and first_hit_rank is None:
                first_hit_rank = rank
            if ov >= OVERLAP_THRESHOLD:
                relevant_ranks.add(rank)

        fm = FactMatch(
            fact_id=fact.get("fact_id", ""),
            fact_text=(fact.get("canonical_form") or fact.get("text") or ""),
            hit=first_hit_rank is not None,
            first_hit_rank=first_hit_rank,
            best_overlap=best_overlap,
        )
        fact_matches.append(fm)

    return MatchResult(
        question=pair.get("question", ""),
        pair_type=pair.get("pair_type", "unknown"),
        depth=pair.get("depth", len(facts)),
        page_spread=pair.get("page_spread", 0),
        necessity_score=pair.get("necessity_score", 0.0),
        n_facts=len(facts),
        retrieved_chunk_ids=ranked_ids,
        fact_matches=fact_matches,
        relevant_ranks=sorted(relevant_ranks),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CP3 Matcher Smoke Test ===")

    from rag_gt.rag.loader import load_chunks, load_gt_pairs
    from rag_gt.rag.retriever import BM25Retriever

    doc_id = "ecma404_json"
    chunks = load_chunks(doc_id)
    pairs = load_gt_pairs(doc_id)
    retriever = BM25Retriever(chunks)
    id_to_text = {c["chunk_id"]: c["text"] for c in chunks}

    hits = 0
    for pair in pairs[:5]:
        results = retriever.retrieve(pair["question"], top_k=10)
        mr = match_pair(pair, results, id_to_text)
        n_hit = sum(1 for fm in mr.fact_matches if fm.hit)
        hits += n_hit
        print(f"\nQ: {pair['question'][:70]}")
        print(f"  type={mr.pair_type} depth={mr.depth}")
        for fm in mr.fact_matches:
            symbol = "HIT" if fm.hit else "MISS"
            print(f"  [{symbol}] rank={fm.first_hit_rank} overlap={fm.best_overlap:.2f} "
                  f"| {fm.fact_text[:60]}")
    print(f"\nFact hits in first 5 pairs: {hits}")
