"""Phase 3 — Context-cluster chain sampler.

Builds 4-fact chains with the structure:

    [A1, A2]  from page X   (same-page single-hop, NLI-verified edge)
    [B1, B2]  from page Y   (same-page single-hop, NLI-verified edge)

Full chain: A1 → A2 ··bridge·· B1 → B2
                       ↑
             cross-page semantic bridge (A and B pages differ by ≥ min_page_gap)

Requirements per chain:
- A1.page == A2.page              (cluster A is one-page)
- B1.page == B2.page              (cluster B is one-page)
- |A.page − B.page| ≥ min_page_gap
- All 4 facts have ≥ min_fact_words words  (no fragments)
- All 4 fact texts are unique

Two modes:
1. From existing pairs (fast, zero LLM): supply a list of already-generated
   2-fact same-page pairs from s7_pairs.json and combine them into clusters.
2. From a live SFG (pipeline integration): walk depth-4 paths through the graph,
   filter for the cluster pattern.

The fast mode is used by the demo script; the SFG mode is the production path
(added to pipeline.py as Stage 6c).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional


# ── Helpers ────────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _page(fact: dict) -> Optional[int]:
    """Return page_start from a fact dict (s7_pairs.json format)."""
    return fact.get("page_start")


def _text(fact: dict) -> str:
    return (fact.get("text") or fact.get("canonical_form") or "").strip()


# ── Cluster dataclass ───────────────────────────────────────────────────────────

@dataclass
class ClusterChain:
    """A 4-fact context cluster spanning two pages."""
    facts: list[dict]           # [A1, A2, B1, B2] — raw fact dicts
    page_a: int                 # page of cluster A
    page_b: int                 # page of cluster B
    page_gap: int               # |page_a - page_b|
    source_pair_a: dict = field(default_factory=dict)  # original 2-fact pair for A
    source_pair_b: dict = field(default_factory=dict)  # original 2-fact pair for B

    @property
    def fact_ids(self) -> list[str]:
        return [f.get("fact_id", "") for f in self.facts]

    def to_dict(self) -> dict:
        return {
            "pair_type": "multi_hop_cluster",
            "depth": 4,
            "page_a": self.page_a,
            "page_b": self.page_b,
            "page_gap": self.page_gap,
            "chain_fact_ids": self.fact_ids,
            "facts": self.facts,
        }


# ── Mode 1: build clusters from existing 2-fact pairs (fast, zero LLM) ────────
#
# NOTE: garbage-fact filtering (patent/CEN boilerplate, bibliography entries,
# watermarks, fragments, weak self-containment) used to live here as a set of
# regex band-aids. That was end-stage cleaning of garbage that should never have
# entered the graph. Those facts are now dropped at their source — Stage 4
# (filter_adaptive._relaxed_reject) — so by the time pairs reach this sampler the
# facts are already clean. The band-aids were removed; see PHASE3_ROOT_CAUSE_FIX.md.


def _is_same_page_pair(pair: dict) -> bool:
    facts = pair.get("facts", [])
    if len(facts) != 2:
        return False
    p0, p1 = _page(facts[0]), _page(facts[1])
    return p0 is not None and p1 is not None and p0 == p1


_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "be", "for",
    "with", "that", "this", "these", "those", "from", "as", "at", "by", "on",
    "it", "its", "not", "if", "so", "such", "can", "may", "shall", "should",
    "must", "will", "have", "has", "been", "was", "were", "also", "any", "all",
    "each", "both", "than", "then", "when", "where", "which", "while", "into",
    "used", "use", "made", "make", "between", "more", "less", "no", "two",
    "only", "order", "result", "value", "values", "method", "test", "tests",
    "measurement", "measurements", "standard", "accordance",
})


def _keyword_tokens(facts: list[dict], min_len: int = 5) -> set[str]:
    """Extract lowercase content words (≥min_len chars) from a list of facts."""
    tokens: set[str] = set()
    for f in facts:
        for tok in re.findall(r"[a-z]{%d,}" % min_len, _text(f).lower()):
            if tok not in _STOPWORDS:
                tokens.add(tok)
    return tokens


def _has_domain_bridge(facts_a: list[dict], facts_b: list[dict], min_shared: int = 1) -> bool:
    """Return True if clusters A and B share at least min_shared domain keywords.

    HEURISTIC — mode-1 only. Combining two same-page pairs from distant pages
    asserts a bridge between them that was never NLI-verified (unlike mode 2,
    ``sample_clusters_from_sfg``, where A<->B is a genuine graph edge). Shared
    vocabulary is a weak proxy: it removes the most obvious unrelated pairings
    but cannot guarantee a real semantic bridge. The principled fix for RC-7 is
    to source clusters from the live SFG (mode 2). See PHASE3_ROOT_CAUSE_FIX.md.
    """
    return len(_keyword_tokens(facts_a) & _keyword_tokens(facts_b)) >= min_shared


def sample_clusters_from_pairs(
    pairs: list[dict],
    n_clusters: int = 30,
    min_fact_words: int = 8,
    min_page_gap: int = 3,
    rng: Optional[random.Random] = None,
    max_attempts: int = 2000,
) -> list[ClusterChain]:
    """Build 4-fact clusters from existing NLI-verified 2-fact same-page pairs.

    Args:
        pairs: list of pair dicts from s7_pairs.json (or merged JSONL).
        n_clusters: how many clusters to return.
        min_fact_words: minimum words per fact (fragment filter).
        min_page_gap: minimum page distance between cluster A and B.
        rng: random state for reproducibility.
        max_attempts: cap on (pairA, pairB) combinations tried.

    Returns:
        List of ClusterChain objects.
    """
    if rng is None:
        rng = random.Random(42)

    # Filter to same-page pairs only. Facts are already domain-clean (Stage 4);
    # the only sampling-time constraint kept here is the fragment word-floor,
    # which is a structural property of the chain, not garbage cleaning.
    same_page = [
        p for p in pairs
        if _is_same_page_pair(p)
        and all(
            _word_count(_text(f)) >= min_fact_words
            for f in p.get("facts", [])
        )
    ]

    if len(same_page) < 2:
        return []

    # Group pairs by page.
    by_page: dict[int, list[dict]] = {}
    for p in same_page:
        pg = _page(p["facts"][0])
        if pg is not None:
            by_page.setdefault(pg, []).append(p)

    pages = sorted(by_page.keys())
    if len(pages) < 2:
        return []

    # Build candidate (pageA, pageB) combinations with sufficient gap.
    page_pairs = [
        (pA, pB)
        for i, pA in enumerate(pages)
        for pB in pages[i + 1:]
        if abs(pA - pB) >= min_page_gap
    ]

    if not page_pairs:
        return []

    rng.shuffle(page_pairs)
    clusters: list[ClusterChain] = []
    seen_combos: set[tuple] = set()
    attempts = 0

    for pA, pB in page_pairs * (max_attempts // max(1, len(page_pairs)) + 1):
        if len(clusters) >= n_clusters or attempts >= max_attempts:
            break
        attempts += 1

        pool_a = by_page.get(pA, [])
        pool_b = by_page.get(pB, [])
        if not pool_a or not pool_b:
            continue

        pair_a = rng.choice(pool_a)
        pair_b = rng.choice(pool_b)

        a1, a2 = pair_a["facts"][0], pair_a["facts"][1]
        b1, b2 = pair_b["facts"][0], pair_b["facts"][1]

        # Deduplicate: no repeated fact IDs across the cluster.
        ids = tuple(sorted([f.get("fact_id","") for f in [a1, a2, b1, b2]]))
        if ids in seen_combos:
            continue

        # No duplicate texts.
        texts = [_text(f) for f in [a1, a2, b1, b2]]
        if len(set(texts)) < 4:
            continue

        # Semantic bridge check: at least 1 shared domain keyword between A and B.
        if not _has_domain_bridge(pair_a["facts"], pair_b["facts"]):
            continue

        seen_combos.add(ids)
        clusters.append(ClusterChain(
            facts=[a1, a2, b1, b2],
            page_a=pA,
            page_b=pB,
            page_gap=abs(pA - pB),
            source_pair_a=pair_a,
            source_pair_b=pair_b,
        ))

    # Sort by largest page gap first (strictest multi-hop at the top).
    clusters.sort(key=lambda c: -c.page_gap)
    return clusters


# ── Mode 2: build clusters from a live TypedSFG (production pipeline) ─────────

def sample_clusters_from_sfg(
    sfg,                       # TypedSFG instance
    n_clusters: int = 30,
    min_fact_words: int = 8,
    min_page_gap: int = 3,
    rng: Optional[random.Random] = None,
    walk_attempts_multiplier: int = 20,
) -> list[ClusterChain]:
    """Walk depth-4 paths through the SFG, filter for the cluster pattern.

    The pattern required:
        facts[0].page == facts[1].page   (A-cluster same-page)
        facts[2].page == facts[3].page   (B-cluster same-page)
        |facts[0].page - facts[2].page| >= min_page_gap
        all facts >= min_fact_words words

    This produces *true* 3-hop depth-4 NLI-verified chains.
    Uses the existing walk_typed_paths machinery; no new NLI calls.
    """
    if rng is None:
        rng = random.Random(42)

    def _sfg_fact_page(fact_id: str) -> Optional[int]:
        fact = sfg.facts.get(fact_id)
        if fact is None:
            return None
        spans = getattr(fact, "supporting_spans", []) or []
        if spans:
            return getattr(spans[0], "page_start", None)
        return None

    def _sfg_fact_text(fact_id: str) -> str:
        fact = sfg.facts.get(fact_id)
        if fact is None:
            return ""
        return str(getattr(fact, "text", "") or getattr(fact, "canonical_form", ""))

    n_walk = n_clusters * walk_attempts_multiplier
    candidates = sfg.walk_typed_paths({4: n_walk}, rng)

    clusters: list[ClusterChain] = []
    seen: set[tuple] = set()

    for chain in candidates:
        if len(clusters) >= n_clusters:
            break
        if len(chain.fact_ids) != 4:
            continue

        fids = chain.fact_ids
        pages = [_sfg_fact_page(fid) for fid in fids]
        if any(p is None for p in pages):
            continue

        p0, p1, p2, p3 = pages
        # Cluster pattern: [same, same] [cross] [same, same]
        if p0 != p1:
            continue
        if p2 != p3:
            continue
        if abs(p0 - p2) < min_page_gap:
            continue

        texts = [_sfg_fact_text(fid) for fid in fids]
        if any(_word_count(t) < min_fact_words for t in texts):
            continue

        key = tuple(sorted(fids))
        if key in seen:
            continue
        seen.add(key)

        fact_dicts = []
        for fid, pg in zip(fids, pages):
            fact = sfg.facts.get(fid)
            fact_dicts.append({
                "fact_id": fid,
                "text": _sfg_fact_text(fid),
                "page_start": pg,
                "self_containment_score": getattr(fact, "self_containment_score", None),
            })

        clusters.append(ClusterChain(
            facts=fact_dicts,
            page_a=p0,
            page_b=p2,
            page_gap=abs(p0 - p2),
        ))

    clusters.sort(key=lambda c: -c.page_gap)
    return clusters
