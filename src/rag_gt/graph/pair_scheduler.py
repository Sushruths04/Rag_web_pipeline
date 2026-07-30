"""Shared candidate-pair scheduler for TF-SFG and pipeline diagnostics.

Both the runtime (TypedSFG._candidate_pairs) and the diagnostics tool
(_embedding_pair_candidates) must use identical scheduling semantics, otherwise
diagnostics over-report coverage and mislead tuning.

Key properties:
- Round-robin over source pages so every page-region gets a chance.
- Per-anchor cap: one source fact cannot dominate the budget.
- Per-page-pair cap: repeated pairs within the same page-pair combo are trimmed.
- Returns a SchedulerReport so callers can surface coverage stats in dashboards.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

from rag_gt.core.types import Fact
from rag_gt.vectorstore.faiss_index import FactIndex

_CANDIDATE_K = 30

_STOPWORDS = {
    "about", "above", "after", "again", "also", "because", "before", "between",
    "from", "have", "into", "more", "most", "only", "other", "same", "some",
    "such", "than", "that", "their", "then", "there", "these", "this", "those",
    "through", "under", "using", "very", "when", "where", "which", "while",
    "with", "would",
}

_ROLE_PAIR_BONUS = {
    # Role is a cheap pre-classification tie-breaker only. Verified relation
    # evidence, not a role label, must drive stronger downstream ranking.
    ("definition", "condition"): 0.08,
    ("definition", "rule"): 0.08,
    ("condition", "definition"): 0.05,
    ("condition", "consequence"): 0.08,
    ("rule", "exception"): 0.08,
    ("rule", "constraint"): 0.08,
    ("rule", "consequence"): 0.07,
    ("condition", "rule"): 0.06,
    ("definition", "consequence"): 0.05,
    ("definition", "example"): 0.03,
}
_CONDITION_CUE_RE = re.compile(
    r"\b(if|when|whenever|unless|assuming|provided|under|on tasks? with|in cases? with)\b",
    re.I,
)
_OUTCOME_CUE_RE = re.compile(
    r"\b(therefore|thus|accordingly|in return|converges?|guaranteed|"
    r"leads? to|results? in|requires?|achieves?|faster learning|identical|must)\b",
    re.I,
)
_SETUP_CUE_RE = re.compile(
    r"\b(defines?|represents?|gives?|estimates?|specifies?|is an? |are )\b",
    re.I,
)
_APPLICATION_CUE_RE = re.compile(
    r"\b(applies?|uses?|computed?|updated?|selected?|allocated?|determines?)\b",
    re.I,
)


@dataclass
class SchedulerReport:
    """Coverage stats for the pair selection process."""
    unique_source_facts: int = 0
    unique_source_pages: int = 0
    top20_anchor_share: float = 0.0
    selected_pairs_by_page_bucket: Dict[int, int] = field(default_factory=dict)
    anchors_seen: int = 0
    raw_candidate_count: int = 0
    selected_count: int = 0
    prefilter_stats: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "unique_source_facts": self.unique_source_facts,
            "unique_source_pages": self.unique_source_pages,
            "top20_anchor_share": round(self.top20_anchor_share, 4),
            "selected_pairs_by_page_bucket": self.selected_pairs_by_page_bucket,
            "anchors_seen": self.anchors_seen,
            "raw_candidate_count": self.raw_candidate_count,
            "selected_count": self.selected_count,
            "prefilter_stats": self.prefilter_stats,
        }


def _fact_page(fact: Fact) -> Optional[int]:
    for span in fact.supporting_spans:
        if span.page_start is not None:
            return int(span.page_start)
    return None


def _fact_char_start(fact: Fact) -> int:
    starts = [
        int(span.char_start)
        for span in fact.supporting_spans
        if span.char_start is not None
    ]
    return min(starts) if starts else 10**12


def _page_key(fact: Fact) -> int:
    page = _fact_page(fact)
    return page if page is not None else -1


def _source_position(fact: Fact) -> tuple[int, int, str]:
    page = _fact_page(fact)
    return (
        page if page is not None else 10**12,
        _fact_char_start(fact),
        fact.fact_id,
    )


def _canonical_source_pair(fact_a: Fact, fact_b: Fact) -> tuple[Fact, Fact]:
    """Order a same-document candidate as it appears in the source."""
    if _source_position(fact_b) < _source_position(fact_a):
        return fact_b, fact_a
    return fact_a, fact_b


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _default_role_bonus(a: Fact, b: Fact) -> float:
    pair = (str(a.role), str(b.role))
    if pair in _ROLE_PAIR_BONUS:
        return _ROLE_PAIR_BONUS[pair]
    if a.role != b.role:
        return 0.03
    return 0.0


def _default_lexical_bonus(a: Fact, b: Fact) -> float:
    ta = _tokens(a.text)
    tb = _tokens(b.text)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
    return min(0.08, overlap * 0.08)


def _default_directional_composition_bonus(a: Fact, b: Fact) -> float:
    """Promote source-ordered pairs that express a usable dependency shape."""
    text_a = a.text or ""
    text_b = b.text or ""
    if _CONDITION_CUE_RE.search(text_a) and _OUTCOME_CUE_RE.search(text_b):
        return 0.10
    if _SETUP_CUE_RE.search(text_a) and (
        _CONDITION_CUE_RE.search(text_b) or _APPLICATION_CUE_RE.search(text_b)
    ):
        return 0.07
    return 0.0


def _collect_neighbours_for_anchor(
    fid: str,
    facts_by_id: Dict[str, Fact],
    index: FactIndex,
    *,
    min_cosine: float,
    candidate_k: int,
    seen_pairs: set,
    scored: list,
    raw_candidate_pairs: int,
    role_bonus_fn: Callable,
    lexical_bonus_fn: Callable,
    directional_composition_bonus_fn: Callable,
    canonical_source_order: bool,
) -> None:
    try:
        emb = index.embedding_for(fid)
    except KeyError:
        return
    fact_a = facts_by_id[fid]
    neighbours = index.search(emb, k=candidate_k)
    for nbr_id, cos in neighbours:
        if nbr_id == fid or nbr_id not in facts_by_id:
            continue
        if float(cos) < min_cosine:
            continue
        fact_b = facts_by_id[nbr_id]
        if canonical_source_order:
            fact_a, fact_b = _canonical_source_pair(fact_a, fact_b)
        pair_key = (fact_a.fact_id, fact_b.fact_id)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        score = (
            float(cos)
            + role_bonus_fn(fact_a, fact_b)
            + lexical_bonus_fn(fact_a, fact_b)
            + directional_composition_bonus_fn(fact_a, fact_b)
        )
        scored.append((score, len(scored), (fact_a, fact_b)))
        if len(scored) >= raw_candidate_pairs:
            return


def _collect_source_neighbours(
    facts: List[Fact],
    *,
    window: int,
    max_page_gap: int,
    seen_pairs: set,
    scored: list,
    role_bonus_fn: Callable,
    lexical_bonus_fn: Callable,
    directional_composition_bonus_fn: Callable,
    canonical_source_order: bool,
) -> int:
    if window <= 0:
        return 0
    ordered = sorted(facts, key=lambda f: (_page_key(f), _fact_char_start(f), f.fact_id))
    added = 0
    for i, fact_a in enumerate(ordered):
        page_a = _fact_page(fact_a)
        for fact_b in ordered[i + 1 : i + 1 + window]:
            page_b = _fact_page(fact_b)
            if page_a is not None and page_b is not None and abs(page_a - page_b) > max_page_gap:
                continue
            oriented_pairs = (
                ((fact_a, fact_b),)
                if canonical_source_order
                else ((fact_a, fact_b), (fact_b, fact_a))
            )
            for src, dst in oriented_pairs:
                pair_key = (src.fact_id, dst.fact_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                same_page_bonus = 0.10 if page_a is not None and page_a == page_b else 0.0
                # Base score for a purely reading-order-adjacent pair. Kept below
                # the semantic-cosine band (~0.45-0.65) so plain adjacency does not
                # crowd out semantically related pairs; a genuine dependency shape
                # (condition->outcome, setup->application) still lifts it via the
                # directional-composition bonus.
                score = (
                    0.60
                    + same_page_bonus
                    + role_bonus_fn(src, dst)
                    + lexical_bonus_fn(src, dst)
                    + directional_composition_bonus_fn(src, dst)
                )
                scored.append((score, len(scored), (src, dst)))
                added += 1
    return added


def schedule_candidate_pairs(
    facts: List[Fact],
    index: FactIndex,
    *,
    min_cosine: float = 0.45,
    candidate_k: int = _CANDIDATE_K,
    raw_candidate_pairs: int = 5000,
    max_pairs: int = 800,
    per_page_pair_cap: int = 24,
    per_anchor_pair_cap: int = 6,
    role_bonus_fn: Optional[Callable] = None,
    lexical_bonus_fn: Optional[Callable] = None,
    directional_composition_bonus_fn: Optional[Callable] = None,
    pair_prefilter_fn: Optional[Callable[[Fact, Fact, float], Tuple[Optional[str], float]]] = None,
    source_neighbor_window: int = 0,
    source_neighbor_max_page_gap: int = 2,
    canonical_source_order: bool = True,
) -> Tuple[List[Tuple[Fact, Fact, float]], SchedulerReport]:
    """Schedule candidate fact pairs for edge classification.

    Round-robins over source pages so every book region gets representation,
    then applies per-anchor and per-page-pair caps during final selection.

    Returns:
        (pairs, report) — pairs is a list of (fact_a, fact_b, score) triples;
        report contains coverage statistics for dashboard display.
    """
    if role_bonus_fn is None:
        role_bonus_fn = _default_role_bonus
    if lexical_bonus_fn is None:
        lexical_bonus_fn = _default_lexical_bonus
    if directional_composition_bonus_fn is None:
        directional_composition_bonus_fn = _default_directional_composition_bonus

    facts_by_id: Dict[str, Fact] = {f.fact_id: f for f in facts}
    seen_pairs: set = set()
    scored: List[Tuple[float, int, Tuple[Fact, Fact]]] = []

    pages: Dict[int, List[str]] = defaultdict(list)
    no_page: List[str] = []
    for fact in facts:
        page = _fact_page(fact)
        if page is None:
            no_page.append(fact.fact_id)
        else:
            pages[page].append(fact.fact_id)

    page_order = sorted(pages)
    cursors = {p: 0 for p in page_order}
    anchors_seen = 0

    # Phase 1: round-robin by page to gather raw candidates
    while len(scored) < raw_candidate_pairs:
        progressed = False
        for page in page_order:
            ids = pages[page]
            idx = cursors[page]
            if idx >= len(ids):
                continue
            cursors[page] += 1
            progressed = True
            anchors_seen += 1
            _collect_neighbours_for_anchor(
                ids[idx], facts_by_id, index,
                min_cosine=min_cosine,
                candidate_k=candidate_k,
                seen_pairs=seen_pairs,
                scored=scored,
                raw_candidate_pairs=raw_candidate_pairs,
                role_bonus_fn=role_bonus_fn,
                lexical_bonus_fn=lexical_bonus_fn,
                directional_composition_bonus_fn=directional_composition_bonus_fn,
                canonical_source_order=canonical_source_order,
            )
            if len(scored) >= raw_candidate_pairs:
                break
        if not progressed:
            break

    for fid in no_page:
        _collect_neighbours_for_anchor(
            fid, facts_by_id, index,
            min_cosine=min_cosine,
            candidate_k=candidate_k,
            seen_pairs=seen_pairs,
            scored=scored,
            raw_candidate_pairs=raw_candidate_pairs,
            role_bonus_fn=role_bonus_fn,
            lexical_bonus_fn=lexical_bonus_fn,
            directional_composition_bonus_fn=directional_composition_bonus_fn,
            canonical_source_order=canonical_source_order,
        )
        if len(scored) >= raw_candidate_pairs:
            break

    source_candidate_count = _collect_source_neighbours(
        facts,
        window=source_neighbor_window,
        max_page_gap=source_neighbor_max_page_gap,
        seen_pairs=seen_pairs,
        scored=scored,
        role_bonus_fn=role_bonus_fn,
        lexical_bonus_fn=lexical_bonus_fn,
        directional_composition_bonus_fn=directional_composition_bonus_fn,
        canonical_source_order=canonical_source_order,
    )

    raw_candidate_count = len(scored)
    prefilter_stats = {
        "input": raw_candidate_count,
        "hard_rejected": {},
        "penalized": 0,
        "passed": len(scored),
        "source_neighbor_candidates": source_candidate_count,
    }
    if pair_prefilter_fn is not None:
        filtered: List[Tuple[float, int, Tuple[Fact, Fact]]] = []
        hard_rejected: Dict[str, int] = defaultdict(int)
        penalized = 0
        for score, ordinal, (fact_a, fact_b) in scored:
            reject_reason, penalty = pair_prefilter_fn(fact_a, fact_b, score)
            if reject_reason is not None:
                hard_rejected[reject_reason] += 1
                continue
            adjusted = max(0.0, score - float(penalty or 0.0))
            if penalty > 0.0:
                penalized += 1
            filtered.append((adjusted, ordinal, (fact_a, fact_b)))
        prefilter_stats = {
            "input": raw_candidate_count,
            "hard_rejected": dict(hard_rejected),
            "penalized": penalized,
            "passed": len(filtered),
            "source_neighbor_candidates": source_candidate_count,
        }
        scored = filtered

    scored.sort(key=lambda x: x[0], reverse=True)

    # Phase 2: select with per-anchor and per-page-pair caps
    selected: List[Tuple[Fact, Fact, float]] = []
    page_pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    anchor_counts: Counter[str] = Counter()

    for score, _ordinal, (fact_a, fact_b) in scored:
        page_pair_key = (_page_key(fact_a), _page_key(fact_b))
        if page_pair_counts[page_pair_key] >= per_page_pair_cap:
            continue
        if anchor_counts[fact_a.fact_id] >= per_anchor_pair_cap:
            continue
        selected.append((fact_a, fact_b, score))
        page_pair_counts[page_pair_key] += 1
        anchor_counts[fact_a.fact_id] += 1
        if len(selected) >= max_pairs:
            break

    # Build report
    unique_src_facts = set(a.fact_id for a, _b, _s in selected)
    unique_src_pages = set(_page_key(a) for a, _b, _s in selected)
    unique_src_pages.discard(-1)

    anchor_pair_counts = Counter(a.fact_id for a, _b, _s in selected)
    top20 = sum(v for _, v in anchor_pair_counts.most_common(20))
    top20_share = top20 / max(1, len(selected))

    bucket_counts: Dict[int, int] = defaultdict(int)
    for fact_a, _fact_b, _score in selected:
        page = _fact_page(fact_a)
        bucket = (page // 10) if page is not None else -1
        bucket_counts[bucket] += 1

    report = SchedulerReport(
        unique_source_facts=len(unique_src_facts),
        unique_source_pages=len(unique_src_pages),
        top20_anchor_share=top20_share,
        selected_pairs_by_page_bucket=dict(sorted(bucket_counts.items())),
        anchors_seen=anchors_seen,
        raw_candidate_count=raw_candidate_count,
        selected_count=len(selected),
        prefilter_stats=prefilter_stats,
    )

    logger.info(
        f"[PairScheduler] raw={len(scored)} selected={len(selected)} "
        f"unique_anchors={len(unique_src_facts)} top20_share={top20_share:.2%} "
        f"pages={len(unique_src_pages)}"
    )
    return selected, report
