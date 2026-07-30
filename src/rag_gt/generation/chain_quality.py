"""Heuristic quality gate for fact chains before question generation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from rag_gt.core.types import Fact
from rag_gt.facts.domain_filter import fact_domain_reject_reason


STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "because", "before",
    "being", "between", "could", "does", "done", "each", "even", "from",
    "have", "into", "itself", "more", "most", "only", "other", "over",
    "same", "some", "such", "than", "that", "their", "then", "there",
    "these", "this", "those", "through", "under", "using", "very", "when",
    "where", "which", "while", "with", "would",
}
WEAK_CONTEXT_START_RE = re.compile(
    r"^\s*(suppose we|the next choice|finally,|just how long|"
    r"waiting for an elevator|it is worth noting|the greatest problem)\b",
    re.I,
)


@dataclass(frozen=True)
class ChainQualityVerdict:
    accepted: bool
    reason: str = "accepted"


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text or "").encode(
        "ascii", "ignore"
    ).decode("ascii")
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{3,}", normalized.lower())
    return {w for w in words if w not in STOPWORDS}


def _pages(facts: Iterable[Fact]) -> list[int]:
    out: list[int] = []
    for fact in facts:
        for span in fact.supporting_spans:
            if span.page_start is not None:
                out.append(int(span.page_start))
                break
    return out


def chain_quality_gate(
    facts: list[Fact],
    *,
    max_page_gap: int = 40,
    min_shared_terms: int = 1,
) -> ChainQualityVerdict:
    if not facts:
        return ChainQualityVerdict(False, "empty_chain")

    ids = [f.fact_id for f in facts]
    if len(ids) != len(set(ids)):
        return ChainQualityVerdict(False, "duplicate_fact_loop")

    normalized_texts = [" ".join((f.text or "").lower().split()) for f in facts]
    if len(normalized_texts) != len(set(normalized_texts)):
        return ChainQualityVerdict(False, "duplicate_fact_text")

    for fact in facts:
        reason = fact_domain_reject_reason(fact)
        if reason:
            return ChainQualityVerdict(False, f"source_artifact:{reason}")
        if WEAK_CONTEXT_START_RE.search(fact.text or ""):
            return ChainQualityVerdict(False, "weak_context_fragment")

    if len(facts) == 1:
        return ChainQualityVerdict(True)

    pages = _pages(facts)
    if len(pages) >= 2 and (max(pages) - min(pages)) > max_page_gap:
        return ChainQualityVerdict(False, "source_gap_too_large")

    token_sets = [_tokens(f.text) for f in facts]
    if any(not ts for ts in token_sets):
        return ChainQualityVerdict(False, "empty_fact_terms")

    for idx, terms in enumerate(token_sets):
        connected = False
        for j, other in enumerate(token_sets):
            if idx == j:
                continue
            if len(terms & other) >= min_shared_terms:
                connected = True
                break
        if not connected:
            return ChainQualityVerdict(False, "unconnected_fact")

    return ChainQualityVerdict(True)
