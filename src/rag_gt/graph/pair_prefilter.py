"""L0 pair pre-filter for V16.2 — runs before TF-SFG LLM classify.

Design philosophy: mostly downrank/penalty; few hard rejects. Hard rejects only
fire when a combo of signals makes the pair provably worthless. Ambiguous pairs
are penalised (score reduced) so the per-page-pair-cap removes them implicitly.

Integrated into typed_sfg.TypedSFG._candidate_pairs after raw scoring and
before LLM classify. The per-page-pair-cap remains and runs after this filter.

See docs/V16_2_FILTER_PLAN_20260519.md §4.2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from rag_gt.core.types import Fact
from rag_gt.facts.domain_filter import fact_domain_reject_reason


@dataclass
class PairPrefilterConfig:
    enabled: bool = True
    min_chars: int = 30
    min_bridge_tokens: int = 2
    bridge_cosine_min: float = 0.55
    max_page_gap_unaided: int = 30
    near_duplicate_overlap_cap: float = 0.75
    same_role_overlap_floor: float = 0.45
    penalty_same_role_high_overlap: float = 0.20
    same_page_role_overlap_cap: float = 0.70
    penalty_same_page_high_overlap: float = 0.25
    penalty_page_gap_outlier_strong_bridge: float = 0.10
    penalty_identical_prefix: float = 0.05
    penalty_definition_definition_weak_anchor: float = 0.15
    penalty_source_fragment_role: float = 0.20
    penalty_deictic_unresolved: float = 0.10
    penalty_weak_role_pair: float = 0.12
    penalty_math_fragment_pair: float = 0.18

    @classmethod
    def from_dict(cls, raw: dict | None) -> "PairPrefilterConfig":
        raw = raw or {}
        return cls(
            enabled=bool(raw.get("enabled", True)),
            min_chars=int(raw.get("min_chars", 30)),
            min_bridge_tokens=int(raw.get("min_bridge_tokens", 2)),
            bridge_cosine_min=float(raw.get("bridge_cosine_min", 0.55)),
            max_page_gap_unaided=int(raw.get("max_page_gap_unaided", 30)),
            near_duplicate_overlap_cap=float(
                raw.get("near_duplicate_overlap_cap", 0.75)
            ),
            same_role_overlap_floor=float(raw.get("same_role_overlap_floor", 0.45)),
            penalty_same_role_high_overlap=float(
                raw.get("penalty_same_role_high_overlap", 0.20)
            ),
            same_page_role_overlap_cap=float(raw.get("same_page_role_overlap_cap", 0.70)),
            penalty_same_page_high_overlap=float(raw.get("penalty_same_page_high_overlap", 0.25)),
            penalty_page_gap_outlier_strong_bridge=float(
                raw.get("penalty_page_gap_outlier_strong_bridge", 0.10)
            ),
            penalty_identical_prefix=float(raw.get("penalty_identical_prefix", 0.05)),
            penalty_definition_definition_weak_anchor=float(
                raw.get("penalty_definition_definition_weak_anchor", 0.15)
            ),
            penalty_source_fragment_role=float(
                raw.get("penalty_source_fragment_role", 0.20)
            ),
            penalty_deictic_unresolved=float(
                raw.get("penalty_deictic_unresolved", 0.10)
            ),
            penalty_weak_role_pair=float(raw.get("penalty_weak_role_pair", 0.12)),
            penalty_math_fragment_pair=float(
                raw.get("penalty_math_fragment_pair", 0.18)
            ),
        )


_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
    "with", "this", "that", "it", "its", "by", "from", "as",
})


def _bridge_tokens(a: Fact, b: Fact) -> int:
    """Shared non-stopword tokens (≥ 4 chars) between two facts."""
    def tok(text: str) -> set[str]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", text.lower())
        return {w for w in words if w not in _STOPWORDS}
    return len(tok(a.text) & tok(b.text))


def _fact_page(fact: Fact) -> Optional[int]:
    for span in fact.supporting_spans:
        if span.page_start is not None:
            return int(span.page_start)
    return None


def _lexical_overlap_fraction(a: Fact, b: Fact) -> float:
    def tok(text: str) -> set[str]:
        words = re.findall(r"[A-Za-z0-9]+", text.lower())
        return {w for w in words if w not in _STOPWORDS and len(w) > 2}
    ta, tb = tok(a.text), tok(b.text)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(min(len(ta), len(tb)), 1)


def _first_n_tokens(text: str, n: int = 10) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z0-9]+", text.lower())[:n])


def _shared_prefix_len(a: str, b: str, n: int = 10) -> int:
    count = 0
    for ta, tb in zip(_first_n_tokens(a, n), _first_n_tokens(b, n)):
        if ta != tb:
            break
        count += 1
    return count


_WEAK_SOURCE_ROLES = {
    "caption",
    "figure_ref",
    "figure",
    "table_cell",
    "table",
    "heading",
}
_WEAK_ROLE_PAIRS = {
    ("definition", "example"),
    ("example", "definition"),
    ("example", "example"),
    ("definition", "definition"),
}
_DEICTIC_START_RE = re.compile(
    r"^\s*(this|these|that|those|it|its|such|they|their)\b",
    re.I,
)
_ANCHOR_RE = re.compile(
    r"\b("
    r"[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*|"
    r"[A-Za-z]+\([^)]{1,12}\)|"
    r"[A-Za-z]*[πλγτ][A-Za-z0-9_()]*|"
    r"[A-Za-z]+(?:-[A-Za-z]+)+"
    r")\b"
)
_MATH_FRAGMENT_RE = re.compile(
    r"(?:[∑∏πλγτ∗∞≈≤≥¯]|\b(?:lim|frac|sum)\b|[A-Za-z]\s*[=<>]\s*[A-Za-z0-9])"
)
_LEADING_EQUATION_RE = re.compile(
    r"^\s*(?:[A-Za-z]\s*,|\(?\d+(?:\.\d+)?\)?|[A-Za-z]\s*[=<>]|[∑∏πλγτ∗∞≈≤≥¯])"
)
_RESTATEMENT_RE = re.compile(
    r"\b(?:another way of saying this|in other words|equivalently|is also called)\b",
    re.I,
)


def _anchor_terms(text: str) -> set[str]:
    anchors = {
        term
        for term in (m.group(1).lower() for m in _ANCHOR_RE.finditer(text or ""))
        if len(term) > 1 and term not in {"an", "the"}
    }
    # Keep multi-word technical anchors that survive stopword filtering.
    words = [
        w for w in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", (text or "").lower())
        if w not in _STOPWORDS
    ]
    for a, b in zip(words, words[1:]):
        anchors.add(f"{a} {b}")
    return anchors


def _has_shared_anchor(a: Fact, b: Fact) -> bool:
    return bool(_anchor_terms(a.text) & _anchor_terms(b.text))


def _starts_deictic_without_bridge(fact: Fact, other: Fact) -> bool:
    text = fact.text or ""
    if not _DEICTIC_START_RE.search(text):
        return False
    # A strong shared anchor usually resolves "this/these" across adjacent facts.
    return not _has_shared_anchor(fact, other)


def _math_fragment_risk(text: str) -> bool:
    """Detect equation-heavy fragments that rarely survive QA-NLI/QGen.

    This is only a downrank signal. Textbook math is valid evidence, but short
    formula fragments currently produce weak relation claims and brittle
    assumption grounding.
    """
    text = text or ""
    compact = " ".join(text.split())
    if not compact:
        return False
    math_hits = len(_MATH_FRAGMENT_RE.findall(compact))
    symbol_chars = sum(1 for ch in compact if ch in "=<>∑∏πλγτ∗∞≈≤≥¯")
    symbol_ratio = symbol_chars / max(len(compact), 1)
    return (
        _LEADING_EQUATION_RE.search(compact) is not None
        or math_hits >= 3
        or symbol_ratio >= 0.035
    )


def prefilter_pair(
    fact_a: Fact,
    fact_b: Fact,
    cosine: float,
    cfg: PairPrefilterConfig,
) -> tuple[str | None, float]:
    """Evaluate one pair; return (reject_reason | None, score_penalty).

    reject_reason non-None means hard reject (pair is dropped entirely).
    score_penalty is subtracted from the raw cosine score (0 if no penalty).
    """
    text_a = (fact_a.canonical_form or fact_a.text or "").strip()
    text_b = (fact_b.canonical_form or fact_b.text or "").strip()

    # Hard reject 1: either fact is below the minimum meaningful length.
    if len(text_a) < cfg.min_chars or len(text_b) < cfg.min_chars:
        return "short_fact", 0.0

    # Hard reject 2: either fact is a fragment (section header, caption, etc.).
    if fact_domain_reject_reason(fact_a) is not None:
        return "fragment_pair", 0.0
    if fact_domain_reject_reason(fact_b) is not None:
        return "fragment_pair", 0.0

    overlap = _lexical_overlap_fraction(fact_a, fact_b)
    # Near-restatements cannot contribute two separately necessary supports.
    # Drop before classification instead of asking the LLM to discover this.
    if overlap >= cfg.near_duplicate_overlap_cap:
        return "near_duplicate_fact_text", 0.0
    if overlap >= 0.45 and (
        _RESTATEMENT_RE.search(text_a) or _RESTATEMENT_RE.search(text_b)
    ):
        return "explicit_restatement_pair", 0.0

    bridge_tok = _bridge_tokens(fact_a, fact_b)
    weak_bridge_tokens = bridge_tok < cfg.min_bridge_tokens
    weak_cosine = cosine < cfg.bridge_cosine_min

    page_a = _fact_page(fact_a)
    page_b = _fact_page(fact_b)
    page_gap = (
        abs(page_a - page_b)
        if page_a is not None and page_b is not None
        else 0
    )

    # Hard reject 4: large page gap AND both bridge signals are weak.
    # Page gap alone is never a hard reject (could be a cross-section multi-hop).
    if page_gap > cfg.max_page_gap_unaided and weak_bridge_tokens and weak_cosine:
        return "page_gap_outlier_combined", 0.0

    # Hard reject 3: no lexical bridge AND no cosine similarity — can't form a
    # meaningful question requiring both facts.
    if weak_bridge_tokens and weak_cosine:
        return "weak_bridge", 0.0

    # --- Penalties (never reject; reduce score so cap eats them) ---
    penalty = 0.0

    same_page = page_a is not None and page_a == page_b
    same_role = str(fact_a.role) == str(fact_b.role)
    # Similar facts playing the same discourse role are commonly alternate
    # formulations of one fact. Downrank them without erasing all candidates.
    if same_role and overlap >= cfg.same_role_overlap_floor:
        penalty += cfg.penalty_same_role_high_overlap

    # Same page + same role + high lexical overlap → likely duplicate or echo.
    # Non-duplicate topical echoes remain available but rank below cleaner pairs.
    if same_page and same_role and overlap > cfg.same_page_role_overlap_cap:
        penalty += cfg.penalty_same_page_high_overlap

    # Large page gap with a strong bridge — valid cross-section multi-hop but
    # slightly risky; apply a small penalty so better-bridged pairs rank first.
    if page_gap > cfg.max_page_gap_unaided and not (weak_bridge_tokens and weak_cosine):
        penalty += cfg.penalty_page_gap_outlier_strong_bridge

    # Identical fact prefix — likely the same fact text split across chunks.
    if _shared_prefix_len(fact_a.text, fact_b.text) >= 3:
        penalty += cfg.penalty_identical_prefix

    role_a = str(fact_a.role or "")
    role_b = str(fact_b.role or "")

    # Definition-definition pairs often restate a concept instead of composing a
    # relation. Keep them available when they share a concrete anchor symbol or
    # named technical object, but downrank the weak-anchor cases.
    if role_a == "definition" and role_b == "definition" and not _has_shared_anchor(fact_a, fact_b):
        penalty += cfg.penalty_definition_definition_weak_anchor

    if (role_a, role_b) in _WEAK_ROLE_PAIRS and not _has_shared_anchor(fact_a, fact_b):
        penalty += cfg.penalty_weak_role_pair

    if role_a in _WEAK_SOURCE_ROLES or role_b in _WEAK_SOURCE_ROLES:
        penalty += cfg.penalty_source_fragment_role

    if _starts_deictic_without_bridge(fact_a, fact_b) or _starts_deictic_without_bridge(fact_b, fact_a):
        penalty += cfg.penalty_deictic_unresolved

    if _math_fragment_risk(text_a) or _math_fragment_risk(text_b):
        penalty += cfg.penalty_math_fragment_pair

    return None, penalty


@dataclass
class _PrefilterStats:
    input_pairs: int = 0
    hard_rejected: dict = field(default_factory=dict)
    penalized: int = 0
    passed: int = 0


def prefilter_pairs(
    pairs_with_scores: list[tuple[Fact, Fact, float]],
    cfg: PairPrefilterConfig,
) -> tuple[list[tuple[Fact, Fact, float]], dict]:
    """Apply L0 pre-filter to (fact_a, fact_b, cosine) triples.

    Returns:
        kept: list of (fact_a, fact_b, adjusted_score) with penalties applied.
        stats: dict for build_summary.json:cascade_stats.L0.
    """
    if not cfg.enabled:
        return pairs_with_scores, {
            "input": len(pairs_with_scores),
            "hard_rejected": {},
            "penalized": 0,
            "passed": len(pairs_with_scores),
        }

    stats = _PrefilterStats(input_pairs=len(pairs_with_scores))
    kept: list[tuple[Fact, Fact, float]] = []

    for fact_a, fact_b, cosine in pairs_with_scores:
        reject_reason, penalty = prefilter_pair(fact_a, fact_b, cosine, cfg)
        if reject_reason is not None:
            stats.hard_rejected[reject_reason] = (
                stats.hard_rejected.get(reject_reason, 0) + 1
            )
            continue
        if penalty > 0.0:
            stats.penalized += 1
        kept.append((fact_a, fact_b, max(0.0, cosine - penalty)))

    stats.passed = len(kept)
    return kept, {
        "input": stats.input_pairs,
        "hard_rejected": dict(stats.hard_rejected),
        "penalized": stats.penalized,
        "passed": stats.passed,
    }
