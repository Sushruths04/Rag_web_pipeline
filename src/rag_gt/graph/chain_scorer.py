"""V16.2 L1 chain scorer and ranker.

Pass 1 scores every candidate chain with cheap structural signals. Pass 2 runs
local NLI only on the top bounded subset, then blends that compatibility score
back into the final rank.

No model is loaded by this module unless pass-2 reranking is actually invoked;
it reuses rag_gt.validation.nli_check.nli_batch.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping

from rag_gt.core.types import Fact, FactChain
from rag_gt.facts.domain_filter import fact_domain_reject_reason
from rag_gt.graph.edge_canonicalize import canonicalize_label, is_typed
from rag_gt.pipeline.yield_controller import ChainCategory
from rag_gt.validation import nli_check


_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
    "with", "this", "that", "it", "its", "by", "from", "as", "into",
    "than", "then", "these", "those", "which", "while", "where", "when",
})
_MATH_FRAGMENT_RE = re.compile(
    r"(?:[∑∏πλγτ∗∞≈≤≥¯]|\b(?:lim|frac|sum)\b|[A-Za-z]\s*[=<>]\s*[A-Za-z0-9])"
)
_LEADING_EQUATION_RE = re.compile(
    r"^\s*(?:[A-Za-z]\s*,|\(?\d+(?:\.\d+)?\)?|[A-Za-z]\s*[=<>]|[∑∏πλγτ∗∞≈≤≥¯])"
)
_ACTION_VERB_RE = re.compile(
    r"\b("
    r"causes?|leads?|requires?|depends?|enables?|prevents?|limits?|changes?|"
    r"updates?|chooses?|observes?|approximates?|estimates?|improves?|reduces?|"
    r"produces?|results?|follows?|affects?|determines?|controls?|uses?|applies?|"
    r"turns?|assigns?|receives?|selects?|maximi[sz]es?|minimi[sz]es?"
    r")\b",
    re.I,
)
_UNRESOLVED_START_RE = re.compile(
    r"^\s*(?:this|these|that|those|it|its|such|they|their|in these cases|accordingly)\b",
    re.I,
)

_ACTIONABLE_RELATION_PRIOR = {
    "mechanism": 1.00,
    "causal": 0.96,
    "procedural": 0.92,
    "condition": 0.86,
    "consequence": 0.86,
    "temporal": 0.82,
    "quantitative": 0.78,
    "contrast": 0.76,
    "comparison": 0.74,
    "intersection": 0.68,
    "rule": 0.62,
    "exception": 0.60,
    "constraint": 0.60,
    "definition": 0.26,
    "descriptive": 0.22,
    "list": 0.14,
}

_ACTIONABLE_ROLE_PAIR_PRIOR = {
    ("definition", "condition"): 0.86,
    ("condition", "definition"): 0.80,
    ("definition", "rule"): 0.72,
    ("rule", "definition"): 0.64,
    ("condition", "rule"): 0.82,
    ("rule", "condition"): 0.78,
    ("condition", "condition"): 0.74,
    ("mechanism", "condition"): 0.84,
    ("condition", "mechanism"): 0.84,
    ("definition", "mechanism"): 0.78,
    ("mechanism", "definition"): 0.72,
    ("definition", "definition"): 0.20,
    ("definition", "example"): 0.28,
    ("example", "definition"): 0.24,
    ("example", "example"): 0.18,
}

_QUESTIONABLE_ROLE_PAIR_PRIOR = {
    ("definition", "condition"): 1.00,
    ("definition", "rule"): 0.86,
    ("condition", "definition"): 0.74,
    ("rule", "definition"): 0.70,
    ("condition", "consequence"): 0.76,
    ("definition", "consequence"): 0.68,
    ("definition", "example"): 0.58,
    ("example", "definition"): 0.54,
    ("condition", "rule"): 0.72,
    ("rule", "condition"): 0.70,
    ("definition", "definition"): 0.42,
}

_CONDITION_CUE_RE = re.compile(
    r"\b(if|when|whenever|unless|provided|only if|in cases?|under|once)\b",
    re.I,
)
_SETUP_CUE_RE = re.compile(
    r"\b(is|are|means|defines?|represents?|refers? to|to use|once one has|we know)\b",
    re.I,
)


@dataclass(frozen=True)
class ChainScorerConfig:
    enabled: bool = True
    nli_rerank_k: int = 80
    min_pass1_score: float = 0.40
    pass1_weights: Mapping[str, float] = field(default_factory=lambda: {
        "length": 0.06,
        "fragment": 0.10,
        "page_compactness": 0.05,
        "role_variety": 0.08,
        "lexical_bridge": 0.10,
        "typed_edge_confidence": 0.16,
        "edge_joint_only_margin": 0.25,
        "answer_contribution_score": 0.30,
        "relation_type_strength": 0.20,
        "source_text_quality": 0.20,
        "compositional_actionability": 0.0,
        "multi_hop_questionability": 0.18,
    })
    pass2_blend: Mapping[str, float] = field(default_factory=lambda: {
        "pass1": 0.85,
        "nli_pair_compat": 0.15,
    })
    per_group_keep: int = 3
    min_final_score_to_keep: float = 0.45
    max_page_gap: int = 40
    relation_type_strength: Mapping[str, float] = field(default_factory=lambda: {
        "mechanism": 1.00,
        "causal": 0.95,
        "procedural": 0.90,
        "temporal": 0.82,
        "quantitative": 0.80,
        "contrast": 0.78,
        "comparison": 0.76,
        "intersection": 0.72,
        "condition": 0.70,
        "consequence": 0.70,
        "rule": 0.68,
        "exception": 0.66,
        "constraint": 0.64,
        "definition": 0.38,
        "descriptive": 0.32,
        "list": 0.20,
    })

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "ChainScorerConfig":
        raw = raw or {}
        return cls(
            enabled=bool(raw.get("enabled", True)),
            nli_rerank_k=int(raw.get("nli_rerank_k", 80)),
            min_pass1_score=float(raw.get("min_pass1_score", 0.40)),
            pass1_weights=dict(raw.get("pass1_weights", {}) or cls().pass1_weights),
            pass2_blend=dict(raw.get("pass2_blend", {}) or cls().pass2_blend),
            per_group_keep=int(raw.get("per_group_keep", 3)),
            min_final_score_to_keep=float(raw.get("min_final_score_to_keep", 0.45)),
            max_page_gap=int(raw.get("max_page_gap", 40)),
            relation_type_strength=dict(
                raw.get("relation_type_strength", {}) or cls().relation_type_strength
            ),
        )


@dataclass(frozen=True)
class ChainScore:
    chain: FactChain
    pass1_score: float
    final_score: float
    category: ChainCategory
    signals: dict[str, float]
    pass2_reranked: bool = False

    def to_dict(self) -> dict:
        return {
            "fact_ids": list(self.chain.fact_ids),
            "pass1_score": self.pass1_score,
            "final_score": self.final_score,
            "category": self.category,
            "signals": dict(self.signals),
            "pass2_reranked": self.pass2_reranked,
        }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _fact_text(fact: Fact) -> str:
    return (fact.canonical_form.strip() or fact.text.strip())


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _lexical_overlap(a: Fact, b: Fact) -> float:
    ta = _tokens(_fact_text(a))
    tb = _tokens(_fact_text(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(min(len(ta), len(tb)), 1)


def _chain_facts(chain: FactChain, facts_by_id: Mapping[str, Fact]) -> list[Fact]:
    return [facts_by_id[fid] for fid in chain.fact_ids if fid in facts_by_id]


def _math_fragment_risk(text: str) -> float:
    text = " ".join((text or "").split())
    if not text:
        return 1.0
    math_hits = len(_MATH_FRAGMENT_RE.findall(text))
    symbol_chars = sum(1 for ch in text if ch in "=<>∑∏πλγτ∗∞≈≤≥¯")
    symbol_ratio = symbol_chars / max(len(text), 1)
    risk = 0.0
    if _LEADING_EQUATION_RE.search(text):
        risk += 0.65
    risk += min(0.50, math_hits * 0.12)
    risk += min(0.45, symbol_ratio * 10.0)
    return _clamp01(risk)


def source_text_quality(chain: FactChain, facts_by_id: Mapping[str, Fact]) -> float:
    """Cheap penalty against formula fragments and equation-heavy supports."""
    facts = _chain_facts(chain, facts_by_id)
    if not facts:
        return 0.0
    qualities = [1.0 - _math_fragment_risk(_fact_text(f)) for f in facts]
    return _clamp01(sum(qualities) / len(qualities))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _action_text_score(text: str) -> float:
    text = text or ""
    hits = len(_ACTION_VERB_RE.findall(text))
    return _clamp01(hits / 2.0)


def _edge_relation_label(edge: dict | None) -> str:
    if edge is None:
        return ""
    return str(edge.get("type", "") or "").strip().lower()


def _edge_text(edge: dict | None) -> str:
    if edge is None:
        return ""
    return " ".join(
        str(edge.get(key, "") or "")
        for key in ("relation_claim", "bridging_quote")
    )


def _role_pair_actionability(a: Fact, b: Fact) -> float:
    pair = (str(a.role or "").strip().lower(), str(b.role or "").strip().lower())
    if pair in _ACTIONABLE_ROLE_PAIR_PRIOR:
        return _ACTIONABLE_ROLE_PAIR_PRIOR[pair]
    if pair[0] != pair[1]:
        return 0.58
    return 0.42


def _unresolved_start_penalty(facts: list[Fact]) -> float:
    risky = sum(1 for fact in facts if _UNRESOLVED_START_RE.search(_fact_text(fact)))
    return _clamp01(0.18 * risky)


def compositional_actionability(
    chain: FactChain,
    facts_by_id: Mapping[str, Fact],
) -> float:
    """Estimate whether a chain can naturally yield a true multi-hop question.

    This is a pre-QGen ranking prior, not a correctness gate. It favors edges
    whose relation, roles, action verbs, and joint-only margin suggest that the
    answer should change when either support is removed.
    """
    facts = _chain_facts(chain, facts_by_id)
    if len(facts) <= 1:
        return 0.0

    relation_scores: list[float] = []
    role_scores: list[float] = []
    fact_action_scores: list[float] = []
    edge_action_scores: list[float] = []
    weak_definition_pairs = 0

    for a, b in zip(facts, facts[1:]):
        edge = _edge_for_pair(chain, a.fact_id, b.fact_id)
        label = _edge_relation_label(edge)
        relation_scores.append(_ACTIONABLE_RELATION_PRIOR.get(label, 0.40))
        role_scores.append(_role_pair_actionability(a, b))

        action_a = _action_text_score(_fact_text(a))
        action_b = _action_text_score(_fact_text(b))
        fact_action_scores.append(0.5 * action_a + 0.5 * action_b)
        edge_action_scores.append(_action_text_score(_edge_text(edge)))

        if label == "definition" and str(a.role) == "definition" and str(b.role) == "definition":
            weak_definition_pairs += 1

    score = (
        0.28 * _mean(relation_scores)
        + 0.18 * _mean(role_scores)
        + 0.20 * edge_joint_only_margin(chain)
        + 0.18 * _mean(fact_action_scores)
        + 0.16 * _mean(edge_action_scores)
    )
    score *= 0.70 + (0.30 * source_text_quality(chain, facts_by_id))
    score -= 0.14 * weak_definition_pairs
    score -= _unresolved_start_penalty(facts)
    return _clamp01(score)


def _role_pair_questionability(a: Fact, b: Fact) -> float:
    pair = (str(a.role or "").strip().lower(), str(b.role or "").strip().lower())
    if pair in _QUESTIONABLE_ROLE_PAIR_PRIOR:
        return _QUESTIONABLE_ROLE_PAIR_PRIOR[pair]
    if pair[0] != pair[1]:
        return 0.50
    return 0.34


def multi_hop_questionability(
    chain: FactChain,
    facts_by_id: Mapping[str, Fact],
) -> float:
    """Prior for chains likely to yield strict minimal multi-hop questions.

    This is a positive selector, not another gate. Recent strict Sutton rows
    that passed support minimality mostly used a setup/definition support plus
    a condition or rule support. This signal promotes those shapes so QGen sees
    more answer-focused multi-hop candidates before generic related pairs.
    """
    facts = _chain_facts(chain, facts_by_id)
    if len(facts) <= 1:
        return 0.0

    scores: list[float] = []
    for a, b in zip(facts, facts[1:]):
        edge = _edge_for_pair(chain, a.fact_id, b.fact_id)
        label = _edge_relation_label(edge)
        role_score = _role_pair_questionability(a, b)

        condition_bonus = 0.0
        if str(b.role or "").lower() == "condition" and _CONDITION_CUE_RE.search(_fact_text(b)):
            condition_bonus += 0.16
        if str(a.role or "").lower() == "condition" and _CONDITION_CUE_RE.search(_fact_text(a)):
            condition_bonus += 0.08

        setup_bonus = 0.0
        if str(a.role or "").lower() == "definition" and _SETUP_CUE_RE.search(_fact_text(a)):
            setup_bonus += 0.08

        relation_bonus = {
            "contrast": 0.12,
            "comparison": 0.12,
            "causal": 0.10,
            "procedural": 0.10,
            "intersection": 0.08,
            "descriptive": 0.02,
            "definition": 0.00,
        }.get(label, 0.04)

        weak_lookup_penalty = 0.0
        if (
            label in {"definition", "descriptive"}
            and str(a.role or "").lower() == "definition"
            and str(b.role or "").lower() == "definition"
        ):
            weak_lookup_penalty = 0.18

        scores.append(
            _clamp01(
                role_score
                + condition_bonus
                + setup_bonus
                + relation_bonus
                - weak_lookup_penalty
            )
        )

    return _clamp01(_mean(scores) * (0.75 + 0.25 * source_text_quality(chain, facts_by_id)))


def _pages(facts: list[Fact]) -> list[int]:
    out: list[int] = []
    for fact in facts:
        for span in fact.supporting_spans:
            if span.page_start is not None:
                out.append(int(span.page_start))
            if span.page_end is not None:
                out.append(int(span.page_end))
    return out


def _edge_for_pair(chain: FactChain, src: str, dst: str) -> dict | None:
    for edge in chain.chain_edges:
        if str(edge.get("src", "")) == src and str(edge.get("dst", "")) == dst:
            return edge
    return None


def _canonical_edge_label(
    chain: FactChain,
    src: str,
    dst: str,
    map_override: Mapping[str, str] | None = None,
) -> str:
    edge = _edge_for_pair(chain, src, dst)
    if edge is None:
        return "UNKNOWN"
    return canonicalize_label(str(edge.get("type", "")), dict(map_override) if map_override else None)


def chain_edge_labels(
    chain: FactChain,
    map_override: Mapping[str, str] | None = None,
) -> list[str]:
    """Canonical labels for each consecutive edge in the chain.

    Missing edges are returned as UNKNOWN so callers route the chain to
    untyped_mh rather than typed_mh.
    """
    labels: list[str] = []
    for src, dst in zip(chain.fact_ids, chain.fact_ids[1:]):
        labels.append(_canonical_edge_label(chain, src, dst, map_override))
    return labels


def typed_edge_confidence(
    chain: FactChain,
    map_override: Mapping[str, str] | None = None,
) -> float:
    """Mean NLI score across canonical non-UNKNOWN consecutive edges."""
    if len(chain.fact_ids) <= 1:
        return 0.0
    scores: list[float] = []
    for src, dst in zip(chain.fact_ids, chain.fact_ids[1:]):
        edge = _edge_for_pair(chain, src, dst)
        if edge is None:
            scores.append(0.0)
            continue
        label = canonicalize_label(str(edge.get("type", "")), dict(map_override) if map_override else None)
        if not is_typed(label):
            scores.append(0.0)
            continue
        scores.append(_clamp01(float(edge.get("nli_score", 0.0) or 0.0)))
    return sum(scores) / len(scores) if scores else 0.0


def edge_joint_only_margin(chain: FactChain) -> float:
    """Mean joint-only edge margin across consecutive edges.

    Edges produced before v16.7 do not carry this field; fall back to zero so
    older cached/debug rows are not over-ranked.
    """
    if len(chain.fact_ids) <= 1:
        return 0.0
    margins: list[float] = []
    for src, dst in zip(chain.fact_ids, chain.fact_ids[1:]):
        edge = _edge_for_pair(chain, src, dst)
        if edge is None:
            margins.append(0.0)
            continue
        margins.append(_clamp01(float(edge.get("joint_only_margin", 0.0) or 0.0)))
    return sum(margins) / len(margins) if margins else 0.0


def edge_answer_contribution_score(chain: FactChain) -> float:
    """Mean verified distinct-answer contribution score for consecutive edges."""
    if len(chain.fact_ids) <= 1:
        return 0.0
    scores: list[float] = []
    for src, dst in zip(chain.fact_ids, chain.fact_ids[1:]):
        edge = _edge_for_pair(chain, src, dst)
        if edge is None:
            scores.append(0.0)
            continue
        scores.append(_clamp01(float(edge.get("answer_contribution_score", 0.0) or 0.0)))
    return sum(scores) / len(scores) if scores else 0.0


def relation_type_strength(
    chain: FactChain,
    strengths: Mapping[str, float] | None = None,
) -> float:
    """Mean relation strength prior for consecutive TF-SFG edges.

    This is a cheap contribution prior: relation types that naturally force
    synthesis get higher scores than definition/descriptive/list edges, which
    often collapse into one-fact questions.
    """
    if len(chain.fact_ids) <= 1:
        return 0.0
    strengths = strengths or ChainScorerConfig().relation_type_strength
    vals: list[float] = []
    for src, dst in zip(chain.fact_ids, chain.fact_ids[1:]):
        edge = _edge_for_pair(chain, src, dst)
        if edge is None:
            vals.append(0.0)
            continue
        label = str(edge.get("type", "") or "").strip().lower()
        vals.append(_clamp01(float(strengths.get(label, 0.40))))
    return sum(vals) / len(vals) if vals else 0.0


def assign_category(
    chain: FactChain,
    map_override: Mapping[str, str] | None = None,
) -> ChainCategory:
    """Assign typed_mh / untyped_mh / fb per V16.2 category rules."""
    if len(chain.fact_ids) <= 1:
        return "fb"
    labels = chain_edge_labels(chain, map_override)
    if labels and all(is_typed(label) for label in labels):
        return "typed_mh"
    return "untyped_mh"


def _doc_median_fact_len(facts_by_id: Mapping[str, Fact]) -> float:
    lengths = [len(_fact_text(f)) for f in facts_by_id.values() if _fact_text(f)]
    if not lengths:
        return 1.0
    return max(1.0, float(statistics.median(lengths)))


def _signals_for_chain(
    chain: FactChain,
    facts_by_id: Mapping[str, Fact],
    cfg: ChainScorerConfig,
    doc_median_len: float,
    map_override: Mapping[str, str] | None = None,
) -> dict[str, float]:
    facts = _chain_facts(chain, facts_by_id)
    if not facts:
        return {
            "length": 0.0,
            "fragment": 0.0,
            "page_compactness": 0.0,
            "role_variety": 0.0,
            "lexical_bridge": 0.0,
            "typed_edge_confidence": 0.0,
            "edge_joint_only_margin": 0.0,
            "answer_contribution_score": 0.0,
            "relation_type_strength": 0.0,
            "source_text_quality": 0.0,
            "compositional_actionability": 0.0,
            "multi_hop_questionability": 0.0,
        }

    mean_len = sum(len(_fact_text(f)) for f in facts) / len(facts)
    s_length = _clamp01(mean_len / doc_median_len)

    fragment_count = sum(1 for f in facts if fact_domain_reject_reason(f) is not None)
    s_fragment = 1.0 - (fragment_count / len(facts))

    pages = _pages(facts)
    if len(pages) >= 2:
        page_span = max(pages) - min(pages)
    else:
        page_span = 0
    s_page_compactness = 1.0 - _clamp01(page_span / max(1, cfg.max_page_gap))

    roles = {str(f.role) for f in facts if f.role}
    s_role_variety = _clamp01(len(roles) / len(facts))

    overlaps = [
        _lexical_overlap(a, b)
        for a, b in zip(facts, facts[1:])
    ]
    s_lexical_bridge = sum(overlaps) / len(overlaps) if overlaps else 0.0

    s_typed_edge_confidence = typed_edge_confidence(chain, map_override)
    s_edge_joint_only_margin = edge_joint_only_margin(chain)
    s_answer_contribution_score = edge_answer_contribution_score(chain)
    s_relation_type_strength = relation_type_strength(
        chain,
        cfg.relation_type_strength,
    )
    s_source_text_quality = source_text_quality(chain, facts_by_id)
    s_compositional_actionability = compositional_actionability(chain, facts_by_id)
    s_multi_hop_questionability = multi_hop_questionability(chain, facts_by_id)

    return {
        "length": _clamp01(s_length),
        "fragment": _clamp01(s_fragment),
        "page_compactness": _clamp01(s_page_compactness),
        "role_variety": _clamp01(s_role_variety),
        "lexical_bridge": _clamp01(s_lexical_bridge),
        "typed_edge_confidence": _clamp01(s_typed_edge_confidence),
        "edge_joint_only_margin": _clamp01(s_edge_joint_only_margin),
        "answer_contribution_score": _clamp01(s_answer_contribution_score),
        "relation_type_strength": _clamp01(s_relation_type_strength),
        "source_text_quality": _clamp01(s_source_text_quality),
        "compositional_actionability": _clamp01(s_compositional_actionability),
        "multi_hop_questionability": _clamp01(s_multi_hop_questionability),
    }


def _weighted_score(signals: Mapping[str, float], weights: Mapping[str, float]) -> float:
    total_weight = sum(max(0.0, float(v)) for v in weights.values())
    if total_weight <= 0:
        return 0.0
    score = sum(_clamp01(signals.get(k, 0.0)) * max(0.0, float(w)) for k, w in weights.items())
    return _clamp01(score / total_weight)


def score_chains_pass1(
    chains: list[FactChain],
    facts_by_id: Mapping[str, Fact],
    cfg: ChainScorerConfig | None = None,
    *,
    map_override: Mapping[str, str] | None = None,
) -> list[ChainScore]:
    """Score every chain with cheap structural signals only."""
    cfg = cfg or ChainScorerConfig()
    doc_median = _doc_median_fact_len(facts_by_id)
    out: list[ChainScore] = []
    for chain in chains:
        signals = _signals_for_chain(chain, facts_by_id, cfg, doc_median, map_override)
        pass1 = _weighted_score(signals, cfg.pass1_weights)
        out.append(
            ChainScore(
                chain=chain,
                pass1_score=pass1,
                final_score=pass1,
                category=assign_category(chain, map_override),
                signals=signals,
                pass2_reranked=False,
            )
        )
    out.sort(key=lambda item: item.pass1_score, reverse=True)
    return out


def _nli_pair_compat(chain: FactChain, facts_by_id: Mapping[str, Fact]) -> float:
    facts = _chain_facts(chain, facts_by_id)
    pairs = [(_fact_text(a), _fact_text(b)) for a, b in zip(facts, facts[1:])]
    if not pairs:
        return 0.0
    scores = nli_check.nli_batch(pairs)
    if not scores:
        return 0.0
    return sum(max(0.0, float(s)) for s in scores) / len(scores)


def pass2_bound(cfg: ChainScorerConfig, attempt_cap: int) -> int:
    return max(0, min(int(cfg.nli_rerank_k), int(math.ceil(1.5 * max(0, attempt_cap)))))


def nli_rerank_pass2(
    scored: list[ChainScore],
    facts_by_id: Mapping[str, Fact],
    cfg: ChainScorerConfig | None = None,
    *,
    attempt_cap: int,
) -> list[ChainScore]:
    """Run local NLI rerank on top bounded chains that pass min_pass1_score."""
    cfg = cfg or ChainScorerConfig()
    eligible = [s for s in scored if s.pass1_score >= cfg.min_pass1_score]
    limit = pass2_bound(cfg, attempt_cap)
    rerank_ids = {id(s) for s in eligible[:limit]}
    out: list[ChainScore] = []

    pass1_weight = float(cfg.pass2_blend.get("pass1", 0.85))
    nli_weight = float(cfg.pass2_blend.get("nli_pair_compat", 0.15))
    blend_total = max(0.000001, pass1_weight + nli_weight)

    for item in scored:
        if id(item) not in rerank_ids:
            out.append(item)
            continue
        compat = _clamp01(_nli_pair_compat(item.chain, facts_by_id))
        signals = dict(item.signals)
        signals["nli_pair_compat"] = compat
        final = _clamp01(
            ((item.pass1_score * pass1_weight) + (compat * nli_weight)) / blend_total
        )
        out.append(
            ChainScore(
                chain=item.chain,
                pass1_score=item.pass1_score,
                final_score=final,
                category=item.category,
                signals=signals,
                pass2_reranked=True,
            )
        )
    out.sort(key=lambda item: item.final_score, reverse=True)
    return out


def rank_and_select(
    chains: list[FactChain],
    facts_by_id: Mapping[str, Fact],
    cfg: ChainScorerConfig | None = None,
    *,
    attempt_cap: int,
    map_override: Mapping[str, str] | None = None,
) -> tuple[list[ChainScore], dict]:
    """Return ranked chains plus build_summary-ready L1 stats."""
    cfg = cfg or ChainScorerConfig()
    pass1 = score_chains_pass1(chains, facts_by_id, cfg, map_override=map_override)
    pass1_pass = [s for s in pass1 if s.pass1_score >= cfg.min_pass1_score]
    reranked = nli_rerank_pass2(pass1, facts_by_id, cfg, attempt_cap=attempt_cap)
    kept = [
        s
        for s in reranked
        if s.final_score >= cfg.min_final_score_to_keep or s.category == "fb"
    ]

    hist = [0] * 10
    for item in reranked:
        idx = min(9, int(_clamp01(item.final_score) * 10))
        hist[idx] += 1

    stats = {
        "pass1_input": len(chains),
        "pass1_pass": len(pass1_pass),
        "pass2_input": sum(1 for s in reranked if s.pass2_reranked),
        "pass2_pass": sum(
            1
            for s in reranked
            if s.pass2_reranked and s.final_score >= cfg.min_final_score_to_keep
        ),
        "hardfail": len(chains) - len(kept),
        "score_histogram": hist,
    }
    return kept, stats
