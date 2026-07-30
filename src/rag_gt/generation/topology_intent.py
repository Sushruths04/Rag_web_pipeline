"""Topology-aware intent gate for V16.2 question generation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rag_gt.core.types import FactChain
from rag_gt.generation.prompts import DEFAULT_DISTRIBUTION
from rag_gt.graph.chain_scorer import (
    ChainScore,
    assign_category,
    chain_edge_labels,
    typed_edge_confidence,
)
from rag_gt.graph.edge_canonicalize import is_typed
from rag_gt.pipeline.yield_controller import ChainCategory


DEFAULT_TOPOLOGY_INTENTS: dict[str, tuple[str, ...]] = {
    "definition": ("factoid", "inferential"),
    "rule": ("factoid", "procedural"),
    "comparative": ("comparative",),
    "causal": ("inferential", "procedural"),
    "quantitative": ("numerical", "factoid"),
    "procedural": ("procedural",),
    "temporal": ("inferential", "factoid"),
    "intersection": ("inferential", "comparative"),
    "descriptive": ("factoid", "list"),
}


@dataclass(frozen=True)
class TopologyIntentConfig:
    enabled: bool = True
    counterfactual_min_typed_confidence: float = 0.65
    unanswerable_max_typed_confidence: float = 0.45
    untyped_mh_min_pass1_for_factoid: float = 0.65
    label_intents: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_TOPOLOGY_INTENTS)
    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "TopologyIntentConfig":
        raw = raw or {}
        raw_map = raw.get("map") or raw.get("label_intents") or {}
        label_intents = dict(DEFAULT_TOPOLOGY_INTENTS)
        for label, intents in raw_map.items():
            if isinstance(intents, str):
                label_intents[str(label)] = (intents,)
            else:
                label_intents[str(label)] = tuple(str(i) for i in intents)
        return cls(
            enabled=bool(raw.get("enabled", True)),
            counterfactual_min_typed_confidence=float(
                raw.get("counterfactual_min_typed_confidence", 0.65)
            ),
            unanswerable_max_typed_confidence=float(
                raw.get("unanswerable_max_typed_confidence", 0.45)
            ),
            untyped_mh_min_pass1_for_factoid=float(
                raw.get("untyped_mh_min_pass1_for_factoid", 0.65)
            ),
            label_intents=label_intents,
        )


@dataclass(frozen=True)
class TopologyIntentDecision:
    intent: str | None
    allowed_intents: tuple[str, ...]
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.intent is not None


def _as_chain(obj: FactChain | ChainScore) -> FactChain:
    return obj.chain if isinstance(obj, ChainScore) else obj


def _category(
    obj: FactChain | ChainScore,
    category: ChainCategory | None,
    map_override: Mapping[str, str] | None,
) -> ChainCategory:
    if category is not None:
        return category
    if isinstance(obj, ChainScore):
        return obj.category
    return assign_category(obj, map_override)


def _pass1_score(obj: FactChain | ChainScore, pass1_score: float | None) -> float:
    if pass1_score is not None:
        return float(pass1_score)
    if isinstance(obj, ChainScore):
        return float(obj.pass1_score)
    return 0.0


def _typed_confidence(
    obj: FactChain | ChainScore,
    typed_confidence: float | None,
    map_override: Mapping[str, str] | None,
) -> float:
    if typed_confidence is not None:
        return float(typed_confidence)
    if isinstance(obj, ChainScore):
        return float(obj.signals.get("typed_edge_confidence", 0.0))
    return typed_edge_confidence(obj, map_override)


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def allowed_intents_for_chain(
    chain_or_score: FactChain | ChainScore,
    cfg: TopologyIntentConfig | None = None,
    *,
    category: ChainCategory | None = None,
    pass1_score: float | None = None,
    typed_confidence: float | None = None,
    map_override: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return topology-compatible intents for a chain.

    Empty tuple means the chain should be dropped before question generation.
    """
    cfg = cfg or TopologyIntentConfig()
    chain = _as_chain(chain_or_score)
    cat = _category(chain_or_score, category, map_override)
    p1 = _pass1_score(chain_or_score, pass1_score)
    conf = _typed_confidence(chain_or_score, typed_confidence, map_override)

    if cat == "fb" or len(chain.fact_ids) <= 1:
        return ("factoid",)

    if cat == "untyped_mh":
        if p1 >= cfg.untyped_mh_min_pass1_for_factoid:
            return ("factoid",)
        return ()

    labels = chain_edge_labels(chain, map_override)
    typed_labels = [label for label in labels if is_typed(label)]
    if len(typed_labels) != len(labels) or not typed_labels:
        return ()

    allowed: list[str] = []
    for label in typed_labels:
        allowed.extend(cfg.label_intents.get(label, ()))

    # Multi-edge typed chains must stay multi-hop; remove factoid when possible.
    if len(typed_labels) >= 2:
        without_factoid = [intent for intent in allowed if intent != "factoid"]
        if without_factoid:
            allowed = without_factoid

    if conf >= cfg.counterfactual_min_typed_confidence:
        allowed.append("counterfactual")
    if conf < cfg.unanswerable_max_typed_confidence:
        allowed.append("unanswerable")

    return _ordered_unique(allowed)


def _renormalized_distribution(
    allowed: Sequence[str],
    target_dist: Mapping[str, float] | None,
) -> dict[str, float]:
    dist = target_dist if target_dist is not None else DEFAULT_DISTRIBUTION
    weights = {intent: float(dist.get(intent, 0.0)) for intent in allowed}
    total = sum(max(0.0, weight) for weight in weights.values())
    if total <= 0:
        return {}
    return {intent: max(0.0, weight) / total for intent, weight in weights.items()}


def _weighted_choice(weights: Mapping[str, float], rng: random.Random) -> str:
    r = rng.random() * sum(weights.values())
    cumulative = 0.0
    last = next(iter(weights))
    for intent, weight in weights.items():
        last = intent
        cumulative += weight
        if r <= cumulative:
            return intent
    return last


def _deficit_choice(
    weights: Mapping[str, float],
    current_counts: Mapping[str, int],
) -> str:
    total_seen = sum(max(0, int(v)) for v in current_counts.values())
    if total_seen <= 0:
        return max(weights, key=weights.get)
    deficits: dict[str, float] = {}
    for intent, target in weights.items():
        actual = current_counts.get(intent, 0) / total_seen
        deficits[intent] = target - actual
    return max(deficits, key=deficits.get)


def pick_topology_intent(
    chain_or_score: FactChain | ChainScore,
    current_counts: Mapping[str, int],
    target_dist: Mapping[str, float] | None = None,
    cfg: TopologyIntentConfig | None = None,
    *,
    category: ChainCategory | None = None,
    pass1_score: float | None = None,
    typed_confidence: float | None = None,
    rng: random.Random | None = None,
    map_override: Mapping[str, str] | None = None,
) -> TopologyIntentDecision:
    """Pick an intent from topology-compatible intents.

    If rng is supplied, sample according to the renormalized target
    distribution. Without rng, choose the most underrepresented compatible
    intent, matching the deterministic V16.1 sampler style.
    """
    cfg = cfg or TopologyIntentConfig()
    allowed = allowed_intents_for_chain(
        chain_or_score,
        cfg,
        category=category,
        pass1_score=pass1_score,
        typed_confidence=typed_confidence,
        map_override=map_override,
    )
    if not allowed:
        return TopologyIntentDecision(
            intent=None,
            allowed_intents=(),
            reason="no_compatible_intent",
        )

    weights = _renormalized_distribution(allowed, target_dist)
    if not weights:
        return TopologyIntentDecision(
            intent=None,
            allowed_intents=allowed,
            reason="no_compatible_intent",
        )

    intent = _weighted_choice(weights, rng) if rng is not None else _deficit_choice(
        weights, current_counts
    )
    return TopologyIntentDecision(intent=intent, allowed_intents=allowed)
