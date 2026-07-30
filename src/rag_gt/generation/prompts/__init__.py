"""Intent-taxonomy prompt registry for v16 diversity sampler."""

from __future__ import annotations

from rag_gt.generation.prompts import (
    comparative,
    counterfactual,
    factoid,
    inferential,
    list_,
    numerical,
    procedural,
    unanswerable,
)

_REGISTRY: dict[str, str] = {
    "factoid": factoid.SYSTEM_PROMPT,
    "inferential": inferential.SYSTEM_PROMPT,
    "comparative": comparative.SYSTEM_PROMPT,
    "numerical": numerical.SYSTEM_PROMPT,
    "procedural": procedural.SYSTEM_PROMPT,
    "counterfactual": counterfactual.SYSTEM_PROMPT,
    "list": list_.SYSTEM_PROMPT,
    "unanswerable": unanswerable.SYSTEM_PROMPT,
}

DEFAULT_DISTRIBUTION: dict[str, float] = {
    "factoid": 0.15,
    "inferential": 0.25,
    "comparative": 0.15,
    "numerical": 0.10,
    "procedural": 0.15,
    "counterfactual": 0.05,
    "list": 0.10,
    "unanswerable": 0.05,
}


def get_prompt(intent: str) -> str:
    """Return the system prompt for the given intent, or empty string if unknown."""
    return _REGISTRY.get(intent, "")


def known_intents() -> list[str]:
    return list(_REGISTRY.keys())
