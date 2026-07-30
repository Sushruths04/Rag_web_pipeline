"""DataMorgana-style intent diversity sampler for v16 question generation.

Tracks intent counts per run and returns the most under-represented intent
relative to a target distribution. This keeps the generated GT balanced
across the 8 intent types without requiring a strict pre-allocation.
"""

from __future__ import annotations

from typing import Dict

from rag_gt.generation.prompts import DEFAULT_DISTRIBUTION, known_intents


def pick_intent(
    current_counts: Dict[str, int],
    target_dist: Dict[str, float] | None = None,
) -> str:
    """Return the intent most below its target fraction.

    Args:
        current_counts: {intent: count_so_far} — missing intents are treated as 0.
        target_dist: {intent: target_fraction}; defaults to DEFAULT_DISTRIBUTION.
    """
    dist = target_dist if target_dist is not None else DEFAULT_DISTRIBUTION
    total = sum(current_counts.values())

    if total == 0:
        return max(dist, key=dist.get)

    # Compute deficit = target_fraction - actual_fraction for each intent.
    deficits: Dict[str, float] = {}
    for intent, target in dist.items():
        actual = current_counts.get(intent, 0) / total
        deficits[intent] = target - actual

    return max(deficits, key=deficits.get)


def make_counts() -> Dict[str, int]:
    """Return a zeroed count dict for all known intents."""
    return {intent: 0 for intent in known_intents()}
