"""Edge-label canonicalization for V16.2 TF-SFG.

Raw labels emitted by the classifier vary (comparison/contrast/mechanism/etc).
This module maps them to a closed canonical set before any downstream routing
so topology-intent, chain scorer, and yield controller all see consistent labels.

UNKNOWN is the canonical label for unrecognised / "none" / "empty" output. Chains
carrying UNKNOWN edges are routed to the untyped_mh bucket — never typed.

The default map mirrors configs/v16.yaml:v16_2.edge_canonicalize.map and is
baked in here so it can be used without loading config at import time.
Pass `map_override` to substitute a config-loaded map.
"""

from __future__ import annotations

# Closed set of canonical labels understood by topology-intent, chain scorer,
# and yield controller. UNKNOWN is the catch-all for anything not mapped.
CANONICAL_UNKNOWN = "UNKNOWN"

CANONICAL_LABELS: frozenset[str] = frozenset({
    "comparative",
    "definition",
    "rule",
    "causal",
    "quantitative",
    "procedural",
    "temporal",
    "intersection",
    "descriptive",
    CANONICAL_UNKNOWN,
})

# Default raw → canonical map.
# Keys are lowercase classifier outputs; values must be in CANONICAL_LABELS.
# Anything absent → CANONICAL_UNKNOWN.
DEFAULT_CANON_MAP: dict[str, str] = {
    "comparative": "comparative",
    "comparison": "comparative",
    "contrast": "comparative",
    "definition": "definition",
    "rule": "rule",
    "causal": "causal",
    "mechanism": "causal",
    "quantitative": "quantitative",
    "procedural": "procedural",
    "temporal": "temporal",
    "intersection": "intersection",
    "descriptive": "descriptive",
    "list": "descriptive",
    # Explicit none/unknown/empty forms → UNKNOWN sentinel
    "none": CANONICAL_UNKNOWN,
    "unknown": CANONICAL_UNKNOWN,
    "empty": CANONICAL_UNKNOWN,
    "": CANONICAL_UNKNOWN,
}


def canonicalize_label(
    raw: str,
    map_override: dict[str, str] | None = None,
) -> str:
    """Return the canonical label for a raw classifier output.

    Lookup is case-insensitive. Any label absent from the map returns
    CANONICAL_UNKNOWN. If a config-provided map contains an invalid canonical
    value it also falls back to CANONICAL_UNKNOWN.
    """
    canon_map = map_override if map_override is not None else DEFAULT_CANON_MAP
    key = (raw or "").strip().lower()
    canonical = canon_map.get(key, CANONICAL_UNKNOWN)
    if canonical not in CANONICAL_LABELS:
        return CANONICAL_UNKNOWN
    return canonical


def is_unknown(label: str) -> bool:
    """True if this canonical label is the UNKNOWN sentinel."""
    return label == CANONICAL_UNKNOWN


def is_typed(label: str) -> bool:
    """True if this canonical label is a typed (non-UNKNOWN) edge."""
    return label in CANONICAL_LABELS and label != CANONICAL_UNKNOWN
