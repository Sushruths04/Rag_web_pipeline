"""Tests for V16.2 edge-label canonicalization."""

from __future__ import annotations

import pytest

from rag_gt.graph.edge_canonicalize import (
    CANONICAL_LABELS,
    CANONICAL_UNKNOWN,
    DEFAULT_CANON_MAP,
    canonicalize_label,
    is_typed,
    is_unknown,
)


# Every raw label in RELATION_TYPES from relation_support_gate.py must map
# to a valid canonical label (never error or produce a garbage string).
CLASSIFIER_OUTPUT_LABELS = [
    "descriptive",
    "definition",
    "comparison",
    "contrast",
    "causal",
    "mechanism",
    "intersection",
    "temporal",
    "quantitative",
    "procedural",
    "list",
]


def test_every_classifier_label_maps_to_canonical() -> None:
    for raw in CLASSIFIER_OUTPUT_LABELS:
        canonical = canonicalize_label(raw)
        assert canonical in CANONICAL_LABELS, (
            f"classifier label '{raw}' → '{canonical}' not in CANONICAL_LABELS"
        )


def test_none_maps_to_unknown() -> None:
    assert canonicalize_label("none") == CANONICAL_UNKNOWN


def test_unknown_maps_to_unknown() -> None:
    assert canonicalize_label("unknown") == CANONICAL_UNKNOWN


def test_empty_maps_to_unknown() -> None:
    assert canonicalize_label("empty") == CANONICAL_UNKNOWN
    assert canonicalize_label("") == CANONICAL_UNKNOWN


def test_unrecognised_maps_to_unknown() -> None:
    assert canonicalize_label("some_future_label") == CANONICAL_UNKNOWN
    assert canonicalize_label("RANDOM_GARBAGE_XYZ") == CANONICAL_UNKNOWN


def test_case_insensitive() -> None:
    assert canonicalize_label("COMPARISON") == "comparative"
    assert canonicalize_label("Causal") == "causal"
    assert canonicalize_label("MECHANISM") == "causal"


def test_comparison_and_contrast_both_map_to_comparative() -> None:
    assert canonicalize_label("comparison") == "comparative"
    assert canonicalize_label("contrast") == "comparative"


def test_causal_and_mechanism_both_map_to_causal() -> None:
    assert canonicalize_label("causal") == "causal"
    assert canonicalize_label("mechanism") == "causal"


def test_list_maps_to_descriptive() -> None:
    assert canonicalize_label("list") == "descriptive"


def test_canonical_labels_are_all_known() -> None:
    for raw, canonical in DEFAULT_CANON_MAP.items():
        assert canonical in CANONICAL_LABELS, (
            f"DEFAULT_CANON_MAP['{raw}'] = '{canonical}' not in CANONICAL_LABELS"
        )


def test_map_override_applies() -> None:
    override = {"custom_type": "causal", "none": CANONICAL_UNKNOWN}
    assert canonicalize_label("custom_type", map_override=override) == "causal"
    # Keys absent from override fall through to UNKNOWN (not DEFAULT_CANON_MAP).
    assert canonicalize_label("comparison", map_override=override) == CANONICAL_UNKNOWN


def test_map_override_bad_canonical_falls_back_to_unknown() -> None:
    override = {"foo": "not_a_real_canonical_label"}
    assert canonicalize_label("foo", map_override=override) == CANONICAL_UNKNOWN


def test_is_typed_and_is_unknown() -> None:
    assert is_unknown(CANONICAL_UNKNOWN) is True
    assert is_typed(CANONICAL_UNKNOWN) is False

    assert is_unknown("causal") is False
    assert is_typed("causal") is True
    assert is_typed("comparative") is True


def test_all_non_unknown_canonical_labels_are_typed() -> None:
    for label in CANONICAL_LABELS:
        if label == CANONICAL_UNKNOWN:
            assert not is_typed(label)
        else:
            assert is_typed(label), f"'{label}' should be typed"


def test_canonical_labels_are_idempotent() -> None:
    for label in CANONICAL_LABELS:
        assert canonicalize_label(label) == label


def test_whitespace_stripped_before_lookup() -> None:
    assert canonicalize_label("  comparison  ") == "comparative"
    assert canonicalize_label("\tnone\n") == CANONICAL_UNKNOWN
