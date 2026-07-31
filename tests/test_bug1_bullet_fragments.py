"""BUG-1: bullet-list items must not ship as orphan fragment facts.

Deterministic coverage (no LLM) for the two guard layers:
- extract._clean_list_markers strips orphan leading/trailing list/clause markers.
- filter_adaptive._is_fragment rejects a fact that still opens with a bullet.

Evidence facts are the real ones found on din_iso_15609_welding_procedure.
"""
import pytest

from rag_gt.allpdf.extract import _clean_list_markers
from rag_gt.allpdf.filter_adaptive import _is_fragment


# (raw fact text, expected cleaned text)
BULLET_FACTS = [
    ("— Torch, electrode and/or wire angle (if required).",
     "Torch, electrode and/or wire angle (if required)."),
    ("— Wire speed feed range for mechanized and automatic welding.",
     "Wire speed feed range for mechanized and automatic welding."),
    ("For multiple electrode systems, the number and configuration of wire electrodes and polarity. —",
     "For multiple electrode systems, the number and configuration of wire electrodes and polarity."),
    ("— Weld run sequence given on the sketch if essential for the properties of the weld.",
     "Weld run sequence given on the sketch if essential for the properties of the weld."),
    ("4.4.6 Back gouging", "Back gouging"),
]

GOOD_FACTS = [
    "For micro-hardness tests, the indenter shall contact the test piece at a velocity of 0,070 mm/s.",
    "At least two hardness measurements shall be made on the calibrated surface of the reference block.",
]


@pytest.mark.parametrize("raw,expected", BULLET_FACTS)
def test_clean_list_markers_strips_orphan_markers(raw, expected):
    assert _clean_list_markers(raw) == expected


@pytest.mark.parametrize("good", GOOD_FACTS)
def test_clean_list_markers_leaves_good_facts(good):
    assert _clean_list_markers(good) == good


@pytest.mark.parametrize("raw", [
    "— Wire speed feed range for mechanized and automatic welding.",
    "— Torch angle",
    "–  Depth and shape.",
])
def test_stage4_backstop_rejects_bare_bullets(raw):
    assert _is_fragment(raw) is True


@pytest.mark.parametrize("good", GOOD_FACTS)
def test_stage4_keeps_good_facts(good):
    assert _is_fragment(good) is False
