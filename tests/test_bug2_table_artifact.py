"""BUG-2: flattened form/table residue must not ship as facts.

Real garbled facts from din_iso_15609_welding_procedure page 15 (the WPS form)
are rejected; a genuine definition that merely contains a colon, and good prose
facts with numbers, are kept.
"""
import pytest

from rag_gt.facts.domain_filter import is_table_artifact


TABLE_GARBLE = [
    # F000112 — colon-separated form field labels
    "Oscillation: amplitude, frequency, dwell time: – backing: Pulse welding "
    "details: Tungsten electrode type/size: Distance contact tube/work piece: "
    "Details of back gouging/backing: Plasma welding details: Preheating temperature:",
    # F000111 — glued footnote superscripts + colons
    "A Voltage2 V Type of current/po larity Wire feed speed2 Run out length1,2/ "
    "travel speed1 Arc energy1, 2/ Heat input 1, 2 Filler material designation "
    "and make: Any special baking or drying: Designation gas/flux: – shielding:",
]

NOT_GARBLE = [
    # F000103 — a real definition with ONE colon (kept; BUG-1 handles the bullet)
    "Distance contact tube/work piece: the distance from the nozzle to the "
    "surface of the work piece.",
    # good prose with numbers
    "For micro-hardness tests, the indenter shall contact the test piece at a "
    "velocity of 0,070 mm/s.",
    # ISO standard reference mentioned once
    "The tungsten electrode is classified in accordance with ISO 6848.",
    # a NOTE with a number
    "NOTE 2 There is evidence that some materials are sensitive to the rate of straining.",
]


@pytest.mark.parametrize("t", TABLE_GARBLE)
def test_table_garble_rejected(t):
    assert is_table_artifact(t) is True


@pytest.mark.parametrize("t", NOT_GARBLE)
def test_prose_kept(t):
    assert is_table_artifact(t) is False
