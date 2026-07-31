"""Tautology gate: drop Q/A pairs where the answer restates the question.

A tautological pair teaches the reader nothing — the answer's content is already
visible in the question. We measure what fraction of the answer's content words
are NEW (not already in the question). Below 25% → tautological → drop.

The two pairs flagged by the user from the v7 run are used as the canonical
bad examples; three known-good v7 pairs are the positive controls.
"""
from rag_gt.allpdf.pipeline import _is_tautological

# ── real bad pairs from v7 (both flagged by user) ─────────────────────────────

# F029001 (scope doc) + F000098 (form-field "Tungsten electrode: the diameter…")
Q_BAD_1 = (
    "Which ISO welding procedure specification requirements incorporate the "
    "tungsten electrode diameter codified by ISO 6848?"
)
A_BAD_1 = (
    "ISO welding procedure specification requirements include the tungsten "
    "electrode diameter, which is codified in accordance with ISO 6848."
)

# F000044 ("A WPS may cover a group of materials.") + F000058 (fragment run-width)
Q_BAD_2 = (
    "Which characteristic of a WPS specifies the maximum run width for "
    "manual and partly mechanized welding?"
)
A_BAD_2 = "WPS specifies the maximum width of the run for manual and partly mechanized welding."


# ── real good pairs from v7 ───────────────────────────────────────────────────

Q_GOOD_1 = (
    "What welding parameters and backing types are defined for fully "
    "mechanized automatic welding?"
)
A_GOOD_1 = (
    "Fully mechanized automatic welding specifies maximum weaving (or amplitude), "
    "frequency and dwell time of oscillation as its parameters, and lists material "
    "backing, gas backing and flux backing as the possible backing options."
)

Q_GOOD_2 = (
    "How does the maximum run width for manual or partly mechanized welding "
    "compare with the maximum weaving parameters for fully mechanized welding?"
)
A_GOOD_2 = (
    "Manual or partly mechanized welding is limited by a maximum run width, while "
    "fully mechanized welding is limited by maximum weaving parameters such as "
    "amplitude, frequency and dwell time."
)

Q_GOOD_3 = (
    "Under what circumstance should ISO/TR 15608 be applied when a WPS covers "
    "a group of materials?"
)
A_GOOD_3 = (
    "ISO/TR 15608 shall be applied when the WPS includes a group of materials "
    "and the specific material is not assigned within that group."
)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_tungsten_electrode_pair_is_tautological():
    """F029001+F000098: answer just copies the question's tungsten/ISO-6848 terms."""
    assert _is_tautological(Q_BAD_1, A_BAD_1) is True


def test_run_width_pair_is_tautological():
    """F000044+F000058: answer says 'WPS specifies X' where Q already said 'which X'."""
    assert _is_tautological(Q_BAD_2, A_BAD_2) is True


def test_weaving_parameters_pair_not_tautological():
    """Good pair: answer introduces weaving amplitude/frequency/oscillation — new info."""
    assert _is_tautological(Q_GOOD_1, A_GOOD_1) is False


def test_comparison_pair_not_tautological():
    """Good comparison pair: answer adds 'limited by', 'amplitude', 'dwell time'."""
    assert _is_tautological(Q_GOOD_2, A_GOOD_2) is False


def test_iso15608_condition_pair_not_tautological():
    """Good conditional pair: answer adds 'specific material not assigned within group'."""
    assert _is_tautological(Q_GOOD_3, A_GOOD_3) is False


def test_empty_answer_not_tautological():
    """Empty answer has no content words — gate does not falsely flag it."""
    assert _is_tautological("What is X?", "") is False
