"""Short-fact expansion: facts under _SHORT_FACT_THRESHOLD words get a neighbouring
sentence attached for context.

Uses only the deterministic _expand_short_fact helper — no LLM, no network.

Threshold is 12 (raised from 10) to also catch the two real-world patterns that
slipped through BUG-1 and BUG-2 (both are 11-word fragments):
  - F000058: "For manual welding and partly mechanized maximum width of the run."
    (prepositional opener, no main verb)
  - F000098: "Tungsten electrode: the diameter, and codification in accordance with ISO 6848."
    (form-field residue: one colon, not caught by BUG-2's ≥4-colon threshold)
"""
from rag_gt.allpdf.extract import _expand_short_fact, _SHORT_FACT_THRESHOLD


# --- ISO-style chunk that has a lead-in followed by short sub-facts -----------

CHUNK_WPS = (
    "Welding procedure specifications (WPS) shall contain all information necessary "
    "to make a weld. A WPS may cover a group of materials. "
    "Ranges and tolerances shall be specified where appropriate."
)

CHUNK_MANUAL = (
    "The following parameters shall be recorded for each welding process. "
    "For manual welding and partly mechanized maximum width of the run. "
    "For fully mechanized and automatic welding, maximum weaving or amplitude, "
    "frequency and dwell time of oscillation."
)

CHUNK_ELECTRODE = (
    "The welding procedure specification shall include the following. "
    "The run-out length of electrode consumed or travel speed. "
    "Current type and polarity shall also be specified."
)

# Real ISO 15609 chunk context for the form-field fact (F000098, page 14)
CHUNK_TUNGSTEN = (
    "For TIG welding, the following additional information shall be specified. "
    "Tungsten electrode: the diameter, and codification in accordance with ISO 6848. "
    "The shielding gas flow rate shall also be recorded."
)

# Real ISO 15609 chunk context for the prepositional-opener fact (F000058, page 12)
CHUNK_RUN_WIDTH = (
    "The welding procedure specification shall record the following oscillation details. "
    "For manual welding and partly mechanized maximum width of the run. "
    "For fully mechanized and automatic welding, maximum weaving or amplitude, "
    "frequency and dwell time of oscillation."
)


# --- tests: short facts get expanded (≤ 8 words, threshold 12) ---------------

def test_wps_materials_gets_leadIn():
    short = "A WPS may cover a group of materials."
    assert len(short.split()) < _SHORT_FACT_THRESHOLD
    expanded = _expand_short_fact(short, CHUNK_WPS)
    assert len(expanded.split()) > len(short.split())
    assert "group of materials" in expanded


def test_electrode_runout_gets_context():
    short = "The run-out length of electrode consumed or travel speed."
    assert len(short.split()) < _SHORT_FACT_THRESHOLD
    expanded = _expand_short_fact(short, CHUNK_ELECTRODE)
    assert len(expanded.split()) > len(short.split())
    assert "electrode" in expanded


# --- tests: 11-word fragments now caught (threshold raised from 10 to 12) ----

def test_F000058_prepositional_fragment_expanded():
    """F000058: 11-word fragment with no finite verb — previously slipped through."""
    frag = "For manual welding and partly mechanized maximum width of the run."
    assert len(frag.split()) == 11
    assert len(frag.split()) < _SHORT_FACT_THRESHOLD
    expanded = _expand_short_fact(frag, CHUNK_RUN_WIDTH)
    assert len(expanded.split()) > len(frag.split())
    assert "maximum width" in expanded


def test_F000098_form_field_residue_expanded():
    """F000098: 11-word form-field residue with single colon — previously slipped through."""
    frag = "Tungsten electrode: the diameter, and codification in accordance with ISO 6848."
    assert len(frag.split()) == 11
    assert len(frag.split()) < _SHORT_FACT_THRESHOLD
    expanded = _expand_short_fact(frag, CHUNK_TUNGSTEN)
    assert len(expanded.split()) > len(frag.split())
    assert "ISO 6848" in expanded


# --- tests: already-long facts are untouched ----------------------------------

def test_long_fact_unchanged():
    long_fact = (
        "For fully mechanized and automatic welding, maximum weaving or amplitude, "
        "frequency and dwell time of oscillation."
    )
    assert len(long_fact.split()) >= _SHORT_FACT_THRESHOLD
    assert _expand_short_fact(long_fact, CHUNK_MANUAL) == long_fact


def test_complete_13word_sentence_unchanged():
    """A complete 13-word sentence should not be expanded (above threshold 12)."""
    complete = "If the material is not assigned in those, ISO/TR 15608 shall be used."
    assert len(complete.split()) == 13
    assert len(complete.split()) >= _SHORT_FACT_THRESHOLD
    assert _expand_short_fact(complete, CHUNK_WPS) == complete


def test_fact_not_in_chunk_unchanged():
    short = "pH values below 7.0 indicate acidity."
    result = _expand_short_fact(short, CHUNK_WPS)
    assert isinstance(result, str)
    assert len(result) >= len(short)


# --- edge cases ---------------------------------------------------------------

def test_empty_chunk_returns_canonical():
    assert _expand_short_fact("Short fact.", "") == "Short fact."


def test_single_sentence_chunk_no_neighbour():
    result = _expand_short_fact("Short fact.", "Short fact.")
    assert "Short fact." in result
