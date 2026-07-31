"""Regression fixture for the Stage-4 filter holes found in the v9 run.

Each fact below is a real canonical_form from pipeline_run_v9. The label encodes
what the filter SHOULD do:
  - "drop": structural junk (TOC, watermark, form-field, bare header, list-dump)
  - "anaphora": depends on an unresolved referent ("in those") — drop or resolve
  - "keep": genuine self-contained technical fact

The test calls the real `_relaxed_reject` (ISO_STANDARD tier) so it exercises the
exact code path the pipeline uses. Front-matter pages mirror the v9 profile.
"""
import pytest

from rag_gt.core.types import Fact, Span
from rag_gt.allpdf.filter_adaptive import _relaxed_reject

FRONT_PAGES = frozenset({2, 6, 7, 8})


def _fact(fid: str, text: str, page: int, sc: float, sc_known: bool) -> Fact:
    span = Span(
        doc_id="din_iso_15609_welding_procedure",
        chunk_id="c", start_token=0, end_token=0,
        char_start=0, char_end=len(text), page_start=page, page_end=page,
        bboxes=[],
    )
    return Fact(
        fact_id=fid, text=text, raw_text=text, canonical_form=text,
        self_containment_score=sc, self_containment_known=sc_known,
        role="descriptive", weight=max(0.01, sc), supporting_spans=[span],
    )


# (fact_id, canonical_form, page, sc, sc_known, label)
CASES = [
    # ---- structural junk that MUST drop ----
    ("F010001",
     "Page 10 is followed by Annex A (informative) titled Welding Procedure "
     "Specification (WPS) with the page reference 11, then a Bibliography with "
     "the page reference 12, and the term Conten.", 6, 0.7, True, "drop"),
    ("F027001",
     "User name: IP Printed copies are uncontrolled Introduction All new welding "
     "procedure specifications need to be prepared in accordance with the document "
     "from the date", 8, 0.0, True, "drop"),
    ("F000107",
     "Pulse welding details: Tungsten electrode type/size: Distance contact "
     "tube/work piece: Details of back gouging/backing: Plasma welding details: "
     "Preheating temperature: Torch angle: Interpass temperature:", 15, 1.0, False, "drop"),
    ("F000108",
     "Interpass temperature: Postheating: Pre-heat maintenance temperature: "
     "Post-weld heat treatment and/or ageing:", 15, 1.0, False, "drop"),
    ("F000109",
     "Time, temperature, method: Heating and cooling rates1: "
     ".......................................................................", 15, 1.0, False, "drop"),
    ("F000039",
     "4 Technical content of welding procedure specification (WPS)", 11, 1.0, False, "drop"),
    ("F000095", "— Additional filler materials.", 14, 1.0, False, "drop"),
    # per-page print-timestamp watermark (v11 finding): NLI-faithful but contentless
    ("F_printout", "The date and time of the printout are 2025-04-23, 12:00:03.",
     9, 1.0, False, "drop"),
    ("F000034",
     "ISO 4063, Welding and allied processes — Nomenclature of processes and "
     "reference numbers ISO 6848, Arc welding and cutting — Nonconsumable "
     "tungsten electrodes — Classification ISO 6947, Welding and allied "
     "processes — Welding positions ISO 14175, Welding consumables — Gases "
     "and gas mixtures for fusion welding ISO 15607, Specification and "
     "qualification of welding procedures — General rules", 10, 1.0, False, "drop"),

    # ---- anaphora fuel: drop OR must be resolved upstream ----
    ("F000049",
     "If the material is not assigned in those, ISO/TR 15608 shall be used.",
     11, 1.0, False, "anaphora"),

    # ---- genuine facts that MUST be kept ----
    ("F000050", "A WPS may cover a group of materials.", 11, 1.0, False, "keep"),
    ("F000041", "The information required in a pWPS/WPS is given in 4.2 to 4.5.",
     11, 1.0, False, "keep"),
    ("F000072",
     "The range of application for the WPS shall then be limited to equipment of "
     "that particular type.", 12, 1.0, False, "keep"),
    ("F000033",
     "For undated references, the latest edition of the referenced document "
     "(including any amendments) applies.", 10, 1.0, False, "keep"),
    ("F000071",
     "If the equipment does not permit control of one of either variable, the "
     "machine settings shall be specified instead.", 12, 1.0, False, "keep"),

    # ---- counter-examples: must NOT be over-rejected by the new detectors ----
    # resolved anaphora (demonstrative followed by a noun) is fine
    ("KEEP_resolved_anaphora",
     "If the material is not assigned in those material groups, ISO/TR 15608 "
     "shall be used.", 11, 1.0, False, "keep"),
    # a dotted ISO clause number is a real clause, not a bare header
    ("KEEP_iso_clause",
     "4.2 The welding process shall be specified for each run.", 11, 1.0, False, "keep"),
    # a sentence that cites two standards is not a reference-list dump
    ("KEEP_two_refs",
     "The terms and definitions given in ISO 15607 and ISO/TR 25901 apply to "
     "this document.", 10, 1.0, False, "keep"),
    # "that particular type" — demonstrative followed by a noun, not dangling
    ("KEEP_that_noun",
     "The application range is limited to equipment of that particular type "
     "only.", 12, 1.0, False, "keep"),
]


@pytest.mark.parametrize("fid,text,page,sc,known,label", CASES, ids=[c[0] for c in CASES])
def test_filter_decision(fid, text, page, sc, known, label):
    fact = _fact(fid, text, page, sc, known)
    reason = _relaxed_reject(fact, FRONT_PAGES)
    if label == "keep":
        assert reason is None, f"{fid} should be kept but was dropped: {reason}"
    else:  # drop / anaphora
        assert reason is not None, f"{fid} ({label}) should be dropped but was kept"


def _fact_with_raw(fid, canonical, raw, page):
    f = _fact(fid, canonical, page, 1.0, False)
    f.raw_text = raw  # pre-rewrite source span (form/table signature preserved)
    return f


def test_annex_form_rejected_via_raw_text_even_after_fluent_rewrite():
    """The v10 corruption: an informative Annex form is rewritten into a fluent
    normative requirement. The fluent text no longer looks like a form, but the
    raw source span does — so the raw-text guard must reject it."""
    # F067001: blank form fields -> "shall be specified" (informative -> normative)
    raw_67 = ("Oscillation: amplitude, frequency, dwell time: – backing: Pulse "
              "welding details: Tungsten electrode type/size: Distance contact "
              "tube/work piece: Plasma welding details: Preheating temperature: "
              "Torch angle: Interpass temperature:")
    canon_67 = ("The following parameters shall be specified: oscillation amplitude, "
                "oscillation frequency, oscillation dwell time, backing, preheating "
                "temperature, torch angle, and interpass temperature.")
    f67 = _fact_with_raw("F067001", canon_67, raw_67, 15)
    assert _relaxed_reject(f67, FRONT_PAGES) is not None, \
        "Annex form (F067001) must be rejected via its raw source span"

    # A genuine prose fact whose raw == canonical must still survive.
    good = _fact_with_raw(
        "F050",
        "A WPS may cover a group of materials.",
        "A WPS may cover a group of materials.", 11)
    assert _relaxed_reject(good, FRONT_PAGES) is None
