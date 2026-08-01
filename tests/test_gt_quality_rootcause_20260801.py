"""Root-cause regression tests for the 2026-08-01 GT quality investigation.

Every fixture in this file is a VERBATIM string taken from the live 5-document
run in ``production/data/runs/fb0ee36df5d7/artifacts/`` (453 pairs, DIN/ISO
welding standards). Nothing here is invented, so a passing suite means the
specific defects observed in that run cannot recur.

Defects covered, with the stage each was traced to:

  1. Stage 3  ``_expand_short_fact`` appends ". " onto text already ending in
     "." producing ".." (extract.py, predecessor-join branch).
  2. Stage 3  ``strip_running_artifacts`` strips only 1 of the 4 lines of the
     RWTH controlled-copy watermark, so the other 3 glue onto real sentences.
  3. Stage 4  no predicate rejects a "deferring fact" -- one that asserts only
     that content lives elsewhere. This is the defect the user reported.
  4. Stage 4  ``fact_has_unresolved_deictic`` is never wired into the relaxed
     tier, and cannot match ISO's ``shall`` register even if it were.
  5. Stage 4  ``ISO_BOILERPLATE_RE`` misses the generic clause-2 normative
     references intro and the CEN approval sentence.
  6. Stage 7  the 20-check ``gt_quality`` suite is never called by the allpdf
     pipeline, and the tautology gate is dead code.
  7. Stage 7  no dedup of any kind runs, so byte-identical questions ship.
"""

from __future__ import annotations

import pytest

from rag_gt.allpdf.extract import _expand_short_fact
from rag_gt.allpdf.filter_adaptive import _relaxed_reject
from rag_gt.core.types import Fact, Span
from rag_gt.facts.domain_filter import (
    ISO_BOILERPLATE_RE,
    is_deferring_fact,
    strip_running_artifacts,
)


def _fact(text: str, *, score: float = 1.0) -> Fact:
    """A fact shaped like the ones the live pipeline produces.

    supporting_spans is populated because Q7_bad_fact_fragment hard-fails on a
    fact with no chunk provenance -- production facts always carry it, and a
    span-less fixture would make every pair-gate test fail for the wrong reason.
    """
    return Fact(
        fact_id="doc_F000001",
        text=text,
        canonical_form=text,
        role="rule",
        supporting_spans=[
            Span(
                doc_id="doc",
                chunk_id="doc_c000001",
                start_token=0,
                end_token=20,
                page_start=12,
                page_end=12,
            )
        ],
        self_containment_score=score,
        self_containment_known=True,
    )


# ---------------------------------------------------------------------------
# 1. Stage 3 -- the ".." joiner bug
# ---------------------------------------------------------------------------

class TestShortFactExpansionPunctuation:
    """_expand_short_fact must not double the terminal period when it joins.

    Note on scope: ``_SENT_SPLIT_RE`` splits *after* ``[.!?]``, so every
    candidate predecessor except a trailing remainder already ends in terminal
    punctuation. The old ``". "`` joiner therefore doubled the period on
    essentially every join it ever performed -- which is why 13.5% of the
    2026-08-01 run's facts carried a "..".
    """

    CHUNK = (
        "The specification for the product could indicate limits for the "
        "differences between the lengths of the two diagonals. "
        "Both diagonals shall be measured and recorded."
    )
    SHORT = "Both diagonals shall be measured and recorded."

    def test_join_does_not_double_the_period(self):
        out = _expand_short_fact(self.SHORT, self.CHUNK)
        assert ".." not in out, f"doubled period leaked into fact text: {out!r}"

    def test_join_still_happens(self):
        """Guard against 'fixing' the bug by disabling the join entirely."""
        out = _expand_short_fact(self.SHORT, self.CHUNK)
        assert "differences between the lengths" in out, (
            f"predecessor context was dropped: {out!r}"
        )
        assert out != self.SHORT

    def test_long_fact_is_returned_unchanged(self):
        long_fact = (
            "Magnifications should be selected so that the diagonal can be "
            "enlarged to greater than 25 percent of the field of view."
        )
        assert _expand_short_fact(long_fact, self.CHUNK) == long_fact


# ---------------------------------------------------------------------------
# 2. Stage 3 -- the full controlled-copy watermark
# ---------------------------------------------------------------------------

class TestRunningArtifactStripping:
    """The RWTH watermark is four lines; all four must be stripped."""

    # Verbatim from page 12 of DIN EN ISO 3834-1-ENG.
    WATERMARK = (
        "Date/time of the printout: 2025-04-23, 11:56:33 "
        "Company name: RWTH Aachen University Universitätsbiblioth.. "
        "User name: IP Printed copies are uncontrolled"
    )

    def test_full_watermark_is_removed(self):
        out = strip_running_artifacts(self.WATERMARK)
        assert out.strip() == "", f"watermark residue survived: {out!r}"

    def test_watermark_glued_to_real_sentence_leaves_only_the_sentence(self):
        real = (
            "The specification for the product could indicate limits for the "
            "differences between the lengths of the two diagonals."
        )
        out = strip_running_artifacts(f"{real} {self.WATERMARK}")
        assert "Aachen" not in out
        assert "printout" not in out.lower()
        assert "differences between the lengths of the two diagonals" in out

    def test_legitimate_company_name_prose_is_not_destroyed(self):
        """Negative control: 'company name' is a real phrase in some standards."""
        real = (
            "The manufacturer shall record the company name and address of the "
            "organization responsible for the welding operation."
        )
        assert strip_running_artifacts(real) == real


# ---------------------------------------------------------------------------
# 3. Stage 4 -- deferring facts (the defect the user reported)
# ---------------------------------------------------------------------------

class TestDeferringFacts:
    """A fact asserting only that content lives elsewhere is not ground truth.

    These produce grounded, fluent, self-contained-scoring pairs whose answer
    can only restate the pointer, because the fact never held the content.
    """

    # -- verbatim defect fixtures from the run -----------------------------
    DEFERRING = [
        # The user's reported example (DIN EN ISO 3834-1-ENG_F019001).
        "Annex A lists criteria which assist in the selection of the "
        "appropriate part of the ISO 3834 series.",
        "The penetration time shall be defined in the written test procedure.",
        "The development time shall be defined in the written test procedure.",
        "The layout of a form that can be used for the test report is given in Annex C.",
        "This annex describes process and control tests used to monitor the "
        "implementation of the method.",
    ]

    # -- facts that LOOK similar but carry real content --------------------
    KEEP = [
        # Names a concrete standard as the answer -- a legitimate retrieval target.
        "The test blocks are specified in ISO 3452-4, Non-destructive testing "
        "— Penetrant testing — Part 4: Equipment.",
        # Instantiates its object with a material list.
        "The Vickers hardness specified in this document is also applicable for "
        "metallic and other inorganic coatings including electrodeposited "
        "coatings, autocatalytic coatings and sprayed coatings.",
        # Carries numeric values.
        "Magnifications should be selected so that the diagonal can be enlarged "
        "to greater than 25 %, but less than 75 % of the maximum possible "
        "optical field of view.",
        # A real normative requirement, no deferral.
        "Working areas shall be sited away from sources of heat, sparks or "
        "naked flames.",
        # Synthetic counter-example: deferral verb, but states the condition.
        "Annex B specifies that the test temperature shall be between 10 °C "
        "and 50 °C for all penetrant systems.",
        # Synthetic counter-example: 'described in' with a concrete payload.
        "The correction factors described in ISO 6507-1 range from 0.99 to 1.08 "
        "for convex surfaces.",
    ]

    @pytest.mark.parametrize("text", DEFERRING)
    def test_deferring_fact_is_detected(self, text):
        assert is_deferring_fact(text) is True, f"missed deferring fact: {text!r}"

    @pytest.mark.parametrize("text", KEEP)
    def test_contentful_fact_is_not_flagged(self, text):
        assert is_deferring_fact(text) is False, f"false positive: {text!r}"

    def test_relaxed_tier_rejects_the_reported_example(self):
        reason = _relaxed_reject(_fact(self.DEFERRING[0]))
        assert reason == "deferring_fact", f"got {reason!r}"

    def test_relaxed_tier_keeps_contentful_facts(self):
        for text in self.KEEP:
            assert _relaxed_reject(_fact(text)) is None, f"over-rejected: {text!r}"


# ---------------------------------------------------------------------------
# 4. Stage 4 -- unresolved deictic openers in ISO 'shall' register
# ---------------------------------------------------------------------------

class TestUnresolvedDeicticOpener:

    def test_this_noun_shall_opener_is_rejected(self):
        # Verbatim: DIN EN ISO 3452-1-ENG_F000201.
        text = (
            "This record shall be used as a comparison for the practical "
            "results obtained using the same test for the daily system "
            "performance check."
        )
        assert _relaxed_reject(_fact(text)) == "unresolved_deictic"

    def test_bare_pronoun_shall_opener_is_rejected(self):
        text = "It shall not be used to remedy inadequate removal of excess penetrant."
        assert _relaxed_reject(_fact(text)) == "unresolved_deictic"

    @pytest.mark.parametrize("text", [
        # Expletive "it": no referent is missing, the subject is extraposed.
        "It is the responsibility of a suitably qualified person, e.g. "
        "ISO 9712, Level 3, to decide which tests are applicable to a "
        "particular process line.",
        "It may be advantageous to also use a component with known natural "
        "discontinuities typical of those normally expected.",
    ])
    def test_expletive_it_is_not_treated_as_anaphora(self, text):
        assert _relaxed_reject(_fact(text)) != "unresolved_deictic"

    def test_resolved_demonstrative_is_kept(self):
        """Negative control: 'This document' names its own referent."""
        text = (
            "This document specifies the sizes of test specimen and the "
            "procedure for carrying out transverse tensile tests."
        )
        assert _relaxed_reject(_fact(text)) != "unresolved_deictic"


# ---------------------------------------------------------------------------
# 5. Stage 4 -- ISO/CEN boilerplate coverage
# ---------------------------------------------------------------------------

class TestIsoBoilerplateCoverage:

    BOILERPLATE = [
        "The following documents are referred to in the text in such a way that "
        "some or all of their content constitutes requirements of this document. "
        "For dated references, only the edition cited applies. For undated "
        "references, the latest edition of the referenced document applies.",
        "This European Standard was approved by CEN on 25 August 2023.",
        "All rights of exploitation in any form and by any means reserved "
        "worldwide for CEN national Members.",
    ]

    TECHNICAL = [
        "The penetrant shall be applied to the surface by spraying, brushing or "
        "immersion so that the whole area under test is completely covered.",
        "For undated references the test force shall be 9,807 N.",
    ]

    @pytest.mark.parametrize("text", BOILERPLATE)
    def test_boilerplate_is_matched(self, text):
        assert ISO_BOILERPLATE_RE.search(text), f"missed boilerplate: {text[:60]!r}"

    @pytest.mark.parametrize("text", TECHNICAL)
    def test_technical_prose_is_not_matched(self, text):
        assert not ISO_BOILERPLATE_RE.search(text), f"false positive: {text[:60]!r}"


# ---------------------------------------------------------------------------
# 6 + 7. Stage 7 -- pair-level quality gate and dedup
# ---------------------------------------------------------------------------

class TestPairRejection:
    """The allpdf pipeline must apply a pair-level quality gate and dedup."""

    def test_tautology_gate_is_wired_into_the_pipeline(self):
        from rag_gt.allpdf import pipeline as P

        assert hasattr(P, "_reject_pair_reason"), (
            "no pair-level rejection hook exists in the allpdf pipeline"
        )

    def test_restatement_pair_is_rejected(self):
        from rag_gt.allpdf.pipeline import _reject_pair_reason

        q = "What defines the penetration time in a written test procedure?"
        a = "The penetration time is defined in the written test procedure."
        f = _fact("The penetration time shall be defined in the written test procedure.")
        assert _reject_pair_reason(q, a, [f]) is not None

    def test_informative_pair_is_kept(self):
        from rag_gt.allpdf.pipeline import _reject_pair_reason

        q = "What percentage range should magnification enlarge the diagonal to?"
        a = (
            "Magnifications should enlarge the diagonal to more than 25 percent "
            "but less than 75 percent of the maximum possible optical field of view."
        )
        f = _fact(
            "Magnifications should be selected so that the diagonal can be "
            "enlarged to greater than 25 %, but less than 75 % of the maximum "
            "possible optical field of view."
        )
        assert _reject_pair_reason(q, a, [f]) is None

    # Pairs the FIRST version of this fix wrongly rejected. The original
    # tautology metric was a ratio ("<25% of the answer's content words are
    # new"), which punishes a short factoid answer for echoing the question's
    # phrasing even when it delivers the actual value. All three are good.
    FALSE_POSITIVES = [
        (
            "What is the minimum required development time according to the "
            "specification?",
            "The minimum required development time according to the "
            "specification is 10 minutes.",
        ),
        (
            "What standard specifies the requirements for test blocks in "
            "penetrant testing equipment?",
            "ISO 3452-4 specifies the requirements for test blocks in "
            "penetrant testing equipment.",
        ),
        (
            "What should be done to the test surface immediately after excess "
            "penetrant removal?",
            "The developer shall be applied to the test surface immediately "
            "after the removal of excess penetrant.",
        ),
    ]

    @pytest.mark.parametrize("question,answer", FALSE_POSITIVES)
    def test_short_factoid_answers_are_not_called_tautological(self, question, answer):
        from rag_gt.allpdf.pipeline import _is_tautological

        assert _is_tautological(question, answer) is False, (
            "answer delivers a real value but was scored circular"
        )

    def test_reported_example_is_still_tautological(self):
        """The user's own example must remain rejected after the metric change."""
        from rag_gt.allpdf.pipeline import _is_tautological

        q = ("What criteria does Annex A provide for selecting the appropriate "
             "part of the ISO 3834 series?")
        a = ("Annex A provides criteria that assist in the selection of the "
             "appropriate part of the ISO 3834 series.")
        assert _is_tautological(q, a) is True

    def test_answer_repeating_question_verbatim_is_tautological(self):
        from rag_gt.allpdf.pipeline import _is_tautological

        q = ("What defines a product family according to the manufacturer, "
             "user, or inspection authority?")
        a = ("A product family may be defined by the manufacturer, user, or "
             "inspection authority.")
        assert _is_tautological(q, a) is True

    def test_question_dedup_helper_collapses_identical_questions(self):
        from rag_gt.allpdf.pipeline import _normalize_question

        a = "What temperature condition applies to all parts of the test?"
        b = "What temperature condition applies to all parts of the test?"
        c = "  What TEMPERATURE condition applies to all parts of the test?!  "
        assert _normalize_question(a) == _normalize_question(b)
        assert _normalize_question(a) == _normalize_question(c)

    def test_distinct_questions_are_not_collapsed(self):
        from rag_gt.allpdf.pipeline import _normalize_question

        a = "What temperature condition applies to all parts of the test?"
        b = "What humidity condition applies to all parts of the test?"
        assert _normalize_question(a) != _normalize_question(b)
