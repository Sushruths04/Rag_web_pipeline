"""Question form must follow what the fact affords, not default to "What".

Measured on the 438-pair run of 2026-08-01:

    what               393  89.7%
    how                 34   7.8%
    under what/why/when/who  11   2.5%

and, decisively, EVERY near-tautological pair opened with "What":

    opener   n     mean Q/A overlap   >=0.80 (near-tautological)
    what    393        0.556           46  (11.7%)
    how      34        0.464            0  ( 0.0%)
    other    11        0.426            0  ( 0.0%)

The mechanism: when a fact's payload is a purpose, condition or mechanism,
"What <noun> ...?" has no noun to ask for, so the payload gets copied into the
question and the answer restates the fact. The user-reported example:

    F: In the other approach, a correction is made to the measurement result
       to compensate for the systematic bias.
    Q: What correction is applied to measurement results to address systematic bias?
    A: A correction is applied to the measurement result to compensate for the
       systematic bias.
"""
from __future__ import annotations

import pytest

from rag_gt.core.types import Fact
from rag_gt.generation.questions import _build_prompt, question_affordances


def _fact(text: str) -> Fact:
    return Fact(fact_id="d_F000001", text=text, canonical_form=text, role="rule")


class TestAffordanceDetection:

    @pytest.mark.parametrize("text,expected", [
        # The reported example.
        ("In the other approach, a correction is made to the measurement "
         "result to compensate for the systematic bias.", "purpose"),
        ("The surface shall be cleaned by wiping with a lint-free cloth.", "mechanism"),
        ("If natural ageing of aluminium alloys takes place, the time between "
         "welding and testing shall be recorded.", "condition"),
        ("The test force shall be 9,807 N applied for 15 s.", "quantity"),
        ("Metrological traceability is defined as the property of a "
         "measurement result.", "definition"),
        ("The operator shall carry out any necessary testing and observations.", "actor"),
        ("Excess penetrant removal results in a loss of indication contrast.", "consequence"),
    ])
    def test_affordance_is_detected(self, text, expected):
        assert expected in question_affordances(text), question_affordances(text)

    def test_plain_statement_affords_nothing_special(self):
        """No match must degrade to no hint, never to a wrong one."""
        assert question_affordances(
            "Vickers hardness applies to metallic coatings."
        ) == []

    def test_empty_text_is_safe(self):
        assert question_affordances("") == []
        assert question_affordances(None) == []


class TestPromptCarriesTheGuidance:

    def test_single_fact_prompt_gets_an_affordance_hint(self):
        """The single-fact path had NO form guidance at all -- role shape hints
        were gated behind len(facts) == 2, and that path is ~99% of output."""
        f = _fact("In the other approach, a correction is made to the "
                  "measurement result to compensate for the systematic bias.")
        prompt = _build_prompt([f])
        assert "Question form for THIS fact" in prompt
        assert "Why" in prompt

    def test_single_fact_prompt_warns_against_defaulting_to_what(self):
        prompt = _build_prompt([_fact("The surface shall be cleaned by wiping.")])
        assert 'do not default to "What' in prompt

    def test_reported_example_appears_as_an_explicit_anti_pattern(self):
        """Pin the user-reported failure into the prompt so it cannot regress."""
        prompt = _build_prompt([_fact("Some fact about hardness testing methods.")])
        assert "compensate for the systematic bias" in prompt
        assert "Why is a correction made to a measurement result?" in prompt

    def test_no_hint_when_the_fact_affords_nothing_specific(self):
        prompt = _build_prompt([_fact("Vickers hardness applies to metallic coatings.")])
        assert "Question form for THIS fact" not in prompt

    def test_multi_fact_prompt_also_gets_the_hint(self):
        facts = [
            _fact("The test force shall be applied smoothly to the specimen."),
            _fact("The force is held so as to avoid vibration of the indenter."),
        ]
        prompt = _build_prompt(facts)
        assert "Question form for THIS fact" in prompt

    def test_existing_multi_fact_guidance_is_not_displaced(self):
        facts = [
            _fact("ISO 9606-1 requires a welder qualification test."),
            _fact("The certificate expires after 3 years unless reconfirmed."),
        ]
        prompt = _build_prompt(facts)
        assert "Required support anchors" in prompt
