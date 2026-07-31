"""BUG-3: the answer-to-evidence grounding gate.

Uses the local cached NLI model. A paraphrased-but-grounded answer is kept; an
answer whose content is not entailed by the cited facts is rejected.
"""
from dataclasses import dataclass, field
from typing import List

import pytest

from rag_gt.allpdf.grounding import answer_grounding_score, is_grounded


@dataclass
class _F:
    text: str
    canonical_form: str = ""


def _facts(*texts: str) -> List[_F]:
    return [_F(text=t) for t in texts]


FACTS = _facts(
    "Wire speed feed range for mechanized and automatic welding is specified.",
    "If the equipment does not permit control of one of either variable, the "
    "machine settings shall be specified instead.",
)


def test_grounded_paraphrase_kept():
    good = ("The wire speed feed range is defined, but when equipment cannot "
            "control a variable, the machine settings are specified instead.")
    assert is_grounded(good, FACTS) is True
    assert answer_grounding_score(good, FACTS) >= 0.5


def test_ungrounded_answer_rejected():
    bad = "The Vickers hardness test requires a minimum load of 10 kgf held for 15 seconds."
    assert is_grounded(bad, FACTS) is False


def test_check_na_when_no_facts():
    assert is_grounded("anything", []) is True
    assert answer_grounding_score("anything", []) is None
