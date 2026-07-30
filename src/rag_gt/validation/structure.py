"""Layer-1 free structural checks on generated (question, answer) pairs."""

from __future__ import annotations

import re
from typing import Tuple

# Yes/no question openers. Includes modal verbs that the previous version missed
# (`will`, `would`, `shall`, `may`, `might`, `must`).
_YES_NO_OPENERS = re.compile(
    r"^(is|are|am|does|do|did|was|were|has|have|had|"
    r"can|could|should|will|would|shall|may|might|must)\b",
    re.IGNORECASE,
)


def check_structure(question: str, answer: str) -> Tuple[bool, str]:
    q = question.strip()
    a = answer.strip()
    if len(q) < 15:
        return False, "question too short"
    if len(q.split()) < 4:
        return False, "question has fewer than 4 words"
    if len(a) < 1:
        return False, "answer is empty"
    if "?" not in q:
        return False, "not a question"
    if _YES_NO_OPENERS.match(q.lower()):
        return False, "yes/no question"
    return True, ""
