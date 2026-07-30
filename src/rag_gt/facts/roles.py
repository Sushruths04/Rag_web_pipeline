"""Rule-based role detection for facts.

Order is significant: more specific patterns first. The `definition` fallback
requires explicit definitional cues (was previously `\\b(is|are)\\b`, which
swallowed every declarative sentence).
"""

from __future__ import annotations

import re

from rag_gt.core.types import Role

_ROLE_PATTERNS: list[tuple[re.Pattern[str], Role]] = [
    (
        re.compile(
            r"\b(shall not|must not|is prohibited|is not allowed|is forbidden)\b",
            re.I,
        ),
        "constraint",
    ),
    (re.compile(r"\b(must|shall|is required to|is mandatory)\b", re.I), "rule"),
    # `unless` lives in `exception`, NOT `condition` — "unless X" is semantically
    # an exception, and having it in both made `condition` always win.
    (
        re.compile(
            r"\b(except|unless|however|notwithstanding|with the exception)\b", re.I
        ),
        "exception",
    ),
    (
        re.compile(r"\b(if|when|provided that|in case|in the event)\b", re.I),
        "condition",
    ),
    (
        re.compile(
            r"\b(therefore|thus|consequently|as a result|causes|leads to|results in)\b",
            re.I,
        ),
        "consequence",
    ),
    (
        re.compile(
            r"\b(for example|for instance|such as|e\.g\.|i\.e\.|including)\b", re.I
        ),
        "example",
    ),
    # Definition: explicit cues only. Plain `\bis\b`/`\bare\b` matched every
    # declarative sentence and skewed the role distribution.
    (
        re.compile(
            r"\b(is defined as|are defined as|means|refers to|"
            r"is\s+(?:a|an|the)\s+\w+|are\s+(?:a|an|the)\s+\w+)\b",
            re.I,
        ),
        "definition",
    ),
]


def detect_role(text: str) -> Role:
    for pattern, role in _ROLE_PATTERNS:
        if pattern.search(text):
            return role
    return "definition"
