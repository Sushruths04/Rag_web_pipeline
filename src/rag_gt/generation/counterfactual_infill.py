"""Counterfactual span infill for CT twins (P4).

Finds a perturb-able span in a fact (numeric value or named entity), asks the
LLM to generate a plausible-but-incorrect replacement, and validates that the
perturbed fact NLI-contradicts the original at or above a threshold.

JSON parsing: json.loads(repair_json(_extract_json(raw))) — never response_format.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from json_repair import repair_json
from loguru import logger
from rag_gt.core.llm import APIError

from rag_gt.core.llm import _extract_json
from rag_gt.core.types import Fact
from rag_gt.validation.nli_check import nli_contradiction

if TYPE_CHECKING:
    from rag_gt.core.llm import LLM


# ---------- Span detection ----------

# Numeric: integers, decimals, scientific notation, percentages, multipliers.
_NUM_RE = re.compile(
    r'\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?:\s*%|\s*×|\s*x)?\b'
)
# Named entities: sequences of 2+ capitalized tokens (e.g. "Monte Carlo", "TD-lambda").
_NE_RE = re.compile(
    r'\b[A-Z][a-zA-Zλε-]+(?:[-\s][A-Z][a-zA-Zλε-]+)*\b'
)
# Minimum span character length — very short spans ("I", "A") are too ambiguous.
_MIN_SPAN_LEN = 2


def _find_candidate_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return (start, end, matched_text) tuples for perturb-able spans.

    Numeric spans are preferred; NE spans fill in when no numeric spans exist.
    Sorted by start position.
    """
    spans: List[Tuple[int, int, str]] = []
    for m in _NUM_RE.finditer(text):
        if len(m.group().strip()) >= _MIN_SPAN_LEN:
            spans.append((m.start(), m.end(), m.group()))
    if not spans:
        for m in _NE_RE.finditer(text):
            val = m.group()
            # Skip common stop-words that appear capitalized at sentence start.
            if val.lower() in {"the", "a", "an", "this", "that", "these", "those",
                                "in", "on", "at", "by", "for", "with", "from", "to"}:
                continue
            if len(val) >= _MIN_SPAN_LEN:
                spans.append((m.start(), m.end(), val))
    return sorted(spans, key=lambda t: t[0])


# ---------- LLM infill ----------

_INFILL_PROMPT = """\
You are generating counterfactual test cases for a RAG evaluation benchmark.

Given a factual statement, replace the highlighted span [{original_span}] with
a different but plausible value of the same type (number with number, name with name).
The replacement must make the statement FACTUALLY INCORRECT compared to the original.

Original statement:
{fact_text}

Highlighted span to replace: [{original_span}]

Rules:
- The replacement must be a single token or short phrase of the same type as the original.
- Do NOT change any other part of the statement.
- The replacement must be clearly different from the original (not a synonym or rephrasing).
- For numbers: use a different but realistic numeric value.
- For names: use a different but plausible name in the same domain.

Return ONLY a JSON object:
{{"replacement": "...", "rationale": "one-sentence explanation of why this is incorrect"}}
"""


# ---------- Result ----------

@dataclass
class CounterfactualResult:
    original_fact_text: str
    perturbed_fact_text: str
    original_span: str
    perturbed_span: str
    span_start: int
    span_end: int
    contradiction_score: float
    is_valid: bool   # True if contradiction_score >= threshold


def infill_counterfactual(
    fact: Fact,
    llm: "LLM",
    *,
    contradiction_threshold: float = 0.40,
    max_retries: int = 3,
) -> Optional[CounterfactualResult]:
    """Attempt to produce a counterfactual version of `fact`.

    Returns None if no perturb-able span is found or if all retries fail the
    NLI contradiction check. The contradiction_threshold is intentionally set
    lower than the plan's 0.60 because generic NLI models often assign neutral
    rather than contradiction to numeric substitutions; tune upward in Phase E.
    """
    fact_text = fact.canonical_form.strip() or fact.text.strip()
    candidates = _find_candidate_spans(fact_text)

    if not candidates:
        logger.debug(f"[CT] no perturb-able span in fact {fact.fact_id}: {fact_text[:80]!r}")
        return None

    # Try each candidate span until one produces a valid contradiction.
    for start, end, original_span in candidates:
        for attempt in range(max_retries):
            prompt = _INFILL_PROMPT.format(
                fact_text=fact_text,
                original_span=original_span,
            )
            try:
                raw = llm.generate(prompt, temperature=0.3, max_tokens=256)
                parsed = json.loads(repair_json(_extract_json(raw)))
                replacement = str(parsed.get("replacement", "")).strip()
            except APIError:
                raise
            except Exception as e:
                logger.debug(f"[CT] LLM/parse error attempt {attempt + 1}: {e}")
                continue

            if not replacement or replacement.lower() == original_span.lower():
                continue

            perturbed_text = (
                fact_text[:start] + replacement + fact_text[end:]
            )

            try:
                score = nli_contradiction(fact_text, perturbed_text)
            except APIError:
                raise
            except Exception as e:
                logger.debug(f"[CT] NLI contradiction error: {e}")
                score = 0.0

            is_valid = score >= contradiction_threshold

            if is_valid:
                return CounterfactualResult(
                    original_fact_text=fact_text,
                    perturbed_fact_text=perturbed_text,
                    original_span=original_span,
                    perturbed_span=replacement,
                    span_start=start,
                    span_end=end,
                    contradiction_score=score,
                    is_valid=True,
                )

            logger.debug(
                f"[CT] attempt {attempt + 1}: contradiction={score:.3f} < {contradiction_threshold}; retrying"
            )

    return None
