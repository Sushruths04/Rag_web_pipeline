"""Atomic-clause decomposition for Constructive Gold Answers (CGA).

Phase 2 of the SFU+CGA+SAE plan (plan v6). Takes a single gold-answer
paragraph and decomposes it into atomic claims (short, independent
statements). The decomposition style is inspired by FActScore but
implemented locally so we can cite it as our own.

Each atomic clause must be:
- A standalone factual claim
- Short (typically 5-25 words)
- NOT contain conjunctions joining two independent claims ("and", "but")
- NOT contain references that require the original passage to resolve
  ("the above", "this method", "they")

The downstream NLI guard (`atomic_clause_entailment` in `nli_check.py`)
then checks each clause against the set of SFU canonical_forms used to
construct the gold answer. Any clause not NLI-entailed by at least one
SFU is flagged as an unsupported claim, and the gold answer is rejected
or rewritten.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, List

from loguru import logger

if TYPE_CHECKING:
    from rag_gt.core.llm import LLM


_DECOMP_PROMPT = """Decompose the passage below into a list of atomic factual claims.

Strict rules:
- Each claim must be a SHORT (5-25 words), STANDALONE sentence.
- Each claim must contain at most ONE independent factual statement.
- Split compound sentences. "A and B" with two facts becomes two claims.
- Do NOT introduce any new facts.
- Do NOT use references that the reader cannot resolve from the claim
  itself: "the above", "this method", "they", "it".
- Resolve pronouns to their antecedents using ONLY the passage.
- Preserve all numerical values and named entities verbatim.

Output STRICT JSON: an array of strings. No preamble, no headers, no
trailing commas.

Passage:
<<PASSAGE>>
{passage}
<</PASSAGE>>

JSON:"""


# Fallback heuristic split when the LLM fails. Splits on period+space and on
# the most common conjunctions. Not as good as the LLM path but at least
# yields some segmentation so the NLI guard has something to operate on.
_FALLBACK_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|(?:,?\s+)(?:and|but|whereas|while|however|moreover)\s+",
    flags=re.IGNORECASE,
)


def _heuristic_decompose(passage: str) -> List[str]:
    if not passage.strip():
        return []
    parts = _FALLBACK_SPLIT_RE.split(passage.strip())
    out: List[str] = []
    for p in parts:
        cleaned = p.strip().rstrip(",")
        if len(cleaned) < 5:
            continue
        if not cleaned.endswith((".", "!", "?")):
            cleaned += "."
        out.append(cleaned)
    return out


def decompose_into_atomic_clauses(
    passage: str,
    llm: "LLM | None" = None,
    *,
    use_llm: bool = True,
) -> List[str]:
    """Return a list of atomic-clause strings extracted from `passage`.

    Falls back to a heuristic split if `llm` is None / `use_llm=False` or
    if the LLM output is unparseable.
    """
    if not passage or not passage.strip():
        return []
    if not use_llm or llm is None:
        return _heuristic_decompose(passage)

    from rag_gt.core.llm import APIError
    from json_repair import repair_json

    prompt = _DECOMP_PROMPT.format(passage=passage.strip())
    try:
        raw = llm.generate(prompt, temperature=0.0, max_tokens=768).strip()
    except APIError:
        raise
    except Exception as e:
        logger.warning(
            f"[atomic_decomp] transient LLM error: {type(e).__name__}: {e}; "
            f"falling back to heuristic split"
        )
        return _heuristic_decompose(passage)

    try:
        data = json.loads(repair_json(raw))
    except Exception as e:
        logger.warning(
            f"[atomic_decomp] could not parse JSON: {e}; "
            f"falling back to heuristic split"
        )
        return _heuristic_decompose(passage)

    if not isinstance(data, list):
        logger.warning(
            f"[atomic_decomp] LLM returned non-list ({type(data)}); falling back"
        )
        return _heuristic_decompose(passage)

    clauses: List[str] = []
    for item in data:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().strip('"').strip("'")
        if len(cleaned) < 5:
            continue
        clauses.append(cleaned)
    return clauses or _heuristic_decompose(passage)
