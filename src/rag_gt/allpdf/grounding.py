"""BUG-3: answer-to-evidence grounding gate.

A ground-truth answer must be entailed by the union of its cited facts. When a
source fact is a fragment or the multi-hop pairing is weak, the answer can drift
to content not supported by the cited evidence — it then looks ungrounded on
cross-verification even if it is true in the wider document.

Uses the local, SQLite-cached cross-encoder NLI (`validation.nli_check.nli_batch`)
so it costs **no external API calls** — the same model already loaded at Stage 5.

This is a precision gate: the threshold is deliberately low (0.5) so only
clearly-unsupported answers are dropped; paraphrased-but-grounded answers (which
score ~0.9-1.0) are kept.
"""

from __future__ import annotations

from typing import List, Optional

from rag_gt.core.types import Fact

# Precision-oriented: grounded paraphrases score ~0.9-1.0, ungrounded ~0.0, so a
# low threshold cleanly drops the bad cases without touching valid paraphrases.
GROUNDING_THRESHOLD = 0.5


def _fact_text(fact: Fact) -> str:
    return (fact.canonical_form or fact.text or "").strip()


def answer_grounding_score(answer: str, facts: List[Fact]) -> Optional[float]:
    """NLI entailment of `answer` by the concatenation of all cited facts.

    Returns None when the check does not apply (no answer / no facts).
    """
    if not answer or not answer.strip() or not facts:
        return None
    premise = " ".join(_fact_text(f) for f in facts).strip()
    if not premise:
        return None
    from rag_gt.validation.nli_check import nli_batch

    return float(nli_batch([(premise, answer)])[0])


def is_grounded(
    answer: str, facts: List[Fact], *, threshold: float = GROUNDING_THRESHOLD
) -> bool:
    """True when the answer is entailed by the cited facts (or check N/A)."""
    score = answer_grounding_score(answer, facts)
    if score is None:
        return True  # nothing to check; abstention is handled separately
    return score >= threshold
