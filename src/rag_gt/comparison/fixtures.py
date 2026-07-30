"""Heuristic dry-run scoring for the comparison harness.

Lets the harness run end-to-end with no API budget and no `ragas` install.
The dry-run mimics RAGAS' four metrics with cheap local proxies so that
correlations with the RAG_GT side are non-degenerate (we deliberately add a
small bounded perturbation to avoid identical-vector traps in Pearson r).
"""

from __future__ import annotations

import random
from typing import Dict, List

from rapidfuzz import fuzz


def _clip(x: float) -> float:
    if x != x:  # NaN
        return float("nan")
    return max(0.0, min(1.0, x))


def heuristic_ragas_scores(
    question: str,
    predicted_answer: str,
    contexts: List[str],
    gold_answer: str,
    rag_gt_metrics: Dict[str, float],
    seed: int = 0,
) -> Dict[str, float]:
    """Return faked-but-sensible RAGAS-shaped scores.

    The dry-run anchors three of the four metrics on RAG_GT side numbers (so
    correlations are believable but not perfect) and synthesises
    `answer_relevancy` from a fuzzy question-answer overlap. A small
    seeded perturbation is added so that subsequent runs are deterministic
    and Spearman is well-defined.
    """
    rng = random.Random(seed)

    def _jitter(base: float, scale: float = 0.05) -> float:
        if base != base:
            return float("nan")
        return _clip(base + rng.gauss(0.0, scale))

    fsr = rag_gt_metrics.get("fact_span_recall", 0.0)
    fsp = rag_gt_metrics.get("fact_span_precision", 0.0)
    nli_faith = rag_gt_metrics.get("faithfulness", 0.0)

    rel = fuzz.token_set_ratio(question or "", predicted_answer or "") / 100.0

    return {
        "context_recall": _jitter(fsr),
        "context_precision": _jitter(fsp),
        "faithfulness": _jitter(nli_faith),
        "answer_relevancy": _jitter(rel),
    }
