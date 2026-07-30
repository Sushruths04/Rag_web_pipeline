"""v16 PASS-GT evaluation metrics.

Computes the paper's five acceptance criteria from a GT JSONL + an answers JSONL:

  1. abstention_rate       — AT twins: fraction correctly abstained (≥ 70% target)
  2. counterfactual_rate   — CT twins: fraction that contradicted the source (≥ 50% target)
  3. arm_overlap_stats     — ARM answers: mean 4-gram NP overlap with source (≤ 35% target)
  4. arm_nli_stats         — ARM answers: mean per-step NLI (≥ 0.55 target)
  5. distractor_fsr_drop   — Clean vs distractor FSR drop (≥ 20pp target)

All functions are pure (no I/O, no LLM calls) — suitable for unit testing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rag_gt.core.types import AnswerLog, QuestionGT
from rag_gt.generation.answers import ABSTENTION_TEXT


@dataclass
class AbstentionReport:
    total_at_twins: int
    correctly_abstained: int
    rate: float  # correctly_abstained / total_at_twins

    def passes(self, threshold: float = 0.70) -> bool:
        return self.rate >= threshold


@dataclass
class CounterfactualReport:
    total_ct_twins: int
    # A CT answer is "correct" when the model does NOT abstain and produces an
    # answer that differs from the original strict row's gold (proxy for detecting
    # the perturbation).  Full contradiction scoring requires an NLI pass that
    # this function defers to the caller via `is_contradicting`.
    correctly_perturbed: int
    rate: float

    def passes(self, threshold: float = 0.50) -> bool:
        return self.rate >= threshold


@dataclass
class ARMOverlapReport:
    n_steps: int
    mean_np_overlap: float
    std_np_overlap: float
    mean_nli_score: float
    std_nli_score: float
    pct_steps_pass_nli: float  # fraction with nli_score >= nli_threshold

    def overlap_passes(self, threshold: float = 0.35) -> bool:
        return self.mean_np_overlap <= threshold

    def nli_passes(self, threshold: float = 0.55) -> bool:
        return self.mean_nli_score >= threshold


@dataclass
class DistractorReport:
    clean_mean_fsr: float
    distractor_mean_fsr: float
    drop_pp: float  # percentage-point drop (clean - distractor) × 100

    def passes(self, threshold_pp: float = 20.0) -> bool:
        return self.drop_pp >= threshold_pp


# ---------- metric functions ----------


def abstention_rate(
    questions: List[QuestionGT],
    answers: List[AnswerLog],
) -> AbstentionReport:
    """Fraction of AT twins whose predicted answer is the abstention text."""
    at_twins = {q.q_id: q for q in questions if q.twin_type == "abstention"}
    ans_by_id: Dict[str, AnswerLog] = {a.q_id: a for a in answers}

    correctly_abstained = 0
    for q_id in at_twins:
        ans = ans_by_id.get(q_id)
        if ans is None:
            continue
        if ans.abstained or _is_abstention_text(ans.predicted_answer):
            correctly_abstained += 1

    total = len(at_twins)
    rate = correctly_abstained / total if total else 0.0
    return AbstentionReport(
        total_at_twins=total,
        correctly_abstained=correctly_abstained,
        rate=rate,
    )


def counterfactual_rate(
    questions: List[QuestionGT],
    answers: List[AnswerLog],
    original_answers: Optional[Dict[str, str]] = None,
) -> CounterfactualReport:
    """Fraction of CT twins whose predicted answer differs from the strict gold.

    `original_answers` maps q_id (of the strict parent) to its gold answer.
    If absent, we fall back to checking that the answer is non-empty and
    non-abstention (a weaker proxy).
    """
    ct_twins = {q.q_id: q for q in questions if q.twin_type == "counterfactual"}
    ans_by_id: Dict[str, AnswerLog] = {a.q_id: a for a in answers}

    correctly_perturbed = 0
    for q_id, q in ct_twins.items():
        ans = ans_by_id.get(q_id)
        if ans is None:
            continue
        if _is_abstention_text(ans.predicted_answer):
            continue  # abstention on CT twin is wrong
        if original_answers:
            parent_gold = original_answers.get(q.twin_of or "", "")
            if parent_gold and ans.predicted_answer.strip() != parent_gold.strip():
                correctly_perturbed += 1
        else:
            # Weak proxy: non-empty, non-abstention response
            correctly_perturbed += 1

    total = len(ct_twins)
    rate = correctly_perturbed / total if total else 0.0
    return CounterfactualReport(
        total_ct_twins=total,
        correctly_perturbed=correctly_perturbed,
        rate=rate,
    )


def arm_overlap_stats(
    questions: List[QuestionGT],
    nli_threshold: float = 0.55,
) -> ARMOverlapReport:
    """Aggregate NP overlap and NLI scores from ARM reasoning_trace fields.

    Only considers rows with non-empty reasoning_trace (i.e., ARM was used).
    """
    overlaps: List[float] = []
    nli_scores: List[float] = []

    for q in questions:
        for step in q.reasoning_trace:
            overlap = float(step.get("np_overlap", 0.0))
            nli = float(step.get("nli_score", 0.0))
            overlaps.append(overlap)
            nli_scores.append(nli)

    if not overlaps:
        return ARMOverlapReport(
            n_steps=0,
            mean_np_overlap=0.0,
            std_np_overlap=0.0,
            mean_nli_score=0.0,
            std_nli_score=0.0,
            pct_steps_pass_nli=0.0,
        )

    pass_nli = sum(1 for s in nli_scores if s >= nli_threshold)
    return ARMOverlapReport(
        n_steps=len(overlaps),
        mean_np_overlap=statistics.mean(overlaps),
        std_np_overlap=statistics.stdev(overlaps) if len(overlaps) > 1 else 0.0,
        mean_nli_score=statistics.mean(nli_scores),
        std_nli_score=statistics.stdev(nli_scores) if len(nli_scores) > 1 else 0.0,
        pct_steps_pass_nli=pass_nli / len(nli_scores),
    )


def distractor_fsr_drop(
    clean_fsr_scores: List[float],
    distractor_fsr_scores: List[float],
) -> DistractorReport:
    """Compare FSR under clean vs distractor-injection retrieval conditions.

    Both lists must be aligned (same questions, same order). FSR values are
    in [0, 1]; the drop is reported as percentage points.
    """
    if not clean_fsr_scores or not distractor_fsr_scores:
        return DistractorReport(
            clean_mean_fsr=0.0,
            distractor_mean_fsr=0.0,
            drop_pp=0.0,
        )

    clean_mean = statistics.mean(clean_fsr_scores)
    dist_mean = statistics.mean(distractor_fsr_scores)
    drop = (clean_mean - dist_mean) * 100.0
    return DistractorReport(
        clean_mean_fsr=clean_mean,
        distractor_mean_fsr=dist_mean,
        drop_pp=drop,
    )


# ---------- helpers ----------


def _is_abstention_text(text: str) -> bool:
    """Check whether a predicted answer is effectively the abstention response."""
    lower = text.strip().lower()
    if not lower:
        return False
    canonical = ABSTENTION_TEXT.lower()
    # Exact match or prefix (model may add trailing punctuation).
    return lower == canonical or lower.startswith(canonical[:30])


def summary_report(
    questions: List[QuestionGT],
    answers: List[AnswerLog],
    clean_fsr: Optional[List[float]] = None,
    distractor_fsr: Optional[List[float]] = None,
    original_strict_gold: Optional[Dict[str, str]] = None,
) -> dict:
    """Return a dict of all v16 metric reports for easy serialisation."""
    abst = abstention_rate(questions, answers)
    cf = counterfactual_rate(questions, answers, original_answers=original_strict_gold)
    arm = arm_overlap_stats(questions)
    dist_report = (
        distractor_fsr_drop(clean_fsr, distractor_fsr)
        if clean_fsr and distractor_fsr
        else None
    )

    out: dict = {
        "abstention": {
            "total_at_twins": abst.total_at_twins,
            "correctly_abstained": abst.correctly_abstained,
            "rate": round(abst.rate, 4),
            "passes_70pct": abst.passes(),
        },
        "counterfactual": {
            "total_ct_twins": cf.total_ct_twins,
            "correctly_perturbed": cf.correctly_perturbed,
            "rate": round(cf.rate, 4),
            "passes_50pct": cf.passes(),
        },
        "arm_overlap": {
            "n_steps": arm.n_steps,
            "mean_np_overlap": round(arm.mean_np_overlap, 4),
            "std_np_overlap": round(arm.std_np_overlap, 4),
            "mean_nli_score": round(arm.mean_nli_score, 4),
            "std_nli_score": round(arm.std_nli_score, 4),
            "pct_steps_pass_nli": round(arm.pct_steps_pass_nli, 4),
            "overlap_passes_35pct": arm.overlap_passes(),  # kept for back-compat
            "overlap_threshold": 0.35,
            "nli_passes_55": arm.nli_passes(),
        },
    }
    if dist_report is not None:
        out["distractor"] = {
            "clean_mean_fsr": round(dist_report.clean_mean_fsr, 4),
            "distractor_mean_fsr": round(dist_report.distractor_mean_fsr, 4),
            "drop_pp": round(dist_report.drop_pp, 2),
            "passes_20pp": dist_report.passes(),
        }
    return out
