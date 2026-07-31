"""Tests for v16 evaluation metrics (abstention, CT, ARM overlap, distractor drop)."""

from __future__ import annotations

import pytest

from rag_gt.core.types import AnswerLog, Fact, MSFS, QuestionGT, Span
from rag_gt.evaluation.metrics import (
    ARMOverlapReport,
    AbstentionReport,
    CounterfactualReport,
    DistractorReport,
    _is_abstention_text,
    abstention_rate,
    arm_overlap_stats,
    counterfactual_rate,
    distractor_fsr_drop,
    summary_report,
)
from rag_gt.generation.answers import ABSTENTION_TEXT


# ---------- helpers ----------


def _span() -> Span:
    return Span(doc_id="d1", chunk_id="d1_c000000", start_token=0, end_token=5)


def _fact(fid: str) -> Fact:
    return Fact(
        fact_id=fid,
        text=f"Fact {fid} text about the topic here.",
        role="definition",
        supporting_spans=[_span()],
    )


def _q(
    q_id: str,
    gold: str = "The answer.",
    twin_type: str | None = None,
    twin_of: str | None = None,
    reasoning_trace: list | None = None,
    adv: str = "clean",
) -> QuestionGT:
    f = _fact("F001")
    return QuestionGT(
        q_id=q_id,
        question="What is the topic?",
        gold_answer=gold,
        msfs_list=[MSFS(msfs_id=f"{q_id}_msfs1", fact_ids=["F001"])],
        doc_ids=["d1"],
        required_fact_ids=["F001"],
        difficulty_reasoning_depth=2,
        difficulty_semantic_distance="intra_doc",
        required_facts=[f],
        twin_type=twin_type,
        twin_of=twin_of,
        reasoning_trace=reasoning_trace or [],
        difficulty_adversarial=adv,
    )


def _ans(q_id: str, text: str, abstained: bool = False) -> AnswerLog:
    return AnswerLog(q_id=q_id, predicted_answer=text, abstained=abstained)


# ---------- _is_abstention_text ----------


def test_is_abstention_exact():
    assert _is_abstention_text(ABSTENTION_TEXT) is True


def test_is_abstention_prefix():
    assert _is_abstention_text("Insufficient information to answer. More context needed.") is True


def test_is_abstention_empty():
    assert _is_abstention_text("") is False


def test_is_abstention_wrong():
    assert _is_abstention_text("The answer is 42.") is False


# ---------- abstention_rate ----------


def test_abstention_rate_all_correct():
    qs = [
        _q("q1_at", gold=ABSTENTION_TEXT, twin_type="abstention", twin_of="q1"),
        _q("q2_at", gold=ABSTENTION_TEXT, twin_type="abstention", twin_of="q2"),
    ]
    answers = [
        _ans("q1_at", ABSTENTION_TEXT),
        _ans("q2_at", ABSTENTION_TEXT),
    ]
    report = abstention_rate(qs, answers)
    assert report.total_at_twins == 2
    assert report.correctly_abstained == 2
    assert report.rate == pytest.approx(1.0)
    assert report.passes()


def test_abstention_rate_none_correct():
    qs = [_q("q1_at", gold=ABSTENTION_TEXT, twin_type="abstention", twin_of="q1")]
    answers = [_ans("q1_at", "Wrong specific answer here.")]
    report = abstention_rate(qs, answers)
    assert report.correctly_abstained == 0
    assert report.rate == pytest.approx(0.0)
    assert not report.passes()


def test_abstention_rate_flag_counts():
    """AnswerLog.abstained=True also counts as correct abstention."""
    qs = [_q("q1_at", gold=ABSTENTION_TEXT, twin_type="abstention", twin_of="q1")]
    answers = [_ans("q1_at", "some text", abstained=True)]
    report = abstention_rate(qs, answers)
    assert report.correctly_abstained == 1


def test_abstention_rate_ignores_strict_rows():
    """Strict rows (twin_type=None) are not counted in AT metrics."""
    qs = [
        _q("q1", gold="Answer.", twin_type=None),
        _q("q1_at", gold=ABSTENTION_TEXT, twin_type="abstention", twin_of="q1"),
    ]
    answers = [_ans("q1", "Answer."), _ans("q1_at", ABSTENTION_TEXT)]
    report = abstention_rate(qs, answers)
    assert report.total_at_twins == 1


def test_abstention_rate_missing_answer():
    """AT twin with no answer entry is not counted as correct."""
    qs = [_q("q1_at", gold=ABSTENTION_TEXT, twin_type="abstention", twin_of="q1")]
    answers = []
    report = abstention_rate(qs, answers)
    assert report.correctly_abstained == 0
    assert report.total_at_twins == 1


# ---------- counterfactual_rate ----------


def test_counterfactual_rate_all_correct():
    qs = [
        _q("q1_ct", gold="Perturbed answer.", twin_type="counterfactual", twin_of="q1"),
    ]
    original = {"q1": "Original gold answer about the topic."}
    answers = [_ans("q1_ct", "Perturbed answer.")]
    report = counterfactual_rate(qs, answers, original_answers=original)
    assert report.correctly_perturbed == 1
    assert report.rate == pytest.approx(1.0)
    assert report.passes()


def test_counterfactual_rate_abstention_not_counted():
    qs = [_q("q1_ct", gold="Perturbed.", twin_type="counterfactual", twin_of="q1")]
    answers = [_ans("q1_ct", ABSTENTION_TEXT)]
    report = counterfactual_rate(qs, answers)
    assert report.correctly_perturbed == 0


def test_counterfactual_rate_same_as_strict_not_counted():
    qs = [_q("q1_ct", twin_type="counterfactual", twin_of="q1")]
    original = {"q1": "Same answer repeated here."}
    answers = [_ans("q1_ct", "Same answer repeated here.")]
    report = counterfactual_rate(qs, answers, original_answers=original)
    assert report.correctly_perturbed == 0


def test_counterfactual_rate_no_original_weak_proxy():
    """Without original_answers, any non-abstention response counts."""
    qs = [_q("q1_ct", twin_type="counterfactual", twin_of="q1")]
    answers = [_ans("q1_ct", "Any specific non-abstention answer.")]
    report = counterfactual_rate(qs, answers, original_answers=None)
    assert report.correctly_perturbed == 1


# ---------- arm_overlap_stats ----------


def _step(overlap: float, nli: float) -> dict:
    return {
        "step_id": "s1",
        "claim": "Compressed claim text.",
        "fact_ids": ["F001"],
        "relation_type": "definition",
        "bbox_refs": [],
        "np_overlap": overlap,
        "nli_score": nli,
        "passes": nli >= 0.55 and overlap <= 0.35,
    }


def test_arm_overlap_stats_basic():
    qs = [
        _q("q1", reasoning_trace=[_step(0.20, 0.80), _step(0.30, 0.75)]),
        _q("q2", reasoning_trace=[_step(0.10, 0.90)]),
    ]
    report = arm_overlap_stats(qs)
    assert report.n_steps == 3
    assert report.mean_np_overlap == pytest.approx((0.20 + 0.30 + 0.10) / 3, abs=1e-4)
    assert report.overlap_passes()


def test_arm_overlap_stats_empty():
    """No ARM trace → zeroed report."""
    qs = [_q("q1", reasoning_trace=[])]
    report = arm_overlap_stats(qs)
    assert report.n_steps == 0
    assert report.mean_np_overlap == pytest.approx(0.0)


def test_arm_overlap_stats_pct_nli():
    qs = [_q("q1", reasoning_trace=[_step(0.10, 0.80), _step(0.10, 0.30)])]
    report = arm_overlap_stats(qs, nli_threshold=0.55)
    assert report.pct_steps_pass_nli == pytest.approx(0.5)


def test_arm_overlap_stats_high_overlap_fails():
    qs = [_q("q1", reasoning_trace=[_step(0.50, 0.80)])]
    report = arm_overlap_stats(qs)
    assert not report.overlap_passes()


# ---------- distractor_fsr_drop ----------


def test_distractor_drop_basic():
    clean = [0.90, 0.85, 0.80]
    distractor = [0.60, 0.55, 0.50]
    report = distractor_fsr_drop(clean, distractor)
    assert report.clean_mean_fsr == pytest.approx(0.85, abs=1e-4)
    assert report.distractor_mean_fsr == pytest.approx(0.55, abs=1e-4)
    assert report.drop_pp == pytest.approx(30.0, abs=1e-3)
    assert report.passes()


def test_distractor_drop_insufficient():
    clean = [0.80]
    distractor = [0.75]  # only 5pp drop
    report = distractor_fsr_drop(clean, distractor)
    assert not report.passes()


def test_distractor_drop_empty():
    report = distractor_fsr_drop([], [])
    assert report.drop_pp == pytest.approx(0.0)


# ---------- summary_report ----------


def test_summary_report_structure():
    qs = [
        _q("q1", gold="Answer."),
        _q("q1_at", gold=ABSTENTION_TEXT, twin_type="abstention", twin_of="q1"),
        _q("q1_ct", gold="Perturbed.", twin_type="counterfactual", twin_of="q1"),
    ]
    answers = [
        _ans("q1", "Answer."),
        _ans("q1_at", ABSTENTION_TEXT),
        _ans("q1_ct", "Perturbed."),
    ]
    report = summary_report(qs, answers)
    assert "abstention" in report
    assert "counterfactual" in report
    assert "arm_overlap" in report
    assert "distractor" not in report  # not passed


def test_summary_report_with_distractor():
    qs = [_q("q1")]
    answers = [_ans("q1", "Answer.")]
    report = summary_report(qs, answers, clean_fsr=[0.80], distractor_fsr=[0.50])
    assert "distractor" in report
    assert report["distractor"]["drop_pp"] == pytest.approx(30.0, abs=1e-3)
