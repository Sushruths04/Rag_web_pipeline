"""Tests for the 3-axis difficulty matrix evaluator (v16)."""

from __future__ import annotations

import pytest

from rag_gt.core.types import AnswerLog, Fact, MSFS, QuestionGT, Span
from rag_gt.evaluation.difficulty_matrix import CellStats, DifficultyMatrix


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
    hops: int = 2,
    sem: str = "intra_doc",
    adv: str = "clean",
    gold: str = "The correct answer.",
) -> QuestionGT:
    f = _fact("F001")
    return QuestionGT(
        q_id=q_id,
        question="What is the topic?",
        gold_answer=gold,
        msfs_list=[MSFS(msfs_id=f"{q_id}_msfs1", fact_ids=["F001"])],
        doc_ids=["d1"],
        required_fact_ids=["F001"],
        difficulty_reasoning_depth=hops,
        difficulty_semantic_distance=sem,
        difficulty_adversarial=adv,
        required_facts=[f],
    )


def _ans(q_id: str, predicted: str, abstained: bool = False) -> AnswerLog:
    return AnswerLog(q_id=q_id, predicted_answer=predicted, abstained=abstained)


def _exact_match(ans: AnswerLog, gold: str) -> bool:
    return ans.predicted_answer.strip().lower() == gold.strip().lower()


# ---------- DifficultyMatrix.from_questions ----------


def test_matrix_groups_by_three_axes():
    qs = [
        _q("q1", hops=2, sem="intra_doc", adv="clean"),
        _q("q2", hops=2, sem="intra_doc", adv="abstention"),
        _q("q3", hops=3, sem="cross_doc", adv="clean"),
    ]
    matrix = DifficultyMatrix.from_questions(qs)
    counts = matrix.cell_counts()
    assert counts.get("hops=2|sem=intra_doc|adv=clean") == 1
    assert counts.get("hops=2|sem=intra_doc|adv=abstention") == 1
    assert counts.get("hops=3|sem=cross_doc|adv=clean") == 1


def test_matrix_same_cell_accumulates():
    qs = [
        _q("q1", hops=1, sem="intra_doc", adv="clean"),
        _q("q2", hops=1, sem="intra_doc", adv="clean"),
        _q("q3", hops=1, sem="intra_doc", adv="clean"),
    ]
    matrix = DifficultyMatrix.from_questions(qs)
    counts = matrix.cell_counts()
    assert counts["hops=1|sem=intra_doc|adv=clean"] == 3


def test_matrix_empty_input():
    matrix = DifficultyMatrix.from_questions([])
    assert matrix.cells == []
    assert matrix.to_dict()["total_questions"] == 0


def test_matrix_unknown_sem_distance():
    qs = [_q("q1", sem="unknown_sem", adv="clean")]
    qs[0].difficulty_semantic_distance = None  # simulate missing field
    matrix = DifficultyMatrix.from_questions(qs)
    counts = matrix.cell_counts()
    assert any("sem=unknown" in k for k in counts)


# ---------- DifficultyMatrix.score ----------


def test_matrix_score_correct():
    qs = [_q("q1", gold="The correct answer.")]
    answers = [_ans("q1", "The correct answer.")]
    matrix = DifficultyMatrix.from_questions(qs)
    matrix.score(answers, _exact_match, qs)
    cell = matrix.cells[0]
    assert cell.total == 1
    assert cell.correct == 1
    assert cell.accuracy == pytest.approx(1.0)
    assert cell.error_rate == pytest.approx(0.0)


def test_matrix_score_incorrect():
    qs = [_q("q1", gold="The correct answer.")]
    answers = [_ans("q1", "Wrong answer.")]
    matrix = DifficultyMatrix.from_questions(qs)
    matrix.score(answers, _exact_match, qs)
    cell = matrix.cells[0]
    assert cell.correct == 0
    assert cell.error_rate == pytest.approx(1.0)


def test_matrix_score_missing_answer_ignored():
    qs = [_q("q1"), _q("q2")]
    answers = [_ans("q1", "The correct answer.")]  # q2 has no answer
    matrix = DifficultyMatrix.from_questions(qs)
    matrix.score(answers, _exact_match, qs)
    # Only q1 counted; q2 ignored
    total = sum(c.total for c in matrix.cells)
    assert total == 1


def test_matrix_score_resets_between_calls():
    qs = [_q("q1", gold="The correct answer.")]
    matrix = DifficultyMatrix.from_questions(qs)
    matrix.score([_ans("q1", "The correct answer.")], _exact_match, qs)
    matrix.score([_ans("q1", "Wrong.")], _exact_match, qs)
    cell = matrix.cells[0]
    assert cell.correct == 0


# ---------- axis_summary ----------


def test_axis_summary_by_adversarial():
    qs = [
        _q("q1", adv="clean"),
        _q("q2", adv="clean"),
        _q("q3", adv="abstention"),
    ]
    matrix = DifficultyMatrix.from_questions(qs)
    summary = matrix.axis_summary("adversarial")
    assert summary["clean"] == 2
    assert summary["abstention"] == 1


def test_axis_summary_by_hops():
    qs = [_q("q1", hops=2), _q("q2", hops=3), _q("q3", hops=2)]
    matrix = DifficultyMatrix.from_questions(qs)
    summary = matrix.axis_summary("hops")
    assert summary["2"] == 2
    assert summary["3"] == 1


# ---------- to_dict ----------


def test_to_dict_structure():
    qs = [_q("q1", hops=2, sem="intra_doc", adv="clean")]
    d = DifficultyMatrix.from_questions(qs).to_dict()
    assert "cells" in d
    assert "total_questions" in d
    assert d["total_questions"] == 1
    cell = d["cells"][0]
    assert cell["hops"] == 2
    assert cell["semantic_distance"] == "intra_doc"
    assert cell["adversarial"] == "clean"
    assert cell["n"] == 1


# ---------- CellStats ----------


def test_cell_stats_accuracy_none_when_no_total():
    cell = CellStats(hops=2, semantic_distance="intra_doc", adversarial="clean")
    assert cell.accuracy is None
    assert cell.error_rate is None
