"""3-axis difficulty matrix for PASS-GT v16.

Axes: hops × semantic_distance × adversarial (clean/distractor/abstention/counterfactual).

GRADE (EMNLP 2025) uses 2 axes; v16 adds the adversarial dimension so cells
report RAG system error rates under distractor injection, abstention, and
counterfactual perturbation independently of the retrieval difficulty.

Usage:
    matrix = DifficultyMatrix.from_questions(questions)
    scores = matrix.score(answers, is_correct_fn)
    print(matrix.to_dict())
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from rag_gt.core.types import AnswerLog, QuestionGT

# Type alias for a matrix cell key.
CellKey = Tuple[int, str, str]


@dataclass
class CellStats:
    hops: int
    semantic_distance: str
    adversarial: str
    q_ids: List[str] = field(default_factory=list)
    # Populated after score().
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> Optional[float]:
        return self.correct / self.total if self.total else None

    @property
    def error_rate(self) -> Optional[float]:
        acc = self.accuracy
        return 1.0 - acc if acc is not None else None

    def to_dict(self) -> dict:
        return {
            "hops": self.hops,
            "semantic_distance": self.semantic_distance,
            "adversarial": self.adversarial,
            "n": len(self.q_ids),
            "correct": self.correct,
            "total": self.total,
            "accuracy": round(self.accuracy, 4) if self.accuracy is not None else None,
            "error_rate": round(self.error_rate, 4) if self.error_rate is not None else None,
        }


class DifficultyMatrix:
    """Groups QuestionGT rows into the 3-axis difficulty cell grid.

    Cells are keyed by (hops, semantic_distance, adversarial). Rows with
    missing or unknown axis values land in a catch-all ("unknown") bucket
    rather than causing a crash.
    """

    def __init__(self) -> None:
        self._cells: Dict[CellKey, CellStats] = {}

    # ---------- construction ----------

    @classmethod
    def from_questions(cls, questions: List[QuestionGT]) -> "DifficultyMatrix":
        matrix = cls()
        for q in questions:
            hops = max(1, q.difficulty_reasoning_depth or 1)
            sem = q.difficulty_semantic_distance or "unknown"
            adv = q.difficulty_adversarial or "clean"
            key: CellKey = (hops, sem, adv)
            if key not in matrix._cells:
                matrix._cells[key] = CellStats(
                    hops=hops, semantic_distance=sem, adversarial=adv
                )
            matrix._cells[key].q_ids.append(q.q_id)
        return matrix

    # ---------- scoring ----------

    def score(
        self,
        answers: List[AnswerLog],
        is_correct: Callable[[AnswerLog, str], bool],
        questions: List[QuestionGT],
    ) -> "DifficultyMatrix":
        """Populate correct/total for each cell.

        `is_correct(answer_log, gold_answer)` returns True when the predicted
        answer is acceptable. The caller supplies the correctness criterion so
        this module stays evaluation-framework-agnostic.
        """
        gold_by_id: Dict[str, str] = {q.q_id: q.gold_answer for q in questions}
        key_by_id: Dict[str, CellKey] = {}
        for q in questions:
            hops = max(1, q.difficulty_reasoning_depth or 1)
            sem = q.difficulty_semantic_distance or "unknown"
            adv = q.difficulty_adversarial or "clean"
            key_by_id[q.q_id] = (hops, sem, adv)

        # Reset counts.
        for cell in self._cells.values():
            cell.correct = 0
            cell.total = 0

        for ans in answers:
            key = key_by_id.get(ans.q_id)
            if key is None or key not in self._cells:
                continue
            gold = gold_by_id.get(ans.q_id, "")
            self._cells[key].total += 1
            if is_correct(ans, gold):
                self._cells[key].correct += 1

        return self

    # ---------- output ----------

    @property
    def cells(self) -> List[CellStats]:
        return list(self._cells.values())

    def cell_counts(self) -> Dict[str, int]:
        """Return a flat {cell_label: count} dict for quick inspection."""
        return {
            f"hops={c.hops}|sem={c.semantic_distance}|adv={c.adversarial}": len(c.q_ids)
            for c in self._cells.values()
        }

    def to_dict(self) -> dict:
        return {
            "cells": [c.to_dict() for c in sorted(
                self._cells.values(),
                key=lambda c: (c.hops, c.semantic_distance, c.adversarial),
            )],
            "total_questions": sum(len(c.q_ids) for c in self._cells.values()),
        }

    def axis_summary(self, axis: str) -> Dict[str, int]:
        """Return total question counts collapsed along one axis.

        axis: "hops" | "semantic_distance" | "adversarial"
        """
        counts: Dict[str, int] = defaultdict(int)
        for cell in self._cells.values():
            key = str(getattr(cell, axis, "unknown"))
            counts[key] += len(cell.q_ids)
        return dict(counts)
