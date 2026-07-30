"""`python -m rag_gt.cli.evaluate_v16` — v16 PASS-GT acceptance-criteria report.

Reads a v16 GT JSONL + an answers JSONL (from `cli/answers_test.py`) and prints
all five paper acceptance criteria:

  1. Abstention rate on AT twins           (target ≥ 70%)
  2. Counterfactual detection rate on CT   (target ≥ 50%)
  3. ARM mean NP overlap                   (target ≤ 35%)
  4. ARM mean per-step NLI                 (target ≥ 0.55)
  5. Distractor FSR drop                   (target ≥ 20pp — requires two answer runs)

Also builds the 3-axis difficulty matrix and writes it to a JSON file.

Usage:
    python -m rag_gt.cli.evaluate_v16 \\
        --gt   data/gt/v16_strict.jsonl \\
        --answers data/eval_runs/v16_answers.jsonl \\
        [--clean_fsr   data/eval_runs/v16_clean_fsr.json] \\
        [--distractor_fsr data/eval_runs/v16_distractor_fsr.json] \\
        [--output data/eval_results/v16_metrics.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from rag_gt.core.types import AnswerLog
from rag_gt.evaluation.difficulty_matrix import DifficultyMatrix
from rag_gt.evaluation.metrics import summary_report
from rag_gt.storage.gt_io import load_gt


def _load_answers(path: Path) -> list[AnswerLog]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(
                AnswerLog(
                    q_id=d["q_id"],
                    predicted_answer=d.get("predicted_answer", d.get("answer", "")),
                    abstained=bool(d.get("abstained", False)),
                )
            )
    return rows


def _load_fsr_list(path: Path) -> list[float]:
    """Load a JSON file that is either a list of floats or {q_id: fsr} dict."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [float(x) for x in data]
    if isinstance(data, dict):
        return [float(v) for v in data.values()]
    raise ValueError(f"Unrecognised FSR file format: {path}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag-gt-evaluate-v16",
        description="v16 PASS-GT acceptance-criteria evaluation report.",
    )
    p.add_argument("--gt", required=True, help="v16 GT JSONL path")
    p.add_argument("--answers", required=True, help="Answers JSONL (q_id + predicted_answer)")
    p.add_argument(
        "--clean_fsr",
        default=None,
        help="JSON file with FSR scores under clean retrieval (list or {q_id: score})",
    )
    p.add_argument(
        "--distractor_fsr",
        default=None,
        help="JSON file with FSR scores under distractor-injected retrieval",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write metrics JSON to this path (default: stdout only)",
    )
    p.add_argument(
        "--matrix_output",
        default=None,
        help="Write difficulty matrix JSON to this path",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    gt_path = Path(args.gt)
    if not gt_path.exists():
        logger.error(f"GT file not found: {gt_path}")
        sys.exit(1)

    ans_path = Path(args.answers)
    if not ans_path.exists():
        logger.error(f"Answers file not found: {ans_path}")
        sys.exit(1)

    logger.info(f"[evaluate_v16] Loading GT from {gt_path}")
    questions = load_gt(gt_path.stem, in_dir=gt_path.parent)
    logger.info(f"[evaluate_v16] {len(questions)} GT rows loaded")

    logger.info(f"[evaluate_v16] Loading answers from {ans_path}")
    answers = _load_answers(ans_path)
    logger.info(f"[evaluate_v16] {len(answers)} answer rows loaded")

    clean_fsr = _load_fsr_list(Path(args.clean_fsr)) if args.clean_fsr else None
    distractor_fsr = _load_fsr_list(Path(args.distractor_fsr)) if args.distractor_fsr else None

    strict_gold = {
        q.q_id: q.gold_answer
        for q in questions
        if q.twin_type is None
    }

    report = summary_report(
        questions,
        answers,
        clean_fsr=clean_fsr,
        distractor_fsr=distractor_fsr,
        original_strict_gold=strict_gold,
    )

    # Difficulty matrix
    matrix = DifficultyMatrix.from_questions(questions)
    matrix_dict = matrix.to_dict()
    report["difficulty_matrix_summary"] = matrix.cell_counts()

    # Print to stdout
    print("\n=== v16 PASS-GT Acceptance Criteria ===\n")

    abst = report["abstention"]
    print(
        f"[P4-AT] Abstention rate : {abst['rate']:.1%}  "
        f"({abst['correctly_abstained']}/{abst['total_at_twins']})  "
        f"{'✓ PASS' if abst['passes_70pct'] else '✗ FAIL'} (target ≥70%)"
    )

    cf = report["counterfactual"]
    print(
        f"[P4-CT] CT detection    : {cf['rate']:.1%}  "
        f"({cf['correctly_perturbed']}/{cf['total_ct_twins']})  "
        f"{'✓ PASS' if cf['passes_50pct'] else '✗ FAIL'} (target ≥50%)"
    )

    arm = report["arm_overlap"]
    if arm["n_steps"] > 0:
        print(
            f"[P5-ARM] NP overlap     : {arm['mean_np_overlap']:.3f} ± {arm['std_np_overlap']:.3f}  "
            f"{'✓ PASS' if arm['overlap_passes_35pct'] else '✗ FAIL'} (target ≤{arm.get('overlap_threshold', 0.35):.0%})"
        )
        print(
            f"[P5-ARM] Step NLI       : {arm['mean_nli_score']:.3f} ± {arm['std_nli_score']:.3f}  "
            f"{'✓ PASS' if arm['nli_passes_55'] else '✗ FAIL'} (target ≥0.55)"
        )
    else:
        print("[P5-ARM] No ARM reasoning_trace found in GT rows (ARM not used?)")

    if "distractor" in report:
        dist = report["distractor"]
        print(
            f"[P3-PASD] FSR drop     : {dist['drop_pp']:.1f}pp  "
            f"(clean={dist['clean_mean_fsr']:.3f}, distractor={dist['distractor_mean_fsr']:.3f})  "
            f"{'✓ PASS' if dist['passes_20pp'] else '✗ FAIL'} (target ≥20pp)"
        )

    print(f"\nDifficulty matrix: {len(matrix_dict['cells'])} cells, "
          f"{matrix_dict['total_questions']} total questions")
    for axis in ("hops", "semantic_distance", "adversarial"):
        print(f"  {axis}: {matrix.axis_summary(axis)}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"[evaluate_v16] Metrics written to {out_path}")

    if args.matrix_output:
        m_path = Path(args.matrix_output)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        m_path.write_text(json.dumps(matrix_dict, indent=2), encoding="utf-8")
        logger.info(f"[evaluate_v16] Difficulty matrix written to {m_path}")


if __name__ == "__main__":
    main()
