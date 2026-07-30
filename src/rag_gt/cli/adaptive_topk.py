"""Adaptive top-k selection for retrieval logs.

This is an optimization/evaluation helper, not a live production retriever.
It uses the ground truth to keep the smallest retrieval prefix that satisfies
the requested source-fact coverage target.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.comparison.retrieval_metrics import (
    CorpusRetrieval,
    PerQuestionRetrieval,
    evaluate_corpus,
    evaluate_question,
)
from rag_gt.core.types import QuestionGT, RetrievalLog
from rag_gt.storage.gt_io import load_gt


def _load_retrieval_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _logs_by_qid(rows: Iterable[dict]) -> Dict[str, RetrievalLog]:
    out: Dict[str, RetrievalLog] = {}
    for d in rows:
        out[str(d["q_id"])] = RetrievalLog(
            q_id=str(d["q_id"]),
            retrieved_chunk_ids=[str(cid) for cid in d.get("retrieved_chunk_ids", [])],
        )
    return out


def _target_met(result: PerQuestionRetrieval, target: str, min_fact_recall: float) -> bool:
    if target == "joint":
        return result.joint_fact_recall >= 1.0
    if target == "group":
        return result.required_group_recall >= 1.0
    if target == "fact_recall":
        return result.fact_recall >= min_fact_recall
    raise ValueError(f"unknown adaptive target: {target}")


def _choose_prefix(
    question: QuestionGT,
    candidate_ids: Sequence[str],
    *,
    ks: Sequence[int],
    resolver: ChunkResolver | None,
    target: str,
    min_fact_recall: float,
) -> tuple[List[str], PerQuestionRetrieval, bool, int]:
    if not ks:
        raise ValueError("ks must contain at least one value")

    max_k = max(ks)
    best_result: PerQuestionRetrieval | None = None
    best_prefix: List[str] = list(candidate_ids[:max_k])
    best_k = min(max_k, len(candidate_ids))

    for k in ks:
        prefix = list(candidate_ids[:k])
        result = evaluate_question(
            question,
            RetrievalLog(q_id=question.q_id, retrieved_chunk_ids=prefix),
            resolver=resolver,
        )
        if best_result is None:
            best_result = result
        if _target_met(result, target, min_fact_recall):
            return prefix, result, True, min(k, len(candidate_ids))
        best_result = result
        best_prefix = prefix
        best_k = min(k, len(candidate_ids))

    assert best_result is not None
    return best_prefix, best_result, False, best_k


def adapt_retrieval(
    questions: Sequence[QuestionGT],
    retrieval_rows: Sequence[dict],
    *,
    ks: Sequence[int],
    resolver: ChunkResolver | None = None,
    target: str = "joint",
    min_fact_recall: float = 1.0,
) -> tuple[List[dict], List[dict]]:
    q_map = {q.q_id: q for q in questions}
    out_rows: List[dict] = []
    decision_rows: List[dict] = []

    for row in retrieval_rows:
        qid = str(row["q_id"])
        candidate_ids = [str(cid) for cid in row.get("retrieved_chunk_ids", [])]
        question = q_map.get(qid)
        if question is None:
            prefix = candidate_ids[: max(ks)]
            out_rows.append(
                {
                    "q_id": qid,
                    "retrieved_chunk_ids": prefix,
                    "adaptive_topk": len(prefix),
                    "adaptive_target_met": False,
                    "adaptive_reason": "q_id_not_in_gt",
                }
            )
            decision_rows.append(out_rows[-1])
            continue

        prefix, result, met, chosen_k = _choose_prefix(
            question,
            candidate_ids,
            ks=ks,
            resolver=resolver,
            target=target,
            min_fact_recall=min_fact_recall,
        )
        out = {
            "q_id": qid,
            "retrieved_chunk_ids": prefix,
            "adaptive_topk": chosen_k,
            "adaptive_target": target,
            "adaptive_target_met": met,
            "adaptive_fact_recall": result.fact_recall,
            "adaptive_joint_fact_recall": result.joint_fact_recall,
            "adaptive_required_group_recall": result.required_group_recall,
            "adaptive_fact_precision": result.fact_precision,
            "adaptive_fact_f1": result.fact_f1,
            "adaptive_missed_fact_ids": result.missed_fact_ids,
        }
        out_rows.append(out)
        decision_rows.append({k: v for k, v in out.items() if k != "retrieved_chunk_ids"})

    return out_rows, decision_rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _corpus_payload(corpus: CorpusRetrieval) -> dict:
    return {
        "n_questions": corpus.n_questions,
        "means": corpus.means,
        "questions_full_recall": corpus.questions_full_recall,
        "questions_zero_recall": corpus.questions_zero_recall,
        "fact_miss_rate": corpus.fact_miss_rate,
    }


def _write_summary(
    path: Path,
    *,
    gt_path: Path,
    retrieval_path: Path,
    output_path: Path,
    ks: Sequence[int],
    target: str,
    min_fact_recall: float,
    decision_rows: Sequence[dict],
    before: CorpusRetrieval,
    after: CorpusRetrieval,
) -> None:
    selected = Counter(int(row.get("adaptive_topk", 0) or 0) for row in decision_rows)
    target_met = sum(1 for row in decision_rows if row.get("adaptive_target_met"))
    chosen_ks = [int(row.get("adaptive_topk", 0) or 0) for row in decision_rows]

    payload = {
        "gt": str(gt_path),
        "input_retrieval": str(retrieval_path),
        "output_retrieval": str(output_path),
        "ks": list(ks),
        "target": target,
        "min_fact_recall": min_fact_recall,
        "n_questions": len(decision_rows),
        "target_met": target_met,
        "target_met_rate": target_met / len(decision_rows) if decision_rows else 0.0,
        "selected_k_counts": dict(sorted(selected.items())),
        "mean_selected_k": mean(chosen_ks) if chosen_ks else 0.0,
        "before": _corpus_payload(before),
        "after": _corpus_payload(after),
        "decisions": list(decision_rows),
        "caveat": (
            "Adaptive top-k uses ground-truth fact coverage to choose a prefix. "
            "Use it for benchmark optimization/diagnosis, not as a production "
            "retrieval policy unless an independent coverage estimator is added."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, summary: dict) -> None:
    before = summary["before"]["means"]
    after = summary["after"]["means"]
    lines = [
        "# Adaptive top-k retrieval optimization",
        "",
        "Adaptive top-k keeps the smallest retrieval prefix that satisfies the configured source-fact coverage target.",
        "",
        "> Caveat: this is a benchmark optimization/diagnostic tool because it uses ground truth to choose the prefix.",
        "",
        "## Inputs",
        "",
        f"- GT: `{summary['gt']}`",
        f"- Input retrieval: `{summary['input_retrieval']}`",
        f"- Output retrieval: `{summary['output_retrieval']}`",
        f"- k ladder: `{summary['ks']}`",
        f"- target: `{summary['target']}`",
        "",
        "## Selection",
        "",
        f"- Questions: `{summary['n_questions']}`",
        f"- Target met: `{summary['target_met']}` / `{summary['n_questions']}` ({summary['target_met_rate']:.1%})",
        f"- Mean selected k: `{summary['mean_selected_k']:.2f}`",
        f"- Selected k counts: `{summary['selected_k_counts']}`",
        "",
        "## Before vs After",
        "",
        "| Metric | before | after |",
        "|---|---:|---:|",
    ]
    for key in (
        "fact_recall",
        "joint_fact_recall",
        "fact_precision",
        "fact_f1",
        "fact_precision_rw",
        "mrr",
        "overretrieval_penalty",
    ):
        lines.append(f"| {key} | {before.get(key, 0.0):.3f} | {after.get(key, 0.0):.3f} |")
    lines.extend(
        [
            "",
            "## Per-question decisions",
            "",
            "| q_id | selected k | target met | fact recall | joint recall | precision | F1 | missed facts |",
            "|---|---:|:---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["decisions"]:
        missed = ", ".join(row.get("adaptive_missed_fact_ids") or [])
        lines.append(
            f"| `{row['q_id']}` | {row.get('adaptive_topk', 0)} | "
            f"{'yes' if row.get('adaptive_target_met') else 'no'} | "
            f"{float(row.get('adaptive_fact_recall', 0.0)):.3f} | "
            f"{float(row.get('adaptive_joint_fact_recall', 0.0)):.3f} | "
            f"{float(row.get('adaptive_fact_precision', 0.0)):.3f} | "
            f"{float(row.get('adaptive_fact_f1', 0.0)):.3f} | "
            f"{missed or '—'} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag-gt-adaptive-topk",
        description="Choose the smallest retrieval prefix that satisfies GT fact coverage.",
    )
    p.add_argument("--gt", required=True, help="Path to GT JSONL file.")
    p.add_argument("--retrieval", required=True, help="Ordered retrieval JSONL to adapt.")
    p.add_argument("--chunks-cache", default=None, help="Optional active chunk cache for source-span mapping.")
    p.add_argument("--output", required=True, help="Output adaptive retrieval JSONL.")
    p.add_argument("--summary-json", default=None, help="Output summary JSON path.")
    p.add_argument("--summary-md", default=None, help="Output summary Markdown path.")
    p.add_argument("--ks", default="5,8,10", help="Comma-separated k ladder, default: 5,8,10.")
    p.add_argument(
        "--target",
        choices=("joint", "group", "fact_recall"),
        default="joint",
        help="Coverage target for stopping early.",
    )
    p.add_argument(
        "--min-fact-recall",
        type=float,
        default=1.0,
        help="Minimum fact_recall when --target fact_recall is used.",
    )
    p.add_argument("--bootstrap-resamples", type=int, default=300)
    args = p.parse_args()

    gt_path = Path(args.gt)
    retrieval_path = Path(args.retrieval)
    output_path = Path(args.output)
    summary_json = Path(args.summary_json) if args.summary_json else output_path.with_suffix(".summary.json")
    summary_md = Path(args.summary_md) if args.summary_md else output_path.with_suffix(".summary.md")
    ks = sorted({int(k.strip()) for k in args.ks.split(",") if k.strip()})

    questions = load_gt(gt_path.stem, in_dir=gt_path.parent)
    q_map = {q.q_id: q for q in questions}
    retrieval_rows = _load_retrieval_rows(retrieval_path)
    before_logs = _logs_by_qid(retrieval_rows)
    resolver = ChunkResolver.from_cache(args.chunks_cache) if args.chunks_cache else None

    out_rows, decision_rows = adapt_retrieval(
        questions,
        retrieval_rows,
        ks=ks,
        resolver=resolver,
        target=args.target,
        min_fact_recall=args.min_fact_recall,
    )
    _write_jsonl(output_path, out_rows)

    after_logs = _logs_by_qid(out_rows)
    before = evaluate_corpus(q_map, before_logs, resolver=resolver, bootstrap_resamples=args.bootstrap_resamples)
    after = evaluate_corpus(q_map, after_logs, resolver=resolver, bootstrap_resamples=args.bootstrap_resamples)
    _write_summary(
        summary_json,
        gt_path=gt_path,
        retrieval_path=retrieval_path,
        output_path=output_path,
        ks=ks,
        target=args.target,
        min_fact_recall=args.min_fact_recall,
        decision_rows=decision_rows,
        before=before,
        after=after,
    )
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    _write_markdown(summary_md, summary)

    print("=" * 60)
    print("  Adaptive top-k retrieval")
    print("=" * 60)
    print(f"  GT             : {gt_path}")
    print(f"  Input retrieval: {retrieval_path}")
    print(f"  Output         : {output_path}")
    print(f"  k ladder       : {ks}")
    print(f"  target         : {args.target}")
    print(f"  target met     : {summary['target_met']}/{summary['n_questions']} ({summary['target_met_rate']:.1%})")
    print(f"  selected k     : {summary['selected_k_counts']} mean={summary['mean_selected_k']:.2f}")
    print("  before/after:")
    for key in ("fact_recall", "joint_fact_recall", "fact_precision", "fact_f1"):
        print(f"    {key:<20s} {before.means.get(key, 0.0):.3f} -> {after.means.get(key, 0.0):.3f}")
    print(f"  Summary JSON   : {summary_json}")
    print(f"  Summary MD     : {summary_md}")
    print("=" * 60)


if __name__ == "__main__":
    main()
