"""`python -m rag_gt.cli.retrieval_eval` -- facts-level retrieval evaluator.

Scores a retriever directly against ground-truth fact spans. Layered:

  L1   fact_recall            chunk_id of any supporting span retrieved
  L2   text_recall            fact text lexically present (partial_ratio ≥ τ_L2)
  L3   text_recall_l3         fact text semantically present (BGE cos ≥ τ_L3)
  L1∧L2  strict_recall
  L1∧L3  strict_recall_l13    ← headline replacement for RAGAS context_recall

Plus rank-weighted precision (RAGAS context_precision analogue), Hit@k, MRR,
per-role recall, per-difficulty breakdowns, and bootstrap 95% CIs.
No NLI model is loaded and no answer is needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.comparison.retrieval_metrics import (
    L2_PARTIAL_RATIO_THRESHOLD,
    L3_COSINE_THRESHOLD,
    CorpusRetrieval,
    PerQuestionRetrieval,
    evaluate_corpus,
)
from rag_gt.core.types import RetrievalLog
from rag_gt.storage.gt_io import load_gt


def _load_retrieval(path: Path) -> Dict[str, RetrievalLog]:
    out: Dict[str, RetrievalLog] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["q_id"]] = RetrievalLog(
                q_id=d["q_id"], retrieved_chunk_ids=d["retrieved_chunk_ids"]
            )
    return out


def _row_to_dict(r: PerQuestionRetrieval) -> dict:
    return {
        "q_id": r.q_id,
        "n_required_facts": r.n_required_facts,
        "n_retrieved_chunks": r.n_retrieved_chunks,
        "fact_recall": r.fact_recall,
        "fact_precision": r.fact_precision,
        "fact_f1": r.fact_f1,
        "fact_precision_rw": r.fact_precision_rw,
        "joint_fact_recall": r.joint_fact_recall,
        "required_group_recall": r.required_group_recall,
        "multi_hop_success": r.multi_hop_success,
        "overretrieval_penalty": r.overretrieval_penalty,
        "span_recall": r.span_recall,
        "span_precision": r.span_precision,
        "text_recall": r.text_recall,
        "strict_recall": r.strict_recall,
        "text_recall_l3": r.text_recall_l3,
        "strict_recall_l13": r.strict_recall_l13,
        "hit_at_1": r.hit_at_1,
        "hit_at_3": r.hit_at_3,
        "hit_at_5": r.hit_at_5,
        "hit_at_10": r.hit_at_10,
        "mrr": r.mrr,
        "missed_fact_ids": r.missed_fact_ids,
        "reasoning_depth": r.reasoning_depth,
        "semantic_distance": r.semantic_distance,
        "facts": [asdict(h) for h in r.facts],
    }


_DISPLAY_KEYS = (
    "fact_recall", "fact_precision", "fact_f1", "fact_precision_rw",
    "joint_fact_recall", "required_group_recall", "multi_hop_success",
    "overretrieval_penalty",
    "span_recall", "span_precision",
    "text_recall", "strict_recall",
    "text_recall_l3", "strict_recall_l13",
    "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "mrr",
)


def _write_json(corpus: CorpusRetrieval, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "retrieval_eval.json"
    payload = {
        "n_questions": corpus.n_questions,
        "means": corpus.means,
        "medians": corpus.medians,
        "stdevs": corpus.stdevs,
        "cis": {k: list(v) for k, v in corpus.cis.items()},
        "per_role_recall": corpus.per_role_recall,
        "fact_miss_rate": corpus.fact_miss_rate,
        "questions_zero_recall": corpus.questions_zero_recall,
        "questions_full_recall": corpus.questions_full_recall,
        "grouped_means": _stringify_keys(corpus.grouped_means),
        "rows": [_row_to_dict(r) for r in corpus.rows],
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def _stringify_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _stringify_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_keys(v) for v in obj]
    return obj


def _write_md(
    corpus: CorpusRetrieval, out_dir: Path,
    gt_label: str, ret_label: str, caveats: List[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "retrieval_eval.md"
    lines: List[str] = []
    lines.append("# Facts-level retrieval evaluation")
    lines.append("")
    lines.append(f"- GT: `{gt_label}`")
    lines.append(f"- Retrieval logs: `{ret_label}`")
    lines.append(f"- N questions: **{corpus.n_questions}**")
    lines.append(
        f"- Questions with full L1 fact recall: **{corpus.questions_full_recall}** "
        f"({_pct(corpus.questions_full_recall, corpus.n_questions)})"
    )
    lines.append(
        f"- Questions with zero L1 fact recall: **{corpus.questions_zero_recall}** "
        f"({_pct(corpus.questions_zero_recall, corpus.n_questions)})"
    )
    lines.append(f"- Overall L1 fact-miss rate: **{_fmt(corpus.fact_miss_rate)}**")
    lines.append("")

    if caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in caveats:
            lines.append(f"- {c}")
        lines.append("")

    lines.append("## Corpus-level metrics (with bootstrap 95% CI)")
    lines.append("")
    lines.append("| Metric | Mean | 95% CI | Median | Std |")
    lines.append("|---|---:|:---:|---:|---:|")
    for k in _DISPLAY_KEYS:
        lo, hi = corpus.cis.get(k, (float("nan"), float("nan")))
        lines.append(
            f"| `{k}` | {_fmt(corpus.means.get(k))} | "
            f"[{_fmt(lo)}, {_fmt(hi)}] | "
            f"{_fmt(corpus.medians.get(k))} | {_fmt(corpus.stdevs.get(k))} |"
        )
    lines.append("")

    lines.append("## Headline replacement metrics for RAGAS")
    lines.append("")
    lines.append("| Replaces | RAG_GT metric | Mean | 95% CI |")
    lines.append("|---|---|---:|:---:|")
    for ragas_name, k in (
        ("RAGAS `context_recall`", "strict_recall_l13"),
        ("RAGAS `context_precision`", "fact_precision_rw"),
    ):
        lo, hi = corpus.cis.get(k, (float("nan"), float("nan")))
        lines.append(
            f"| {ragas_name} | `{k}` | {_fmt(corpus.means.get(k))} | "
            f"[{_fmt(lo)}, {_fmt(hi)}] |"
        )
    lines.append("")

    lines.append("## Recall by fact role (L1)")
    lines.append("")
    if corpus.per_role_recall:
        lines.append("| Role | Recall |")
        lines.append("|---|---:|")
        for role, val in corpus.per_role_recall.items():
            lines.append(f"| `{role}` | {_fmt(val)} |")
    else:
        lines.append("_(no roles observed)_")
    lines.append("")

    # Per-difficulty breakdowns
    grouped = corpus.grouped_means or {}
    for field_name, label in (
        ("reasoning_depth", "Reasoning depth (n hops)"),
        ("semantic_distance", "Semantic distance"),
    ):
        if field_name in grouped and grouped[field_name]:
            lines.append(f"## Per-difficulty: {label}")
            lines.append("")
            lines.append(
                "| value | n | fact_recall (L1) | text_recall (L2) | "
                "text_recall_l3 (L3) | strict_recall_l13 | fact_precision_rw |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for value in sorted(grouped[field_name].keys(), key=str):
                row = grouped[field_name][value]
                lines.append(
                    f"| `{value}` | {int(row.get('n', 0))} | "
                    f"{_fmt(row.get('fact_recall'))} | "
                    f"{_fmt(row.get('text_recall'))} | "
                    f"{_fmt(row.get('text_recall_l3'))} | "
                    f"{_fmt(row.get('strict_recall_l13'))} | "
                    f"{_fmt(row.get('fact_precision_rw'))} |"
                )
            lines.append("")

    lines.append("## Per-question summary (top 30 by miss count)")
    lines.append("")
    lines.append(
        "| q_id | required | retrieved | fact_recall | flat_precision | "
        "fact_f1 | joint | hit@1 | hit@5 | mrr | missed |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    sorted_rows = sorted(
        corpus.rows, key=lambda r: (-len(r.missed_fact_ids), r.q_id)
    )[:30]
    for r in sorted_rows:
        lines.append(
            f"| `{r.q_id}` | {r.n_required_facts} | {r.n_retrieved_chunks} "
            f"| {_fmt(r.fact_recall)} | {_fmt(r.fact_precision)} "
            f"| {_fmt(r.fact_f1)} | {_fmt(r.joint_fact_recall)} "
            f"| {r.hit_at_1} | {r.hit_at_5} | {_fmt(r.mrr)} "
            f"| {', '.join(r.missed_fact_ids[:3]) or '—'} |"
        )
    lines.append("")

    lines.append("## Per-fact diagnostics (first 50 missed L1 facts)")
    lines.append("")
    flat_misses = [
        (r.q_id, h)
        for r in corpus.rows
        for h in r.facts
        if not h.retrieved
    ][:50]
    if flat_misses:
        lines.append(
            "| q_id | fact_id | role | required chunk_ids | "
            "L2 ratio | L3 cos | fact text |"
        )
        lines.append("|---|---|---|---|---:|---:|---|")
        for qid, h in flat_misses:
            lines.append(
                f"| `{qid}` | `{h.fact_id}` | `{h.role}` | "
                f"{', '.join(h.required_chunk_ids[:3]) or '—'} "
                f"| {_fmt(h.best_partial_ratio)} | {_fmt(h.best_cosine)} "
                f"| {_truncate(h.text, 80)} |"
            )
    else:
        lines.append("_(no missed facts)_")
    lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (x != x):
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{num / denom:.1%}"


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s.replace("|", "/").replace("\n", " ")
    return (s[: n - 1] + "…").replace("|", "/").replace("\n", " ")


def _print_console(corpus: CorpusRetrieval) -> None:
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  N questions               : {corpus.n_questions}")
    print(f"  Full-recall (L1)          : {corpus.questions_full_recall}")
    print(f"  Zero-recall (L1)          : {corpus.questions_zero_recall}")
    print(f"  Fact-miss rate (L1)       : {_fmt(corpus.fact_miss_rate)}")
    print()
    bar_width = 30
    for k in _DISPLAY_KEYS:
        m = corpus.means.get(k)
        if m is None or m != m:
            print(f"  [N/A] {k:<22s} | (no data)")
            continue
        lo, hi = corpus.cis.get(k, (float("nan"), float("nan")))
        filled = int(max(0.0, min(1.0, m)) * bar_width)
        bar = "#" * filled + "." * (bar_width - filled)
        print(
            f"        {k:<22s} |{bar}| mean={m:.3f} "
            f"95%CI=[{lo:.3f}, {hi:.3f}]"
        )
    print("=" * 60)


def _coverage_caveat(
    resolver: Optional[ChunkResolver], questions
) -> Optional[str]:
    if resolver is None:
        return None
    cov = resolver.verify_coverage(questions)
    if cov.is_complete:
        return None
    return (
        f"Chunk-resolver coverage: {cov.found}/{cov.requested} "
        f"({cov.coverage:.1%}) — {len(cov.missing)} required chunk_ids "
        f"missing from cache. L2/L3 metrics for these facts are computed "
        f"against the chunks that *were* retrievable."
    )


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag-gt-retrieval-eval",
        description="Facts-level retrieval evaluator with L1/L2/L3 layers.",
    )
    p.add_argument("--gt", required=True, help="Path to GT JSONL file.")
    p.add_argument("--retrieval", required=True, help="Path to retrieval_logs.jsonl.")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: data/eval_results/retrieval_<gt-stem>).",
    )
    p.add_argument(
        "--chunks-cache",
        default=None,
        help="Path to chunks.jsonl. Required for L2 / L3.",
    )
    p.add_argument(
        "--l2-threshold",
        type=float,
        default=L2_PARTIAL_RATIO_THRESHOLD,
        help=f"rapidfuzz.partial_ratio cutoff for L2 (default {L2_PARTIAL_RATIO_THRESHOLD}).",
    )
    p.add_argument(
        "--enable-l3",
        action="store_true",
        help="Enable L3 semantic recall (BGE cosine). Requires --chunks-cache.",
    )
    p.add_argument(
        "--l3-threshold",
        type=float,
        default=L3_COSINE_THRESHOLD,
        help=f"BGE cosine cutoff for L3 (default {L3_COSINE_THRESHOLD}).",
    )
    p.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Bootstrap resamples for 95%% CI (default 1000).",
    )
    p.add_argument(
        "--allow-partial-coverage",
        type=float,
        default=None,
        metavar="MIN_FRACTION",
        help=(
            "Maximum tolerated chunk-coverage fraction below 1.0 "
            "(e.g. 0.95 = accept ≥95%% coverage). Without this flag, any "
            "missing chunk fails fast."
        ),
    )
    args = p.parse_args()

    gt_path = Path(args.gt)
    ret_path = Path(args.retrieval)
    out_dir = Path(args.output_dir or f"data/eval_results/retrieval_{gt_path.stem}")

    print("=" * 60)
    print("  Facts-level retrieval evaluation")
    print("=" * 60)
    print(f"  GT             : {gt_path}")
    print(f"  Retrieval logs : {ret_path}")
    print(f"  Output dir     : {out_dir}")
    print("=" * 60 + "\n")

    questions = load_gt(gt_path.stem, in_dir=gt_path.parent or None)
    q_map = {q.q_id: q for q in questions}
    ret_logs = _load_retrieval(ret_path)

    resolver = None
    embedder = None
    caveats: List[str] = []
    if args.chunks_cache:
        resolver = ChunkResolver.from_cache(args.chunks_cache)
        print(f"  L2 enabled (chunks cache: {args.chunks_cache}, threshold: {args.l2_threshold})")

        cov = resolver.verify_coverage(questions)
        print(f"  Resolver coverage: {cov.found}/{cov.requested} ({cov.coverage:.1%})")
        if not cov.is_complete:
            min_required = args.allow_partial_coverage
            if min_required is None:
                sys.stderr.write(
                    f"[retrieval_eval] coverage incomplete ({cov.coverage:.1%}); "
                    "pass --allow-partial-coverage MIN_FRACTION to proceed.\n"
                )
                sys.exit(3)
            if cov.coverage < min_required:
                sys.stderr.write(
                    f"[retrieval_eval] coverage {cov.coverage:.1%} below "
                    f"required {min_required:.1%}.\n"
                )
                sys.exit(3)
            caveats.append(_coverage_caveat(resolver, questions))

        source_cov = resolver.verify_source_mapping(questions)
        if source_cov.requested:
            print(
                f"  Source-span mapping: {source_cov.found}/{source_cov.requested} "
                f"({source_cov.coverage:.1%})"
            )
            if not source_cov.is_complete:
                min_required = args.allow_partial_coverage
                if min_required is None:
                    sys.stderr.write(
                        f"[retrieval_eval] source-span mapping incomplete "
                        f"({source_cov.coverage:.1%}); chunks must expose canonical "
                        "char_start/char_end offsets or pass "
                        "--allow-partial-coverage MIN_FRACTION to proceed.\n"
                    )
                    sys.exit(3)
                if source_cov.coverage < min_required:
                    sys.stderr.write(
                        f"[retrieval_eval] source-span mapping {source_cov.coverage:.1%} "
                        f"below required {min_required:.1%}.\n"
                    )
                    sys.exit(3)
                caveats.append(
                    f"Source-span mapping: {source_cov.found}/{source_cov.requested} "
                    f"({source_cov.coverage:.1%}); unmapped facts fell back to stored chunk IDs."
                )

    if args.enable_l3:
        if resolver is None:
            sys.stderr.write("--enable-l3 requires --chunks-cache.\n")
            sys.exit(2)
        from rag_gt.core.models import MM
        embedder = MM.get_embedding()
        print(f"  L3 enabled (BGE embedder, threshold: {args.l3_threshold})")

    corpus = evaluate_corpus(
        q_map, ret_logs,
        resolver=resolver, l2_threshold=args.l2_threshold,
        embedder=embedder, l3_threshold=args.l3_threshold,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    if corpus.n_questions == 0:
        sys.stderr.write("No matching q_ids between GT and retrieval logs.\n")
        sys.exit(1)

    json_path = _write_json(corpus, out_dir)
    md_path = _write_md(corpus, out_dir, str(gt_path), str(ret_path), caveats)
    _print_console(corpus)
    print(f"\n[retrieval_eval] wrote: {md_path}")
    print(f"[retrieval_eval] wrote: {json_path}")


if __name__ == "__main__":
    main()
