"""`python -m rag_gt.cli.validate_replacement` — gate the RAG_GT-replaces-RAGAS claim.

Runs RAG_GT (with L3 enabled and rank-weighted precision) and RAGAS over the
same (GT, retrieval_logs) on a single retriever, aligns by `q_id`, and
computes the §7 validation evidence:

  - Pearson r and Spearman ρ between RAG_GT strict_recall_l13 ↔ RAGAS context_recall
    and fact_precision_rw ↔ context_precision, with bootstrap 95% CIs
  - Mean absolute error per metric
  - Top-5 disagreements with hand-traceable per-row evidence
  - Pass/fail checklist against §7.2 thresholds

Outputs `data/eval_results/replacement_validation/<run_id>/`:
  validation.json   per-question scores from both sides
  validation.md     human-readable verdict + tables + disagreement deep-dive
  scatter_recall.png, scatter_precision.png  visual evidence
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.comparison.ragas_adapter import RagasAdapter, RagasConfig
from rag_gt.comparison.retrieval_metrics import (
    L3_COSINE_THRESHOLD,
    evaluate_corpus,
)
from rag_gt.core.types import AnswerLog, RetrievalLog
from rag_gt.storage.gt_io import load_gt


# §7.2 thresholds.
PASS_SPEARMAN_LO = 0.55          # lower bound of bootstrap 95% CI
PASS_MAX_MAE = 0.20              # mean absolute error per metric
PASS_AUDIT_QUALITY = 0.80        # fact-corpus audit pass rate (informational)


@dataclass
class MetricPair:
    name: str
    rag_gt_key: str
    ragas_key: str


PAIRS = [
    MetricPair("recall (RAGAS context_recall ↔ RAG_GT strict_recall_l13)",
               "strict_recall_l13", "context_recall"),
    MetricPair("precision (RAGAS context_precision ↔ RAG_GT fact_precision_rw)",
               "fact_precision_rw", "context_precision"),
]


@dataclass
class CorrelationStats:
    pair_name: str
    n: int
    pearson_r: float
    spearman_rho: float
    spearman_rho_ci: Tuple[float, float]
    mae: float
    rag_gt_mean: float
    ragas_mean: float


@dataclass
class Disagreement:
    q_id: str
    rag_gt_value: float
    ragas_value: float
    abs_diff: float
    metric_name: str


@dataclass
class ValidationReport:
    n_questions: int
    n_judge_calls: int
    ragas_tokens_in: int
    ragas_tokens_out: int
    ragas_usd: float
    rag_gt_seconds: float
    ragas_seconds: float
    correlations: List[CorrelationStats]
    top_disagreements: Dict[str, List[Disagreement]]
    rows: List[Dict[str, float]]
    pass_recall: bool
    pass_precision: bool
    pass_mae_recall: bool
    pass_mae_precision: bool


def _load_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _ranks(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def _pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(dx2 * dy2) if dx2 and dy2 else 0.0
    return num / denom if denom else float("nan")


def _spearman(xs: List[float], ys: List[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def _bootstrap_spearman(
    xs: List[float], ys: List[float], n_resamples: int = 1000, seed: int = 0,
) -> Tuple[float, float]:
    if len(xs) < 3:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    rhos: List[float] = []
    n = len(xs)
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [xs[i] for i in idx]
        by = [ys[i] for i in idx]
        rho = _spearman(bx, by)
        if not math.isnan(rho):
            rhos.append(rho)
    if not rhos:
        return (float("nan"), float("nan"))
    rhos.sort()
    lo = rhos[int(len(rhos) * 0.025)]
    hi = rhos[int(len(rhos) * 0.975) - 1]
    return (lo, hi)


def _make_dummy_answers(q_ids: List[str], gt_path: Path) -> Dict[str, AnswerLog]:
    """RAGAS' adapter expects answers; for retrieval-only validation we use the
    gold answer as the predicted answer (this only affects faithfulness/
    answer_relevancy, which we do not score here)."""
    questions = load_gt(gt_path.stem, in_dir=gt_path.parent or None)
    q_map = {q.q_id: q for q in questions}
    return {
        qid: AnswerLog(q_id=qid, predicted_answer=q_map[qid].gold_answer, abstained=False)
        for qid in q_ids if qid in q_map
    }


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag-gt-validate-replacement",
        description="Validate the RAG_GT-replaces-RAGAS claim for retrieval metrics.",
    )
    p.add_argument("--gt", required=True)
    p.add_argument("--retrieval", required=True)
    p.add_argument("--chunks-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap n questions (for cost control).")
    p.add_argument("--ragas-llm", default="api", choices=("api", "ollama", "dry_run"))
    p.add_argument("--ragas-model", default=None)
    p.add_argument("--l3-threshold", type=float, default=L3_COSINE_THRESHOLD)
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--allow-partial-coverage", type=float, default=0.0,
                   help="Min coverage fraction tolerated; 0 = ignore coverage.")
    args = p.parse_args()

    gt_path = Path(args.gt)
    ret_path = Path(args.retrieval)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  RAG_GT vs RAGAS — Replacement validation (retrieval metrics)")
    print("=" * 70)
    print(f"  GT             : {gt_path}")
    print(f"  Retrieval logs : {ret_path}")
    print(f"  Chunks cache   : {args.chunks_cache}")
    print(f"  RAGAS backend  : {args.ragas_llm}")
    print(f"  Output dir     : {out_dir}")
    print("=" * 70 + "\n")

    # Load GT and retrieval logs
    questions = load_gt(gt_path.stem, in_dir=gt_path.parent or None)
    q_map = {q.q_id: q for q in questions}
    ret_logs = {
        d["q_id"]: RetrievalLog(
            q_id=d["q_id"], retrieved_chunk_ids=d["retrieved_chunk_ids"]
        )
        for d in _load_jsonl(ret_path)
    }
    matched = sorted(qid for qid in q_map if qid in ret_logs)
    if args.limit:
        matched = matched[: args.limit]
    if not matched:
        sys.stderr.write("No matching q_ids.\n")
        sys.exit(1)
    q_map_m = {qid: q_map[qid] for qid in matched}
    ret_logs_m = {qid: ret_logs[qid] for qid in matched}
    n = len(matched)
    print(f"  Evaluating n = {n} questions\n")

    # Resolver + embedder
    resolver = ChunkResolver.from_cache(args.chunks_cache)
    if args.allow_partial_coverage > 0:
        cov = resolver.verify_coverage(q_map_m.values())
        if cov.coverage < args.allow_partial_coverage:
            sys.stderr.write(
                f"Coverage {cov.coverage:.1%} below required {args.allow_partial_coverage:.1%}.\n"
            )
            sys.exit(3)
        source_cov = resolver.verify_source_mapping(q_map_m.values())
        if source_cov.requested and source_cov.coverage < args.allow_partial_coverage:
            sys.stderr.write(
                f"Source-span mapping {source_cov.coverage:.1%} below required "
                f"{args.allow_partial_coverage:.1%}. Chunks must expose canonical "
                "char_start/char_end offsets for source-anchored GT.\n"
            )
            sys.exit(3)
        if source_cov.requested:
            print(
                f"  Source-span mapping: {source_cov.found}/{source_cov.requested} "
                f"({source_cov.coverage:.1%})"
            )
    from rag_gt.core.models import MM
    embedder = MM.get_embedding()

    # ---- RAG_GT side ----
    print("[RAG_GT] computing layered retrieval metrics...")
    t0 = time.perf_counter()
    rg_corpus = evaluate_corpus(
        q_map_m, ret_logs_m,
        resolver=resolver, embedder=embedder, l3_threshold=args.l3_threshold,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    rag_gt_seconds = time.perf_counter() - t0
    rg_by_qid = {r.q_id: r for r in rg_corpus.rows}
    print(f"[RAG_GT] done in {rag_gt_seconds:.2f}s")

    # ---- RAGAS side ----
    print(f"[RAGAS] running with backend={args.ragas_llm}...")
    cfg = RagasConfig.from_env(backend=args.ragas_llm)
    if args.ragas_model:
        cfg.judge_model = args.ragas_model
    ragas_adapter = RagasAdapter(cfg, resolver)
    ans_logs = _make_dummy_answers(matched, gt_path)
    rg_for_dryrun = {
        qid: {
            "fact_span_recall": rg_by_qid[qid].fact_recall,
            "fact_span_precision": rg_by_qid[qid].fact_precision,
            "faithfulness": 0.0,
        }
        for qid in matched
    }
    t0 = time.perf_counter()
    rs = ragas_adapter.run(q_map_m, ret_logs_m, ans_logs, rg_for_dryrun)
    ragas_seconds = time.perf_counter() - t0
    rs_by_qid = {row["q_id"]: row for row in rs.per_question}
    print(f"[RAGAS] done in {ragas_seconds:.2f}s "
          f"(judge_calls={rs.judge_calls}, tokens={rs.prompt_tokens}+{rs.completion_tokens})")

    # ---- Per-question rows ----
    rows: List[Dict[str, float]] = []
    for qid in matched:
        rg = rg_by_qid[qid]
        rsr = rs_by_qid.get(qid, {})
        rows.append({
            "q_id": qid,
            "rag_gt_strict_recall_l13": rg.strict_recall_l13,
            "rag_gt_text_recall_l3": rg.text_recall_l3,
            "rag_gt_text_recall_l2": rg.text_recall,                # L2 lexical
            "rag_gt_text_recall_any": rg.text_recall_any,           # L1 ∨ L2 ∨ L3
            "rag_gt_fact_recall_l1": rg.fact_recall,
            "rag_gt_fact_precision_rw": rg.fact_precision_rw,
            "ragas_context_recall": rsr.get("context_recall"),
            "ragas_context_precision": rsr.get("context_precision"),
        })

    # ---- Correlation stats ----
    correlations: List[CorrelationStats] = []
    disagreements: Dict[str, List[Disagreement]] = {}
    for pair in PAIRS:
        rg_key_full = f"rag_gt_{pair.rag_gt_key}"
        rs_key_full = f"ragas_{pair.ragas_key}"
        xs: List[float] = []; ys: List[float] = []; aligned: List[Tuple[str, float, float]] = []
        for r in rows:
            x = r.get(rg_key_full); y = r.get(rs_key_full)
            if x is None or y is None: continue
            if isinstance(x, float) and x != x: continue
            if isinstance(y, float) and y != y: continue
            xs.append(float(x)); ys.append(float(y))
            aligned.append((r["q_id"], float(x), float(y)))
        if len(xs) < 3:
            correlations.append(CorrelationStats(
                pair.name, len(xs), float("nan"), float("nan"),
                (float("nan"), float("nan")), float("nan"),
                float("nan"), float("nan"),
            ))
            disagreements[pair.name] = []
            continue
        r_pearson = _pearson(xs, ys)
        rho = _spearman(xs, ys)
        rho_ci = _bootstrap_spearman(xs, ys, n_resamples=args.bootstrap_resamples)
        mae = sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)
        correlations.append(CorrelationStats(
            pair_name=pair.name, n=len(xs),
            pearson_r=r_pearson, spearman_rho=rho, spearman_rho_ci=rho_ci,
            mae=mae,
            rag_gt_mean=sum(xs) / len(xs), ragas_mean=sum(ys) / len(ys),
        ))
        # top-5 disagreements
        ranked = sorted(aligned, key=lambda t: -abs(t[1] - t[2]))[:5]
        disagreements[pair.name] = [
            Disagreement(q_id=qid, rag_gt_value=x, ragas_value=y,
                         abs_diff=abs(x - y), metric_name=pair.name)
            for qid, x, y in ranked
        ]

    # ---- Pass/fail checks ----
    recall_corr = correlations[0]
    precision_corr = correlations[1]
    pass_recall = (
        not math.isnan(recall_corr.spearman_rho_ci[0])
        and recall_corr.spearman_rho_ci[0] >= PASS_SPEARMAN_LO
    )
    pass_precision = (
        not math.isnan(precision_corr.spearman_rho_ci[0])
        and precision_corr.spearman_rho_ci[0] >= PASS_SPEARMAN_LO
    )
    pass_mae_recall = (not math.isnan(recall_corr.mae)) and recall_corr.mae <= PASS_MAX_MAE
    pass_mae_precision = (not math.isnan(precision_corr.mae)) and precision_corr.mae <= PASS_MAX_MAE

    # Cost estimate
    ragas_usd = _estimate_usd(cfg.judge_model, rs.prompt_tokens, rs.completion_tokens)

    report = ValidationReport(
        n_questions=n, n_judge_calls=rs.judge_calls,
        ragas_tokens_in=rs.prompt_tokens, ragas_tokens_out=rs.completion_tokens,
        ragas_usd=ragas_usd,
        rag_gt_seconds=rag_gt_seconds, ragas_seconds=ragas_seconds,
        correlations=correlations, top_disagreements=disagreements, rows=rows,
        pass_recall=pass_recall, pass_precision=pass_precision,
        pass_mae_recall=pass_mae_recall, pass_mae_precision=pass_mae_precision,
    )

    _write_json(report, out_dir, cfg.judge_model)
    _write_md(report, out_dir, gt_path, ret_path, cfg.judge_model)
    _write_plots(report, out_dir)
    _print_console(report)


def _estimate_usd(model: str, in_tokens: int, out_tokens: int) -> float:
    prices = {
        "gpt-4o":             {"in": 0.0025, "out": 0.0100},
        "gpt-4o-mini":        {"in": 0.00015, "out": 0.0006},
        "openai/gpt-oss-20b": {"in": 0.00020, "out": 0.0006},
        "openai/gpt-oss-120b": {"in": 0.00040, "out": 0.0015},
    }
    p = prices.get(model, {"in": 0.0, "out": 0.0})
    return (in_tokens / 1000.0) * p["in"] + (out_tokens / 1000.0) * p["out"]


def _write_json(report: ValidationReport, out_dir: Path, judge_model: str) -> None:
    payload = {
        "n_questions": report.n_questions,
        "judge_model": judge_model,
        "ragas_tokens_in": report.ragas_tokens_in,
        "ragas_tokens_out": report.ragas_tokens_out,
        "ragas_judge_calls": report.n_judge_calls,
        "ragas_usd": report.ragas_usd,
        "rag_gt_seconds": report.rag_gt_seconds,
        "ragas_seconds": report.ragas_seconds,
        "speedup_rag_gt_over_ragas": (
            report.ragas_seconds / report.rag_gt_seconds
            if report.rag_gt_seconds else None
        ),
        "correlations": [
            {
                "pair_name": c.pair_name, "n": c.n,
                "pearson_r": c.pearson_r,
                "spearman_rho": c.spearman_rho,
                "spearman_rho_ci": list(c.spearman_rho_ci),
                "mae": c.mae,
                "rag_gt_mean": c.rag_gt_mean, "ragas_mean": c.ragas_mean,
            }
            for c in report.correlations
        ],
        "top_disagreements": {
            name: [
                {
                    "q_id": d.q_id, "rag_gt_value": d.rag_gt_value,
                    "ragas_value": d.ragas_value, "abs_diff": d.abs_diff,
                }
                for d in disagreements
            ]
            for name, disagreements in report.top_disagreements.items()
        },
        "pass_criteria": {
            "spearman_lo_threshold": PASS_SPEARMAN_LO,
            "max_mae_threshold": PASS_MAX_MAE,
            "pass_recall_correlation": report.pass_recall,
            "pass_precision_correlation": report.pass_precision,
            "pass_mae_recall": report.pass_mae_recall,
            "pass_mae_precision": report.pass_mae_precision,
            "overall_pass": all([
                report.pass_recall, report.pass_precision,
                report.pass_mae_recall, report.pass_mae_precision,
            ]),
        },
        "rows": report.rows,
    }
    with open(out_dir / "validation.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _write_md(
    report: ValidationReport, out_dir: Path, gt_path: Path, ret_path: Path,
    judge_model: str,
) -> None:
    lines: List[str] = []
    lines.append("# Replacement validation: RAG_GT vs RAGAS (retrieval metrics)")
    lines.append("")
    lines.append(f"- GT: `{gt_path}`")
    lines.append(f"- Retrieval logs: `{ret_path}`")
    lines.append(f"- N questions: **{report.n_questions}**")
    lines.append(f"- RAGAS judge model: `{judge_model}`")
    lines.append("")

    lines.append("## Pass / fail summary")
    lines.append("")
    overall = all([
        report.pass_recall, report.pass_precision,
        report.pass_mae_recall, report.pass_mae_precision,
    ])
    verdict = "**PASS — RAG_GT replaces RAGAS for retrieval metrics**" if overall \
        else "**FAIL — see §5 of replace_ragas_retrieval_plan.md for mitigations**"
    lines.append(f"### Overall: {verdict}")
    lines.append("")
    lines.append("| Criterion | Threshold | Value | Pass |")
    lines.append("|---|---|---|:---:|")
    rc = report.correlations[0]; pc = report.correlations[1]
    lines.append(
        f"| Spearman ρ lower CI on recall pair | ≥ {PASS_SPEARMAN_LO} "
        f"| {_fmt(rc.spearman_rho_ci[0])} | {'✓' if report.pass_recall else '✗'} |"
    )
    lines.append(
        f"| Spearman ρ lower CI on precision pair | ≥ {PASS_SPEARMAN_LO} "
        f"| {_fmt(pc.spearman_rho_ci[0])} | {'✓' if report.pass_precision else '✗'} |"
    )
    lines.append(
        f"| MAE on recall pair | ≤ {PASS_MAX_MAE} "
        f"| {_fmt(rc.mae)} | {'✓' if report.pass_mae_recall else '✗'} |"
    )
    lines.append(
        f"| MAE on precision pair | ≤ {PASS_MAX_MAE} "
        f"| {_fmt(pc.mae)} | {'✓' if report.pass_mae_precision else '✗'} |"
    )
    lines.append("")

    lines.append("## Cost & wall-time")
    lines.append("")
    lines.append("| Side | Wall-time | API tokens | API USD |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| RAG_GT | **{report.rag_gt_seconds:.2f} s** ({report.rag_gt_seconds / report.n_questions:.3f} s/q) "
        f"| 0 | **$0.0000** |"
    )
    lines.append(
        f"| RAGAS  | {report.ragas_seconds:.2f} s ({report.ragas_seconds / report.n_questions:.3f} s/q) "
        f"| {report.ragas_tokens_in} in + {report.ragas_tokens_out} out "
        f"| **${report.ragas_usd:.4f}** ({report.ragas_usd / report.n_questions:.4f}/q) |"
    )
    if report.rag_gt_seconds > 0:
        lines.append(
            f"\nRAG_GT speedup: **{report.ragas_seconds / report.rag_gt_seconds:.2f}×** "
            f"(RAGAS wall-time / RAG_GT wall-time)"
        )
    lines.append("")

    lines.append("## Correlation between paired metrics")
    lines.append("")
    lines.append("| Pair | n | Pearson r | Spearman ρ | 95% CI on ρ | MAE | RAG_GT mean | RAGAS mean |")
    lines.append("|---|---:|---:|---:|:---:|---:|---:|---:|")
    for c in report.correlations:
        lines.append(
            f"| {c.pair_name} | {c.n} | {_fmt(c.pearson_r)} | {_fmt(c.spearman_rho)} "
            f"| [{_fmt(c.spearman_rho_ci[0])}, {_fmt(c.spearman_rho_ci[1])}] "
            f"| {_fmt(c.mae)} | {_fmt(c.rag_gt_mean)} | {_fmt(c.ragas_mean)} |"
        )
    lines.append("")

    lines.append("## Top-5 disagreements (per metric pair)")
    lines.append("")
    for pair_name, ds in report.top_disagreements.items():
        lines.append(f"### {pair_name}")
        lines.append("")
        if not ds:
            lines.append("_(no rows scored on both sides)_")
            lines.append("")
            continue
        lines.append("| q_id | RAG_GT | RAGAS | abs diff |")
        lines.append("|---|---:|---:|---:|")
        for d in ds:
            lines.append(
                f"| `{d.q_id}` | {d.rag_gt_value:.3f} | {d.ragas_value:.3f} | "
                f"**{d.abs_diff:.3f}** |"
            )
        lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append("![scatter_recall](scatter_recall.png)")
    lines.append("")
    lines.append("![scatter_precision](scatter_precision.png)")
    lines.append("")

    (out_dir / "validation.md").write_text("\n".join(lines), encoding="utf-8")


def _write_plots(report: ValidationReport, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[validate_replacement] matplotlib unavailable; skipping plots: {e}")
        return

    pairs = [
        ("rag_gt_strict_recall_l13", "ragas_context_recall",
         "RAG_GT strict_recall_l13", "RAGAS context_recall",
         report.correlations[0], "scatter_recall.png"),
        ("rag_gt_fact_precision_rw", "ragas_context_precision",
         "RAG_GT fact_precision_rw", "RAGAS context_precision",
         report.correlations[1], "scatter_precision.png"),
    ]
    for rg_key, rs_key, x_label, y_label, corr, fname in pairs:
        xs: List[float] = []; ys: List[float] = []
        for r in report.rows:
            x = r.get(rg_key); y = r.get(rs_key)
            if x is None or y is None: continue
            if isinstance(x, float) and x != x: continue
            if isinstance(y, float) and y != y: continue
            xs.append(float(x)); ys.append(float(y))
        if not xs: continue
        fig, ax = plt.subplots(figsize=(5.4, 5.4))
        ax.scatter(xs, ys, alpha=0.7)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(x_label); ax.set_ylabel(y_label)
        ax.set_title(
            f"{x_label} vs {y_label}\n"
            f"n={corr.n}, ρ={corr.spearman_rho:.2f} "
            f"[{corr.spearman_rho_ci[0]:.2f}, {corr.spearman_rho_ci[1]:.2f}], "
            f"MAE={corr.mae:.2f}"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=120)
        plt.close(fig)


def _fmt(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:.3f}"


def _print_console(report: ValidationReport) -> None:
    # Use ASCII-safe characters so this works in cp1252 console codepages.
    print("=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)
    overall = all([
        report.pass_recall, report.pass_precision,
        report.pass_mae_recall, report.pass_mae_precision,
    ])
    print(f"  Overall verdict: {'PASS' if overall else 'FAIL'}")
    for c in report.correlations:
        # Replace any non-ASCII glyphs in the pair_name for safety on Windows.
        name = c.pair_name.replace("↔", "<->").replace("ρ", "rho")
        print(f"\n  {name}")
        print(f"    n={c.n}  Pearson r={c.pearson_r:.3f}  "
              f"Spearman rho={c.spearman_rho:.3f} 95%CI=[{c.spearman_rho_ci[0]:.3f}, {c.spearman_rho_ci[1]:.3f}]")
        print(f"    MAE={c.mae:.3f}  RAG_GT mean={c.rag_gt_mean:.3f}  RAGAS mean={c.ragas_mean:.3f}")
    print(f"\n  RAG_GT wall-time : {report.rag_gt_seconds:.2f} s")
    print(f"  RAGAS wall-time  : {report.ragas_seconds:.2f} s")
    print(f"  RAGAS API spend  : ${report.ragas_usd:.4f} "
          f"({report.ragas_tokens_in}+{report.ragas_tokens_out} tokens)")
    print("=" * 70)


if __name__ == "__main__":
    main()
