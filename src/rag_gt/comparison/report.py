"""Write the comparison artifacts: comparison.json, summary.md, and PNG plots.

Matplotlib is forced to the Agg backend before pyplot is imported so headless
environments and CI never trip over a missing GUI backend.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from rag_gt.comparison.comparator import (
    METRIC_PAIRS,
    RAG_GT_ONLY_METRICS,
    RAGAS_ONLY_METRICS,
    ComparisonReport,
    PerQuestionRow,
)


def write_json(report: ComparisonReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "comparison.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    return p


def write_cost_json(report: ComparisonReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "cost_report.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report.cost.to_dict(), f, ensure_ascii=False, indent=2)
    return p


def write_markdown(report: ComparisonReport, out_dir: Path, plot_files: List[Path]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "summary.md"
    lines: List[str] = []
    lines.append("# RAG_GT vs RAGAS Comparison")
    lines.append("")
    lines.append(f"- N questions evaluated: **{len(report.rows)}**")
    lines.append(f"- RAGAS backend used: **{report.ragas_backend_used}**")
    lines.append(f"- Composite rank-agreement (Spearman): **{_fmt(report.rank_agreement)}**")
    lines.append("")
    lines.append("## Corpus-level metrics")
    lines.append("")
    lines.append("### RAG_GT (facts-based, no LLM API calls)")
    lines.append(_corpus_table(report.rag_gt_corpus))
    lines.append("")
    lines.append("### RAGAS")
    lines.append(_corpus_table(report.ragas_corpus))
    lines.append("")
    lines.append("## Correlation between paired metrics")
    lines.append("")
    lines.append(
        "| Pair | n | Pearson r | p | Spearman ρ | p | Kendall τ |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in report.correlations:
        lines.append(
            "| {label} | {n} | {pr} | {pp} | {sr} | {sp} | {kt} |".format(
                label=c.pair_label,
                n=c.n,
                pr=_fmt(c.pearson_r),
                pp=_fmt(c.pearson_p),
                sr=_fmt(c.spearman_rho),
                sp=_fmt(c.spearman_p),
                kt=_fmt(c.kendall_tau),
            )
        )
    lines.append("")
    lines.append("## Cost & wall-time")
    lines.append("")
    cost = report.cost
    lines.append(f"- RAG_GT wall time: **{cost.rag_gt_seconds:.2f}s** ({_per_q(cost.rag_gt_seconds, cost.n_questions)})")
    lines.append(f"- RAGAS wall time:  **{cost.ragas_seconds:.2f}s** ({_per_q(cost.ragas_seconds, cost.n_questions)})")
    speedup = cost.speedup
    if _is_finite(speedup):
        lines.append(f"- Speedup of RAG_GT over RAGAS: **{speedup:.2f}x**")
    lines.append(f"- RAG_GT API cost: **$0.00** (no LLM calls)")
    if cost.judge_model:
        lines.append(
            f"- RAGAS judge model: `{cost.judge_model}`"
            + (" *(price unknown — using $0)*" if cost.price_unknown else "")
        )
    lines.append(
        f"- RAGAS tokens: {cost.ragas_prompt_tokens} prompt + "
        f"{cost.ragas_completion_tokens} completion across "
        f"{cost.ragas_judge_calls} judge calls"
    )
    lines.append(
        f"- RAGAS estimated cost: **${cost.ragas_usd:.4f}** "
        f"(${cost.usd_per_question_ragas:.4f} per question)"
    )
    lines.append("")
    lines.append("## Observability matrix")
    lines.append("")
    lines.append("- RAG_GT-only metrics (no RAGAS counterpart): "
                 + ", ".join(f"`{m}`" for m in RAG_GT_ONLY_METRICS))
    lines.append("- RAGAS-only metrics (no RAG_GT counterpart): "
                 + ", ".join(f"`{m}`" for m in RAGAS_ONLY_METRICS))
    if report.ragas_metric_failures:
        nan_str = ", ".join(f"`{k}`={v}" for k, v in report.ragas_metric_failures.items() if v)
        if nan_str:
            lines.append(f"- RAGAS metric NaN counts: {nan_str}")
    lines.append("")
    if plot_files:
        lines.append("## Plots")
        lines.append("")
        for pf in plot_files:
            lines.append(f"![{pf.stem}]({pf.name})")
            lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(_recommendation(report))
    lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_plots(report: ComparisonReport, out_dir: Path) -> List[Path]:
    """Render scatter / bar / heatmap PNGs. Returns paths in display order."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    out_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []

    # 1. Scatter plots, one per metric pair.
    for rg_metric, rs_metric, label in METRIC_PAIRS:
        xs, ys = _xy(report.rows, rg_metric, rs_metric)
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xs, ys, alpha=0.7)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(f"RAG_GT {rg_metric}")
        ax.set_ylabel(f"RAGAS {rs_metric}")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        fname = out_dir / f"scatter_{rg_metric}__{rs_metric}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        out.append(fname)

    # 2. Bar: mean of paired metrics, side by side.
    pairs = []
    for rg_metric, rs_metric, label in METRIC_PAIRS:
        rg_mean = report.rag_gt_corpus.get(rg_metric, {}).get("mean", float("nan"))
        rs_mean = report.ragas_corpus.get(rs_metric, {}).get("mean", float("nan"))
        if _is_finite(rg_mean) and _is_finite(rs_mean):
            pairs.append((label, rg_mean, rs_mean))
    if pairs:
        labels = [p[0] for p in pairs]
        rg_vals = [p[1] for p in pairs]
        rs_vals = [p[2] for p in pairs]
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bw = 0.4
        ax.bar([i - bw / 2 for i in x], rg_vals, bw, label="RAG_GT")
        ax.bar([i + bw / 2 for i in x], rs_vals, bw, label="RAGAS")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean score")
        ax.set_title("Paired metric means")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fname = out_dir / "bar_metric_means.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        out.append(fname)

    # 3. Cost bar.
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["RAG_GT", "RAGAS"], [report.cost.rag_gt_usd, report.cost.ragas_usd])
    ax.set_ylabel("USD")
    ax.set_title("API cost per evaluation run")
    for i, v in enumerate([report.cost.rag_gt_usd, report.cost.ragas_usd]):
        ax.text(i, v, f" ${v:.4f}", ha="center", va="bottom")
    ax.grid(True, axis="y", alpha=0.3)
    fname = out_dir / "bar_cost_usd.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    out.append(fname)

    # 4. Wall-time bar.
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["RAG_GT", "RAGAS"], [report.cost.rag_gt_seconds, report.cost.ragas_seconds]
    )
    ax.set_ylabel("Seconds")
    ax.set_title("Wall time per evaluation run")
    for i, v in enumerate([report.cost.rag_gt_seconds, report.cost.ragas_seconds]):
        ax.text(i, v, f" {v:.1f}s", ha="center", va="bottom")
    ax.grid(True, axis="y", alpha=0.3)
    fname = out_dir / "bar_wall_time.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    out.append(fname)

    # 5. Correlation heatmap (paired metrics only).
    if report.correlations:
        labels = [c.pair_label for c in report.correlations]
        vals = [
            [c.pearson_r, c.spearman_rho, c.kendall_tau] for c in report.correlations
        ]
        fig, ax = plt.subplots(figsize=(6, max(2.4, 0.6 + 0.5 * len(labels))))
        im = ax.imshow(vals, vmin=-1, vmax=1, aspect="auto", cmap="RdBu_r")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Pearson r", "Spearman ρ", "Kendall τ"])
        ax.set_yticks(list(range(len(labels))))
        ax.set_yticklabels(labels)
        for i, row in enumerate(vals):
            for j, v in enumerate(row):
                if _is_finite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color=("white" if abs(v) > 0.5 else "black"))
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        ax.set_title("Correlation between paired metrics")
        fname = out_dir / "heatmap_correlations.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        out.append(fname)

    return out


def write_all(
    report: ComparisonReport, out_dir: Path | str, plots: bool = True
) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    paths["json"] = write_json(report, out)
    paths["cost"] = write_cost_json(report, out)
    plot_paths: List[Path] = []
    if plots:
        try:
            plot_paths = write_plots(report, out)
        except Exception as e:
            print(f"[report] plot generation failed: {e!r}; continuing without plots.")
    paths["markdown"] = write_markdown(report, out, plot_paths)
    paths["plots"] = plot_paths  # type: ignore[assignment]
    return paths


# ---------- helpers ----------


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and x == x and not math.isinf(x)


def _fmt(x: float) -> str:
    if not _is_finite(x):
        return "—"
    return f"{x:.3f}"


def _per_q(seconds: float, n: int) -> str:
    if n <= 0:
        return ""
    return f"{seconds / n:.3f}s/q"


def _xy(
    rows: List[PerQuestionRow], rg_metric: str, rs_metric: str
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for r in rows:
        x = r.rag_gt.get(rg_metric, float("nan"))
        y = r.ragas.get(rs_metric, float("nan"))
        if _is_finite(x) and _is_finite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


def _corpus_table(stats: Dict[str, Dict[str, float]]) -> str:
    if not stats:
        return "_(no metrics — RAG_GT/RAGAS side was skipped)_"
    lines = ["| Metric | n | Mean | Median | Std |", "|---|---:|---:|---:|---:|"]
    for k, v in stats.items():
        lines.append(
            f"| `{k}` | {int(v.get('n', 0))} | {_fmt(v.get('mean'))} | "
            f"{_fmt(v.get('median'))} | {_fmt(v.get('std'))} |"
        )
    return "\n".join(lines)


def _recommendation(report: ComparisonReport) -> str:
    rho = report.rank_agreement
    cost = report.cost
    if not _is_finite(rho):
        return (
            "Insufficient overlapping per-question data to compute rank "
            "agreement. Re-run with more questions or a non-zero limit."
        )
    if rho >= 0.6:
        verdict = (
            f"RAG_GT is a viable cheap baseline for RAGAS in this setting "
            f"(composite Spearman ρ = {rho:.2f}). It costs $0 vs ${cost.ragas_usd:.4f} "
            f"per run and finishes {('about ' + format(cost.speedup, '.1f') + 'x faster') if _is_finite(cost.speedup) else 'in less time'}."
        )
    elif rho >= 0.3:
        verdict = (
            f"RAG_GT broadly tracks RAGAS rank order (ρ = {rho:.2f}) at "
            f"a fraction of the cost ($0 vs ${cost.ragas_usd:.4f}). Use it for "
            f"routine evaluation and keep RAGAS for periodic spot checks."
        )
    else:
        verdict = (
            f"RAG_GT and RAGAS rank questions differently here (ρ = {rho:.2f}). "
            "Investigate the per-question scatter plots and the chunk-resolver "
            "coverage before treating either as the reference."
        )
    return verdict
