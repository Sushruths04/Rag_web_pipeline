"""`python -m rag_gt.cli.compare` -- benchmark RAG_GT against RAGAS.

Inputs (mirrors cli/evaluate.py):
    --gt          Path to GT JSONL (e.g. datagt/test20.jsonl)
    --retrieval   retrieval_logs.jsonl
    --answers     answer_logs.jsonl

Outputs (in --output-dir):
    comparison.json     full per-question dump (both sides aligned by q_id)
    cost_report.json    wall time + token cost per side
    summary.md          corpus tables + correlations + cost + plots
    *.png               scatter, bar, heatmap (omitted with --no-plots)

Chunk-text resolution requires data/cache/chunks.jsonl (rebuild with
`python -m rag_gt.cli.cache_chunks`). The harness defaults --ragas-llm to
`dry_run` when `LLM_BACKEND` env is unset to prevent surprise API spend.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from rag_gt.comparison.chunk_resolver import ChunkResolver
from rag_gt.comparison.comparator import Comparator
from rag_gt.comparison.cost_tracker import CostTracker
from rag_gt.comparison.ragas_adapter import RagasConfig
from rag_gt.comparison.report import write_all


def _default_backend() -> str:
    return (os.getenv("LLM_BACKEND") or "dry_run").strip().lower()


def _default_output_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("data/eval_results") / f"run_{ts}")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag-gt-compare",
        description="RAG_GT vs RAGAS comparison harness.",
    )
    p.add_argument("--gt", required=True, help="Path to GT JSONL file.")
    p.add_argument("--retrieval", required=True, help="Path to retrieval_logs.jsonl.")
    p.add_argument("--answers", required=True, help="Path to answer_logs.jsonl.")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: data/eval_results/run_{timestamp}).",
    )
    p.add_argument(
        "--chunks-cache",
        default="data/cache/chunks.jsonl",
        help="Path to the chunks cache JSONL (build with `rag-gt-cache-chunks`).",
    )
    p.add_argument(
        "--ragas-llm",
        choices=("dry_run", "api", "ollama"),
        default=None,
        help="RAGAS judge backend. Default: dry_run when LLM_BACKEND is unset.",
    )
    p.add_argument(
        "--ragas-model",
        default=None,
        help="Override the RAGAS judge model (else from API_GT_MODEL / OLLAMA_GT_MODEL).",
    )
    p.add_argument(
        "--ragas-embed",
        default=None,
        help="Override the RAGAS embedding model (default: BAAI/bge-base-en-v1.5).",
    )
    p.add_argument("--limit", type=int, default=None, help="Max questions to evaluate.")
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Proceed even if the chunks cache is missing some required chunk_ids.",
    )
    p.add_argument("--skip-rag-gt", action="store_true")
    p.add_argument("--skip-ragas", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--price-table",
        default=None,
        help="Optional JSON override for the per-1k-token USD price table.",
    )
    args = p.parse_args()

    backend = (args.ragas_llm or _default_backend()).lower()
    out_dir = Path(args.output_dir or _default_output_dir())

    print("=" * 60)
    print("  RAG_GT vs RAGAS Comparison")
    print("=" * 60)
    print(f"  GT             : {args.gt}")
    print(f"  Retrieval logs : {args.retrieval}")
    print(f"  Answer logs    : {args.answers}")
    print(f"  Chunks cache   : {args.chunks_cache}")
    print(f"  RAGAS backend  : {backend}")
    print(f"  Output dir     : {out_dir}")
    print("=" * 60 + "\n")

    cache_path = Path(args.chunks_cache)
    if not cache_path.exists():
        sys.stderr.write(
            "[compare] chunks cache missing.\n"
            f"  Run: python -m rag_gt.cli.cache_chunks --input_dir data/docs --output {cache_path}\n"
        )
        sys.exit(2)

    resolver = ChunkResolver.from_cache(cache_path)

    # Coverage check before we burn any time.
    from rag_gt.storage.gt_io import load_gt

    gt_path = Path(args.gt)
    questions = load_gt(gt_path.stem, in_dir=gt_path.parent or None)
    coverage = resolver.verify_coverage(questions)
    print(
        f"[compare] chunks cache covers {coverage.found}/{coverage.requested} "
        f"required chunk_ids ({coverage.coverage:.1%})"
    )
    if not coverage.is_complete and not args.allow_partial:
        sample = ", ".join(coverage.missing[:5])
        sys.stderr.write(
            f"[compare] missing {len(coverage.missing)} chunk_ids; "
            f"first few: {sample}\n"
            "Re-run cache_chunks with the same --chunk_size / --chunk_overlap "
            "as GT generation, or pass --allow-partial.\n"
        )
        sys.exit(3)

    cfg = RagasConfig.from_env(backend=backend)
    if args.ragas_model:
        cfg.judge_model = args.ragas_model
    if args.ragas_embed:
        cfg.embed_model = args.ragas_embed

    cost = CostTracker(price_table_path=args.price_table, judge_model=cfg.judge_model)

    comparator = Comparator(
        gt_path=args.gt,
        retrieval_path=args.retrieval,
        answers_path=args.answers,
        resolver=resolver,
        ragas_cfg=cfg,
        cost_tracker=cost,
        limit=args.limit,
        skip_rag_gt=args.skip_rag_gt,
        skip_ragas=args.skip_ragas,
    )

    report = comparator.run()
    paths = write_all(report, out_dir, plots=not args.no_plots)

    _print_console_summary(report)
    print(f"\n[compare] wrote: {paths['markdown']}")
    print(f"[compare] wrote: {paths['json']}")
    print(f"[compare] wrote: {paths['cost']}")
    if paths.get("plots"):
        print(f"[compare] wrote {len(paths['plots'])} plot(s) under {out_dir}")
    sys.exit(0)


def _print_console_summary(report) -> None:
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  N questions: {len(report.rows)}")
    print(f"  RAGAS backend: {report.ragas_backend_used}")
    print(f"  Composite rank-agreement (Spearman): {_fmt(report.rank_agreement)}")
    for c in report.correlations:
        print(
            f"  {c.pair_label:<48s} | n={c.n:>3d} | "
            f"r={_fmt(c.pearson_r)} | rho={_fmt(c.spearman_rho)} | tau={_fmt(c.kendall_tau)}"
        )
    print(
        f"  Cost: RAG_GT $0 vs RAGAS ${report.cost.ragas_usd:.4f} "
        f"({report.cost.ragas_prompt_tokens}+{report.cost.ragas_completion_tokens} tokens)"
    )
    print(
        f"  Wall time: RAG_GT {report.cost.rag_gt_seconds:.2f}s "
        f"vs RAGAS {report.cost.ragas_seconds:.2f}s"
    )
    print("=" * 60)


def _fmt(x: float) -> str:
    if not isinstance(x, (int, float)) or x != x:
        return "—"
    return f"{x:.3f}"


if __name__ == "__main__":
    main()
