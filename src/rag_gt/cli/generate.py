"""`python -m rag_gt.cli.generate` -- Phase 1 GT generation CLI."""

from __future__ import annotations

import argparse
import os

from rag_gt.pipeline.gt_pipeline import run_gt_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag-gt-generate",
        description="RAG GT Pipeline V9.1 -- Ground Truth Generator",
    )
    p.add_argument("--input_dir", required=True, help="Dir with PDF files")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument(
        "--doc_type",
        default="UNKNOWN",
        help="ISO_STANDARD|KT_DOC|REPORT|NARRATIVE|UNKNOWN",
    )
    p.add_argument("--n_questions", type=int, default=8, help="Questions per document")
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--chunk_overlap", type=int, default=64)
    p.add_argument(
        "--fast_mode",
        action="store_true",
        help="Skip minimality + NLI checks (development only)",
    )
    p.add_argument(
        "--max_concurrent_llm_calls",
        type=int,
        default=None,
        help="Override concurrency for parallel question generation",
    )
    p.add_argument(
        "--disable_folder_heuristic",
        action="store_true",
        help="Skip Layer-0 path-based DocType inference",
    )
    p.add_argument(
        "--pdf_backend",
        choices=["legacy", "auto", "docling"],
        default=None,
        help=(
            "Override ingestion.pdf_backend for this run. "
            "legacy uses pdfplumber/PyMuPDF; auto tries Docling then falls back; "
            "docling requires Docling success."
        ),
    )
    p.add_argument(
        "--docling_export_format",
        choices=["markdown", "text"],
        default=None,
        help="Override Docling export format for this run.",
    )
    p.add_argument(
        "--docling_do_ocr",
        action="store_true",
        help="Enable Docling OCR for this run.",
    )
    p.add_argument(
        "--docling_page_range_size",
        type=int,
        default=None,
        help="Docling PDF page-window size for long documents.",
    )
    p.add_argument(
        "--question_mode",
        choices=["singlehop", "multihop", "mixed"],
        default="mixed",
        help="Control reasoning depth for generated questions.",
    )
    p.add_argument(
        "--min_hops",
        type=int,
        default=1,
        help="Minimum fact-chain depth to accept.",
    )
    p.add_argument(
        "--max_hops",
        type=int,
        default=3,
        help="Maximum fact-chain depth to accept.",
    )
    p.add_argument(
        "--min_distinct_chunks",
        type=int,
        default=1,
        help="Require a chain to span at least this many original chunks.",
    )
    p.add_argument(
        "--min_distinct_roles",
        type=int,
        default=1,
        help="Require a chain to contain at least this many fact roles.",
    )
    p.add_argument(
        "--min_distinct_pages",
        type=int,
        default=1,
        help="Require a chain to span at least this many source pages when page metadata exists.",
    )
    p.add_argument(
        "--min_char_gap",
        type=int,
        default=0,
        help="Require max(char_start)-min(char_start) across chain facts to be at least this value.",
    )
    p.add_argument(
        "--max_wall_minutes",
        type=float,
        default=None,
        help="Stop submitting/collecting generation work after this many wall-clock minutes.",
    )
    p.add_argument(
        "--progress_every",
        type=int,
        default=None,
        help="Log generation progress every N completed futures.",
    )
    p.add_argument(
        "--pair_budget",
        type=int,
        default=None,
        help="Override the v16 TF-SFG candidate-pair classification budget for this run.",
    )
    p.add_argument(
        "--max_live_api_calls",
        type=int,
        default=None,
        help="Hard stop before exceeding this many uncached LLM calls (paid-run safety cap).",
    )
    p.add_argument(
        "--disable_v16_singlehop_fallback",
        action="store_true",
        help="Disable v16 single-hop fallback candidates for a multi-hop-only validation run.",
    )
    p.add_argument(
        "--disable_v16_twins",
        action="store_true",
        help="Disable v16 abstention/counterfactual twin derivation for a validation run.",
    )
    p.add_argument(
        "--v16",
        action="store_true",
        help="Enable v16 PASS-GT mode (TF-SFG + QA-NLI gate + twins + ARM).",
    )
    p.add_argument(
        "--v16_2",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override configs/v16.yaml v16_2.enabled for this run.",
    )
    p.add_argument(
        "--trace_path",
        default=None,
        help="Structured JSONL trace path. Defaults to <output>.trace.jsonl.",
    )
    p.add_argument(
        "--disable_trace",
        action="store_true",
        help="Disable structured observability trace output.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.pdf_backend:
        os.environ["RAG_GT_PDF_BACKEND"] = args.pdf_backend
    if args.docling_export_format:
        os.environ["RAG_GT_DOCLING_EXPORT_FORMAT"] = args.docling_export_format
    if args.docling_do_ocr:
        os.environ["RAG_GT_DOCLING_DO_OCR"] = "true"
    if args.docling_page_range_size is not None:
        os.environ["RAG_GT_DOCLING_PAGE_RANGE_SIZE"] = str(
            args.docling_page_range_size
        )
    run_gt_pipeline(
        input_dir=args.input_dir,
        output_path=args.output,
        doc_type=args.doc_type,
        n_questions=args.n_questions,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        fast_mode=args.fast_mode,
        max_concurrent_llm_calls=args.max_concurrent_llm_calls,
        disable_folder_heuristic=args.disable_folder_heuristic,
        question_mode=args.question_mode,
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        min_distinct_chunks=args.min_distinct_chunks,
        min_distinct_roles=args.min_distinct_roles,
        min_distinct_pages=args.min_distinct_pages,
        min_char_gap=args.min_char_gap,
        max_wall_minutes=args.max_wall_minutes,
        progress_every=args.progress_every,
        pair_budget=args.pair_budget,
        max_live_api_calls=args.max_live_api_calls,
        disable_v16_singlehop_fallback=args.disable_v16_singlehop_fallback,
        disable_v16_twins=args.disable_v16_twins,
        v16=args.v16,
        v16_2=args.v16_2,
        trace_path=args.trace_path,
        enable_trace=not args.disable_trace,
    )


if __name__ == "__main__":
    main()
