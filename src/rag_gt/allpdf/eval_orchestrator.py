"""Full GRAFT allpdf pipeline evaluation orchestrator.

Runs the full pipeline + Stage 10 evaluation on every document in a corpus,
then produces a consolidated cross-doc evaluation report.

Usage:
    python -m rag_gt.allpdf.eval_orchestrator \\
        --manifest data/test_corpus_allpdf/corpus_manifest.json \\
        --out-dir data/test_corpus_allpdf/eval_run \\
        --max-pairs 150 --target-chains 8 \\
        [--dry-run]  # stages 0-4 only (no LLM graph or QGen)

Produces:
    data/test_corpus_allpdf/eval_run/
        {doc_id}/pipeline_result.json   (per-doc pipeline gates)
        {doc_id}/pipeline_report.md
        {doc_id}/eval_result.json        (Stage 10 retrieval metrics)
    cross_doc_eval_report.md             (consolidated table)
    cross_doc_eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

from loguru import logger

from rag_gt.allpdf.evaluate import evaluate_pairs
from rag_gt.allpdf.pipeline import run_doc_pipeline


def run_and_evaluate(
    doc_id: str,
    pdf_path: str,
    out_dir: str,
    *,
    dry_run: bool = False,
    max_pairs: int = 150,
    target_chains: int = 8,
    chain_depth: int = 2,
    llm_chunk_cap: int = 30,
    docling_page_cap: int = 8,
    seed: int = 42,
) -> dict:
    """Run pipeline + Stage 10 evaluation for one document."""
    t0 = time.time()

    # Run the pipeline.
    pipeline_result = run_doc_pipeline(
        doc_id, pdf_path, out_dir,
        dry_run=dry_run,
        max_pairs=max_pairs,
        target_chains=target_chains,
        chain_depth=chain_depth,
        llm_chunk_cap=llm_chunk_cap,
        docling_page_cap=docling_page_cap,
        seed=seed,
    )

    eval_summary = None
    if not dry_run:
        # Load Stage-2 chunks + Stage-7 pairs from checkpoints.
        ckpt = Path(out_dir) / "checkpoints"
        pairs_path = ckpt / "s7_pairs.json"
        chunks_path = ckpt / "s2_chunks_full.json"
        if pairs_path.exists() and chunks_path.exists():
            pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            eval_result = evaluate_pairs(doc_id, pairs, chunks)
            eval_summary = eval_result.summary()
            eval_out = Path(out_dir) / "eval_result.json"
            eval_out.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
            logger.info(
                f"[eval_orch] {doc_id}: eval done — "
                f"fact_recall@5={eval_summary['fact_recall@5']:.3f} "
                f"hit@5={eval_summary['hit@5']:.3f} "
                f"mrr={eval_summary['mrr']:.3f}"
            )
        else:
            logger.warning(f"[eval_orch] {doc_id}: pairs/chunks not found, skipping Stage 10")

    return {
        "doc_id": doc_id,
        "pipeline": pipeline_result,
        "evaluation": eval_summary,
        "total_wall_sec": round(time.time() - t0, 1),
    }


def run_corpus(
    docs: List[dict],
    out_dir: str,
    *,
    dry_run: bool = False,
    max_pairs: int = 150,
    target_chains: int = 8,
    chain_depth: int = 2,
    llm_chunk_cap: int = 30,
    docling_page_cap: int = 8,
) -> dict:
    """Run the pipeline on every doc in the corpus and collect results."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results = []
    t0 = time.time()

    for doc in docs:
        doc_id = doc["doc_id"]
        pdf_path = doc["path"]
        doc_out = str(out / doc_id)

        logger.info(f"\n{'='*60}\n[eval_orch] Processing: {doc_id}\n{'='*60}")
        try:
            result = run_and_evaluate(
                doc_id, pdf_path, doc_out,
                dry_run=dry_run,
                max_pairs=max_pairs,
                target_chains=target_chains,
                chain_depth=chain_depth,
                llm_chunk_cap=llm_chunk_cap,
                docling_page_cap=docling_page_cap,
            )
        except Exception as e:
            logger.error(f"[eval_orch] {doc_id} FAILED: {type(e).__name__}: {e}")
            result = {"doc_id": doc_id, "pipeline": None, "evaluation": None, "error": str(e)}

        all_results.append(result)

    report = {
        "n_docs": len(docs),
        "dry_run": dry_run,
        "wall_sec": round(time.time() - t0, 1),
        "results": all_results,
    }
    report_path = out / "cross_doc_eval.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_cross_doc_report(report, str(out / "cross_doc_eval_report.md"))
    return report


def _write_cross_doc_report(report: dict, path: str) -> None:
    lines = [
        "# GRAFT All-PDF Pipeline — Cross-Document Evaluation Report", "",
        f"- Documents: {report['n_docs']}",
        f"- Dry run: {report['dry_run']}",
        f"- Total wall time: {report['wall_sec']}s", "",
    ]
    if not report["dry_run"]:
        lines += [
            "## Evaluation Metrics (Stage 10 BM25 Retrieval)", "",
            "| doc_id | pairs | fact_recall@5 | hit@5 | mrr | coverage |",
            "|--------|-------|--------------|-------|-----|---------|",
        ]
        for r in report["results"]:
            ev = r.get("evaluation") or {}
            n = ev.get("n_pairs", 0)
            fr5 = ev.get("fact_recall@5", "-")
            h5 = ev.get("hit@5", "-")
            mrr = ev.get("mrr", "-")
            cov = ev.get("coverage", "-")
            lines.append(f"| {r['doc_id']} | {n} | {fr5} | {h5} | {mrr} | {cov} |")
        lines.append("")

    lines += [
        "## Pipeline Gate Summary", "",
        "| doc_id | facts | edges | chains | q_pairs | pipeline |",
        "|--------|-------|-------|--------|---------|----------|",
    ]
    for r in report["results"]:
        if r.get("pipeline"):
            p = r["pipeline"]
            facts = p.get("n_facts_after_filter", "?")
            edges = p.get("n_edges", "-")
            chains = p.get("n_chains", "-")
            pairs = p.get("n_pairs", "-")
            status = "PASS" if p.get("pipeline_passed") else "FAIL"
            lines.append(f"| {r['doc_id']} | {facts} | {edges} | {chains} | {pairs} | {status} |")
        else:
            lines.append(f"| {r['doc_id']} | ERROR | | | | FAIL |")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="GRAFT all-PDF corpus evaluation")
    ap.add_argument("--manifest", required=True, help="corpus_manifest.json")
    ap.add_argument("--out-dir", default="data/test_corpus_allpdf/eval_run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=150)
    ap.add_argument("--target-chains", type=int, default=8)
    ap.add_argument("--chain-depth", type=int, default=2)
    ap.add_argument("--llm-chunk-cap", type=int, default=30)
    ap.add_argument("--docling-page-cap", type=int, default=8)
    ap.add_argument("--doc-ids", nargs="+", help="Subset of docs to run (default: all)")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    docs = [d for d in manifest if d.get("status") == "ok"]
    if args.doc_ids:
        docs = [d for d in docs if d["doc_id"] in args.doc_ids]
    if not docs:
        raise SystemExit("No valid docs found in manifest")

    report = run_corpus(
        docs, args.out_dir,
        dry_run=args.dry_run,
        max_pairs=args.max_pairs,
        target_chains=args.target_chains,
        chain_depth=args.chain_depth,
        llm_chunk_cap=args.llm_chunk_cap,
        docling_page_cap=args.docling_page_cap,
    )
    passed = sum(1 for r in report["results"]
                 if r.get("pipeline", {}).get("pipeline_passed", False))
    print(f"\n{'='*60}")
    print(f"Cross-doc eval complete: {passed}/{len(docs)} pipeline PASS")
    print(f"Report: {args.out_dir}/cross_doc_eval_report.md")
    return 0 if passed == len(docs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
