"""Autonomous all-PDF pipeline orchestrator.

Runs the document-adaptive front-end end-to-end on any input with no manual
steps: Stage 0 preflight -> Stage 1 adaptive ingestion -> Stage 2 agentic
chunking, each followed by its verification gate. Writes per-stage checkpoints
and a consolidated report.

Usage:
    python -m rag_gt.allpdf.run --input data/test_corpus_allpdf --out-dir data/test_corpus_allpdf/run
    python -m rag_gt.allpdf.run --input some.pdf
    python -m rag_gt.allpdf.run --manifest data/test_corpus_allpdf/corpus_manifest.json

Exit code is 0 only when every stage gate passes for every document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import List, Tuple

from loguru import logger

from rag_gt.allpdf.chunk import agentic_chunk
from rag_gt.allpdf.ingest import ingest_document
from rag_gt.allpdf.preflight import profile_pdf
from rag_gt.allpdf.verify import VerificationGate, save_checkpoint


def _discover(args) -> List[Tuple[str, str]]:
    """Return [(doc_id, path)] from --manifest, --input dir, or --input file."""
    if args.manifest:
        m = json.loads(Path(args.manifest).read_text())
        return [(d["doc_id"], d["path"]) for d in m if d.get("status") == "ok"]
    p = Path(args.input)
    if p.is_dir():
        return [(f.stem, str(f)) for f in sorted(p.glob("*.pdf"))]
    if p.is_file():
        return [(p.stem, str(p))]
    raise SystemExit(f"--input not found: {args.input}")


def run_pipeline(
    docs: List[Tuple[str, str]], out_dir: str, docling_page_cap: int = 60
) -> dict:
    out = Path(out_dir)
    ckpt = out / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    g0 = VerificationGate("stage0_preflight")
    g1 = VerificationGate("stage1_ingest")
    g2 = VerificationGate("stage2_chunk")
    profiles, ingests, chunks = [], [], []
    per_doc = []
    t0 = time.time()

    for doc_id, path in docs:
        rec = {"doc_id": doc_id, "path": path}
        try:
            prof = profile_pdf(path, doc_id)
            profiles.append(prof.to_dict())
            ing = ingest_document(prof, docling_page_cap=docling_page_cap)
            ingests.append(ing.summary())
            cr = agentic_chunk(prof, ing)
            chunks.append(cr.summary())
            # Persist chunks per doc (downstream stages consume these).
            save_checkpoint(
                cr.chunks, str(ckpt / f"chunks_{doc_id}.json")
            )
            rec.update({
                "status": "ok",
                "doc_type": prof.doc_type_guess,
                "backend": ing.backend_used,
                "strategy": cr.strategy,
                "pages": prof.page_count,
                "units": ing.n_units,
                "chunks": len(cr.chunks),
            })
            logger.info(
                f"[run] {doc_id}: {prof.doc_type_guess}/{ing.backend_used}/{cr.strategy} "
                f"pages={prof.page_count} units={ing.n_units} chunks={len(cr.chunks)}"
            )
        except Exception as e:  # noqa: BLE001
            rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            logger.error(f"[run] {doc_id} FAILED: {type(e).__name__}: {e}")
        per_doc.append(rec)

    ok_docs = [r for r in per_doc if r["status"] == "ok"]
    # Aggregate gates.
    g0.expect("every doc profiled", lambda: len(profiles) == len(docs))
    g0.expect("all page_count>0", lambda: all(p["page_count"] > 0 for p in profiles))
    g1.expect("all ok docs: units>0 (non-scanned) ",
              lambda: all(s["n_units"] > 0 for s, p in zip(ingests, profiles) if not p["scanned"]))
    g1.expect("all ok docs: chars>0 (non-scanned)",
              lambda: all(s["char_count"] > 0 for s, p in zip(ingests, profiles) if not p["scanned"]))
    g1.expect(
        "all ok docs: pages_covered>=90% of page_count (non-scanned)",
        lambda: all(
            s["pages_covered"] >= 0.9 * p["page_count"]
            for s, p in zip(ingests, profiles) if not p["scanned"]
        ),
        "catches silent truncation (e.g. docling_page_cap slicing) before it ships",
    )
    g2.expect("every ok doc has chunks>0", lambda: all(s["n_chunks"] > 0 for s in chunks))
    g2.expect(">=3 chunking strategies used", lambda: len({s["strategy"] for s in chunks}) >= 3)

    g0.metrics = {"docs": len(profiles)}
    g1.metrics = {"backends": sorted({s["backend_used"] for s in ingests})}
    g2.metrics = {"strategies": sorted({s["strategy"] for s in chunks})}

    save_checkpoint(profiles, str(ckpt / "stage0_profiles.json"))
    save_checkpoint(ingests, str(ckpt / "stage1_ingest.json"))
    save_checkpoint(chunks, str(ckpt / "stage2_chunk.json"))

    gates = [g0, g1, g2]
    report = {
        "n_docs": len(docs),
        "n_ok": len(ok_docs),
        "n_error": len(docs) - len(ok_docs),
        "wall_sec": round(time.time() - t0, 1),
        "pipeline_passed": all(g.passed for g in gates) and len(ok_docs) == len(docs),
        "gates": [g.to_dict() for g in gates],
        "per_doc": per_doc,
    }
    save_checkpoint(report, str(out / "pipeline_report.json"))
    _write_markdown(report, str(out / "pipeline_report.md"))
    return report


def _write_markdown(report: dict, path: str) -> None:
    lines = [
        "# All-PDF Pipeline Run Report", "",
        f"- Documents: {report['n_docs']} ({report['n_ok']} ok, {report['n_error']} error)",
        f"- Wall time: {report['wall_sec']}s",
        f"- **Pipeline {'PASS' if report['pipeline_passed'] else 'FAIL'}**", "",
        "## Per-document", "",
        "| doc | status | doc_type | backend | strategy | pages | units | chunks |",
        "|---|---|---|---|---|--:|--:|--:|",
    ]
    for r in report["per_doc"]:
        if r["status"] == "ok":
            lines.append(
                f"| {r['doc_id']} | ok | {r['doc_type']} | {r['backend']} | "
                f"{r['strategy']} | {r['pages']} | {r['units']} | {r['chunks']} |"
            )
        else:
            lines.append(f"| {r['doc_id']} | ERROR | {r.get('error','')} | | | | | |")
    lines += ["", "## Gate results", ""]
    for g in report["gates"]:
        lines.append(f"### {g['stage']} — {'PASS' if g['passed'] else 'FAIL'}")
        for c in g["checks"]:
            lines.append(f"- [{'x' if c['passed'] else ' '}] {c['name']}"
                         + (f" — {c['detail']}" if c['detail'] else ""))
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(prog="rag-gt-allpdf", description="Autonomous all-PDF pipeline")
    ap.add_argument("--input", help="PDF file or directory of PDFs")
    ap.add_argument("--manifest", help="corpus_manifest.json")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--docling-page-cap", type=int, default=60)
    args = ap.parse_args()
    if not args.input and not args.manifest:
        raise SystemExit("provide --input or --manifest")
    docs = _discover(args)
    out_dir = args.out_dir or (
        str(Path(args.manifest).parent / "run") if args.manifest
        else str(Path(args.input).parent / "allpdf_run")
    )
    report = run_pipeline(docs, out_dir, docling_page_cap=args.docling_page_cap)
    print(f"\nReport: {out_dir}/pipeline_report.md")
    print(f"Pipeline {'PASS' if report['pipeline_passed'] else 'FAIL'} "
          f"({report['n_ok']}/{report['n_docs']} docs ok, {report['wall_sec']}s)")
    return 0 if report["pipeline_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
