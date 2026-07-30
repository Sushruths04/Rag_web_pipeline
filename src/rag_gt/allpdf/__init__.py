"""GRAFT all-PDF pipeline: document-adaptive ingestion, agentic chunking, and
per-stage verification. New front-end that makes the GT/eval pipeline work on
arbitrary heterogeneous PDFs.

Stages:
  0  preflight.py       — DocProfile: doc_type, scanned, table_density, etc.
  1  ingest.py          — IngestResult: SourceUnits with page/bbox provenance
  2  chunk.py           — ChunkResult: agentic strategy selection + packing
  3  extract.py         — ExtractResult: SFU facts with canonical_form
  4  filter_adaptive.py — doc-type-aware domain filter
  5–7 pipeline.py       — TypedSFG (hybrid graph) → chain sampling → QGen
  10  evaluate.py       — BM25 retrieval + fact_recall metrics ($0 / no-API)

Orchestrators:
  run.py               — Stages 0-2 batch run (corpus manifest → checkpoints)
  pipeline.py          — Stages 0-7 per-doc (full or dry-run)
  eval_orchestrator.py — Stages 0-10 per-doc + cross-doc eval report

See docs/superpowers/specs/2026-06-22-graft-allpdf-pipeline-design.md
"""

from rag_gt.allpdf.preflight import DocProfile, profile_pdf
from rag_gt.allpdf.ingest import IngestResult, ingest_document
from rag_gt.allpdf.chunk import ChunkResult, agentic_chunk, select_strategy
from rag_gt.allpdf.extract import ExtractResult, extract_sfu_facts
from rag_gt.allpdf.filter_adaptive import filter_facts_adaptive
from rag_gt.allpdf.verify import VerificationGate, save_checkpoint
from rag_gt.allpdf.evaluate import evaluate_pairs, EvalResult

__all__ = [
    "DocProfile", "profile_pdf",
    "IngestResult", "ingest_document",
    "ChunkResult", "agentic_chunk", "select_strategy",
    "ExtractResult", "extract_sfu_facts",
    "filter_facts_adaptive",
    "VerificationGate", "save_checkpoint",
    "evaluate_pairs", "EvalResult",
]
