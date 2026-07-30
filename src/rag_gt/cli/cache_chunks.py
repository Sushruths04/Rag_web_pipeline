"""`python -m rag_gt.cli.cache_chunks` -- one-shot rebuild of the chunks cache.

Reads PDFs/DOCX from `--input_dir`, runs the canonical pipeline chunker, and
writes one JSONL row per chunk to `--output`. Existing GT JSONLs are not
touched. Chunk IDs match `cli/retrieve_test.py` exactly because both go
through `chunking.strategies.chunk_document`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_gt.comparison.chunk_resolver import build_chunks_cache


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag-gt-cache-chunks",
        description=(
            "Rebuild the chunk_id -> text JSONL cache by re-running the "
            "RAG_GT pipeline chunker over a document directory."
        ),
    )
    p.add_argument("--input_dir", required=True, help="Directory with PDF/DOCX files.")
    p.add_argument(
        "--output",
        default="data/cache/chunks.jsonl",
        help="Output JSONL path (default: data/cache/chunks.jsonl).",
    )
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--chunk_overlap", type=int, default=64)
    p.add_argument(
        "--doc_type",
        default=None,
        help="Optional doc_type hint (ISO_STANDARD, KT_DOC, REPORT, ...).",
    )
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        sys.stderr.write(f"input_dir not found or not a directory: {in_dir}\n")
        sys.exit(2)

    n = build_chunks_cache(
        in_dir,
        Path(args.output),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        doc_type_hint=args.doc_type,
    )
    print(f"[cache_chunks] wrote {n} chunks to {args.output}")


if __name__ == "__main__":
    main()
