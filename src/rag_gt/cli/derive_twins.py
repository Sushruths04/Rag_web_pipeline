"""`python -m rag_gt.cli.derive_twins` — offline AT/CT twin derivation for v16 GT files.

Reads a strict GT JSONL, derives AT + CT twins for every accepted row, and
writes a new JSONL that contains the originals interleaved with their twins.

Twin derivation needs only the LLM and the NLI model — no FAISS index.

Usage:
    python -m rag_gt.cli.derive_twins \\
        --input  data/gt/v16_strict.jsonl \\
        --output data/gt/v16_with_twins.jsonl \\
        [--ct_threshold 0.40] \\
        [--ct_max_retries 3] \\
        [--llm_role answer] \\
        [--no_twins]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from rag_gt.core.llm import get_llm
from rag_gt.generation.twins import derive_twins
from rag_gt.storage.gt_io import load_gt, save_gt


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag-gt-derive-twins",
        description="Derive AT/CT adversarial twins from a strict v16 GT file.",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input JSONL path (strict GT rows produced by generate.py --v16)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSONL path (originals + twins)",
    )
    p.add_argument(
        "--ct_threshold",
        type=float,
        default=0.40,
        help="NLI contradiction threshold for CT twin acceptance (default: 0.40)",
    )
    p.add_argument(
        "--ct_max_retries",
        type=int,
        default=3,
        help="Max LLM retries per span for CT twin generation (default: 3)",
    )
    p.add_argument(
        "--llm_role",
        default="answer",
        help="LLM role for CT answer regeneration (default: answer)",
    )
    p.add_argument(
        "--no_twins",
        action="store_true",
        help="Dry-run: load and re-save the GT file without deriving any twins",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"[derive_twins] Loading GT rows from {input_path}")
    rows = load_gt(input_path.stem, in_dir=input_path.parent)
    logger.info(f"[derive_twins] Loaded {len(rows)} rows")

    if args.no_twins:
        logger.info("[derive_twins] --no_twins: skipping twin derivation")
        all_rows = rows
    else:
        llm = get_llm(args.llm_role)
        logger.info(
            f"[derive_twins] Deriving twins "
            f"(ct_threshold={args.ct_threshold}, ct_max_retries={args.ct_max_retries})"
        )
        twins = derive_twins(
            rows,
            llm,
            twins_enabled=True,
            ct_threshold=args.ct_threshold,
            ct_max_retries=args.ct_max_retries,
        )
        logger.info(
            f"[derive_twins] {len(twins)} twins derived "
            f"({sum(1 for t in twins if t.twin_type == 'abstention')} AT, "
            f"{sum(1 for t in twins if t.twin_type == 'counterfactual')} CT)"
        )
        all_rows = rows + twins

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_gt(all_rows, output_path.stem, out_dir=output_path.parent)
    logger.info(
        f"[derive_twins] Done — {len(all_rows)} rows saved to {output_path} "
        f"({len(rows)} strict + {len(all_rows) - len(rows)} twins)"
    )


if __name__ == "__main__":
    main()
