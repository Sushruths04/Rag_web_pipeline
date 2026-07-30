"""`python -m rag_gt.cli.dry_run_budget` — V16.2 calibration gate.

Computes per-document `DocBudget` for a set of real PDFs without running any
LLM stage, and prints a Markdown table for the user to inspect before any live
TF-SFG wiring is enabled.

Acceptance rule (the user enforces by reading the table):
  - the docs must produce meaningfully different `tf_sfg_pairs`;
  - the smallest doc must have `strict_target >= 4`;
  - no global saturation/collapse at the clamps.

If saturation/collapse is observed, recalibrate constants in
`configs/v16.yaml::v16_2.adaptive_budget` before proceeding to task #11.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import yaml

from rag_gt.budget import AdaptiveBudgetConfig, compute_doc_budget
from rag_gt.chunking.strategies import chunk_document
from rag_gt.core.config import load_config
from rag_gt.facts.extraction import extract_candidate_facts
from rag_gt.facts.domain_filter import filter_fact_domain
from rag_gt.ingestion import ingest_document
from rag_gt.profiling.profiler import profile_document
from rag_gt.spans.normalization import find_fact_spans, tokenize_document

logger = logging.getLogger("rag_gt.cli.dry_run_budget")


def _expand_doc_args(values: Iterable[str]) -> list[str]:
    """Accept either PDF paths or glob patterns; return a sorted unique list."""
    paths: list[str] = []
    for v in values:
        # Allow comma-separated lists too: "--docs a.pdf,b.pdf".
        for part in v.split(","):
            part = part.strip()
            if not part:
                continue
            if any(c in part for c in "*?["):
                matched = sorted(glob.glob(part))
                paths.extend(matched)
            else:
                paths.append(part)
    # de-dup while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _load_v16_2_adaptive_cfg(v16_yaml_path: str) -> AdaptiveBudgetConfig:
    with open(v16_yaml_path, "r", encoding="utf-8") as f:
        v16_cfg = yaml.safe_load(f) or {}
    v16_2 = v16_cfg.get("v16_2", {}) or {}
    return AdaptiveBudgetConfig.from_dict(v16_2.get("adaptive_budget"))


def _yield_fractions(v16_yaml_path: str) -> tuple[float, float, float]:
    with open(v16_yaml_path, "r", encoding="utf-8") as f:
        v16_cfg = yaml.safe_load(f) or {}
    yc = ((v16_cfg.get("v16_2") or {}).get("yield_controller") or {})
    return (
        float(yc.get("mh_floor_fraction", 0.60)),
        float(yc.get("untyped_mh_cap_fraction", 0.10)),
        float(yc.get("fb_cap_fraction", 0.40)),
    )


def _count_pages(facts: list, fallback_doc_units: int = 0) -> int:
    """Effective page count = distinct page numbers seen in fact spans.

    Falls back to `fallback_doc_units` if no spans carry page metadata.
    """
    pages: set[int] = set()
    for f in facts:
        for span in getattr(f, "supporting_spans", []) or []:
            for bbox in getattr(span, "bboxes", []) or []:
                page_no = getattr(bbox, "page_no", None)
                if page_no is not None:
                    pages.add(int(page_no))
            ps = getattr(span, "page_start", None)
            pe = getattr(span, "page_end", None)
            if ps is not None:
                pages.add(int(ps))
            if pe is not None:
                pages.add(int(pe))
    if pages:
        return len(pages)
    return max(1, fallback_doc_units)


def _ingest_and_count(
    path: str, default_doc_type: str, chunk_size: int, chunk_overlap: int
) -> tuple[str, int, int]:
    """Run the cheap, no-LLM stages and return (doc_type, fact_count, page_count).

    fact_count reflects the post-domain-filter count, matching what TF-SFG will
    actually see in the live pipeline.
    """
    doc = ingest_document(path, doc_type=default_doc_type)
    profile = profile_document(doc, path=path, enable_fast_path=True)
    doc_type = profile["doc_type"]
    chunks = chunk_document(doc, profile, chunk_size, chunk_overlap)
    facts = extract_candidate_facts(doc, chunks, llm=None)
    doc_toks = tokenize_document(doc)
    facts = find_fact_spans(doc, doc_toks, facts, chunks)
    facts = [f for f in facts if f.supporting_spans]
    # Apply the domain filter so fact_count matches the TF-SFG input.
    facts, _ = filter_fact_domain(facts)
    page_count = _count_pages(facts, fallback_doc_units=len(chunks))
    return doc_type, len(facts), page_count


def _build_table(rows: list[dict]) -> str:
    """Markdown table of dry-run budgets, with a saturation hint per row."""
    header = (
        "| doc | doc_type | facts | pages | size_signal | tf_sfg_pairs | "
        "fallback_singlehop | strict_target | attempt_cap | note |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    body_lines: list[str] = []
    for r in rows:
        body_lines.append(
            f"| `{r['doc']}` | {r['doc_type']} | {r['facts']} | {r['pages']} | "
            f"{r['size_signal']:.2f} | {r['tf_sfg_pairs']} | {r['fallback_singlehop']} | "
            f"{r['strict_target']} | {r['attempt_cap']} | {r['note']} |"
        )
    return header + "\n".join(body_lines) + "\n"


def _saturation_note(
    pairs: int, target: int, lo_pairs: int, hi_pairs: int, lo_target: int, hi_target: int
) -> str:
    notes: list[str] = []
    if pairs <= lo_pairs:
        notes.append("tf_sfg@floor")
    if pairs >= hi_pairs:
        notes.append("tf_sfg@ceiling")
    if target <= lo_target:
        notes.append("target@floor")
    if target >= hi_target:
        notes.append("target@ceiling")
    return ",".join(notes) if notes else "ok"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag-gt-dry-run-budget",
        description="V16.2 calibration gate: compute per-doc DocBudget and print a Markdown table.",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to v16.yaml. Defaults to configs/v16.yaml relative to repo root.",
    )
    p.add_argument(
        "--docs",
        nargs="+",
        required=True,
        help="PDF paths or globs (or comma-separated list). E.g., 'data/docs_rl/*.pdf'.",
    )
    p.add_argument("--doc_type", default="UNKNOWN", help="Default doc_type hint before profiling.")
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--chunk_overlap", type=int, default=64)
    p.add_argument(
        "--out",
        default=None,
        help="Write Markdown table to this path in addition to stdout.",
    )
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()

    v16_yaml = args.config
    if v16_yaml is None:
        # Default: configs/v16.yaml via load_config's resolution.
        cfg = load_config()
        v16_top = cfg.get("v16") or {}
        v16_yaml = v16_top.get("config_path", "configs/v16.yaml")
    if not Path(v16_yaml).is_absolute():
        v16_yaml = str(Path.cwd() / v16_yaml)

    if not Path(v16_yaml).exists():
        print(f"ERROR: v16 config not found at {v16_yaml}", file=sys.stderr)
        return 2

    adaptive_cfg = _load_v16_2_adaptive_cfg(v16_yaml)
    mh_floor, untyped_cap, fb_cap = _yield_fractions(v16_yaml)

    paths = _expand_doc_args(args.docs)
    if not paths:
        print("ERROR: no PDF paths matched.", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for path in paths:
        if not os.path.exists(path):
            logger.warning(f"skipping missing file: {path}")
            continue
        logger.info(f"ingesting + counting: {path}")
        try:
            doc_type, fact_count, page_count = _ingest_and_count(
                path,
                default_doc_type=args.doc_type,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"failed to ingest {path}: {exc}")
            continue

        budget = compute_doc_budget(
            fact_count=fact_count,
            page_count=page_count,
            doc_type=doc_type,
            cfg=adaptive_cfg,
            mh_floor_fraction=mh_floor,
            untyped_mh_cap_fraction=untyped_cap,
            fb_cap_fraction=fb_cap,
        )
        base = adaptive_cfg.base_budget.get(doc_type) or adaptive_cfg.base_budget.get("REPORT")
        hi_pairs = base.tf_sfg_pairs_hi if base else 400
        rows.append(
            {
                "doc": os.path.basename(path),
                "doc_type": budget.doc_type,
                "facts": budget.fact_count,
                "pages": budget.page_count,
                "size_signal": budget.size_signal,
                "tf_sfg_pairs": budget.tf_sfg_pairs,
                "fallback_singlehop": budget.fallback_singlehop,
                "strict_target": budget.strict_target,
                "attempt_cap": budget.attempt_cap,
                "note": _saturation_note(
                    pairs=budget.tf_sfg_pairs,
                    target=budget.strict_target,
                    lo_pairs=24,
                    hi_pairs=hi_pairs,
                    lo_target=4,
                    hi_target=adaptive_cfg.global_strict_per_doc_cap,
                ),
            }
        )

    if not rows:
        print("ERROR: no docs ingested successfully.", file=sys.stderr)
        return 1

    table = _build_table(rows)
    print()
    print("## V16.2 dry-run budget")
    print()
    print(table)

    saturated = [r for r in rows if "@ceiling" in r["note"] or "@floor" in r["note"]]
    if saturated:
        print("WARNING: one or more docs saturated a clamp; review calibration before wiring.")
        for r in saturated:
            print(f"  - {r['doc']}: {r['note']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# V16.2 dry-run budget\n\n")
            f.write(table)
            if saturated:
                f.write("\n## Saturation warnings\n\n")
                for r in saturated:
                    f.write(f"- `{r['doc']}`: {r['note']}\n")
        logger.info(f"wrote table to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
