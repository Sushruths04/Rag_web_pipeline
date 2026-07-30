"""Evaluate retrieval against the v2 evidence-dense GT (no API cost).

Adapts v2 QA records (answer_clauses / gold_fact_ids) to the CP3 matcher
schema. The matcher scores token overlap between FACT text and chunk text, so
facts are joined by EXACT gold_fact_ids from the grounded fact files — never
from the answer clauses (paraphrases would understate recall) and never by
fuzzy matching (NEXT_AGENT_HANDOFF.md Track A).

Usage:
    python -m rag_gt.rag.eval_v2 \
        --v2-dir data/eval_results/allpdf_v2_gt \
        --facts-dir data/eval_results/facts_v1_grounded \
        --out data/eval_results/V2_FULLCORPUS_EVAL
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

from rag_gt.rag.loader import load_chunks, load_gt_pairs
from rag_gt.rag.matcher import match_pair
from rag_gt.rag.metrics import aggregate, fill_metrics
from rag_gt.rag.retriever import build_retriever

STRATEGIES = ("bm25", "hybrid")


def doc_to_pipeline_id(doc: str) -> str:
    """v2 fact/GT files use '<pipeline_doc_id>_full'."""
    return doc[:-5] if doc.endswith("_full") else doc


def v2_pair_to_eval(pair: dict, fact_text_by_id: Dict[str, str]) -> dict:
    """Map a v2 QA record onto the matcher's expected pair schema.

    Raises KeyError if a gold fact id is missing from the lookup — a missing
    id means the GT and fact set are out of sync, which must fail loudly.
    """
    fact_ids = [str(f) for f in pair.get("gold_fact_ids", [])]
    facts = [{"fact_id": fid, "text": fact_text_by_id[fid]} for fid in fact_ids]
    pages = [p for p in pair.get("gold_pages", []) if isinstance(p, (int, float))]
    return {
        "question": str(pair.get("question", "")),
        "pair_type": str(pair.get("evidence_strategy") or pair.get("hop_type") or "unknown"),
        "depth": len(fact_ids),
        "page_spread": int(max(pages) - min(pages)) if pages else 0,
        "necessity_score": float((pair.get("necessity") or {}).get("necessity_score", 0.0)),
        "facts": facts,
    }


def build_fact_text_lookup(facts_dir: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for path in sorted(facts_dir.glob("facts_*.json")):
        for rec in json.loads(path.read_text(encoding="utf-8")):
            lookup[str(rec["fact_id"])] = str(
                rec.get("canonical_form") or rec.get("text") or ""
            )
    return lookup


def evaluate_v2(v2_dir: Path, facts_dir: Path, top_k: int = 10,
                strategies=STRATEGIES, embed_source: str = "local") -> dict:
    lookup = build_fact_text_lookup(facts_dir)
    out: dict = {"top_k": top_k, "strategies": {}}
    v2_files = sorted(p for p in v2_dir.glob("*.json"))
    for strategy in strategies:
        t0 = time.time()
        v2_results: List = []
        v1_results: List = []
        per_doc: Dict[str, dict] = {}
        for path in v2_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pairs = payload.get("pairs", [])
            if not pairs:
                continue
            doc_full = str(payload.get("stats", {}).get("doc") or pairs[0].get("doc", ""))
            doc_id = doc_to_pipeline_id(doc_full)
            chunks = load_chunks(doc_id)
            retriever = build_retriever(chunks, strategy=strategy, embed_source=embed_source)
            id_to_text = {c.get("chunk_id", ""): c.get("text", "") for c in chunks}

            doc_mrs = []
            for pair in pairs:
                eval_pair = v2_pair_to_eval(pair, lookup)
                ranked = retriever.retrieve(eval_pair["question"], top_k=top_k)
                doc_mrs.append(fill_metrics(match_pair(eval_pair, ranked, id_to_text)))
            v2_results.extend(doc_mrs)
            per_doc[doc_id] = aggregate(doc_mrs).summary()

            for pair in load_gt_pairs(doc_id):
                ranked = retriever.retrieve(pair["question"], top_k=top_k)
                v1_results.append(fill_metrics(match_pair(pair, ranked, id_to_text)))

        agg = aggregate(v2_results)
        out["strategies"][strategy] = {
            "wall_sec": round(time.time() - t0, 1),
            "v2": agg.summary(),
            "v2_by_strategy": {pt: a.summary() for pt, a in sorted(agg.by_pair_type.items())},
            "v2_per_doc": per_doc,
            "v1_baseline": aggregate(v1_results).summary(),
        }
    return out


_HEAD_KEYS = ("n_pairs", "fact_recall@5", "precision@5", "precision_rw@5",
              "hit@5", "mrr", "coverage")


def _row(label: str, summary: dict) -> str:
    cells = [str(summary.get("n_pairs", 0))]
    for key in _HEAD_KEYS[1:]:
        cells.append(f"{summary.get(key, 0.0):.3f}")
    return "| " + label + " | " + " | ".join(cells) + " |"


def _fmt_md(result: dict) -> str:
    lines = [
        "# v2 GT full-corpus retrieval evaluation",
        "",
        f"top_k={result['top_k']}, chunking=original, matcher=token-overlap>=60%",
        "precision_rw@5 = rank-weighted precision (mean Precision@r over relevant",
        "ranks r<=5) — does not punish questions with fewer gold units than k.",
        "",
    ]
    header = ("| set | n | recall@5 | precision@5 | precision_rw@5 | hit@5 | mrr | coverage |\n"
              "|---|---:|---:|---:|---:|---:|---:|---:|")
    for strategy, block in result["strategies"].items():
        lines += [f"## Retriever: {strategy}", "", header,
                  _row("**v2 GT (evidence-dense)**", block["v2"]),
                  _row("v1 GT baseline (same docs)", block["v1_baseline"]), ""]
        lines += ["### v2 by evidence strategy", "", header]
        for pt, summary in block["v2_by_strategy"].items():
            lines.append(_row(pt, summary))
        lines += ["", "### v2 per document", "", header]
        for doc_id, summary in sorted(block["v2_per_doc"].items()):
            lines.append(_row(doc_id, summary))
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2-dir", type=Path, required=True)
    ap.add_argument("--facts-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="output path prefix (writes .json and .md)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--embed-source", choices=("local", "api"), default="local",
                    help="dense/hybrid embedding source: local SentenceTransformer "
                         "(fully offline, default) or api (MM.get_embedding(), may "
                         "call a paid endpoint if ACTIVE_EMBEDDING is set)")
    args = ap.parse_args(argv)

    result = evaluate_v2(args.v2_dir, args.facts_dir, top_k=args.top_k,
                         embed_source=args.embed_source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    args.out.with_suffix(".md").write_text(_fmt_md(result), encoding="utf-8")
    for strategy, block in result["strategies"].items():
        s = block["v2"]
        print(f"[{strategy}] v2 n={s['n_pairs']} recall@5={s.get('fact_recall@5', 0):.3f} "
              f"prec_rw@5={s.get('precision_rw@5', 0):.3f} mrr={s.get('mrr', 0):.3f}",
              file=sys.stderr)
    print(f"written: {args.out.with_suffix('.json')} + .md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
