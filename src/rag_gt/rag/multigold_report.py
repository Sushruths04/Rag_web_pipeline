"""Stage E multi-gold retrieval evaluation and standalone HTML report."""

from __future__ import annotations

import argparse
import html
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from loguru import logger

from rag_gt.core.config import load_config
from rag_gt.core.llm import get_llm
from rag_gt.rag.decomposition import DecompositionRetriever
from rag_gt.rag.matcher import MatchResult, fact_overlap, match_pair
from rag_gt.rag.metrics import aggregate, fill_metrics
from rag_gt.rag.reranker import CrossEncoderReranker, CrossEncoderScorer
from rag_gt.rag.retriever import DenseRetriever, build_retriever


def _fact_id(record: Mapping[str, object]) -> str:
    return str(record.get("fact_id") or record.get("id") or "").strip()


def _fact_text(record: Mapping[str, object]) -> str:
    return str(record.get("canonical_form") or record.get("text") or "").strip()


def _load_chunks(chunk_root: Path, doc: str) -> list[dict]:
    path = chunk_root / doc / "checkpoints" / "s2_chunks_full.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing chunk checkpoint for {doc}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Chunk checkpoint must contain a JSON list: {path}")
    return payload


def _chunk_has_page(chunk: Mapping[str, object], page: int) -> bool:
    start = int(chunk.get("page_start", chunk.get("page", 0)) or 0)
    end = int(chunk.get("page_end", start) or start)
    return start <= page <= end


def _resolve_source_path(value: object, source_root: Path) -> str:
    if not value:
        return ""
    path = Path(str(value))
    resolved = path if path.is_absolute() else source_root / path
    return str(resolved.resolve()) if resolved.exists() else str(resolved)


def _fact_bboxes(
    fact: Mapping[str, object],
    chunk: Mapping[str, object],
    page: int,
    *,
    fuzzy_min: float,
) -> tuple[list[dict], str, float]:
    units = [
        unit
        for unit in chunk.get("source_units") or []
        if int(unit.get("page_no", 0) or 0) == page
    ]
    if not units:
        return [], "unavailable", 0.0
    fact_text = _fact_text(fact)
    chunk_text = str(chunk.get("text") or "")
    position = chunk_text.casefold().find(fact_text.casefold())
    alignment_score = 1.0 if position >= 0 else 0.0
    fuzzy_aligned = False
    if position < 0:
        from rapidfuzz.fuzz import partial_ratio_alignment

        alignment = partial_ratio_alignment(fact_text, chunk_text)
        if alignment and float(alignment.score) / 100.0 >= fuzzy_min:
            position = int(alignment.dest_start)
            alignment_score = float(alignment.score) / 100.0
            fuzzy_aligned = True
    selected_units: list[Mapping[str, object]] = []
    method = "chunk_page_fallback"
    if position >= 0:
        source_start = int(
            chunk.get("source_char_start", chunk.get("char_start", 0)) or 0
        )
        fact_start = source_start + position
        fact_end = fact_start + len(fact_text)
        selected_units = [
            unit
            for unit in units
            if int(unit.get("char_start", 0) or 0) < fact_end
            and int(unit.get("char_end", 0) or 0) > fact_start
        ]
        if selected_units:
            method = (
                "fuzzy_char_overlap"
                if fuzzy_aligned
                else "approximate_char_overlap"
            )
    if not selected_units:
        selected_units = units
    boxes: list[dict] = []
    seen: set[str] = set()
    for unit in selected_units:
        for bbox in unit.get("bboxes") or []:
            if int(bbox.get("page_no", page) or page) != page:
                continue
            key = json.dumps(bbox, sort_keys=True)
            if key not in seen:
                seen.add(key)
                boxes.append(dict(bbox))
    return boxes, method if boxes else "unavailable", alignment_score


def _ground_gold_supports(
    qa: Mapping[str, object],
    facts_by_id: Mapping[str, Mapping[str, object]],
    chunks: Sequence[Mapping[str, object]],
    *,
    overlap_min: float,
    source_root: Path,
) -> tuple[list[dict], str | None]:
    supports: list[dict] = []
    pages = list(qa.get("gold_pages") or [])
    bbox_map = qa.get("gold_bboxes") or {}
    fuzzy_min = float(
        load_config()["multigold_evaluation"]["bbox_fuzzy_alignment_min"]
    )
    for index, fact_id in enumerate(qa.get("gold_fact_ids") or []):
        fact_id = str(fact_id)
        fact = facts_by_id.get(fact_id)
        if fact is None:
            return [], "missing_fact"
        page = int(pages[index] if index < len(pages) else fact.get("page", 0) or 0)
        page_candidates = [chunk for chunk in chunks if _chunk_has_page(chunk, page)]
        candidates = page_candidates or list(chunks)
        if not candidates:
            return [], "no_chunks"
        scored = [
            (fact_overlap(dict(fact), str(chunk.get("text") or "")), chunk)
            for chunk in candidates
        ]
        score, best = max(scored, key=lambda item: item[0])
        if score < overlap_min:
            return [], "fact_not_grounded"
        derived_bboxes, bbox_method, bbox_alignment = _fact_bboxes(
            fact, best, page, fuzzy_min=fuzzy_min
        )
        existing_bboxes = list(bbox_map.get(fact_id) or [])
        bboxes = existing_bboxes or derived_bboxes
        if existing_bboxes:
            bbox_method = "fact_checkpoint"
            bbox_alignment = 1.0
        supports.append(
            {
                "fact_id": fact_id,
                "text": _fact_text(fact),
                "page": page,
                "bboxes": bboxes,
                "bbox_method": bbox_method,
                "bbox_alignment_score": round(bbox_alignment, 3),
                "chunk_id": str(best.get("chunk_id") or ""),
                "chunk_page_start": best.get("page_start"),
                "chunk_page_end": best.get("page_end"),
                "overlap": round(float(score), 3),
                "source_path": _resolve_source_path(
                    best.get("source_path"), source_root
                ),
            }
        )
    if qa.get("hop_type") == "bridge" and len({s["chunk_id"] for s in supports}) < 2:
        return [], "bridge_not_multichunk"
    return supports, None


def _matcher_pair(qa: Mapping[str, object], supports: Sequence[Mapping[str, object]]) -> dict:
    pages = [int(s["page"]) for s in supports]
    return {
        "question": str(qa.get("question") or ""),
        "pair_type": str(qa.get("hop_type") or "unknown"),
        "depth": len(supports),
        "page_spread": max(pages) - min(pages) if pages else 0,
        "necessity_score": float((qa.get("necessity") or {}).get("necessity_score", 0.0) or 0.0),
        "facts": [
            {"fact_id": support["fact_id"], "text": support["text"]}
            for support in supports
        ],
    }


def _summary(results: Sequence[MatchResult], top_k_values: list[int]) -> dict:
    summary = aggregate(list(results), k_values=top_k_values).summary()
    for k in top_k_values:
        summary[f"hop_completeness@{k}"] = summary[f"hit@{k}"]
    return summary


def _per_retriever_result(
    mr: MatchResult,
    ranked: Sequence[tuple[str, float]],
    chunks_by_id: Mapping[str, Mapping[str, object]],
    *,
    top_k_values: list[int],
    snippet_chars: int,
) -> dict:
    fact_hits = [
        {
            "fact_id": match.fact_id,
            "hit": match.hit,
            "first_hit_rank": match.first_hit_rank,
            "best_overlap": round(match.best_overlap, 3),
        }
        for match in mr.fact_matches
    ]
    retrieved = []
    for rank, (chunk_id, score) in enumerate(ranked, start=1):
        chunk = chunks_by_id.get(chunk_id, {})
        retrieved.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "score": round(float(score), 6),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "text": " ".join(str(chunk.get("text") or "").split())[
                    :snippet_chars
                ],
                "covers_fact_ids": [
                    match.fact_id
                    for match in mr.fact_matches
                    if match.first_hit_rank == rank
                ],
            }
        )
    return {
        "metrics": {
            **{f"recall@{k}": round(mr.fact_recall_at_k[k], 4) for k in top_k_values},
            **{f"precision@{k}": round(mr.precision_at_k[k], 4) for k in top_k_values},
            **{f"hop_completeness@{k}": mr.hit_at_k[k] for k in top_k_values},
            **{f"ndcg@{k}": round(mr.ndcg_at_k[k], 4) for k in top_k_values},
            "mrr": round(mr.mrr, 4),
        },
        "fact_hits": fact_hits,
        "retrieved": retrieved,
    }


def evaluate_multigold(
    facts: Sequence[Mapping[str, object]],
    qa_pairs: Sequence[Mapping[str, object]],
    chunk_root: Path,
    *,
    strategies: Sequence[str] | None = None,
    retriever_factory: Callable[..., object] = build_retriever,
) -> dict:
    """Evaluate all Stage D PASS rows against per-document source chunks."""
    started = time.time()
    cfg = load_config()["multigold_evaluation"]
    selected_strategies = list(strategies or cfg["retrievers"])
    top_k_values = [int(value) for value in cfg["top_k_values"]]
    max_top_k = max(top_k_values)
    overlap_min = float(cfg["match_overlap_min"])
    snippet_chars = int(cfg["report_snippet_chars"])
    facts_by_id = {_fact_id(fact): fact for fact in facts if _fact_id(fact)}
    accepted = [qa for qa in qa_pairs if (qa.get("verify") or {}).get("verdict") == "PASS"]
    by_doc: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for qa in accepted:
        by_doc[str(qa.get("doc") or "")].append(qa)

    all_results: dict[str, list[MatchResult]] = defaultdict(list)
    type_results: dict[str, dict[str, list[MatchResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_qa: list[dict] = []
    grounding_rejections: Counter = Counter()
    n_chunks_by_doc: dict[str, int] = {}
    reranker_scorer = (
        CrossEncoderScorer()
        if any(name in {"hybrid_rerank", "bridge_rerank"} for name in selected_strategies)
        else None
    )
    decomp_llm = get_llm("gt") if "decompose" in selected_strategies else None
    source_root = chunk_root.parents[2]

    for doc, doc_qas in sorted(by_doc.items()):
        chunks = _load_chunks(chunk_root, doc)
        n_chunks_by_doc[doc] = len(chunks)
        chunks_by_id = {str(chunk.get("chunk_id") or ""): chunk for chunk in chunks}
        grounded: list[tuple[Mapping[str, object], list[dict], dict]] = []
        for qa in doc_qas:
            supports, reason = _ground_gold_supports(
                qa,
                facts_by_id,
                chunks,
                overlap_min=overlap_min,
                source_root=source_root,
            )
            if reason:
                grounding_rejections[reason] += 1
                continue
            grounded.append((qa, supports, _matcher_pair(qa, supports)))

        queries = [str(qa.get("question") or "") for qa, _, _ in grounded]
        retrievers: dict[str, object] = {}
        dense: DenseRetriever | None = None
        hybrid: object | None = None
        for strategy in selected_strategies:
            if strategy in {"hybrid_rerank", "bridge_rerank", "decompose"}:
                if hybrid is None:
                    if dense is None:
                        dense_candidate = retriever_factory(chunks, strategy="dense")
                        if not isinstance(dense_candidate, DenseRetriever):
                            raise TypeError("Dense retriever factory returned an incompatible object")
                        dense = dense_candidate
                        dense.prime_queries(queries)
                    hybrid = retriever_factory(
                        chunks, strategy="hybrid", dense_retriever=dense
                    )
                if strategy == "decompose":
                    if decomp_llm is None:
                        raise RuntimeError("Decomposition LLM was not initialized")
                    retriever = DecompositionRetriever(hybrid, decomp_llm)
                else:
                    if reranker_scorer is None:
                        raise RuntimeError("Reranker scorer was not initialized")
                    retriever = CrossEncoderReranker(
                        hybrid,
                        reranker_scorer,
                        mode=(
                            "relevance"
                            if strategy == "hybrid_rerank"
                            else "bridge_complementarity"
                        ),
                    )
            elif strategy == "hybrid" and dense is not None:
                hybrid = retriever_factory(
                    chunks, strategy="hybrid", dense_retriever=dense
                )
                retriever = hybrid
            else:
                retriever = retriever_factory(chunks, strategy=strategy)
            if strategy == "dense" and isinstance(retriever, DenseRetriever):
                dense = retriever
            if strategy == "hybrid":
                hybrid = retriever
            prime = getattr(retriever, "prime_queries", None)
            if callable(prime):
                prime(queries)
            retrievers[strategy] = retriever

        for qa, supports, matcher_input in grounded:
            qa_output = {
                "qa_id": qa.get("qa_id"),
                "doc": doc,
                "hop_type": qa.get("hop_type"),
                "bridge_entity": qa.get("bridge_entity"),
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "gold_supports": supports,
                "retrieval": {},
            }
            for strategy, retriever in retrievers.items():
                ranked = retriever.retrieve(str(qa.get("question") or ""), top_k=max_top_k)
                id_to_text = {
                    chunk_id: str(chunk.get("text") or "")
                    for chunk_id, chunk in chunks_by_id.items()
                }
                mr = fill_metrics(
                    match_pair(
                        matcher_input,
                        ranked,
                        id_to_text,
                        top_k_values=top_k_values,
                    ),
                    k_values=top_k_values,
                )
                all_results[strategy].append(mr)
                type_results[strategy][str(qa.get("hop_type"))].append(mr)
                qa_output["retrieval"][strategy] = _per_retriever_result(
                    mr,
                    ranked,
                    chunks_by_id,
                    top_k_values=top_k_values,
                    snippet_chars=snippet_chars,
                )
            per_qa.append(qa_output)

    summaries: dict[str, dict] = {}
    for strategy in selected_strategies:
        summaries[strategy] = {"all": _summary(all_results[strategy], top_k_values)}
        for hop_type in ("bridge", "single"):
            summaries[strategy][hop_type] = _summary(
                type_results[strategy][hop_type], top_k_values
            )
    bbox_supports = sum(
        1
        for qa in per_qa
        for support in qa["gold_supports"]
        if support["bboxes"]
    )
    total_supports = sum(len(qa["gold_supports"]) for qa in per_qa)
    return {
        "metadata": {
            "n_input": len(qa_pairs),
            "n_stage_d_pass": len(accepted),
            "n_evaluated": len(per_qa),
            "grounding_rejections": dict(sorted(grounding_rejections.items())),
            "n_chunks_by_doc": n_chunks_by_doc,
            "retrievers": selected_strategies,
            "top_k_values": top_k_values,
            "wall_seconds": round(time.time() - started, 2),
            "bbox_supports": bbox_supports,
            "total_supports": total_supports,
            "bbox_coverage": round(bbox_supports / total_supports, 4)
            if total_supports
            else 0.0,
        },
        "summary": summaries,
        "per_qa": per_qa,
    }


def _metric(summary: Mapping[str, object], name: str) -> str:
    return f"{float(summary.get(name, 0.0)):.3f}"


def enrich_qa_grounding(
    qa_payload: Mapping[str, object], report: Mapping[str, object]
) -> dict:
    """Write Stage E's real chunk and bbox grounding back into the QA contract."""
    supports_by_id = {
        str(item["qa_id"]): item["gold_supports"] for item in report["per_qa"]
    }
    output = []
    for qa in qa_payload["pairs"]:
        enriched = dict(qa)
        supports = supports_by_id.get(str(qa.get("qa_id")))
        if supports:
            enriched["gold_chunk_ids"] = [support["chunk_id"] for support in supports]
            enriched["gold_bboxes"] = {
                support["fact_id"]: support["bboxes"] for support in supports
            }
            enriched["grounding"] = {
                support["fact_id"]: {
                    "chunk_id": support["chunk_id"],
                    "page": support["page"],
                    "bbox_method": support["bbox_method"],
                    "bbox_alignment_score": support["bbox_alignment_score"],
                    "source_path": support["source_path"],
                    "overlap": support["overlap"],
                }
                for support in supports
            }
        output.append(enriched)
    return {
        "pairs": output,
        "stats": {
            **dict(qa_payload.get("stats") or {}),
            "bbox_supports": report["metadata"]["bbox_supports"],
            "total_supports": report["metadata"]["total_supports"],
            "bbox_coverage": report["metadata"]["bbox_coverage"],
        },
    }


def render_html(report: Mapping[str, object]) -> str:
    """Render a dependency-free interactive report suitable for local review."""
    metadata = report["metadata"]
    summary_rows = []
    for retriever, tracks in report["summary"].items():
        for track, metrics in tracks.items():
            summary_rows.append(
                "<tr>"
                f"<td>{html.escape(str(retriever))}</td><td>{html.escape(str(track))}</td>"
                f"<td>{int(metrics.get('n_pairs', 0))}</td>"
                f"<td>{_metric(metrics, 'fact_recall@5')}</td>"
                f"<td>{_metric(metrics, 'precision@5')}</td>"
                f"<td>{_metric(metrics, 'hop_completeness@5')}</td>"
                f"<td>{_metric(metrics, 'mrr')}</td>"
                f"<td>{_metric(metrics, 'ndcg@10')}</td>"
                "</tr>"
            )

    cards = []
    for qa in report["per_qa"]:
        supports = []
        for support in qa["gold_supports"]:
            bbox = html.escape(json.dumps(support["bboxes"])) if support["bboxes"] else "unavailable"
            source_link = ""
            if support.get("source_path"):
                try:
                    href = Path(str(support["source_path"])).as_uri() + f"#page={support['page']}"
                    source_link = f" · <a href='{html.escape(href, quote=True)}'>open PDF page</a>"
                except ValueError:
                    source_link = ""
            supports.append(
                "<div class='support'>"
                f"<div><b>{html.escape(support['fact_id'])}</b> · page {support['page']} · "
                f"chunk <code>{html.escape(support['chunk_id'])}</code></div>"
                f"<p>{html.escape(support['text'])}</p>"
                f"<small>bbox ({html.escape(support['bbox_method'])}, alignment "
                f"{support['bbox_alignment_score']:.3f}): {bbox} · "
                f"grounding overlap {support['overlap']:.3f}{source_link}</small>"
                "</div>"
            )
        retrieval_blocks = []
        for name, result in qa["retrieval"].items():
            metrics = result["metrics"]
            chips = []
            for hit in result["fact_hits"]:
                css_class = "hit" if hit["hit"] else "miss"
                symbol = "✓" if hit["hit"] else "✗"
                rank_label = (
                    f" @ {hit['first_hit_rank']}" if hit["first_hit_rank"] else ""
                )
                chips.append(
                    f"<span class='chip {css_class}'>{symbol} "
                    f"{html.escape(hit['fact_id'])}{rank_label}</span>"
                )
            hit_chips = " ".join(chips)
            result_rows = "".join(
                "<tr>"
                f"<td>{row['rank']}</td><td><code>{html.escape(row['chunk_id'])}</code></td>"
                f"<td>{html.escape(str(row['page_start']))}–{html.escape(str(row['page_end']))}</td>"
                f"<td>{html.escape(', '.join(row['covers_fact_ids']) or '—')}</td>"
                f"<td>{html.escape(' '.join(row['text'].split()))}</td></tr>"
                for row in result["retrieved"]
            )
            retrieval_blocks.append(
                "<details class='retriever'><summary>"
                f"{html.escape(name)} · R@5 {_metric(metrics, 'recall@5')} · "
                f"P@5 {_metric(metrics, 'precision@5')} · "
                f"Hop@5 {_metric(metrics, 'hop_completeness@5')}</summary>"
                f"<div class='chips'>{hit_chips}</div>"
                "<table><thead><tr><th>#</th><th>Chunk</th><th>Pages</th><th>Gold hit</th><th>Snippet</th></tr></thead>"
                f"<tbody>{result_rows}</tbody></table></details>"
            )
        search_text = " ".join(
            [str(qa.get("qa_id")), str(qa.get("doc")), str(qa.get("question")), str(qa.get("bridge_entity"))]
        ).casefold()
        cards.append(
            f"<details class='qa' data-hop='{html.escape(str(qa['hop_type']))}' "
            f"data-search='{html.escape(search_text, quote=True)}'>"
            f"<summary><span class='qid'>{html.escape(str(qa['qa_id']))}</span> "
            f"<span class='tag'>{html.escape(str(qa['hop_type']))}</span> "
            f"{html.escape(str(qa['question']))}</summary>"
            f"<p class='answer'><b>Answer:</b> {html.escape(str(qa['answer']))}</p>"
            f"<h4>Gold supports</h4>{''.join(supports)}"
            f"<h4>Retrieval</h4>{''.join(retrieval_blocks)}</details>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-gold Retrieval Report</title><style>
:root{{--bg:#f5f7fb;--panel:#fff;--ink:#172033;--muted:#64748b;--line:#dbe2ea;--accent:#2155cd;--good:#147d50;--bad:#b42318}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}} h1{{margin:0 0 4px;font-size:28px}} .meta{{color:var(--muted);margin-bottom:20px}}
.panel,.qa{{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin:12px 0;box-shadow:0 1px 2px #1020400d}}
.panel{{padding:18px;overflow:auto}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{background:#eef3fb}}
.controls{{display:flex;gap:10px;position:sticky;top:0;background:var(--bg);padding:10px 0;z-index:2}} input,select{{padding:9px 12px;border:1px solid var(--line);border-radius:7px;background:white}} input{{flex:1}}
.qa>summary{{cursor:pointer;padding:14px 16px;font-weight:600}} .qa[open]{{padding-bottom:16px}} .qa>p,.qa>h4,.qa>.support,.qa>.retriever{{margin-left:16px;margin-right:16px}}
.qid{{color:var(--accent)}} .tag,.chip{{display:inline-block;padding:2px 7px;border-radius:99px;background:#e8eefb;font-size:12px}} .support{{border-left:4px solid var(--accent);padding:10px 12px;background:#f8faff;margin-bottom:8px}} .support p{{margin:5px 0}} small{{color:var(--muted)}}
.retriever{{border:1px solid var(--line);border-radius:8px;margin-top:8px;overflow:auto}} .retriever>summary{{padding:10px;cursor:pointer;background:#f8fafc;font-weight:600}} .chips{{padding:8px}} .chip{{margin-right:5px}} .hit{{color:var(--good);background:#e8f7ef}} .miss{{color:var(--bad);background:#fff0ee}} code{{font-size:12px}} .hidden{{display:none}}
</style></head><body><main>
<h1>Multi-gold Retrieval Report</h1>
<div class="meta">{metadata['n_evaluated']} evaluated of {metadata['n_stage_d_pass']} Stage D PASS rows · {html.escape(', '.join(metadata['retrievers']))} · bbox coverage {metadata['bbox_coverage']:.1%} · generated in {metadata['wall_seconds']} s</div>
<section class="panel"><h2>Metric summary</h2><table><thead><tr><th>Retriever</th><th>Track</th><th>N</th><th>Recall@5</th><th>Precision@5</th><th>Hop completeness@5</th><th>MRR</th><th>NDCG@10</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></section>
<div class="controls"><input id="search" placeholder="Filter by ID, document, bridge, or question"><select id="hop"><option value="all">All tracks</option><option value="bridge">Bridge</option><option value="single">Single</option></select></div>
<section id="qas">{''.join(cards)}</section>
<script>const search=document.getElementById('search'),hop=document.getElementById('hop'),cards=[...document.querySelectorAll('.qa')];function filter(){{const q=search.value.toLowerCase(),h=hop.value;cards.forEach(c=>c.classList.toggle('hidden',!(c.dataset.search.includes(q)&&(h==='all'||c.dataset.hop===h))))}}search.addEventListener('input',filter);hop.addEventListener('change',filter);</script>
</main></body></html>"""


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage E multi-gold retrieval report")
    parser.add_argument("facts", type=Path)
    parser.add_argument("qa_pairs", type=Path)
    parser.add_argument("chunk_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_html", type=Path)
    parser.add_argument("--retrievers", nargs="+")
    parser.add_argument("--enriched-qa", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    qa_payload = json.loads(args.qa_pairs.read_text(encoding="utf-8"))
    report = evaluate_multigold(
        facts,
        qa_payload["pairs"],
        args.chunk_root,
        strategies=args.retrievers,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_html.write_text(
        render_html(report), encoding="utf-8", newline="\n"
    )
    if args.enriched_qa:
        args.enriched_qa.parent.mkdir(parents=True, exist_ok=True)
        enriched = enrich_qa_grounding(qa_payload, report)
        args.enriched_qa.write_text(
            json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    logger.info(f"Stage E wrote {args.output_json} and {args.output_html}: {report['metadata']}")
    return 0 if report["metadata"]["n_evaluated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
