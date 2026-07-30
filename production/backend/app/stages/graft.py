"""Real GRAFT pipeline stages — thin wrappers around the worktree's
`src/rag_gt` building blocks (the same call sites `rag_gt.allpdf.pipeline.
run_doc_pipeline` uses, verified against that file).

Heavy imports (rag_gt, torch, docling, pdfplumber) happen INSIDE stage
functions so the API process never pays for them — only the worker does.

Inter-stage state: picklable objects (profiles, ingests, chunk results,
facts, chains, pairs) are persisted to `<run_dir>/state.pkl` after each
stage, which is what makes per-stage retry/resume work. Non-picklable
objects (FAISS index, TypedSFG) live in the in-process `_MEM` carry; if a
retry lands on a stage whose carry is gone, the stage raises with a clear
"retry from graph" message.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

from app.orchestrator.context import StageContext
from app.orchestrator.dag import PipelineDAG, StageDef
from app.stages.topology import GRAFT_TOPOLOGY

_WORKTREE_ROOT = Path(__file__).resolve().parents[4]
_SRC = _WORKTREE_ROOT / "src"

# in-process carry for non-picklable stage products, keyed by run_id
_MEM: dict[str, dict[str, Any]] = {}


def _bootstrap() -> None:
    """Make rag_gt importable and load API credentials."""
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from dotenv import load_dotenv

    for cand in (_WORKTREE_ROOT / ".env", _WORKTREE_ROOT.parents[1] / ".env"):
        if cand.exists():
            load_dotenv(cand)
            break


# -- state ---------------------------------------------------------------


def _state_path(ctx: StageContext) -> Path:
    return ctx.run_dir / "state.pkl"


def _load_state(ctx: StageContext) -> dict:
    p = _state_path(ctx)
    if p.exists():
        with open(p, "rb") as fh:
            return pickle.load(fh)
    return {}


def _save_state(ctx: StageContext, state: dict) -> None:
    with open(_state_path(ctx), "wb") as fh:
        pickle.dump(state, fh)


def _require(state: dict, key: str, produced_by: str) -> Any:
    if key not in state:
        raise RuntimeError(
            f"missing {key!r} in run state — retry from the {produced_by!r} stage"
        )
    return state[key]


def _docs(ctx: StageContext) -> list[tuple[str, str]]:
    pdfs = sorted((ctx.run_dir / "uploads").glob("*.pdf"))
    if not pdfs:
        raise RuntimeError("no PDFs uploaded — the graft pipeline needs at least one PDF")
    return [(p.stem, str(p)) for p in pdfs]


# -- Track 1 stages ---------------------------------------------------------


def profile_stage(ctx: StageContext) -> None:
    _bootstrap()
    from rag_gt.allpdf.preflight import profile_pdf

    docs = _docs(ctx)
    state = _load_state(ctx)
    profiles: dict[str, Any] = {}
    total_pages = 0
    for i, (doc_id, path) in enumerate(docs, 1):
        ctx.check_cancel()
        with ctx.span("processing", f"profile_{doc_id}", input={"path": path}) as sp:
            prof = profile_pdf(path, doc_id)
            profiles[doc_id] = prof
            total_pages += prof.page_count
            sp.output = {
                "pages": prof.page_count,
                "doc_type": prof.doc_type_guess,
                "backend": prof.recommended_backend,
                "scanned": prof.scanned,
            }
        ctx.log(
            "info",
            f"{doc_id}: {prof.page_count} pages, type={prof.doc_type_guess}, "
            f"backend={prof.recommended_backend}",
        )
        ctx.progress(i, len(docs), doc_id)
    state["profiles"] = profiles
    _save_state(ctx, state)
    ctx.metric("docs", len(docs))
    ctx.metric("total_pages", total_pages)


def ingest_stage(ctx: StageContext) -> None:
    _bootstrap()
    from rag_gt.allpdf.ingest import ingest_document

    state = _load_state(ctx)
    profiles = _require(state, "profiles", "profile")
    page_cap = int(ctx.config.get("docling_page_cap", 8))
    ingests: dict[str, Any] = {}
    total_units = 0
    for i, (doc_id, prof) in enumerate(profiles.items(), 1):
        ctx.check_cancel()
        with ctx.span("processing", f"ingest_{doc_id}", input={"backend": prof.recommended_backend}) as sp:
            ing = ingest_document(prof, docling_page_cap=page_cap)
            ingests[doc_id] = ing
            total_units += ing.n_units
            sp.output = {"units": ing.n_units, "chars": ing.char_count, "backend": ing.backend_used}
        ctx.log("info", f"{doc_id}: {ing.n_units} units, {ing.char_count} chars via {ing.backend_used}")
        ctx.progress(i, len(profiles), doc_id)
    state["ingests"] = ingests
    _save_state(ctx, state)
    ctx.metric("units", total_units)


def chunk_stage(ctx: StageContext) -> None:
    _bootstrap()
    from rag_gt.allpdf.chunk import agentic_chunk

    state = _load_state(ctx)
    profiles = _require(state, "profiles", "profile")
    ingests = _require(state, "ingests", "ingest")
    chunk_results: dict[str, Any] = {}
    total_chunks = 0
    for i, doc_id in enumerate(profiles, 1):
        ctx.check_cancel()
        with ctx.span("processing", f"chunk_{doc_id}") as sp:
            cr = agentic_chunk(profiles[doc_id], ingests[doc_id])
            chunk_results[doc_id] = cr
            total_chunks += len(cr.chunks)
            sp.output = {"chunks": len(cr.chunks), "strategy": cr.strategy}
        ctx.log("info", f"{doc_id}: {len(cr.chunks)} chunks via strategy={cr.strategy}")
        # chunks are the contract with the RAG track — persist per doc
        art = ctx.run_dir / "artifacts" / f"chunks_{doc_id}.json"
        art.write_text(json.dumps(cr.chunks, ensure_ascii=False), encoding="utf-8")
        ctx.artifact(f"chunks_{doc_id}.json", art, kind="json")
        ctx.progress(i, len(profiles), doc_id)
    state["chunk_results"] = chunk_results
    _save_state(ctx, state)
    ctx.metric("chunks", total_chunks)


def _fact_provenance(f: Any) -> dict:
    """Auditable fact record — mirrors `_prov()` in rag_gt.allpdf.pipeline."""
    sp = f.supporting_spans[0] if f.supporting_spans else None
    return {
        "fact_id": f.fact_id,
        "text": f.text,
        "canonical_form": f.canonical_form,
        "raw_text": getattr(f, "raw_text", None),
        "self_containment_score": f.self_containment_score,
        "page_start": getattr(sp, "page_start", None) if sp else None,
        "char_start": getattr(sp, "char_start", None) if sp else None,
        "char_end": getattr(sp, "char_end", None) if sp else None,
        "chunk_id": getattr(sp, "chunk_id", None) if sp else None,
        "bboxes": [
            b.to_dict() if hasattr(b, "to_dict") else b
            for b in (getattr(sp, "bboxes", []) or [])
        ] if sp else [],
    }


def extract_stage(ctx: StageContext) -> None:
    _bootstrap()
    from rag_gt.allpdf.extract import extract_sfu_facts

    state = _load_state(ctx)
    profiles = _require(state, "profiles", "profile")
    chunk_results = _require(state, "chunk_results", "chunk")
    live = ctx.config.get("llm_mode", "import") == "live"
    llm = _traced_llm(ctx, "gt") if live else None
    cap = int(ctx.config.get("llm_chunk_cap", 80)) if live else 0

    extracts: dict[str, Any] = {}
    total = 0
    for i, doc_id in enumerate(profiles, 1):
        ctx.check_cancel()
        with ctx.span(
            "processing", f"extract_{doc_id}",
            input={"path": "llm" if live else "rule-based", "llm_chunk_cap": cap},
        ) as sp:
            res = extract_sfu_facts(
                profiles[doc_id], chunk_results[doc_id], llm=llm,
                llm_chunk_cap=cap,
                cache_dir=str(ctx.run_dir / "llm_cache" / f"extract_{doc_id}") if live else None,
            )
            extracts[doc_id] = res
            total += res.n_facts
            sp.output = {"facts": res.n_facts, "canonical": res.n_canonical, "path": res.extraction_path}
        ctx.log("info", f"{doc_id}: {res.n_facts} facts ({res.extraction_path}), {res.n_canonical} canonical")
        ctx.progress(i, len(profiles), doc_id)
    state["extracts"] = extracts
    _save_state(ctx, state)
    ctx.metric("facts_extracted", total)


def clean_stage(ctx: StageContext) -> None:
    _bootstrap()
    from rag_gt.allpdf.filter_adaptive import filter_facts_adaptive

    state = _load_state(ctx)
    profiles = _require(state, "profiles", "profile")
    extracts = _require(state, "extracts", "extract")
    kept_by_doc: dict[str, list] = {}
    total_kept = total_dropped = 0
    for i, doc_id in enumerate(profiles, 1):
        ctx.check_cancel()
        with ctx.span("processing", f"clean_{doc_id}") as sp:
            kept, drop_counts = filter_facts_adaptive(extracts[doc_id].facts, profiles[doc_id])
            kept_by_doc[doc_id] = kept
            dropped = sum(drop_counts.values())
            total_kept += len(kept)
            total_dropped += dropped
            sp.output = {"kept": len(kept), "dropped": dropped, "reasons": drop_counts}
        for reason, n in sorted(drop_counts.items(), key=lambda kv: -kv[1]):
            ctx.log("info", f"{doc_id}: dropped {n} facts — {reason}")
        # provenance artifact: every kept fact with page/char span/bbox
        art = ctx.run_dir / "artifacts" / f"facts_{doc_id}.json"
        art.write_text(
            json.dumps([_fact_provenance(f) for f in kept], ensure_ascii=False),
            encoding="utf-8",
        )
        ctx.artifact(f"facts_{doc_id}.json", art, kind="json")
        ctx.progress(i, len(profiles), doc_id)
    state["kept_facts"] = kept_by_doc
    _save_state(ctx, state)
    ctx.metric("facts_kept", total_kept)
    ctx.metric("facts_dropped", total_dropped)


class BudgetExceeded(RuntimeError):
    pass


class CostBudget:
    """Shared across all TracedLLM instances of a run; aborts at the cap."""

    def __init__(self, max_cost_usd: float) -> None:
        self.max_cost_usd = float(max_cost_usd)
        self.spent = 0.0

    def add(self, cost: float) -> None:
        self.spent += cost
        if self.max_cost_usd > 0 and self.spent > self.max_cost_usd:
            raise BudgetExceeded(
                f"LLM budget exceeded: ${self.spent:.4f} > ${self.max_cost_usd:.4f} cap"
            )


_PAYLOAD_CAP = 4000  # chars of prompt/output stored per span


class TracedLLM:
    """Wraps the rag_gt `LLM` protocol; one `llm` span per generate() call."""

    def __init__(self, inner: Any, ctx: StageContext, role: str, budget: CostBudget) -> None:
        self._inner = inner
        self._ctx = ctx
        self._role = role
        self._budget = budget
        self._price_in = float(ctx.config.get("price_in_per_mtok", 0.0))
        self._price_out = float(ctx.config.get("price_out_per_mtok", 0.0))

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        with self._ctx.span(
            "llm", f"{self._role}.generate", input={"prompt": prompt[:_PAYLOAD_CAP]}
        ) as sp:
            out = self._inner.generate(prompt, temperature=temperature, max_tokens=max_tokens)
            tok_in = len(prompt) // 4  # char-estimate, same convention as cost_tracker
            tok_out = len(out) // 4
            cost = tok_in / 1e6 * self._price_in + tok_out / 1e6 * self._price_out
            sp.output = {"text": out[:_PAYLOAD_CAP]}
            sp.tokens_in = tok_in
            sp.tokens_out = tok_out
            sp.cost_usd = cost
            sp.extra = {"temperature": temperature, "max_tokens": max_tokens}
            self._budget.add(cost)
        return out


def _traced_llm(ctx: StageContext, role: str, budget: CostBudget | None = None):
    from rag_gt.core.llm import get_llm

    budget = budget or CostBudget(float(ctx.config.get("max_cost_usd", 5.0)))
    return TracedLLM(get_llm(role), ctx, role, budget)


def _is_import(ctx: StageContext) -> bool:
    return ctx.config.get("llm_mode", "import") != "live"


def _skip_import(ctx: StageContext, what: str) -> bool:
    if _is_import(ctx):
        ctx.log("info", f"{ctx.stage}: llm_mode=import — {what} taken from the imported GT")
        return True
    return False


DEFAULT_GT_IMPORT = _WORKTREE_ROOT / "data" / "eval_results" / "allpdf_qa_pairs_verified_bbox.json"


def graph_stage(ctx: StageContext) -> None:
    _bootstrap()
    if _skip_import(ctx, "the fact graph"):
        return
    from rag_gt.allpdf.pipeline import _build_graph, _embed_and_index

    state = _load_state(ctx)
    profiles = _require(state, "profiles", "profile")
    kept = _require(state, "kept_facts", "clean")
    budget = CostBudget(float(ctx.config.get("max_cost_usd", 5.0)))
    answer_llm = _traced_llm(ctx, "answer", budget)
    max_pairs = int(ctx.config.get("max_pairs", 400))

    mem = _MEM.setdefault(ctx.run_id, {})
    mem["budget"] = budget
    mem["sfg"] = {}
    total_edges = 0
    for i, doc_id in enumerate(profiles, 1):
        ctx.check_cancel()
        facts = kept[doc_id]
        eff = min(max_pairs, max(60, len(facts) * 3))
        with ctx.span("embedding", f"embed_{doc_id}", input={"facts": len(facts)}) as esp:
            _, index, embed_sec = _embed_and_index(facts)
            esp.output = {"seconds": embed_sec}
            esp.extra = {"index": "faiss"}
        ctx.log("info", f"{doc_id}: embedded {len(facts)} facts in {embed_sec}s")
        with ctx.span("processing", f"graph_{doc_id}", input={"candidate_budget": eff}) as sp:
            sfg = _build_graph(facts, index, answer_llm, profiles[doc_id].doc_type_guess, eff)
            mem["sfg"][doc_id] = sfg
            total_edges += sfg.edge_count
            sp.output = {"edges": sfg.edge_count, "classified_pairs": sfg.classified_pairs}
        ctx.log("info", f"{doc_id}: {sfg.edge_count} edges from {sfg.classified_pairs} classified pairs")
        ctx.progress(i, len(profiles), doc_id)
    ctx.metric("edges", total_edges)
    ctx.metric("llm_cost_usd", round(budget.spent, 6))


def sample_stage(ctx: StageContext) -> None:
    _bootstrap()
    if _skip_import(ctx, "the chains"):
        return
    import random

    from rag_gt.allpdf.pipeline import _build_chains

    state = _load_state(ctx)
    kept = _require(state, "kept_facts", "clean")
    mem = _MEM.get(ctx.run_id, {})
    if "sfg" not in mem:
        raise RuntimeError("fact graph not in memory — retry from the 'graph' stage")
    rng = random.Random(int(ctx.config.get("seed", 42)))
    target = int(ctx.config.get("target_chains", 20))
    mh_chains = int(ctx.config.get("multihop_chains", 0))

    chains_by_doc: dict[str, list] = {}
    n_single_total = n_multi_total = 0
    docs = list(kept)
    for i, doc_id in enumerate(docs, 1):
        ctx.check_cancel()
        sfg = mem["sfg"][doc_id]
        with ctx.span("processing", f"sample_{doc_id}") as sp:
            chains, n_single, n_two = _build_chains(sfg, kept[doc_id], target, rng)
            if mh_chains > 0:
                local = {tuple(c.fact_ids) for c in chains}
                mh = sfg.walk_typed_paths(
                    {2: mh_chains}, rng, cross_page_bias=2.0,
                    require_cross_page=True, min_page_gap=1,
                )
                chains += [c for c in mh if tuple(c.fact_ids) not in local]
            chains_by_doc[doc_id] = chains
            n_single_total += n_single
            n_multi_total += len(chains) - n_single
            sp.output = {"chains": len(chains), "single": n_single}
        ctx.log("info", f"{doc_id}: {len(chains)} chains ({n_single} single-fact)")
        ctx.progress(i, len(docs), doc_id)
    state["chains"] = chains_by_doc
    _save_state(ctx, state)
    ctx.metric("chains_single", n_single_total)
    ctx.metric("chains_multihop", n_multi_total)


def qagen_stage(ctx: StageContext) -> None:
    _bootstrap()
    if _skip_import(ctx, "the QA pairs"):
        return
    from rag_gt.allpdf.pipeline import _run_qgen

    state = _load_state(ctx)
    kept = _require(state, "kept_facts", "clean")
    chains_by_doc = _require(state, "chains", "sample")
    mem = _MEM.get(ctx.run_id, {})
    budget = mem.get("budget") or CostBudget(float(ctx.config.get("max_cost_usd", 5.0)))
    llm = _traced_llm(ctx, "gt", budget)
    answer_llm = _traced_llm(ctx, "answer", budget)
    score_necessity = bool(ctx.config.get("score_necessity", False))

    pairs_by_doc: dict[str, list] = {}
    obs_by_doc: dict[str, dict] = {}
    total = 0
    docs = list(chains_by_doc)
    for i, doc_id in enumerate(docs, 1):
        ctx.check_cancel()
        facts_by_id = {f.fact_id: f for f in kept[doc_id]}
        with ctx.span("processing", f"qagen_{doc_id}", input={"chains": len(chains_by_doc[doc_id])}) as sp:
            pairs = _run_qgen(
                chains_by_doc[doc_id], facts_by_id, llm, answer_llm,
                score_necessity=score_necessity,
            )
            pairs_by_doc[doc_id] = pairs
            obs_by_doc[doc_id] = dict(getattr(_run_qgen, "last_obs", {}) or {})
            total += len(pairs)
            sp.output = {"pairs": len(pairs)}
        ctx.log("info", f"{doc_id}: {len(pairs)} QA pairs generated")
        ctx.progress(i, len(docs), doc_id)
    state["pairs"] = pairs_by_doc
    state["qgen_obs"] = obs_by_doc
    _save_state(ctx, state)
    ctx.metric("pairs_generated", total)
    ctx.metric("llm_cost_usd", round(budget.spent, 6))


def verify_stage(ctx: StageContext) -> None:
    _bootstrap()
    if _skip_import(ctx, "verification results"):
        return
    state = _load_state(ctx)
    obs_by_doc = state.get("qgen_obs", {})
    for doc_id, obs in obs_by_doc.items():
        for k, v in sorted(obs.items()):
            if isinstance(v, (int, float)):
                ctx.log("info", f"{doc_id}: {k}={v}")
    # verification is inline in _run_qgen (reframe/tautology/grounding gates);
    # surface the totals as metrics
    total_pairs = sum(len(p) for p in state.get("pairs", {}).values())
    ctx.metric("pairs_verified", total_pairs)


def gt_dataset_stage(ctx: StageContext) -> None:
    _bootstrap()
    state = _load_state(ctx)
    art = ctx.run_dir / "artifacts" / "gt.jsonl"

    if _is_import(ctx):
        src = Path(str(ctx.config.get("gt_import_path", DEFAULT_GT_IMPORT)))
        if not src.exists():
            raise RuntimeError(f"gt_import_path not found: {src}")
        uploaded = [doc_id for doc_id, _ in _docs(ctx)]
        with ctx.span("processing", "import_gt", input={"path": str(src)}) as sp:
            data = json.loads(src.read_text(encoding="utf-8"))
            all_pairs = data["pairs"] if isinstance(data, dict) and "pairs" in data else data
            # keep only pairs whose source doc matches an uploaded PDF, so the
            # eval track scores retrieval over a corpus that actually contains
            # the evidence
            def _matches(pair_doc: str) -> bool:
                base = pair_doc[:-5] if pair_doc.endswith("_full") else pair_doc
                return any(u.startswith(base) or base.startswith(u) for u in uploaded)

            pairs = [p for p in all_pairs if _matches(str(p.get("doc", "")))]
            sp.output = {"pairs_total": len(all_pairs), "pairs_matching_corpus": len(pairs)}
        ctx.log(
            "info",
            f"imported {len(pairs)}/{len(all_pairs)} verified GT pairs from "
            f"{src.name} matching uploaded docs {uploaded}",
        )
        if not pairs:
            gt_docs = sorted({str(p.get("doc", "")) for p in all_pairs})
            raise RuntimeError(
                f"no imported GT pairs match the uploaded PDFs {uploaded}. "
                f"GT covers: {gt_docs}"
            )
    else:
        pairs_by_doc = _require(state, "pairs", "qagen")
        pairs = [p for doc_pairs in pairs_by_doc.values() for p in doc_pairs]

    with open(art, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")
    ctx.artifact("gt.jsonl", art, kind="jsonl")
    state["gt_pairs"] = pairs
    _save_state(ctx, state)
    ctx.metric("pairs_kept", len(pairs))


def _todo_stage(ctx: StageContext) -> None:
    """Placeholder for stages not yet implemented."""
    ctx.log("info", f"{ctx.stage}: arrives with a later plan")


# -- Track 2: RAG build ------------------------------------------------------


def _corpus_chunks(state: dict) -> list[dict]:
    chunk_results = _require(state, "chunk_results", "chunk")
    chunks: list[dict] = []
    for cr in chunk_results.values():
        chunks.extend(cr.chunks)
    bad = [c for c in chunks if not c.get("chunk_id") or not c.get("text")]
    if bad:
        raise RuntimeError(f"{len(bad)} chunks missing chunk_id/text — cannot index")
    return chunks


def bm25_index_stage(ctx: StageContext) -> None:
    _bootstrap()
    from rag_gt.rag.retriever import BM25Retriever

    state = _load_state(ctx)
    chunks = _corpus_chunks(state)
    with ctx.span("processing", "build_bm25", input={"chunks": len(chunks)}) as sp:
        retriever = BM25Retriever(chunks)
        sp.output = {"chunks": retriever.n_chunks}
    _MEM.setdefault(ctx.run_id, {})["bm25"] = retriever
    ctx.log("info", f"BM25 index over {len(chunks)} chunks")
    ctx.metric("indexed_chunks", len(chunks))


def vector_index_stage(ctx: StageContext) -> None:
    _bootstrap()
    from rag_gt.rag.retriever import DenseRetriever

    state = _load_state(ctx)
    chunks = _corpus_chunks(state)
    model = str(ctx.config.get("embed_model", "all-MiniLM-L6-v2"))
    with ctx.span("embedding", "embed_chunks", input={"chunks": len(chunks)}) as sp:
        retriever = DenseRetriever(chunks, model=model)
        sp.output = {"vectors": len(chunks), "dim": int(retriever._embeddings.shape[1])}
        sp.extra = {"model": model, "device": "cpu"}
    _MEM.setdefault(ctx.run_id, {})["dense"] = retriever
    ctx.log("info", f"dense index over {len(chunks)} chunks ({model})")
    ctx.metric("indexed_chunks", len(chunks))


def fusion_stage(ctx: StageContext) -> None:
    _bootstrap()
    from rag_gt.rag.retriever import HybridRetriever

    state = _load_state(ctx)
    chunks = _corpus_chunks(state)
    mem = _MEM.get(ctx.run_id, {})
    if "dense" not in mem or "bm25" not in mem:
        raise RuntimeError("indexes not in memory — retry from 'bm25_index'/'vector_index'")
    with ctx.span("processing", "build_fusion") as sp:
        hybrid = HybridRetriever(chunks, dense_retriever=mem["dense"])
        sp.output = {"rrf_k": hybrid.rrf_k}
    mem["hybrid"] = hybrid
    ctx.log("info", f"hybrid RRF fusion ready (rrf_k={hybrid.rrf_k})")


def rerank_warmup_stage(ctx: StageContext) -> None:
    _bootstrap()
    from app.ragengine.rerank import Reranker

    model = str(ctx.config.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    with ctx.span("embedding", "load_reranker", input={"model": model}) as sp:
        reranker = Reranker(model)
        # warm-up inference so first sweep query isn't cold
        reranker._model.predict([("warm", "up")])
        sp.output = {"ready": True}
        sp.extra = {"model": model, "device": "cpu"}
    _MEM.setdefault(ctx.run_id, {})["reranker"] = reranker
    ctx.log("info", f"cross-encoder reranker ready ({model})")


# -- Track 3: evaluation & self-improvement ---------------------------------


def _retrieve(mem: dict, mode: str, query: str, top_k: int) -> list[tuple[str, float]]:
    """Retrieve (chunk_id, score) with the given mode from the run carry."""
    if mode == "bm25":
        return mem["bm25"].retrieve(query, top_k=top_k)
    if mode == "vector":
        return mem["dense"].retrieve(query, top_k=top_k)
    if mode == "hybrid":
        return mem["hybrid"].retrieve(query, top_k=top_k)
    if mode == "hybrid+rerank":
        candidates = mem["hybrid"].retrieve(query, top_k=max(top_k * 3, 30))
        return mem["reranker"].rerank(
            query, candidates, mem["bm25"].get_chunk_text, top_k=top_k
        )
    raise ValueError(f"unknown retrieval mode {mode!r}")


def _eval_mem(ctx: StageContext) -> dict:
    mem = _MEM.get(ctx.run_id, {})
    missing = {"bm25", "dense", "hybrid", "reranker"} - mem.keys()
    if missing:
        raise RuntimeError(
            f"retrievers not in memory ({sorted(missing)}) — retry from 'bm25_index'"
        )
    return mem


def _eval_cell(
    ctx: StageContext,
    mem: dict,
    pairs: list[dict],
    mode: str,
    top_k: int,
    span_cap: int,
) -> tuple[dict, list[dict]]:
    """Score all pairs for one (mode, top_k) cell. Returns (aggregate, rows)."""
    from app.evaluation.matching import aggregate, score_pair, unit_hit

    rows = []
    for i, pair in enumerate(pairs):
        ctx.check_cancel()
        query = str(pair.get("question", ""))
        ranked = _retrieve(mem, mode, query, top_k)
        texts = [mem["bm25"].get_chunk_text(cid) for cid, _ in ranked]
        row = score_pair(pair, texts)
        row["retrieved"] = [cid for cid, _ in ranked]
        rows.append(row)
        if i < span_cap:
            units = [c["text"] for c in (pair.get("answer_clauses") or []) if c.get("text")] or [pair.get("answer", "")]
            with ctx.span(
                "retrieval", f"{mode}@{top_k}",
                input={"query": query, "mode": mode, "top_k": top_k},
            ) as sp:
                sp.output = {
                    "recall": row["recall"],
                    "hit": row["hit"],
                    "candidates": [
                        {
                            "chunk_id": cid,
                            "score": round(score, 5),
                            "relevant": any(unit_hit(u, [mem["bm25"].get_chunk_text(cid)]) for u in units),
                            "text": mem["bm25"].get_chunk_text(cid)[:300],
                        }
                        for cid, score in ranked
                    ],
                    "gold_units": [u[:300] for u in units],
                }
                sp.extra = {"qa_id": pair.get("qa_id"), "hop_type": pair.get("hop_type")}
    return aggregate(rows), rows


def sweep_stage(ctx: StageContext) -> None:
    _bootstrap()
    import random

    state = _load_state(ctx)
    pairs = _require(state, "gt_pairs", "gt_dataset")
    mem = _eval_mem(ctx)

    modes = list(ctx.config.get("sweep_modes", ["bm25", "vector", "hybrid", "hybrid+rerank"]))
    top_ks = [int(k) for k in ctx.config.get("sweep_top_ks", [3, 5, 10])]
    sample_n = int(ctx.config.get("sweep_sample", 100))
    rng = random.Random(int(ctx.config.get("seed", 42)))
    sample = pairs if (sample_n <= 0 or sample_n >= len(pairs)) else rng.sample(pairs, sample_n)

    # one batched query embedding for all dense/hybrid cells
    mem["dense"].prime_queries([str(p.get("question", "")) for p in sample])

    cells = [(m, k) for m in modes for k in top_ks]
    leaderboard = []
    for i, (mode, top_k) in enumerate(cells, 1):
        agg, _ = _eval_cell(ctx, mem, sample, mode, top_k, span_cap=3)
        leaderboard.append(
            {
                "config": f"{mode}@{top_k}",
                "chunking": "profiler",
                "retrieval": mode,
                "top_k": top_k,
                "recall": round(agg["recall"], 4),
                "precision": round(agg["precision"], 4),
                "precision_rw": round(agg["precision_rw"], 4),
                "f1": round(agg["f1"], 4),
                "f1_rw": round(agg["f1_rw"], 4),
                "hit_rate": round(agg["hit_rate"], 4),
                "n": agg["n"],
            }
        )
        ctx.log(
            "info",
            f"{mode}@{top_k}: recall={agg['recall']:.3f} prec_rw={agg['precision_rw']:.3f} "
            f"f1_rw={agg['f1_rw']:.3f} (raw precision={agg['precision']:.3f}, n={agg['n']})",
        )
        ctx.progress(i, len(cells), f"{mode}@{top_k}")

    from app.evaluation.matching import pick_winner

    best = pick_winner(leaderboard)
    for r in leaderboard:
        r["winner"] = r is best
    art = ctx.run_dir / "artifacts" / "leaderboard.json"
    art.write_text(json.dumps({"rows": leaderboard}, indent=1), encoding="utf-8")
    ctx.artifact("leaderboard.json", art, kind="json")
    state["leaderboard"] = leaderboard
    _save_state(ctx, state)
    ctx.metric("configs_evaluated", len(cells))
    ctx.metric("sweep_pairs", len(sample))


def leaderboard_stage(ctx: StageContext) -> None:
    _bootstrap()
    state = _load_state(ctx)
    rows = _require(state, "leaderboard", "sweep")
    for r in sorted(rows, key=lambda r: -r["f1_rw"])[:5]:
        ctx.log("info", f"{'* ' if r.get('winner') else '  '}{r['config']}: f1_rw={r['f1_rw']:.3f} (raw f1={r['f1']:.3f})")
    ctx.metric("best_f1", max(r["f1"] for r in rows))
    ctx.metric("best_f1_rw", max(r["f1_rw"] for r in rows))


def select_config_stage(ctx: StageContext) -> None:
    _bootstrap()
    state = _load_state(ctx)
    rows = _require(state, "leaderboard", "sweep")
    winner = next(r for r in rows if r.get("winner"))
    state["winner"] = winner
    _save_state(ctx, state)
    ctx.log(
        "info",
        f"selected {winner['config']} — f1_rw={winner['f1_rw']:.3f} "
        f"(recall={winner['recall']:.3f}, precision_rw={winner['precision_rw']:.3f}, raw f1={winner['f1']:.3f})",
    )
    ctx.metric("winner_top_k", winner["top_k"])


def final_eval_stage(ctx: StageContext) -> None:
    _bootstrap()
    from app.evaluation.matching import aggregate

    state = _load_state(ctx)
    pairs = _require(state, "gt_pairs", "gt_dataset")
    winner = _require(state, "winner", "select_config")
    mem = _eval_mem(ctx)
    span_cap = int(ctx.config.get("eval_span_cap", 300))

    mem["dense"].prime_queries([str(p.get("question", "")) for p in pairs])
    agg, rows = _eval_cell(ctx, mem, pairs, winner["retrieval"], winner["top_k"], span_cap)

    results = {
        "winner": winner,
        "aggregate": agg,
        "rows": [
            {
                "qa_id": r["qa_id"], "hop_type": r["hop_type"],
                "recall": round(r["recall"], 4), "precision": round(r["precision"], 4),
                "precision_rw": round(r["precision_rw"], 4),
                "hit": r["hit"], "retrieved": r["retrieved"],
            }
            for r in rows
        ],
    }
    art = ctx.run_dir / "artifacts" / "eval_results.json"
    art.write_text(json.dumps(results, indent=1), encoding="utf-8")
    ctx.artifact("eval_results.json", art, kind="json")
    state["eval_results"] = results
    _save_state(ctx, state)
    ctx.metric("final_recall", round(agg["recall"], 4))
    ctx.metric("final_precision", round(agg["precision"], 4))
    ctx.metric("final_precision_rw", round(agg["precision_rw"], 4))
    ctx.metric("final_f1", round(agg["f1"], 4))
    ctx.metric("final_f1_rw", round(agg["f1_rw"], 4))
    ctx.log(
        "info",
        f"final eval at {winner['config']}: recall={agg['recall']:.3f} "
        f"precision={agg['precision']:.3f} f1={agg['f1']:.3f} over {agg['n']} pairs",
    )


def report_stage(ctx: StageContext) -> None:
    _bootstrap()
    from app.evaluation.report import build_report

    state = _load_state(ctx)
    eval_results = _require(state, "eval_results", "final_eval")
    leaderboard = _require(state, "leaderboard", "sweep")
    gt_pairs = _require(state, "gt_pairs", "gt_dataset")
    chunks = _corpus_chunks(state)
    chunk_texts = {c["chunk_id"]: c.get("text", "") for c in chunks}
    docs = [doc_id for doc_id, _ in _docs(ctx)]

    with ctx.span("processing", "build_report") as sp:
        html_doc = build_report(
            run_id=ctx.run_id,
            docs=docs,
            n_pairs=len(gt_pairs),
            leaderboard=leaderboard,
            eval_results=eval_results,
            gt_pairs=gt_pairs,
            chunk_texts=chunk_texts,
        )
        art = ctx.run_dir / "artifacts" / "report.html"
        art.write_text(html_doc, encoding="utf-8")
        sp.output = {"bytes": len(html_doc)}
    ctx.artifact("report.html", art, kind="html")
    ctx.log("info", f"report.html written ({len(html_doc)} bytes)")


# -- DAG builders ---------------------------------------------------------

_P3_IMPLEMENTED = {
    "profile": profile_stage,
    "ingest": ingest_stage,
    "chunk": chunk_stage,
    "extract": extract_stage,
    "clean": clean_stage,
}

_IMPLEMENTED = {
    **_P3_IMPLEMENTED,
    "graph": graph_stage,
    "sample": sample_stage,
    "qagen": qagen_stage,
    "verify": verify_stage,
    "gt_dataset": gt_dataset_stage,
    "bm25_index": bm25_index_stage,
    "vector_index": vector_index_stage,
    "fusion": fusion_stage,
    "rerank_warmup": rerank_warmup_stage,
    "sweep": sweep_stage,
    "leaderboard": leaderboard_stage,
    "select_config": select_config_stage,
    "final_eval": final_eval_stage,
    "report": report_stage,
}


def build_graft_p3_dag() -> PipelineDAG:
    """Front-end only (profile -> ingest -> chunk -> extract -> clean):
    integration-test pipeline, zero LLM."""
    stages = [
        StageDef(s.name, s.label, s.deps, s.track, _P3_IMPLEMENTED[s.name])
        for s in GRAFT_TOPOLOGY
        if s.name in _P3_IMPLEMENTED
    ]
    return PipelineDAG(stages)


def build_graft_dag() -> PipelineDAG:
    """Full pipeline. GT track is real; rag/eval stages are Plan-4 no-ops."""
    stages = [
        StageDef(s.name, s.label, s.deps, s.track, _IMPLEMENTED.get(s.name, _todo_stage))
        for s in GRAFT_TOPOLOGY
    ]
    return PipelineDAG(stages)


def register(pipelines: dict) -> None:
    pipelines["graft_p3"] = build_graft_p3_dag
    pipelines["graft"] = build_graft_dag
