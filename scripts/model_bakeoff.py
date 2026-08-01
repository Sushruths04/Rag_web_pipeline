"""Multi-model Q&A bake-off over a frozen set of fact chains.

Stages 0-6 run once to produce 30 fact chains; only stage 7 (question
generation + answer generation) varies by model. That keeps the evidence
identical across columns, so the comparison measures the generator and not the
extractor.

Usage:
    python scripts/model_bakeoff.py --build-chains     # stages 0-6, writes chains.json
    python scripts/model_bakeoff.py --run              # stage 7 for all 5 models
    python scripts/model_bakeoff.py --build-chains --run

See docs/superpowers/specs/2026-08-02-model-bakeoff-design.md
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

OUT_DIR = Path("data/model_bakeoff")
CHAINS_PATH = OUT_DIR / "chains.json"
RESULTS_PATH = OUT_DIR / "results.json"

# The anchor model drives stages 0-6 (extraction cache hits + edge
# classification), so the evidence is built once by a single known model.
ANCHOR_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"

MODELS = [
    ("google/gemma-3-27b-it", "Gemma 3 27B", "~27B small"),
    ("meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B", "70B mid"),
    ("Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen3 235B-A22B", "anchor (current)"),
    ("deepseek-ai/DeepSeek-V4-Pro", "DeepSeek V4 Pro", "frontier open"),
    ("moonshotai/Kimi-K3", "Kimi K3", "frontier open"),
]

DOCS = [
    ("din_iso_6507_1",
     "data/gt_input_DINENISO65071ENG/DIN EN ISO 6507-1-ENG.pdf",
     "data/reruns_postfix/din_iso_6507_1/checkpoints/s3_chunk_cache"),
    ("din_iso_3452_1",
     "data/gt_input_DINENISO34521ENG/DIN EN ISO 3452-1-ENG.pdf",
     "data/reruns_postfix/din_iso_3452_1/checkpoints/s3_chunk_cache"),
]

CHAINS_PER_DOC = 15
SEED = 42
CONCURRENCY = 6
CANDIDATE_BUDGET = 150   # Stage-5 edge-classification budget; see _build_doc_chains

# Mirrors pipeline._run_qgen's hint so the reframe path behaves identically.
_CONCISE_HINT = (
    "Ask ONE focused question in a single short clause. Do not join two "
    "questions with 'and'; combine the facts into a single relationship "
    "question. Do NOT count or number the words — just write the question."
)


# --------------------------------------------------------------------------
# Stage 0-6: build the frozen chain set
# --------------------------------------------------------------------------

def _make_llm(model: str):
    """An APILLM bound to one model id, using the .env endpoint and key."""
    from rag_gt.core.llm import APILLM

    base = os.getenv("API_BASE_URL", "").strip().strip('"').strip("'")
    key = os.getenv("API_KEY", "").strip().strip('"').strip("'")
    return APILLM(base_url=base, api_key=key, model=model)


def _fact_payload(f) -> dict:
    """Serialize a Fact losslessly enough to rebuild it for generation.

    Critically includes `role` — questions._format_role_shape_hint steers the
    question form from it, and the existing gt_pairs.json does not persist it.
    """
    return {
        "fact_id": f.fact_id,
        "text": f.text,
        "raw_text": getattr(f, "raw_text", "") or f.text,
        "canonical_form": f.canonical_form or f.text,
        "role": str(f.role),
        "weight": f.weight,
        "self_containment_score": f.self_containment_score,
        "self_containment_known": getattr(f, "self_containment_known", False),
        "spans": [s.to_dict() for s in (f.supporting_spans or [])],
        "page_start": (
            f.supporting_spans[0].page_start if f.supporting_spans else None
        ),
    }


def _fact_from_payload(d: dict):
    from rag_gt.core.types import Fact, Span

    return Fact(
        fact_id=d["fact_id"],
        text=d["text"],
        raw_text=d.get("raw_text") or d["text"],
        canonical_form=d.get("canonical_form") or d["text"],
        role=d.get("role", "descriptive"),
        weight=d.get("weight", 0.5),
        self_containment_score=d.get("self_containment_score", 1.0),
        self_containment_known=d.get("self_containment_known", False),
        supporting_spans=[Span.from_any(s) for s in d.get("spans", [])],
    )


def _build_doc_chains(doc_id: str, pdf_path: str, cache_dir: str, llm) -> List[dict]:
    """Run stages 0-6 for one document and return selectable chain dicts."""
    from rag_gt.allpdf.chunk import agentic_chunk
    from rag_gt.allpdf.extract import extract_sfu_facts
    from rag_gt.allpdf.filter_adaptive import filter_facts_adaptive
    from rag_gt.allpdf.ingest import ingest_document
    from rag_gt.allpdf.preflight import profile_pdf
    from rag_gt.allpdf.pipeline import (
        _any_fragment_fact,
        _build_chains,
        _build_graph,
        _embed_and_index,
    )

    logger.info(f"[bakeoff] {doc_id} — stages 0-2 (ingest + chunk)")
    profile = profile_pdf(pdf_path, doc_id)
    ingest = ingest_document(profile)
    chunk_result = agentic_chunk(profile, ingest)

    logger.info(f"[bakeoff] {doc_id} — stage 3 (extraction; cache_dir={cache_dir})")
    extract = extract_sfu_facts(
        profile, chunk_result, llm=llm,
        llm_chunk_cap=30,
        cache_dir=cache_dir,
    )
    logger.info(f"[bakeoff] {doc_id} — stage 3 gave {extract.n_facts} facts")

    kept_facts, drop_counts = filter_facts_adaptive(extract.facts, profile)
    logger.info(f"[bakeoff] {doc_id} — stage 4 kept {len(kept_facts)} facts")

    logger.info(f"[bakeoff] {doc_id} — stage 5 (embed + typed graph)")
    _, index, embed_sec = _embed_and_index(kept_facts)
    # The pipeline default would be min(400, n_facts*3) = 400 here. Edge
    # classification is one LLM call per candidate and ran at ~7 pairs/min, so
    # 400 costs ~55 min/doc. Only ~5 two-fact chains per document are needed,
    # and a probe run had already accepted 41 edges by pair 125. 150 keeps a
    # comfortable margin at a third of the wall time. This narrows the two-fact
    # POOL; it does not bias the comparison, since all models see whichever
    # chains are selected.
    effective_max_pairs = CANDIDATE_BUDGET
    sfg = _build_graph(
        kept_facts, index, llm, profile.doc_type_guess, effective_max_pairs
    )
    logger.info(f"[bakeoff] {doc_id} — stage 5 gave {sfg.edge_count} edges")

    rng = random.Random(SEED)
    chains, n_single, n_two = _build_chains(sfg, kept_facts, 40, rng)
    logger.info(f"[bakeoff] {doc_id} — stage 6: {n_single} single + {n_two} two-fact")

    facts_by_id = {f.fact_id: f for f in kept_facts}
    out: List[dict] = []
    for c in chains:
        cf = [facts_by_id[fid] for fid in c.fact_ids if fid in facts_by_id]
        if not cf:
            continue
        # The pipeline skips fragment-anchored chains, so they would render as
        # empty cells for every model. Exclude them at selection time instead.
        if _any_fragment_fact(cf):
            continue
        out.append({
            "doc_id": doc_id,
            "fact_ids": list(c.fact_ids),
            "depth": len(cf),
            "chain_edges": list(c.chain_edges or []),
            "facts": [_fact_payload(f) for f in cf],
        })
    return out


_TWO_FACT_SHARE = 1 / 3


def _select(chain_dicts: List[dict], n: int, rng: random.Random) -> List[dict]:
    """A deliberate mix of two-fact and single-fact chains.

    Two-fact chains are capped at a third of the panel. They are the
    interesting ones -- they exercise ``chain_edges`` and the role-shape hint --
    but single-fact chains are ~99% of what the pipeline actually emits, so a
    panel made entirely of two-fact chains would not represent the output being
    compared. The remainder is spread across fact roles so the sample is not
    entirely `definition` facts.
    """
    two = [c for c in chain_dicts if c["depth"] >= 2]
    single = [c for c in chain_dicts if c["depth"] == 1]
    rng.shuffle(two)
    picked = two[:int(n * _TWO_FACT_SHARE)]
    if len(picked) >= n:
        return picked[:n]

    # Round-robin over role so the panel is not entirely `definition` facts.
    by_role: Dict[str, List[dict]] = {}
    for c in single:
        by_role.setdefault(c["facts"][0]["role"], []).append(c)
    for bucket in by_role.values():
        rng.shuffle(bucket)
    roles = sorted(by_role)
    i = 0
    while len(picked) < n and any(by_role[r] for r in roles):
        r = roles[i % len(roles)]
        if by_role[r]:
            picked.append(by_role[r].pop())
        i += 1
    return picked[:n]


def build_chains() -> List[dict]:
    llm = _make_llm(ANCHOR_MODEL)
    rng = random.Random(SEED)
    selected: List[dict] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for doc_id, pdf_path, cache_dir in DOCS:
        pool = _build_doc_chains(doc_id, pdf_path, cache_dir, llm)
        # Persist the FULL pool per document. Stages 0-6 are the expensive part;
        # saving only the 15 selected chains would make any change to the
        # selection rule cost another full graph build.
        pool_path = OUT_DIR / f"pool_{doc_id}.json"
        pool_path.write_text(
            json.dumps(pool, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"[bakeoff] {doc_id} — {len(pool)} usable chains → {pool_path}")
        selected.extend(_select(pool, CHAINS_PER_DOC, rng))

    return _finalize(selected)


def _finalize(selected: List[dict]) -> List[dict]:
    for i, c in enumerate(selected):
        c["chain_id"] = f"c{i:02d}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHAINS_PATH.write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    n_two = sum(1 for c in selected if c["depth"] > 1)
    logger.info(
        f"[bakeoff] wrote {len(selected)} chains "
        f"({n_two} two-fact, {len(selected) - n_two} single-fact) → {CHAINS_PATH}"
    )
    return selected


def select_from_pools() -> List[dict]:
    """Re-run selection from the saved per-document pools. No LLM calls."""
    rng = random.Random(SEED)
    selected: List[dict] = []
    for doc_id, _, _ in DOCS:
        pool_path = OUT_DIR / f"pool_{doc_id}.json"
        if not pool_path.exists():
            raise SystemExit(f"{pool_path} missing — run --build-chains first")
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        logger.info(f"[bakeoff] {doc_id} — {len(pool)} pooled chains")
        selected.extend(_select(pool, CHAINS_PER_DOC, rng))
    return _finalize(selected)


# --------------------------------------------------------------------------
# Stage 7: generate per model
# --------------------------------------------------------------------------

def _question_stem(q: str) -> str:
    w = (q or "").strip().split()
    return w[0].strip('",.').capitalize() if w else ""


def _generate_one(chain: dict, llm) -> dict:
    """One chain through one model. Mirrors pipeline._run_qgen's control flow,
    but records gate verdicts instead of dropping the pair."""
    from rag_gt.generation.questions import generate_question
    from rag_gt.generation.answers import generate_answer, is_abstention
    from rag_gt.allpdf.grounding import is_grounded
    from rag_gt.allpdf.pipeline import (
        _is_corrupted_question,
        _needs_reframe,
        _reject_pair_reason,
    )

    facts = [_fact_from_payload(d) for d in chain["facts"]]
    edges = chain.get("chain_edges") or None
    cell: dict = {
        "chain_id": chain["chain_id"],
        "question": None, "answer": None,
        "qgen_sec": None, "agen_sec": None,
        "stem": None, "reject_reason": None,
        "is_abstention": False, "is_grounded": None,
        "reframed": False, "error": None,
    }

    try:
        t0 = time.time()
        question = generate_question(
            facts, llm, chain_edges=edges if len(facts) > 1 else None
        )
        cell["qgen_sec"] = round(time.time() - t0, 2)
        if not question:
            cell["error"] = "qgen returned empty"
            return cell

        if _needs_reframe(question):
            q2 = generate_question(
                facts, llm, chain_edges=edges if len(facts) > 1 else None,
                extra_hint=_CONCISE_HINT, temperature=0.4,
            )
            if q2 and not _needs_reframe(q2):
                question, cell["reframed"] = q2, True

        cell["question"] = question
        cell["stem"] = _question_stem(question)
        if _is_corrupted_question(question):
            cell["reject_reason"] = "corrupted_question"
            return cell

        t1 = time.time()
        answer = generate_answer(question, facts, llm)
        cell["agen_sec"] = round(time.time() - t1, 2)
        if not answer:
            cell["error"] = "answer returned empty"
            return cell
        cell["answer"] = answer

        if is_abstention(answer):
            cell["is_abstention"] = True
            cell["reject_reason"] = "abstention"
            return cell

        cell["is_grounded"] = bool(is_grounded(answer, facts))
        if not cell["is_grounded"]:
            cell["reject_reason"] = "ungrounded"
            return cell

        cell["reject_reason"] = _reject_pair_reason(question, answer, facts)
    except Exception as e:            # a model that cannot complete IS a result
        cell["error"] = f"{type(e).__name__}: {e}"
    return cell


def run_models(chains: List[dict]) -> dict:
    results: Dict[str, List[dict]] = {}
    timings: Dict[str, float] = {}
    usage_by_model: Dict[str, dict] = {}

    for model_id, label, tier in MODELS:
        logger.info(f"[bakeoff] === {label} ({model_id}) over {len(chains)} chains ===")
        llm = _make_llm(model_id)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            cells = list(ex.map(lambda c: _generate_one(c, llm), chains))
        timings[model_id] = round(time.time() - t0, 1)

        ok = sum(1 for c in cells if not c["reject_reason"] and not c["error"])
        errs = sum(1 for c in cells if c["error"])
        logger.info(
            f"[bakeoff] {label}: {ok}/{len(cells)} passed the gate, "
            f"{errs} errors, {timings[model_id]}s"
        )
        results[model_id] = cells

        # Real provider-reported token counts, not an estimate.
        usage = dict(getattr(llm, "usage_totals", {}) or {})
        logger.info(f"[bakeoff] {label} usage: {usage}")
        usage_by_model[model_id] = usage

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": [
            {"id": m, "label": l, "tier": t} for m, l, t in MODELS
        ],
        "chains": chains,
        "results": {m: results[m] for m, _, _ in MODELS},
        "usage": usage_by_model,
        "wall_sec": timings,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"[bakeoff] wrote results → {RESULTS_PATH}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-chains", action="store_true")
    ap.add_argument("--select-only", action="store_true",
                    help="re-run selection from saved pool_*.json (no LLM calls)")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    if args.build_chains:
        chains = build_chains()
    elif args.select_only:
        chains = select_from_pools()
    else:
        if not CHAINS_PATH.exists():
            logger.error(f"{CHAINS_PATH} missing — run with --build-chains first")
            return 1
        chains = json.loads(CHAINS_PATH.read_text(encoding="utf-8"))
        logger.info(f"[bakeoff] loaded {len(chains)} frozen chains")

    if args.run:
        run_models(chains)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
