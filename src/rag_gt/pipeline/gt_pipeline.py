"""V9.4 Ground-Truth pipeline orchestrator.

Changes vs V9 (see CHANGES_APPLIED.md for the full audit trail):
  - Per-doc work runs under narrow try/except; in-progress questions
    survive a doc-level failure via incremental persistence (`append_gt`).
  - Random seed uses MD5(doc_id) instead of built-in `hash()`, which is
    randomized per process by default.
  - Each MSFS gets a unique id derived from (doc_id, question index).
  - Per-depth drop schema is pre-initialized to {"kept": N, "minimality": N}
    so JSON consumers can rely on a stable shape.
  - Cross-doc question dedup increments `drops.duplicates`.
  - File glob is case-insensitive (`.pdf` and `.PDF`, `.docx` and `.DOCX`).
  - `as_completed` carries an explicit timeout to keep one hung LLM call
    from stalling the corpus.
  - `RunLogger` no longer touches the global `logging.basicConfig`; it
    attaches a fresh FileHandler per run, so back-to-back runs in the same
    process don't share a log file.
  - Doc-cache key includes a schema version tag.
  - The legacy random sampler now dedups in the same way as the new one.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import math
import os
import pickle
import random
import sys
import time
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from rag_gt.core.config import load_config

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)
logger.add(
    "data/logs/pipeline_{time:YYYYMMDD}.log",
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
)

# Bumped whenever the doc-cache pickle format changes. Bump this when you add
# fields to Fact/Document/Span — old pickles will be rejected automatically.
DOC_CACHE_SCHEMA = "v10.0-source-anchored"


class RunLogger:
    """Logs each generation run to a file. Each instance gets its own file
    and its own handler — back-to-back runs in the same process do not share
    state (the previous version's `logging.basicConfig` was a no-op after
    the first call)."""

    def __init__(self, log_dir: str = "data/logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"gt_run_{self.run_id}.log"

        self.logger = logging.getLogger(f"gt_pipeline.{self.run_id}")
        self.logger.setLevel(logging.DEBUG)
        # Remove any pre-existing handlers (re-use of the same logger name is unlikely
        # because we suffix with run_id, but be defensive).
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
        handler = logging.FileHandler(self.log_file, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        )
        self.logger.addHandler(handler)
        self.logger.propagate = False
        self.logger.info(f"=== RUN STARTED: {self.run_id} ===")

    def log_config(self, cfg: dict) -> None:
        self.logger.info("CONFIG:")
        for section, values in cfg.items():
            self.logger.info(f"  {section}: {values}")

    def log_document(self, doc_id: str, stats: dict) -> None:
        self.logger.info(f"DOC: {doc_id}")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")

    def log_question(
        self, doc_id: str, depth: int, cos: float, question: str, status: str = "OK"
    ) -> None:
        if status == "OK":
            self.logger.info(
                f"Q_OK | doc={doc_id} | depth={depth} | cos={cos:.2f} | {question[:70]}..."
            )
        else:
            self.logger.warning(f"Q_FAIL | doc={doc_id} | reason={question}")

    def log_summary(self, stats: dict) -> None:
        self.logger.info("=== SUMMARY ===")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=== RUN COMPLETE ===")

    def get_log_path(self) -> str:
        return str(self.log_file)


from rag_gt.core.llm import APIError, LLM, default_concurrency, get_llm
from rag_gt.core.types import MSFS, Fact, FactChain, QuestionGT
from rag_gt.validation.relation_support_gate import RelationVerdict, relation_support_gate
from rag_gt.chunking.strategies import chunk_document
from rag_gt.facts.domain_filter import fact_has_unresolved_deictic, filter_fact_domain
from rag_gt.facts.extraction import extract_candidate_facts
from rag_gt.generation.answers import ABSTENTION_TEXT, generate_answer
from rag_gt.generation.chain_quality import chain_quality_gate
from rag_gt.generation.multihop_sampler import enumerate_candidate_chains
from rag_gt.generation.questions import generate_question, premise_leakage_indices
from rag_gt.ingestion import ingest_document
from rag_gt.profiling.profiler import profile_document
from rag_gt.spans.normalization import find_fact_spans, tokenize_document
from rag_gt.storage.gt_io import append_gt, save_build_summary, save_gt
from rag_gt.validation.minimality import minimal_evidence_check, support_minimality_check
from rag_gt.validation.nli_check import batch_answer_entailment
from rag_gt.validation.structure import check_structure
from rag_gt.validation.gt_quality import score_generated_pair
from rag_gt.vectorstore.embedding import embed_facts, embed_query
from rag_gt.vectorstore.faiss_index import FactIndex
from rag_gt.observability.tracing import (
    PipelineTracer,
    chain_snapshot,
    chunk_sample,
    fact_snapshot,
    make_chain_id,
)


_DOC_CACHE_DIR = Path("data/cache/doc_cache")
_LLM_CALL_TIMEOUT = 300  # seconds; per-future cap for as_completed


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_key(path: str, doc_type: str, chunk_size: int, chunk_overlap: int) -> str:
    cfg = load_config()
    ingestion_cfg = cfg.get("ingestion", {})
    pdf_backend = str(ingestion_cfg.get("pdf_backend", "legacy")).lower()
    docling_export = str(ingestion_cfg.get("docling_export_format", "")).lower()
    docling_ocr = str(bool(ingestion_cfg.get("docling_do_ocr", False))).lower()
    docling_tables = str(
        bool(ingestion_cfg.get("docling_do_table_structure", False))
    ).lower()
    docling_batch = str(ingestion_cfg.get("docling_batch_size", ""))
    docling_page_range = str(ingestion_cfg.get("docling_page_range_size", ""))
    docling_min_chars = str(ingestion_cfg.get("docling_min_text_chars", ""))
    docling_ratio = str(ingestion_cfg.get("docling_min_text_file_ratio", ""))
    return (
        f"{_file_hash(path)}_{doc_type}_{chunk_size}_{chunk_overlap}_"
        f"{pdf_backend}_{docling_export}_{docling_ocr}_{docling_tables}_"
        f"{docling_batch}_{docling_page_range}_{docling_min_chars}_"
        f"{docling_ratio}_{DOC_CACHE_SCHEMA}"
    )


def _load_doc_cache(key: str):
    p = _DOC_CACHE_DIR / f"{key}.pkl"
    if p.exists():
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            p.unlink(missing_ok=True)
    return None


def _save_doc_cache(key: str, payload) -> None:
    _DOC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_DOC_CACHE_DIR / f"{key}.pkl", "wb") as f:
        pickle.dump(payload, f)


def _stable_doc_seed(doc_id: str) -> int:
    """Process-stable seed derived from doc_id. Built-in `hash()` is randomized
    per Python process, which silently broke reproducibility across runs."""
    digest = hashlib.md5(doc_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _normalize_question_for_dedup(q: str) -> str:
    text = unicodedata.normalize("NFKC", q or "").lower()
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


@dataclass
class _DropStats:
    fact_domain: int = 0
    provenance: int = 0
    chain_quality: int = 0
    questiongen: int = 0
    answer: int = 0
    structure: int = 0
    quality: int = 0
    nli: int = 0
    minimality: int = 0
    duplicates: int = 0
    qrsg: int = 0
    qa_nli: int = 0   # v16: QA-NLI gate rejects
    per_depth: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "failed_fact_domain": self.fact_domain,
            "failed_provenance": self.provenance,
            "failed_chain_quality": self.chain_quality,
            "failed_questiongen": self.questiongen,
            "failed_answergen": self.answer,
            "failed_structure": self.structure,
            "failed_quality": self.quality,
            "failed_nli": self.nli,
            "failed_minimality": self.minimality,
            "duplicates": self.duplicates,
            "failed_qrsg": self.qrsg,
            "failed_qa_nli": self.qa_nli,
            "per_depth": self.per_depth,
        }

    def init_depths(self, depths: Iterable[int]) -> None:
        """Pre-populate the per-depth schema so consumers see a stable shape."""
        for d in depths:
            self.per_depth.setdefault(int(d), {"kept": 0, "minimality": 0})

    def bump(self, depth: int, key: str) -> None:
        bucket = self.per_depth.setdefault(int(depth), {"kept": 0, "minimality": 0})
        bucket[key] = bucket.get(key, 0) + 1


def _facts_by_id(facts: List[Fact]) -> dict[str, Fact]:
    return {f.fact_id: f for f in facts}


def _chain_facts(chain: FactChain, by_id: dict[str, Fact]) -> List[Fact]:
    return [by_id[fid] for fid in chain.fact_ids if fid in by_id]


def _sem_distance(chain_facts: List[Fact]) -> str:
    chunk_ids = {
        span.chunk_id
        for f in chain_facts
        for span in f.supporting_spans
        if span.chunk_id
    }
    if len(chunk_ids) <= 1:
        return "local"
    return "intra_doc"


def _fact_primary_chunk_id(fact: Fact) -> str:
    for span in fact.supporting_spans:
        if span.chunk_id:
            return span.chunk_id
    return ""


def _distinct_chain_chunks(chain_facts: List[Fact]) -> set[str]:
    return {
        cid for cid in (_fact_primary_chunk_id(f) for f in chain_facts) if cid
    }


def _normalized_fact_text(fact: Fact) -> str:
    return " ".join((fact.text or "").lower().split())


def _distinct_chain_pages(chain_facts: List[Fact]) -> set[int]:
    pages: set[int] = set()
    for fact in chain_facts:
        for span in fact.supporting_spans:
            if span.page_start is not None:
                pages.add(int(span.page_start))
            if span.page_end is not None:
                pages.add(int(span.page_end))
    return pages


def _fact_page_count(facts: List[Fact]) -> int:
    pages: set[int] = set()
    for fact in facts:
        for span in fact.supporting_spans:
            if span.page_start is not None:
                pages.add(int(span.page_start))
            if span.page_end is not None:
                pages.add(int(span.page_end))
            for bbox in span.bboxes:
                page_no = getattr(bbox, "page_no", None)
                if page_no is not None:
                    pages.add(int(page_no))
    return max(1, len(pages))


def _chain_char_gap(chain_facts: List[Fact]) -> int:
    starts = [
        int(span.char_start)
        for fact in chain_facts
        for span in fact.supporting_spans
        if span.char_start is not None
    ]
    if len(starts) < 2:
        return 0
    return max(starts) - min(starts)


def _hop_type_for_chain(chain: FactChain) -> str:
    roles = [str(r) for r in chain.role_path if r]
    if not roles:
        return "single_fact" if chain.depth == 1 else "multi_fact"
    pairs = {
        ("definition", "rule"): "definition_plus_rule",
        ("definition", "condition"): "definition_plus_condition",
        ("condition", "consequence"): "condition_plus_consequence",
        ("rule", "exception"): "rule_plus_exception",
        ("rule", "constraint"): "rule_plus_constraint",
    }
    for key, label in pairs.items():
        if len(roles) >= 2 and tuple(roles[:2]) == key:
            return label
    return "_plus_".join(roles[:3])


def _filter_chains(
    chains: List[FactChain],
    by_id: dict[str, Fact],
    *,
    question_mode: str,
    min_hops: int,
    max_hops: int,
    min_distinct_chunks: int,
    min_distinct_roles: int,
    min_distinct_pages: int,
    min_char_gap: int,
    c3_enabled: bool = False,
    c3_score_threshold: float = 0.7,
    c3_reject_unknown: bool = False,
    c3_stats: Optional[dict] = None,
    chain_quality_enabled: bool = True,
    chain_max_page_gap: int = 40,
    chain_quality_stats: Optional[dict] = None,
    filter_stats: Optional[dict] = None,
) -> List[FactChain]:
    out: List[FactChain] = []

    def _reject(reason: str, chain: FactChain, cf: List[Fact] | None = None) -> None:
        if filter_stats is None:
            return
        filter_stats["rejected"] = filter_stats.get("rejected", 0) + 1
        by_reason = filter_stats.setdefault("by_reason", {})
        by_reason[reason] = by_reason.get(reason, 0) + 1
        examples = filter_stats.setdefault("examples", {})
        if reason not in examples:
            examples[reason] = {
                "chain_id": make_chain_id(list(chain.fact_ids)),
                "fact_ids": list(chain.fact_ids),
                "depth": chain.depth,
                "roles": [str(getattr(f, "role", "")) for f in (cf or [])],
            }

    if filter_stats is not None:
        filter_stats["input"] = len(chains)
    for chain in chains:
        if question_mode == "singlehop" and chain.depth != 1:
            _reject("question_mode_singlehop", chain)
            continue
        if question_mode == "multihop" and chain.depth < 2:
            _reject("question_mode_multihop", chain)
            continue
        if chain.depth < min_hops or chain.depth > max_hops:
            _reject("depth_out_of_range", chain)
            continue
        cf = _chain_facts(chain, by_id)
        texts = [_normalized_fact_text(f) for f in cf]
        if len(set(texts)) < len(texts):
            _reject("duplicate_fact_text", chain, cf)
            continue
        if len(_distinct_chain_chunks(cf)) < min_distinct_chunks:
            _reject("insufficient_distinct_chunks", chain, cf)
            continue
        if len({str(f.role) for f in cf if f.role}) < min_distinct_roles:
            _reject("insufficient_distinct_roles", chain, cf)
            continue
        pages = _distinct_chain_pages(cf)
        if pages and len(pages) < min_distinct_pages:
            _reject("insufficient_distinct_pages", chain, cf)
            continue
        if min_char_gap > 0 and _chain_char_gap(cf) < min_char_gap:
            _reject("insufficient_char_gap", chain, cf)
            continue
        if chain_quality_enabled:
            verdict = chain_quality_gate(cf, max_page_gap=chain_max_page_gap)
            if not verdict.accepted:
                if chain_quality_stats is not None:
                    chain_quality_stats[verdict.reason] = (
                        chain_quality_stats.get(verdict.reason, 0) + 1
                    )
                _reject(f"chain_quality:{verdict.reason}", chain, cf)
                continue
        if c3_enabled:
            # C3 self-containment gate: only hard-reject facts that are scored
            # (self_containment_known=True). Unscored facts pass unless
            # c3_reject_unknown=True (v14+).
            c3_drop = False
            c3_unknown_seen = False
            for f in cf:
                if f.self_containment_known and f.self_containment_score < c3_score_threshold:
                    if c3_stats is not None:
                        c3_stats["low_score_rejected"] = (
                            c3_stats.get("low_score_rejected", 0) + 1
                        )
                    c3_drop = True
                    _reject("c3_low_self_containment", chain, cf)
                    break
                if not f.self_containment_known and c3_reject_unknown:
                    if c3_stats is not None:
                        c3_stats["unknown_rejected"] = (
                            c3_stats.get("unknown_rejected", 0) + 1
                        )
                    c3_drop = True
                    _reject("c3_unknown_self_containment", chain, cf)
                    break
                if not f.self_containment_known:
                    c3_unknown_seen = True
            if c3_drop:
                continue
            if c3_unknown_seen and c3_stats is not None:
                c3_stats["unknown_warned"] = c3_stats.get("unknown_warned", 0) + 1
        # Reject chains where any fact opens with an unresolved deictic pronoun.
        # Such facts hard-fail Q7_bad_fact_fragment; skip them before expensive QGen/ARM.
        deictic_drop = False
        for f in cf:
            if fact_has_unresolved_deictic(f):
                _reject("deictic_opener_unresolved", chain, cf)
                deictic_drop = True
                break
        if deictic_drop:
            continue
        out.append(chain)
    if filter_stats is not None:
        filter_stats["output"] = len(out)
    return out


def _build_depth_distribution(n_questions: int, cfg: dict) -> dict[int, int]:
    dist = cfg.get("multihop", {}).get(
        "depth_distribution", {1: 0.35, 2: 0.45, 3: 0.20}
    )
    oversample = cfg.get("multihop", {}).get("n_chains_oversample", 3)
    out: dict[int, int] = {}
    for depth_key, frac in dist.items():
        d = int(depth_key)
        out[d] = max(1, int(round(n_questions * float(frac) * oversample)))
    return out


def _apply_depth_controls(
    depth_dist: dict[int, int],
    *,
    question_mode: str,
    min_hops: int,
    max_hops: int,
) -> dict[int, int]:
    out: dict[int, int] = {}
    for depth, count in depth_dist.items():
        if question_mode == "singlehop" and depth != 1:
            continue
        if question_mode == "multihop" and depth < 2:
            continue
        if depth < min_hops or depth > max_hops:
            continue
        out[depth] = count
    if not out:
        fallback_depth = 1 if question_mode == "singlehop" else max(2, min_hops)
        out[fallback_depth] = max(1, sum(depth_dist.values()) or 1)
    return out


def _target_question_count(n_questions: int, cfg: dict) -> int:
    """Requested questions plus configurable review buffer.

    A stricter quality gate can discard many candidates. Keeping the final
    target slightly above the user-requested count gives reviewers enough rows
    without weakening the gate.
    """
    multiplier = float(
        cfg.get("question_generation", {}).get("output_buffer_multiplier", 1.0)
    )
    return max(n_questions, int(math.ceil(n_questions * multiplier)))


def _candidate_submit_limit(target_questions: int, cfg: dict) -> int:
    qrsg_cfg = cfg.get("v13_qrsg", {})
    qrsg_enabled = bool(qrsg_cfg.get("enabled", False))
    if qrsg_enabled and qrsg_cfg.get("candidate_submit_multiplier") is not None:
        multiplier = int(qrsg_cfg.get("candidate_submit_multiplier", 4))
    else:
        multiplier = int(
            cfg.get("question_generation", {}).get("candidate_submit_multiplier", 3)
        )
    return max(target_questions, target_questions * max(1, multiplier))


def _deadline_from_minutes(start_time: float, max_wall_minutes: Optional[float]) -> Optional[float]:
    if not max_wall_minutes or max_wall_minutes <= 0:
        return None
    return start_time + (float(max_wall_minutes) * 60.0)


def _deadline_remaining_seconds(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return max(0.0, deadline - time.time())


def _deadline_expired(deadline: Optional[float]) -> bool:
    remaining = _deadline_remaining_seconds(deadline)
    return remaining is not None and remaining <= 0.0


def _tf_sfg_stage_deadline(
    deadline: Optional[float],
    now: float,
    *,
    max_wall_fraction: float = 0.35,
    reserve_generation_minutes: float = 20.0,
) -> Optional[float]:
    """Reserve run time for QGen when TF-SFG API calls are throttled."""
    if deadline is None:
        return None
    remaining = max(0.0, deadline - now)
    reserve_seconds = max(0.0, float(reserve_generation_minutes) * 60.0)
    fraction_budget = remaining * max(0.0, min(1.0, float(max_wall_fraction)))
    if remaining > reserve_seconds:
        fraction_budget = min(fraction_budget, remaining - reserve_seconds)
    return now + fraction_budget


def _is_semantic_duplicate_embedding(
    vector,
    seen_vectors: list,
    threshold: float,
) -> bool:
    if vector is None or not seen_vectors or threshold <= 0:
        return False
    return any(
        float(sum(float(a) * float(b) for a, b in zip(vector, old))) >= threshold
        for old in seen_vectors
    )


QUESTION_DOMAIN_ARTIFACT_RE = re.compile(
    r"\b("
    r"copyright|license|creative commons|free pdf|low[- ]cost print|"
    r"website|url|textbook equity|saylor|"
    r"author|authors|translated|translator|publisher|isbn|"
    r"study guide|examination study guides?|catalog|"
    r"about the authors|foreword|preface|table of contents"
    r")\b",
    re.I,
)


def _question_domain_reject_reason(question: str, answer: str) -> Optional[str]:
    text = f"{question or ''} {answer or ''}"
    if QUESTION_DOMAIN_ARTIFACT_RE.search(text):
        return "source_meta_question"
    if re.search(r"\baccording to the statement\b|\bprovided facts\b", question or "", re.I):
        return "source_framed_question"
    return None


def _provenance_reject_reason(facts: List[Fact]) -> Optional[str]:
    for fact in facts:
        if not fact.supporting_spans:
            return "missing_supporting_spans"
        for span in fact.supporting_spans:
            if not span.doc_id:
                return "missing_doc_id"
            if not span.chunk_id:
                return "missing_chunk_id"
            if span.char_start is None or span.char_end is None:
                return "missing_char_offsets"
            if span.page_start is None:
                return "missing_page"
            if not span.bboxes:
                return "missing_bbox"
            if not span.source_path:
                return "missing_source_path"
            if not span.source_sha1:
                return "missing_source_sha1"
    return None


def _fact_page_for_fallback(fact: Fact) -> int:
    for span in fact.supporting_spans or []:
        page = getattr(span, "page_start", None)
        if page is not None:
            return int(page)
    return 0


_DEICTIC_SUBJECT_RE = re.compile(
    r"^(this|that|these|those|it|they|their|such|here|there|both|some|many|"
    r"most|all|each|every|another|other|however|moreover|additionally|"
    r"furthermore|therefore|thus|hence|consequently|nevertheless|"
    r"for example|for instance|in addition|in particular|in contrast|"
    r"before we|we consider|we use|we define|we have|we can|we note|"
    r"as (noted|mentioned|described|discussed|shown|seen|stated|above|below))\b",
    re.I,
)
_MID_DEICTIC_RE = re.compile(
    r"\b(this kind of|this type of|this class of|this sort of|"
    r"this problem|this approach|this setting|this task|this case|"
    r"such problems|such methods|such tasks|such cases|such an agent)\b",
    re.I,
)


def _fallback_fact_score(fact: Fact) -> int:
    """Score 0-4: higher = more self-contained, better Q candidate.

    4 — Clean definition "X is/are/involves/specifies Y" (not "is like/similar to/also")
    3 — Article-led definition "A/An/The X is/are/involves Y" or contrast fact
    2 — Capital, not deictic, ends with period, other patterns
    1 — Capital, not deictic, ends poorly
    0 — Deictic subject, lowercase start, or fragment
    """
    text = (fact.text or "").strip()
    if not text:
        return 0
    first_alpha = next((c for c in text if c.isalpha()), None)
    if first_alpha is None or first_alpha.islower():
        return 0
    if _DEICTIC_SUBJECT_RE.match(text):
        return 0
    ends_cleanly = text.endswith(".")
    lead60 = text[:60].lower()
    # Exclude weak patterns: "X is like/similar to/also/part of" — comparison not definition
    is_weak_comparison = bool(re.search(r"\bis\s+(like|similar to|also\b|part of)", lead60))
    # Strong definition verbs: "X is/are/defines/involves/requires Y"
    is_definition = bool(re.match(
        r"^[A-Z][a-zA-Z].*\b(is|are|was|were|defines?|refers? to|involves?|"
        r"requires?|specifies?|consists?\s+of|represents?|enables?|allows?)\b",
        text,
    ))
    # Article-led: "A/An/The X is/are/involves..."
    is_article_led = bool(re.match(r"^(A|An|The)\s+[A-Za-z]", text))
    # Mid-sentence deictic ("this kind of problem") → can't ask self-contained Q from this alone
    has_mid_deictic = bool(_MID_DEICTIC_RE.search(text))
    if is_definition and not is_weak_comparison and not has_mid_deictic:
        return 4 if ends_cleanly else 3
    if is_article_led and ends_cleanly and not has_mid_deictic:
        return 3
    return 2 if ends_cleanly else 1


def _v16_singlehop_fallback_chains(
    facts: List[Fact],
    existing_chains: List[FactChain],
    *,
    limit: int,
    min_words: int,
    max_words: int,
    rng: random.Random,
) -> List[FactChain]:
    """Add bounded provenance-clean single-hop candidates when TF-SFG is sparse."""
    if limit <= 0:
        return []
    existing = {fid for chain in existing_chains for fid in chain.fact_ids}
    role_rank = {
        "definition": 0,
        "rule": 1,
        "condition": 2,
        "constraint": 3,
        "consequence": 4,
        "exception": 5,
        "example": 8,
    }
    candidates: List[Fact] = []
    for fact in facts:
        if fact.fact_id in existing:
            continue
        if _provenance_reject_reason([fact]):
            continue
        words = re.findall(r"\w+", fact.text or "")
        if len(words) < min_words or len(words) > max_words:
            continue
        if (fact.role or "") == "example":
            continue
        candidates.append(fact)

    rng.shuffle(candidates)
    candidates.sort(
        key=lambda f: (
            -_fallback_fact_score(f),   # higher score = selected first
            _fact_page_for_fallback(f),
            role_rank.get(str(f.role or ""), 6),
            len(f.text or ""),
        )
    )
    out: List[FactChain] = []
    used_pages: set[int] = set()

    for fact in candidates:
        page = _fact_page_for_fallback(fact)
        if page in used_pages and len(used_pages) < limit:
            continue
        out.append(
            FactChain(
                fact_ids=[fact.fact_id],
                anchor_id=fact.fact_id,
                mean_cosine=1.0,
                role_path=[fact.role],
            )
        )
        used_pages.add(page)
        if len(out) >= limit:
            return out

    for fact in candidates:
        if len(out) >= limit:
            break
        if any(fact.fact_id in chain.fact_ids for chain in out):
            continue
        out.append(
            FactChain(
                fact_ids=[fact.fact_id],
                anchor_id=fact.fact_id,
                mean_cosine=1.0,
                role_path=[fact.role],
            )
        )
    return out


def _append_qrsg_trace(
    path: Optional[Path],
    question: str,
    facts: List[Fact],
    verdict: RelationVerdict,
) -> None:
    """Append one JSONL trace row. Appends to `path` using open('a'); the caller
    is responsible for clearing the file at run start."""
    if path is None:
        return
    row = {
        "question": question,
        "fact_ids": [f.fact_id for f in facts],
        "facts": [
            {
                "fact_id": f.fact_id,
                "text": f.text,
                "canonical_form": f.canonical_form,
                "role": f.role,
                "self_containment_score": f.self_containment_score,
                "self_containment_known": f.self_containment_known,
            }
            for f in facts
        ],
        **verdict.to_dict(),
        "llm_raw": verdict.llm_raw,
    }
    try:
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.warning(f"[QRSG] trace write failed: {e}")


def _generate_and_answer(
    chain_facts: List[Fact],
    gt_llm: LLM,
    answer_llm: LLM,
    qrsg_llm: Optional[LLM] = None,
    qrsg_gate_version: str = "v1",
    qrsg_max_tokens: int = 2048,
    doc_id: str = "",
    cost_tracker: object = None,
) -> Tuple[Optional[str], Optional[RelationVerdict], Optional[str]]:
    if cost_tracker is not None:
        from rag_gt.observability.cost_tracker import TrackedLLM

        gt_llm = TrackedLLM(
            gt_llm, cost_tracker, stage="question_gen", doc_id=doc_id
        )
        answer_llm = TrackedLLM(
            answer_llm, cost_tracker, stage="answer_gen", doc_id=doc_id
        )
        if qrsg_llm is not None:
            qrsg_llm = TrackedLLM(
                qrsg_llm, cost_tracker, stage="qrsg", doc_id=doc_id
            )

    q = generate_question(chain_facts, gt_llm)
    if not q:
        return None, None, None

    verdict: Optional[RelationVerdict] = None
    if qrsg_llm is not None:
        verdict = relation_support_gate(
            q,
            chain_facts,
            qrsg_llm,
            gate_version=qrsg_gate_version,
            max_tokens=qrsg_max_tokens,
        )
        if not verdict.accepted:
            return q, verdict, None

    a = generate_answer(q, chain_facts, answer_llm)
    return q, verdict, a


def _generate_and_answer_v16(
    chain_facts: List[Fact],
    chain_edges: List[dict],
    gt_llm: LLM,
    answer_llm: LLM,
    v16_cfg: dict,
    qrsg_llm: Optional[LLM] = None,
    qrsg_gate_version: str = "v1",
    qrsg_max_tokens: int = 2048,
    intent: str = "",
    doc_id: str = "",
    cost_tracker: object = None,
    qa_nli_profile: Optional[dict] = None,
    question_generation_profile: Optional[dict] = None,
) -> Tuple[Optional[str], Optional[RelationVerdict], Optional[str], dict]:
    """v16 generation path: QA-NLI gate runs between question gen and answer gen.

    Returns (question, qrsg_verdict, answer, v16_extras) where v16_extras is a
    dict with keys 'question_assumptions' (List[dict]) and 'qa_nli_accepted' (bool).
    Returns (q, verdict, None, {}) on any gate failure to keep the caller's
    drop accounting simple.
    """
    from rag_gt.validation.question_assumption_gate import (
        check_question_assumptions,
        verdicts_to_dicts,
    )
    from rag_gt.observability.cost_tracker import LiveCallBudgetExceeded
    qgen_cfg = dict(v16_cfg.get("question_generation", {}) or {})
    if question_generation_profile:
        base_guard_cfg = qgen_cfg.get("premise_leakage_guard", {}) or {}
        profile_guard_cfg = (
            question_generation_profile.get("premise_leakage_guard", {}) or {}
        )
        qgen_cfg.update(question_generation_profile)
        qgen_cfg["premise_leakage_guard"] = {
            **base_guard_cfg,
            **profile_guard_cfg,
        }
    use_answer_llm_for_qgen = bool(qgen_cfg.get("use_answer_llm", False))
    fallback_qgen_on_api_error = bool(
        qgen_cfg.get("fallback_to_answer_llm_on_api_error", True)
    )
    primary_question_base_llm = answer_llm if use_answer_llm_for_qgen else gt_llm
    fallback_question_base_llm = gt_llm if use_answer_llm_for_qgen else answer_llm

    if cost_tracker is not None:
        from rag_gt.observability.cost_tracker import TrackedLLM

        question_llm = TrackedLLM(
            primary_question_base_llm,
            cost_tracker,
            stage="question_gen",
            doc_id=doc_id,
        )
        fallback_question_llm = TrackedLLM(
            fallback_question_base_llm,
            cost_tracker,
            stage="question_gen_fallback",
            doc_id=doc_id,
        )
        qa_llm = TrackedLLM(
            answer_llm, cost_tracker, stage="qa_nli", doc_id=doc_id
        )
        answer_stage_llm = TrackedLLM(
            answer_llm, cost_tracker, stage="answer_gen", doc_id=doc_id
        )
        qrsg_stage_llm = (
            TrackedLLM(qrsg_llm, cost_tracker, stage="qrsg", doc_id=doc_id)
            if qrsg_llm is not None
            else None
        )
    else:
        question_llm = primary_question_base_llm
        fallback_question_llm = fallback_question_base_llm
        qa_llm = answer_llm
        answer_stage_llm = answer_llm
        qrsg_stage_llm = qrsg_llm

    qa_nli_cfg = v16_cfg.get("qa_nli", {})
    max_regen = int(qa_nli_cfg.get("max_regen_attempts", 3))
    threshold = float(qa_nli_cfg.get("threshold", 0.55))
    max_rel_fail_rate = float(qa_nli_cfg.get("max_rel_fail_rate", 0.10))
    max_total_fail_rate = float(qa_nli_cfg.get("max_total_fail_rate", 0.25))
    fail_open = bool(qa_nli_cfg.get("fail_open_on_decomposition_error", True))
    enforce_edge_relation_match = bool(
        qa_nli_cfg.get("enforce_edge_relation_match", False)
    )
    max_rel_mismatch_rate = float(qa_nli_cfg.get("max_rel_mismatch_rate", 0.5))
    require_per_fact_coverage = bool(
        qa_nli_cfg.get("require_per_fact_coverage", False)
    )
    try:
        q = generate_question(
            chain_facts,
            question_llm,
            intent=intent,
            chain_edges=chain_edges,
        )
    except LiveCallBudgetExceeded:
        raise
    except APIError:
        if not fallback_qgen_on_api_error:
            raise
        logger.warning("[QGen] primary question LLM failed; retrying with fallback LLM")
        q = generate_question(
            chain_facts,
            fallback_question_llm,
            intent=intent,
            chain_edges=chain_edges,
        )
    if not q:
        return None, None, None, {}

    premise_guard_cfg = qgen_cfg.get("premise_leakage_guard", {}) or {}
    premise_guard_enabled = bool(premise_guard_cfg.get("enabled", False))
    premise_guard_reject_after_regen = bool(
        premise_guard_cfg.get("reject_after_regen", True)
    )
    premise_guard_max_regen = int(premise_guard_cfg.get("max_regen_attempts", 2))
    premise_guard_params = {
        "min_common_run": int(premise_guard_cfg.get("min_common_run", 5)),
        "min_overlap_tokens": int(premise_guard_cfg.get("min_overlap_tokens", 7)),
        "min_overlap_ratio": float(premise_guard_cfg.get("min_overlap_ratio", 0.55)),
    }
    leaked_indices: list[int] = []
    if premise_guard_enabled:
        leaked_indices = premise_leakage_indices(q, chain_facts, **premise_guard_params)
        leak_regen_attempt = 0
        while leaked_indices and leak_regen_attempt < premise_guard_max_regen:
            leak_regen_attempt += 1
            guidance = (
                "\n\nGuidance: The previous question restated too much of "
                f"Support indices {leaked_indices} as a premise. Rewrite it "
                "using only compact concept anchors. Do not copy long phrases "
                "from any support into the question. The answer itself must "
                "need every support."
            )
            try:
                q_regen = generate_question(
                    chain_facts,
                    question_llm,
                    extra_hint=guidance,
                    intent=intent,
                    chain_edges=chain_edges,
                    temperature=0.4,
                )
            except LiveCallBudgetExceeded:
                raise
            except APIError:
                if not fallback_qgen_on_api_error:
                    raise
                logger.warning("[QGen] primary premise-leak regen failed; retrying with fallback LLM")
                q_regen = generate_question(
                    chain_facts,
                    fallback_question_llm,
                    extra_hint=guidance,
                    intent=intent,
                    chain_edges=chain_edges,
                    temperature=0.4,
                )
            if not q_regen or q_regen == q:
                break
            q = q_regen
            leaked_indices = premise_leakage_indices(q, chain_facts, **premise_guard_params)

        if leaked_indices and premise_guard_reject_after_regen:
            return q, None, None, {
                "question_assumptions": [],
                "qa_nli_accepted": False,
                "qa_nli_fail_rate": 0.0,
                "qa_nli_reason": "premise_leakage:" + ",".join(str(i) for i in leaked_indices),
                "qa_nli_missing_fact_indices": list(leaked_indices),
                "intent": intent,
            }

    # QA-NLI gate: runs before QRSG and answer LLM to save cost on vacuous chains.
    qa_verdict = check_question_assumptions(
        q,
        chain_facts,
        qa_llm,  # same model as QRSG for JSON stability
        chain_edges=chain_edges,
        threshold=threshold,
        max_rel_fail_rate=max_rel_fail_rate,
        max_total_fail_rate=max_total_fail_rate,
        fail_open_on_decomposition_error=fail_open,
        enforce_edge_relation_match=enforce_edge_relation_match,
        max_rel_mismatch_rate=max_rel_mismatch_rate,
        qa_nli_profile=qa_nli_profile,
        require_per_fact_coverage=require_per_fact_coverage,
    )

    regen_attempt = 0
    while not qa_verdict.accepted and regen_attempt < max_regen and qa_verdict.guidance:
        regen_attempt += 1
        logger.debug(
            f"[QA-NLI] regen attempt {regen_attempt}/{max_regen}: {qa_verdict.reason}"
        )
        guided_prompt_suffix = f"\n\nGuidance: {qa_verdict.guidance}"
        try:
            q_regen = generate_question(
                chain_facts,
                question_llm,
                extra_hint=guided_prompt_suffix,
                intent=intent,
                chain_edges=chain_edges,
            )
        except LiveCallBudgetExceeded:
            raise
        except APIError:
            if not fallback_qgen_on_api_error:
                raise
            logger.warning("[QGen] primary regen LLM failed; retrying with fallback LLM")
            q_regen = generate_question(
                chain_facts,
                fallback_question_llm,
                extra_hint=guided_prompt_suffix,
                intent=intent,
                chain_edges=chain_edges,
            )
        if not q_regen or q_regen == q:
            break
        q = q_regen
        qa_verdict = check_question_assumptions(
            q,
            chain_facts,
            qa_llm,
            chain_edges=chain_edges,
            threshold=threshold,
            max_rel_fail_rate=max_rel_fail_rate,
            max_total_fail_rate=max_total_fail_rate,
            fail_open_on_decomposition_error=fail_open,
            enforce_edge_relation_match=enforce_edge_relation_match,
            max_rel_mismatch_rate=max_rel_mismatch_rate,
            qa_nli_profile=qa_nli_profile,
            require_per_fact_coverage=require_per_fact_coverage,
        )

    v16_extras: dict = {
        "question_assumptions": verdicts_to_dicts(qa_verdict.assumption_verdicts),
        "qa_nli_accepted": qa_verdict.accepted,
        "qa_nli_fail_rate": qa_verdict.rel_fail_rate,
        "qa_nli_reason": qa_verdict.reason,
        "qa_nli_missing_fact_indices": list(getattr(qa_verdict, "missing_fact_indices", [])),
        "premise_leakage_indices": (
            premise_leakage_indices(q, chain_facts, **premise_guard_params)
            if premise_guard_enabled
            else []
        ),
        "premise_leakage_advisory_only": (
            premise_guard_enabled and not premise_guard_reject_after_regen
        ),
        "intent": intent,
    }

    if not qa_verdict.accepted:
        return q, None, None, v16_extras

    # QRSG gate (unchanged from v15).
    verdict: Optional[RelationVerdict] = None
    if qrsg_stage_llm is not None:
        verdict = relation_support_gate(
            q,
            chain_facts,
            qrsg_stage_llm,
            gate_version=qrsg_gate_version,
            max_tokens=qrsg_max_tokens,
        )
        if not verdict.accepted:
            return q, verdict, None, v16_extras

    arm_cfg = v16_cfg.get("arm", {})
    if bool(arm_cfg.get("enabled", False)):
        from rag_gt.generation.cga import build_constructive_gold_answer

        try:
            cga = build_constructive_gold_answer(
                q,
                chain_facts,
                answer_llm,
                threshold=float(arm_cfg.get("step_nli_threshold", 0.55)),
                max_retries=int(arm_cfg.get("max_step_retries", 3)),
                max_np_overlap=float(arm_cfg.get("max_np_overlap", 0.35)),
                use_arm=True,
                tf_sfg_edges=chain_edges,
                cost_tracker=cost_tracker,
                doc_id=doc_id,
            )
        except LiveCallBudgetExceeded:
            raise
        except Exception as e:
            logger.debug(f"[ARM] failed for q={q[:80]!r}: {type(e).__name__}: {e}")
            v16_extras["answer_failure_reason"] = "arm_exception"
            return q, verdict, None, v16_extras
        if not cga.all_clauses_pass:
            v16_extras["arm_all_clauses_pass"] = False
            v16_extras["answer_failure_reason"] = "arm_grounding_failed"
            v16_extras["reasoning_trace"] = list(cga.reasoning_trace)
            return q, verdict, None, v16_extras
        a = cga.gold_answer
        v16_extras["reasoning_trace"] = list(cga.reasoning_trace)
        v16_extras["arm_all_clauses_pass"] = True
    else:
        a = generate_answer(q, chain_facts, answer_stage_llm)
    return q, verdict, a, v16_extras


def _list_input_paths(input_dir: str, enable_docx: bool) -> List[str]:
    """Case-insensitive PDF/DOCX listing. `glob` is case-sensitive on Linux/macOS,
    so a `Standard.PDF` file dropped into the input dir was previously invisible."""
    base = Path(input_dir)
    if not base.exists():
        return []
    suffixes = {".pdf"}
    if enable_docx:
        suffixes.add(".docx")
    out: List[str] = []
    for p in sorted(base.iterdir()):
        if p.is_file() and p.suffix.lower() in suffixes:
            out.append(str(p))
    return out


def run_gt_pipeline(
    input_dir: str,
    output_path: str,
    doc_type: str = "UNKNOWN",
    n_questions: int = 8,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    fast_mode: bool = False,
    max_concurrent_llm_calls: Optional[int] = None,
    disable_folder_heuristic: bool = False,
    question_mode: str = "mixed",
    min_hops: int = 1,
    max_hops: int = 3,
    min_distinct_chunks: int = 1,
    min_distinct_roles: int = 1,
    min_distinct_pages: int = 1,
    min_char_gap: int = 0,
    max_wall_minutes: Optional[float] = None,
    progress_every: Optional[int] = None,
    pair_budget: Optional[int] = None,
    max_live_api_calls: Optional[int] = None,
    disable_v16_singlehop_fallback: bool = False,
    disable_v16_twins: bool = False,
    v16: bool = False,
    v16_2: Optional[bool] = None,
    trace_path: Optional[str] = None,
    enable_trace: bool = True,
) -> None:
    cfg = load_config()
    t0 = time.time()
    # v16 PASS-GT flag: CLI flag takes precedence over config.
    v16_top_cfg = cfg.get("v16", {})
    v16_enabled = v16 or (v16_2 is True) or bool(v16_top_cfg.get("enabled", False)) or (
        os.getenv("RAG_GT_V16", "").lower() in ("1", "true", "yes")
    )
    v16_cfg: dict = {}
    if v16_enabled:
        v16_cfg_path = v16_top_cfg.get("config_path", "configs/v16.yaml")
        try:
            import yaml
            with open(v16_cfg_path, "r", encoding="utf-8") as _f:
                v16_cfg = yaml.safe_load(_f) or {}
        except Exception as _e:
            logger.warning(f"[v16] Could not load v16 config from {v16_cfg_path!r}: {_e}; using defaults")
        if pair_budget is not None:
            _pair_budget = max(1, int(pair_budget))
            v16_cfg.setdefault("tf_sfg", {})["max_candidate_pairs"] = _pair_budget
            logger.info(f"[v16] TF-SFG pair budget override: {_pair_budget}")
        if disable_v16_singlehop_fallback:
            v16_cfg.setdefault("fallback", {})["enabled"] = False
            logger.info("[v16] Single-hop fallback disabled for this run")
        if disable_v16_twins:
            v16_cfg.setdefault("twins", {})["enabled"] = False
            logger.info("[v16] Twin derivation disabled for this run")
        if v16_2 is not None:
            v16_cfg.setdefault("v16_2", {})["enabled"] = bool(v16_2)
        logger.info(f"[v16] PASS-GT mode enabled — config: {v16_cfg_path}")
    v16_2_cfg = v16_cfg.get("v16_2", {}) or {}
    v16_2_enabled = v16_enabled and bool(v16_2_cfg.get("enabled", False))
    if v16_2_enabled:
        logger.info("[v16.2] Cost-adaptive cascade enabled")
    cost_tracker = None
    cost_tracker_cfg = v16_2_cfg.get("cost_tracker", {}) or {}
    if v16_enabled and (
        bool(cost_tracker_cfg.get("enabled", False))
        or max_live_api_calls is not None
    ):
        from rag_gt.observability.cost_tracker import CostTracker

        cost_tracker = CostTracker(
            chars_per_token=float(
                cost_tracker_cfg.get("token_estimate_chars_per_token", 3.5)
            ),
            max_live_api_calls=max_live_api_calls,
        )
    qgen_cfg = cfg.get("question_generation", {})
    concurrency = max_concurrent_llm_calls or int(
        cfg.get("performance", {}).get("max_concurrent_llm_calls")
        or default_concurrency()
        or 4
    )
    min_cosine = float(cfg.get("multihop", {}).get("min_cosine", 0.55))
    role_bias = bool(cfg.get("multihop", {}).get("role_bias", True))
    use_random_fallback = bool(cfg.get("legacy", {}).get("use_random_sampler", False))
    enable_nli = bool(cfg.get("validation", {}).get("enable_nli_checks", True))
    enable_min = bool(cfg.get("validation", {}).get("enable_minimality_check", True))
    min_gt_quality = float(cfg.get("validation", {}).get("min_gt_quality", 0.7))
    enable_docx = bool(cfg.get("ingestion", {}).get("enable_docx", False))
    fact_domain_enabled = bool(
        cfg.get("fact_domain_filter", {}).get("enabled", True)
    )
    chain_quality_cfg = cfg.get("chain_quality", {})
    chain_quality_enabled = bool(chain_quality_cfg.get("enabled", True))
    chain_max_page_gap = int(chain_quality_cfg.get("max_page_gap", 40) or 40)
    semantic_duplicate_threshold = float(
        qgen_cfg.get("semantic_duplicate_threshold", 0.0) or 0.0
    )
    v16_qgen_cfg = v16_cfg.get("question_generation", {}) if v16_enabled else {}
    qgen_api_error_fail_fast_after = int(
        v16_qgen_cfg.get(
            "api_error_fail_fast_after",
            qgen_cfg.get("api_error_fail_fast_after", 0),
        )
        or 0
    )
    if progress_every is None:
        progress_every = int(qgen_cfg.get("progress_every", 0) or 0)
    if max_wall_minutes is None:
        max_wall_minutes = float(qgen_cfg.get("max_wall_minutes", 0) or 0)
    deadline = _deadline_from_minutes(t0, max_wall_minutes)
    target_questions = _target_question_count(n_questions, cfg)
    candidate_limit = _candidate_submit_limit(target_questions, cfg)
    qrsg_cfg = cfg.get("v13_qrsg", {})
    qrsg_enabled = bool(qrsg_cfg.get("enabled", False))
    qrsg_gate_version = str(qrsg_cfg.get("gate_version", "v1"))
    qrsg_llm_role = str(qrsg_cfg.get("llm_role", "gt") or "gt")
    qrsg_max_tokens = int(qrsg_cfg.get("max_tokens", 2048) or 2048)
    c3_score_threshold = float(qrsg_cfg.get("c3_score_threshold", 0.7))
    c3_reject_unknown = bool(qrsg_cfg.get("reject_unknown_self_containment", False))
    corpus_name = os.path.splitext(os.path.basename(output_path))[0]
    out_dir = Path(output_path).parent
    resolved_trace_path = (
        Path(trace_path)
        if trace_path
        else out_dir / f"{corpus_name}.trace.jsonl"
    )

    run_logger = RunLogger()
    tracer = PipelineTracer(resolved_trace_path, enabled=enable_trace)
    tracer.emit(
        "run_setup",
        "run_start",
        data={
            "input_dir": input_dir,
            "output_path": output_path,
            "trace_path": str(resolved_trace_path),
            "pipeline_mode": {
                "v16_enabled": v16_enabled,
                "v16_2_enabled": v16_2_enabled,
                "fast_mode": fast_mode,
            },
        },
        thresholds={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "min_cosine": min_cosine,
            "min_gt_quality": min_gt_quality,
            "target_questions_with_buffer": target_questions,
            "candidate_submit_limit": candidate_limit,
            "semantic_duplicate_threshold": semantic_duplicate_threshold,
            "qrsg_enabled": qrsg_enabled,
            "c3_score_threshold": c3_score_threshold,
        },
    )
    run_logger.log_config(
        {
            "input_dir": input_dir,
            "output_path": output_path,
            "doc_type": doc_type,
            "n_questions": n_questions,
            "chunk_size": chunk_size,
            "fast_mode": fast_mode,
            "min_cosine": min_cosine,
            "min_gt_quality": min_gt_quality,
            "target_questions_with_buffer": target_questions,
            "candidate_submit_limit": candidate_limit,
            "question_mode": question_mode,
            "min_hops": min_hops,
            "max_hops": max_hops,
            "min_distinct_chunks": min_distinct_chunks,
            "min_distinct_roles": min_distinct_roles,
            "min_distinct_pages": min_distinct_pages,
            "min_char_gap": min_char_gap,
            "fact_domain_filter": fact_domain_enabled,
            "chain_quality_enabled": chain_quality_enabled,
            "chain_max_page_gap": chain_max_page_gap,
            "semantic_duplicate_threshold": semantic_duplicate_threshold,
            "max_wall_minutes": max_wall_minutes,
            "max_live_api_calls": max_live_api_calls,
            "disable_v16_singlehop_fallback": disable_v16_singlehop_fallback,
            "disable_v16_twins": disable_v16_twins,
            "progress_every": progress_every,
        }
    )

    print("\n" + "=" * 60)
    print("  RAG GT Pipeline V10 -- Source-Anchored Ground Truth Generation")
    print("=" * 60)
    print(f"  Input dir  : {input_dir}")
    print(f"  Output     : {output_path}")
    print(f"  Doc type   : {doc_type}")
    print(f"  N questions: {n_questions} requested per document")
    print(f"  Target     : {target_questions} with review buffer")
    print(f"  Concurrency: {concurrency} parallel LLM calls")
    print(
        f"  Fast mode  : {'YES (minimality + NLI skipped)' if fast_mode else 'NO (full validation)'}"
    )
    print("=" * 60 + "\n")

    logger.info("[Stage] Initialising LLM...")
    with tracer.stage("run_setup", data={"operation": "initialise_llms"}):
        gt_llm = get_llm("gt")
        answer_llm = get_llm("answer")
    logger.info("[Stage] LLMs ready")
    qrsg_llm = get_llm(qrsg_llm_role) if qrsg_enabled else None
    if qrsg_enabled:
        logger.info(
            "[QRSG] Gate enabled — gate_version=%s role=%s max_tokens=%s cache=%s",
            qrsg_gate_version,
            qrsg_llm_role,
            qrsg_max_tokens,
            qrsg_cfg.get("cache_path", "default"),
        )

    paths = _list_input_paths(input_dir, enable_docx=enable_docx)
    if not paths:
        suffix_label = "PDF or DOCX" if enable_docx else "PDF"
        raise FileNotFoundError(f"No {suffix_label} files found in {input_dir}")
    tracer.emit(
        "run_setup",
        "input_paths_discovered",
        counts={"documents": len(paths)},
        data={"paths": paths, "enable_docx": enable_docx},
    )
    logger.info(f"[Stage] Loading documents... found {len(paths)} file(s)")
    print(f"  Found {len(paths)} document(s): {[os.path.basename(p) for p in paths]}\n")

    # Prepare incremental output sink. We append per-doc and run a final
    # consolidating save_gt() that atomically replaces the file.
    incremental_path = out_dir / f"{corpus_name}.partial.jsonl"
    if incremental_path.exists():
        try:
            incremental_path.unlink()
        except OSError:
            pass

    # QRSG trace file setup (Task 12).
    qrsg_trace_path: Optional[Path] = None
    qrsg_rejected_path: Optional[Path] = None
    qrsg_stats_path: Optional[Path] = None
    qrsg_total = 0
    qrsg_accepted_count = 0
    qrsg_reject_reasons: dict = {}
    qrsg_relation_type_pre: dict = {}
    qrsg_relation_type_post: dict = {}
    qrsg_by_depth: dict = {}
    qrsg_concept_coverage_sum = 0.0
    qrsg_concept_coverage_count = 0
    qrsg_risky_frame_rejects = 0
    c3_stats = {
        "low_score_rejected": 0,
        "unknown_warned": 0,
        "unknown_rejected": 0,
    }
    if qrsg_enabled:
        out_dir.mkdir(parents=True, exist_ok=True)
        qrsg_trace_path = out_dir / f"{corpus_name}.qrsg_trace.jsonl"
        qrsg_rejected_path = out_dir / f"{corpus_name}.qrsg_rejected.jsonl"
        qrsg_stats_path = out_dir / f"{corpus_name}.qrsg_stats.json"
        qrsg_trace_path.write_text("", encoding="utf-8")
        qrsg_rejected_path.write_text("", encoding="utf-8")
        logger.info("[QRSG] trace=%s rejected=%s", qrsg_trace_path, qrsg_rejected_path)

    all_questions: List[QuestionGT] = []
    seen_question_norms: set[str] = set()
    seen_question_embeddings: list = []
    total_facts_extracted = 0
    fact_domain_drop_reasons: dict[str, int] = {}
    chain_quality_drop_reasons: dict[str, int] = {}
    budget_per_doc: dict[str, dict] = {}
    cascade_stats: dict[str, dict] = {}
    drops = _DropStats()
    initial_depth_dist = _apply_depth_controls(
        _build_depth_distribution(target_questions, cfg),
        question_mode=question_mode,
        min_hops=min_hops,
        max_hops=max_hops,
    )
    drops.init_depths(initial_depth_dist.keys())
    distractor_id_offset = 0  # v16: global counter for unique distractor IDs across docs
    intent_counts: dict[str, int] = {}
    intent_target_dist = dict(v16_cfg.get("intent_distribution", {}) or {})
    for path in tqdm(paths, desc="Documents", unit="doc"):
        if _deadline_expired(deadline):
            logger.warning("[WallClock] max wall time reached before next document")
            break
        doc_name = os.path.basename(path)
        tracer.emit(
            "run_setup",
            "document_start",
            doc_id=os.path.splitext(doc_name)[0],
            data={"path": path, "filename": doc_name},
        )
        # Note: per-stage try/except inside; the outer guard only catches the
        # very-coarse case (e.g., file-system read failure). Per-stage failures
        # do NOT discard all in-progress questions for the doc.
        try:
            cache_key = _cache_key(path, doc_type, chunk_size, chunk_overlap)
            cached = _load_doc_cache(cache_key)
            if cached is not None:
                doc, profile, facts = cached
                tracer.emit(
                    "run_setup",
                    "doc_cache_hit",
                    doc_id=doc.doc_id,
                    counts={"facts": len(facts)},
                    data={"cache_key": cache_key, "profile": profile},
                )
                print(
                    f"  [{doc.doc_id}] (cached) type={profile['doc_type']} | {len(facts)} facts"
                )
                logger.info(f"  {doc.doc_id}: loaded from cache ({len(facts)} facts)")
            else:
                tracer.emit(
                    "run_setup",
                    "doc_cache_miss",
                    doc_id=os.path.splitext(doc_name)[0],
                    data={"cache_key": cache_key},
                )
                logger.info(f"[Stage] Ingesting: {doc_name}")
                with tracer.stage("ingestion", doc_id=os.path.splitext(doc_name)[0]):
                    doc = ingest_document(path, doc_type=doc_type)
                tracer.emit(
                    "ingestion",
                    "document_ingested",
                    doc_id=doc.doc_id,
                    counts={
                        "chars": len(doc.text),
                        "source_units": len(doc.source_units),
                    },
                    data={
                        "source_backend": doc.source_backend,
                        "source_path": doc.source_path,
                        "source_sha1": doc.source_sha1,
                    },
                )
                logger.info(f"  {doc.doc_id}: {len(doc.text)} chars ingested")

                logger.info(f"[Stage] Profiling: {doc.doc_id}")
                with tracer.stage("profiling", doc_id=doc.doc_id):
                    profile = profile_document(
                        doc,
                        path=path,
                        enable_fast_path=not disable_folder_heuristic,
                    )
                tracer.emit(
                    "profiling",
                    "document_profiled",
                    doc_id=doc.doc_id,
                    counts={"char_count": profile.get("char_count")},
                    data=profile,
                )
                logger.info(
                    f"  {doc.doc_id}: classified as {profile['doc_type']} "
                    f"(signals: {profile['signal_scores']}, fast_path={profile['fast_path_hit']})"
                )
                print(
                    f"  [{doc.doc_id}] type={profile['doc_type']} | {profile['char_count']} chars"
                )

                logger.info(f"[Stage] Chunking: {doc.doc_id}")
                with tracer.stage("chunking", doc_id=doc.doc_id):
                    chunks = chunk_document(doc, profile, chunk_size, chunk_overlap)
                tracer.emit(
                    "chunking",
                    "chunks_created",
                    doc_id=doc.doc_id,
                    counts={"chunks": len(chunks)},
                    thresholds={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
                    data={"sample": chunk_sample(chunks)},
                )
                logger.info(f"  {doc.doc_id}: {len(chunks)} chunk(s) produced")

                logger.info(f"[Stage] Extracting facts: {doc.doc_id}")
                with tracer.stage("fact_extraction", doc_id=doc.doc_id):
                    facts = extract_candidate_facts(doc, chunks, llm=None)
                    candidate_fact_count = len(facts)
                    doc_toks = tokenize_document(doc)
                    facts = find_fact_spans(doc, doc_toks, facts, chunks)
                    mapped_fact_count = sum(1 for f in facts if f.supporting_spans)
                    facts = [f for f in facts if f.supporting_spans]
                tracer.emit(
                    "fact_extraction",
                    "facts_extracted",
                    doc_id=doc.doc_id,
                    counts={
                        "candidate_facts": candidate_fact_count,
                        "mapped_facts": mapped_fact_count,
                        "dropped_without_spans": candidate_fact_count - mapped_fact_count,
                    },
                    thresholds={
                        "min_fact_length": cfg.get("fact_extraction", {}).get("min_fact_length"),
                        "max_fact_length": cfg.get("fact_extraction", {}).get("max_fact_length"),
                        "quality_threshold": cfg.get("fact_extraction", {}).get("quality_threshold"),
                        "span_fuzzy_threshold": cfg.get("span_normalization", {}).get("fuzzy_threshold"),
                        "iou_tau_recall": cfg.get("span_normalization", {}).get("iou_tau_recall"),
                        "iou_tau_precision": cfg.get("span_normalization", {}).get("iou_tau_precision"),
                    },
                    data={"sample": [fact_snapshot(f) for f in facts[:8]]},
                )
                if candidate_fact_count > mapped_fact_count:
                    tracer.drop(
                        "fact_extraction",
                        "missing_supporting_span",
                        doc_id=doc.doc_id,
                        counts={"dropped": candidate_fact_count - mapped_fact_count},
                    )
                logger.info(
                    f"  {doc.doc_id}: {len(facts)} facts extracted (with spans)"
                )
                print(f"  [{doc.doc_id}] {len(facts)} facts extracted")
                _save_doc_cache(cache_key, (doc, profile, facts))
                logger.info(f"  {doc.doc_id}: saved to doc cache")

            raw_fact_count = len(facts)
            if fact_domain_enabled and facts:
                with tracer.stage("fact_domain_filter", doc_id=doc.doc_id):
                    facts, domain_drops = filter_fact_domain(facts)
                for reason, count in domain_drops.items():
                    fact_domain_drop_reasons[reason] = (
                        fact_domain_drop_reasons.get(reason, 0) + count
                    )
                dropped_count = raw_fact_count - len(facts)
                drops.fact_domain += dropped_count
                tracer.emit(
                    "fact_domain_filter",
                    "facts_filtered",
                    doc_id=doc.doc_id,
                    counts={
                        "input": raw_fact_count,
                        "output": len(facts),
                        "dropped": dropped_count,
                    },
                    data={"drop_by_reason": domain_drops},
                )
                for reason, count in domain_drops.items():
                    tracer.drop(
                        "fact_domain_filter",
                        reason,
                        doc_id=doc.doc_id,
                        counts={"dropped": count},
                    )
                if dropped_count:
                    logger.info(
                        f"  {doc.doc_id}: fact-domain filter dropped "
                        f"{dropped_count}/{raw_fact_count} facts: {domain_drops}"
                    )
                    print(
                        f"  [{doc.doc_id}] fact-domain filter dropped "
                        f"{dropped_count}/{raw_fact_count}"
                    )
            total_facts_extracted += len(facts)

            if not facts:
                logger.warning(f"  {doc.doc_id}: 0 facts -- skipping document")
                tracer.drop(
                    "fact_domain_filter",
                    "no_facts_after_filtering",
                    doc_id=doc.doc_id,
                )
                continue

            doc_budget = None
            if v16_2_enabled:
                from rag_gt.budget import AdaptiveBudgetConfig, compute_doc_budget
                from rag_gt.pipeline.yield_controller import YieldControllerConfig

                adaptive_cfg = AdaptiveBudgetConfig.from_dict(
                    v16_2_cfg.get("adaptive_budget")
                )
                yield_cfg = YieldControllerConfig.from_dict(
                    v16_2_cfg.get("yield_controller")
                )
                doc_budget = compute_doc_budget(
                    fact_count=len(facts),
                    page_count=_fact_page_count(facts),
                    doc_type=str(profile.get("doc_type", "UNKNOWN")),
                    cfg=adaptive_cfg,
                    mh_floor_fraction=yield_cfg.mh_floor_fraction,
                    untyped_mh_cap_fraction=yield_cfg.untyped_mh_cap_fraction,
                    fb_cap_fraction=yield_cfg.fb_cap_fraction,
                )
                if pair_budget is not None:
                    doc_budget = replace(
                        doc_budget,
                        tf_sfg_pairs=max(1, int(pair_budget)),
                    )
                budget_per_doc[doc.doc_id] = {
                    "fact_count": doc_budget.fact_count,
                    "page_count": doc_budget.page_count,
                    "doc_type": doc_budget.doc_type,
                    "size_signal": doc_budget.size_signal,
                    "tf_sfg_pairs": doc_budget.tf_sfg_pairs,
                    "fallback_singlehop": doc_budget.fallback_singlehop,
                    "strict_target": doc_budget.strict_target,
                    "attempt_cap": doc_budget.attempt_cap,
                    "mh_floor_fraction": doc_budget.mh_floor_fraction,
                    "untyped_mh_cap_fraction": doc_budget.untyped_mh_cap_fraction,
                    "fb_cap_fraction": doc_budget.fb_cap_fraction,
                }
                tracer.emit(
                    "budget",
                    "doc_budget_computed",
                    doc_id=doc.doc_id,
                    counts={
                        "facts": doc_budget.fact_count,
                        "pages": doc_budget.page_count,
                        "strict_target": doc_budget.strict_target,
                        "attempt_cap": doc_budget.attempt_cap,
                    },
                    metrics={"size_signal": doc_budget.size_signal},
                    data=budget_per_doc[doc.doc_id],
                )
                cascade_stats.setdefault(doc.doc_id, {})
                logger.info(
                    f"  [v16.2] {doc.doc_id}: budget "
                    f"tf_sfg_pairs={doc_budget.tf_sfg_pairs} "
                    f"fallback_singlehop={doc_budget.fallback_singlehop} "
                    f"strict_target={doc_budget.strict_target} "
                    f"attempt_cap={doc_budget.attempt_cap}"
                )

            logger.info(f"[Stage] Building fact vector store: {doc.doc_id}")
            with tracer.stage("vector_index", doc_id=doc.doc_id):
                embeddings = embed_facts(facts)
                index = FactIndex(dim=embeddings.shape[1])
                index.add(embeddings, [f.fact_id for f in facts])
            tracer.emit(
                "vector_index",
                "index_built",
                doc_id=doc.doc_id,
                counts={"facts": len(facts), "vectors": len(facts)},
                data={"dim": int(embeddings.shape[1])},
            )
            logger.info(f"  {doc.doc_id}: index built with {len(facts)} vectors")

            # v16: build the typed fact sub-graph (P1).
            typed_sfg = None
            if v16_enabled:
                from rag_gt.graph.typed_sfg import TypedSFG
                # Inject resolved doc_type so TypedSFG can apply doc_type_profiles.
                _v16_cfg_with_type = {**v16_cfg, "_doc_type": str(profile.get("doc_type", ""))}
                typed_sfg = TypedSFG(
                    facts,
                    index,
                    _v16_cfg_with_type,
                    v16_2_enabled=v16_2_enabled,
                    doc_budget=doc_budget,
                )
                tf_sfg_llm = answer_llm
                if cost_tracker is not None:
                    from rag_gt.observability.cost_tracker import TrackedLLM

                    tf_sfg_llm = TrackedLLM(
                        answer_llm,
                        cost_tracker,
                        stage="tf_sfg_classify",
                        doc_id=doc.doc_id,
                    )
                tf_sfg_cfg = v16_cfg.get("tf_sfg", {}) or {}
                tf_sfg_deadline = _tf_sfg_stage_deadline(
                    deadline,
                    time.time(),
                    max_wall_fraction=float(tf_sfg_cfg.get("max_wall_fraction", 0.35)),
                    reserve_generation_minutes=float(
                        tf_sfg_cfg.get("reserve_generation_minutes", 20.0)
                    ),
                )
                with tracer.stage("tf_sfg", doc_id=doc.doc_id):
                    typed_sfg.build(tf_sfg_llm, deadline=tf_sfg_deadline)
                _edge_type_counts: dict[str, int] = {}
                _edge_single_fact_risk = 0
                _edge_margin_values: list[float] = []
                _edge_contribution_values: list[float] = []
                for _edge in typed_sfg.edge_map.values():
                    _edge_type_counts[_edge.type] = _edge_type_counts.get(_edge.type, 0) + 1
                    _margin = float(getattr(_edge, "joint_only_margin", 0.0) or 0.0)
                    _edge_margin_values.append(_margin)
                    _edge_contribution_values.append(
                        float(getattr(_edge, "answer_contribution_score", 0.0) or 0.0)
                    )
                    if _margin <= 0.05:
                        _edge_single_fact_risk += 1
                tracer.emit(
                    "tf_sfg",
                    "graph_built",
                    doc_id=doc.doc_id,
                    counts={
                        "classified_pairs": typed_sfg.classified_pairs,
                        "accepted_edges": typed_sfg.edge_count,
                        "source_facts_with_edges": len(typed_sfg.adj),
                        "pre_llm_redundant_pairs_skipped": typed_sfg.redundant_pairs_skipped,
                    },
                    thresholds={
                        "min_cosine": typed_sfg.min_cosine,
                        "nli_edge_threshold": typed_sfg.nli_threshold,
                        "max_pairs": typed_sfg.max_pairs,
                        "edge_minimality_enabled": typed_sfg.edge_minimality_enabled,
                        "edge_single_fact_threshold": typed_sfg.edge_single_fact_threshold,
                        "edge_min_joint_only_margin": typed_sfg.edge_min_joint_only_margin,
                        "answer_contribution_enabled": typed_sfg.answer_contribution_enabled,
                        "contribution_own_threshold": typed_sfg.contribution_own_threshold,
                        "pre_llm_redundancy_enabled": typed_sfg.pre_llm_redundancy_enabled,
                        "bidirectional_entailment_threshold": typed_sfg.pre_llm_redundancy_threshold,
                        "max_wall_fraction": float(tf_sfg_cfg.get("max_wall_fraction", 0.35)),
                        "reserve_generation_minutes": float(
                            tf_sfg_cfg.get("reserve_generation_minutes", 20.0)
                        ),
                    },
                    data={
                        "build_timed_out": typed_sfg.build_timed_out,
                        "build_budget_exhausted": typed_sfg.build_budget_exhausted,
                        "L0": dict(typed_sfg.l0_stats),
                        "edge_count_by_type": _edge_type_counts,
                        "edge_single_fact_risk_count": _edge_single_fact_risk,
                        "edge_joint_only_margin_min": (
                            min(_edge_margin_values) if _edge_margin_values else None
                        ),
                        "edge_joint_only_margin_mean": (
                            sum(_edge_margin_values) / len(_edge_margin_values)
                            if _edge_margin_values else None
                        ),
                        "answer_contribution_score_mean": (
                            sum(_edge_contribution_values) / len(_edge_contribution_values)
                            if _edge_contribution_values else None
                        ),
                        "edge_sample": [
                            e.to_dict()
                            for e in list(typed_sfg.edge_map.values())[:50]
                        ],
                    },
                )
                if v16_2_enabled:
                    edge_labels: dict[str, int] = {}
                    for edge in typed_sfg.edge_map.values():
                        edge_labels[edge.type] = edge_labels.get(edge.type, 0) + 1
                    cascade_stats.setdefault(doc.doc_id, {})["L0"] = dict(
                        typed_sfg.l0_stats
                    )
                    cascade_stats.setdefault(doc.doc_id, {})["edge_canonicalize"] = {
                        "by_label": edge_labels,
                        "unknown_count": int(edge_labels.get("UNKNOWN", 0)),
                    }
                    tf_sfg_cache_hits = 0
                    if cost_tracker is not None:
                        stage_summary = (
                            cost_tracker.get_summary_for_doc(doc.doc_id)
                            .by_stage.get("tf_sfg_classify")
                        )
                        if stage_summary is not None:
                            tf_sfg_cache_hits = int(stage_summary.cache_hit_calls)
                    cascade_stats.setdefault(doc.doc_id, {})["tf_sfg"] = {
                        "classified": typed_sfg.classified_pairs,
                        "accepted_edges": typed_sfg.edge_count,
                        "cache_hits": tf_sfg_cache_hits,
                    }
                logger.info(
                    f"  [v16] {doc.doc_id}: TF-SFG built — {typed_sfg.edge_count} typed edges"
                )

            # Resolve doc-type profile overrides for chain_quality and qa_nli gates.
            # Must be computed before _filter_chains (which uses _doc_chain_max_page_gap).
            _resolved_doc_type = str(profile.get("doc_type", ""))
            _dt_profile = (
                v16_cfg.get("doc_type_profiles", {}).get(_resolved_doc_type.upper(), {})
                if v16_enabled else {}
            )
            _profile_chain_quality = _dt_profile.get("chain_quality", {})
            _doc_chain_max_page_gap = int(
                _profile_chain_quality.get("max_page_gap", chain_max_page_gap)
            )
            _profile_qa_nli = _dt_profile.get("qa_nli", {})
            _profile_question_generation = _dt_profile.get("question_generation", {})
            qa_nli_cfg_base = v16_cfg.get("qa_nli", {}) if v16_enabled else {}
            qa_nli_cfg = {**qa_nli_cfg_base, **_profile_qa_nli} if _profile_qa_nli else qa_nli_cfg_base

            logger.info(f"[Stage] Sampling multi-hop chains: {doc.doc_id}")
            depth_dist = _apply_depth_controls(
                _build_depth_distribution(target_questions, cfg),
                question_mode=question_mode,
                min_hops=min_hops,
                max_hops=max_hops,
            )
            rng = random.Random(_stable_doc_seed(doc.doc_id))
            if v16_enabled and typed_sfg is not None and typed_sfg.edge_count > 0:
                sampler_name = "typed_sfg_walk"
                with tracer.stage("chain_sampling", doc_id=doc.doc_id):
                    chains = typed_sfg.walk_typed_paths(depth_dist, rng)
                logger.info(
                    f"  [v16] {doc.doc_id}: {len(chains)} typed-graph chains sampled"
                )
            elif use_random_fallback:
                sampler_name = "legacy_random"
                with tracer.stage("chain_sampling", doc_id=doc.doc_id):
                    chains = _legacy_random_chains(facts, depth_dist, rng)
            else:
                sampler_name = "semantic_multihop"
                with tracer.stage("chain_sampling", doc_id=doc.doc_id):
                    chains = enumerate_candidate_chains(
                        facts,
                        index,
                        n_chains_per_depth=depth_dist,
                        min_cosine=min_cosine,
                        role_bias=role_bias,
                        rng=rng,
                    )
            tracer.emit(
                "chain_sampling",
                "chains_sampled",
                doc_id=doc.doc_id,
                counts={"chains": len(chains)},
                thresholds={"depth_distribution": depth_dist, "min_cosine": min_cosine},
                data={
                    "sampler": sampler_name,
                    "sample": [chain_snapshot(c, _facts_by_id(facts)) for c in chains[:12]],
                },
            )
            chain_filter_stats: dict = {}
            chains = _filter_chains(
                chains,
                by_id=_facts_by_id(facts),
                question_mode=question_mode,
                min_hops=min_hops,
                max_hops=max_hops,
                min_distinct_chunks=min_distinct_chunks,
                min_distinct_roles=min_distinct_roles,
                min_distinct_pages=min_distinct_pages,
                min_char_gap=min_char_gap,
                chain_quality_enabled=chain_quality_enabled and not v16_2_enabled,
                chain_max_page_gap=_doc_chain_max_page_gap,
                chain_quality_stats=chain_quality_drop_reasons,
                c3_enabled=qrsg_enabled,
                c3_score_threshold=c3_score_threshold,
                c3_reject_unknown=c3_reject_unknown,
                c3_stats=c3_stats,
                filter_stats=chain_filter_stats,
            )
            tracer.emit(
                "chain_filter",
                "chains_filtered",
                doc_id=doc.doc_id,
                counts={
                    "input": chain_filter_stats.get("input", 0),
                    "output": chain_filter_stats.get("output", len(chains)),
                    "dropped": chain_filter_stats.get("rejected", 0),
                },
                thresholds={
                    "question_mode": question_mode,
                    "min_hops": min_hops,
                    "max_hops": max_hops,
                    "min_distinct_chunks": min_distinct_chunks,
                    "min_distinct_roles": min_distinct_roles,
                    "min_distinct_pages": min_distinct_pages,
                    "min_char_gap": min_char_gap,
                    "chain_quality_enabled": chain_quality_enabled and not v16_2_enabled,
                    "chain_max_page_gap": _doc_chain_max_page_gap,
                    "c3_enabled": qrsg_enabled,
                    "c3_score_threshold": c3_score_threshold,
                },
                data=chain_filter_stats,
            )
            for reason, count in (chain_filter_stats.get("by_reason") or {}).items():
                tracer.drop(
                    "chain_filter",
                    reason,
                    doc_id=doc.doc_id,
                    counts={"dropped": count},
                    data={
                        "example": (chain_filter_stats.get("examples") or {}).get(reason)
                    },
                )
            if v16_enabled:
                fallback_cfg = v16_cfg.get("fallback", {})
                if bool(fallback_cfg.get("enabled", True)) and len(chains) < candidate_limit:
                    configured_fallback_limit = int(
                        fallback_cfg.get("singlehop_candidates", 24)
                    )
                    if v16_2_enabled and doc_budget is not None:
                        configured_fallback_limit = int(doc_budget.fallback_singlehop)
                    fallback_limit = min(
                        configured_fallback_limit,
                        max(0, candidate_limit - len(chains)),
                    )
                    fallback_chains = _v16_singlehop_fallback_chains(
                        facts,
                        chains,
                        limit=fallback_limit,
                        min_words=int(fallback_cfg.get("min_fact_words", 8)),
                        max_words=int(fallback_cfg.get("max_fact_words", 45)),
                        rng=rng,
                    )
                    if fallback_chains:
                        chains.extend(fallback_chains)
                        tracer.emit(
                            "chain_sampling",
                            "singlehop_fallback_added",
                            doc_id=doc.doc_id,
                            counts={"added": len(fallback_chains), "chains_total": len(chains)},
                            thresholds={
                                "fallback_limit": fallback_limit,
                                "min_fact_words": int(fallback_cfg.get("min_fact_words", 8)),
                                "max_fact_words": int(fallback_cfg.get("max_fact_words", 45)),
                            },
                            data={
                                "sample": [
                                    chain_snapshot(c, _facts_by_id(facts))
                                    for c in fallback_chains[:12]
                                ]
                            },
                        )
                        logger.info(
                            f"  [v16] {doc.doc_id}: added {len(fallback_chains)} "
                            "single-hop fallback chains"
                        )
            logger.info(f"  {doc.doc_id}: {len(chains)} candidate chains")

            by_id = _facts_by_id(facts)
            chain_scores_by_key: dict[tuple[str, ...], object] = {}
            l1_stats: dict = {}
            yield_controller = None
            doc_strict_target = target_questions
            doc_candidate_limit = candidate_limit
            if v16_enabled and not v16_2_enabled:
                from rag_gt.graph.chain_scorer import ChainScorerConfig, rank_and_select

                scorer_raw = v16_cfg.get("chain_scorer", {})
                scorer_cfg = ChainScorerConfig.from_dict(scorer_raw)
                if scorer_cfg.enabled:
                    with tracer.stage("chain_scoring", doc_id=doc.doc_id):
                        ranked_scores, l1_stats = rank_and_select(
                            chains,
                            by_id,
                            scorer_cfg,
                            attempt_cap=doc_candidate_limit,
                            map_override=None,
                        )
                    chains = [item.chain for item in ranked_scores]
                    chain_scores_by_key = {
                        tuple(item.chain.fact_ids): item for item in ranked_scores
                    }
                    cascade_stats.setdefault(doc.doc_id, {})["L1_plain_v16"] = dict(l1_stats)
                    tracer.emit(
                        "chain_scoring",
                        "chains_ranked",
                        doc_id=doc.doc_id,
                        counts={
                            "input": l1_stats.get("pass1_input", 0),
                            "output": len(chains),
                            "pass2_input": l1_stats.get("pass2_input", 0),
                            "hardfail": l1_stats.get("hardfail", 0),
                        },
                        thresholds={
                            "min_pass1_score": scorer_cfg.min_pass1_score,
                            "min_final_score_to_keep": scorer_cfg.min_final_score_to_keep,
                            "nli_rerank_k": scorer_cfg.nli_rerank_k,
                        },
                        data={
                            "stats": l1_stats,
                            "top_scores": [
                                item.to_dict()
                                for item in ranked_scores[:20]
                            ],
                        },
                    )
                    logger.info(
                        f"  [v16] {doc.doc_id}: relation-aware L1 ranked "
                        f"{len(chains)} chains"
                    )

            if v16_2_enabled and doc_budget is not None:
                from rag_gt.graph.chain_scorer import ChainScorerConfig, rank_and_select
                from rag_gt.pipeline.yield_controller import YieldController, YieldControllerConfig

                scorer_cfg = ChainScorerConfig.from_dict(
                    v16_2_cfg.get("chain_scorer")
                )
                with tracer.stage("chain_scoring", doc_id=doc.doc_id):
                    ranked_scores, l1_stats = rank_and_select(
                        chains,
                        by_id,
                        scorer_cfg,
                        attempt_cap=doc_budget.attempt_cap,
                        map_override=(v16_2_cfg.get("edge_canonicalize") or {}).get("map"),
                    )
                chains = [item.chain for item in ranked_scores]
                chain_scores_by_key = {
                    tuple(item.chain.fact_ids): item for item in ranked_scores
                }
                yield_cfg = YieldControllerConfig.from_dict(
                    v16_2_cfg.get("yield_controller")
                )
                yield_controller = YieldController(
                    cfg=yield_cfg,
                    strict_target=doc_budget.strict_target,
                )
                doc_strict_target = doc_budget.strict_target
                doc_candidate_limit = min(candidate_limit, doc_budget.attempt_cap)
                logger.info(
                    f"  [v16.2] {doc.doc_id}: L1 kept {len(chains)} chains "
                    f"(pass2={l1_stats.get('pass2_input', 0)})"
                )
                cascade_stats.setdefault(doc.doc_id, {})["L1"] = dict(l1_stats)
                tracer.emit(
                    "chain_scoring",
                    "chains_ranked",
                    doc_id=doc.doc_id,
                    counts={
                        "input": l1_stats.get("pass1_input", 0),
                        "output": len(chains),
                        "pass2_input": l1_stats.get("pass2_input", 0),
                        "hardfail": l1_stats.get("hardfail", 0),
                    },
                    thresholds={
                        "min_pass1_score": scorer_cfg.min_pass1_score,
                        "min_final_score_to_keep": scorer_cfg.min_final_score_to_keep,
                        "nli_rerank_k": scorer_cfg.nli_rerank_k,
                    },
                    data={
                        "stats": l1_stats,
                        "top_scores": [
                            item.to_dict()
                            for item in ranked_scores[:20]
                        ],
                    },
                )

            doc_questions: List[QuestionGT] = []
            q_idx = 0
            topology_stats = {
                "dropped_no_intent": 0,
                "by_chain_type": {"typed_mh": 0, "untyped_mh": 0, "fb": 0},
            }

            def _pg_cat_entry() -> dict:
                return {"attempted": 0, "failed_qa_nli": 0, "failed_quality": 0, "failed_answer_nli": 0, "accepted": 0}

            post_gen_by_cat: dict[str, dict] = {}

            def _pg(cat: str | None) -> dict:
                key = cat if cat in ("typed_mh", "untyped_mh", "fb") else "unknown"
                return post_gen_by_cat.setdefault(key, _pg_cat_entry())

            # Tuple: (chain, facts, question, qrsg_verdict, answer, v16_extras)
            # v16_extras is {} for v15 runs; dict with 'question_assumptions' etc. for v16.
            generated: List[
                Tuple[FactChain, List[Fact], str, Optional[RelationVerdict], str, dict]
            ] = []

            pool = ThreadPoolExecutor(max_workers=concurrency)
            futures: dict = {}
            wall_timeout_hit = False
            qgen_abort_reason: Optional[str] = None
            qgen_api_error_failures = 0
            try:
                for chain in chains:
                    if len(futures) >= doc_candidate_limit or _deadline_expired(deadline):
                        break
                    cf = _chain_facts(chain, by_id)
                    if not cf:
                        continue
                    chain_category = None
                    if v16_enabled:
                        if v16_2_enabled and yield_controller is not None and doc_budget is not None:
                            from rag_gt.generation.topology_intent import (
                                TopologyIntentConfig,
                                pick_topology_intent,
                            )
                            from rag_gt.graph.chain_scorer import assign_category

                            if yield_controller.is_hard_stopped(doc_budget.attempt_cap):
                                break
                            score = chain_scores_by_key.get(tuple(chain.fact_ids))
                            chain_category = (
                                getattr(score, "category")
                                if score is not None
                                else assign_category(
                                    chain,
                                    (v16_2_cfg.get("edge_canonicalize") or {}).get("map"),
                                )
                            )
                            topology_stats["by_chain_type"][chain_category] = (
                                topology_stats["by_chain_type"].get(chain_category, 0) + 1
                            )
                            if not yield_controller.can_attempt(chain_category):
                                continue
                            intent_decision = pick_topology_intent(
                                score if score is not None else chain,
                                intent_counts,
                                intent_target_dist if intent_target_dist else None,
                                TopologyIntentConfig.from_dict(
                                    v16_2_cfg.get("topology_intent")
                                ),
                                category=chain_category,
                                map_override=(
                                    (v16_2_cfg.get("edge_canonicalize") or {}).get("map")
                                ),
                            )
                            if not intent_decision.accepted:
                                topology_stats["dropped_no_intent"] += 1
                                continue
                            intent = str(intent_decision.intent)
                            yield_controller.record_attempt(chain_category)
                        else:
                            from rag_gt.generation.intent_sampler import pick_intent

                            if len(cf) == 1:
                                # Single-fact fallback rows cannot honestly satisfy
                                # comparative/inferential/procedural multi-fact
                                # intents. Keep them simple and let typed chains
                                # carry the novelty-heavy intents.
                                intent = "factoid"
                            else:
                                intent = pick_intent(
                                    intent_counts,
                                    intent_target_dist if intent_target_dist else None,
                                )
                        intent_counts[intent] = intent_counts.get(intent, 0) + 1
                        future = pool.submit(
                            _generate_and_answer_v16,
                            cf,
                            chain.chain_edges,
                            gt_llm,
                            answer_llm,
                            v16_cfg,
                            qrsg_llm,
                            qrsg_gate_version,
                            qrsg_max_tokens,
                            intent,
                            doc.doc_id,
                            cost_tracker,
                            _profile_qa_nli or None,
                            _profile_question_generation or None,
                        )
                        futures[future] = (chain, cf, chain_category)
                        tracer.emit(
                            "candidate_generation",
                            "candidate_submitted",
                            doc_id=doc.doc_id,
                            item_id=make_chain_id(list(chain.fact_ids)),
                            counts={"facts": len(cf), "depth": chain.depth},
                            data={
                                "chain": chain_snapshot(chain, by_id),
                                "category": chain_category,
                                "intent": intent,
                            },
                        )
                    else:
                        future = pool.submit(
                            _generate_and_answer,
                            cf,
                            gt_llm,
                            answer_llm,
                            qrsg_llm,
                            qrsg_gate_version,
                            qrsg_max_tokens,
                            doc.doc_id,
                            cost_tracker,
                        )
                        futures[future] = (chain, cf, chain_category)
                        tracer.emit(
                            "candidate_generation",
                            "candidate_submitted",
                            doc_id=doc.doc_id,
                            item_id=make_chain_id(list(chain.fact_ids)),
                            counts={"facts": len(cf), "depth": chain.depth},
                            data={"chain": chain_snapshot(chain, by_id)},
                        )

                collection_timeout = max(
                    _LLM_CALL_TIMEOUT * 4,
                    int(math.ceil(len(futures) / max(concurrency, 1)))
                    * _LLM_CALL_TIMEOUT,
                )
                remaining = _deadline_remaining_seconds(deadline)
                if remaining is not None:
                    if remaining <= 0:
                        raise TimeoutError
                    collection_timeout = min(collection_timeout, int(math.ceil(remaining)))
                completed = as_completed(futures, timeout=collection_timeout)
                completed_count = 0
                for fut in completed:
                    if _deadline_expired(deadline):
                        wall_timeout_hit = True
                        raise TimeoutError
                    completed_count += 1
                    if progress_every and completed_count % progress_every == 0:
                        logger.info(
                            f"[QGen] doc={doc.doc_id} completed "
                            f"{completed_count}/{len(futures)} candidates"
                        )
                    chain, cf, chain_category = futures[fut]
                    chain_id = make_chain_id(list(chain.fact_ids))
                    try:
                        raw_result = fut.result(timeout=_LLM_CALL_TIMEOUT)
                        if v16_enabled:
                            q, verdict, a, v16_extras = raw_result
                        else:
                            q, verdict, a = raw_result
                            v16_extras = {}
                    except Exception as e:
                        logger.warning(
                            f"[QGen] doc={doc.doc_id} future failed: {type(e).__name__}: {e}"
                        )
                        drops.questiongen += 1
                        tracer.drop(
                            "candidate_generation",
                            f"future_failed:{type(e).__name__}",
                            doc_id=doc.doc_id,
                            item_id=chain_id,
                            data={"error": str(e), "chain": chain_snapshot(chain, by_id)},
                        )
                        from rag_gt.observability.cost_tracker import LiveCallBudgetExceeded

                        if isinstance(e, LiveCallBudgetExceeded):
                            qgen_abort_reason = "live_api_call_cap_reached"
                            wall_timeout_hit = True
                            for pending in futures:
                                pending.cancel()
                            logger.warning(
                                f"[QGen] aborting {doc.doc_id}: live API call cap reached"
                            )
                            tracer.drop(
                                "candidate_generation",
                                qgen_abort_reason,
                                doc_id=doc.doc_id,
                                counts={
                                    "submitted": len(futures),
                                    "survivors": len(generated),
                                },
                            )
                            raise TimeoutError
                        if isinstance(e, APIError):
                            qgen_api_error_failures += 1
                            if (
                                qgen_api_error_fail_fast_after > 0
                                and qgen_api_error_failures >= qgen_api_error_fail_fast_after
                                and not generated
                            ):
                                qgen_abort_reason = "api_error_fail_fast"
                                wall_timeout_hit = True
                                for pending in futures:
                                    pending.cancel()
                                logger.warning(
                                    f"[QGen] aborting {doc.doc_id} after "
                                    f"{qgen_api_error_failures} consecutive API failures"
                                )
                                tracer.drop(
                                    "candidate_generation",
                                    qgen_abort_reason,
                                    doc_id=doc.doc_id,
                                    counts={
                                        "api_errors": qgen_api_error_failures,
                                        "submitted": len(futures),
                                        "survivors": len(generated),
                                    },
                                    thresholds={
                                        "api_error_fail_fast_after": qgen_api_error_fail_fast_after
                                    },
                                )
                                raise TimeoutError
                        continue
                    if not q:
                        drops.questiongen += 1
                        tracer.drop(
                            "candidate_generation",
                            "question_generation_failed",
                            doc_id=doc.doc_id,
                            item_id=chain_id,
                            data={"chain": chain_snapshot(chain, by_id), "category": chain_category},
                        )
                        run_logger.log_question(
                            doc_id=doc.doc_id, depth=0, cos=0.0,
                            question="failed_questiongen", status="FAIL",
                        )
                        continue
                    _pg(chain_category)["attempted"] += 1
                    # v16: drop rows that failed the QA-NLI gate after all retries.
                    if v16_enabled and not v16_extras.get("qa_nli_accepted", True):
                        drops.qa_nli += 1
                        _pg(chain_category)["failed_qa_nli"] += 1
                        tracer.drop(
                            "post_generation_gates",
                            "qa_nli_rejected",
                            doc_id=doc.doc_id,
                            item_id=chain_id,
                            metrics={
                                "rel_fail_rate": v16_extras.get("qa_nli_fail_rate", 0),
                            },
                            thresholds=qa_nli_cfg,
                            data={
                                "question": q,
                                "category": chain_category,
                                "intent": v16_extras.get("intent"),
                                "qa_nli_reason": v16_extras.get("qa_nli_reason"),
                                "qa_nli_missing_fact_indices": v16_extras.get("qa_nli_missing_fact_indices", []),
                                "premise_leakage_indices": v16_extras.get("premise_leakage_indices", []),
                                "premise_leakage_advisory_only": v16_extras.get("premise_leakage_advisory_only", False),
                                "question_assumptions": v16_extras.get("question_assumptions", []),
                                "chain": chain_snapshot(chain, by_id),
                            },
                        )
                        logger.debug(
                            f"[QA-NLI] drop q={q[:80]!r} "
                            f"rel_fail_rate={v16_extras.get('qa_nli_fail_rate', 0):.3f}"
                        )
                        continue
                    # QRSG gate: write trace for every verdict; drop on reject.
                    if verdict is not None:
                        _append_qrsg_trace(qrsg_trace_path, q, cf, verdict)
                        qrsg_total += 1
                        rel = verdict.relation_type or "unknown"
                        qrsg_relation_type_pre[rel] = (
                            qrsg_relation_type_pre.get(rel, 0) + 1
                        )
                        depth_key = str(chain.depth)
                        depth_bucket = qrsg_by_depth.setdefault(
                            depth_key, {"total": 0, "accepted": 0}
                        )
                        depth_bucket["total"] += 1
                        concept_keys = set(verdict.evidence_map.keys())
                        concept_keys.update(verdict.unsupported_concepts)
                        concept_total = len(concept_keys)
                        if concept_total > 0:
                            qrsg_concept_coverage_sum += (
                                1.0
                                - (
                                    len(verdict.unsupported_concepts)
                                    / concept_total
                                )
                            )
                            qrsg_concept_coverage_count += 1
                        if not verdict.accepted:
                            drops.qrsg += 1
                            reason_key = verdict.reason.split(":")[0]
                            qrsg_reject_reasons[reason_key] = (
                                qrsg_reject_reasons.get(reason_key, 0) + 1
                            )
                            if verdict.risky_frame_hit:
                                qrsg_risky_frame_rejects += 1
                            _append_qrsg_trace(qrsg_rejected_path, q, cf, verdict)
                            tracer.drop(
                                "post_generation_gates",
                                f"qrsg:{reason_key}",
                                doc_id=doc.doc_id,
                                item_id=chain_id,
                                data={
                                    "question": q,
                                    "verdict": verdict.to_dict(),
                                    "chain": chain_snapshot(chain, by_id),
                                },
                            )
                            logger.debug(
                                f"[QRSG] rejected q={q[:80]!r} reason={verdict.reason}"
                            )
                            continue
                        qrsg_accepted_count += 1
                        depth_bucket["accepted"] += 1
                        qrsg_relation_type_post[rel] = (
                            qrsg_relation_type_post.get(rel, 0) + 1
                        )
                    if not a:
                        drops.answer += 1
                        _pg(chain_category)["failed_quality"] += 1
                        answer_failure_reason = str(
                            v16_extras.get(
                                "answer_failure_reason", "answer_generation_failed"
                            )
                        )
                        tracer.drop(
                            "candidate_generation",
                            answer_failure_reason,
                            doc_id=doc.doc_id,
                            item_id=chain_id,
                            data={
                                "question": q,
                                "arm_all_clauses_pass": v16_extras.get(
                                    "arm_all_clauses_pass"
                                ),
                                "reasoning_trace": v16_extras.get("reasoning_trace", []),
                                "chain": chain_snapshot(chain, by_id),
                            },
                        )
                        run_logger.log_question(
                            doc_id=doc.doc_id, depth=0, cos=0.0,
                            question=answer_failure_reason, status="FAIL",
                        )
                        continue
                    if v16_enabled:
                        prov_reason = _provenance_reject_reason(cf)
                        if prov_reason:
                            drops.provenance += 1
                            _pg(chain_category)["failed_quality"] += 1
                            tracer.drop(
                                "post_generation_gates",
                                f"provenance:{prov_reason}",
                                doc_id=doc.doc_id,
                                item_id=chain_id,
                                data={"question": q, "answer": a, "chain": chain_snapshot(chain, by_id)},
                            )
                            logger.debug(
                                f"[Provenance] drop q={q[:80]!r} reason={prov_reason}"
                            )
                            continue
                        domain_reason = _question_domain_reject_reason(q, a)
                        if domain_reason:
                            drops.quality += 1
                            _pg(chain_category)["failed_quality"] += 1
                            tracer.drop(
                                "post_generation_gates",
                                f"question_domain:{domain_reason}",
                                doc_id=doc.doc_id,
                                item_id=chain_id,
                                data={"question": q, "answer": a, "chain": chain_snapshot(chain, by_id)},
                            )
                            logger.debug(
                                f"[DomainQuestion] drop q={q[:100]!r} reason={domain_reason}"
                            )
                            continue
                    ok, _reason = check_structure(q, a)
                    if not ok:
                        drops.structure += 1
                        _pg(chain_category)["failed_quality"] += 1
                        tracer.drop(
                            "post_generation_gates",
                            f"structure:{_reason}",
                            doc_id=doc.doc_id,
                            item_id=chain_id,
                            data={"question": q, "answer": a, "chain": chain_snapshot(chain, by_id)},
                        )
                        continue
                    quality = score_generated_pair(q, a, cf)
                    if quality["quality"] < min_gt_quality:
                        drops.quality += 1
                        _pg(chain_category)["failed_quality"] += 1
                        flagged = [
                            k for k, v in quality["flags"].items()
                            if v
                        ]
                        logger.debug(
                            f"[GTQuality] drop q={q[:100]!r} "
                            f"score={quality['quality']:.3f} flags={flagged}"
                        )
                        tracer.drop(
                            "post_generation_gates",
                            "quality_below_threshold",
                            doc_id=doc.doc_id,
                            item_id=chain_id,
                            metrics={"quality": quality["quality"]},
                            thresholds={"min_gt_quality": min_gt_quality},
                            data={
                                "question": q,
                                "answer": a,
                                "quality": quality,
                                "flagged": flagged,
                                "chain": chain_snapshot(chain, by_id),
                            },
                        )
                        continue
                    # Cross-doc dedup: skip questions seen earlier in the run.
                    norm = _normalize_question_for_dedup(q)
                    if norm in seen_question_norms:
                        drops.duplicates += 1
                        tracer.drop(
                            "post_generation_gates",
                            "duplicate_question_exact",
                            doc_id=doc.doc_id,
                            item_id=chain_id,
                            data={"question": q, "normalized": norm, "chain": chain_snapshot(chain, by_id)},
                        )
                        continue
                    q_embedding = None
                    if semantic_duplicate_threshold > 0:
                        try:
                            q_embedding = embed_query(q)
                        except Exception as e:
                            logger.warning(
                                f"[Dedup] semantic embedding failed; "
                                f"falling back to exact dedup: {type(e).__name__}: {e}"
                            )
                            semantic_duplicate_threshold = 0.0
                        if _is_semantic_duplicate_embedding(
                            q_embedding,
                            seen_question_embeddings,
                            semantic_duplicate_threshold,
                        ):
                            drops.duplicates += 1
                            tracer.drop(
                                "post_generation_gates",
                                "duplicate_question_semantic",
                                doc_id=doc.doc_id,
                                item_id=chain_id,
                                thresholds={"semantic_duplicate_threshold": semantic_duplicate_threshold},
                                data={"question": q, "chain": chain_snapshot(chain, by_id)},
                            )
                            continue
                    seen_question_norms.add(norm)
                    if q_embedding is not None:
                        seen_question_embeddings.append(q_embedding)
                    if v16_2_enabled and chain_category is not None:
                        v16_extras["v16_2_category"] = chain_category
                    tracer.emit(
                        "post_generation_gates",
                        "candidate_pre_nli_survivor",
                        status="accepted",
                        doc_id=doc.doc_id,
                        item_id=chain_id,
                        counts={"depth": chain.depth, "facts": len(cf)},
                        metrics={"quality": quality["quality"]},
                        data={
                            "question": q,
                            "answer": a,
                            "category": chain_category,
                            "quality": quality,
                            "chain": chain_snapshot(chain, by_id),
                        },
                    )
                    generated.append((chain, cf, q, verdict, a, v16_extras))
            except TimeoutError:
                wall_timeout_hit = _deadline_expired(deadline) or bool(qgen_abort_reason)
                for fut in futures:
                    fut.cancel()
                reason = qgen_abort_reason or ("wall clock" if wall_timeout_hit else "collection")
                logger.warning(
                    f"[QGen] {reason} timeout for {doc.doc_id}; "
                    f"continuing with {len(generated)} survivors"
                )
                tracer.drop(
                    "candidate_generation",
                    f"timeout:{reason}",
                    doc_id=doc.doc_id,
                    counts={"submitted": len(futures), "survivors": len(generated)},
                    thresholds={"collection_timeout_seconds": collection_timeout},
                )
            finally:
                pool.shutdown(wait=not wall_timeout_hit, cancel_futures=True)

            if enable_nli and not fast_mode and generated:
                # generated tuple is (chain, cf, q, verdict, a, v16_extras)
                # answer is at index 4 — unchanged from v15.
                pairs = [(g[4], g[1]) for g in generated]
                nli_pass = batch_answer_entailment(pairs)
            else:
                nli_pass = [True] * len(generated)

            survivors = [
                (chain, cf, q, verdict, a, v16_extras)
                for (chain, cf, q, verdict, a, v16_extras), passed in zip(generated, nli_pass)
                if passed
            ]
            drops.nli += len(generated) - len(survivors)
            for (chain, cf, q, verdict, a, v16_extras), passed in zip(generated, nli_pass):
                if not passed:
                    _pg(v16_extras.get("v16_2_category"))["failed_answer_nli"] += 1
                    tracer.drop(
                        "answer_nli",
                        "answer_not_entailed",
                        doc_id=doc.doc_id,
                        item_id=make_chain_id(list(chain.fact_ids)),
                        thresholds={
                            "enabled": enable_nli,
                            "fast_mode": fast_mode,
                            "nli_gt_consistency_threshold": cfg.get("validation", {}).get("nli_gt_consistency_threshold"),
                        },
                        data={
                            "question": q,
                            "answer": a,
                            "category": v16_extras.get("v16_2_category"),
                            "chain": chain_snapshot(chain, by_id),
                        },
                    )
            tracer.emit(
                "answer_nli",
                "batch_answer_nli_complete",
                doc_id=doc.doc_id,
                counts={
                    "input": len(generated),
                    "passed": len(survivors),
                    "dropped": len(generated) - len(survivors),
                },
                thresholds={"enabled": enable_nli and not fast_mode},
            )

            for chain, cf, q, verdict, a, v16_extras in survivors:
                if len(doc_questions) >= doc_strict_target:
                    break
                if v16_2_enabled and yield_controller is not None:
                    category = v16_extras.get("v16_2_category")
                    if category and not yield_controller.can_accept(category):
                        tracer.drop(
                            "minimality",
                            f"yield_accept_cap:{category}",
                            doc_id=doc.doc_id,
                            item_id=make_chain_id(list(chain.fact_ids)),
                            data={"question": q, "category": category, "yield": yield_controller.get_summary()},
                        )
                        continue
                depth = chain.depth
                if enable_min and not fast_mode and depth > 1:
                    minimality_details = None
                    try:
                        if v16_enabled:
                            support_verdict = support_minimality_check(
                                q,
                                cf,
                                a,
                                question_assumptions=list(
                                    v16_extras.get("question_assumptions", [])
                                ),
                                reasoning_trace=list(
                                    v16_extras.get("reasoning_trace", [])
                                ),
                                qa_nli_profile=_profile_qa_nli or None,
                            )
                            passes_minimality = support_verdict.passed
                            minimality_details = support_verdict.to_dict()
                        else:
                            minimality_llm = answer_llm
                            if cost_tracker is not None:
                                from rag_gt.observability.cost_tracker import TrackedLLM

                                minimality_llm = TrackedLLM(
                                    answer_llm,
                                    cost_tracker,
                                    stage="minimality_check",
                                    doc_id=doc.doc_id,
                                )
                            passes_minimality = minimal_evidence_check(
                                q, cf, a, minimality_llm
                            )
                    except Exception as e:
                        drops.minimality += 1
                        drops.bump(depth, "minimality")
                        tracer.drop(
                            "minimality",
                            f"minimality_error:{type(e).__name__}",
                            doc_id=doc.doc_id,
                            item_id=make_chain_id(list(chain.fact_ids)),
                            data={"question": q, "answer": a, "error": str(e), "chain": chain_snapshot(chain, by_id)},
                        )
                        logger.warning(
                            f"[Minimality] doc={doc.doc_id} q={q[:100]!r} "
                            f"failed with {type(e).__name__}: {e}"
                        )
                        continue
                    if not passes_minimality:
                        drops.minimality += 1
                        drops.bump(depth, "minimality")
                        reason = (
                            str(minimality_details.get("reason", "not_minimal"))
                            if minimality_details
                            else "not_minimal"
                        )
                        tracer.drop(
                            "minimality",
                            reason,
                            doc_id=doc.doc_id,
                            item_id=make_chain_id(list(chain.fact_ids)),
                            data={
                                "question": q,
                                "answer": a,
                                "minimality": minimality_details,
                                "chain": chain_snapshot(chain, by_id),
                            },
                        )
                        continue

                if v16_2_enabled and yield_controller is not None:
                    category = v16_extras.get("v16_2_category")
                    if category:
                        yield_controller.record_accepted(category)
                        _pg(category)["accepted"] += 1

                q_idx += 1
                msfs_id = f"{doc.doc_id}_q{q_idx:03d}_msfs1"
                msfs = MSFS(msfs_id=msfs_id, fact_ids=[f.fact_id for f in cf])

                # v16 P3: mine provenance-anchored distractors for this chain.
                distractor_spans: List[dict] = []
                if v16_enabled:
                    from rag_gt.generation.distractors import mine_distractors
                    distractors_cfg = v16_cfg.get("distractors", {})
                    try:
                        distractor_spans = mine_distractors(
                            a,
                            cf,
                            by_id,
                            index,
                            per_fact=int(distractors_cfg.get("per_fact", 2)),
                            min_cosine=float(
                                distractors_cfg.get("min_cosine_to_support", 0.55)
                            ),
                            distractor_id_offset=distractor_id_offset,
                        )
                        distractor_id_offset += len(distractor_spans)
                    except Exception as _de:
                        logger.warning(
                            f"[PASD] distractor mining failed for q{q_idx}: {_de}"
                        )
                        tracer.drop(
                            "augmentation",
                            f"distractor_mining_failed:{type(_de).__name__}",
                            doc_id=doc.doc_id,
                            item_id=make_chain_id(list(chain.fact_ids)),
                            data={"question": q, "error": str(_de)},
                        )

                doc_questions.append(
                    QuestionGT(
                        q_id=f"{doc.doc_id}_q{q_idx:03d}",
                        question=q,
                        gold_answer=a,
                        msfs_list=[msfs],
                        doc_ids=[doc.doc_id],
                        required_fact_ids=[f.fact_id for f in cf],
                        difficulty_reasoning_depth=depth,
                        difficulty_semantic_distance=_sem_distance(cf),
                        required_facts=cf,
                        required_fact_groups=[[f.fact_id] for f in cf],
                        hop_type=_hop_type_for_chain(chain),
                        minimum_required_fact_count=len(cf),
                        relation_type=verdict.relation_type if verdict is not None else "",
                        risky_frame_hit=verdict.risky_frame_hit if verdict is not None else None,
                        qrsg_evidence_map=dict(verdict.evidence_map) if verdict is not None else {},
                        # v16 PASS-GT fields — empty when v16 is disabled.
                        tf_sfg_edges=list(chain.chain_edges) if v16_enabled else [],
                        question_assumptions=list(v16_extras.get("question_assumptions", [])),
                        distractor_spans=distractor_spans,
                        reasoning_trace=list(v16_extras.get("reasoning_trace", [])),
                        intent=str(v16_extras.get("intent", "") or ""),
                    )
                )
                tracer.emit(
                    "minimality",
                    "candidate_accepted",
                    status="accepted",
                    doc_id=doc.doc_id,
                    item_id=make_chain_id(list(chain.fact_ids)),
                    counts={
                        "depth": depth,
                        "facts": len(cf),
                        "distractors": len(distractor_spans),
                    },
                    data={
                        "q_id": f"{doc.doc_id}_q{q_idx:03d}",
                        "question": q,
                        "answer": a,
                        "category": v16_extras.get("v16_2_category"),
                        "intent": v16_extras.get("intent"),
                        "chain": chain_snapshot(chain, by_id),
                    },
                )
                drops.bump(depth, "kept")
                run_logger.log_question(
                    doc_id=doc.doc_id, depth=depth, cos=chain.mean_cosine,
                    question=q, status="OK",
                )
                logger.debug(
                    f"  [OK] Q{q_idx}: depth={depth} cos={chain.mean_cosine:.2f} | {q[:60]}"
                )
                if v16_2_enabled and yield_controller is not None and yield_controller.is_halted():
                    logger.info(
                        f"  [v16.2] {doc.doc_id}: yield target met "
                        f"({yield_controller.total_accepted}/{doc_strict_target})"
                    )
                    break

            if v16_2_enabled:
                cascade_stats.setdefault(doc.doc_id, {})["topology_intent"] = dict(
                    topology_stats
                )
                if yield_controller is not None:
                    cascade_stats.setdefault(doc.doc_id, {})["yield"] = (
                        yield_controller.get_summary()
                    )
                if post_gen_by_cat:
                    cascade_stats.setdefault(doc.doc_id, {})["post_generation_by_category"] = (
                        dict(post_gen_by_cat)
                    )
            if doc_questions and "yield" not in cascade_stats.get(doc.doc_id, {}):
                cascade_stats.setdefault(doc.doc_id, {})["yield"] = {
                    "strict_total": len(doc_questions),
                    "strict_target": doc_strict_target,
                }

            # v16 P4: derive AT+CT twins for this doc's accepted strict rows.
            if v16_enabled and doc_questions:
                from rag_gt.generation.twins import derive_twins
                from rag_gt.observability.cost_tracker import LiveCallBudgetExceeded
                twins_cfg = v16_cfg.get("twins", {})
                try:
                    twin_rows = derive_twins(
                        doc_questions,
                        answer_llm,
                        twins_enabled=bool(twins_cfg.get("enabled", True)),
                        ct_threshold=float(
                            twins_cfg.get("ct_nli_contradiction_threshold", 0.40)
                        ),
                        ct_max_retries=int(twins_cfg.get("ct_max_retries", 3)),
                        ct_sample_rate=float(twins_cfg.get("ct_sample_rate", 1.0)),
                        require_answer_change=bool(
                            twins_cfg.get("require_answer_change", True)
                        ),
                        validate_answer_entailment=bool(
                            twins_cfg.get("validate_answer_entailment", True)
                        ),
                        cost_tracker=cost_tracker,
                        doc_id=doc.doc_id,
                    )
                    if twin_rows:
                        doc_questions.extend(twin_rows)
                        tracer.emit(
                            "augmentation",
                            "twins_appended",
                            doc_id=doc.doc_id,
                            counts={"twins": len(twin_rows), "doc_questions_total": len(doc_questions)},
                            thresholds={
                                "ct_threshold": float(
                                    twins_cfg.get("ct_nli_contradiction_threshold", 0.40)
                                ),
                                "ct_max_retries": int(twins_cfg.get("ct_max_retries", 3)),
                                "ct_sample_rate": float(twins_cfg.get("ct_sample_rate", 1.0)),
                            },
                        )
                        logger.info(
                            f"  [v16] {doc.doc_id}: {len(twin_rows)} twins appended "
                            f"(total={len(doc_questions)})"
                        )
                except LiveCallBudgetExceeded as _te:
                    logger.warning(
                        f"[Twins] live API call cap reached for {doc.doc_id}; "
                        "retaining strict rows without further twin generation"
                    )
                    tracer.drop(
                        "augmentation",
                        "twins_skipped_live_api_call_cap_reached",
                        doc_id=doc.doc_id,
                        data={"error": str(_te)},
                    )
                except Exception as _te:
                    logger.warning(f"[Twins] twin derivation failed for {doc.doc_id}: {_te}")
                    tracer.drop(
                        "augmentation",
                        f"twins_failed:{type(_te).__name__}",
                        doc_id=doc.doc_id,
                        data={"error": str(_te)},
                    )

            # Incremental persistence: append per doc so a later crash doesn't
            # discard the run's accumulated questions.
            if doc_questions:
                try:
                    append_gt(doc_questions, f"{corpus_name}.partial", out_dir=out_dir)
                    tracer.emit(
                        "persistence",
                        "partial_gt_appended",
                        doc_id=doc.doc_id,
                        counts={"questions": len(doc_questions)},
                        data={"path": str(incremental_path)},
                    )
                except Exception as e:
                    logger.warning(f"[GT] incremental append failed: {e}")
                    tracer.drop(
                        "persistence",
                        f"partial_append_failed:{type(e).__name__}",
                        doc_id=doc.doc_id,
                        data={"error": str(e), "path": str(incremental_path)},
                    )

            all_questions.extend(doc_questions)
            tracer.emit(
                "run_setup",
                "document_end",
                doc_id=doc.doc_id,
                counts={
                    "facts": len(facts),
                    "candidate_chains": len(chains),
                    "questions": len(doc_questions),
                    "strict_target": doc_strict_target,
                },
                data={
                    "cascade_stats": cascade_stats.get(doc.doc_id, {}),
                    "budget": budget_per_doc.get(doc.doc_id, {}),
                },
            )
            print(
                f"  [{doc.doc_id}] Generated {len(doc_questions)}/{doc_strict_target} questions"
            )

        except FileNotFoundError as e:
            logger.error(f"Doc not found {path}: {e}")
            tracer.drop(
                "ingestion",
                "file_not_found",
                doc_id=os.path.splitext(doc_name)[0],
                data={"path": path, "error": str(e)},
            )
            print(f"  [ERROR] {doc_name}: file not found")
        except Exception as e:
            logger.exception(f"Failed on {path}: {e}")
            tracer.drop(
                "run_setup",
                f"document_failed:{type(e).__name__}",
                doc_id=os.path.splitext(doc_name)[0],
                data={"path": path, "error": str(e)},
            )
            print(f"  [ERROR] {doc_name}: {type(e).__name__} -- {e}")

    final_output_path = save_gt(all_questions, corpus_name, out_dir=out_dir)
    build_summary_path = None
    if budget_per_doc or cascade_stats or cost_tracker is not None:
        build_summary = {
            "budget_per_doc": budget_per_doc,
            "cascade_stats": cascade_stats,
            "cost_tracker": (
                cost_tracker.to_build_summary_dict()
                if cost_tracker is not None
                else {"aggregate": {}}
            ),
        }
        build_summary_path = save_build_summary(final_output_path, build_summary)

    # Clean up the partial file once the final atomic save succeeds.
    try:
        if incremental_path.exists():
            incremental_path.unlink()
    except OSError:
        pass

    stats_path = out_dir / f"{corpus_name}_drop_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    if chain_quality_drop_reasons:
        drops.chain_quality = sum(chain_quality_drop_reasons.values())
    drop_stats = drops.to_dict()
    if fact_domain_drop_reasons:
        drop_stats["fact_domain_reject_by_reason"] = fact_domain_drop_reasons
    if chain_quality_drop_reasons:
        drop_stats["chain_quality_reject_by_reason"] = chain_quality_drop_reasons
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(drop_stats, f, indent=2, ensure_ascii=False)

    # Write QRSG run stats (Task 12).
    if qrsg_enabled and qrsg_stats_path is not None:
        depth_accept_rates = {
            depth: (
                bucket["accepted"] / bucket["total"]
                if bucket.get("total", 0)
                else 0.0
            )
            for depth, bucket in sorted(qrsg_by_depth.items())
        }
        concept_coverage = (
            qrsg_concept_coverage_sum / qrsg_concept_coverage_count
            if qrsg_concept_coverage_count
            else 0.0
        )
        qrsg_stats_path.write_text(
            json.dumps(
                {
                    "gate_version": qrsg_gate_version,
                    "llm_role": qrsg_llm_role,
                    "max_tokens": qrsg_max_tokens,
                    "qrsg_total_questions": qrsg_total,
                    "qrsg_accepted": qrsg_accepted_count,
                    "qrsg_rejected": qrsg_total - qrsg_accepted_count,
                    "qrsg_accept_rate": (
                        qrsg_accepted_count / qrsg_total if qrsg_total else 0.0
                    ),
                    "qrsg_reject_by_reason": qrsg_reject_reasons,
                    "qrsg_risky_frame_reject_rate": (
                        qrsg_risky_frame_rejects / qrsg_total
                        if qrsg_total
                        else 0.0
                    ),
                    "qrsg_chain_answerability_rate_by_depth": depth_accept_rates,
                    "qrsg_question_concept_coverage_mean": concept_coverage,
                    "qrsg_relation_type_distribution": {
                        "pre_gate": qrsg_relation_type_pre,
                        "post_gate": qrsg_relation_type_post,
                    },
                    "c3_self_containment": c3_stats,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info("[QRSG] stats written to %s", qrsg_stats_path)

    trace_summary_path = None
    dashboard_path = None
    try:
        trace_summary = tracer.close(
            {
                "documents_processed": len(paths),
                "total_facts_found": total_facts_extracted,
                "questions_saved": len(all_questions),
                "drop_stats": drop_stats,
                "output_file": final_output_path,
                "build_summary": build_summary_path,
                "drop_stats_path": str(stats_path),
            }
        )
        trace_summary_path = str(trace_summary) if trace_summary else None
        if enable_trace:
            from rag_gt.cli.build_pipeline_dashboard import build_dashboard

            dashboard_path = str(
                build_dashboard(Path(resolved_trace_path), Path(f"{final_output_path}.dashboard.html"))
            )
    except Exception as e:
        logger.warning(f"[Trace] finalization/dashboard failed: {type(e).__name__}: {e}")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  GT GENERATION SUMMARY")
    print("=" * 60)
    print(f"  Documents processed : {len(paths)}")
    print(f"  Total facts found   : {total_facts_extracted}")
    print(f"  Questions saved     : {len(all_questions)}")
    print(f"  Drops               : {drop_stats}")
    if qrsg_enabled:
        print(f"  QRSG verdicts       : {qrsg_total} ({qrsg_accepted_count} accepted, {qrsg_total - qrsg_accepted_count} rejected)")
    print(f"  Elapsed             : {elapsed / 60:.1f} min")
    print(f"  Output file         : {output_path}")
    if build_summary_path:
        print(f"  Build summary       : {build_summary_path}")
    print(f"  Drop stats          : {stats_path}")
    if enable_trace:
        print(f"  Trace JSONL         : {resolved_trace_path}")
        if trace_summary_path:
            print(f"  Trace summary       : {trace_summary_path}")
        if dashboard_path:
            print(f"  Trace dashboard     : {dashboard_path}")
    print(f"  Log file            : {run_logger.get_log_path()}")
    print("=" * 60 + "\n")

    run_logger.log_summary(
        {
            "documents_processed": len(paths),
            "total_facts_found": total_facts_extracted,
            "questions_saved": len(all_questions),
            "drops": drop_stats,
            "elapsed_minutes": round(elapsed / 60, 1),
            "output_file": output_path,
        }
    )


def _legacy_random_chains(
    facts: List[Fact], depth_dist: dict[int, int], rng: random.Random
) -> List[FactChain]:
    """Rollback path: depth-aware random.sample() for comparison runs only.

    Now applies the same `tuple(fact_ids)` dedup as the new sampler so A/B
    comparisons are not biased by hidden duplicates."""
    out: List[FactChain] = []
    seen: set[tuple] = set()
    for depth, n in depth_dist.items():
        if len(facts) < depth:
            continue
        attempts = 0
        max_attempts = n * 10
        accepted = 0
        while accepted < n and attempts < max_attempts:
            attempts += 1
            picked = rng.sample(facts, depth)
            key = tuple(p.fact_id for p in picked)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                FactChain(
                    fact_ids=list(key),
                    anchor_id=picked[0].fact_id,
                    mean_cosine=0.0,
                    role_path=[p.role for p in picked],
                )
            )
            accepted += 1
    rng.shuffle(out)
    return out
