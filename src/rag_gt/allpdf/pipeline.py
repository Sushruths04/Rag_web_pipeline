"""Full GRAFT all-PDF pipeline — Stages 0 through 9.

Stages 0-4 are the new adaptive front-end (allpdf package).
Stages 5-9 call the existing typed-SFG / QGen / validation building blocks,
injecting allpdf facts instead of gt_pipeline's rule-based extractions.

Key differences from gt_pipeline.py:
  - source_neighbor_window = 3  (structural adjacency edges enabled)
  - SFU canonical_form flows into TF-SFG NLI and edge disambiguation
  - Adaptive domain filter (doc-type-aware rule relaxation)
  - Verification gates logged at every stage boundary

Usage (dry-run, no LLM):
    python -m rag_gt.allpdf.pipeline --doc-id ecma404 \\
        --pdf data/test_corpus_allpdf/ecma404_json.pdf \\
        --out-dir data/test_corpus_allpdf/pipeline_run/ecma404 \\
        --dry-run

Usage (full, needs LLM configured):
    python -m rag_gt.allpdf.pipeline --doc-id sutton_rl \\
        --pdf data/docs_rl/SuttonBartoIPRLBook2ndEd.pdf \\
        --out-dir data/test_corpus_allpdf/pipeline_run/sutton_rl \\
        --max-pairs 400 --target-chains 20
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from rag_gt.allpdf.chunk import agentic_chunk
from rag_gt.allpdf.extract import extract_sfu_facts
from rag_gt.allpdf.filter_adaptive import filter_facts_adaptive
from rag_gt.allpdf.ingest import ingest_document
from rag_gt.allpdf.preflight import profile_pdf
from rag_gt.allpdf.verify import VerificationGate, save_checkpoint
from rag_gt.core.types import Fact, FactChain


# ---------------------------------------------------------------------------
# Stage 5 config: hybrid graph (structural edges enabled)
# ---------------------------------------------------------------------------

_HYBRID_TF_SFG_CFG = {
    "min_cosine": 0.40,
    "nli_edge_threshold": 0.50,
    "max_candidate_pairs": 800,
    "source_neighbor_window": 3,       # structural adjacency edges
    "source_neighbor_max_page_gap": 3,
    "canonical_source_order": True,
    "per_page_pair_cap": 24,
    "per_anchor_pair_cap": 8,
    "raw_candidate_pairs": 20000,
    "max_wall_fraction": 0.5,
    "reserve_generation_minutes": 10.0,
    "pair_prefilter": {
        "enabled": True,
        "min_chars": 30,
        "min_bridge_tokens": 2,
        "bridge_cosine_min": 0.50,
        "max_page_gap_unaided": 50,
        "near_duplicate_overlap_cap": 0.75,
        "same_role_overlap_floor": 0.45,
        "penalty_same_role_high_overlap": 0.15,
        "same_page_role_overlap_cap": 0.7,
        "penalty_same_page_high_overlap": 0.20,
    },
    # Generation-first redesign: single-fact questions are the default volume
    # source, so two-fact pairs can now be held to a strict standard without
    # starving the dataset. Both facts must contribute DISTINCT supported answer
    # clauses (answer_contribution) and neither fact alone may entail the edge
    # claim (edge_minimality). Together these enforce a genuine reasoning
    # dependency rather than a page-proximity coincidence.
    "edge_minimality": {
        "enabled": True,
        "single_fact_threshold": 0.85,
        "min_joint_only_margin": 0.05,
    },
    "answer_contribution": {
        "enabled": True,
        "own_entailment_threshold": 0.45,
    },
    "pre_llm_redundancy": {
        "enabled": True,
        "bidirectional_entailment_threshold": 0.86,
        "single_fact_entailment_threshold": 0.94,
    },
    "cache_path": "data/cache/allpdf_tfsfg_cache.db",
}


def _embed_and_index(facts: List[Fact]):
    """Build FAISS index for a list of facts. Returns (embeddings, index, embed_sec)."""
    from rag_gt.vectorstore.embedding import embed_facts
    from rag_gt.vectorstore.faiss_index import FactIndex

    t0 = time.time()
    embeddings = embed_facts(facts)
    index = FactIndex(dim=embeddings.shape[1])
    index.add(embeddings, [f.fact_id for f in facts])
    return embeddings, index, round(time.time() - t0, 2)


def _build_graph(facts: List[Fact], index, llm, doc_type: str, max_pairs: int):
    """Stage 5: build TypedSFG with structural edges."""
    from rag_gt.graph.typed_sfg import TypedSFG

    v16_cfg = {
        "tf_sfg": {**_HYBRID_TF_SFG_CFG, "max_candidate_pairs": max_pairs},
        "_doc_type": doc_type,
        "doc_type_profiles": {},
    }
    sfg = TypedSFG(facts, index, v16_cfg)
    sfg.build(llm)
    return sfg


def _sample_chains(sfg, n_chains: int, depth: int, rng: random.Random) -> List[FactChain]:
    """Stage 5b: walk typed paths to produce multi-hop chains."""
    depth_dist = {depth: n_chains}
    return sfg.walk_typed_paths(depth_dist, rng)


# Edge relation types that express a genuine reasoning dependency (the answer
# needs BOTH facts). Page proximity / topical co-mention ("list", "descriptive",
# "intersection", "comparison") do NOT qualify — those produced the artificial
# two-fact aggregation questions. definition/procedural are kept because a
# definition->application or process->parameter link is a real dependency.
# Narrowed to the relation types where the answer GENUINELY needs both facts and
# the dependency is a logical one, not a shared-document-section coincidence.
# Preview evidence: with "procedural", "definition", "rule", "temporal" included,
# the classifier laundered topical aggregations ("joint prep AND temperature")
# into accepted two-fact pairs. A real WPS dependency is a condition/consequence/
# exception (e.g. "if preheating is not required, use the lowest temperature").
_GENUINE_DEPENDENCY_TYPES = frozenset({
    "causal", "mechanism", "condition", "consequence", "exception",
})


def _build_chains(
    sfg,
    kept_facts: List[Fact],
    n_two_fact: int,
    rng: random.Random,
) -> Tuple[List[FactChain], int, int]:
    """Generation-first chain set.

    Returns (chains, n_single, n_two_fact_genuine):
      - one depth-1 chain per kept fact  → single-fact questions (the default);
      - depth-2 chains, kept ONLY when their edge expresses a genuine reasoning
        dependency (``_GENUINE_DEPENDENCY_TYPES``). Page-proximity edges are
        dropped so no artificial two-fact aggregation question is produced.
    """
    single_chains = [FactChain(fact_ids=[f.fact_id]) for f in kept_facts]
    two_fact_raw = _sample_chains(sfg, n_two_fact, 2, rng)
    two_fact = [
        c for c in two_fact_raw
        if c.chain_edges
        and str(c.chain_edges[0].get("type", "")).lower() in _GENUINE_DEPENDENCY_TYPES
    ]
    return single_chains + two_fact, len(single_chains), len(two_fact)


import re as _re

_FRAGMENT_BULLET_RE = _re.compile(r"^[•\-\*\?\d]+[\.\)]\s*", _re.UNICODE)


def _any_fragment_fact(facts) -> bool:
    """Return True if any fact in the chain is a content-poor fragment.

    Fragment criteria:
    - text under 8 words after stripping whitespace/bullets
    - text starts with a bullet/numbering prefix (pure list item)
    - text is a bare cross-reference sentence (matches 'see also', 'refer to', 'as shown in')

    NOTE: the word-floor was 15, which double-filtered on top of Stage 4 and
    skipped valid short ISO clauses ("If possible, avoid testing etched
    surfaces." = 6 words). Stage 4 (`_relaxed_reject._is_fragment`) now removes
    genuine fragments at the source, so this is a light backstop, not the
    primary fragment filter. Lowered to 8 to match `_RELAXED_MIN_WORDS`.
    On din_iso_15609 this recovered chain yield from 8/45 to the full clean set.
    """
    _XREF_RE = _re.compile(
        r"^(?:see\s+also|refer\s+to|as\s+shown\s+in|an\s+example\s+of|"
        r"for\s+(?:more\s+)?(?:details?|information)\s+see)",
        _re.IGNORECASE,
    )
    for f in facts:
        text = (f.canonical_form or f.text or "").strip()
        stripped = _FRAGMENT_BULLET_RE.sub("", text).strip()
        word_count = len(stripped.split())
        if word_count < 8:
            return True
        if _XREF_RE.match(stripped):
            return True
    return False



_COMPOUND_Q_RE = _re.compile(
    r"\b and (what|how|why|which|when|who|where)\b", _re.IGNORECASE
)
# Reasoning models asked to hit a word budget sometimes number the words inline
# ("What(1) equipment2 control3 ...") and the counting leaks into the question.
_WORD_NUMBERING_RE = _re.compile(r"[A-Za-z]+\(\d+\)|[A-Za-z]{2,}\d+\s+[A-Za-z]{2,}\d+")


def _is_corrupted_question(q: str) -> bool:
    """True when a question contains leaked word-count annotations."""
    if not q:
        return False
    return len(_WORD_NUMBERING_RE.findall(q)) >= 1


def _is_compound_question(q: str) -> bool:
    """True when a question crams two asks together (essay-like, not book-style)."""
    if not q:
        return False
    if len(q.split()) > 26:
        return True
    if _COMPOUND_Q_RE.search(q):
        return True
    if q.count("?") > 1:
        return True
    return False


def _needs_reframe(q: str) -> bool:
    return _is_compound_question(q) or _is_corrupted_question(q)


_TAUTOLOGY_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "and", "or", "but", "not", "this", "that", "these", "those",
    "it", "its", "what", "which", "who", "how", "when", "where", "why",
    "as", "also", "than", "then", "there", "their", "they", "both", "include",
    "includes", "including", "such", "any", "all", "each", "some", "its",
}
# A pair is tautological when the answer adds almost no new content words
# beyond what the question already states. Threshold: fewer than 25% of the
# answer's content words are new (not already in the question).
_TAUTOLOGY_NEW_WORD_FLOOR = 0.25


# Verbs and connectives that carry no answer content on their own. An answer
# whose ONLY new material is drawn from this set has not answered anything --
# it has re-worded the question's own predicate.
_WEAK_NEW_STEMS = frozenset({
    "assist", "help", "aid", "provid", "give", "us", "use", "includ", "appli",
    "relat", "serv", "consist", "compris", "involv", "describ", "specifi",
    "defin", "list", "present", "show", "contain", "cover", "indic", "accord",
    "refer", "correspond", "pertain", "concern", "regard", "address", "state",
    "mention", "note", "set", "mak", "do", "be", "have", "may", "might", "can",
    "could", "shall", "should", "must", "will", "would", "requir", "need",
    "follow", "obtain", "achiev", "result", "occur", "exist", "avail",
})

_STEM_SUFFIXES = (
    "ications", "ication", "ations", "ation", "ements", "ement", "ments",
    "ment", "ances", "ance", "ences", "ence", "ings", "ing", "ions", "ion",
    "ives", "ive", "ers", "er", "als", "al", "ed", "es", "s",
)


def _stem(word: str) -> str:
    """Crude suffix stripper.

    Only needs to make morphological variants collide -- "selection"/"selecting"
    -> "select", "provide"/"provides" -> "provid" -- so an answer that merely
    re-inflects the question's own words is not credited with new information.
    """
    w = word
    changed = True
    while changed and len(w) > 4:
        changed = False
        for suf in _STEM_SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                changed = True
                break
    return w.rstrip("e") if len(w) > 3 else w


def _is_substantive(stem: str) -> bool:
    """A stem that could plausibly BE an answer."""
    if any(ch.isdigit() for ch in stem):
        return True          # a quantity, a threshold, a standard number
    return len(stem) >= 3 and stem not in _WEAK_NEW_STEMS


def _is_tautological(question: str, answer: str) -> bool:
    """True when the answer restates the question with no new information.

    Measures NOVELTY PRESENCE, not a novelty ratio. The earlier ratio form
    ("fewer than 25% of the answer's content words are new") was wrong for
    short factoid answers: replaying it over the 2026-08-01 run rejected
    "What is the minimum required development time?" / "... is 10 minutes."
    because the one new token was diluted by the question's own phrasing.
    Well-formed answers legitimately echo the question; what makes a pair
    circular is that nothing substantive is added at all.

    A pair is tautological when every content word the answer introduces is
    either a morphological variant of a question word or a weak predicate.
    """
    def _content(text: str) -> list[str]:
        out = []
        for w in (text or "").split():
            w = w.lower().strip(".,;:?!\"'()[]{}")
            if not w or w in _TAUTOLOGY_STOPWORDS:
                continue
            # Keep short NUMERIC tokens. The >2-character floor silently
            # discarded "2", "10", "5" -- precisely the tokens that most often
            # ARE the answer ("Level 2", "a type 2 reference block",
            # "10 minutes"), which made those pairs look contentless.
            if len(w) > 2 or any(ch.isdigit() for ch in w):
                out.append(w)
        return out

    q_stems = {_stem(w) for w in _content(question)}
    a_stems = [_stem(w) for w in _content(answer)]
    if not a_stems:
        return False
    new_stems = [s for s in a_stems if s not in q_stems]
    if not new_stems:
        return True
    return not any(_is_substantive(s) for s in new_stems)


def _is_tautological_ratio_legacy(question: str, answer: str) -> bool:
    """Original ratio-based form, retained for reference. Not used as a gate."""
    def _content(text: str) -> set:
        return {
            w.lower().strip(".,;:?!\"'()[]{}") for w in text.split()
            if len(w) > 2 and w.lower().strip(".,;:?!\"'()[]{}") not in _TAUTOLOGY_STOPWORDS
        }

    q_words = _content(question)
    a_words = _content(answer)
    if not a_words:
        return False
    new_in_answer = a_words - q_words
    return len(new_in_answer) / len(a_words) < _TAUTOLOGY_NEW_WORD_FLOOR


_PUNCT_STRIP_RE = _re.compile(r"[^\w\s]")
_WS_RE = _re.compile(r"\s+")


def _normalize_question(question: str) -> str:
    """Canonical key for exact-duplicate question detection.

    The allpdf pipeline generates one question per chain independently and had
    no dedup of any kind, so byte-identical questions shipped in the same
    dataset (18 near-duplicate pairs, several byte-identical, in the
    2026-08-01 run). Mirrors ``gt_pipeline._normalize_question_for_dedup``.
    """
    import unicodedata

    text = unicodedata.normalize("NFKC", question or "").lower()
    return _WS_RE.sub(" ", _PUNCT_STRIP_RE.sub(" ", text)).strip()


def _reject_pair_reason(question: str, answer: str, facts) -> Optional[str]:
    """Pair-level quality gate for the allpdf pipeline.

    Two mechanisms, both of which already existed but neither of which was
    reachable from this pipeline:

    * ``rag_gt.validation.gt_quality`` — the 20-check suite (Q1..Q20). It was
      only ever imported by ``pipeline/gt_pipeline.py`` and
      ``cli/arm_continuation.py``. Replaying it over the 2026-08-01 run's 453
      pairs showed 93 (20.5%) would have been rejected on hard-fail flags.
    * ``_is_tautological`` — implemented, unit-tested, and called from nowhere
      but its own test since the "FIX 5" removal. The removal reasoning held
      for multi-hop pairs (whose circularity came from fabricated edges) but
      the single-fact path, which produces ~99% of pairs, was never covered.

    Only the hard-fail flag set is enforced. The weighted score is deliberately
    NOT used as a threshold here: replaying it showed `quality < 0.7` firing on
    0 of 453 pairs, because the per-check weights are small relative to
    TOTAL_WEIGHT. Gating on an inert number would be theatre.
    """
    from rag_gt.validation.gt_quality import (
        CHECKS,
        GENERIC_SOURCE_RE,
        HARD_FAIL_FLAGS,
    )

    if _is_tautological(question, answer):
        return "tautological_pair"

    # Q4_generic_referent is enforced in SPLIT form here, deliberately.
    #
    # It bundles two unrelated tests. GENERIC_SOURCE_RE ("according to the
    # provided text", "in the given statements") is a genuine defect in any
    # domain and is enforced below. GENERIC_RE, however, fires on a noun list
    # -- method / process / procedure / operation / operator / graph / table --
    # that was tuned for academic prose, where "the method" is vague. In an ISO
    # standard those are the domain's own subjects. Replaying the full check
    # over the 2026-08-01 run rejected 34 pairs, and reading them shows good
    # ones being lost:
    #   "What information can the penetrant method provide about
    #    discontinuities?"  -> rejected on "the ... method"
    #   "How does the drying process for the part utilize forced-air
    #    circulation?"      -> rejected on "the drying process"
    # Both name a real subject and carry strong retrieval anchors.
    # Under-specified questions are still caught by Q1_deictic_no_antecedent
    # (kept below) and by the Stage-4 deictic-opener rule on the fact itself.
    if GENERIC_SOURCE_RE.search(question or ""):
        return "quality_generic_source_frame"

    fact_dicts = []
    for f in facts:
        fact_dicts.append({
            "fact_id": getattr(f, "fact_id", ""),
            "text": getattr(f, "text", "") or "",
            "canonical_form": getattr(f, "canonical_form", "") or "",
            "supporting_spans": getattr(f, "supporting_spans", []) or [],
        })

    for check in CHECKS:
        if check.name not in HARD_FAIL_FLAGS:
            continue
        if check.name == "Q4_generic_referent":
            continue  # enforced in split form above -- see the comment there
        try:
            if check.fn(question, answer, fact_dicts):
                return f"quality_{check.name}"
        except Exception as e:  # a broken check must not silently pass a pair
            logger.warning(f"[pipeline] quality check {check.name} errored: {e}")
    return None


def _run_qgen(
    chains: List[FactChain],
    facts_by_id: Dict[str, Fact],
    llm,
    answer_llm,
    *,
    score_necessity: bool = False,
) -> List[dict]:
    """Stage 7: generate Q&A pairs from fact chains.

    When score_necessity is True, every multi-hop pair gets a leave-one-out
    necessity_score (local NLI, no API cost).
    """
    from rag_gt.generation.questions import generate_question
    from rag_gt.generation.answers import generate_answer, is_abstention
    from rag_gt.allpdf.grounding import is_grounded
    from rag_gt.allpdf.hop_classify import (
        build_source_order_index,
        classify_pair_facts,
    )

    _CONCISE_HINT = (
        "Ask ONE focused question in a single short clause. Do not join two "
        "questions with 'and'; combine the facts into a single relationship "
        "question. Do NOT count or number the words — just write the question."
    )

    order_index = build_source_order_index(list(facts_by_id.values()))

    pairs: List[dict] = []
    n_fragment_skipped = 0
    qgen_times: List[float] = []
    agen_times: List[float] = []
    qgen_fail = 0
    agen_fail = 0
    n_compound_regen = 0
    n_abstain_dropped = 0
    n_ungrounded_dropped = 0
    n_tautological_dropped = 0
    n_quality_dropped = 0
    n_duplicate_dropped = 0
    quality_reasons: Dict[str, int] = {}
    seen_questions: set[str] = set()

    def _accept(question: str, answer: str, chain_facts) -> bool:
        """Pair-level gate + dedup. Mutates the counters above."""
        nonlocal n_quality_dropped, n_duplicate_dropped, n_tautological_dropped
        reason = _reject_pair_reason(question, answer, chain_facts)
        if reason:
            if reason == "tautological_pair":
                n_tautological_dropped += 1
            else:
                n_quality_dropped += 1
            quality_reasons[reason] = quality_reasons.get(reason, 0) + 1
            return False
        key = _normalize_question(question)
        if key in seen_questions:
            n_duplicate_dropped += 1
            return False
        seen_questions.add(key)
        return True

    n_single_pairs = 0
    for chain in chains:
        chain_facts = [facts_by_id[fid] for fid in chain.fact_ids if fid in facts_by_id]
        if not chain_facts:
            continue

        # --- Single-fact path (RD1): the default natural-information-need question.
        if len(chain_facts) == 1:
            fact = chain_facts[0]
            if _any_fragment_fact(chain_facts):
                n_fragment_skipped += 1
                continue
            try:
                question = generate_question([fact], llm)
                if not question or _is_corrupted_question(question):
                    qgen_fail += 1
                    continue
                answer = generate_answer(question, [fact], answer_llm)
                if not answer or is_abstention(answer):
                    n_abstain_dropped += 1
                    continue
                if not is_grounded(answer, [fact]):
                    n_ungrounded_dropped += 1
                    continue
                # Grounding proves the answer is TRUE given the fact; it cannot
                # tell whether the answer is INFORMATIVE. A restatement of a
                # deferring fact is perfectly entailed. Hence the pair gate.
                if not _accept(question, answer, [fact]):
                    continue
                pairs.append({
                    "chain_fact_ids": chain.fact_ids,
                    "question": question,
                    "answer": answer,
                    "depth": 1,
                    "pair_type": "single_fact",
                    "page_spread": 0,
                    "neighbor_distance": 0,
                    "necessity_score": None,
                    "facts": [{
                        "fact_id": fact.fact_id,
                        "text": fact.text,
                        "canonical_form": fact.canonical_form,
                        "self_containment_score": fact.self_containment_score,
                        "page_start": (
                            fact.supporting_spans[0].page_start
                            if fact.supporting_spans else None
                        ),
                    }],
                })
                n_single_pairs += 1
            except Exception as e:
                logger.warning(f"[pipeline] single-fact QGen failed {chain.fact_ids}: {e}")
            continue

        # --- Two-fact path: genuine-dependency pairs only.
        if _any_fragment_fact(chain_facts):
            logger.debug(
                f"[pipeline] skipping chain {chain.fact_ids} — fragment anchor fact"
            )
            n_fragment_skipped += 1
            continue
        try:
            tq = time.time()
            edges = chain.chain_edges if chain.chain_edges else None
            question = generate_question(chain_facts, llm, chain_edges=edges)
            qgen_times.append(round(time.time() - tq, 2))
            if not question:
                qgen_fail += 1
                continue

            # Single-focus framing: a compound/essay-like or word-numbered
            # (CoT-counting leak) question is regenerated once with a concise
            # hint and a little temperature.
            if _needs_reframe(question):
                q2 = generate_question(
                    chain_facts, llm, chain_edges=edges,
                    extra_hint=_CONCISE_HINT, temperature=0.4,
                )
                if q2 and not _needs_reframe(q2):
                    question = q2
                    n_compound_regen += 1
            # Never emit a corrupted (word-numbered) question.
            if _is_corrupted_question(question):
                qgen_fail += 1
                continue

            ta = time.time()
            answer = generate_answer(question, chain_facts, answer_llm)
            agen_times.append(round(time.time() - ta, 2))
            if not answer:
                agen_fail += 1
                continue

            # Answerability gate: if the answer model abstained, the question
            # asked for something the facts do not support. Regenerate the
            # question once (more variation), re-answer; drop the pair if it
            # still abstains — abstentions are noise, not ground truth.
            if is_abstention(answer):
                q2 = generate_question(
                    chain_facts, llm, chain_edges=edges,
                    extra_hint=_CONCISE_HINT, temperature=0.5,
                )
                if q2:
                    a2 = generate_answer(q2, chain_facts, answer_llm)
                    if not is_abstention(a2):
                        question, answer = q2, a2
                if is_abstention(answer):
                    n_abstain_dropped += 1
                    continue

            # Grounding gate (BUG-3): drop a pair whose answer is not entailed by
            # the union of its cited facts (local NLI, no API cost).
            if not is_grounded(answer, chain_facts):
                n_ungrounded_dropped += 1
                logger.debug(
                    f"[pipeline] dropping ungrounded answer for chain {chain.fact_ids}"
                )
                continue

            # The tautology gate was removed in "FIX 5" on the theory that
            # circular pairs were caused by junk facts (Layer A) and fabricated
            # edges (Layer B), both fixed at the source. That held for two-fact
            # pairs, but the single-fact path — ~99% of output — was never
            # covered, so it is restored here for both paths alongside the
            # gt_quality hard-fail checks and dedup.
            if not _accept(question, answer, chain_facts):
                continue

            hop = classify_pair_facts(chain_facts, order_index_by_id=order_index)
            necessity = None
            if score_necessity and hop["is_multi"]:
                from rag_gt.allpdf.necessity import leave_one_out_necessity
                necessity = leave_one_out_necessity(answer, chain_facts)
            pairs.append({
                "chain_fact_ids": chain.fact_ids,
                "question": question,
                "answer": answer,
                "depth": len(chain.fact_ids),
                "pair_type": hop["pair_type"],
                "page_spread": hop["page_spread"],
                "neighbor_distance": hop["neighbor_distance"],
                "necessity_score": (
                    necessity["necessity_score"] if necessity else None
                ),
                "facts": [
                    {
                        "fact_id": f.fact_id,
                        "text": f.text,
                        "canonical_form": f.canonical_form,
                        "self_containment_score": f.self_containment_score,
                        "page_start": (
                            f.supporting_spans[0].page_start
                            if f.supporting_spans else None
                        ),
                    }
                    for f in chain_facts
                ],
            })
        except Exception as e:
            logger.warning(f"[pipeline] QGen failed for chain {chain.fact_ids}: {e}")

    avg_q = round(sum(qgen_times) / max(1, len(qgen_times)), 2)
    avg_a = round(sum(agen_times) / max(1, len(agen_times)), 2)
    logger.info(
        f"[pipeline] Stage 7 done: {len(pairs)} pairs from {len(chains)} chains "
        f"[fragment_skip={n_fragment_skipped} qgen_fail={qgen_fail} agen_fail={agen_fail} "
        f"compound_regen={n_compound_regen} abstain_dropped={n_abstain_dropped} "
        f"ungrounded_dropped={n_ungrounded_dropped} tautological_dropped={n_tautological_dropped} "
        f"quality_dropped={n_quality_dropped} duplicate_dropped={n_duplicate_dropped} "
        f"quality_reasons={quality_reasons}] "
        f"[qgen_avg={avg_q}s agen_avg={avg_a}s "
        f"qgen_total={sum(qgen_times):.1f}s agen_total={sum(agen_times):.1f}s]"
    )
    # Attach observability stats for pipeline_result.json
    _run_qgen.last_obs = {  # type: ignore[attr-defined]
        "n_chains_input": len(chains),
        "n_fragment_skipped": n_fragment_skipped,
        "n_qgen_fail": qgen_fail,
        "n_agen_fail": agen_fail,
        "n_compound_regen": n_compound_regen,
        "n_abstain_dropped": n_abstain_dropped,
        "n_ungrounded_dropped": n_ungrounded_dropped,
        "n_tautological_dropped": n_tautological_dropped,
        "n_quality_dropped": n_quality_dropped,
        "n_duplicate_dropped": n_duplicate_dropped,
        "quality_reasons": dict(quality_reasons),
        "n_pairs_produced": len(pairs),
        "qgen_total_sec": round(sum(qgen_times), 2),
        "agen_total_sec": round(sum(agen_times), 2),
        "qgen_avg_sec": avg_q,
        "agen_avg_sec": avg_a,
    }
    return pairs


def run_doc_pipeline(
    doc_id: str,
    pdf_path: str,
    out_dir: str,
    *,
    dry_run: bool = False,
    max_pairs: int = 400,
    target_chains: int = 20,
    chain_depth: int = 2,
    multihop_chains: int = 0,
    multihop_depths: Tuple[int, ...] = (2,),
    cross_page_bias: float = 2.0,
    min_page_gap: int = 1,
    score_necessity: bool = False,
    llm_chunk_cap: int = 80,
    docling_page_cap: int = 60,
    seed: int = 42,
    extract_only: bool = False,
    extract_workers: Optional[int] = None,
) -> dict:
    """Run the full GRAFT allpdf pipeline for one document.

    dry_run=True: use rule-based extraction (no LLM), skip graph+QGen.
    """
    out = Path(out_dir)
    ckpt = out / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _stage_times: dict = {}   # stage_name -> elapsed seconds at stage end
    rng = random.Random(seed)

    def _mark(name: str) -> None:
        _stage_times[name] = round(time.time() - t0, 1)

    # ---- STAGE 0: Preflight ----
    g0 = VerificationGate("stage0_preflight")
    logger.info(f"[pipeline] {doc_id} — Stage 0: preflight")
    profile = profile_pdf(pdf_path, doc_id)
    save_checkpoint(profile.to_dict(), str(ckpt / "s0_profile.json"))
    g0.check("page_count>0", profile.page_count > 0)
    g0.check("backend_set", bool(profile.recommended_backend))
    logger.info(g0.report())
    _mark("s0_preflight")

    # ---- STAGE 1: Ingest ----
    g1 = VerificationGate("stage1_ingest")
    logger.info(f"[pipeline] {doc_id} — Stage 1: ingest")
    ingest = ingest_document(profile, docling_page_cap=docling_page_cap)
    save_checkpoint(ingest.summary(), str(ckpt / "s1_ingest.json"))
    g1.check("units>0", ingest.n_units > 0 or profile.scanned)
    g1.check("chars>0", ingest.char_count > 0 or profile.scanned)
    g1.check(
        "pages_covered>=90% of page_count",
        profile.scanned or ingest.pages_covered >= 0.9 * profile.page_count,
        f"covered {ingest.pages_covered}/{profile.page_count} pages",
    )
    logger.info(g1.report())
    _mark("s1_ingest")

    # ---- STAGE 2: Chunk ----
    g2 = VerificationGate("stage2_chunk")
    logger.info(f"[pipeline] {doc_id} — Stage 2: chunk")
    chunk_result = agentic_chunk(profile, ingest)
    save_checkpoint(chunk_result.summary(), str(ckpt / "s2_chunk.json"))
    # Save full chunk list for Stage 10 evaluation retrieval.
    save_checkpoint(chunk_result.chunks, str(ckpt / "s2_chunks_full.json"))
    g2.check("chunks>0", len(chunk_result.chunks) > 0)
    g2.check("strategy_set", bool(chunk_result.strategy))
    logger.info(g2.report())
    _mark("s2_chunk")

    if dry_run:
        # Dry-run: validate stages 0-2 only; do rule-based extraction for gate check.
        llm = None
    else:
        from rag_gt.core.llm import get_llm
        llm = get_llm("gt")
        answer_llm = get_llm("answer")

    # ---- STAGE 3: SFU fact extraction ----
    g3 = VerificationGate("stage3_extract")
    logger.info(f"[pipeline] {doc_id} — Stage 3: SFU extraction (llm={'yes' if llm else 'rule-based'})")
    extract = extract_sfu_facts(
        profile, chunk_result, llm=llm,
        llm_chunk_cap=llm_chunk_cap if not dry_run else 0,
        cache_dir=str(ckpt / "s3_chunk_cache") if not dry_run else None,
        workers=extract_workers,
    )
    save_checkpoint(extract.summary(), str(ckpt / "s3_extract.json"))
    g3.check("facts>0", extract.n_facts > 0)
    # Only gate on canonical_rate when LLM extraction was actually used.
    if not dry_run and extract.extraction_path == "llm" and extract.n_facts > 0:
        canon_rate = extract.n_canonical / extract.n_facts
        g3.check("canonical_rate>=0.1", canon_rate >= 0.1,
                 detail=f"got {canon_rate:.2f}")
    logger.info(g3.report())
    _mark("s3_extract")

    # ---- STAGE 4: Adaptive domain filter ----
    g4 = VerificationGate("stage4_filter")
    logger.info(f"[pipeline] {doc_id} — Stage 4: adaptive filter")
    kept_facts, drop_counts = filter_facts_adaptive(extract.facts, profile)
    save_checkpoint(
        {"kept": len(kept_facts), "dropped": sum(drop_counts.values()), "reasons": drop_counts},
        str(ckpt / "s4_filter.json"),
    )
    # Full kept-fact list with provenance (RD3): exact source span, page, chunk
    # and bbox so a fact can be audited against the original PDF without
    # re-running extraction. raw_text is the PRE-rewrite source span.
    def _prov(f):
        sp = f.supporting_spans[0] if f.supporting_spans else None
        return {
            "fact_id": f.fact_id,
            "text": f.text,
            "canonical_form": f.canonical_form,
            "raw_text": getattr(f, "raw_text", None),
            "self_containment_score": f.self_containment_score,
            "self_containment_known": getattr(f, "self_containment_known", False),
            "page_start": getattr(sp, "page_start", None) if sp else None,
            "char_start": getattr(sp, "char_start", None) if sp else None,
            "char_end": getattr(sp, "char_end", None) if sp else None,
            "chunk_id": getattr(sp, "chunk_id", None) if sp else None,
            "bboxes": [b.to_dict() if hasattr(b, "to_dict") else b
                       for b in (getattr(sp, "bboxes", []) or [])] if sp else [],
        }
    save_checkpoint(
        [_prov(f) for f in kept_facts],
        str(ckpt / "s4_facts.json"),
    )
    g4.check("facts_remain>0", len(kept_facts) > 0)
    g4.check("drop_rate<0.9",
             len(kept_facts) / max(1, extract.n_facts) >= 0.1,
             detail=f"{len(kept_facts)}/{extract.n_facts} kept")
    logger.info(g4.report())
    _mark("s4_filter")

    if extract_only:
        elapsed = time.time() - t0
        result = {
            "doc_id": doc_id,
            "extract_only": True,
            "stages_completed": ["s0", "s1", "s2", "s3", "s4"],
            "n_facts_after_filter": len(kept_facts),
            "s3_extract": extract.summary(),
            "s4_drop_reasons": drop_counts,
            "pipeline_passed": all(g.passed for g in [g0, g1, g2, g3, g4]),
            "wall_sec": round(elapsed, 1),
        }
        save_checkpoint(result, str(out / "pipeline_result.json"))
        logger.info(f"[pipeline] {doc_id} — extract-only done: {len(kept_facts)} facts")
        return result

    if dry_run:
        elapsed = time.time() - t0
        result = {
            "doc_id": doc_id,
            "dry_run": True,
            "stages_completed": ["s0", "s1", "s2", "s3", "s4"],
            "n_facts_after_filter": len(kept_facts),
            "gates": {
                "s0": g0.to_dict(), "s1": g1.to_dict(),
                "s2": g2.to_dict(), "s3": g3.to_dict(), "s4": g4.to_dict(),
            },
            "pipeline_passed": all(g.passed for g in [g0, g1, g2, g3, g4]),
            "wall_sec": round(elapsed, 1),
            "stage_times_sec": _stage_times,
        }
        save_checkpoint(result, str(out / "pipeline_result.json"))
        return result

    # ---- STAGE 5: Hybrid candidate graph ----
    # Scale candidate budget with document size: ~3 candidates per kept fact,
    # capped at max_pairs. A 10-page doc with 50 facts → 150 candidates max;
    # a 100-page doc with 500 facts → max_pairs ceiling applies.
    n_facts = len(kept_facts)
    effective_max_pairs = min(max_pairs, max(60, n_facts * 3))
    g5 = VerificationGate("stage5_graph")
    logger.info(
        f"[pipeline] {doc_id} — Stage 5: hybrid candidate graph "
        f"(facts={n_facts}, candidate_budget={effective_max_pairs})"
    )
    facts_by_id = {f.fact_id: f for f in kept_facts}
    _, index, embed_sec = _embed_and_index(kept_facts)
    logger.info(f"[pipeline] {doc_id} — Stage 5 embedding: {embed_sec}s for {n_facts} facts")
    sfg = _build_graph(kept_facts, index, answer_llm, profile.doc_type_guess, effective_max_pairs)
    save_checkpoint(
        {"n_facts": n_facts, "candidate_budget": effective_max_pairs,
         "edge_count": sfg.edge_count, "classified_pairs": sfg.classified_pairs},
        str(ckpt / "s5_graph.json"),
    )
    g5.check("edges>=1", sfg.edge_count >= 1, detail=f"got {sfg.edge_count}")
    g5.check("anchor_facts>=1", len(sfg.adj) >= 1)
    logger.info(g5.report())
    _mark("s5_graph")

    # ---- STAGE 6: Chain sampling ----
    g6 = VerificationGate("stage6_chains")
    logger.info(f"[pipeline] {doc_id} — Stage 6: building single-fact + genuine-dependency chains")
    # Generation-first: one single-fact chain per kept fact (default questions)
    # + genuine-dependency two-fact chains only.
    chains, n_single, n_two_fact = _build_chains(sfg, kept_facts, target_chains, rng)
    logger.info(
        f"[pipeline] {doc_id} — Stage 6: {n_single} single-fact + {n_two_fact} "
        f"genuine two-fact chains"
    )
    n_local = len(chains)
    n_multihop = n_two_fact
    if multihop_chains > 0:
        # Phase 2: dedicated cross-page multi-hop pass over the same built graph.
        local_keys = {tuple(c.fact_ids) for c in chains}
        depths = tuple(multihop_depths) or (2,)
        per_depth = max(1, multihop_chains // len(depths))
        depth_dist = {d: per_depth for d in depths}
        logger.info(
            f"[pipeline] {doc_id} — Stage 6b: cross-page multi-hop pass "
            f"depths={depths} per_depth={per_depth} min_page_gap={min_page_gap}"
        )
        mh = sfg.walk_typed_paths(
            depth_dist, rng,
            cross_page_bias=cross_page_bias,
            require_cross_page=True,
            min_page_gap=min_page_gap,
        )
        new_mh = [c for c in mh if tuple(c.fact_ids) not in local_keys]
        n_multihop = len(new_mh)
        chains = chains + new_mh
    save_checkpoint(
        {"n_chains": len(chains), "target": target_chains, "depth": chain_depth,
         "n_local": n_local, "n_multihop": n_multihop},
        str(ckpt / "s6_chains.json"),
    )
    g6.check("chains>=1", len(chains) >= 1, detail=f"got {len(chains)}")
    logger.info(g6.report())
    _mark("s6_chains")

    # ---- STAGE 7: QGen ----
    g7 = VerificationGate("stage7_qgen")
    logger.info(f"[pipeline] {doc_id} — Stage 7: question generation")
    pairs = _run_qgen(
        chains, facts_by_id, llm, answer_llm,
        score_necessity=score_necessity,
    )
    save_checkpoint(pairs, str(ckpt / "s7_pairs.json"))
    g7.check("pairs>=1", len(pairs) >= 1, detail=f"got {len(pairs)}")
    logger.info(g7.report())
    _mark("s7_qgen")

    # ---- Final result ----
    elapsed = time.time() - t0
    all_gates = [g0, g1, g2, g3, g4, g5, g6, g7]

    # Per-stage durations (difference between consecutive marks).
    _stage_order = ["s0_preflight", "s1_ingest", "s2_chunk", "s3_extract",
                    "s4_filter", "s5_graph", "s6_chains", "s7_qgen"]
    prev = 0.0
    stage_durations: dict = {}
    for s in _stage_order:
        end = _stage_times.get(s, prev)
        stage_durations[s] = round(end - prev, 1)
        prev = end

    qgen_obs = getattr(_run_qgen, "last_obs", {})
    nli_obs = sfg.l0_stats
    result = {
        "doc_id": doc_id,
        "dry_run": False,
        "stages_completed": ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"],
        "n_facts_after_filter": n_facts,
        "candidate_budget": effective_max_pairs,
        "n_edges": sfg.edge_count,
        "n_chains": len(chains),
        "n_pairs": len(pairs),
        "pipeline_passed": all(g.passed for g in all_gates),
        "gates": {f"s{i}": g.to_dict() for i, g in enumerate(all_gates)},
        "wall_sec": round(elapsed, 1),
        "stage_times_sec": _stage_times,
        "stage_durations_sec": stage_durations,
        # --- Detailed observability ---
        "obs": {
            "s3_extract": extract.summary(),
            "s5_embed_sec": embed_sec,
            "s5_nli": {
                "candidate_budget": effective_max_pairs,
                "pairs_attempted": nli_obs.get("nli_pairs_attempted", 0),
                "nli_total_sec": nli_obs.get("nli_total_sec", 0),
                "nli_avg_sec_per_pair": nli_obs.get("nli_avg_sec_per_pair", 0),
                "acceptance_rate": nli_obs.get("nli_acceptance_rate", 0),
                "edges_accepted": sfg.edge_count,
                "pairs_skipped_redundant": nli_obs.get("pre_llm_redundant_pairs_skipped", 0),
            },
            "s6_chains": {
                "n_local": n_local,
                "n_multihop": n_multihop,
                "total": len(chains),
            },
            "s7_qgen": qgen_obs,
        },
    }
    save_checkpoint(result, str(out / "pipeline_result.json"))
    _write_report(result, pairs, str(out / "pipeline_report.md"))

    # Per-stage timing table in the log.
    timing_rows = "  ".join(
        f"{s.split('_')[0]}={stage_durations[s]:.0f}s"
        for s in _stage_order
    )
    logger.info(
        f"[pipeline] {doc_id}: DONE — "
        f"{len(pairs)} pairs, {sfg.edge_count} edges, {elapsed:.1f}s total\n"
        f"  Timing: {timing_rows}"
    )
    return result


def _write_report(result: dict, pairs: list, path: str) -> None:
    lines = [
        f"# GRAFT All-PDF Pipeline Report — {result['doc_id']}", "",
        f"- Dry run: {result['dry_run']}",
        f"- Facts after filter: {result['n_facts_after_filter']}",
    ]
    if not result["dry_run"]:
        lines += [
            f"- Edges: {result['n_edges']}",
            f"- Chains: {result['n_chains']}",
            f"- Q&A pairs: {result['n_pairs']}",
        ]
    lines += [
        f"- Wall time: {result['wall_sec']}s",
        f"- **Pipeline {'PASS' if result['pipeline_passed'] else 'FAIL'}**", "",
    ]
    if result.get("stage_durations_sec"):
        lines += ["## Stage timing", "", "| Stage | Duration |", "|-------|----------|"]
        for stage, dur in result["stage_durations_sec"].items():
            mins = dur / 60
            lines.append(f"| {stage} | {dur:.0f}s ({mins:.1f}m) |")
        lines.append("")
    lines += ["## Gate results", ""]
    for key, g in result["gates"].items():
        lines.append(f"### {g['stage']} — {'PASS' if g['passed'] else 'FAIL'}")
        for c in g["checks"]:
            lines.append(f"- [{'x' if c['passed'] else ' '}] {c['name']}"
                         + (f" — {c['detail']}" if c.get("detail") else ""))
        lines.append("")
    if pairs:
        lines += ["## Sample Q&A Pairs (first 3)", ""]
        for p in pairs[:3]:
            lines += [
                f"**Q:** {p['question']}",
                f"**A:** {p['answer']}",
                f"Depth: {p['depth']} | Facts: {[f['fact_id'] for f in p['facts']]}",
                "",
            ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="GRAFT all-PDF pipeline")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--pdf", required=True, help="Path to source PDF")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Run stages 0-4 only (no LLM / no graph / no QGen)")
    ap.add_argument("--max-pairs", type=int, default=400)
    ap.add_argument("--target-chains", type=int, default=20)
    ap.add_argument("--chain-depth", type=int, default=2)
    ap.add_argument("--multihop-chains", type=int, default=0,
                    help="Phase 2: extra cross-page multi-hop chains to sample")
    ap.add_argument("--multihop-depths", type=str, default="2",
                    help="comma-separated depths for the multi-hop pass, e.g. 2,3")
    ap.add_argument("--cross-page-bias", type=float, default=2.0,
                    help="walk-weight multiplier for cross-page edges")
    ap.add_argument("--min-page-gap", type=int, default=1,
                    help="minimum page gap to count a step as cross-page")
    ap.add_argument("--score-necessity", action="store_true",
                    help="compute leave-one-out necessity for multi-hop pairs")
    ap.add_argument("--llm-chunk-cap", type=int, default=80)
    ap.add_argument("--extract-workers", type=int, default=None,
                    help="Stage-3 LLM extraction parallelism (BUG-A). "
                         "Default: performance.max_concurrent_llm_calls. Use 1 for "
                         "the sequential path.")
    ap.add_argument("--extract-only", action="store_true",
                    help="Run Stages 0-4 with LLM then stop (writes s4_facts). "
                         "Per-chunk extraction is cached, so re-running resumes; "
                         "use scripts/resume_from_s4.py for Stages 5-7.")
    ap.add_argument("--docling-page-cap", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = args.out_dir or f"data/test_corpus_allpdf/pipeline_run/{args.doc_id}"
    multihop_depths = tuple(
        int(x) for x in str(args.multihop_depths).split(",") if x.strip()
    ) or (2,)
    result = run_doc_pipeline(
        args.doc_id, args.pdf, out_dir,
        dry_run=args.dry_run,
        max_pairs=args.max_pairs,
        target_chains=args.target_chains,
        chain_depth=args.chain_depth,
        multihop_chains=args.multihop_chains,
        multihop_depths=multihop_depths,
        cross_page_bias=args.cross_page_bias,
        min_page_gap=args.min_page_gap,
        score_necessity=args.score_necessity,
        llm_chunk_cap=args.llm_chunk_cap,
        docling_page_cap=args.docling_page_cap,
        seed=args.seed,
        extract_only=args.extract_only,
        extract_workers=args.extract_workers,
    )
    status = "PASS" if result["pipeline_passed"] else "FAIL"
    print(f"\nPipeline {status} — {args.doc_id}")
    if not args.dry_run:
        print(f"  {result.get('n_pairs', 0)} Q&A pairs generated")
    print(f"  Report: {out_dir}/pipeline_report.md")
    return 0 if result["pipeline_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
