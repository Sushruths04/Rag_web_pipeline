"""Typed Fact Sub-Graph (TF-SFG) builder and typed-path walker (P1).

Replaces cosine-only chain sampling with walks over NLI-verified typed edges.
Chains produced here carry `chain_edges` metadata so the pipeline can store
them in the v16 QuestionGT row without re-deriving them.

Integration point: called from pipeline/gt_pipeline.py when v16 is enabled,
after the FAISS index is built, before chain sampling.
"""

from __future__ import annotations

import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from rag_gt.core.llm import APIError
from rag_gt.core.types import Fact, FactChain
from rag_gt.budget.adaptive_budget import DocBudget
from rag_gt.graph.edge_classifier import EdgeRecord, classify_edge
from rag_gt.graph.pair_prefilter import PairPrefilterConfig, prefilter_pair
from rag_gt.graph.pair_scheduler import SchedulerReport, schedule_candidate_pairs
from rag_gt.observability.cost_tracker import LiveCallBudgetExceeded
from rag_gt.validation.nli_check import nli_batch
from rag_gt.vectorstore.faiss_index import FactIndex

# Default pre-filter cosine: lower than the main pipeline threshold (0.55) so
# typed edges can bridge semantically adjacent but not identical facts.
_DEFAULT_MIN_COSINE = 0.45
# Top-K neighbours checked per fact for candidate edge pairs.
_CANDIDATE_K = 30

_ROLE_PAIR_BONUS = {
    # Keep the unverified role prior bounded: it may break a near-tie, but
    # cannot displace semantically stronger pairs before edge verification.
    ("definition", "condition"): 0.08,
    ("definition", "rule"): 0.08,
    ("condition", "definition"): 0.05,
    ("condition", "consequence"): 0.08,
    ("rule", "exception"): 0.08,
    ("rule", "constraint"): 0.08,
    ("rule", "consequence"): 0.07,
    ("condition", "rule"): 0.06,
    ("definition", "consequence"): 0.05,
    ("definition", "example"): 0.03,
}
_EDGE_TYPE_WALK_PRIOR = {
    "mechanism": 1.00,
    "causal": 0.95,
    "procedural": 0.90,
    "temporal": 0.82,
    "quantitative": 0.80,
    "contrast": 0.78,
    "comparison": 0.76,
    "intersection": 0.72,
    "condition": 0.70,
    "consequence": 0.70,
    "rule": 0.68,
    "exception": 0.66,
    "constraint": 0.64,
    "definition": 0.38,
    "descriptive": 0.32,
    "list": 0.20,
}
_ROLE_PAIR_WALK_PRIOR = {
    ("definition", "condition"): 1.00,
    ("definition", "rule"): 0.86,
    ("condition", "definition"): 0.74,
    ("condition", "consequence"): 0.76,
    ("definition", "consequence"): 0.68,
    ("definition", "example"): 0.56,
    ("definition", "definition"): 0.38,
}
_STOPWORDS = {
    "about", "above", "after", "again", "also", "because", "before", "between",
    "from", "have", "into", "more", "most", "only", "other", "same", "some",
    "such", "than", "that", "their", "then", "there", "these", "this", "those",
    "through", "under", "using", "very", "when", "where", "which", "while",
    "with", "would",
}


def _fact_page(fact: Fact) -> Optional[int]:
    for span in fact.supporting_spans:
        if span.page_start is not None:
            return int(span.page_start)
    return None


def _page_key(fact: Fact) -> int:
    page = _fact_page(fact)
    return page if page is not None else -1


def _role_bonus(a: Fact, b: Fact) -> float:
    pair = (str(a.role), str(b.role))
    if pair in _ROLE_PAIR_BONUS:
        return _ROLE_PAIR_BONUS[pair]
    if a.role != b.role:
        return 0.03
    return 0.0


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _lexical_bonus(a: Fact, b: Fact) -> float:
    ta = _tokens(a.text)
    tb = _tokens(b.text)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / max(1, min(len(ta), len(tb)))
    return min(0.08, overlap * 0.08)


def _edge_walk_weight(edge: EdgeRecord) -> float:
    """Prioritize edges likely to produce genuinely compositional questions."""
    relation_prior = _EDGE_TYPE_WALK_PRIOR.get(str(edge.type).lower(), 0.40)
    margin = max(0.0, min(1.0, float(getattr(edge, "joint_only_margin", 0.0) or 0.0)))
    nli = max(0.0, min(1.0, float(edge.nli_score or 0.0)))
    contribution = max(
        0.0,
        min(1.0, float(getattr(edge, "answer_contribution_score", 0.0) or 0.0)),
    )
    return max(
        0.001,
        (0.30 * nli) + (0.25 * margin) + (0.30 * contribution) + (0.15 * relation_prior),
    )


def _role_pair_walk_prior(a: Fact, b: Fact) -> float:
    pair = (str(a.role or "").strip().lower(), str(b.role or "").strip().lower())
    if pair in _ROLE_PAIR_WALK_PRIOR:
        return _ROLE_PAIR_WALK_PRIOR[pair]
    if pair[0] != pair[1]:
        return 0.50
    return 0.34


class TypedSFG:
    """Typed Fact Sub-Graph for one document.

    Attributes:
        adj: adjacency list — adj[src_fact_id] = List[EdgeRecord]
        edge_map: (src, dst) → EdgeRecord for O(1) chain-edge lookup
    """

    def __init__(
        self,
        facts: List[Fact],
        index: FactIndex,
        v16_cfg: Optional[dict] = None,
        *,
        v16_2_enabled: Optional[bool] = None,
        doc_budget: Optional[DocBudget] = None,
    ) -> None:
        self.facts: Dict[str, Fact] = {f.fact_id: f for f in facts}
        self.index = index
        self.v16_cfg = v16_cfg or {}
        cfg = self.v16_cfg.get("tf_sfg", {})
        v16_2_cfg = self.v16_cfg.get("v16_2", {}) or {}
        self.v16_2_enabled = (
            bool(v16_2_cfg.get("enabled", False))
            if v16_2_enabled is None
            else bool(v16_2_enabled)
        )
        self.doc_budget = doc_budget
        # --- doc-type profile overrides global tf_sfg defaults ---
        doc_type = str(self.v16_cfg.get("_doc_type", "")).upper()
        profile_cfg = self.v16_cfg.get("doc_type_profiles", {}).get(doc_type, {})
        profile_tf_sfg = profile_cfg.get("tf_sfg", {})
        # merge: global defaults → global tf_sfg config → profile override
        self.min_cosine: float = float(
            profile_tf_sfg.get("min_cosine", cfg.get("min_cosine", _DEFAULT_MIN_COSINE))
        )
        self.max_pairs: int = int(
            profile_tf_sfg.get("max_candidate_pairs", cfg.get("max_candidate_pairs", 5000))
        )
        if self.v16_2_enabled and doc_budget is not None:
            self.max_pairs = int(doc_budget.tf_sfg_pairs)
        self.raw_candidate_pairs: int = int(cfg.get("raw_candidate_pairs", 5000))
        self.source_neighbor_window: int = int(
            profile_tf_sfg.get("source_neighbor_window", cfg.get("source_neighbor_window", 0))
        )
        self.source_neighbor_max_page_gap: int = int(
            profile_tf_sfg.get(
                "source_neighbor_max_page_gap",
                cfg.get("source_neighbor_max_page_gap", 2),
            )
        )
        self.canonical_source_order: bool = bool(
            profile_tf_sfg.get(
                "canonical_source_order",
                cfg.get("canonical_source_order", True),
            )
        )
        self.per_page_pair_cap: int = int(cfg.get("per_page_pair_cap", 12))
        self.per_anchor_pair_cap: int = int(cfg.get("per_anchor_pair_cap", 6))
        self.nli_threshold: float = float(
            profile_tf_sfg.get("nli_edge_threshold", cfg.get("nli_edge_threshold", 0.55))
        )
        self.allow_low_nli_with_quote: bool = bool(
            profile_tf_sfg.get(
                "allow_low_nli_with_quote",
                cfg.get("allow_low_nli_with_quote", False),
            )
        )
        self.low_nli_floor: float = float(
            profile_tf_sfg.get("low_nli_floor", cfg.get("low_nli_floor", 0.0))
        )
        edge_min_cfg = cfg.get("edge_minimality", {}) or {}
        profile_edge_min_cfg = profile_tf_sfg.get("edge_minimality", {}) or {}
        resolved_edge_min_cfg = {**edge_min_cfg, **profile_edge_min_cfg}
        self.edge_minimality_enabled: bool = bool(
            resolved_edge_min_cfg.get("enabled", False)
        )
        self.edge_single_fact_threshold: float = float(
            resolved_edge_min_cfg.get("single_fact_threshold", 0.85)
        )
        self.edge_min_joint_only_margin: float = float(
            resolved_edge_min_cfg.get("min_joint_only_margin", 0.05)
        )
        contribution_cfg = cfg.get("answer_contribution", {}) or {}
        profile_contribution_cfg = profile_tf_sfg.get("answer_contribution", {}) or {}
        resolved_contribution_cfg = {**contribution_cfg, **profile_contribution_cfg}
        self.answer_contribution_enabled: bool = bool(
            resolved_contribution_cfg.get("enabled", False)
        )
        self.contribution_own_threshold: float = float(
            resolved_contribution_cfg.get("own_entailment_threshold", 0.45)
        )
        redundancy_cfg = cfg.get("pre_llm_redundancy", {}) or {}
        profile_redundancy_cfg = profile_tf_sfg.get("pre_llm_redundancy", {}) or {}
        resolved_redundancy_cfg = {**redundancy_cfg, **profile_redundancy_cfg}
        self.pre_llm_redundancy_enabled: bool = bool(
            resolved_redundancy_cfg.get("enabled", False)
        )
        self.pre_llm_redundancy_threshold: float = float(
            resolved_redundancy_cfg.get("bidirectional_entailment_threshold", 0.86)
        )
        self.pre_llm_single_fact_threshold: float = float(
            resolved_redundancy_cfg.get("single_fact_entailment_threshold", 1.01)
        )
        self.progress_interval: int = max(1, int(cfg.get("progress_interval", 25)))
        self.max_consecutive_api_errors: int = max(
            1,
            int(cfg.get("max_consecutive_api_errors", 3)),
        )
        self.scheduler_report: Optional[SchedulerReport] = None
        pair_prefilter_raw = (
            profile_tf_sfg.get("pair_prefilter")
            or cfg.get("pair_prefilter")
            or v16_2_cfg.get("pair_prefilter")
        )
        self.pair_prefilter_cfg = PairPrefilterConfig.from_dict(
            pair_prefilter_raw
        )
        edge_canon_cfg = v16_2_cfg.get("edge_canonicalize", {}) or {}
        self.edge_canonicalize_enabled = self.v16_2_enabled and bool(
            edge_canon_cfg.get("enabled", True)
        )
        self.edge_canonicalize_map = edge_canon_cfg.get("map")
        self.l0_stats: dict = {}
        self.classified_pairs: int = 0
        self.build_timed_out: bool = False
        self.build_budget_exhausted: bool = False
        self.redundant_pairs_skipped: int = 0
        self.bidirectional_redundant_pairs_skipped: int = 0
        self.single_fact_entailed_pairs_skipped: int = 0
        self.adj: Dict[str, List[EdgeRecord]] = defaultdict(list)
        self.edge_map: Dict[Tuple[str, str], EdgeRecord] = {}
        self._built = False

    # ---------- build ----------

    def build(self, llm: object, *, deadline: Optional[float] = None) -> None:
        """Enumerate candidate pairs and classify all edges.

        Candidates are found via the FAISS index: for each fact, find its top-K
        neighbours with cosine >= min_cosine, then classify each pair.
        """
        fact_ids = list(self.facts.keys())
        if not fact_ids:
            self._built = True
            return

        logger.info(f"[TF-SFG] Building typed graph over {len(fact_ids)} facts...")

        pairs = self._candidate_pairs(fact_ids)
        self.classified_pairs = 0
        self.redundant_pairs_skipped = 0
        self.bidirectional_redundant_pairs_skipped = 0
        self.single_fact_entailed_pairs_skipped = 0

        logger.info(f"[TF-SFG] Classifying {len(pairs)} candidate pairs...")
        accepted = 0
        edge_counter = [0]
        consecutive_api_errors = 0
        _t_nli_start = time.time()
        _pair_times: list = []
        for i, (fact_a, fact_b) in enumerate(pairs, start=1):
            if deadline is not None and time.time() >= deadline:
                self.build_timed_out = True
                logger.warning(
                    "[TF-SFG] Graph-stage deadline reached after "
                    f"{i - 1}/{len(pairs)} pairs; continuing with "
                    f"{accepted} accepted edges"
                )
                break
            if self.pre_llm_redundancy_enabled:
                text_a = (fact_a.canonical_form or fact_a.text).strip()
                text_b = (fact_b.canonical_form or fact_b.text).strip()
                try:
                    ab_score, ba_score = nli_batch(
                        [(text_a, text_b), (text_b, text_a)]
                    )
                    if (
                        ab_score >= self.pre_llm_redundancy_threshold
                        and ba_score >= self.pre_llm_redundancy_threshold
                    ):
                        self.redundant_pairs_skipped += 1
                        self.bidirectional_redundant_pairs_skipped += 1
                        continue
                    if max(ab_score, ba_score) >= self.pre_llm_single_fact_threshold:
                        self.redundant_pairs_skipped += 1
                        self.single_fact_entailed_pairs_skipped += 1
                        continue
                except Exception as e:
                    logger.debug(
                        "[TF-SFG] Pre-LLM redundancy NLI failed for "
                        f"{fact_a.fact_id}->{fact_b.fact_id}: {e}"
                    )
            try:
                self.classified_pairs += 1
                _tp = time.time()
                record = classify_edge(
                    fact_a,
                    fact_b,
                    llm,  # type: ignore[arg-type]
                    edge_counter=edge_counter,
                    nli_threshold=self.nli_threshold,
                    canonicalize_edges=self.edge_canonicalize_enabled,
                    canonical_map=self.edge_canonicalize_map,
                    allow_low_nli_with_quote=self.allow_low_nli_with_quote,
                    low_nli_floor=self.low_nli_floor,
                    edge_minimality_enabled=self.edge_minimality_enabled,
                    edge_single_fact_threshold=self.edge_single_fact_threshold,
                    edge_min_joint_only_margin=self.edge_min_joint_only_margin,
                    answer_contribution_enabled=self.answer_contribution_enabled,
                    contribution_own_threshold=self.contribution_own_threshold,
                )
                _pair_times.append(round(time.time() - _tp, 3))
                consecutive_api_errors = 0
            except LiveCallBudgetExceeded as e:
                self.classified_pairs -= 1
                self.build_budget_exhausted = True
                logger.warning(f"[TF-SFG] {e} Continuing with {accepted} accepted edges")
                break
            except APIError as e:
                consecutive_api_errors += 1
                logger.warning(
                    f"[TF-SFG] API error while classifying pair {i}/{len(pairs)} "
                    f"({fact_a.fact_id}->{fact_b.fact_id}); "
                    f"consecutive_api_errors={consecutive_api_errors}/"
                    f"{self.max_consecutive_api_errors}: {e}"
                )
                if consecutive_api_errors >= self.max_consecutive_api_errors:
                    raise RuntimeError(
                        "[TF-SFG] Aborting edge classification after "
                        f"{consecutive_api_errors} consecutive API errors. "
                        "Retry when the LLM endpoint is stable; transient API failures "
                        "were not cached."
                    ) from e
                continue

            if record is not None:
                self.adj[record.src].append(record)
                self.edge_map[(record.src, record.dst)] = record
                accepted += 1
            if i % self.progress_interval == 0 or i == len(pairs):
                logger.info(
                    f"[TF-SFG] Progress: classified {i}/{len(pairs)} pairs; "
                    f"accepted={accepted}; sources={len(self.adj)}"
                )

        nli_total = time.time() - _t_nli_start
        avg_pair = round(nli_total / max(1, len(_pair_times)), 3)
        self.l0_stats["pre_llm_redundant_pairs_skipped"] = self.redundant_pairs_skipped
        self.l0_stats["pre_llm_bidirectional_pairs_skipped"] = (
            self.bidirectional_redundant_pairs_skipped
        )
        self.l0_stats["pre_llm_single_fact_entailed_pairs_skipped"] = (
            self.single_fact_entailed_pairs_skipped
        )
        self.l0_stats["nli_total_sec"] = round(nli_total, 2)
        self.l0_stats["nli_avg_sec_per_pair"] = avg_pair
        self.l0_stats["nli_pairs_attempted"] = len(_pair_times)
        self.l0_stats["nli_acceptance_rate"] = round(accepted / max(1, len(_pair_times)), 3)
        self._built = True
        logger.info(
            f"[TF-SFG] Built: {accepted}/{len(pairs)} edges accepted "
            f"from {self.classified_pairs} attempted pairs "
            f"({len(self.adj)} source facts with outgoing edges) "
            f"[NLI: {nli_total:.1f}s total, {avg_pair}s/pair, "
            f"accept={self.l0_stats['nli_acceptance_rate']:.0%}]"
        )

    def _candidate_pairs(self, fact_ids: List[str]) -> List[Tuple[Fact, Fact]]:
        """Delegate to shared pair_scheduler, then apply v16_2 prefilter if enabled."""
        facts = [self.facts[fid] for fid in fact_ids if fid in self.facts]
        pair_prefilter_fn = None
        if self.pair_prefilter_cfg.enabled:
            pair_prefilter_fn = (
                lambda fa, fb, score: prefilter_pair(fa, fb, score, self.pair_prefilter_cfg)
            )
        selected_triples, report = schedule_candidate_pairs(
            facts,
            self.index,
            min_cosine=self.min_cosine,
            candidate_k=_CANDIDATE_K,
            raw_candidate_pairs=self.raw_candidate_pairs,
            max_pairs=self.max_pairs,
            per_page_pair_cap=self.per_page_pair_cap,
            per_anchor_pair_cap=self.per_anchor_pair_cap,
            role_bonus_fn=_role_bonus,
            lexical_bonus_fn=_lexical_bonus,
            pair_prefilter_fn=pair_prefilter_fn,
            source_neighbor_window=self.source_neighbor_window,
            source_neighbor_max_page_gap=self.source_neighbor_max_page_gap,
            canonical_source_order=self.canonical_source_order,
        )
        self.scheduler_report = report
        self.l0_stats = dict(report.prefilter_stats)

        return [(fa, fb) for fa, fb, _s in selected_triples]

    # ---------- query ----------

    @property
    def edge_count(self) -> int:
        return len(self.edge_map)

    def edges_for_chain(self, fact_ids: List[str]) -> List[dict]:
        """Return the edge records that connect consecutive facts in a chain."""
        out: List[dict] = []
        for i in range(len(fact_ids) - 1):
            key = (fact_ids[i], fact_ids[i + 1])
            rec = self.edge_map.get(key)
            if rec is not None:
                out.append(rec.to_dict())
        return out

    # ---------- path walker ----------

    def walk_typed_paths(
        self,
        depth_dist: Dict[int, int],
        rng: random.Random,
        *,
        cross_page_bias: float = 0.0,
        require_cross_page: bool = False,
        min_page_gap: int = 1,
    ) -> List[FactChain]:
        """Sample FactChain objects by walking typed edges.

        For each (depth, target_count) in depth_dist, sample up to target_count
        chains of the given depth. Chains that cannot be completed via typed
        edges at any step are discarded.

        Cross-page controls (Phase 2 multi-hop pass — see
        docs/allpdf_singlehop_multihop_redesign_plan_20260622.md §5a):
          - cross_page_bias: multiply an edge's walk weight by (1 + bias) when
            it steps to a fact whose page differs by >= min_page_gap. Soft
            preference; 0.0 reproduces the original local-biased walk.
          - require_cross_page: hard-filter each hop to cross-page steps only.
            A walk that cannot make a cross-page step at any hop is discarded
            (reduces yield on sparse graphs but never fabricates edges).
          - min_page_gap: minimum |page(dst) - page(src)| to count as cross-page.

        Returns chains with `chain_edges` populated for downstream use.
        Falls back gracefully when the graph is too sparse.
        """
        if not self._built:
            raise RuntimeError("TypedSFG.build() must be called before walk_typed_paths()")

        out: List[FactChain] = []
        seen: set = set()

        # Facts that have at least one outgoing edge are eligible anchors.
        anchor_ids = [fid for fid in self.adj if self.adj[fid]]
        if not anchor_ids:
            logger.warning("[TF-SFG] No outgoing edges found; typed walk will yield 0 chains")
            return []

        for depth, target in sorted(depth_dist.items()):
            chains_at_depth = 0
            attempts = 0
            # Cross-page walks fail more often, so allow more attempts.
            attempt_mult = 60 if require_cross_page else 20
            max_attempts = target * attempt_mult

            while chains_at_depth < target and attempts < max_attempts:
                attempts += 1
                chain = self._walk_one(
                    depth, anchor_ids, rng,
                    cross_page_bias=cross_page_bias,
                    require_cross_page=require_cross_page,
                    min_page_gap=min_page_gap,
                )
                if chain is None:
                    continue
                key = tuple(chain.fact_ids)
                if key in seen:
                    continue
                seen.add(key)
                out.append(chain)
                chains_at_depth += 1

            if chains_at_depth < target:
                logger.debug(
                    f"[TF-SFG] depth={depth}: yielded {chains_at_depth}/{target} chains "
                    f"(sparse graph or low yield)"
                )

        rng.shuffle(out)
        return out

    def _walk_one(
        self,
        depth: int,
        anchor_ids: List[str],
        rng: random.Random,
        *,
        cross_page_bias: float = 0.0,
        require_cross_page: bool = False,
        min_page_gap: int = 1,
    ) -> Optional[FactChain]:
        """Attempt a single depth-d walk from a random anchor."""
        anchor_id = rng.choice(anchor_ids)
        fact_ids = [anchor_id]
        edges: List[EdgeRecord] = []
        visited = {anchor_id}

        for _ in range(depth - 1):
            current = fact_ids[-1]
            candidates = [
                e for e in self.adj.get(current, [])
                if e.dst not in visited
                and e.dst in self.facts
            ]
            if not candidates:
                return None
            cur_page = _fact_page(self.facts[current]) if current in self.facts else None

            def _page_gap(edge: EdgeRecord) -> Optional[int]:
                dst_fact = self.facts.get(edge.dst)
                dst_page = _fact_page(dst_fact) if dst_fact is not None else None
                if cur_page is None or dst_page is None:
                    return None
                return abs(dst_page - cur_page)

            if require_cross_page:
                cross = [
                    e for e in candidates
                    if (_page_gap(e) is not None and _page_gap(e) >= min_page_gap)
                ]
                if not cross:
                    return None
                candidates = cross
            # Weight by relation contribution, not only NLI. High NLI alone
            # over-samples definition/descriptive edges that later fail
            # minimality; joint-only margin and relation type make this a
            # TF-SFG-specific contribution prior.
            weights = []
            for edge in candidates:
                src_fact = self.facts.get(edge.src)
                dst_fact = self.facts.get(edge.dst)
                role_prior = (
                    _role_pair_walk_prior(src_fact, dst_fact)
                    if src_fact is not None and dst_fact is not None
                    else 0.50
                )
                # Strict minimality-era successes are dominated by setup
                # definition -> condition/rule pairs. Promote those walks
                # without removing other typed relations from consideration.
                weight = (0.76 * _edge_walk_weight(edge)) + (0.24 * role_prior)
                if cross_page_bias > 0.0:
                    gap = _page_gap(edge)
                    if gap is not None and gap >= min_page_gap:
                        weight *= (1.0 + cross_page_bias)
                weights.append(weight)
            total = sum(weights)
            if total <= 0:
                chosen = rng.choice(candidates)
            else:
                r = rng.random() * total
                cumulative = 0.0
                chosen = candidates[-1]
                for c, weight in zip(candidates, weights):
                    cumulative += weight
                    if r <= cumulative:
                        chosen = c
                        break
            fact_ids.append(chosen.dst)
            edges.append(chosen)
            visited.add(chosen.dst)

        if len(fact_ids) != depth:
            return None

        # Compute mean cosine from edge NLI scores as a proxy (used by _filter_chains).
        mean_cos = float(np.mean([e.nli_score for e in edges])) if edges else 0.0

        facts_in_chain = [self.facts[fid] for fid in fact_ids if fid in self.facts]
        role_path = [f.role for f in facts_in_chain]

        return FactChain(
            fact_ids=fact_ids,
            anchor_id=anchor_id,
            mean_cosine=mean_cos,
            role_path=role_path,
            chain_edges=[e.to_dict() for e in edges],
        )
