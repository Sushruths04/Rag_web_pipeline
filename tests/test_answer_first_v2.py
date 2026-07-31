import threading

from rag_gt.generation.answer_first_v2 import (
    _cluster_cache_key,
    _compose_multi,
    _doc_name,
    _fact_id,
    _form_hint,
    _pair_cache_key,
    _pair_prompt,
    _QUESTION_FORMS,
    _question_cache_key,
    _question_prompt,
    _settings,
    _side_prompt,
    build_v2_pairs,
    draft_clusters,
    draft_neighbor_pairs,
    gate_clusters,
    gate_neighbor_pairs,
)
from rag_gt.generation.cluster_bridge import build_clusters
from rag_gt.generation.neighbor_pairs import sample_neighbor_pairs


def _fact(fid, page, char, text):
    return {"fact_id": fid, "text": text, "canonical_form": text,
            "page_start": page, "char_start": char, "chunk_id": f"ck_{fid}",
            "bboxes": [], "doc": "din_iso_15609_welding_procedure_full"}


FACTS = [
    _fact("A", 1, 0, "The welding procedure specification defines ranges for essential variables."),
    _fact("A2", 1, 90, "Each essential variable range must be recorded before qualification."),
    _fact("B", 3, 0, "ISO 15607 defines the general rules for specification and qualification."),
    _fact("B2", 3, 90, "Qualification records must reference the governing standard edition."),
]

CLUSTER = {"kind": "cluster_2plus2", "doc": "din_iso_15609_welding_procedure_full",
           "fact_a": "A", "fact_a2": "A2", "fact_b": "B", "fact_b2": "B2",
           "bridge_entity": "ISO 15607", "bridge_norm": "iso 15607",
           "pages": [1, 3], "source_pair_id": "BP0001"}

BRIDGE_PAIRS = [
    {"doc": "din_iso_15609_welding_procedure_full", "fact_a": "A", "fact_b": "B",
     "bridge_entity": "ISO 15607", "bridge_norm": "iso 15607",
     "bridge_type": "STANDARD_REF", "pages": [1, 3], "pair_id": "BP0001"}
]


GOOD_QUESTION = "Which values does a WPS define, and which records must name the governing edition?"


class FakeLLM:
    """Serves neighbor-pair drafts plus the two-stage cluster modes."""

    def __init__(self):
        self.prompts = []

    def generate_json(self, prompt, temperature=0.0, max_tokens=0):
        self.prompts.append(prompt)
        if "MODE: NEIGHBOR-PAIR" in prompt:
            return {"clause_a": "The WPS defines ranges for essential variables.",
                    "clause_b": "Each variable range is recorded before qualification.",
                    "question": "Which values does a welding procedure specification define, and when are they recorded?"}
        if "MODE: CLUSTER-SIDE" in prompt:
            if "ISO 15607 defines" in prompt:  # B-side (facts B + B2)
                return {"clause_1": "General qualification rules come from a governing standard.",
                        "clause_2": "Qualification records must reference that standard's edition."}
            return {"clause_1": "The WPS defines ranges for essential variables.",
                    "clause_2": "Each variable range is recorded before qualification."}
        if "MODE: CLUSTER-QUESTION" in prompt:
            return {"question": GOOD_QUESTION}
        if "MODE: BRIDGE" in prompt:  # v1 fallback path (2-fact cross-page bridge)
            return {"clause_a": "The WPS defines ranges for essential variables.",
                    "clause_b": "General qualification rules come from a governing standard.",
                    "question": "Which values does the specification define, and where do the general rules come from?"}
        raise AssertionError(f"unexpected prompt mode: {prompt[:60]}")


def _accepting_nli(pairs):
    # scripted: clause->own-fact entailed (0.9); single fact -> full answer NOT
    # sufficient (0.2); joint multi-fact premise -> answer entailed (0.95)
    scores = []
    for premise, hypothesis in pairs:
        joint = premise.count(".") >= 2  # multi-fact premise
        full_answer = hypothesis.count(";") >= 1
        if joint:
            scores.append(0.95)
        elif full_answer:
            scores.append(0.2)
        else:
            scores.append(0.9)
    return scores


def _passing_necessity(answer, facts, threshold=0.5):
    return {"necessity_score": 1.0,
            "necessary_fact_ids": [f.fact_id for f in facts],
            "loo_entailment": [0.1] * len(facts)}


def test_compose_multi_joins_with_semicolons_preserving_acronyms():
    out = _compose_multi(["ISO 15607 sets the rules.", "The records must comply.",
                          "WPS ranges are defined."])
    assert out == "ISO 15607 sets the rules; the records must comply; WPS ranges are defined."


def test_prompts_carry_modes_and_sources():
    p = _pair_prompt("fact a text", "fact b text", "ISO 15609 (welding procedure specification)")
    assert "MODE: NEIGHBOR-PAIR" in p and "fact a text" in p
    s = _side_prompt("fact one text", "fact two text", "the source document")
    assert "MODE: CLUSTER-SIDE" in s and "fact one text" in s and "fact two text" in s
    q = _question_prompt(["ca", "cb", "cc", "cd"], "ISO 15607", "the source document")
    assert "MODE: CLUSTER-QUESTION" in q and "<<BRIDGE>>ISO 15607<</BRIDGE>>" in q
    assert all(c in q for c in ("ca", "cb", "cc", "cd"))


def test_cluster_cache_key_is_content_derived():
    assert _cluster_cache_key(CLUSTER) == "A|A2|B|B2|iso 15607"


# ---------------------------------------------------------------------------
# M4 refactor -- draft_* / gate_* extraction (separable, individually
# cacheable steps for the block SDK; see 05_BLOCK_CATALOG.md sec. 4)
# ---------------------------------------------------------------------------


def test_draft_neighbor_pairs_is_draft_only_no_gating():
    """draft_neighbor_pairs is the pure LLM-drafting step for 2-fact
    neighbor pairs: one cache entry per pair, no NLI scoring, no rejection
    decisions -- those live in gate_neighbor_pairs."""
    settings = _settings()
    facts_by_id = {_fact_id(f): f for f in FACTS}
    pair = {"fact_a": "A", "fact_b": "A2"}
    drafts = draft_neighbor_pairs(
        [pair], facts_by_id, FakeLLM(), settings,
        doc_name=_doc_name("din_iso_15609_welding_procedure_full", settings),
        workers=1, cache_path=None,
    )
    key = f"pairv2:{_pair_cache_key(pair)}"
    assert key in drafts
    assert drafts[key]["clauses"] == [
        "The WPS defines ranges for essential variables.",
        "Each variable range is recorded before qualification.",
    ]
    assert "question" in drafts[key]


def test_draft_clusters_is_draft_only_public_alias_of_two_stage_drafting():
    """draft_clusters exposes the existing two-stage cluster drafting
    (side clauses -> pre-gate -> question) as a public, block-SDK-callable
    step -- same signature and behavior as the pre-refactor private
    ``_draft_clusters_two_stage``, just no leading underscore."""
    settings = _settings()
    facts_by_id = {_fact_id(f): f for f in FACTS}
    thresholds_clause_min = float(settings["clause_entailment_min"])
    drafts, meta = draft_clusters(
        [CLUSTER], facts_by_id, FakeLLM(), settings,
        workers=1, cache_path=None, nli_fn=_accepting_nli,
        doc_name=_doc_name("din_iso_15609_welding_procedure_full", settings),
        clause_min=thresholds_clause_min,
    )
    key = f"cluster:{_cluster_cache_key(CLUSTER)}"
    assert key in drafts
    assert drafts[key]["question"] == GOOD_QUESTION
    assert len(drafts[key]["clauses"]) == 4
    assert meta["n_sides"] == 2


def test_gate_neighbor_pairs_accepts_a_valid_draft():
    """gate_neighbor_pairs is the gate-only step for 2-fact pairs: draft in,
    accepted QA record (or rejection reason) out -- no LLM call happens
    here, only local NLI/necessity scoring."""
    settings = _settings()
    facts_by_id = {_fact_id(f): f for f in FACTS}
    pair = {"fact_a": "A", "fact_b": "A2"}
    drafts = draft_neighbor_pairs(
        [pair], facts_by_id, FakeLLM(), settings,
        doc_name=_doc_name("din_iso_15609_welding_procedure_full", settings),
        workers=1, cache_path=None,
    )
    thresholds = {"clause_min": float(settings["clause_entailment_min"]),
                  "single_max": float(settings["single_fact_answer_max"]),
                  "joint_min": float(settings["joint_answer_min"])}
    accepted, rejected = gate_neighbor_pairs(
        [pair], facts_by_id, drafts, thresholds=thresholds,
        require_chunk_ids=True, nli_fn=_accepting_nli,
        necessity_fn=lambda items, threshold=None: [_passing_necessity(a, f) for a, f in items],
        seen_questions=set(), doc="din_iso_15609_welding_procedure_full",
    )
    assert len(accepted) == 1 and not rejected
    assert accepted[0]["evidence_strategy"] == "neighbor_pair_v2"
    assert accepted[0]["gold_fact_ids"] == ["A", "A2"]


def test_gate_clusters_accepts_a_valid_draft():
    settings = _settings()
    facts_by_id = {_fact_id(f): f for f in FACTS}
    drafts, _meta = draft_clusters(
        [CLUSTER], facts_by_id, FakeLLM(), settings,
        workers=1, cache_path=None, nli_fn=_accepting_nli,
        doc_name=_doc_name("din_iso_15609_welding_procedure_full", settings),
        clause_min=float(settings["clause_entailment_min"]),
    )
    thresholds = {"clause_min": float(settings["clause_entailment_min"]),
                  "single_max": float(settings["single_fact_answer_max"]),
                  "joint_min": float(settings["joint_answer_min"])}
    accepted, rejected = gate_clusters(
        [CLUSTER], facts_by_id, drafts, thresholds=thresholds,
        require_chunk_ids=True, nli_fn=_accepting_nli,
        necessity_fn=lambda items, threshold=None: [_passing_necessity(a, f) for a, f in items],
        seen_questions=set(), doc="din_iso_15609_welding_procedure_full",
    )
    assert len(accepted) == 1 and not rejected
    assert accepted[0]["evidence_strategy"] == "cluster_2plus2"
    assert accepted[0]["gold_fact_ids"] == ["A", "A2", "B", "B2"]


def test_resolve_necessity_group_fn_defaults_to_batched_loo():
    """When the caller leaves necessity_fn/necessity_batch_fn at their
    defaults, resolution must pick the genuinely batched
    leave_one_out_necessity_batch (T8), not per-candidate wrapping."""
    from rag_gt.allpdf.necessity import leave_one_out_necessity, leave_one_out_necessity_batch
    from rag_gt.generation.answer_first_v2 import _resolve_necessity_group_fn

    fn = _resolve_necessity_group_fn(leave_one_out_necessity, None)
    assert fn is leave_one_out_necessity_batch


def test_resolve_necessity_group_fn_wraps_single_item_override():
    """Backward compat: a caller-supplied single-item necessity_fn override
    (the pre-T8 contract used by every existing test fixture) is auto-
    wrapped into batch form."""
    from rag_gt.generation.answer_first_v2 import _resolve_necessity_group_fn

    calls = []

    def single(answer, facts, threshold=None):
        calls.append((answer, facts, threshold))
        return {"necessity_score": 1.0, "necessary_fact_ids": [], "loo_entailment": []}

    fn = _resolve_necessity_group_fn(single, None)
    out = fn([("ans", ["f1"])], threshold=0.5)
    assert calls == [("ans", ["f1"], 0.5)]
    assert out[0]["necessity_score"] == 1.0


def test_resolve_necessity_group_fn_prefers_explicit_batch_override():
    from rag_gt.allpdf.necessity import leave_one_out_necessity
    from rag_gt.generation.answer_first_v2 import _resolve_necessity_group_fn

    sentinel = object()

    def batch_fn(items, *, threshold=None):
        return sentinel

    fn = _resolve_necessity_group_fn(leave_one_out_necessity, batch_fn)
    assert fn is batch_fn


def test_manual_draft_gate_composition_matches_build_v2_pairs_output():
    """MANDATORY golden-file regression (M4 refactor spec sec. 3a-2): calling
    draft_neighbor_pairs -> draft_clusters -> gate_neighbor_pairs ->
    gate_clusters directly and composing the results by hand must reproduce
    exactly what build_v2_pairs returns for the equivalent inputs. This
    proves the extraction is a transparent recomposition, not a function
    that merely happens to still be called correctly from inside
    build_v2_pairs.

    include_fallback=False on both sides so the comparison is not entangled
    with the (deliberately un-split, threaded) demotion/fallback path.
    """
    doc = "din_iso_15609_welding_procedure_full"
    settings = _settings()
    doc_name = _doc_name(doc, settings)
    facts_by_id = {_fact_id(f): f for f in FACTS}
    thresholds = {"clause_min": float(settings["clause_entailment_min"]),
                  "single_max": float(settings["single_fact_answer_max"]),
                  "joint_min": float(settings["joint_answer_min"])}

    neighbor_pairs = sample_neighbor_pairs(FACTS, doc=doc, max_uses_per_fact=1,
                                           min_cosine=0.40, max_cosine=0.95)
    clusters, _fallback = build_clusters(BRIDGE_PAIRS, FACTS, min_cosine=0.40, max_cosine=0.95)

    llm_manual = FakeLLM()
    pair_drafts = draft_neighbor_pairs(
        neighbor_pairs, facts_by_id, llm_manual, settings,
        doc_name=doc_name, workers=1, cache_path=None,
    )
    cluster_drafts, _meta = draft_clusters(
        clusters, facts_by_id, llm_manual, settings,
        workers=1, cache_path=None, nli_fn=_accepting_nli,
        doc_name=doc_name, clause_min=thresholds["clause_min"],
    )
    drafts = {**pair_drafts, **cluster_drafts}

    # replicate build_v2_pairs' single-item -> batch necessity auto-wrap
    # exactly (it is not exported; this mirrors the same three lines that
    # live inside build_v2_pairs for the non-default necessity_fn case).
    def necessity_group_fn(items, *, threshold=None, _fn=_passing_necessity):
        return [_fn(answer, facts, threshold=threshold) for answer, facts in items]

    seen_questions: set[str] = set()
    pair_out, _pair_rej = gate_neighbor_pairs(
        neighbor_pairs, facts_by_id, drafts, thresholds=thresholds,
        require_chunk_ids=True, nli_fn=_accepting_nli,
        necessity_fn=necessity_group_fn, seen_questions=seen_questions, doc=doc,
    )
    cluster_out, _cluster_rej = gate_clusters(
        clusters, facts_by_id, drafts, thresholds=thresholds,
        require_chunk_ids=True, nli_fn=_accepting_nli,
        necessity_fn=necessity_group_fn, seen_questions=seen_questions, doc=doc,
    )
    manual_pairs = pair_out + cluster_out
    for index, item in enumerate(manual_pairs, start=1):
        item["qa_id"] = f"V2Q{index:05d}"

    llm_reference = FakeLLM()
    reference = build_v2_pairs(
        FACTS, BRIDGE_PAIRS, llm_reference,
        doc=doc, nli_fn=_accepting_nli, necessity_fn=_passing_necessity,
        include_fallback=False,
    )

    assert manual_pairs == reference["pairs"]
    assert len(manual_pairs) == 2  # one neighbor pair + one cluster, sanity check


def test_build_v2_pairs_end_to_end():
    result = build_v2_pairs(
        FACTS, BRIDGE_PAIRS, FakeLLM(),
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=_accepting_nli,
        necessity_fn=_passing_necessity,
        include_fallback=False,
    )
    pairs = result["pairs"]
    strategies = {p["evidence_strategy"] for p in pairs}
    assert "neighbor_pair_v2" in strategies and "cluster_2plus2" in strategies
    single = next(p for p in pairs if p["evidence_strategy"] == "neighbor_pair_v2")
    assert single["hop_type"] == "single" and len(single["answer_clauses"]) == 2
    cluster = next(p for p in pairs if p["evidence_strategy"] == "cluster_2plus2")
    assert cluster["hop_type"] == "bridge" and len(cluster["answer_clauses"]) == 4
    assert len(cluster["gold_chunk_ids"]) == 4 and cluster["grounding_complete"] is True
    assert all(p["qa_id"].startswith("V2Q") for p in pairs)
    assert result["stats"]["n_cluster_pass"] == 1


def test_strict_grounding_rejects_missing_chunk_id():
    facts = [dict(f) for f in FACTS]
    facts[0].pop("chunk_id")
    result = build_v2_pairs(
        facts, [], FakeLLM(),
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=_accepting_nli, necessity_fn=_passing_necessity,
        include_fallback=False, require_chunk_ids=True,
    )
    reasons = result["stats"]["pair_rejection_reasons"]
    assert reasons.get("ungrounded_chunk_ids", 0) >= 1


# ---------------------------------------------------------------------------
# T1 -- pre-draft bridge validity filter (kills F6)
# ---------------------------------------------------------------------------


def test_pre_draft_filter_drops_bridge_referencing_unknown_fact():
    """A bridge pair naming a fact_id absent from the input set must be
    dropped BEFORE any drafting call, not discovered after paying for one."""
    bridge_pairs = BRIDGE_PAIRS + [
        {"doc": "din_iso_15609_welding_procedure_full", "fact_a": "A", "fact_b": "ZZZ_MISSING",
         "bridge_entity": "Missing Bridge", "bridge_norm": "missing bridge",
         "bridge_type": "STANDARD_REF", "pages": [1, 9], "pair_id": "BP9999"}
    ]
    llm = FakeLLM()
    result = build_v2_pairs(
        FACTS, bridge_pairs, llm,
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=_accepting_nli, necessity_fn=_passing_necessity,
        include_fallback=True, pair_limit=0,
    )
    assert result["stats"]["pre_rejected_unresolvable"] == 1
    assert all("ZZZ_MISSING" not in p for p in llm.prompts)
    # the bogus pair never even reaches build_clusters' own fallback list
    assert result["stats"]["n_fallback_bridges"] == 0


def test_pre_draft_filter_drops_bridge_missing_chunk_id_before_any_llm_call():
    """Under strict grounding, a bridge referencing a fact with no real
    chunk_id must be dropped pre-draft -- previously this fact would still
    reach a full cluster-side/question draft (paid) before being rejected
    downstream as ungrounded_chunk_ids."""
    facts = [dict(f) for f in FACTS]
    facts[2] = dict(facts[2])
    facts[2].pop("chunk_id")  # fact B loses its chunk_id
    llm = FakeLLM()
    result = build_v2_pairs(
        facts, BRIDGE_PAIRS, llm,
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=_accepting_nli, necessity_fn=_passing_necessity,
        include_fallback=True, pair_limit=0, require_chunk_ids=True,
    )
    assert result["stats"]["pre_rejected_unresolvable"] == 1
    assert not llm.prompts


def test_pre_draft_filter_off_when_require_chunk_ids_false():
    """When strict grounding is disabled, a fact missing only its chunk_id is
    still resolvable (it will emit a fact:<id> pseudo-ID downstream), so it
    must NOT be pre-rejected."""
    facts = [dict(f) for f in FACTS]
    facts[2] = dict(facts[2])
    facts[2].pop("chunk_id")
    result = build_v2_pairs(
        facts, BRIDGE_PAIRS, FakeLLM(),
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=_accepting_nli, necessity_fn=_passing_necessity,
        include_fallback=True, pair_limit=0, require_chunk_ids=False,
    )
    assert result["stats"]["pre_rejected_unresolvable"] == 0


# ---------------------------------------------------------------------------
# T2 -- bridge surface quality gate (kills F1)
# ---------------------------------------------------------------------------


def test_junk_bridge_surface_dropped_before_any_llm_call():
    junk_bridges = [
        {"doc": "din_iso_15609_welding_procedure_full", "fact_a": "A", "fact_b": "B",
         "bridge_entity": "than 0", "bridge_norm": "than 0",
         "bridge_type": "STANDARD_REF", "pages": [1, 3], "pair_id": "BP0002"}
    ]
    llm = FakeLLM()
    result = build_v2_pairs(
        FACTS, junk_bridges, llm,
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=_accepting_nli, necessity_fn=_passing_necessity,
        include_fallback=True, pair_limit=0,
    )
    assert result["stats"]["n_bridge_surface_rejected"] == 1
    assert not llm.prompts
    assert result["stats"]["n_clusters_built"] == 0
    assert result["stats"]["n_fallback_bridges"] == 0


def test_repairable_bridge_surface_flows_through_repaired():
    """'di erent' repairs to 'different' -- the cluster must still build and
    the accepted QA record must carry the REPAIRED surface, not the OCR junk."""
    repairable_bridges = [
        {"doc": "din_iso_15609_welding_procedure_full", "fact_a": "A", "fact_b": "B",
         "bridge_entity": "di erent", "bridge_norm": "di erent",
         "bridge_type": "STANDARD_REF", "pages": [1, 3], "pair_id": "BP0003"}
    ]
    result = build_v2_pairs(
        FACTS, repairable_bridges, FakeLLM(),
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=_accepting_nli, necessity_fn=_passing_necessity,
        include_fallback=False, pair_limit=0,
    )
    assert result["stats"]["n_bridge_surface_rejected"] == 0
    assert result["stats"]["n_cluster_pass"] == 1
    cluster = next(p for p in result["pairs"] if p["evidence_strategy"] == "cluster_2plus2")
    assert cluster["bridge_entity"] == "different"


def test_single_sufficiency_gate_rejects():
    def sufficient_nli(pairs):
        return [0.9 for _ in pairs]  # every premise entails everything -> one fact suffices
    result = build_v2_pairs(
        FACTS, [], FakeLLM(),
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=sufficient_nli, necessity_fn=_passing_necessity,
        include_fallback=False,
    )
    assert result["stats"]["n_pair_pass"] == 0
    assert result["stats"]["pair_rejection_reasons"].get("not_jointly_necessary", 0) >= 1


# ---------------------------------------------------------------------------
# Two-stage cluster drafting
# ---------------------------------------------------------------------------

BAD_CLAUSE = "BAD claim entirely unrelated to any source text present here."


def _nli_bad_aware(pairs):
    base = _accepting_nli(pairs)
    return [0.05 if "BAD" in hyp else s for (prem, hyp), s in zip(pairs, base)]


class RetrySideLLM(FakeLLM):
    """First B-side draft has a non-entailed clause; the retry fixes it."""

    def __init__(self):
        super().__init__()
        self.bside_calls = 0

    def generate_json(self, prompt, temperature=0.0, max_tokens=0):
        if "MODE: CLUSTER-SIDE" in prompt and "ISO 15607 defines" in prompt:
            self.prompts.append(prompt)
            self.bside_calls += 1
            if self.bside_calls == 1:
                return {"clause_1": BAD_CLAUSE,
                        "clause_2": "Qualification records must reference that standard's edition."}
            return {"clause_1": "General qualification rules come from a governing standard.",
                    "clause_2": "Qualification records must reference that standard's edition."}
        return super().generate_json(prompt, temperature, max_tokens)


class AlwaysBadSideLLM(FakeLLM):
    """The B-side clause is never entailed, even after the feedback retry."""

    def generate_json(self, prompt, temperature=0.0, max_tokens=0):
        if "MODE: CLUSTER-SIDE" in prompt and "ISO 15607 defines" in prompt:
            self.prompts.append(prompt)
            return {"clause_1": BAD_CLAUSE,
                    "clause_2": "Qualification records must reference that standard's edition."}
        return super().generate_json(prompt, temperature, max_tokens)


class LeakQuestionLLM(FakeLLM):
    """First question leaks the bridge surface; the retry hides it."""

    def __init__(self):
        super().__init__()
        self.q_calls = 0

    def generate_json(self, prompt, temperature=0.0, max_tokens=0):
        if "MODE: CLUSTER-QUESTION" in prompt:
            self.prompts.append(prompt)
            self.q_calls += 1
            if self.q_calls == 1:
                return {"question": "How does ISO 15607 connect the WPS ranges to the qualification records?"}
            return {"question": GOOD_QUESTION}
        return super().generate_json(prompt, temperature, max_tokens)


def _run_cluster_build(llm, nli_fn):
    return build_v2_pairs(
        FACTS, BRIDGE_PAIRS, llm,
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=nli_fn, necessity_fn=_passing_necessity,
        include_fallback=False, pair_limit=0,
    )


def test_two_stage_side_retry_with_feedback_recovers_cluster():
    llm = RetrySideLLM()
    result = _run_cluster_build(llm, _nli_bad_aware)
    assert result["stats"]["n_cluster_pass"] == 1
    assert result["stats"]["n_side_retries"] == 1
    retry_prompts = [p for p in llm.prompts if "not entailed" in p]
    assert retry_prompts and "MODE: CLUSTER-SIDE" in retry_prompts[0]


def test_pregate_failure_skips_question_call():
    llm = AlwaysBadSideLLM()
    result = _run_cluster_build(llm, _nli_bad_aware)
    assert result["stats"]["n_cluster_pass"] == 0
    assert result["stats"]["cluster_rejection_reasons"].get("clause_pregate", 0) == 1
    assert not any("MODE: CLUSTER-QUESTION" in p for p in llm.prompts)


def test_bridge_leak_question_retried():
    llm = LeakQuestionLLM()
    result = _run_cluster_build(llm, _accepting_nli)
    assert result["stats"]["n_cluster_pass"] == 1
    assert llm.q_calls == 2
    cluster = next(p for p in result["pairs"] if p["evidence_strategy"] == "cluster_2plus2")
    assert "iso 15607" not in cluster["question"].lower()


# ---------------------------------------------------------------------------
# T6 -- bridge-leak waste reduction (halves F5)
# ---------------------------------------------------------------------------


def test_question_prompt_lists_forbidden_bridge_words():
    q = _question_prompt(["ca", "cb", "cc", "cd"], "ISO 15607", "the source document",
                         bridge_norm="iso 15607")
    assert "FORBIDDEN" in q
    assert "ISO 15607" in q
    assert "iso 15607" in q


def test_bridge_leak_retry_quotes_the_exact_leaked_substring():
    """The retry message must name what actually leaked, not a generic
    'invalid or leaked the bridge' notice."""
    llm = LeakQuestionLLM()
    _run_cluster_build(llm, _accepting_nli)
    assert llm.q_calls == 2
    retried_prompt = llm.prompts[-1]
    assert "'ISO 15607'" in retried_prompt
    assert "remove it" in retried_prompt


def test_question_attempts_override_allows_third_try(monkeypatch):
    """question_generation_attempts gives the cheap question call more tries
    than the global generation_attempts used by the expensive drafts."""
    from rag_gt.generation import answer_first_v2 as m

    class DoubleLeakLLM(LeakQuestionLLM):
        def generate_json(self, prompt, temperature=0.0, max_tokens=0):
            if "MODE: CLUSTER-QUESTION" in prompt:
                self.prompts.append(prompt)
                self.q_calls += 1
                if self.q_calls <= 2:
                    return {"question": "How does ISO 15607 connect the WPS ranges to the qualification records?"}
                return {"question": GOOD_QUESTION}
            return FakeLLM.generate_json(self, prompt, temperature, max_tokens)

    base_settings = m._settings()
    monkeypatch.setattr(m, "_settings", lambda: {**base_settings,
                                                 "generation_attempts": 2,
                                                 "question_generation_attempts": 3})
    llm = DoubleLeakLLM()
    result = _run_cluster_build(llm, _accepting_nli)
    assert llm.q_calls == 3
    assert result["stats"]["n_cluster_pass"] == 1


class AlwaysLeakQuestionLLM(FakeLLM):
    """Every question attempt leaks the bridge — the cluster must fail."""

    def generate_json(self, prompt, temperature=0.0, max_tokens=0):
        if "MODE: CLUSTER-QUESTION" in prompt:
            self.prompts.append(prompt)
            return {"question": "How does ISO 15607 connect the WPS ranges to the qualification records?"}
        return super().generate_json(prompt, temperature, max_tokens)


def test_failed_cluster_demoted_to_two_fact_bridge():
    """A failed 4-fact cluster is still a verified cross-page bridge pair —
    demote it to the v1 2-fact path instead of discarding the multi-hop."""
    llm = AlwaysLeakQuestionLLM()
    result = build_v2_pairs(
        FACTS, BRIDGE_PAIRS, llm,
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=_accepting_nli, necessity_fn=_passing_necessity,
        include_fallback=True, pair_limit=0,
    )
    stats = result["stats"]
    assert stats["n_cluster_pass"] == 0
    assert stats["n_clusters_demoted"] == 1
    fallback = [p for p in result["pairs"] if p["evidence_strategy"] == "bridge_pair_v1"]
    assert len(fallback) == 1
    assert fallback[0]["gold_fact_ids"] == ["A", "B"]
    assert fallback[0]["hop_type"] == "bridge"


def test_demotion_off_without_fallback():
    llm = AlwaysLeakQuestionLLM()
    result = build_v2_pairs(
        FACTS, BRIDGE_PAIRS, llm,
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=_accepting_nli, necessity_fn=_passing_necessity,
        include_fallback=False, pair_limit=0,
    )
    assert result["stats"]["n_clusters_demoted"] == 0
    assert result["stats"]["n_fallback_pass"] == 0


def test_stats_carry_stage_timings():
    result = _run_cluster_build(FakeLLM(), _accepting_nli)
    timings = result["stats"]["timings_sec"]
    assert set(timings) >= {"sampling", "drafting", "scoring", "fallback"}
    assert all(isinstance(v, float) for v in timings.values())


def test_question_cache_key_changes_with_clauses():
    k1 = _question_cache_key("A|A2|B|B2|iso 15607", ["a", "b", "c", "d"])
    k2 = _question_cache_key("A|A2|B|B2|iso 15607", ["a", "b", "c", "e"])
    assert k1 != k2
    assert k1.startswith("A|A2|B|B2|iso 15607|q")


# ---------------------------------------------------------------------------
# T5 -- question-form rotation (fixes F4: 90% "What" monoculture)
# ---------------------------------------------------------------------------


def test_form_hint_is_deterministic_and_varies_across_keys():
    keys = [f"cache-key-{i}" for i in range(12)]
    hints = [_form_hint(k) for k in keys]
    assert all(h in _QUESTION_FORMS for h in hints)
    assert len(set(hints)) > 1, "12 distinct keys should not all rotate to the same form"
    # same key -> same hint, every time
    assert _form_hint("cache-key-0") == _form_hint("cache-key-0")


def test_pair_prompt_carries_form_hint():
    p = _pair_prompt("fact a text", "fact b text", "the source document", form_hint="Which")
    assert "Which" in p


def test_question_prompt_carries_form_hint():
    q = _question_prompt(["ca", "cb", "cc", "cd"], "ISO 15607", "the source document",
                         form_hint="Under what condition")
    assert "Under what condition" in q


def test_stats_include_question_first_word_histogram():
    result = build_v2_pairs(
        FACTS, BRIDGE_PAIRS, FakeLLM(),
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=_accepting_nli, necessity_fn=_passing_necessity,
        include_fallback=False,
    )
    histogram = result["stats"]["question_first_word_histogram"]
    n_questions = len(result["pairs"])
    assert sum(histogram.values()) == n_questions
    assert n_questions > 0


# ---------------------------------------------------------------------------
# T7 -- overlap fallback drafting with cluster scoring (fixes F7)
# ---------------------------------------------------------------------------

# A second, lonely fact with no same-page neighbour, so its bridge pair
# becomes STRUCTURAL fallback known immediately at sampling time (before any
# drafting), independent of the cluster A/A2/B/B2 which drafts and passes
# normally.
FACTS_WITH_LONELY = FACTS + [
    _fact("C", 7, 0, "A standalone fact on its own page with no reading-order neighbour."),
]
BRIDGE_PAIRS_WITH_LONELY = BRIDGE_PAIRS + [
    {"doc": "din_iso_15609_welding_procedure_full", "fact_a": "A", "fact_b": "C",
     "bridge_entity": "Lonely Bridge", "bridge_norm": "lonely bridge",
     "bridge_type": "STANDARD_REF", "pages": [1, 7], "pair_id": "BP0005"}
]


def test_fallback_drafting_overlaps_with_cluster_scoring():
    """The v1 fallback draft for the structural (lonely) bridge pair must be
    launched BEFORE/WHILE the OUTER pair/cluster NLI scoring runs on the main
    thread, not only after scoring finishes.

    Proven deterministically with a blocking Event rather than a timing race:
    the fallback LLM call sets the event on entry; nli_fn blocks on that
    event (bounded wait) the first time it is called AFTER the drafting
    phase (call #1 is the cluster's own clause pre-gate, which legitimately
    runs during drafting, before fallback launches, in both old and new
    code -- it is deliberately excluded). If fallback only starts after ALL
    scoring finishes (old sequential behaviour), nothing can ever set the
    event while nli_fn's wait() blocks the single-threaded call chain, so
    the wait genuinely times out.
    """
    fallback_started = threading.Event()

    class SignallingFallbackLLM(FakeLLM):
        def generate_json(self, prompt, temperature=0.0, max_tokens=0):
            if "MODE: BRIDGE" in prompt:
                fallback_started.set()
            return super().generate_json(prompt, temperature, max_tokens)

    call_index = {"n": 0}
    post_drafting_wait_result = {"ok": None}

    def _nli_that_waits_for_fallback(pairs):
        call_index["n"] += 1
        if call_index["n"] >= 2 and post_drafting_wait_result["ok"] is None:
            post_drafting_wait_result["ok"] = fallback_started.wait(timeout=2.0)
        return _accepting_nli(pairs)

    llm = SignallingFallbackLLM()
    result = build_v2_pairs(
        FACTS_WITH_LONELY, BRIDGE_PAIRS_WITH_LONELY, llm,
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=_nli_that_waits_for_fallback, necessity_fn=_passing_necessity,
        include_fallback=True, pair_limit=0,
    )
    assert post_drafting_wait_result["ok"] is True, (
        "pair/cluster scoring ran without the fallback draft ever having "
        "started -- fallback drafting is not overlapping with scoring"
    )
    assert fallback_started.is_set()
    assert "fallback_overlap_sec" in result["stats"]["timings_sec"]
    assert result["stats"]["timings_sec"]["fallback_overlap_sec"] >= 0.0
    # the lonely pair must still make it through as a v1 bridge pair
    lonely = [p for p in result["pairs"] if p["evidence_strategy"] == "bridge_pair_v1"]
    assert any(p["gold_fact_ids"] == ["A", "C"] for p in lonely)


# ---------------------------------------------------------------------------
# T8 -- batch leave-one-out necessity across candidates (fixes F8, speeds
# scoring); asserts nli_check.truncation_stats() is surfaced in the stats
# block with a WARN on cluster premise truncation.
# ---------------------------------------------------------------------------


def test_leave_one_out_necessity_batch_makes_one_nli_call_for_many_candidates(monkeypatch):
    from rag_gt.allpdf.necessity import leave_one_out_necessity_batch
    from rag_gt.core.types import Fact

    def _f(fid, text):
        return Fact(fact_id=fid, text=text, canonical_form=text, role="definition",
                    self_containment_score=1.0)

    facts_1 = [_f("x1", "fact text one here"), _f("x2", "fact text two here")]
    facts_2 = [_f("y1", "fact text three here"), _f("y2", "fact text four here"),
               _f("y3", "fact text five here")]

    calls = {"n": 0}

    def fake_nli_batch(pairs):
        calls["n"] += 1
        return [0.1 for _ in pairs]  # below any reasonable threshold -> all "necessary"

    monkeypatch.setattr("rag_gt.validation.nli_check.nli_batch", fake_nli_batch)

    items = [("answer one", facts_1), ("answer two three", facts_2)]
    results = leave_one_out_necessity_batch(items, threshold=0.5)

    assert calls["n"] == 1, "must batch every candidate's LOO pairs into ONE nli_batch call"
    assert len(results) == 2
    assert results[0]["necessity_score"] == 1.0
    assert results[0]["necessary_fact_ids"] == ["x1", "x2"]
    assert results[1]["necessary_fact_ids"] == ["y1", "y2", "y3"]


def test_leave_one_out_necessity_batch_handles_ineligible_items(monkeypatch):
    from rag_gt.allpdf.necessity import leave_one_out_necessity_batch
    from rag_gt.core.types import Fact

    def _f(fid, text):
        return Fact(fact_id=fid, text=text, canonical_form=text, role="definition",
                    self_containment_score=1.0)

    eligible = [_f("x1", "fact text one here"), _f("x2", "fact text two here")]
    single = [_f("z1", "fact text alone here")]

    calls = {"n": 0}

    def fake_nli_batch(pairs):
        calls["n"] += 1
        return [0.1 for _ in pairs]

    monkeypatch.setattr("rag_gt.validation.nli_check.nli_batch", fake_nli_batch)

    items = [
        ("", eligible),
        ("insufficient information to answer", eligible),
        ("real answer", single),
        ("real multi-fact answer", eligible),
    ]
    results = leave_one_out_necessity_batch(items)
    assert results[0] is None
    assert results[1] is None
    assert results[2] is None
    assert results[3] is not None
    assert results[3]["necessity_score"] == 1.0
    assert calls["n"] == 1


def test_leave_one_out_necessity_batch_skips_nli_call_when_nothing_eligible(monkeypatch):
    from rag_gt.allpdf.necessity import leave_one_out_necessity_batch

    calls = {"n": 0}

    def fake_nli_batch(pairs):
        calls["n"] += 1
        return []

    monkeypatch.setattr("rag_gt.validation.nli_check.nli_batch", fake_nli_batch)
    results = leave_one_out_necessity_batch([("", [])])
    assert results == [None]
    assert calls["n"] == 0


def test_score_group_via_build_v2_pairs_batches_necessity_across_candidates():
    """Two independent neighbour-pair candidates on the same page must share
    ONE necessity_batch_fn call, not one call each."""
    page5_facts = [
        _fact("D1", 5, 0, "The header lists the applicable code and edition for compliance."),
        _fact("D2", 5, 60, "The edition determines which clauses apply during review."),
        _fact("D3", 5, 400, "The appendix defines abbreviations used throughout the document."),
        _fact("D4", 5, 460, "Abbreviations must be expanded on first use in each section."),
    ]
    spy_calls: list[int] = []

    def spy_batch(items, *, threshold=None):
        spy_calls.append(len(items))
        return [
            {"necessity_score": 1.0, "necessary_fact_ids": [f.fact_id for f in facts],
             "loo_entailment": [0.1] * len(facts)}
            for _, facts in items
        ]

    build_v2_pairs(
        page5_facts, [], FakeLLM(),
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=_accepting_nli, necessity_batch_fn=spy_batch,
        include_fallback=False,
    )
    assert spy_calls == [2]


def test_necessity_fn_override_still_applies_when_no_batch_override_given():
    """Backward compat: a caller overriding only the single-item necessity_fn
    (the pre-T8 contract, e.g. every existing test fixture in this file)
    must still control acceptance -- it gets auto-wrapped into batch form."""

    def rejecting_necessity(answer, facts, threshold=0.5):
        return {"necessity_score": 0.0, "necessary_fact_ids": [], "loo_entailment": [0.9] * len(facts)}

    result = build_v2_pairs(
        FACTS, BRIDGE_PAIRS, FakeLLM(),
        doc="din_iso_15609_welding_procedure_full",
        nli_fn=_accepting_nli, necessity_fn=rejecting_necessity,
        include_fallback=False,
    )
    assert result["stats"]["n_cluster_pass"] == 0
    assert result["stats"]["cluster_rejection_reasons"].get("loo_failed", 0) == 1


def test_stats_include_nli_truncation_block():
    result = _run_cluster_build(FakeLLM(), _accepting_nli)
    truncation = result["stats"]["nli_truncation"]
    assert "cluster" in truncation
    assert "premise_truncated" in truncation["cluster"]


def test_warns_when_cluster_scoring_truncates_a_premise(monkeypatch):
    from rag_gt.generation import answer_first_v2 as m

    counter = {"n": 0}

    def fake_truncation_stats():
        counter["n"] += 1
        # first call (snapshot before cluster scoring) reports 0; every call
        # after reports a jump of 2 premise truncations during that window
        return {"premise_truncated": 0 if counter["n"] == 1 else 2,
                "hypothesis_truncated": 0, "total_inputs": 0}

    monkeypatch.setattr(m, "truncation_stats", fake_truncation_stats)

    class FakeLogger:
        def __init__(self):
            self.warnings = []

        def warning(self, msg):
            self.warnings.append(msg)

        def info(self, msg):
            pass

    fake_logger = FakeLogger()
    monkeypatch.setattr(m, "logger", fake_logger)

    _run_cluster_build(FakeLLM(), _accepting_nli)
    assert any("premise" in w.lower() or "truncat" in w.lower() for w in fake_logger.warnings)
