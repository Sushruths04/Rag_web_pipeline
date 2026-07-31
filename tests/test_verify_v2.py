"""Stage D verification for v2 evidence-dense QA (n=2 singles, n=4 clusters).

verify_qa_pairs (bridge_verifier.py) is hard-coded to the v1 schema and would
reject every v2 record by construction (see verify_v2.py module docstring).
These tests prove the generalized cascade handles both shapes, that
bridge_pair_v1 records route to the unmodified v1 verifier, and that the
deterministic pre-checks (bridge leak, clause-count, missing fact) short
-circuit before any NLI call.
"""
from rag_gt.validation.verify_v2 import verify_v2_pairs

SETTINGS = {
    "duplicate_jaccard_max": 0.60,
    "mutual_entailment_min": 0.85,
    "borderline_margin": 0.05,
    "use_llm_borderline": False,
    "judge_temperature": 0.0,
    "judge_max_tokens": 192,
    "clause_entailment_min": 0.65,
    "single_fact_answer_max": 0.50,
    "joint_answer_min": 0.85,
}


def _facts():
    return [
        {"id": "A", "text": "The welding procedure specification defines ranges for essential variables."},
        {"id": "A2", "text": "Each essential variable range must be recorded before qualification."},
        {"id": "B", "text": "ISO 15607 defines the general rules for specification and qualification."},
        {"id": "B2", "text": "Qualification records must reference the governing standard edition."},
    ]


def _neighbor_pair_qa():
    return {
        "qa_id": "V2Q00001",
        "hop_type": "single",
        "evidence_strategy": "neighbor_pair_v2",
        "bridge_entity": "",
        "question": "Which values does a WPS define, and when are they recorded?",
        "answer": "The WPS defines ranges for essential variables; each range is recorded before qualification.",
        "answer_clauses": [
            {"text": "The WPS defines ranges for essential variables.", "fact_id": "A"},
            {"text": "Each range is recorded before qualification.", "fact_id": "A2"},
        ],
        "gold_fact_ids": ["A", "A2"],
        "necessity": {"nli_singles": [0.1, 0.12], "nli_joint": 0.95},
    }


def _cluster_qa():
    return {
        "qa_id": "V2Q00002",
        "hop_type": "bridge",
        "evidence_strategy": "cluster_2plus2",
        "bridge_entity": "ISO 15607",
        "question": "Which values does a WPS define, and which records must name the governing edition?",
        "answer": ("The WPS defines ranges for essential variables; each range is recorded before "
                  "qualification; general qualification rules come from a governing standard; "
                  "qualification records must reference that standard's edition."),
        "answer_clauses": [
            {"text": "The WPS defines ranges for essential variables.", "fact_id": "A"},
            {"text": "Each range is recorded before qualification.", "fact_id": "A2"},
            {"text": "General qualification rules come from a governing standard.", "fact_id": "B"},
            {"text": "Qualification records must reference that standard's edition.", "fact_id": "B2"},
        ],
        "gold_fact_ids": ["A", "A2", "B", "B2"],
        "necessity": {"nli_singles": [0.1, 0.1, 0.1, 0.1], "nli_joint": 0.97},
    }


def _v1_bridge_facts():
    # the shared _facts() only has the bridge surface in fact B (fine for
    # cluster_2plus2, which checks bridge_hidden(question) not fact presence);
    # verify_qa_pairs' v1 path additionally requires the bridge surface be
    # present in BOTH source facts (_bridge_present), so this fixture is local.
    return [
        {"id": "A", "text": "The welding procedure specification defines ranges for essential variables per ISO 15607."},
        {"id": "B", "text": "ISO 15607 defines the general rules for specification and qualification."},
    ]


def _bridge_v1_qa():
    return {
        "qa_id": "V2Q00003",
        "hop_type": "bridge",
        "evidence_strategy": "bridge_pair_v1",
        "bridge_entity": "ISO 15607",
        "question": "Which values does the specification define, and where do the general rules come from?",
        "answer": "The WPS defines ranges for essential variables; general qualification rules come from a governing standard.",
        "answer_clauses": [
            {"text": "The WPS defines ranges for essential variables.", "fact_id": "A"},
            {"text": "General qualification rules come from a governing standard.", "fact_id": "B"},
        ],
        "gold_fact_ids": ["A", "B"],
        "necessity": {"passed": True, "nli_a": 0.1, "nli_b": 0.1},
    }


def _make_accepting_nli(clause_texts):
    """Scripted NLI keyed on input SHAPE, not fuzzy text matching:
    joint premise = concatenation of >=2 fact sentences (>=2 periods);
    pairwise duplicate-check premise = one of the known CLAUSE strings
    (never a fact text); anything else = a (fact_text, its own clause)
    entailment check."""
    clause_set = set(clause_texts)

    def nli(pairs):
        scores = []
        for premise, hypothesis in pairs:
            if premise.count(".") >= 2:
                scores.append(0.95)
            elif premise in clause_set:
                scores.append(0.05)
            else:
                scores.append(0.92)
        return scores

    return nli


_ALL_CLAUSES = [
    "The WPS defines ranges for essential variables.",
    "Each range is recorded before qualification.",
    "General qualification rules come from a governing standard.",
    "Qualification records must reference that standard's edition.",
]
_accepting_nli = _make_accepting_nli(_ALL_CLAUSES)


def test_neighbor_pair_v2_deterministic_pass():
    result = verify_v2_pairs([_neighbor_pair_qa()], _facts(), nli_fn=_accepting_nli, settings=SETTINGS)
    verify = result["pairs"][0]["verify"]
    assert verify["verdict"] == "PASS"
    assert verify["faithful"] is True
    assert verify["duplicate"] is False
    assert verify["judge_used"] is False
    assert result["stats"]["n_v2_native"] == 1
    assert result["stats"]["n_v1_bridge_routed"] == 0


def test_cluster_2plus2_deterministic_pass():
    result = verify_v2_pairs([_cluster_qa()], _facts(), nli_fn=_accepting_nli, settings=SETTINGS)
    verify = result["pairs"][0]["verify"]
    assert verify["verdict"] == "PASS"
    assert verify["faithful"] is True
    assert verify["bridge_hidden"] is True
    assert len(verify["scores"]["clauses"]) == 4


def test_bridge_pair_v1_routes_to_existing_verifier():
    def five_input_nli(pairs):
        assert len(pairs) == 5  # the v1 verifier's exact fixed layout
        return [0.98, 0.97, 0.96, 0.10, 0.12]

    result = verify_v2_pairs([_bridge_v1_qa()], _v1_bridge_facts(), nli_fn=five_input_nli, settings=SETTINGS)
    verify = result["pairs"][0]["verify"]
    assert verify["verdict"] == "PASS"
    assert "bridge_in_both" in verify  # proves it went through verify_qa_pairs, not the n-ary path
    assert result["stats"]["n_v1_bridge_routed"] == 1
    assert result["stats"]["n_v2_native"] == 0


def test_cluster_bridge_leak_rejected_before_nli():
    qa = _cluster_qa()
    qa["question"] = "Which values relate to ISO 15607's governing edition?"  # leaks the bridge
    calls = []

    def counting_nli(pairs):
        calls.extend(pairs)
        return _accepting_nli(pairs)

    result = verify_v2_pairs([qa], _facts(), nli_fn=counting_nli, settings=SETTINGS)
    verify = result["pairs"][0]["verify"]
    assert verify["verdict"] == "REJECT"
    assert verify["reason"] == "bridge_leaked"
    assert calls == []  # rejected before any NLI call


def test_invalid_answer_clauses_length_rejected():
    qa = _neighbor_pair_qa()
    qa["answer_clauses"] = qa["answer_clauses"][:1]  # only 1 clause for a 2-fact strategy
    result = verify_v2_pairs([qa], _facts(), nli_fn=_accepting_nli, settings=SETTINGS)
    verify = result["pairs"][0]["verify"]
    assert verify["verdict"] == "REJECT"
    assert verify["reason"] == "invalid_answer_clauses"


def test_missing_fact_rejected():
    qa = _neighbor_pair_qa()
    qa["gold_fact_ids"] = ["A", "ZZZ_MISSING"]
    result = verify_v2_pairs([qa], _facts(), nli_fn=_accepting_nli, settings=SETTINGS)
    verify = result["pairs"][0]["verify"]
    assert verify["verdict"] == "REJECT"
    assert verify["reason"] == "missing_fact"


def test_v1_bridge_judge_reason_is_bucketed_not_raw_llm_text():
    """A judge-escalated bridge_pair_v1 row's stats-level reason must be the
    bucketed 'judge_pass'/'judge_fix'/'judge_reject' label, not the LLM's
    free-text justification (which lives on the per-pair verify['reason']
    field, not the aggregate histogram)."""
    class BorderlineJudgeLLM:
        def generate_json(self, prompt, temperature=0.0, max_tokens=0):
            return {"verdict": "PASS", "reason": "The answer is fully supported by both sources."}

    settings = dict(SETTINGS)
    settings["use_llm_borderline"] = True

    def borderline_nli(pairs):
        # own_a/own_b/joint right at the clause_entailment_min/joint_answer_min
        # margin so `borderline` is True and the judge fires.
        return [0.65, 0.65, 0.85, 0.10, 0.12]

    result = verify_v2_pairs(
        [_bridge_v1_qa()], _v1_bridge_facts(), llm=BorderlineJudgeLLM(),
        nli_fn=borderline_nli, settings=settings,
    )
    assert result["stats"]["reasons"].get("judge_pass") == 1
    assert not any(len(k) > 40 for k in result["stats"]["reasons"])  # no raw LLM sentences


def test_mixed_batch_preserves_order_and_aggregates_stats():
    pairs = [_neighbor_pair_qa(), _bridge_v1_qa(), _cluster_qa()]

    def mixed_nli(inputs):
        # route by exact input count seen so far isn't reliable across calls;
        # bridge_verifier and verify_v2 each call nli_fn once per routed group,
        # so this must handle whichever group size arrives.
        if len(inputs) == 5:
            return [0.98, 0.97, 0.96, 0.10, 0.12]
        return _accepting_nli(inputs)

    result = verify_v2_pairs(pairs, _facts() + _v1_bridge_facts(), nli_fn=mixed_nli, settings=SETTINGS)
    assert [row["qa_id"] for row in result["pairs"]] == [
        "V2Q00001", "V2Q00003", "V2Q00002",
    ]
    assert result["stats"]["n_input"] == 3
    assert result["stats"]["n_v1_bridge_routed"] == 1
    assert result["stats"]["n_v2_native"] == 2
    assert sum(result["stats"]["verdicts"].values()) == 3
