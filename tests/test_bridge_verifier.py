"""Stage D deterministic cascade tests ($0)."""

from rag_gt.validation.bridge_verifier import verify_qa_pairs


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
        {"id": "F1", "text": "Preheating temperature is an essential variable."},
        {"id": "F2", "text": "Changing an essential variable requires requalification."},
    ]


def _qa(question="Which temperature classification triggers requalification?"):
    return {
        "qa_id": "Q1",
        "hop_type": "bridge",
        "bridge_entity": "essential variable",
        "question": question,
        "answer": "Preheating temperature has the classification; changing it requires requalification.",
        "answer_clauses": [
            {"text": "Preheating temperature has the classification.", "fact_id": "F1"},
            {"text": "Changing it requires requalification.", "fact_id": "F2"},
        ],
        "gold_fact_ids": ["F1", "F2"],
        "necessity": {"passed": True, "nli_a": 0.1, "nli_b": 0.2},
    }


def _passing_nli(inputs):
    assert len(inputs) == 5
    return [0.98, 0.97, 0.96, 0.10, 0.12]


def test_deterministic_pass_records_full_cascade():
    result = verify_qa_pairs(
        [_qa()], _facts(), nli_fn=_passing_nli, settings=SETTINGS
    )
    verify = result["pairs"][0]["verify"]
    assert verify["verdict"] == "PASS"
    assert verify["bridge_in_both"] is True
    assert verify["bridge_hidden"] is True
    assert verify["duplicate"] is False
    assert verify["faithful"] is True
    assert verify["judge_used"] is False


def test_bridge_leak_rejects_before_nli():
    result = verify_qa_pairs(
        [_qa("What does the essential variable require?")],
        _facts(),
        nli_fn=lambda inputs: (_ for _ in ()).throw(AssertionError(inputs)),
        settings=SETTINGS,
    )
    assert result["pairs"][0]["verify"]["verdict"] == "REJECT"
    assert result["pairs"][0]["verify"]["reason"] == "bridge_leaked"


def test_duplicate_clauses_rejected():
    qa = _qa()
    qa["answer_clauses"][1]["text"] = qa["answer_clauses"][0]["text"]
    result = verify_qa_pairs(
        [qa],
        _facts(),
        nli_fn=lambda _inputs: [0.98, 0.97, 0.96, 0.99, 0.99],
        settings=SETTINGS,
    )
    assert result["pairs"][0]["verify"]["reason"] == "duplicate_clauses"


def test_compact_bridge_variant_is_detected_as_leak():
    result = verify_qa_pairs(
        [{**_qa("What does ISO3834 require?"), "bridge_entity": "ISO 3834"}],
        [
            {"id": "F1", "text": "ISO 3834 defines quality requirements."},
            {"id": "F2", "text": "ISO 3834 includes selection criteria."},
        ],
        nli_fn=lambda inputs: (_ for _ in ()).throw(AssertionError(inputs)),
        settings=SETTINGS,
    )
    assert result["pairs"][0]["verify"]["reason"] == "bridge_leaked"
