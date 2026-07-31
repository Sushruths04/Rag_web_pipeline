"""Stage C acceptance tests. All LLM and NLI behavior is stubbed ($0)."""

from rag_gt.generation.answer_first import (
    bridge_is_hidden,
    build_answer_first_pairs,
)


class FakeLLM:
    def generate_json(self, prompt, temperature=0.0, max_tokens=0):
        if "MODE: BRIDGE" in prompt:
            return {
                "clause_a": "Preheating temperature is classified as an essential variable",
                "clause_b": "A change beyond the qualified range requires requalification",
                "question": "Which temperature classification triggers requalification after a change beyond its qualified range?",
            }
        if "Preheating temperature" in prompt:
            question = "How is preheating temperature classified for qualification purposes?"
        else:
            question = "What consequence follows a change beyond the qualified range?"
        return {"question": question}


def _facts():
    return [
        {
            "id": "F1",
            "doc": "d",
            "text": "Preheating temperature is classified as an essential variable.",
            "page": 13,
        },
        {
            "id": "F2",
            "doc": "d",
            "text": "A change in an essential variable beyond the qualified range requires requalification.",
            "page": 16,
        },
    ]


def _pairs():
    return [
        {
            "pair_id": "BP1",
            "doc": "d",
            "fact_a": "F1",
            "fact_b": "F2",
            "bridge_entity": "essential variable",
            "bridge_norm": "essential variable",
            "pages": [13, 16],
        }
    ]


def _passing_nli(inputs):
    # Five scores per bridge: clause A, clause B, A-only answer, B-only answer,
    # joint answer. The exact ordering is part of the Stage C contract.
    assert len(inputs) == 5
    return [0.97, 0.96, 0.12, 0.20, 0.93]


def _passing_necessity(answer, facts, threshold=None):
    assert len(facts) == 2
    assert threshold == 0.5
    return {
        "necessity_score": 1.0,
        "necessary_fact_ids": ["F1", "F2"],
        "loo_entailment": [0.20, 0.12],
    }


def test_bridge_answer_first_passes_and_has_grounding_contract():
    result = build_answer_first_pairs(
        _facts(),
        _pairs(),
        FakeLLM(),
        include_singles=False,
        workers=1,
        nli_fn=_passing_nli,
        necessity_fn=_passing_necessity,
    )
    assert result["stats"]["n_bridge_pass"] == 1
    qa = result["pairs"][0]
    assert qa["hop_type"] == "bridge"
    assert qa["gold_fact_ids"] == ["F1", "F2"]
    assert qa["gold_chunk_ids"] == ["fact:F1", "fact:F2"]
    assert qa["gold_pages"] == [13, 16]
    assert qa["necessity"]["passed"] is True
    assert qa["necessity"]["nli_joint"] == 0.93
    assert qa["verify"]["verdict"] == "PENDING_STAGE_D"
    assert bridge_is_hidden(qa["question"], qa["bridge_entity"])


def test_rejects_pair_when_answer_is_not_jointly_necessary():
    def topical_nli(_inputs):
        return [0.97, 0.96, 0.82, 0.20, 0.93]

    result = build_answer_first_pairs(
        _facts(),
        _pairs(),
        FakeLLM(),
        include_singles=False,
        workers=1,
        nli_fn=topical_nli,
        necessity_fn=_passing_necessity,
    )
    assert result["stats"]["n_bridge_pass"] == 0
    assert result["stats"]["bridge_rejection_reasons"] == {
        "not_jointly_necessary": 1
    }


def test_bridge_hidden_normalizes_case_and_punctuation():
    assert not bridge_is_hidden("What does ISO-15607 require?", "ISO 15607")
    assert bridge_is_hidden("What does the qualification standard require?", "ISO 15607")


def test_single_control_is_emitted_with_one_gold_fact():
    result = build_answer_first_pairs(
        _facts(),
        [],
        FakeLLM(),
        include_singles=True,
        workers=1,
        nli_fn=lambda _inputs: [],
        necessity_fn=_passing_necessity,
    )
    assert result["stats"]["n_single_pass"] == 2
    assert all(pair["hop_type"] == "single" for pair in result["pairs"])
    assert all(len(pair["gold_fact_ids"]) == 1 for pair in result["pairs"])
