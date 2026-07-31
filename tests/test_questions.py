from __future__ import annotations

from rag_gt.core.types import Fact
from rag_gt.generation.questions import (
    _parse_candidates,
    _question_max_tokens,
    generate_question,
    premise_leakage_indices,
)


class _CapturingLLM:
    model = "fake-question"

    def __init__(self) -> None:
        self.prompt = ""
        self.max_tokens = 0

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 512) -> str:
        self.prompt = prompt
        self.max_tokens = max_tokens
        return "How does the eligibility trace connect TD error updates to later value changes?"


def _fact(fid: str, text: str, role: str = "definition") -> Fact:
    return Fact(fact_id=fid, text=text, role=role, supporting_spans=[])


def test_multihop_question_prompt_includes_verified_chain_edges() -> None:
    llm = _CapturingLLM()
    facts = [
        _fact("F001", "TD errors update value estimates after observed transitions."),
        _fact("F002", "Eligibility traces assign credit to recently visited states."),
    ]
    q = generate_question(
        facts,
        llm,  # type: ignore[arg-type]
        chain_edges=[
            {
                "src": "F001",
                "dst": "F002",
                "type": "causal",
                "relation_claim": "TD errors interact with eligibility traces to update recent states.",
                "bridging_quote": "Eligibility traces assign credit",
                "source_contribution": "TD errors update value estimates.",
                "destination_contribution": "Eligibility traces assign credit to recent states.",
                "question_seed": "How do TD errors and eligibility traces combine during value updates?",
                "question_seed_score": 1.0,
            }
        ],
    )
    assert q is not None
    assert "Support 1:" in llm.prompt
    assert "Support 1 -> Support 2" in llm.prompt
    assert "Verified relations between the facts" in llm.prompt
    assert "relation: causal" in llm.prompt
    assert "premise-safe question scaffold: How do TD errors and eligibility traces combine" in llm.prompt
    assert "TD errors interact with eligibility traces to update recent states." not in llm.prompt
    assert "bridge quote:" not in llm.prompt
    assert "Edge-type question shape guidance" in llm.prompt
    assert "For causal:" in llm.prompt
    assert "property and resulting practical consequence together" in llm.prompt
    assert "For two supports, prefer a two-part answer form" in llm.prompt
    assert "Required support anchors" in llm.prompt
    assert "Support 1: TD" in llm.prompt
    assert "Support 2: Eligibility" in llm.prompt
    assert "Support 1 contribution:" not in llm.prompt
    assert "Support 2 contribution:" not in llm.prompt
    assert "both supports contribute distinct answer content" in llm.prompt
    assert "Do not quote, paraphrase, or supply either support's answer content" in llm.prompt
    assert "answer must change if any one" in llm.prompt
    assert llm.max_tokens == _question_max_tokens()


def test_multihop_question_prompt_uses_comparison_shape_guidance() -> None:
    llm = _CapturingLLM()
    facts = [
        _fact("F001", "One-step TD methods update values after each transition."),
        _fact("F002", "Eligibility traces can learn faster when rewards are delayed."),
    ]
    generate_question(
        facts,
        llm,  # type: ignore[arg-type]
        chain_edges=[
            {
                "src": "F001",
                "dst": "F002",
                "type": "contrast",
                "relation_claim": "Eligibility traces differ from one-step methods in cost and delayed-reward speed.",
                "bridging_quote": "learn faster when rewards are delayed",
            }
        ],
    )
    assert "For contrast:" in llm.prompt
    assert "differ" in llm.prompt


def test_definition_condition_prompt_prefers_consequence_under_condition() -> None:
    llm = _CapturingLLM()
    facts = [
        _fact("F001", "The return is the function of future rewards an agent seeks to maximize."),
        _fact(
            "F002",
            "If each action influences only immediate reward, a myopic agent can maximize each reward separately.",
            role="condition",
        ),
    ]
    generate_question(facts, llm)  # type: ignore[arg-type]
    assert "Preferred multi-hop form for this support shape" in llm.prompt
    assert "what follows, applies, or should be done under that condition" in llm.prompt
    assert "do not repeat Support 1 as the premise" in llm.prompt


def test_question_parser_rejects_planning_artifact_with_question_mark() -> None:
    raw = 'Make sure at least 12 words. Sentence starts with "What". End with "?". No extra. Let\'s craft:'
    assert _parse_candidates(raw) == []
    assert _parse_candidates(
        "What mechanism connects TD errors with eligibility traces during value updates?"
    ) == ["What mechanism connects TD errors with eligibility traces during value updates?"]
    assert _parse_candidates(
        "Question: How do TD errors and eligibility traces interact during value updates?"
    ) == ["How do TD errors and eligibility traces interact during value updates?"]
    assert _parse_candidates(
        "Accordingly, how do TD errors and eligibility traces interact during value updates?"
    ) == ["How do TD errors and eligibility traces interact during value updates?"]


def test_premise_leakage_detects_long_support_restatement() -> None:
    facts = [
        _fact(
            "F001",
            "For some state s we would like to know whether we should change "
            "the policy to deterministically choose an action.",
        ),
        _fact(
            "F002",
            "If π is a deterministic policy, then following π observes returns "
            "only for one action from each state.",
        ),
    ]
    question = (
        "Why does determining whether to change the policy to deterministically "
        "choose an action lead to observing returns only for one action from each state?"
    )
    assert premise_leakage_indices(question, facts) == [0]


def test_premise_leakage_allows_compact_anchor_question() -> None:
    facts = [
        _fact("F001", "TD errors update value estimates after observed transitions."),
        _fact("F002", "Eligibility traces assign credit to recently visited states."),
    ]
    question = "How do TD errors and eligibility traces combine during value updates?"
    assert premise_leakage_indices(question, facts) == []
