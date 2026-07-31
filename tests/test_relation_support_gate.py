"""Regression suite for the Question Relation-Support Gate (v13).

LLM responses are stubbed with canned JSON so the test runs hermetically.
The five negative fixtures come verbatim from the source audit
`docs/graph_based_synthetic_qa_grounding_audit.md` §10. The five positives
are constructed to exercise each non-rejecting code path.

A live-LLM smoke test is gated behind `@pytest.mark.live_llm` and skipped
by default; opt in locally with `pytest -m live_llm`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pytest

from rag_gt.core.types import Fact, Span
from rag_gt.validation import relation_support_gate as rsg
from rag_gt.validation.relation_support_gate import (
    RELATIONAL_REQUIRES_LINK,
    RelationVerdict,
    relation_support_gate,
)


# ---------- helpers ----------


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Each test gets its own QRSG cache file so they don't poison each other."""
    cache_path = tmp_path / "qrsg_cache.db"
    monkeypatch.setattr(rsg, "_QRSG_CACHE_PATH", cache_path)
    monkeypatch.setattr(rsg, "_thread_local", _FreshThreadLocal())
    yield


class _FreshThreadLocal:
    """Stand-in for threading.local() that is fresh per test."""
    pass


class StubLLM:
    """Deterministic LLM stub. `.generate()` pops from `responses` in order."""

    def __init__(self, responses: List[str], model: str = "stub-qrsg-llm"):
        self.responses = list(responses)
        self.model = model
        self.calls: List[str] = []

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        self.calls.append(prompt)
        if not self.responses:
            return ""
        return self.responses.pop(0)


def _fact(fact_id: str, text: str, role: str = "definition") -> Fact:
    return Fact(
        fact_id=fact_id,
        text=text,
        role=role,
        supporting_spans=[
            Span(doc_id="doc", chunk_id="doc_c000000", start_token=0, end_token=1)
        ],
        canonical_form=text,
        raw_text=text,
    )


def _canned(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


# ---------- negative fixtures (verbatim from audit §10) ----------


def test_neg_q004_comparison_without_linking_fact():
    """q004 — 'distinguishing characteristic of on-policy methods compared with...'."""
    facts = [
        _fact("F2131", "On-policy methods estimate a policy value while using that policy for control."),
        _fact("F1910", "Values are useful for suggesting a policy when each action value is explicitly estimated."),
        _fact("F2521", "A conventional action-value function maps positions and moves to an estimated value."),
    ]
    question = (
        "What is the distinguishing characteristic of on-policy methods compared with "
        "a conventional action-value function regarding how they estimate values and use them for control?"
    )
    llm = StubLLM([_canned({
        "relation_type": "comparison",
        "concepts": ["on-policy methods", "conventional action-value function"],
        "evidence_map": {
            "on-policy methods": [
                {"fact_id": "F2131", "quote": "On-policy methods estimate a policy value"}
            ],
            "conventional action-value function": [
                {"fact_id": "F2521", "quote": "A conventional action-value function maps positions and moves"}
            ],
        },
        "linking_fact_id": None,
        "unsupported_concepts": [],
        "unsupported_relation": True,
        "reason": "No single fact connects on-policy methods with the conventional action-value function.",
    })])

    v = relation_support_gate(question, facts, llm)
    assert v.accepted is False
    assert v.reason == "unsupported_relation"
    assert v.relation_type == "comparison"
    assert v.risky_frame_hit == "comparison"


def test_neg_q006_mechanism_no_handling_fact():
    """q006 — 'How does backgammon handle its large branching factor...'."""
    facts = [
        _fact("F4082", "Backgammon involves several further complications, but the basic idea remains."),
        _fact("F3394", "Backgammon has a large branching factor, yet moves must be made within a few seconds."),
    ]
    question = (
        "How does backgammon handle its large branching factor while still requiring "
        "each move to be made within only a few seconds?"
    )
    llm = StubLLM([_canned({
        "relation_type": "mechanism",
        "concepts": ["backgammon handling large branching factor"],
        "evidence_map": {
            "backgammon": [
                {"fact_id": "F3394", "quote": "Backgammon has a large branching factor"}
            ],
        },
        "linking_fact_id": None,
        "unsupported_concepts": ["how backgammon handles its large branching factor"],
        "unsupported_relation": True,
        "reason": "No fact explains a handling mechanism.",
    })])

    v = relation_support_gate(question, facts, llm)
    assert v.accepted is False
    # Either unsupported_relation or unsupported_concepts depending on which check trips first;
    # both indicate the same root cause. Order in _verify: relation first.
    assert v.reason == "unsupported_relation"
    assert v.risky_frame_hit == "mechanism_handle"


def test_neg_q021_intersection_no_linking_fact():
    """q021 — 'Which people are mentioned in both acknowledgments...'."""
    facts = [
        _fact("F0070", "List of people who read drafts: Smith, Jones, Brown, Wilson."),
        _fact("F0066", "List of contributors who helped develop the overall view: Jones, Brown, Davis."),
    ]
    question = (
        "Which people are mentioned in both acknowledgments as draft readers "
        "and as contributors who helped develop the overall view presented in the book?"
    )
    llm = StubLLM([_canned({
        "relation_type": "intersection",
        "concepts": ["draft readers", "contributors"],
        "evidence_map": {
            "draft readers": [{"fact_id": "F0070", "quote": "List of people who read drafts"}],
            "contributors": [{"fact_id": "F0066", "quote": "List of contributors who helped develop the overall view"}],
        },
        "linking_fact_id": None,
        "unsupported_concepts": [],
        "unsupported_relation": False,
        "reason": "Two independent lists; no explicit intersection fact.",
    })])

    v = relation_support_gate(question, facts, llm)
    assert v.accepted is False
    assert v.reason == "risky_frame_no_link:intersection"
    assert v.risky_frame_hit == "intersection"


def test_neg_q027_causal_implicit_relation():
    """q027 — 'How does a decreasing epsilon-greedy strategy influence...'."""
    facts = [
        _fact("F0950", "Maximizing immediate reward can reduce access to future rewards."),
        _fact("F0200", "If an action selected by the policy is followed by low reward, the policy may change to select another action."),
        _fact("F4559", "Actions were selected by an epsilon-greedy policy with epsilon decreasing during learning."),
    ]
    question = (
        "How does a decreasing epsilon-greedy exploration strategy influence the policy's "
        "tendency to switch away from actions that yield low immediate reward but may reduce future returns?"
    )
    llm = StubLLM([_canned({
        "relation_type": "causal",
        "concepts": ["decreasing epsilon", "policy switching"],
        "evidence_map": {
            "decreasing epsilon": [
                {"fact_id": "F4559", "quote": "epsilon decreasing during learning"}
            ],
            "policy switching": [
                {"fact_id": "F0200", "quote": "the policy may change to select another action"}
            ],
        },
        "linking_fact_id": None,
        "unsupported_concepts": [],
        "unsupported_relation": True,
        "reason": "No fact states the influence of decreasing epsilon on switching.",
    })])

    v = relation_support_gate(question, facts, llm)
    assert v.accepted is False
    assert v.reason == "unsupported_relation"


def test_neg_q033_comparison_quote_mismatch():
    """q033 — even when relation type is plausible, a non-substring quote forces reject."""
    facts = [
        _fact("F0516", "The discussion moves closer to the full reinforcement learning problem when the bandit problem becomes associative."),
        _fact("F0773", "The phrase associative reinforcement learning is reserved as a synonym for the full reinforcement learning problem."),
        _fact("F0711", "Associative search tasks are intermediate between the n-armed bandit problem and full reinforcement learning."),
    ]
    question = (
        "How does associative reinforcement learning differ from associative search tasks "
        "regarding their positions between the n-armed bandit problem and the full reinforcement learning problem?"
    )
    llm = StubLLM([_canned({
        "relation_type": "comparison",
        "concepts": ["associative reinforcement learning", "associative search"],
        "evidence_map": {
            "associative reinforcement learning": [
                {"fact_id": "F0773",
                 # NOTE: deliberately not a substring of F0773 — should trip quote_mismatch.
                 "quote": "associative RL is a different beast entirely"}
            ],
            "associative search": [
                {"fact_id": "F0711", "quote": "Associative search tasks are intermediate"}
            ],
        },
        "linking_fact_id": "F0711",
        "unsupported_concepts": [],
        "unsupported_relation": False,
        "reason": "Both endpoints share an axis; F0711 links them.",
    })])

    v = relation_support_gate(question, facts, llm)
    assert v.accepted is False
    assert v.reason.startswith("quote_mismatch:")


# ---------- positive fixtures ----------


def test_pos_descriptive_single_fact():
    facts = [_fact("F1000", "Sarsa is an on-policy TD control algorithm.")]
    llm = StubLLM([_canned({
        "relation_type": "definition",
        "concepts": ["Sarsa"],
        "evidence_map": {"Sarsa": [{"fact_id": "F1000", "quote": "Sarsa is an on-policy TD control algorithm"}]},
        "linking_fact_id": None,
        "unsupported_concepts": [],
        "unsupported_relation": False,
        "reason": "Direct definition.",
    })])
    v = relation_support_gate("What is Sarsa, the on-policy TD control algorithm described here?", facts, llm)
    assert v.accepted is True, v.reason
    assert v.relation_type == "definition"


def test_pos_mechanism_with_explicit_fact():
    facts = [
        _fact("F2000", "Backgammon handles its large branching factor by approximating the value function with a neural network."),
        _fact("F2001", "Backgammon positions are evaluated by TD-Gammon's network."),
    ]
    llm = StubLLM([_canned({
        "relation_type": "mechanism",
        "concepts": ["backgammon handling branching factor"],
        "evidence_map": {
            "backgammon handling branching factor": [
                {"fact_id": "F2000",
                 "quote": "approximating the value function with a neural network"}
            ],
        },
        "linking_fact_id": "F2000",
        "unsupported_concepts": [],
        "unsupported_relation": False,
        "reason": "F2000 explicitly states the mechanism.",
    })])
    v = relation_support_gate(
        "How does backgammon handle its large branching factor during training?", facts, llm,
    )
    assert v.accepted is True, v.reason
    assert v.linking_fact_id == "F2000"


def test_pos_causal_with_because_fact():
    facts = [
        _fact("F3000", "Off-policy methods diverge because the target policy and behaviour policy mismatch."),
    ]
    llm = StubLLM([_canned({
        "relation_type": "causal",
        "concepts": ["off-policy divergence"],
        "evidence_map": {
            "off-policy divergence": [
                {"fact_id": "F3000", "quote": "Off-policy methods diverge because the target policy"}
            ],
        },
        "linking_fact_id": "F3000",
        "unsupported_concepts": [],
        "unsupported_relation": False,
        "reason": "Direct causal fact present.",
    })])
    v = relation_support_gate("Why does off-policy learning sometimes diverge in practice?", facts, llm)
    assert v.accepted is True, v.reason


def test_pos_comparison_with_shared_axis():
    facts = [
        _fact("F4000", "Sarsa is on-policy whereas Q-learning is off-policy, both updating action-value estimates online."),
    ]
    llm = StubLLM([_canned({
        "relation_type": "comparison",
        "concepts": ["Sarsa", "Q-learning"],
        "evidence_map": {
            "Sarsa": [{"fact_id": "F4000", "quote": "Sarsa is on-policy"}],
            "Q-learning": [{"fact_id": "F4000", "quote": "Q-learning is off-policy"}],
        },
        "linking_fact_id": "F4000",
        "unsupported_concepts": [],
        "unsupported_relation": False,
        "reason": "F4000 names both endpoints on the same axis.",
    })])
    v = relation_support_gate(
        "How does Sarsa compare with Q-learning on the on-policy/off-policy axis?", facts, llm,
    )
    assert v.accepted is True, v.reason


def test_relational_linking_fact_must_ground_two_endpoints():
    """A relational linking fact id must connect endpoints, not just exist."""
    facts = [
        _fact("F1", "Alpha is an on-policy method."),
        _fact("F2", "Beta maps state-action pairs to values."),
    ]
    llm = StubLLM([_canned({
        "relation_type": "comparison",
        "concepts": ["Alpha", "Beta"],
        "evidence_map": {
            "Alpha": [{"fact_id": "F1", "quote": "Alpha is an on-policy method"}],
            "Beta": [{"fact_id": "F2", "quote": "Beta maps state-action pairs"}],
        },
        "linking_fact_id": "F1",
        "unsupported_concepts": [],
        "unsupported_relation": False,
        "reason": "F1 is incorrectly claimed as the comparison link.",
    })])

    v = relation_support_gate("How does Alpha compare with Beta?", facts, llm)
    assert v.accepted is False
    assert v.reason == "linking_fact_lacks_endpoints"


def test_pos_list_enumeration():
    facts = [
        _fact("F5000", "The four standard reinforcement learning elements are policy, reward, value, and model."),
    ]
    llm = StubLLM([_canned({
        "relation_type": "list",
        "concepts": ["reinforcement learning elements"],
        "evidence_map": {
            "reinforcement learning elements": [
                {"fact_id": "F5000",
                 "quote": "policy, reward, value, and model"}
            ],
        },
        "linking_fact_id": None,
        "unsupported_concepts": [],
        "unsupported_relation": False,
        "reason": "Single fact enumerates all items.",
    })])
    v = relation_support_gate("What are the standard reinforcement learning elements?", facts, llm)
    assert v.accepted is True, v.reason
    # list-type questions do NOT require a linking fact
    assert v.relation_type == "list"


# ---------- supplemental: structural / contract ----------


def test_trace_contract_carries_debug_fields():
    """Reviewer guardrail: every reject must carry enough to debug.

    Required: question (via context), facts (via fact_ids upstream), relation_type,
    unsupported_concepts, unsupported_relation, risky_frame_hit, llm_raw,
    final Python reason.
    """
    facts = [_fact("F9000", "An unrelated fact about a different topic entirely.")]
    llm_raw = _canned({
        "relation_type": "comparison",
        "concepts": ["a", "b"],
        "evidence_map": {},
        "linking_fact_id": None,
        "unsupported_concepts": ["a", "b"],
        "unsupported_relation": True,
        "reason": "nothing supports this",
    })
    llm = StubLLM([llm_raw])
    v = relation_support_gate("Why does Sarsa differ from Q-learning here?", facts, llm)

    # All required debug fields must be populated.
    assert v.accepted is False
    assert v.reason
    assert v.relation_type == "comparison"
    assert v.unsupported_relation is True
    assert v.unsupported_concepts == ["a", "b"]
    assert v.risky_frame_hit is not None
    assert v.llm_raw == llm_raw


def test_parse_error_short_circuits_to_reject():
    facts = [_fact("F0001", "Some fact text that is long enough.")]
    llm = StubLLM(["not json at all — model misbehaved"])
    v = relation_support_gate("What is described?", facts, llm)
    assert v.accepted is False
    assert v.reason == "parse_error"
    assert v.llm_raw == "not json at all — model misbehaved"


def test_empty_question_rejected_without_llm_call():
    facts = [_fact("F0001", "Some fact text that is long enough.")]
    llm = StubLLM([])
    v = relation_support_gate("   ", facts, llm)
    assert v.accepted is False
    assert v.reason == "empty_question"
    assert llm.calls == []


def test_no_facts_rejected_without_llm_call():
    llm = StubLLM([])
    v = relation_support_gate("Anything?", [], llm)
    assert v.accepted is False
    assert v.reason == "no_facts"
    assert llm.calls == []


def test_cache_hit_skips_llm_call(tmp_path, monkeypatch):
    facts = [_fact("F0001", "Sarsa is an on-policy TD control method here.")]
    payload = _canned({
        "relation_type": "definition",
        "concepts": ["Sarsa"],
        "evidence_map": {"Sarsa": [{"fact_id": "F0001", "quote": "Sarsa is an on-policy TD control method"}]},
        "linking_fact_id": None,
        "unsupported_concepts": [],
        "unsupported_relation": False,
        "reason": "ok",
    })
    llm1 = StubLLM([payload])
    v1 = relation_support_gate("What is Sarsa as described in this passage?", facts, llm1)
    assert v1.accepted is True

    # Second call with the same (question, facts, model_id, gate_version) → cache hit.
    llm2 = StubLLM([])  # empty: would return "" → parse error if it called.
    v2 = relation_support_gate("What is Sarsa as described in this passage?", facts, llm2)
    assert v2.accepted is True
    assert llm2.calls == []


def test_unknown_relation_type_rejected():
    facts = [_fact("F0001", "Some long enough fact text here.")]
    llm = StubLLM([_canned({
        "relation_type": "vibes-based",
        "concepts": [],
        "evidence_map": {},
        "linking_fact_id": None,
        "unsupported_concepts": [],
        "unsupported_relation": False,
        "reason": "n/a",
    })])
    v = relation_support_gate("What is described in this passage?", facts, llm)
    assert v.accepted is False
    assert v.reason.startswith("unknown_relation_type:")


# ---------- live LLM smoke (skipped by default) ----------


@pytest.mark.live_llm
def test_live_llm_smoke():
    """Manual sanity check against the configured LLM. Run with `pytest -m live_llm`."""
    from rag_gt.core.llm import get_llm

    llm = get_llm("gt")
    facts = [_fact("F0001", "Sarsa is an on-policy temporal-difference control algorithm.")]
    v = relation_support_gate("What is Sarsa described as in this passage?", facts, llm)
    # We do not assert v.accepted is True — only that the call returns a structured verdict.
    assert isinstance(v, RelationVerdict)
    assert v.relation_type in {
        "descriptive", "definition", "comparison", "contrast", "causal",
        "mechanism", "intersection", "temporal", "quantitative", "procedural", "list",
    }
