"""APILLM must keep the provider's own token counts.

Every OpenAI-compatible response carries a `usage` block counted by the
provider's tokenizer. It used to be parsed and discarded, which forced callers
to guess with `len(text) // 4` -- a heuristic that is badly wrong on ISO text,
which is dense with numbers, units and standard identifiers.

See docs/COST_TRACKING_IS_OPTIONAL.md.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from rag_gt.core.llm import APILLM


def _response(content: str = "hello", usage: dict | None = None, status: int = 200):
    class FakeResponse:
        status_code = status
        text = ""
        headers: dict = {}

        def json(self):
            body: dict = {
                "choices": [{"message": {"content": content}}],
            }
            if usage is not None:
                body["usage"] = usage
            return body

    return FakeResponse()


def _client() -> APILLM:
    return APILLM(base_url="https://example.test/v1", api_key="k", model="m")


def test_usage_block_is_recorded():
    llm = _client()
    usage = {"prompt_tokens": 1234, "completion_tokens": 56, "total_tokens": 1290}
    with patch.object(llm._session, "post", return_value=_response(usage=usage)):
        out = llm.generate("hi")

    assert out == "hello"
    assert llm.last_usage == {
        "prompt_tokens": 1234,
        "completion_tokens": 56,
        "total_tokens": 1290,
    }
    assert llm.usage_totals["total_tokens"] == 1290
    assert llm.usage_totals["calls_with_usage"] == 1


def test_usage_totals_accumulate_across_calls():
    llm = _client()
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    with patch.object(llm._session, "post", return_value=_response(usage=usage)):
        llm.generate("a")
        llm.generate("b")
        llm.generate("c")

    assert llm.usage_totals["total_tokens"] == 45
    assert llm.usage_totals["prompt_tokens"] == 30
    assert llm.usage_totals["calls_with_usage"] == 3


def test_missing_usage_block_is_recorded_as_absent_not_zero():
    """An endpoint that omits usage must be distinguishable from a free one."""
    llm = _client()
    with patch.object(llm._session, "post", return_value=_response(usage=None)):
        llm.generate("hi")

    assert llm.last_usage is None
    assert llm.usage_totals["calls_without_usage"] == 1
    assert llm.usage_totals["total_tokens"] == 0


def test_total_tokens_is_derived_when_the_provider_omits_it():
    llm = _client()
    with patch.object(
        llm._session,
        "post",
        return_value=_response(usage={"prompt_tokens": 7, "completion_tokens": 3}),
    ):
        llm.generate("hi")

    assert llm.last_usage["total_tokens"] == 10


@pytest.mark.parametrize("usage", ["not-a-dict", 42, {"prompt_tokens": "abc"}])
def test_malformed_usage_does_not_break_generation(usage):
    llm = _client()
    with patch.object(llm._session, "post", return_value=_response(usage=usage)):
        out = llm.generate("hi")

    assert out == "hello", "a bad usage block must never fail the call"
    assert llm.last_usage is None
