import pytest

from rag_gt.core.llm import APIError, APILLM, get_llm

_MODEL_ENV_VARS = (
    "RAG_LLM_CHAT_MODEL",
    "API_GT_MODEL",
    "API_ANSWER_MODEL",
    "API_BASE_URL",
    "API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_model_env(monkeypatch):
    """Isolate get_llm() precedence tests from any real .env / shell env.

    Without this, a developer's own .env (which sets these same names) would
    silently change which branch of get_llm()'s precedence logic runs.
    """
    for name in _MODEL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("API_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("API_KEY", "secret")


class _RateLimitedResponse:
    status_code = 429
    headers = {"Retry-After": "30"}
    text = ""


def test_api_retry_controls_can_be_reduced_for_paid_probe(monkeypatch):
    monkeypatch.setenv("API_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("API_REQUEST_TIMEOUT_SECONDS", "11")
    monkeypatch.setenv("API_RETRY_SLEEP_CAP_SECONDS", "2")

    llm = APILLM("https://provider.example/v1", "secret", "model-name")

    assert llm.max_attempts == 1
    assert llm.request_timeout_seconds == 11.0
    assert llm.retry_sleep_cap_seconds == 2.0


def test_rate_limit_on_final_attempt_does_not_sleep(monkeypatch):
    monkeypatch.setenv("API_MAX_ATTEMPTS", "1")
    llm = APILLM("https://provider.example/v1", "secret", "model-name")
    monkeypatch.setattr(llm._session, "post", lambda *args, **kwargs: _RateLimitedResponse())
    sleep_calls = []
    monkeypatch.setattr(
        "rag_gt.core.llm._backoff_sleep",
        lambda *args, **kwargs: sleep_calls.append((args, kwargs)),
    )

    with pytest.raises(APIError, match="rate-limited"):
        llm.generate("test")

    assert sleep_calls == []


# ---------- get_llm() model precedence ----------
#
# Pinned intent (see get_llm()'s docstring in rag_gt/core/llm.py):
# RAG_LLM_CHAT_MODEL, when set, is used for EVERY role and wins over the
# per-role API_GT_MODEL / API_ANSWER_MODEL fallbacks. The per-role variables
# are only consulted once RAG_LLM_CHAT_MODEL is unset/blank. This is a
# deliberate later simplification (single global chat-model knob), not an
# accidental bug — .env.example and the README document this precedence so
# API_GT_MODEL/API_ANSWER_MODEL don't look like independent knobs. These
# tests pin the behavior so it cannot silently regress.


def test_global_chat_model_wins_over_gt_role_override(monkeypatch):
    monkeypatch.setenv("RAG_LLM_CHAT_MODEL", "global-model")
    monkeypatch.setenv("API_GT_MODEL", "gt-only-model")

    llm = get_llm("gt")

    assert llm.model == "global-model"


def test_global_chat_model_wins_over_answer_role_override(monkeypatch):
    monkeypatch.setenv("RAG_LLM_CHAT_MODEL", "global-model")
    monkeypatch.setenv("API_ANSWER_MODEL", "answer-only-model")

    llm = get_llm("answer")

    assert llm.model == "global-model"


def test_gt_role_override_used_when_global_chat_model_unset(monkeypatch):
    monkeypatch.setenv("API_GT_MODEL", "gt-only-model")
    monkeypatch.setenv("API_ANSWER_MODEL", "answer-only-model")

    llm = get_llm("gt")

    assert llm.model == "gt-only-model"


def test_answer_role_override_used_when_global_chat_model_unset(monkeypatch):
    monkeypatch.setenv("API_GT_MODEL", "gt-only-model")
    monkeypatch.setenv("API_ANSWER_MODEL", "answer-only-model")

    llm = get_llm("answer")

    assert llm.model == "answer-only-model"


def test_get_llm_raises_clearly_when_no_model_configured():
    with pytest.raises(ValueError, match="No chat model configured"):
        get_llm("gt")
