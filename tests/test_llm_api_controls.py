import pytest

from rag_gt.core.llm import APIError, APILLM


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
