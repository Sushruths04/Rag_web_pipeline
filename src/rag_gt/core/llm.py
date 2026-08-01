"""Unified LLM interface: Protocol + factory + Ollama / API adapters.

Pipeline modules must only import `LLM` and `get_llm` from this module.

Security note: this module never logs raw API keys or `Authorization` headers.
`__repr__` is overridden on adapters that hold credentials. If you add new
logging here, redact `self.api_key` and `self._headers` explicitly.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Optional, Protocol, runtime_checkable

import requests
from dotenv import load_dotenv
from json_repair import repair_json
from loguru import logger

load_dotenv()

# Suppress loguru's diagnose mode so local variables (which may include
# self.api_key / self._headers) are never embedded in tracebacks. Apps that
# want richer tracebacks should re-enable this themselves.
try:
    logger.configure(extra={})
    # `logger.add` callers can override; this only nudges defaults.
    logger.opt(exception=False)
except Exception:
    pass


@runtime_checkable
class LLM(Protocol):
    def generate(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 512
    ) -> str: ...


# ---------- JSON extraction helper ----------

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _extract_json(raw: str) -> str:
    """Strip code fences and locate the first balanced JSON object/array.

    Handles model output like 'Here is the JSON: ```json {...} ``` thanks!'.
    If no balanced JSON is found (truncated output), returns the incomplete
    fragment from the first '{' or '[' so repair_json can attempt recovery.
    """
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE).strip()
    obj_match = _JSON_OBJECT_RE.search(s)
    arr_match = _JSON_ARRAY_RE.search(s)
    if obj_match and arr_match:
        # Pick whichever starts earlier
        return obj_match.group(0) if obj_match.start() <= arr_match.start() else arr_match.group(0)
    if obj_match:
        return obj_match.group(0)
    if arr_match:
        return arr_match.group(0)
    # Truncated output — return fragment from first brace so repair_json can close it.
    for start_char, close_char in [('{', '}'), ('[', ']')]:
        idx = s.find(start_char)
        if idx != -1:
            return s[idx:] + close_char
    raise ValueError(f"No JSON object/array found in LLM output: {s[:200]!r}")


def _backoff_sleep(attempt: int, base: float = 1.0, cap: float = 30.0, retry_after: Optional[float] = None) -> None:
    """Exponential backoff with jitter, honouring server-supplied Retry-After."""
    if retry_after is not None and retry_after > 0:
        time.sleep(min(retry_after, cap))
        return
    delay = min(cap, base * (2 ** attempt)) + random.uniform(0.0, 0.5)
    time.sleep(delay)


# ---------- env helpers ----------


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.1) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default


# Ollama adapter removed — this pipeline is API-only (see get_llm).


# ---------- OpenAI-compatible HTTP API ----------


class APIError(RuntimeError):
    """Raised when the API returns an unrecoverable error (auth, bad URL, etc.)."""


class APILLM:
    """Adapter for any OpenAI-compatible chat-completions endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        if not base_url:
            raise ValueError("API_BASE_URL not set in .env")
        api_key = (api_key or "").strip()
        if not api_key:
            raise ValueError("API_KEY not set or empty/whitespace in .env")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_attempts = _env_int("API_MAX_ATTEMPTS", 5)
        self.request_timeout_seconds = _env_float("API_REQUEST_TIMEOUT_SECONDS", 120.0)
        self.retry_sleep_cap_seconds = _env_float("API_RETRY_SLEEP_CAP_SECONDS", 30.0)
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._session = requests.Session()
        # Provider-reported token usage. `last_usage` is the most recent call's
        # {prompt_tokens, completion_tokens, total_tokens} or None when the
        # endpoint omitted the block; the totals accumulate over the client's
        # lifetime. Callers should prefer these over any local estimate.
        self.last_usage: Optional[dict] = None
        self.usage_totals: dict = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls_with_usage": 0,
            "calls_without_usage": 0,
        }
        logger.info(f"[APILLM] model={self.model} @ {self.base_url}")

    def _record_usage(self, usage: object) -> None:
        """Store the provider's token counts for the call just completed."""
        if not isinstance(usage, dict):
            self.last_usage = None
            self.usage_totals["calls_without_usage"] += 1
            return
        try:
            prompt = int(usage.get("prompt_tokens") or 0)
            completion = int(usage.get("completion_tokens") or 0)
        except (TypeError, ValueError):
            self.last_usage = None
            self.usage_totals["calls_without_usage"] += 1
            return
        total = int(usage.get("total_tokens") or (prompt + completion))
        self.last_usage = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }
        self.usage_totals["prompt_tokens"] += prompt
        self.usage_totals["completion_tokens"] += completion
        self.usage_totals["total_tokens"] += total
        self.usage_totals["calls_with_usage"] += 1

    def __repr__(self) -> str:
        # Never include api_key or _headers in repr — they reach tracebacks.
        return f"APILLM(model={self.model!r}, base_url={self.base_url!r})"

    def generate(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_error: Optional[str] = None
        max_attempts = self.max_attempts
        for attempt in range(max_attempts):
            try:
                r = self._session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                    timeout=self.request_timeout_seconds,
                )
            except requests.exceptions.Timeout:
                last_error = "timeout"
                logger.warning(f"[APILLM] Timeout attempt {attempt + 1}/{max_attempts}")
                if attempt < max_attempts - 1:
                    _backoff_sleep(attempt, cap=self.retry_sleep_cap_seconds)
                continue
            except requests.exceptions.RequestException as e:
                last_error = f"network: {type(e).__name__}"
                logger.warning(f"[APILLM] Network error attempt {attempt + 1}/{max_attempts}: {type(e).__name__}")
                if attempt < max_attempts - 1:
                    _backoff_sleep(attempt, cap=self.retry_sleep_cap_seconds)
                continue

            # 401/403 are unrecoverable; do not retry.
            if r.status_code in (401, 403):
                raise APIError(
                    f"[APILLM] Auth failed ({r.status_code}). "
                    "Check API_KEY in .env. Response body redacted."
                )

            # 429 honours Retry-After when present.
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                retry_after = float(ra) if ra and ra.isdigit() else None
                last_error = "rate-limited"
                logger.warning(
                    f"[APILLM] Rate limited (attempt {attempt + 1}/{max_attempts}); "
                    f"retry_after={retry_after}"
                )
                if attempt < max_attempts - 1:
                    _backoff_sleep(
                        attempt,
                        base=2.0,
                        cap=self.retry_sleep_cap_seconds,
                        retry_after=retry_after,
                    )
                continue

            # 5xx: transient — retry with backoff.
            if 500 <= r.status_code < 600:
                last_error = f"server-{r.status_code}"
                logger.warning(f"[APILLM] {r.status_code} server error attempt {attempt + 1}/{max_attempts}")
                if attempt < max_attempts - 1:
                    _backoff_sleep(attempt, cap=self.retry_sleep_cap_seconds)
                continue

            # Other 4xx: surface body fragment, do not retry.
            if 400 <= r.status_code < 500:
                snippet = r.text[:300] if r.text else ""
                raise APIError(
                    f"[APILLM] {r.status_code} client error (not retried). Body: {snippet!r}"
                )

            # 2xx path
            try:
                result = r.json()
            except (ValueError, json.JSONDecodeError) as e:
                snippet = r.text[:300] if r.text else ""
                raise APIError(
                    f"[APILLM] Non-JSON response from {self.base_url}. "
                    f"Did you forget '/v1' on API_BASE_URL? Body: {snippet!r}"
                ) from e

            if not result or "choices" not in result or not result["choices"]:
                last_error = f"malformed-response"
                logger.error(f"[APILLM] Malformed response shape (attempt {attempt + 1}/{max_attempts})")
                if attempt < max_attempts - 1:
                    _backoff_sleep(attempt, cap=self.retry_sleep_cap_seconds)
                continue

            # Record the provider's OWN token counts before returning. They are
            # in every OpenAI-compatible response and were previously discarded,
            # which forced callers to guess with len(text)//4. The provider's
            # tokenizer is authoritative; a character heuristic is badly wrong on
            # ISO text, which is dense with numbers, units and identifiers.
            self._record_usage(result.get("usage"))

            message = result["choices"][0].get("message", {})
            content = message.get("content") or message.get("reasoning_content") or ""
            if not content:
                refusal = message.get("refusal")
                if refusal:
                    logger.warning(f"[APILLM] Model refused: {refusal}")
                    return ""  # treat refusal as empty output, not retryable
                last_error = "empty-content"
                if attempt < max_attempts - 1:
                    _backoff_sleep(attempt, cap=self.retry_sleep_cap_seconds)
                continue
            return content.strip()

        logger.error(f"[APILLM] All {max_attempts} attempts failed: {last_error}")
        raise APIError(f"[APILLM] All {max_attempts} attempts failed: {last_error}")

    def generate_json(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048
    ) -> dict:
        raw = self.generate(prompt, temperature, max_tokens)
        return json.loads(repair_json(_extract_json(raw)))

    def unload(self) -> None:
        return None


# ---------- Factory ----------


def active_backend() -> str:
    return os.getenv("LLM_BACKEND", "api").strip().lower()


def get_llm(role: str = "gt") -> LLM:
    """Return the API chat adapter. The model is controlled entirely from .env.

    ``RAG_LLM_CHAT_MODEL`` is the single chat model used for ALL roles. If it is
    unset, fall back to the role-specific ``API_GT_MODEL`` / ``API_ANSWER_MODEL``.
    """
    model = os.getenv("RAG_LLM_CHAT_MODEL", "").strip()
    if not model:
        model = (
            os.getenv("API_GT_MODEL", "")
            if role == "gt"
            else os.getenv("API_ANSWER_MODEL", "")
        ).strip()
    if not model:
        raise ValueError(
            "No chat model configured. Set RAG_LLM_CHAT_MODEL in .env."
        )
    return APILLM(
        base_url=os.getenv("API_BASE_URL", ""),
        api_key=os.getenv("API_KEY", ""),
        model=model,
    )


def default_concurrency() -> int:
    """Default parallelism (override with RAG_GT_MAX_CONCURRENT_LLM_CALLS)."""
    override = os.getenv("RAG_GT_MAX_CONCURRENT_LLM_CALLS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return 8


def model_identifier(llm: LLM) -> str:
    """Stable identifier for cache keys.

    Raises if the adapter does not expose a `.model` attribute. This prevents
    silently caching every adapter under the same fake key.
    """
    model = getattr(llm, "model", None)
    if not model:
        raise AttributeError(
            f"LLM adapter {type(llm).__name__} has no .model attribute; cannot key cache."
        )
    return str(model)
