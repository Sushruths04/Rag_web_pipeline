"""LLM answer generation from a question + its supporting facts.

Fact text is fenced inside `<<FACT>> ... <</FACT>>` so a fact whose body
contains literal `Question:` / `Answer:` cannot collide with the prompt
template. Auth and parse errors propagate; only transient errors are mapped
to the abstention output, and we tag the abstention with a marker so callers
can distinguish "model decided no info" from "all retries failed".

Phase 3 (plan v4) changes:
  - `max_tokens` is sourced from `configs/config.yaml`
    (`llm_options.num_predict_answer`, default 768), not hard-coded to 256.
    Reasoning models like gpt-oss-120b emit visible CoT before the answer,
    and 256 tokens was being consumed entirely by deliberation, leaving no
    final answer.
  - `_normalize_answer_output` is hardened to strip residual CoT patterns
    that the old whitelist missed (quoted-question restatements, "Look at
    facts:", "Need to use only provided facts"). `_INSUFFICIENT_PATTERNS`
    is now applied as a whole-line match, not a substring, so a partial
    answer that contains the phrase in passing is no longer demoted to
    abstention.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, List

from loguru import logger

from rag_gt.core.types import Fact

if TYPE_CHECKING:
    from rag_gt.core.llm import LLM

ABSTENTION_TEXT = "Insufficient information to answer."

_FENCE_OPEN = "<<FACT>>"
_FENCE_CLOSE = "<</FACT>>"
_INTERNAL_TOKENS = re.compile(re.escape(_FENCE_OPEN) + r"|" + re.escape(_FENCE_CLOSE))

# Substring-match: if any of these phrases appears anywhere in the model output
# AS A LEADING PHRASE, drop everything before "Answer:" (if present) or treat
# the whole thing as a meta-deliberation that didn't reach a final answer.
_META_PREFIXES = (
    "we need to answer",
    "the provided facts",
    "the facts supplied",
    "the given facts",
    "look through facts",
    "look at facts",
    "need to use only provided facts",
    "use only provided facts",
)

# Whole-line match (not substring): a line that consists of one of these
# phrases is an abstention. A partial answer that happens to contain the
# phrase in passing is NOT an abstention.
_INSUFFICIENT_LINE_PATTERNS = (
    r"^\s*insufficient information( to answer)?\.?\s*$",
    r"^\s*the (provided |given )?(facts|information) (do|does) not contain (any )?information.*$",
    r"^\s*cannot be answered from the (given|provided)( material| facts)?\.?\s*$",
    r"^\s*the question cannot be answered.*$",
    r"^\s*consequently,? the question cannot be answered.*$",
    r"^\s*none of the individuals listed.*$",
)
_INSUFFICIENT_LINE_RE = re.compile(
    "|".join(_INSUFFICIENT_LINE_PATTERNS), flags=re.IGNORECASE
)

# Reasoning models often begin by restating the question in quotes before
# starting to deliberate. Match a leading "..."? line (smart or straight
# quotes) and strip everything up to the first explicit "Answer:" marker.
_LEADING_QUOTED_QUESTION_RE = re.compile(
    r'^[\s ]*["“”‘’].*?\?["“”‘’][\s ]*',
    flags=re.DOTALL,
)

# Find the first "Answer:" or "Final answer:" marker (case-insensitive,
# whole-word for "Answer" so we don't catch "Answering").
_ANSWER_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s+)?answer\s*:\s*", flags=re.IGNORECASE
)


@lru_cache(maxsize=1)
def _answer_max_tokens() -> int:
    """Resolve answer-generation token budget from config.yaml.

    Falls back to 768 if the key is missing (e.g. legacy config). Cached so
    a config-edit during a long run is picked up only on next process start,
    which matches how config is otherwise loaded.
    """
    try:
        from rag_gt.core.config import load_config
        cfg = load_config()
        return int(cfg.get("llm_options", {}).get("num_predict_answer", 768))
    except Exception as e:  # config-read failures are non-fatal
        logger.warning(f"[AnsGen] could not read num_predict_answer from config: {e}; falling back to 768")
        return 768


def _sanitize_fact(text: str) -> str:
    """Strip our fence delimiters from fact text before fencing."""
    return _INTERNAL_TOKENS.sub("", text)


def _strip_leading_cot(text: str) -> str:
    """Remove leading CoT scaffolding (quoted-question restatement +
    meta-deliberation) before any explicit Answer: marker, if present.

    If no marker is present, leave the text alone — the heuristics below will
    handle it. Idempotent."""
    out = text

    # If there's an explicit "Answer:" / "Final answer:" marker, jump past
    # the LAST such marker so reasoning models that emit "Answer: ... wait,
    # actually... Final answer: ..." don't fool us.
    last_marker_end = None
    for m in _ANSWER_MARKER_RE.finditer(out):
        last_marker_end = m.end()
    if last_marker_end is not None:
        out = out[last_marker_end:].strip()
        return out

    # No explicit marker — strip a single leading quoted-question restatement
    # if present. Common pattern from gpt-oss-120b.
    out = _LEADING_QUOTED_QUESTION_RE.sub("", out, count=1).strip()
    return out


def _normalize_answer_output(text: str) -> str:
    out = (text or "").strip()
    if not out:
        return ABSTENTION_TEXT

    # Strip leading CoT scaffolding first. This must happen BEFORE the
    # insufficient-pattern check, because reasoning models often write
    # "the facts do not contain ..." inside their deliberation before
    # arriving at a real answer.
    out = _strip_leading_cot(out)
    if not out:
        return ABSTENTION_TEXT

    lowered = out.lower().lstrip()

    # If the cleaned output STARTS with a meta-deliberation prefix and no
    # explicit Answer: marker was found, treat it as abstention.
    if lowered.startswith(_META_PREFIXES):
        return ABSTENTION_TEXT

    # Keep only the first paragraph when the model rambles past the
    # answer (multi-paragraph output is a CoT-leak signature).
    paras = [p.strip() for p in re.split(r"\n\s*\n", out) if p.strip()]
    if paras:
        out = paras[0]

    if len(out) < 10:
        return ABSTENTION_TEXT

    # Whole-line abstention check: if ANY line of the cleaned output is a
    # bare abstention, treat the answer as abstention.
    for line in out.splitlines():
        if _INSUFFICIENT_LINE_RE.match(line):
            return ABSTENTION_TEXT

    # Complete-sentence polish: collapse internal newlines (extraction often
    # leaves mid-sentence line breaks) and ensure terminal punctuation.
    out = " ".join(out.split())
    if out and out[-1] not in ".?!":
        out += "."

    return out


def is_abstention(text: str) -> bool:
    """True when an answer is an abstention ('Insufficient information ...').

    Public helper so the pipeline can gate/regenerate pairs the answer model
    declined to answer (the question required information not in the facts).
    """
    if not text:
        return True
    t = text.strip()
    if t == ABSTENTION_TEXT:
        return True
    first = t.splitlines()[0] if t.splitlines() else t
    return bool(_INSUFFICIENT_LINE_RE.match(first))


_ANSWER_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "of", "and", "or", "but", "not", "with", "this", "that", "it",
    "its", "by", "from", "as", "into",
}


def _answer_token_overlap(answer: str, facts: List[Fact]) -> float:
    """Fraction of answer content tokens present in source facts.

    Returns a value in [0, 1]. Low overlap suggests the answer added
    outside-source content. Symbols and formulas are included as tokens.
    """
    def _tok(text: str) -> set:
        words = re.findall(r"[\w*(),.]+", (text or "").lower())
        return {w for w in words if len(w) >= 3 and w not in _ANSWER_STOPWORDS}

    fact_tokens = _tok(" ".join(f.text or "" for f in facts))
    answer_tokens = _tok(answer)
    if not answer_tokens:
        return 1.0
    return len(answer_tokens & fact_tokens) / len(answer_tokens)


def _build_answer_prompt(question: str, fenced: str, *, strict: bool = False) -> str:
    strictness = (
        "IMPORTANT — your previous answer shared too much phrasing with the question. "
        "Rewrite: keep every fact-grounded claim, but use different sentence structure "
        "and different word choices for anything that was also in the question. "
    ) if strict else ""
    return (
        "Answer the question using ONLY the information from the provided facts. "
        "Each fact appears between <<FACT>> and <</FACT>> markers.\n\n"
        "The question names a subject and asks for a specific property or relationship. "
        "Your answer must STATE that property's value or the relationship's outcome — "
        "drawn directly from the facts.\n\n"
        "The question and answer share the SUBJECT NAME. They must NOT share the "
        "property value, outcome, or causal description — that content belongs only "
        "in the answer, not in the question. If your answer repeats multi-word "
        "phrases from the question, rewrite it with your own sentence structure.\n\n"
        + strictness +
        "Format: 1-2 complete sentences. "
        "Begin with the technical subject (e.g. 'BERT', 'the Transformer', 'ISO 9606-1'). "
        "State the fact's claim directly. "
        "Do NOT add any information not in the facts. "
        "Do NOT start with 'The document states', 'According to the text', "
        "'Based on the provided facts', or any similar source-frame opener. "
        "Return exactly 'Insufficient information to answer.' when the facts do "
        "NOT jointly support the question's PREMISE — for example if the question "
        "assumes a relationship, attribution, or specific subject that no fact "
        "states. Do NOT answer just because one fact matches part of the question; "
        "a false or unsupported premise must be rejected, not partially answered.\n\n"
        "SELF-CHECK before outputting:\n"
        "(0) Is every assumption in the QUESTION (its subject, attribution, and any "
        "claimed relationship) supported by the facts? If NO — abstain.\n"
        "(1) Does every claim trace to a provided fact? YES required.\n"
        "(2) Does the answer share a multi-word phrase with the question (beyond the subject name)? "
        "If YES — rewrite that phrase in your own words.\n"
        "(3) Does the answer begin with the subject, not a clause that echoes the question? YES required.\n\n"
        f"Facts:\n{fenced}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_answer(question: str, facts: List[Fact], llm: "LLM") -> str:
    fenced = "\n".join(
        f"{_FENCE_OPEN}\n{_sanitize_fact(f.text).strip()}\n{_FENCE_CLOSE}"
        for f in facts
    )
    prompt = _build_answer_prompt(question, fenced, strict=False)

    # Import here so the module remains importable when the optional `requests`
    # / network stack isn't installed (used by some test paths).
    from rag_gt.core.llm import APIError

    try:
        result = llm.generate(
            prompt, temperature=0.1, max_tokens=_answer_max_tokens()
        ).strip()
    except APIError:
        # Auth / unrecoverable client error: re-raise so the caller can stop.
        raise
    except Exception as e:
        # Transient — log type and continue with abstention. Do NOT swallow
        # silently: callers reading drop stats need to know this case happened.
        logger.warning(f"[AnsGen] transient error from LLM: {type(e).__name__}: {e}")
        return ABSTENTION_TEXT

    answer = _normalize_answer_output(result)

    # Token-overlap reprompt: if answer tokens barely overlap with source facts
    # (< 30%), the model likely added outside content → retry with explicit
    # instruction to restate facts in own words (NOT verbatim copy).
    # Also reprompt if overlap is very high (> 0.90) AND answer is long (> 80
    # chars), which signals the model copy-pasted question phrasing back.
    overlap = _answer_token_overlap(answer, facts)
    needs_reprompt = (
        answer != ABSTENTION_TEXT
        and (overlap < 0.30 or (overlap > 0.90 and len(answer) > 80))
    )
    if needs_reprompt:
        logger.debug(f"[AnsGen] low token overlap — retrying with strict prompt")
        strict_prompt = _build_answer_prompt(question, fenced, strict=True)
        try:
            result2 = llm.generate(
                strict_prompt, temperature=0.0, max_tokens=_answer_max_tokens()
            ).strip()
            answer2 = _normalize_answer_output(result2)
            if answer2 != ABSTENTION_TEXT:
                answer = answer2
        except APIError:
            raise
        except Exception as e:
            logger.debug(f"[AnsGen] strict reprompt failed: {e}")

    return answer
