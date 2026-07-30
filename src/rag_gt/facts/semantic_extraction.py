"""Semantic Fact Unit (SFU) extraction + upgrade utilities.

Phase 1 of the SFU+CGA+SAE plan (plan v6). Two entry points:

1. ``upgrade_fact_to_sfu(fact, context_text, llm, nli_model)`` — takes an
   existing single-sentence ``Fact`` and its surrounding chunk text, runs the
   canonical-form rewrite + NLI guard + self-containment scorer, and returns
   the same Fact with ``canonical_form``, ``self_containment_score``, and
   ``raw_text`` populated. This is the MVP path: cheap, works on existing GT.

2. ``extract_sfus_from_chunk(chunk_text, llm, nli_model)`` — full path:
   sends a chunk to an LLM segmenter that returns semantically-coherent
   spans (variable length: 1 sentence, 2 sentences, or a paragraph), then
   runs each span through the canonical-form + self-containment pipeline.
   For Phase 5 (full GT regeneration).

Both paths share the same canonical-form rewrite + NLI-guard helpers so the
research story is consistent: an SFU's ``canonical_form`` is always a rewrite
that is NLI-entailed by the verbatim source span — never adds content.

NLI threshold for the canonical-form guard defaults to 0.7. Self-containment
scores < ``MIN_SELF_CONTAINMENT_THRESHOLD`` (default 0.5) are flagged as
weak; the caller decides whether to drop or keep them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

from loguru import logger

from rag_gt.core.types import Fact

if TYPE_CHECKING:
    from rag_gt.core.llm import LLM


# ---- Thresholds (override per call if needed) ----

CANONICAL_NLI_THRESHOLD = 0.7
MIN_SELF_CONTAINMENT_THRESHOLD = 0.5
MAX_REWRITE_RETRIES = 3

# Sentinel passed as ``nli_model`` to ENABLE the canonical-rewrite NLI guard
# using the module-level self-loading NLI functions (the guard only needs a
# truthy gate; the actual entailment is computed by ``nli_batch``). Callers that
# genuinely have no NLI available pass ``nli_model=None`` to skip (unit tests).
# Production callers (the all-PDF pipeline) MUST pass this so a rewrite that
# changes meaning — modality, units, optionality — is rejected instead of being
# silently accepted with a synthetic perfect score.
USE_MODULE_NLI = object()


# Reasoning models (gpt-oss-120b) sometimes emit their chain-of-thought as the
# rewrite instead of the rewritten passage — e.g. "We need to rewrite the
# passage so it can be understood in isolation ... The passage is: ...". When
# the API returns an empty `content` channel, the LLM wrapper falls back to the
# raw `reasoning_content`, and that CoT then becomes a "fact". This guard
# rejects such meta-text so the verbatim passage is kept instead (passed=False),
# preventing the artefact at the source rather than filtering it downstream.
_COT_META_RE = re.compile(
    r"\b("
    r"we need to (?:rewrite|resolve|produce|make)|"
    r"let'?s (?:rewrite|resolve|craft)|"
    r"i (?:need|should|must|will) (?:rewrite|resolve|produce)|"
    r"the passage (?:is|reads|says)\s*:|"
    r"rewrite the passage|resolve pronouns,? (?:and )?discourse|"
    r"so it can be understood in isolation|"
    r"here is the (?:rewritten|rewrite)|the rewritten passage"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_cot_meta(text: str) -> bool:
    """True when an LLM rewrite is reasoning/meta-text rather than the rewrite itself."""
    return bool(_COT_META_RE.search(text or ""))


# ---- Prompts ----

_CANONICAL_REWRITE_PROMPT = """Rewrite the passage below so it can be understood in isolation.

Strict rules — violating any of these is a failure:
- Add NO new factual content not already in the passage.
- Do NOT import outside knowledge (no definitions you bring in yourself).
- PRESERVE MODALITY exactly. If the passage does not contain "shall", "must",
  "is required", do NOT add them. Never turn a label, heading, or blank form
  field (e.g. "Preheating temperature:", "Torch angle:") into a requirement
  ("... shall be specified"). A field label is NOT a normative statement.
- PRESERVE OPTIONALITY. Do not drop or change qualifiers like "if required",
  "where applicable", "if any". An optional item must stay optional.
- PRESERVE UNITS AND MARKERS. A superscript or trailing digit after a word
  (e.g. "thickness2", "Current1") is a FOOTNOTE MARKER, not a unit. Never invent
  a unit (do NOT write "mm²", "kA", etc.) from such a marker; drop the marker.
- PRESERVE NAMED ENTITIES. Never replace a named entity, standard designation
  (e.g. "ISO 15607", "ECMA-404"), proper noun, organisation, or specific technical
  term that is ALREADY in the passage with a generic reference like "the referenced
  standard", "the relevant standard", "the document", or "the organisation". Keep
  the exact name verbatim. You may resolve a vague pronoun/demonstrative TO such a
  name using context, but NEVER rewrite an explicit name into a generic phrase.
- If the passage is a blank form/template region (a run of "Label:" fields with
  no values, or a table of field names), it has no factual claim — return the
  single word NONE and nothing else.
- Resolve pronouns ("it", "this", "they", "both", "the two") to their antecedents
  using ONLY the supplied context. For "both"/"the two", name the two things
  explicitly (e.g. "practical issues with both" -> "practical issues with the
  forward and reverse traceability paths"). If the two referents are NOT in the
  context, leave the passage unchanged — do NOT guess.
- Resolve demonstratives used as bare objects ("in those", "of these", "to
  that", "from them") by replacing the demonstrative with the actual noun it
  refers to from the context (e.g. "if the material is not assigned in those"
  -> "if the material is not assigned in the material grouping lists"). If the
  antecedent noun is NOT present in the context, leave the passage unchanged —
  do NOT guess.
- Replace discourse markers ("Thus", "However", "In particular", "Therefore")
  with explicit referents from the context, OR drop the marker if it has no
  resolvable antecedent.
- If the passage names a term ("denoted X", "let X be ...") without defining
  it in the passage or the context, KEEP the name but DO NOT invent a
  definition.
- If the passage is a list item or sentence fragment (e.g. "Wire speed feed
  range for mechanized and automatic welding"), turn it into a COMPLETE sentence
  using only the lead-in already present in the context (e.g. "For mechanized
  and automatic welding, the wire speed feed range shall be specified."). Do not
  add facts beyond the lead-in.
- If the passage is a question (it ends with "?" or only restates an interrogative
  without answering it), rewrite it as the declarative claim it implies using ONLY
  the context. If the context does not supply the answer, return the single word
  NONE and nothing else.
- Output ONE plain paragraph. No bullets, no headers, no preamble.

Context (for reference only — do not paraphrase from here unless it's the
antecedent of a pronoun or discourse marker in the passage):
<<CONTEXT>>
{context}
<</CONTEXT>>

Passage to rewrite:
<<PASSAGE>>
{passage}
<</PASSAGE>>

Rewrite:"""


_SELF_CONTAINMENT_PROMPT = """Read the passage below IN ISOLATION (no other context).
Score how well it can be understood on its own.

- 1.0  Completely self-contained. All technical terms are either defined in
       the passage or are common terminology a domain reader would know
       without further explanation.
- 0.7  Mostly self-contained. One minor unresolved reference.
- 0.5  Partial. Some pronouns / discourse markers without antecedents, or
       one named-but-undefined technical term.
- 0.2  Weak. Multiple unresolved references; reader cannot reconstruct the
       full claim.
- 0.0  Cannot be understood without prior context.

Return ONLY the number, e.g. "0.85". No explanation.

Passage:
{passage}

Score:"""


_SEGMENT_CHUNK_PROMPT = """You will receive a passage of text from a technical document.
Identify each self-contained factual claim. A claim MAY span:
- 1 sentence (a complete claim)
- 2 sentences (a setup + claim, or claim + necessary qualifier)
- A whole paragraph (when the claim only completes across multiple sentences)

The boundary is decided by MEANING: a span must read as ONE coherent
statement when isolated. If a sentence depends on the previous one for its
antecedent ("Thus, ...", "It then ...", "denoted X"), MERGE them into one
span. Do NOT split mid-claim just because there is a period.

ENUMERATED LISTS: when a lead-in clause introduces a list ("... the following
shall be specified:", "X includes:", "For process Y:"), KEEP the lead-in
together with its list items as ONE span. NEVER emit a bare list item
("— Wire speed feed range ...", "— Torch angle") as a standalone claim — it is
not self-contained without its lead-in. Drop loose clause-number headings
("4.4.6 Back gouging") rather than attaching them to a claim.

For each claim you find, return:
- start_char, end_char: zero-based char offsets in the input passage
- role_hint: one of {{"definition", "rule", "constraint", "condition",
  "exception", "consequence", "example"}} — your best guess
- claim_summary: one short plain-English sentence summarising the claim

Output STRICT JSON: an array of objects, no preamble, no trailing commas.

CRITICAL OUTPUT RULE: Respond with ONLY the JSON array, starting with the
character '['. Do NOT think out loud, do NOT restate the passage, do NOT write
"We need to" or any reasoning. The very first character of your reply must be '['.

Passage (offsets are 0-based into this text):
<<PASSAGE>>
{passage}
<</PASSAGE>>

JSON:"""


# ---- Helpers ----


_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_float_score(raw: str) -> Optional[float]:
    """Pull a 0–1 float out of an LLM response and clamp to [0, 1].

    Reasoning models (gpt-oss-120b in particular) emit CoT before the final
    score, so the *last* in-range float in the output is usually the actual
    answer. Strategy:
      1. Find all floats.
      2. Prefer the last one that falls naturally in [0, 1] (the rubric range).
      3. Fall back to clamping any value to [0, 1] if no in-range float exists.
    """
    if not raw:
        return None
    matches = _FLOAT_RE.findall(raw)
    if not matches:
        return None
    parsed: List[float] = []
    for m in matches:
        try:
            parsed.append(float(m))
        except ValueError:
            continue
    if not parsed:
        return None
    in_range = [v for v in parsed if 0.0 <= v <= 1.0]
    if in_range:
        return in_range[-1]
    v = parsed[-1]
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _forward_nli_guard(
    raw: str, canonical: str, nli_model, threshold: float
) -> Tuple[bool, float, float]:
    """Forward NLI guard: raw must entail canonical.

    The thesis claim for SFU canonical_form is *non-additivity*: the rewrite
    introduces no content beyond the raw span. That property is captured by
    the forward direction NLI(raw → canonical) alone. The backward direction
    (canonical → raw) would additionally require the canonical form to
    contain ALL raw content — which rejects legitimate shortening rewrites
    that strip rhetorical framing ("We are now ready to consider how X" → "X")
    even though X is exactly the load-bearing claim.

    Returns (passes, ab_score, ba_score) for diagnostic logging — the
    backward score is computed and surfaced but does NOT gate the result.
    """
    from rag_gt.validation.nli_check import nli_batch

    if not raw.strip() or not canonical.strip():
        return (False, 0.0, 0.0)
    scores = nli_batch([(raw, canonical), (canonical, raw)])
    ab, ba = float(scores[0]), float(scores[1])
    return (ab >= threshold, ab, ba)


# ---- Public API ----


@dataclass
class CanonicalRewrite:
    """Result of the canonical-form rewrite + NLI guard step."""

    canonical_form: str
    nli_forward: float  # raw -> canonical
    nli_backward: float  # canonical -> raw
    passed: bool
    retries: int


@dataclass
class SFUSegment:
    """One span the chunk-segmenter returned. Pre-canonicalisation."""

    start_char: int
    end_char: int
    text: str
    role_hint: str
    claim_summary: str


def canonical_form_rewrite(
    passage: str,
    context: str,
    llm: "LLM",
    nli_model=None,
    threshold: float = CANONICAL_NLI_THRESHOLD,
    max_retries: int = MAX_REWRITE_RETRIES,
) -> CanonicalRewrite:
    """Rewrite `passage` into a self-contained canonical form, then NLI-guard it.

    Retries up to `max_retries` times if the NLI guard fails. If still failing
    after retries, returns the verbatim passage as the canonical form (with
    `passed=False` so callers can detect this).
    """
    from rag_gt.core.llm import APIError

    last_canonical = passage.strip()
    last_ab = 0.0
    last_ba = 0.0

    for attempt in range(max_retries):
        prompt = _CANONICAL_REWRITE_PROMPT.format(
            context=context.strip()[:2000] or "(no surrounding context)",
            passage=passage.strip(),
        )
        try:
            raw = llm.generate(prompt, temperature=0.0, max_tokens=512).strip()
        except APIError:
            raise
        except Exception as e:
            logger.warning(
                f"[SFU rewrite] transient LLM error attempt {attempt + 1}: "
                f"{type(e).__name__}: {e}"
            )
            continue
        candidate = raw.strip().strip('"').strip("'")
        if not candidate:
            continue

        # If the model says this passage has no usable claim, honour it BEFORE the
        # NLI guard. NLI(passage -> "NONE") scores ~0 and would fail all retries,
        # reverting to the verbatim garbage (post-loop fallback). This is the model
        # abstaining on a label/blank-form/rhetorical region, not a heuristic gate.
        if candidate.strip().upper() == "NONE":
            return CanonicalRewrite(
                canonical_form="NONE",
                nli_forward=0.0,
                nli_backward=0.0,
                passed=True,
                retries=attempt,
            )

        # Reject reasoning-model CoT/meta-text leaking in as the rewrite. The
        # candidate is discarded (not stored as last_canonical), so the post-loop
        # fallback returns either an earlier non-CoT rewrite (if one was produced)
        # or the verbatim passage — never the CoT string. passed stays False so
        # the caller can tell the rewrite did not cleanly succeed.
        if _looks_like_cot_meta(candidate):
            logger.debug(
                f"[SFU rewrite] discarded CoT/meta candidate attempt {attempt + 1}: "
                f"{candidate[:80]!r}"
            )
            continue

        if nli_model is None:
            # Skip NLI guard when running without the model (unit tests).
            return CanonicalRewrite(
                canonical_form=candidate,
                nli_forward=1.0,
                nli_backward=1.0,
                passed=True,
                retries=attempt,
            )

        passes, ab, ba = _forward_nli_guard(
            passage, candidate, nli_model, threshold
        )
        last_canonical = candidate
        last_ab = ab
        last_ba = ba
        if passes:
            return CanonicalRewrite(
                canonical_form=candidate,
                nli_forward=ab,
                nli_backward=ba,
                passed=True,
                retries=attempt,
            )
        logger.debug(
            f"[SFU rewrite] forward NLI guard failed attempt {attempt + 1}: "
            f"raw→canon={ab:.3f} (need ≥ {threshold:.2f}); canon→raw={ba:.3f} (diagnostic only); retrying"
        )

    # All retries exhausted / NLI never satisfied: keep the VERBATIM ORIGINAL
    # passage, not the last (entailment-failing) rewrite. A rewrite that cannot
    # be NLI-entailed by its source may have changed meaning, so trusting the
    # original wording is the safe choice (audit recommendation).
    return CanonicalRewrite(
        canonical_form=passage.strip(),
        nli_forward=last_ab,
        nli_backward=last_ba,
        passed=False,
        retries=max_retries,
    )


def self_containment_score(
    passage: str, llm: "LLM", default_on_error: float = 0.5
) -> float:
    """Score how self-contained `passage` is on its own. Returns 0..1."""
    from rag_gt.core.llm import APIError

    if not passage.strip():
        return 0.0
    prompt = _SELF_CONTAINMENT_PROMPT.format(passage=passage.strip())
    try:
        # Reasoning models (gpt-oss-120b) emit CoT before the score, so a
        # 16-token budget is far too tight — they hit the cap inside the
        # deliberation and return nothing parseable. 256 leaves room for
        # CoT plus a clearly-isolated final score.
        raw = llm.generate(prompt, temperature=0.0, max_tokens=256).strip()
    except APIError:
        raise
    except Exception as e:
        logger.warning(
            f"[SFU score] transient LLM error: {type(e).__name__}: {e}"
        )
        return default_on_error
    score = _parse_float_score(raw)
    if score is None:
        logger.warning(f"[SFU score] could not parse score from {raw!r}")
        return default_on_error
    return score


def upgrade_fact_to_sfu(
    fact: Fact,
    context_text: str,
    llm: "LLM",
    nli_model=None,
    *,
    canonical_threshold: float = CANONICAL_NLI_THRESHOLD,
    min_self_containment: float = MIN_SELF_CONTAINMENT_THRESHOLD,
) -> Fact:
    """Upgrade an existing fact in-place by adding SFU fields.

    - Stores the original `fact.text` in `fact.raw_text`.
    - Runs canonical-form rewrite under bidirectional NLI guard.
    - Scores self-containment.
    - Mutates the passed Fact and returns it.

    `context_text` is the chunk(s) of source text surrounding the fact. It is
    passed to the LLM during the canonical-form rewrite to help resolve
    anaphora. It is NOT used as source content (the prompt forbids that).
    """
    if not fact.raw_text:
        fact.raw_text = fact.text

    rewrite = canonical_form_rewrite(
        fact.raw_text, context_text, llm, nli_model, threshold=canonical_threshold
    )
    fact.canonical_form = rewrite.canonical_form
    if not rewrite.passed:
        logger.info(
            f"[SFU upgrade] fact={fact.fact_id} canonical-form NLI guard failed "
            f"after retries; canonical_form left as verbatim. "
            f"raw→canon={rewrite.nli_forward:.3f}, canon→raw={rewrite.nli_backward:.3f}"
        )

    fact.self_containment_score = self_containment_score(
        fact.canonical_form or fact.raw_text, llm
    )
    if fact.self_containment_score < min_self_containment:
        logger.info(
            f"[SFU upgrade] fact={fact.fact_id} weak self-containment "
            f"({fact.self_containment_score:.2f}) — flagged but kept"
        )
    return fact


# ---- Full chunk segmentation (for Phase 5 GT regeneration) ----


def segment_chunk_into_sfus(
    chunk_text: str, llm: "LLM"
) -> List[SFUSegment]:
    """LLM-segment a chunk into a list of variable-length SFU spans.

    Returns spans with `text` taken verbatim from `chunk_text` using the
    LLM-supplied character offsets. Robust to JSON-parse failures (returns
    an empty list on irrecoverable output).
    """
    from rag_gt.core.llm import APIError, _extract_json
    from json_repair import repair_json

    if not chunk_text or not chunk_text.strip():
        return []
    prompt = _SEGMENT_CHUNK_PROMPT.format(passage=chunk_text)
    # gpt-oss-120b is a reasoning model: it burns tokens on chain-of-thought
    # BEFORE emitting the JSON array, so a 1024 budget is frequently exhausted
    # mid-deliberation and returns an empty/short content channel (root cause of
    # the ~60% "Expecting value: char 0" parse failures). 2048 leaves room for
    # CoT plus the array; _extract_json strips any prose wrapper before parsing,
    # matching the edge-classifier's robust path. One retry covers a flaky call.
    data = None
    for attempt in range(2):
        try:
            raw = llm.generate(prompt, temperature=0.0, max_tokens=4096).strip()
        except APIError:
            raise
        except Exception as e:
            logger.warning(
                f"[SFU segment] transient LLM error attempt {attempt + 1}: "
                f"{type(e).__name__}: {e}"
            )
            continue
        try:
            data = json.loads(repair_json(_extract_json(raw)))
            break
        except Exception as e:
            logger.debug(
                f"[SFU segment] parse failed attempt {attempt + 1} "
                f"(raw[:60]={raw[:60]!r}): {e}"
            )
            data = None
    if data is None:
        logger.warning("[SFU segment] could not parse JSON from segmenter after retries")
        return []
    if not isinstance(data, list):
        logger.warning(f"[SFU segment] segmenter returned non-list: {type(data)}")
        return []

    out: List[SFUSegment] = []
    chunk_len = len(chunk_text)
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start_char", -1))
            end = int(item.get("end_char", -1))
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start or end > chunk_len:
            continue
        text = chunk_text[start:end].strip()
        if len(text) < 10:
            continue
        out.append(
            SFUSegment(
                start_char=start,
                end_char=end,
                text=text,
                role_hint=str(item.get("role_hint", "") or ""),
                claim_summary=str(item.get("claim_summary", "") or ""),
            )
        )
    return out
