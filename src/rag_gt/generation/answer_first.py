"""Stage C: answer-first generation for verified cross-page bridge pairs.

The LLM only drafts two source-bounded clauses and a bridge-hidden question.
All acceptance decisions are made afterward with the local NLI model and the
existing leave-one-out necessity implementation.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from loguru import logger

from rag_gt.allpdf.necessity import leave_one_out_necessity
from rag_gt.core.config import load_config
from rag_gt.core.llm import get_llm
from rag_gt.core.types import Fact
from rag_gt.validation.nli_check import nli_batch


_DEFAULT_DOC_NAME = "the source document"


def _settings() -> dict:
    cfg = load_config()
    settings = dict(cfg["answer_first"])
    settings["workers"] = int(cfg["performance"]["max_concurrent_llm_calls"])
    settings.setdefault("doc_display_names", {})
    return settings


def _doc_name(doc_id: object, settings: Mapping[str, object]) -> str:
    """Resolve a doc id to its human display name (Track B). Falls back to a
    generic phrase so a new/unknown corpus never leaks a bare id into a prompt."""
    names = settings.get("doc_display_names") or {}
    return str(names.get(str(doc_id or ""), _DEFAULT_DOC_NAME))


def _fact_text(record: Mapping[str, object]) -> str:
    return str(record.get("canonical_form") or record.get("text") or "").strip()


def _fact_id(record: Mapping[str, object]) -> str:
    return str(record.get("fact_id") or record.get("id") or "").strip()


def _as_fact(record: Mapping[str, object]) -> Fact:
    return Fact(
        fact_id=_fact_id(record),
        text=_fact_text(record),
        canonical_form=_fact_text(record),
        role="definition",
        self_containment_score=float(
            record.get("self_containment_score", record.get("self_containment", 1.0))
            or 1.0
        ),
    )


def _load_records(path: Path, collection_key: str | None = None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if collection_key and isinstance(payload.get(collection_key), list):
        return payload[collection_key]
    raise ValueError(f"{path} must contain a JSON list or a '{collection_key}' list")


def _normal_form(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def bridge_is_hidden(question: str, *bridge_forms: str) -> bool:
    """Return True when no normalized bridge surface occurs in the question."""
    question_norm = f" {_normal_form(question)} "
    question_compact = question_norm.replace(" ", "")
    question_tokens = question_norm.split()
    for form in bridge_forms:
        form_norm = _normal_form(form)
        if form_norm and f" {form_norm} " in question_norm:
            return False
        # Catch punctuation/spacing variants such as ISO3834 vs ISO 3834.
        form_compact = form_norm.replace(" ", "")
        if len(form_compact) >= 6 and form_compact in question_compact:
            return False
        # BUG-G: a short all-alpha bridge (e.g. "WPS", "pWPS") escapes the padded
        # and compact checks when it appears as a plural/possessive ("WPSs",
        # "WPS's" -> normalized "wpss"/"wps s"). Catch a question token that
        # starts with the whole short bridge form.
        if (
            form_norm
            and len(form_norm) <= 5
            and form_norm.isalpha()
            and any(tok.startswith(form_norm) for tok in question_tokens)
        ):
            return False
    return True


def _clean_clause(value: object, *, min_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip(" \t\r\n;,.:")
    if len(text) < min_chars:
        return ""
    return text + ("" if text[-1] in ".!?" else ".")


def _clean_question(value: object, *, min_chars: int) -> str:
    question = " ".join(str(value or "").split()).strip().strip('"\'')
    if not question:
        return ""
    if not question.endswith("?"):
        question = question.rstrip(".!;") + "?"
    return question if len(question) >= min_chars and question.count("?") == 1 else ""


def _lower_first(text: str) -> str:
    """Lowercase the first character only when it is an ordinary capitalized word.

    Leaves acronyms / standard references intact: "ISO 3834 requires…" must not
    become "iSO 3834 requires…". Heuristic: only downcase when the SECOND
    character is already lowercase (i.e. a normal Titlecase word like "The").
    """
    if len(text) >= 2 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def _compose_answer(clause_a: str, clause_b: str) -> str:
    return f"{clause_a.rstrip('.!?')}; {_lower_first(clause_b)}".strip()


def _bridge_prompt(
    pair: Mapping[str, object], fact_a: str, fact_b: str, doc_name: str = _DEFAULT_DOC_NAME
) -> str:
    bridge = str(pair.get("bridge_entity") or pair.get("bridge_norm") or "")
    return f"""MODE: BRIDGE
DOCUMENT: {doc_name}
Create one answer-first, two-support QA draft from the source facts below.

SOURCE A:
<<FACT_A>>{fact_a}<</FACT_A>>

SOURCE B:
<<FACT_B>>{fact_b}<</FACT_B>>

The verified bridge surface is: <<BRIDGE>>{bridge}<</BRIDGE>>

Rules:
1. clause_a states only a claim directly entailed by SOURCE A.
2. clause_b states only a different claim directly entailed by SOURCE B.
3. The two clauses together form a useful answer; do not add outside knowledge.
4. Write one standalone question whose complete answer requires BOTH clauses.
5. The question MUST NOT contain the bridge surface, its abbreviation, or a trivial
   spelling variant. Use other concrete anchors from the sources so it is retrievable.
6. Do not mention sources, facts, pages, documents, or these instructions.
7. The question MUST NOT contain "according to the fact/source/passage/document",
   "as mentioned in the source", "this document", "the provided/given fact", or any
   reference to the dataset. Ask a natural question about the subject matter. If the
   question must name where the information comes from, call it "{doc_name}" — never
   "this document", "the fact", or "the source".
8. Ask "why"/"how" only if SOURCE A or SOURCE B contains an explicit reason, cause,
   or mechanism; for a rule or requirement with no stated reason, ask "what"/"which".
9. Build the question only on a real relation between SOURCE A and SOURCE B
   (cause->effect, condition->consequence, definition->application, sequence). If
   they merely share a topic, do not invent a contrast.
10. Each clause must faithfully preserve its source's scope and claim. Do not
    generalise (e.g. "Required" when the source says it varies by level), soften
    ("becomes laborious" when the source says "cannot be applied"), or attribute a
    property to the wrong subject.

Return only JSON:
{{"clause_a":"...","clause_b":"...","question":"...?"}}"""


def _single_prompt(fact_text: str, doc_name: str = _DEFAULT_DOC_NAME) -> str:
    return f"""MODE: SINGLE
DOCUMENT: {doc_name}
Write one standalone retrieval question answered completely by this source fact.

Rules:
1. The question MUST NOT contain "according to the fact/source/passage/document",
   "as mentioned in the source", "this document", "the provided/given fact", "the
   given details", or any reference to the dataset. Ask a natural question about the
   subject matter itself. If the question must name where the information comes from,
   call it "{doc_name}" — never "this document", "the fact", or "the source".
2. Ask only what this fact fully answers. If it names a field or heading, ask about
   that field; do not ask for a value the fact does not contain.
3. Ask "why"/"how" only if the fact states a reason or mechanism; for a rule,
   requirement, or definition, ask "what"/"which".
4. Resolve every pronoun/quantifier ("this", "both") to the concrete noun in the fact.
5. Do not reveal the answer in the question, and add no outside knowledge.

<<FACT>>{fact_text}<</FACT>>

Return only JSON: {{"question":"...?"}}"""


def _draft_bridge(
    pair: Mapping[str, object],
    facts_by_id: Mapping[str, Mapping[str, object]],
    llm: object,
    settings: Mapping[str, object],
) -> dict:
    pair_id = _bridge_cache_key(pair)  # BUG-B: content-derived, not positional
    fact_a_record = facts_by_id.get(str(pair.get("fact_a") or ""))
    fact_b_record = facts_by_id.get(str(pair.get("fact_b") or ""))
    if not fact_a_record or not fact_b_record:
        return {"kind": "bridge", "key": pair_id, "error": "missing_fact"}
    bridge = str(pair.get("bridge_entity") or "")
    bridge_norm = str(pair.get("bridge_norm") or "")
    doc_name = _doc_name(
        pair.get("doc") or fact_a_record.get("doc") or fact_b_record.get("doc"),
        settings,
    )
    prompt = _bridge_prompt(
        pair, _fact_text(fact_a_record), _fact_text(fact_b_record), doc_name
    )
    attempts = int(settings["generation_attempts"])
    for attempt in range(attempts):
        try:
            temperature = float(
                settings["temperature_initial"]
                if attempt == 0
                else settings["temperature_retry"]
            )
            raw = llm.generate_json(
                prompt,
                temperature=temperature,
                max_tokens=int(settings["bridge_max_tokens"]),
            )
        except Exception as exc:  # preserve a per-item failure without losing the run
            if attempt == attempts - 1:
                return {
                    "kind": "bridge",
                    "key": pair_id,
                    "error": f"generation_error:{type(exc).__name__}:{exc}",
                }
            continue
        clause_a = _clean_clause(
            raw.get("clause_a"), min_chars=int(settings["min_clause_chars"])
        )
        clause_b = _clean_clause(
            raw.get("clause_b"), min_chars=int(settings["min_clause_chars"])
        )
        question = _clean_question(
            raw.get("question"), min_chars=int(settings["min_question_chars"])
        )
        if clause_a and clause_b and question and bridge_is_hidden(question, bridge, bridge_norm):
            return {
                "kind": "bridge",
                "key": pair_id,
                "clause_a": clause_a,
                "clause_b": clause_b,
                "question": question,
            }
        prompt += "\nYour previous draft was invalid or leaked the bridge. Return a corrected JSON object."
    return {"kind": "bridge", "key": pair_id, "error": "invalid_or_bridge_leak"}


def _draft_single(
    record: Mapping[str, object],
    llm: object,
    settings: Mapping[str, object],
) -> dict:
    fact_id = _fact_id(record)
    doc_name = _doc_name(record.get("doc"), settings)
    try:
        raw = llm.generate_json(
            _single_prompt(_fact_text(record), doc_name),
            temperature=float(settings["temperature_initial"]),
            max_tokens=int(settings["single_max_tokens"]),
        )
        question = _clean_question(
            raw.get("question"), min_chars=int(settings["min_question_chars"])
        )
    except Exception as exc:
        return {
            "kind": "single",
            "key": fact_id,
            "error": f"generation_error:{type(exc).__name__}:{exc}",
        }
    if not question:
        return {"kind": "single", "key": fact_id, "error": "invalid_question"}
    return {"kind": "single", "key": fact_id, "question": question}


# Draft-cache schema version. Bumped when the cache KEY scheme changes so a stale
# cache written under the old scheme is ignored rather than misread (BUG-B: the
# v1 cache keyed bridges by positional pair_id; v2 keys by fact-pair content).
_DRAFT_CACHE_VERSION = 2


def _load_draft_cache(path: Path | None) -> dict[str, dict]:
    cached: dict[str, dict] = {}
    if not path or not path.exists():
        return cached
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        # Ignore entries from an older key scheme so positional keys are never
        # matched against the new content keys.
        if int(item.get("cache_version", 1)) != _DRAFT_CACHE_VERSION:
            continue
        cached[f"{item['kind']}:{item['key']}"] = item
    return cached


def _append_draft(path: Path | None, item: Mapping[str, object]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(item)
    record.setdefault("cache_version", _DRAFT_CACHE_VERSION)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_drafting(
    jobs: Sequence[tuple[str, str, Callable[[], dict]]],
    *,
    workers: int,
    cache_path: Path | None,
    progress_every: int,
) -> dict[str, dict]:
    cached = _load_draft_cache(cache_path)
    # Failed calls are recorded for auditability but retried on the next run;
    # only a valid draft is a completed cache entry.
    missing = [
        job
        for job in jobs
        if f"{job[0]}:{job[1]}" not in cached
        or cached[f"{job[0]}:{job[1]}"].get("error")
    ]
    if missing:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(job[2]): (job[0], job[1]) for job in missing}
            completed = 0
            for future in as_completed(futures):
                kind, key = futures[future]
                try:
                    item = future.result()
                except Exception as exc:  # defensive: one bad item must not erase prior API work
                    item = {"kind": kind, "key": key, "error": f"worker_error:{type(exc).__name__}:{exc}"}
                cached[f"{kind}:{key}"] = item
                _append_draft(cache_path, item)
                completed += 1
                if completed % progress_every == 0 or completed == len(missing):
                    logger.info(f"Stage C drafting: {completed}/{len(missing)} new items")
    return cached


def _grounding(record: Mapping[str, object]) -> tuple[str, int, list, bool]:
    """Return (chunk_id, page, bboxes, grounded).

    ``grounded`` is False when the source record carried no real ``chunk_id``
    and we fell back to a ``fact:<id>`` pseudo-ID. That pseudo-ID is NOT a real
    corpus chunk and must never be silently treated as retrievable gold (this
    is the exact defect that produced the V19 fine-tuning fiasco). Callers set
    ``grounding_complete`` on the QA record from this flag and may enforce a
    strict mode that refuses to emit ungrounded pairs.
    """
    fact_id = _fact_id(record)
    real_chunk_id = str(record.get("chunk_id") or "").strip()
    chunk_id = real_chunk_id or f"fact:{fact_id}"
    page = int(record.get("page", record.get("page_start", 0)) or 0)
    bboxes = list(record.get("bboxes") or [])
    grounded = bool(real_chunk_id)
    return chunk_id, page, bboxes, grounded


def _bridge_cache_key(pair: Mapping[str, object]) -> str:
    """Content-derived draft-cache key for a bridge pair (BUG-B).

    The old key was the POSITIONAL ``pair_id`` (BP0001…) assigned by
    ``bridge_linker`` in sorted order. Regenerating the pairs file after any
    upstream change reshuffles that numbering, so cached drafts were silently
    re-attributed to different fact pairs. Key on the pair's actual content so a
    cached draft always maps back to the same two facts + bridge.
    """
    fact_a = str(pair.get("fact_a") or "")
    fact_b = str(pair.get("fact_b") or "")
    bridge = str(pair.get("bridge_norm") or pair.get("bridge_entity") or "")
    return f"{fact_a}|{fact_b}|{bridge}"


def _score_bridge_drafts(
    pairs: Sequence[Mapping[str, object]],
    facts_by_id: Mapping[str, Mapping[str, object]],
    drafts: Mapping[str, Mapping[str, object]],
    *,
    clause_threshold: float,
    single_max: float,
    joint_min: float,
    require_chunk_ids: bool = False,
    nli_fn: Callable[[list[tuple[str, str]]], list[float]] = nli_batch,
    necessity_fn: Callable[..., dict | None] = leave_one_out_necessity,
) -> tuple[list[dict], Counter]:
    candidates: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    rejected: Counter = Counter()
    nli_inputs: list[tuple[str, str]] = []
    for pair in pairs:
        draft = drafts.get(f"bridge:{_bridge_cache_key(pair)}", {})
        if draft.get("error"):
            rejected[str(draft["error"]).split(":", 1)[0]] += 1
            continue
        a = facts_by_id.get(str(pair.get("fact_a") or ""))
        b = facts_by_id.get(str(pair.get("fact_b") or ""))
        if not a or not b:
            rejected["missing_fact"] += 1
            continue
        clause_a = str(draft["clause_a"])
        clause_b = str(draft["clause_b"])
        answer = _compose_answer(clause_a, clause_b)
        candidates.append((pair, draft))
        nli_inputs.extend(
            [
                (_fact_text(a), clause_a),
                (_fact_text(b), clause_b),
                (_fact_text(a), answer),
                (_fact_text(b), answer),
                (f"{_fact_text(a)} {_fact_text(b)}", answer),
            ]
        )

    scores = nli_fn(nli_inputs)
    if len(scores) != len(nli_inputs):
        raise RuntimeError(f"NLI returned {len(scores)} scores for {len(nli_inputs)} inputs")

    output: list[dict] = []
    seen_questions: set[str] = set()
    for index, (pair, draft) in enumerate(candidates):
        clause_a_score, clause_b_score, nli_a, nli_b, nli_joint = [
            float(value) for value in scores[index * 5 : index * 5 + 5]
        ]
        if clause_a_score < clause_threshold or clause_b_score < clause_threshold:
            rejected["clause_not_entailed"] += 1
            continue
        if nli_a >= single_max or nli_b >= single_max or nli_joint < joint_min:
            rejected["not_jointly_necessary"] += 1
            continue

        a_record = facts_by_id[str(pair["fact_a"])]
        b_record = facts_by_id[str(pair["fact_b"])]
        fact_a = _as_fact(a_record)
        fact_b = _as_fact(b_record)
        answer = _compose_answer(str(draft["clause_a"]), str(draft["clause_b"]))
        necessity = necessity_fn(answer, [fact_a, fact_b], threshold=single_max)
        if not necessity or float(necessity.get("necessity_score", 0.0)) < 1.0:
            rejected["loo_failed"] += 1
            continue
        question_norm = _normal_form(str(draft["question"]))
        if question_norm in seen_questions:
            rejected["duplicate_question"] += 1
            continue
        seen_questions.add(question_norm)

        chunk_a, page_a, bbox_a, grounded_a = _grounding(a_record)
        chunk_b, page_b, bbox_b, grounded_b = _grounding(b_record)
        grounding_complete = grounded_a and grounded_b
        if require_chunk_ids and not grounding_complete:
            rejected["ungrounded_chunk_ids"] += 1
            continue
        fact_a_id = _fact_id(a_record)
        fact_b_id = _fact_id(b_record)
        output.append(
            {
                "qa_id": "",  # assigned after bridge + single tracks are merged
                "hop_type": "bridge",
                "bridge_entity": str(pair.get("bridge_entity") or ""),
                "question": str(draft["question"]),
                "answer": answer,
                "answer_clauses": [
                    {"text": str(draft["clause_a"]), "fact_id": fact_a_id, "nli": round(clause_a_score, 3)},
                    {"text": str(draft["clause_b"]), "fact_id": fact_b_id, "nli": round(clause_b_score, 3)},
                ],
                "gold_fact_ids": [fact_a_id, fact_b_id],
                "gold_chunk_ids": [chunk_a, chunk_b],
                "gold_pages": [page_a, page_b],
                "gold_bboxes": {fact_a_id: bbox_a, fact_b_id: bbox_b},
                "grounding_complete": grounding_complete,
                "necessity": {
                    "nli_a": round(nli_a, 3),
                    "nli_b": round(nli_b, 3),
                    "nli_joint": round(nli_joint, 3),
                    "necessity_score": necessity["necessity_score"],
                    "necessary_fact_ids": necessity["necessary_fact_ids"],
                    "loo_entailment": necessity["loo_entailment"],
                    "passed": True,
                },
                "verify": {
                    "bridge_hidden": True,
                    "duplicate": False,
                    "faithful": True,
                    "verdict": "PENDING_STAGE_D",
                },
                "source_pair_id": str(pair.get("pair_id") or ""),
                "doc": str(pair.get("doc") or a_record.get("doc") or ""),
            }
        )
    return output, rejected


def _make_single_pairs(
    facts: Sequence[Mapping[str, object]],
    drafts: Mapping[str, Mapping[str, object]],
    *,
    require_chunk_ids: bool = False,
) -> tuple[list[dict], Counter]:
    output: list[dict] = []
    rejected: Counter = Counter()
    seen_questions: set[str] = set()
    for record in facts:
        fact_id = _fact_id(record)
        draft = drafts.get(f"single:{fact_id}", {})
        if draft.get("error"):
            rejected[str(draft["error"]).split(":", 1)[0]] += 1
            continue
        question = str(draft.get("question") or "")
        norm = _normal_form(question)
        if not norm or norm in seen_questions:
            rejected["duplicate_question"] += 1
            continue
        seen_questions.add(norm)
        chunk_id, page, bboxes, grounded = _grounding(record)
        if require_chunk_ids and not grounded:
            rejected["ungrounded_chunk_ids"] += 1
            continue
        answer = _fact_text(record)
        output.append(
            {
                "qa_id": "",
                "hop_type": "single",
                "bridge_entity": "",
                "question": question,
                "answer": answer,
                "answer_clauses": [{"text": answer, "fact_id": fact_id}],
                "gold_fact_ids": [fact_id],
                "gold_chunk_ids": [chunk_id],
                "gold_pages": [page],
                "gold_bboxes": {fact_id: bboxes},
                "grounding_complete": grounded,
                "necessity": {"passed": None, "reason": "single_control"},
                "verify": {
                    "bridge_hidden": None,
                    "duplicate": False,
                    "faithful": True,
                    "verdict": "PENDING_STAGE_D",
                },
                "doc": str(record.get("doc") or ""),
            }
        )
    return output, rejected


def build_answer_first_pairs(
    facts: Sequence[Mapping[str, object]],
    bridge_pairs: Sequence[Mapping[str, object]],
    llm: object,
    *,
    include_singles: bool = True,
    workers: int | None = None,
    draft_cache_path: Path | None = None,
    clause_threshold: float | None = None,
    single_max: float | None = None,
    joint_min: float | None = None,
    require_chunk_ids: bool = False,
    nli_fn: Callable[[list[tuple[str, str]]], list[float]] = nli_batch,
    necessity_fn: Callable[..., dict | None] = leave_one_out_necessity,
) -> dict:
    """Generate and validate Stage C QA records from in-memory inputs.

    ``require_chunk_ids``: strict grounding. When True, any pair whose facts lack
    a real ``chunk_id`` (would fall back to a ``fact:`` pseudo-ID) is rejected
    with reason ``ungrounded_chunk_ids`` instead of shipped as fake gold (BUG-C).
    """
    settings = _settings()
    workers = int(settings["workers"] if workers is None else workers)
    clause_threshold = float(
        settings["clause_entailment_min"]
        if clause_threshold is None
        else clause_threshold
    )
    single_max = float(
        settings["single_fact_answer_max"] if single_max is None else single_max
    )
    joint_min = float(settings["joint_answer_min"] if joint_min is None else joint_min)
    facts_by_id = {_fact_id(record): record for record in facts if _fact_id(record)}
    jobs: list[tuple[str, str, Callable[[], dict]]] = []
    for pair in bridge_pairs:
        pair_copy = dict(pair)
        key = _bridge_cache_key(pair_copy)  # BUG-B: content-derived key
        jobs.append(
            (
                "bridge",
                key,
                lambda pair=pair_copy: _draft_bridge(
                    pair, facts_by_id, llm, settings
                ),
            )
        )
    if include_singles:
        for record in facts:
            record_copy = dict(record)
            key = _fact_id(record_copy)
            jobs.append(
                (
                    "single",
                    key,
                    lambda fact=record_copy: _draft_single(fact, llm, settings),
                )
            )

    drafts = _run_drafting(
        jobs,
        workers=workers,
        cache_path=draft_cache_path,
        progress_every=int(settings["progress_every"]),
    )
    bridge_output, bridge_rejected = _score_bridge_drafts(
        bridge_pairs,
        facts_by_id,
        drafts,
        clause_threshold=clause_threshold,
        single_max=single_max,
        joint_min=joint_min,
        require_chunk_ids=require_chunk_ids,
        nli_fn=nli_fn,
        necessity_fn=necessity_fn,
    )
    single_output: list[dict] = []
    single_rejected: Counter = Counter()
    if include_singles:
        single_output, single_rejected = _make_single_pairs(
            facts, drafts, require_chunk_ids=require_chunk_ids
        )

    merged = bridge_output + single_output
    for index, item in enumerate(merged, start=1):
        item["qa_id"] = f"Q{index:05d}"
    n_ungrounded = sum(1 for qa in merged if not qa.get("grounding_complete", False))
    return {
        "pairs": merged,
        "stats": {
            "n_input_facts": len(facts),
            "n_input_bridge_pairs": len(bridge_pairs),
            "n_bridge_pass": len(bridge_output),
            "n_bridge_rejected": len(bridge_pairs) - len(bridge_output),
            "bridge_rejection_reasons": dict(sorted(bridge_rejected.items())),
            "n_single_pass": len(single_output),
            "n_single_rejected": len(facts) - len(single_output) if include_singles else 0,
            "single_rejection_reasons": dict(sorted(single_rejected.items())),
            "thresholds": {
                "clause_entailment_min": clause_threshold,
                "single_fact_answer_max": single_max,
                "joint_answer_min": joint_min,
            },
            "require_chunk_ids": require_chunk_ids,
            "n_pairs_ungrounded": n_ungrounded,
            "grounding_note": (
                "grounding_complete=false marks a pair whose gold_chunk_ids fell "
                "back to a fact:<id> pseudo-ID (source record had no real chunk_id). "
                "Such IDs are NOT retrievable corpus chunks; run with "
                "require_chunk_ids=True to reject them instead of emitting fake gold."
            ),
        },
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage C answer-first bridge QA generation")
    parser.add_argument("facts", type=Path)
    parser.add_argument("bridge_pairs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--bridge-limit", type=int)
    parser.add_argument("--single-limit", type=int)
    parser.add_argument("--no-singles", action="store_true")
    parser.add_argument("--draft-cache", type=Path)
    parser.add_argument("--clause-threshold", type=float)
    parser.add_argument("--single-max", type=float)
    parser.add_argument("--joint-min", type=float)
    parser.add_argument(
        "--require-chunk-ids",
        action="store_true",
        help="Reject pairs whose facts lack a real chunk_id instead of emitting "
        "fact:<id> pseudo-IDs as gold (BUG-C strict grounding).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    facts = _load_records(args.facts)
    pairs = _load_records(args.bridge_pairs, "pairs")
    if args.bridge_limit is not None:
        pairs = pairs[: max(0, args.bridge_limit)]
    if args.single_limit is not None:
        facts_for_singles = facts[: max(0, args.single_limit)]
        required = {str(p.get("fact_a")) for p in pairs} | {str(p.get("fact_b")) for p in pairs}
        selected = {_fact_id(f): f for f in facts_for_singles}
        for fact in facts:
            if _fact_id(fact) in required:
                selected[_fact_id(fact)] = fact
        facts = list(selected.values())

    cache_path = args.draft_cache or args.output.with_suffix(".drafts.jsonl")
    result = build_answer_first_pairs(
        facts,
        pairs,
        get_llm("gt"),
        include_singles=not args.no_singles,
        workers=args.workers,
        draft_cache_path=cache_path,
        clause_threshold=args.clause_threshold,
        single_max=args.single_max,
        joint_min=args.joint_min,
        require_chunk_ids=args.require_chunk_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Stage C wrote {args.output}: {result['stats']}")
    return 0 if result["stats"]["n_bridge_pass"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
