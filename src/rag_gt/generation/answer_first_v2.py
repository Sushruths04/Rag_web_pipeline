"""Stage C v2 — evidence-dense answer-first QA generation.

Two products, both multi-evidence so retrieval precision is measurable:

  single v2  ``neighbor_pair_v2``  2 same-page neighbour facts -> a question
             requiring BOTH clauses (2 evidence units).
  bridge v2  ``cluster_2plus2``    verified cross-page bridge pair + one
             same-page neighbour per side -> a question requiring ALL FOUR
             clauses, bridge surface hidden (4 evidence units). At 4 facts the
             leave-one-out necessity check is genuinely stronger than pairwise
             gates (audit BUG-H / N1).

Clusters that cannot find neighbours fall back to plain 2-fact bridges via the
v1 pipeline, tagged ``bridge_pair_v1``. All records carry the v1 QA schema plus
``evidence_strategy``. Strict grounding by default: pairs whose facts lack a
real chunk_id are rejected, never emitted with ``fact:`` pseudo-IDs.

Audit trail: the draft cache JSONL persists EVERY draft — including ones later
rejected by the NLI/LOO/grounding gates — and the stats block records
per-reason rejection counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from loguru import logger

from rag_gt.allpdf.bridge_quality import surface_ok
from rag_gt.allpdf.necessity import leave_one_out_necessity, leave_one_out_necessity_batch
from rag_gt.core.llm import get_llm
from rag_gt.generation.answer_first import (
    _as_fact,
    _clean_clause,
    _clean_question,
    _doc_name,
    _fact_id,
    _fact_text,
    _grounding,
    _load_records,
    _lower_first,
    _normal_form,
    _run_drafting,
    _settings,
    bridge_is_hidden,
    build_answer_first_pairs,
)
from rag_gt.generation.cluster_bridge import build_clusters
from rag_gt.generation.neighbor_pairs import sample_neighbor_pairs
from rag_gt.validation.nli_check import nli_batch, truncation_stats

_RULES_COMMON = """\
- Do not mention sources, facts, pages, documents, or these instructions.
- The question MUST NOT contain "according to the fact/source/passage/document",
  "as mentioned in the source", "this document", "the provided/given fact", or any
  reference to the dataset. If the question must name where the information comes
  from, call it "{doc_name}" — never "this document", "the fact", or "the source".
- Ask "why"/"how" only if a source states an explicit reason, cause, or mechanism;
  otherwise ask "what"/"which".
- Each clause must faithfully preserve its source's scope and claim: no
  generalising, softening, or attributing a property to the wrong subject."""

# T5 (fixes F4): 90% of the round-2 shipped questions opened with "What" and
# 69% were compound "…, and …" questions. Each draft job gets a FORM hint
# sampled deterministically from its own cache key (stable across re-runs
# reading the same cache; a prompt-text change is expected to invalidate the
# cache, which is the existing cache-key contract). The hint is advisory: the
# why/how faithfulness rule below still wins if a form does not fit the
# source content.
_QUESTION_FORMS: tuple[str, ...] = (
    "What", "Which", "How", "Under what condition", "When", "For what purpose",
)


def _form_hint(cache_key: str) -> str:
    digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
    return _QUESTION_FORMS[int(digest, 16) % len(_QUESTION_FORMS)]


def _form_hint_rule(form_hint: str) -> str:
    return (
        f"- Preferred opening form: start the question with \"{form_hint}\" if it "
        "fits the content naturally; otherwise pick the closest natural fit from "
        "What/Which/How/Under what condition/When/For what purpose. This preference "
        "never overrides the why/how faithfulness rule below."
    )


def _pair_prompt(fact_a: str, fact_b: str, doc_name: str, *, form_hint: str = "What") -> str:
    return f"""MODE: NEIGHBOR-PAIR
DOCUMENT: {doc_name}
Create one answer-first, two-support QA draft from two adjacent source facts.

SOURCE A:
<<FACT_A>>{fact_a}<</FACT_A>>

SOURCE B:
<<FACT_B>>{fact_b}<</FACT_B>>

Rules:
1. clause_a states only a claim directly entailed by SOURCE A.
2. clause_b states only a different claim directly entailed by SOURCE B.
3. Write one standalone question whose complete answer requires BOTH clauses;
   an answer built from only one source must be incomplete.
{_form_hint_rule(form_hint)}
{_RULES_COMMON.format(doc_name=doc_name)}

Return only JSON:
{{"clause_a":"...","clause_b":"...","question":"...?"}}"""


def _side_prompt(fact_1: str, fact_2: str, doc_name: str) -> str:
    return f"""MODE: CLUSTER-SIDE
DOCUMENT: {doc_name}
Rewrite each source as one standalone clause. These clauses become part of a
verified answer, so each must be checkable against its source alone.

SOURCE 1:
<<FACT_1>>{fact_1}<</FACT_1>>

SOURCE 2:
<<FACT_2>>{fact_2}<</FACT_2>>

Rules:
1. clause_1 states only a claim directly entailed by SOURCE 1.
2. clause_2 states only a different claim directly entailed by SOURCE 2.
- Stay close to each source's own wording; prefer its exact terms over synonyms.
- Each clause must faithfully preserve its source's scope and claim: no
  generalising, softening, or attributing a property to the wrong subject.
- Do not mention sources, facts, pages, documents, or these instructions.

Return only JSON:
{{"clause_1":"...","clause_2":"..."}}"""


def _question_prompt(clauses: Sequence[str], bridge: str, doc_name: str, *,
                     bridge_norm: str = "", form_hint: str = "What") -> str:
    a, a2, b, b2 = clauses
    # T6 (halves F5): name the exact forbidden words up front rather than
    # relying solely on the retry loop to teach the model what leaked --
    # 26% of round-2 draft jobs died invalid_or_bridge_leak.
    forbidden_words = list(dict.fromkeys(w for w in (bridge, bridge_norm) if w))
    forbidden = ", ".join(f'"{w}"' for w in forbidden_words)
    return f"""MODE: CLUSTER-QUESTION
DOCUMENT: {doc_name}
The complete answer to your question is exactly these four verified clauses.
Clauses A/B come from one page; clauses C/D come from another page of the same
document.

CLAUSE A: {a}
CLAUSE B: {a2}
CLAUSE C: {b}
CLAUSE D: {b2}

The verified concept linking the two pages is: <<BRIDGE>>{bridge}<</BRIDGE>>

Rules:
1. Write ONE standalone question whose complete answer requires ALL FOUR
   clauses; dropping any single clause must leave the answer incomplete.
2. The question MUST NOT contain the bridge surface, its abbreviation, or a
   trivial spelling variant. Use other concrete anchors so it stays retrievable.
   FORBIDDEN words in the question: {forbidden}. Do not use these words, or
   any abbreviation or trivial spelling variant of them, anywhere in the
   question.
{_form_hint_rule(form_hint)}
{_RULES_COMMON.format(doc_name=doc_name)}

Return only JSON:
{{"question":"...?"}}"""


def _compose_multi(clauses: Sequence[str]) -> str:
    head, *rest = [c.strip() for c in clauses if c.strip()]
    parts = [head.rstrip(".!?")] + [_lower_first(c).rstrip(".!?") for c in rest]
    return "; ".join(parts) + "."


def _bridge_resolvable(
    pair: Mapping[str, object],
    facts_by_id: Mapping[str, Mapping[str, object]],
    require_chunk_ids: bool,
) -> bool:
    """T1: a bridge pair can only be drafted if both its facts still exist in
    the current input set and (when strict grounding is on) each carries a
    real chunk_id. Checked BEFORE any drafting call -- the old flow only
    discovered a missing/ungrounded fact after paying for a cluster-side or
    v1 bridge draft (F6: 66 fallback jobs drafted for facts that could never
    ground)."""
    rec_a = facts_by_id.get(str(pair.get("fact_a") or ""))
    rec_b = facts_by_id.get(str(pair.get("fact_b") or ""))
    if rec_a is None or rec_b is None:
        return False
    if require_chunk_ids:
        if not str(rec_a.get("chunk_id") or "").strip():
            return False
        if not str(rec_b.get("chunk_id") or "").strip():
            return False
    return True


def _bridge_surface_gate(pair: Mapping[str, object]) -> tuple[bool, dict]:
    """T2: gate a bridge pair's surface through ``bridge_quality.surface_ok``
    BEFORE it may seed a cluster or fall back to the v1 2-fact path (kills
    F1: junk bridge surfaces like "di erent"/"than 0" shipped in round-2).

    Returns (ok, pair). When the surface was repaired (e.g. an OCR split),
    the returned pair is a COPY with ``bridge_entity``/``bridge_norm``
    updated to the repaired text so every downstream consumer (cluster
    build, prompts, the final QA record) sees the repaired surface, never
    the junk original."""
    raw = str(pair.get("bridge_entity") or pair.get("bridge_norm") or "")
    ok, detail = surface_ok(raw)
    if not ok:
        return False, dict(pair)
    if detail.strip() != raw.strip():
        repaired = dict(pair)
        repaired["bridge_entity"] = detail
        repaired["bridge_norm"] = detail.lower()
        return True, repaired
    return True, dict(pair)


def _pair_cache_key(pair: Mapping[str, object]) -> str:
    return f"{pair.get('fact_a')}|{pair.get('fact_b')}"


def _cluster_cache_key(cluster: Mapping[str, object]) -> str:
    return (
        f"{cluster.get('fact_a')}|{cluster.get('fact_a2')}|"
        f"{cluster.get('fact_b')}|{cluster.get('fact_b2')}|{cluster.get('bridge_norm')}"
    )


def _side_cache_key(fid1: str, fid2: str) -> str:
    return f"{fid1}|{fid2}"


def _question_cache_key(cluster_key: str, clauses: Sequence[str]) -> str:
    # a retried side changes its clauses, which must invalidate the question
    digest = hashlib.md5("|".join(clauses).encode("utf-8")).hexdigest()[:12]
    return f"{cluster_key}|q{digest}"


def _draft_question(prompt: str, llm: object, settings: Mapping[str, object],
                    *, key: str, bridge_forms: Sequence[str]) -> dict:
    # the question call is short and cheap; hiding the bridge in a 4-clause
    # question is the hardest constraint, so it gets its own attempt budget
    attempts = int(settings.get("question_generation_attempts",
                                settings["generation_attempts"]))
    for attempt in range(attempts):
        try:
            temperature = float(
                settings["temperature_initial"] if attempt == 0 else settings["temperature_retry"]
            )
            raw = llm.generate_json(
                prompt, temperature=temperature, max_tokens=int(settings["bridge_max_tokens"])
            )
        except Exception as exc:
            if attempt == attempts - 1:
                return {"kind": "clq", "key": key,
                        "error": f"generation_error:{type(exc).__name__}:{exc}"}
            continue
        question = _clean_question(raw.get("question"), min_chars=int(settings["min_question_chars"]))
        if question and bridge_is_hidden(question, *bridge_forms):
            return {"kind": "clq", "key": key, "question": question}
        # T6: quote the exact form that leaked instead of a generic notice --
        # 26% of round-2 draft jobs died invalid_or_bridge_leak, and a vague
        # retry message left the model guessing what to remove.
        leaked = next(
            (form for form in bridge_forms if form and not bridge_is_hidden(question or "", form)),
            None,
        )
        if question and leaked:
            prompt += f"\nYour previous question contained '{leaked}' — remove it. Return a corrected JSON object."
        else:
            prompt += "\nYour previous question was invalid or contained the bridge concept. Return a corrected JSON object."
    return {"kind": "clq", "key": key, "error": "invalid_or_bridge_leak"}


def draft_neighbor_pairs(
    pairs: Sequence[Mapping[str, object]],
    facts_by_id: Mapping[str, Mapping[str, object]],
    llm: object,
    settings: Mapping[str, object],
    *,
    doc_name: str,
    workers: int,
    cache_path: Path | None,
) -> dict[str, dict]:
    """Draft-only step for neighbor-pair (2 same-page facts, single-hop)
    candidates: one LLM draft call per pair (or a cache hit), no NLI
    scoring and no accept/reject decision -- that is ``gate_neighbor_pairs``.

    Artifact-in/artifact-out: ``pairs`` is the FREE candidate list from
    ``sample_neighbor_pairs``; the return value is the same
    ``{"pairv2:<key>": draft}`` cache shape ``_run_drafting`` has always
    produced, so it composes unchanged with ``gate_neighbor_pairs``.
    """
    jobs: list[tuple[str, str, Callable[[], dict]]] = []
    for pair in pairs:
        a, b = facts_by_id[pair["fact_a"]], facts_by_id[pair["fact_b"]]
        key = _pair_cache_key(pair)
        prompt = _pair_prompt(_fact_text(a), _fact_text(b), doc_name, form_hint=_form_hint(key))
        jobs.append(("pairv2", key, lambda p=prompt, k=key: _draft_with_retries(
            p, llm, settings, kind="pairv2", key=k, clause_keys=("clause_a", "clause_b"))))
    return _run_drafting(jobs, workers=workers, cache_path=cache_path,
                         progress_every=int(settings["progress_every"]))


def draft_clusters(
    clusters: Sequence[Mapping[str, object]],
    facts_by_id: Mapping[str, Mapping[str, object]],
    llm: object,
    settings: Mapping[str, object],
    *,
    workers: int,
    cache_path: Path | None,
    nli_fn: Callable[[list[tuple[str, str]]], list[float]],
    doc_name: str,
    clause_min: float,
) -> tuple[dict[str, dict], dict]:
    """Draft-only step for cluster (4-fact bridge) candidates: two-stage
    drafting, artifact-in (clusters + facts) / artifact-out (drafts + meta).

    Stage 1 drafts two clauses per page-side (short calls, sides deduplicated
    across clusters), pre-gates every clause with the free local NLI, and
    retries a failing side ONCE with feedback naming the failing clause.
    Stage 2 spends the question call only on clusters whose four clauses
    already pass, so a hard clause never wastes a question draft. The NLI
    pre-gate lives here (not in gate_clusters) because it decides whether a
    side is even worth spending a question call on -- it is part of the
    drafting cost-control, not the final accept/reject decision.

    Returns ({f"cluster:{cluster_key}": draft}, meta). A draft is either
    {"clauses": [4], "question": str} or {"error": "clause_pregate" | ...}.
    """
    progress = int(settings["progress_every"])

    sides: dict[str, tuple[str, str]] = {}
    for c in clusters:
        for f1, f2 in ((str(c["fact_a"]), str(c["fact_a2"])),
                       (str(c["fact_b"]), str(c["fact_b2"]))):
            sides.setdefault(_side_cache_key(f1, f2), (f1, f2))

    def _side_job(f1: str, f2: str, kind: str, feedback: str = ""):
        prompt = _side_prompt(_fact_text(facts_by_id[f1]), _fact_text(facts_by_id[f2]), doc_name) + feedback
        key = _side_cache_key(f1, f2)
        return (kind, key, lambda p=prompt, k=key, kd=kind: _draft_with_retries(
            p, llm, settings, kind=kd, key=k, clause_keys=("clause_1", "clause_2"),
            require_question=False))

    side_drafts = _run_drafting(
        [_side_job(f1, f2, "side") for f1, f2 in sides.values()],
        workers=workers, cache_path=cache_path, progress_every=progress,
    )

    def _gate(keys: Sequence[str], drafts: Mapping[str, Mapping[str, object]],
              kind: str) -> dict[str, list[float]]:
        """Clause NLI per side; sides whose draft errored are simply absent."""
        valid = [k for k in keys if drafts.get(f"{kind}:{k}", {}).get("clauses")]
        inputs: list[tuple[str, str]] = []
        for k in valid:
            f1, f2 = sides[k]
            clauses = drafts[f"{kind}:{k}"]["clauses"]
            inputs.append((_fact_text(facts_by_id[f1]), str(clauses[0])))
            inputs.append((_fact_text(facts_by_id[f2]), str(clauses[1])))
        scores = nli_fn(inputs)
        if len(scores) != len(inputs):
            raise RuntimeError(f"NLI returned {len(scores)} scores for {len(inputs)} inputs")
        return {k: [float(scores[2 * i]), float(scores[2 * i + 1])] for i, k in enumerate(valid)}

    def _passes(sc: list[float] | None) -> bool:
        return sc is not None and all(s >= clause_min for s in sc)

    all_keys = sorted(sides)
    first_scores = _gate(all_keys, side_drafts, "side")

    final_clauses: dict[str, list[str]] = {}
    failing_keys: list[str] = []
    for k in all_keys:
        if _passes(first_scores.get(k)):
            final_clauses[k] = [str(c) for c in side_drafts[f"side:{k}"]["clauses"]]
        else:
            failing_keys.append(k)

    n_side_retries = len(failing_keys)
    if failing_keys:
        retry_jobs = []
        for k in failing_keys:
            f1, f2 = sides[k]
            sc = first_scores.get(k)
            if sc is None:
                feedback = "\nYour previous draft was invalid. Return a corrected JSON object."
            else:
                notes = [
                    f"{name} was not entailed by {src}"
                    for (name, src), s in zip(
                        (("clause_1", "SOURCE 1"), ("clause_2", "SOURCE 2")), sc)
                    if s < clause_min
                ]
                feedback = (
                    "\nFeedback on your previous draft: " + "; ".join(notes)
                    + ". Rewrite BOTH clauses, staying strictly within each source's"
                    " own wording. Return only the JSON object."
                )
            retry_jobs.append(_side_job(f1, f2, "side_r1", feedback))
        retry_drafts = _run_drafting(retry_jobs, workers=workers,
                                     cache_path=cache_path, progress_every=progress)
        retry_scores = _gate(failing_keys, retry_drafts, "side_r1")
        for k in failing_keys:
            if _passes(retry_scores.get(k)):
                final_clauses[k] = [str(c) for c in retry_drafts[f"side_r1:{k}"]["clauses"]]

    out: dict[str, dict] = {}
    ready: list[tuple[Mapping[str, object], str, list[str]]] = []
    for c in clusters:
        ckey = _cluster_cache_key(c)
        ka = _side_cache_key(str(c["fact_a"]), str(c["fact_a2"]))
        kb = _side_cache_key(str(c["fact_b"]), str(c["fact_b2"]))
        if ka in final_clauses and kb in final_clauses:
            ready.append((c, ckey, final_clauses[ka] + final_clauses[kb]))
        else:
            out[f"cluster:{ckey}"] = {"kind": "cluster", "key": ckey, "error": "clause_pregate"}

    q_jobs = []
    for c, ckey, clauses in ready:
        qkey = _question_cache_key(ckey, clauses)
        prompt = _question_prompt(clauses, str(c["bridge_entity"]), doc_name,
                                  bridge_norm=str(c["bridge_norm"]), form_hint=_form_hint(qkey))
        forms = (str(c["bridge_entity"]), str(c["bridge_norm"]))
        q_jobs.append(("clq", qkey, lambda p=prompt, k=qkey, f=forms: _draft_question(
            p, llm, settings, key=k, bridge_forms=f)))
    q_drafts = _run_drafting(q_jobs, workers=workers, cache_path=cache_path,
                             progress_every=progress)

    for c, ckey, clauses in ready:
        qd = q_drafts.get(f"clq:{_question_cache_key(ckey, clauses)}", {})
        if qd.get("question"):
            out[f"cluster:{ckey}"] = {"kind": "cluster", "key": ckey,
                                      "clauses": clauses, "question": str(qd["question"])}
        else:
            out[f"cluster:{ckey}"] = {"kind": "cluster", "key": ckey,
                                      "error": str(qd.get("error") or "invalid_or_bridge_leak")}
    meta = {"n_side_retries": n_side_retries, "n_sides": len(sides),
            "n_questions_drafted": len(q_jobs)}
    return out, meta


# Back-compat alias: this function was private (`_draft_clusters_two_stage`)
# before the M4 block-SDK refactor exposed it as `draft_clusters`.
_draft_clusters_two_stage = draft_clusters


def _draft_with_retries(prompt: str, llm: object, settings: Mapping[str, object],
                        *, kind: str, key: str, clause_keys: Sequence[str],
                        bridge_forms: Sequence[str] = (),
                        require_question: bool = True) -> dict:
    attempts = int(settings["generation_attempts"])
    for attempt in range(attempts):
        try:
            temperature = float(
                settings["temperature_initial"] if attempt == 0 else settings["temperature_retry"]
            )
            raw = llm.generate_json(
                prompt, temperature=temperature, max_tokens=int(settings["bridge_max_tokens"])
            )
        except Exception as exc:
            if attempt == attempts - 1:
                return {"kind": kind, "key": key,
                        "error": f"generation_error:{type(exc).__name__}:{exc}"}
            continue
        clauses = [
            _clean_clause(raw.get(k), min_chars=int(settings["min_clause_chars"]))
            for k in clause_keys
        ]
        question = _clean_question(raw.get("question"), min_chars=int(settings["min_question_chars"]))
        question_ok = bool(question) or not require_question
        if all(clauses) and question_ok and (
            not bridge_forms or not question or bridge_is_hidden(question, *bridge_forms)
        ):
            out = {"kind": kind, "key": key, "clauses": clauses}
            if question:
                out["question"] = question
            return out
        prompt += "\nYour previous draft was invalid or leaked the bridge. Return a corrected JSON object."
    return {"kind": kind, "key": key, "error": "invalid_or_bridge_leak"}


def _qa_record(*, hop_type: str, strategy: str, question: str, clauses: Sequence[str],
               records: Sequence[Mapping[str, object]], clause_nli: Sequence[float],
               necessity: Mapping[str, object], nli_extra: Mapping[str, object],
               bridge_entity: str = "", source_pair_id: str = "",
               doc: str = "") -> dict:
    answer = _compose_multi(clauses)
    fact_ids = [_fact_id(r) for r in records]
    groundings = [_grounding(r) for r in records]
    return {
        "qa_id": "",
        "hop_type": hop_type,
        "evidence_strategy": strategy,
        "bridge_entity": bridge_entity,
        "question": question,
        "answer": answer,
        "answer_clauses": [
            {"text": clause, "fact_id": fid, "nli": round(score, 3)}
            for clause, fid, score in zip(clauses, fact_ids, clause_nli)
        ],
        "gold_fact_ids": fact_ids,
        "gold_chunk_ids": [g[0] for g in groundings],
        "gold_pages": [g[1] for g in groundings],
        "gold_bboxes": {fid: g[2] for fid, g in zip(fact_ids, groundings)},
        "grounding_complete": all(g[3] for g in groundings),
        "necessity": {**dict(nli_extra), **{
            "necessity_score": necessity["necessity_score"],
            "necessary_fact_ids": necessity["necessary_fact_ids"],
            "loo_entailment": necessity["loo_entailment"],
            "passed": True,
        }},
        "verify": {"bridge_hidden": bool(bridge_entity) or None, "duplicate": False,
                   "faithful": True, "verdict": "PENDING_STAGE_D"},
        "source_pair_id": source_pair_id,
        "doc": doc,
    }


def gate_qa_group(
    items: Sequence[Mapping[str, object]],
    facts_by_id: Mapping[str, Mapping[str, object]],
    drafts: Mapping[str, Mapping[str, object]],
    *,
    kind: str,
    fact_keys: Sequence[str],
    strategy: str,
    hop_type: str,
    cache_key_fn: Callable[[Mapping[str, object]], str],
    thresholds: Mapping[str, float],
    require_chunk_ids: bool,
    nli_fn: Callable[[list[tuple[str, str]]], list[float]],
    necessity_fn: Callable[[Sequence[tuple[str, Sequence[object]]]], Sequence[dict | None]],
    seen_questions: set[str],
    doc: str,
) -> tuple[list[dict], Counter]:
    """Gate-only step, shared for neighbour pairs (2 facts) and clusters
    (4 facts): candidates + their drafts in, accepted QA records + a
    per-reason rejection Counter out. No drafting happens here -- a
    candidate whose draft already errored is simply rejected on its
    recorded error reason.

    NLI layout per candidate (n = len(fact_keys)):
      n clause checks (fact_i -> clause_i)              each >= clause_min
      n single-sufficiency checks (fact_i -> answer)    each <  single_max
      1 joint check (all facts concatenated -> answer)  >= joint_min

    T8: ``necessity_fn`` here is BATCH-shaped -- ``[(answer, facts), ...] ->
    [dict|None, ...]`` -- so every candidate surviving the clause/single/
    joint NLI gates has its leave-one-out necessity scored in ONE call
    instead of one ``nli_batch`` round-trip per candidate.
    """
    n = len(fact_keys)
    rejected: Counter = Counter()
    candidates: list[tuple[Mapping[str, object], Mapping[str, object], list]] = []
    nli_inputs: list[tuple[str, str]] = []
    for item in items:
        draft = drafts.get(f"{kind}:{cache_key_fn(item)}", {})
        if draft.get("error"):
            rejected[str(draft["error"]).split(":", 1)[0]] += 1
            continue
        records = [facts_by_id.get(str(item.get(k) or "")) for k in fact_keys]
        if any(r is None for r in records):
            rejected["missing_fact"] += 1
            continue
        clauses = [str(c) for c in draft["clauses"]]
        answer = _compose_multi(clauses)
        candidates.append((item, draft, records))
        texts = [_fact_text(r) for r in records]
        nli_inputs.extend((texts[i], clauses[i]) for i in range(n))
        nli_inputs.extend((texts[i], answer) for i in range(n))
        nli_inputs.append((" ".join(texts), answer))

    scores = nli_fn(nli_inputs)
    if len(scores) != len(nli_inputs):
        raise RuntimeError(f"NLI returned {len(scores)} scores for {len(nli_inputs)} inputs")

    per = 2 * n + 1
    staged: list[dict] = []
    for index, (item, draft, records) in enumerate(candidates):
        block = [float(v) for v in scores[index * per:(index + 1) * per]]
        clause_scores, single_scores, joint = block[:n], block[n:2 * n], block[2 * n]
        if any(s < thresholds["clause_min"] for s in clause_scores):
            rejected["clause_not_entailed"] += 1
            continue
        if any(s >= thresholds["single_max"] for s in single_scores) or joint < thresholds["joint_min"]:
            rejected["not_jointly_necessary"] += 1
            continue
        clauses = [str(c) for c in draft["clauses"]]
        answer = _compose_multi(clauses)
        staged.append({
            "item": item, "draft": draft, "records": records,
            "clause_scores": clause_scores, "single_scores": single_scores,
            "joint": joint, "clauses": clauses, "answer": answer,
        })

    # T8: every surviving candidate's leave-one-out check goes into ONE
    # necessity_fn call instead of one per candidate.
    loo_items = [(s["answer"], [_as_fact(r) for r in s["records"]]) for s in staged]
    necessities = necessity_fn(loo_items, threshold=thresholds["single_max"]) if staged else []

    output: list[dict] = []
    for s, necessity in zip(staged, necessities):
        if not necessity or float(necessity.get("necessity_score", 0.0)) < 1.0:
            rejected["loo_failed"] += 1
            continue
        question_norm = _normal_form(str(s["draft"]["question"]))
        if question_norm in seen_questions:
            rejected["duplicate_question"] += 1
            continue
        record = _qa_record(
            hop_type=hop_type, strategy=strategy, question=str(s["draft"]["question"]),
            clauses=s["clauses"], records=s["records"], clause_nli=s["clause_scores"],
            necessity=necessity,
            nli_extra={"nli_singles": [round(x, 3) for x in s["single_scores"]],
                       "nli_joint": round(s["joint"], 3)},
            bridge_entity=str(s["item"].get("bridge_entity") or ""),
            source_pair_id=str(s["item"].get("source_pair_id") or s["item"].get("pair_id") or ""),
            doc=doc,
        )
        if require_chunk_ids and not record["grounding_complete"]:
            rejected["ungrounded_chunk_ids"] += 1
            continue
        seen_questions.add(question_norm)
        output.append(record)
    return output, rejected


# Back-compat alias: this function was private (`_score_group`) before the
# M4 block-SDK refactor exposed it as `gate_qa_group`.
_score_group = gate_qa_group


def gate_neighbor_pairs(
    pairs: Sequence[Mapping[str, object]],
    facts_by_id: Mapping[str, Mapping[str, object]],
    drafts: Mapping[str, Mapping[str, object]],
    *,
    thresholds: Mapping[str, float],
    require_chunk_ids: bool,
    nli_fn: Callable[[list[tuple[str, str]]], list[float]],
    necessity_fn: Callable[[Sequence[tuple[str, Sequence[object]]]], Sequence[dict | None]],
    seen_questions: set[str],
    doc: str,
) -> tuple[list[dict], Counter]:
    """Gate-only step for neighbor-pair drafts (matches the ``qa_gen_pairs``
    block's gate boundary): thin partial application of ``gate_qa_group``
    fixing the neighbor-pair-specific constants (kind, fact_keys, strategy,
    hop_type, cache key)."""
    return gate_qa_group(
        pairs, facts_by_id, drafts,
        kind="pairv2", fact_keys=("fact_a", "fact_b"),
        strategy="neighbor_pair_v2", hop_type="single",
        cache_key_fn=_pair_cache_key, thresholds=thresholds,
        require_chunk_ids=require_chunk_ids, nli_fn=nli_fn,
        necessity_fn=necessity_fn, seen_questions=seen_questions, doc=doc,
    )


def gate_clusters(
    clusters: Sequence[Mapping[str, object]],
    facts_by_id: Mapping[str, Mapping[str, object]],
    drafts: Mapping[str, Mapping[str, object]],
    *,
    thresholds: Mapping[str, float],
    require_chunk_ids: bool,
    nli_fn: Callable[[list[tuple[str, str]]], list[float]],
    necessity_fn: Callable[[Sequence[tuple[str, Sequence[object]]]], Sequence[dict | None]],
    seen_questions: set[str],
    doc: str,
) -> tuple[list[dict], Counter]:
    """Gate-only step for cluster (4-fact bridge) drafts (matches the
    ``qa_gen_clusters`` block's gate boundary). Same partial-application
    shape as ``gate_neighbor_pairs``, fixing the 4-fact cluster constants."""
    return gate_qa_group(
        clusters, facts_by_id, drafts,
        kind="cluster", fact_keys=("fact_a", "fact_a2", "fact_b", "fact_b2"),
        strategy="cluster_2plus2", hop_type="bridge",
        cache_key_fn=_cluster_cache_key, thresholds=thresholds,
        require_chunk_ids=require_chunk_ids, nli_fn=nli_fn,
        necessity_fn=necessity_fn, seen_questions=seen_questions, doc=doc,
    )


def _demoted_pair(cluster: Mapping[str, object]) -> dict:
    """Recast a failed 4-fact cluster back into the v1 2-fact bridge-pair
    shape it was seeded from."""
    return {
        "doc": cluster["doc"], "fact_a": cluster["fact_a"], "fact_b": cluster["fact_b"],
        "bridge_entity": cluster["bridge_entity"], "bridge_norm": cluster["bridge_norm"],
        "pair_id": cluster["source_pair_id"],
    }


def _merge_v1_stats(first: Mapping[str, object], second: Mapping[str, object]) -> dict:
    """T7: the fallback/demotion drafting may now run in two waves (an early
    one overlapped with cluster scoring, plus a small late one for clusters
    only discovered as failed AFTER scoring) -- merge their v1 stats blocks
    into one so callers see a single, complete picture."""
    if not first:
        return dict(second)
    if not second:
        return dict(first)
    merged = dict(first)
    for key in ("n_input_facts", "n_input_bridge_pairs", "n_bridge_pass",
               "n_bridge_rejected", "n_single_pass", "n_single_rejected",
               "n_pairs_ungrounded"):
        merged[key] = int(first.get(key) or 0) + int(second.get(key) or 0)
    for key in ("bridge_rejection_reasons", "single_rejection_reasons"):
        combined = Counter(first.get(key) or {}) + Counter(second.get(key) or {})
        merged[key] = dict(sorted(combined.items()))
    return merged


def _resolve_necessity_group_fn(
    necessity_fn: Callable[..., dict | None],
    necessity_batch_fn: Callable[..., Sequence[dict | None]] | None,
) -> Callable[..., Sequence[dict | None]]:
    """T8: ``gate_qa_group`` scores a whole GROUP of candidates in one
    necessity call. ``necessity_fn`` stays single-item (answer, facts) ->
    dict|None for backward compat with the v1 fallback path
    (``build_answer_first_pairs`` still calls it per-candidate) and with
    existing single-item test fixtures; when the caller has not supplied a
    real batch function, one is derived here: the real default resolves to
    the genuinely batched ``leave_one_out_necessity_batch``, and any OTHER
    single-item override (e.g. a test fake) is auto-wrapped so it still
    controls acceptance without every caller needing to migrate to the
    batch signature. Shared by ``build_v2_pairs`` and the ``qa_gen_pairs``/
    ``qa_gen_clusters`` block adapters so the wrapping rule lives in one
    place."""
    if necessity_batch_fn is not None:
        return necessity_batch_fn
    if necessity_fn is leave_one_out_necessity:
        return leave_one_out_necessity_batch

    def necessity_group_fn(items, *, threshold=None, _fn=necessity_fn):
        return [_fn(answer, facts, threshold=threshold) for answer, facts in items]

    return necessity_group_fn


def build_v2_pairs(
    facts: Sequence[Mapping[str, object]],
    bridge_pairs: Sequence[Mapping[str, object]],
    llm: object,
    *,
    doc: str,
    workers: int | None = None,
    draft_cache_path: Path | None = None,
    pair_limit: int | None = None,
    cluster_limit: int | None = None,
    include_fallback: bool = True,
    require_chunk_ids: bool = True,
    embed_fn: Callable[[list[str]], object] | None = None,
    min_cosine: float = 0.40,
    max_cosine: float | None = 0.95,
    pair_overlap: int = 1,
    demote_failed_clusters: bool = True,
    nli_fn: Callable[[list[tuple[str, str]]], list[float]] = nli_batch,
    necessity_fn: Callable[..., dict | None] = leave_one_out_necessity,
    necessity_batch_fn: Callable[..., Sequence[dict | None]] | None = None,
) -> dict:
    import time as _time
    timings: dict[str, float] = {}
    _t0 = _time.perf_counter()
    settings = _settings()
    workers = int(settings["workers"] if workers is None else workers)
    thresholds = {
        "clause_min": float(settings["clause_entailment_min"]),
        "single_max": float(settings["single_fact_answer_max"]),
        "joint_min": float(settings["joint_answer_min"]),
    }

    necessity_group_fn = _resolve_necessity_group_fn(necessity_fn, necessity_batch_fn)
    doc_name = _doc_name(doc, settings)
    facts = [dict(f) for f in facts]
    facts_by_id = {_fact_id(f): f for f in facts if _fact_id(f)}
    doc_bridges_raw = [p for p in bridge_pairs if str(p.get("doc") or "") in ("", doc)]

    n_bridge_surface_rejected = 0
    doc_bridges = []
    for p in doc_bridges_raw:
        ok, gated = _bridge_surface_gate(p)
        if ok:
            doc_bridges.append(gated)
        else:
            n_bridge_surface_rejected += 1

    n_pre_rejected_unresolvable = sum(
        1 for p in doc_bridges if not _bridge_resolvable(p, facts_by_id, require_chunk_ids)
    )
    doc_bridges = [
        p for p in doc_bridges if _bridge_resolvable(p, facts_by_id, require_chunk_ids)
    ]

    neighbor_pairs = sample_neighbor_pairs(
        facts, doc=doc, embed_fn=embed_fn, max_pairs=pair_limit,
        min_cosine=min_cosine, max_cosine=max_cosine,
        max_uses_per_fact=pair_overlap,
    )
    clusters, fallback = build_clusters(
        doc_bridges, facts, embed_fn=embed_fn,
        min_cosine=min_cosine, max_cosine=max_cosine,
    )
    timings["sampling"] = round(_time.perf_counter() - _t0, 2)
    _t0 = _time.perf_counter()
    if cluster_limit is not None:
        overflow = clusters[cluster_limit:]
        clusters = clusters[:cluster_limit]
        # over-limit clusters are NOT silently dropped; their seeds go to fallback
        fallback = fallback + [
            {"doc": c["doc"], "fact_a": c["fact_a"], "fact_b": c["fact_b"],
             "bridge_entity": c["bridge_entity"], "bridge_norm": c["bridge_norm"],
             "pair_id": c["source_pair_id"]}
            for c in overflow
        ]

    drafts = draft_neighbor_pairs(
        neighbor_pairs, facts_by_id, llm, settings,
        doc_name=doc_name, workers=workers, cache_path=draft_cache_path,
    )
    cluster_drafts, cluster_meta = draft_clusters(
        clusters, facts_by_id, llm, settings,
        workers=workers, cache_path=draft_cache_path, nli_fn=nli_fn,
        doc_name=doc_name, clause_min=thresholds["clause_min"],
    )
    drafts = {**drafts, **cluster_drafts}
    timings["drafting"] = round(_time.perf_counter() - _t0, 2)

    # T7 (fixes F7: fallback drafting was 65% of wall time, running
    # sequentially AFTER cluster scoring): the moment drafting is done we
    # already know an EARLY demotion set -- structural fallback (no
    # neighbour found), cluster_limit overflow, and clusters whose draft
    # itself already errored (clause_pregate / invalid_or_bridge_leak).
    # Launch the v1 fallback/demotion drafting (network-bound) for that set
    # on a background thread now, concurrently with pair/cluster NLI scoring
    # (GPU-bound) on the main thread -- the two do not contend for the same
    # resource. A cluster that only fails at the final NLI/LOO gate is
    # discovered too late for this wave and is drafted in a small second,
    # sequential wave after scoring.
    early_draft_failed_ids: set[str] = set()
    if include_fallback and demote_failed_clusters:
        early_draft_failed_ids = {
            c["source_pair_id"] for c in clusters
            if cluster_drafts.get(f"cluster:{_cluster_cache_key(c)}", {}).get("error")
        }
    existing_fallback_ids = {str(p.get("pair_id") or "") for p in fallback}
    early_demoted = [
        _demoted_pair(c) for c in clusters
        if c["source_pair_id"] in early_draft_failed_ids
        and c["source_pair_id"] not in existing_fallback_ids
    ]
    early_fallback_input = fallback + early_demoted

    fallback_executor: ThreadPoolExecutor | None = None
    fallback_future = None
    _fb_t0 = _time.perf_counter()
    if include_fallback and early_fallback_input:
        fallback_executor = ThreadPoolExecutor(max_workers=1)
        fallback_future = fallback_executor.submit(
            build_answer_first_pairs, facts, early_fallback_input, llm,
            include_singles=False, workers=workers,
            draft_cache_path=draft_cache_path, require_chunk_ids=require_chunk_ids,
            nli_fn=nli_fn, necessity_fn=necessity_fn,
        )

    _t0 = _time.perf_counter()
    seen_questions: set[str] = set()
    pair_out, pair_rej = gate_neighbor_pairs(
        neighbor_pairs, facts_by_id, drafts, thresholds=thresholds,
        require_chunk_ids=require_chunk_ids, nli_fn=nli_fn,
        necessity_fn=necessity_group_fn, seen_questions=seen_questions, doc=doc,
    )

    # T8 (fixes F8): snapshot the NLI truncation counters around cluster
    # scoring specifically -- the 4-fact joint premise (up to 628 chars in
    # round-2) is the risk case, so its truncation is tracked separately
    # from the rest of the run and WARNed on rather than silently absorbed.
    _cluster_trunc_before = truncation_stats()
    cluster_out, cluster_rej = gate_clusters(
        clusters, facts_by_id, drafts, thresholds=thresholds,
        require_chunk_ids=require_chunk_ids, nli_fn=nli_fn,
        necessity_fn=necessity_group_fn, seen_questions=seen_questions, doc=doc,
    )
    _cluster_trunc_after = truncation_stats()
    cluster_truncation = {
        key: _cluster_trunc_after.get(key, 0) - _cluster_trunc_before.get(key, 0)
        for key in _cluster_trunc_after
    }
    if cluster_truncation.get("premise_truncated", 0) > 0:
        logger.warning(
            f"NLI premise truncation occurred during cluster scoring for doc={doc}: "
            f"{cluster_truncation}"
        )

    _scoring_end = _time.perf_counter()
    timings["scoring"] = round(_scoring_end - _t0, 2)
    _t0 = _time.perf_counter()

    # A failed 4-fact cluster still rests on a VERIFIED cross-page bridge
    # pair — demote it to the v1 2-fact path rather than discarding the
    # multi-hop entirely (the 2-fact question faces the same NLI/LOO gates).
    n_demoted = 0
    late_demoted: list[dict] = []
    if include_fallback and demote_failed_clusters:
        passed_ids = {qa["source_pair_id"] for qa in cluster_out}
        n_demoted = sum(1 for c in clusters if c["source_pair_id"] not in passed_ids)
        already_handled = early_draft_failed_ids | existing_fallback_ids
        late_demoted = [
            _demoted_pair(c) for c in clusters
            if c["source_pair_id"] not in passed_ids
            and c["source_pair_id"] not in already_handled
        ]

    fallback_out: list[dict] = []
    fallback_stats: dict = {}
    fallback_overlap_sec = 0.0
    if fallback_future is not None:
        v1_early = fallback_future.result()
        fallback_executor.shutdown(wait=True)
        _fb_end = _time.perf_counter()
        # the wall-clock window during which the fallback thread and the
        # main-thread scoring genuinely coexisted (both started at _fb_t0)
        fallback_overlap_sec = round(max(0.0, min(_scoring_end, _fb_end) - _fb_t0), 2)
        fallback_out.extend(v1_early["pairs"])
        fallback_stats = v1_early["stats"]
    if include_fallback and late_demoted:
        v1_late = build_answer_first_pairs(
            facts, late_demoted, llm, include_singles=False, workers=workers,
            draft_cache_path=draft_cache_path, require_chunk_ids=require_chunk_ids,
            nli_fn=nli_fn, necessity_fn=necessity_fn,
        )
        fallback_out.extend(v1_late["pairs"])
        fallback_stats = _merge_v1_stats(fallback_stats, v1_late["stats"])
    for qa in fallback_out:
        qa["evidence_strategy"] = "bridge_pair_v1"
    timings["fallback"] = round(_time.perf_counter() - _t0, 2)
    timings["fallback_overlap_sec"] = fallback_overlap_sec

    merged = pair_out + cluster_out + fallback_out
    for index, item in enumerate(merged, start=1):
        item["qa_id"] = f"V2Q{index:05d}"

    # T5: first-word histogram makes the F4 "What" monoculture (90% in
    # round-2) directly observable per run.
    first_word_histogram: Counter = Counter(
        str(item.get("question", "")).strip().split(" ", 1)[0].rstrip(",:;")
        for item in merged
        if str(item.get("question", "")).strip()
    )

    return {
        "pairs": merged,
        "stats": {
            "doc": doc,
            "n_input_facts": len(facts),
            "n_bridge_surface_rejected": n_bridge_surface_rejected,
            "pre_rejected_unresolvable": n_pre_rejected_unresolvable,
            "n_neighbor_pairs_sampled": len(neighbor_pairs),
            "n_clusters_built": len(clusters),
            "n_fallback_bridges": len(fallback),
            "n_pair_pass": len(pair_out),
            "pair_rejection_reasons": dict(sorted(pair_rej.items())),
            "n_cluster_pass": len(cluster_out),
            "cluster_rejection_reasons": dict(sorted(cluster_rej.items())),
            "n_side_retries": cluster_meta["n_side_retries"],
            "n_sides_drafted": cluster_meta["n_sides"],
            "n_questions_drafted": cluster_meta["n_questions_drafted"],
            "n_fallback_pass": len(fallback_out),
            "n_clusters_demoted": n_demoted,
            "fallback_stats": fallback_stats,
            "thresholds": thresholds,
            "require_chunk_ids": require_chunk_ids,
            "timings_sec": timings,
            "question_first_word_histogram": dict(
                sorted(first_word_histogram.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "nli_truncation": {
                "cluster": cluster_truncation,
                "cumulative": truncation_stats(),
            },
        },
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage C v2 evidence-dense QA generation")
    parser.add_argument("facts", type=Path)
    parser.add_argument("bridge_pairs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--doc", required=True, help="doc id (filters bridge pairs, names prompts)")
    parser.add_argument("--pair-limit", type=int)
    parser.add_argument("--cluster-limit", type=int)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--draft-cache", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--require-chunk-ids", action="store_true", default=True)
    parser.add_argument(
        "--embed", action="store_true",
        help="apply the cosine band to neighbour selection using the local "
        "sentence-transformer (vectorstore.embedding_model)",
    )
    parser.add_argument("--min-cosine", type=float, default=0.40)
    parser.add_argument("--max-cosine", type=float, default=0.95)
    parser.add_argument(
        "--pair-overlap", type=int, default=1,
        help="max pairs a fact may join for neighbor-pair singles (2 ~doubles volume)",
    )
    return parser.parse_args(argv)


def _local_embed_fn():
    from sentence_transformers import SentenceTransformer

    from rag_gt.core.config import load_config

    model_name = str(load_config()["vectorstore"]["embedding_model"])
    model = SentenceTransformer(model_name)

    def embed(texts: list[str]):
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    return embed


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    facts = _load_records(args.facts)
    bridge_pairs = _load_records(args.bridge_pairs, "pairs")
    result = build_v2_pairs(
        facts, bridge_pairs, get_llm("gt"),
        doc=args.doc,
        workers=args.workers,
        draft_cache_path=args.draft_cache or args.output.with_suffix(".drafts.jsonl"),
        pair_limit=args.pair_limit,
        cluster_limit=args.cluster_limit,
        include_fallback=not args.no_fallback,
        require_chunk_ids=args.require_chunk_ids,
        embed_fn=_local_embed_fn() if args.embed else None,
        min_cosine=args.min_cosine,
        max_cosine=args.max_cosine,
        pair_overlap=args.pair_overlap,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Stage C v2 wrote {args.output}: {result['stats']}")
    return 0 if (result["stats"]["n_pair_pass"] + result["stats"]["n_cluster_pass"]) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
