"""GT JSONL read/write. Output is immutable after Phase 1.

Writes go to a temp file and `os.replace` onto the final path so a crash
never leaves a half-written, partially-readable JSONL behind.
JSON output uses `ensure_ascii=False` so non-ASCII text is preserved
verbatim in the file (instead of `\\u00e9`-style escapes).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from rag_gt.core.config import repo_root
from rag_gt.core.types import Fact, MSFS, QuestionGT, Span

load_dotenv()


def _resolve_gt_dir(out_dir: Optional[Path | str]) -> Path:
    if out_dir is not None:
        return Path(out_dir)
    return Path(os.getenv("GT_OUTPUT_DIR") or str(repo_root() / "data" / "gt"))


def _fact_to_dict(f: Fact) -> dict:
    """Serialize a Fact, including optional SFU fields when populated.

    Phase 1 (plan v6): canonical_form, self_containment_score, and raw_text
    are emitted only when non-default so V10/V11 round-trip output is
    byte-identical to the previous serialiser.
    """
    out = {
        "fact_id": f.fact_id,
        "text": f.text,
        "role": f.role,
        "supporting_spans": [s.to_dict() for s in f.supporting_spans],
    }
    if f.canonical_form:
        out["canonical_form"] = f.canonical_form
    # 1.0 is the default for legacy facts; only emit when the SFU pipeline
    # has actually scored this fact below the default.
    if f.self_containment_score < 1.0:
        out["self_containment_score"] = f.self_containment_score
    if f.raw_text:
        out["raw_text"] = f.raw_text
    if f.self_containment_known:
        out["self_containment_known"] = True
    return out


def _to_dict(q: QuestionGT) -> dict:
    out = {
        "q_id": q.q_id,
        "question": q.question,
        "gold_answer": q.gold_answer,
        "msfs_list": [
            {"msfs_id": m.msfs_id, "fact_ids": m.fact_ids} for m in q.msfs_list
        ],
        "doc_ids": q.doc_ids,
        "required_fact_ids": q.required_fact_ids,
        "required_facts": [
            _fact_to_dict(f) for f in q.required_facts
        ],
        "difficulty_reasoning_depth": q.difficulty_reasoning_depth,
        "difficulty_semantic_distance": q.difficulty_semantic_distance,
    }
    optional = {
        "required_fact_groups": q.required_fact_groups,
        "hop_type": q.hop_type,
        "minimum_required_fact_count": q.minimum_required_fact_count,
        "chunk_profile_id": q.chunk_profile_id,
        "fact_to_chunk_mappings": q.fact_to_chunk_mappings,
        "relation_type": q.relation_type,
        "risky_frame_hit": q.risky_frame_hit,
        "qrsg_evidence_map": q.qrsg_evidence_map,
        # v16 PASS-GT fields — emitted only when populated so v11–v15 output is
        # byte-identical to the previous serialiser.
        "tf_sfg_edges": q.tf_sfg_edges,
        "question_assumptions": q.question_assumptions,
        "distractor_spans": q.distractor_spans,
        "twin_of": q.twin_of,
        "twin_type": q.twin_type,
        "twin_metadata": q.twin_metadata,
        "reasoning_trace": q.reasoning_trace,
        "intent": q.intent,
    }
    for key, value in optional.items():
        if value not in (None, "", [], {}):
            out[key] = value
    # difficulty_adversarial: emit only for non-default rows to preserve back-compat.
    if q.difficulty_adversarial and q.difficulty_adversarial != "clean":
        out["difficulty_adversarial"] = q.difficulty_adversarial
    return out


def _from_dict(d: dict) -> QuestionGT:
    if "required_facts" not in d:
        raise ValueError(
            "GT entry is missing 'required_facts'. Regenerate GT with the current pipeline."
        )
    required_facts = []
    for f in d["required_facts"]:
        spans = []
        for s in f.get("supporting_spans", []):
            spans.append(Span.from_any(s))
        score_raw = f.get("self_containment_score")
        self_containment_score = float(
            score_raw if score_raw is not None else 1.0
        )
        self_containment_known = bool(f.get("self_containment_known", False))
        # Span requirement is conditional on the fact's role. Roles like
        # "definition" / "consequence" produced via inference may legitimately
        # have no extractive span; only enforce the requirement when present.
        required_facts.append(
            Fact(
                fact_id=f["fact_id"],
                text=f["text"],
                role=f["role"],
                supporting_spans=spans,
                # Phase 1 (plan v6): optional SFU fields. Default values keep
                # V10/V11 GT files loading unchanged.
                canonical_form=str(f.get("canonical_form", "") or ""),
                self_containment_score=self_containment_score,
                raw_text=str(f.get("raw_text", "") or ""),
                self_containment_known=self_containment_known,
            )
        )
    return QuestionGT(
        q_id=d["q_id"],
        question=d["question"],
        gold_answer=d["gold_answer"],
        msfs_list=[
            MSFS(msfs_id=m["msfs_id"], fact_ids=m["fact_ids"]) for m in d["msfs_list"]
        ],
        doc_ids=d["doc_ids"],
        required_fact_ids=d["required_fact_ids"],
        difficulty_reasoning_depth=d["difficulty_reasoning_depth"],
        difficulty_semantic_distance=d["difficulty_semantic_distance"],
        required_facts=required_facts,
        required_fact_groups=[
            [str(fid) for fid in group]
            for group in d.get("required_fact_groups", [])
            if isinstance(group, list)
        ],
        hop_type=str(d.get("hop_type", "") or ""),
        minimum_required_fact_count=int(
            d.get("minimum_required_fact_count", 0) or 0
        ),
        chunk_profile_id=str(d.get("chunk_profile_id", "") or ""),
        fact_to_chunk_mappings=list(d.get("fact_to_chunk_mappings", []) or []),
        relation_type=str(d.get("relation_type", "") or ""),
        risky_frame_hit=d.get("risky_frame_hit") or None,
        qrsg_evidence_map=dict(d.get("qrsg_evidence_map", {}) or {}),
        # v16 PASS-GT fields — default to empty/None for v11–v15 GT files.
        tf_sfg_edges=list(d.get("tf_sfg_edges", []) or []),
        question_assumptions=list(d.get("question_assumptions", []) or []),
        distractor_spans=list(d.get("distractor_spans", []) or []),
        twin_of=d.get("twin_of") or None,
        twin_type=d.get("twin_type") or None,
        twin_metadata=d.get("twin_metadata") or None,
        reasoning_trace=list(d.get("reasoning_trace", []) or []),
        difficulty_adversarial=str(d.get("difficulty_adversarial", "clean") or "clean"),
        intent=str(d.get("intent", "") or ""),
    )


def save_gt(
    questions: List[QuestionGT],
    corpus_name: str,
    out_dir: Path | str | None = None,
) -> str:
    """Atomically write `corpus_name.jsonl` to `out_dir`."""
    target_dir = _resolve_gt_dir(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    final = target_dir / f"{corpus_name}.jsonl"
    tmp = target_dir / f"{corpus_name}.jsonl.tmp"

    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for q in questions:
            f.write(json.dumps(_to_dict(q), ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # fsync isn't supported on every filesystem (e.g., some networked
            # ones); fall back gracefully.
            pass
    os.replace(tmp, final)
    print(f"[GT] Saved {len(questions)} questions -> {final}")
    return str(final)


def save_build_summary(
    output_path: Path | str,
    summary: dict,
) -> str:
    """Write `<output_path>.build_summary.json` atomically."""
    final = Path(f"{output_path}.build_summary.json")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{final}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, final)
    return str(final)


def append_gt(
    questions: List[QuestionGT],
    corpus_name: str,
    out_dir: Path | str | None = None,
) -> str:
    """Append questions to a per-corpus JSONL file. Used by the pipeline for
    incremental persistence — one bad doc no longer wipes the run."""
    target_dir = _resolve_gt_dir(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{corpus_name}.jsonl"
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for q in questions:
            f.write(json.dumps(_to_dict(q), ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    return str(path)


def load_gt(corpus_name: str, in_dir: Path | str | None = None) -> List[QuestionGT]:
    src_dir = _resolve_gt_dir(in_dir)
    path = src_dir / f"{corpus_name}.jsonl"
    out: List[QuestionGT] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                out.append(_from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                raise ValueError(
                    f"Failed to parse GT JSONL at {path}:{lineno}: {e}"
                ) from e
    return out
