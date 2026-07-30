"""`python -m rag_gt.cli.audit_gt_quality` — score every GT question on
quality and emit a triage report.

The motivating bug: questions like "Why is *this distribution* considered a
better choice than the uniform distribution?" have no antecedent for "this
distribution" — the question is not self-contained, and the gold answer is
similarly stripped of context. RAGAS and the SUT can both regurgitate the
gold answer because neither has to ground the referent. RAG_GT cannot tell
the question is bad because the metrics score retrieval, not GT writing.

Heuristics applied (each yields a 0/1 flag and weight):
  Q1 — deictic with no antecedent: "this/that/these/those/it/they" inside
       the question without a clear noun phrase preceding it
  Q2 — answer too short: < 40 chars total — likely "Yes" / "No" / numeric only
  Q3 — answer ≈ question: cosine similarity > 0.95 between Q and A (asking
       and answering with the same words)
  Q4 — generic referent: "the result", "the figure", "the table",
       "the example" without naming which one
  Q5 — multiple unrelated facts: required_facts come from > 1 doc OR span
       > 5 chunks apart (genuine multi-hop is rare; usually a packing artefact)
  Q6 — answer mentions a number not in any gold fact: hallucinated
       quantitative content (only flagged if facts are present)

Scoring:
  quality = 1 - (sum(weights * flags) / total_weight)
  Anything below `--min-quality` (default 0.5) goes into the "bad" bucket.

Output:
  data/eval_results/gt_audit/<corpus>/quality.json   — per-question scores
  data/eval_results/gt_audit/<corpus>/quality.md     — triage report
  data/eval_results/gt_audit/<corpus>/questions_review.md
                                                       — full Q/A/fact review
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import List

from rag_gt.validation.gt_quality import CHECKS, score_question


def _render_md(corpus: str, audited: List[dict], min_q: float) -> str:
    bad = [a for a in audited if a["quality"] < min_q]
    flag_counts = Counter()
    for a in audited:
        for k, v in a["flags"].items():
            if v:
                flag_counts[k] += 1
    qs = [a["quality"] for a in audited]
    parts = [
        f"# GT quality audit — {corpus}\n",
        f"- n questions: **{len(audited)}**",
        f"- mean quality: **{sum(qs)/len(qs):.3f}**",
        f"- median quality: **{sorted(qs)[len(qs)//2]:.3f}**",
        f"- below threshold ({min_q}): **{len(bad)} ({100*len(bad)/len(audited):.1f}%)**\n",
        "## Flag frequency",
        "| flag | count | % |",
        "|---|---:|---:|",
    ]
    for check in CHECKS:
        c = flag_counts.get(check.name, 0)
        parts.append(
            f"| {check.name} (w={check.weight}) | {c} | {100*c/len(audited):.1f}% |"
        )
    parts.append("\n## Worst 20 questions (lowest quality)\n")
    parts.append("| q_id | quality | flags |")
    parts.append("|---|---:|---|")
    for a in sorted(audited, key=lambda x: x["quality"])[:20]:
        flagged = [k for k, v in a["flags"].items() if v]
        parts.append(f"| `{a['q_id']}` | {a['quality']:.3f} | {', '.join(flagged) or '—'} |")
    parts.append("\n## What to do with bad questions")
    parts.append("- Drop them, or rewrite them to include the missing antecedent.")
    parts.append("- For Q1/Q4 violations: prepend the missing context.")
    parts.append("- For Q2 (too short): demand a fuller gold answer from the GT generator.")
    parts.append("- For Q5 (scattered): split into separate questions or accept higher tolerance.")
    parts.append("- For Q6 (unsupported number): the GT or the generator hallucinated; verify.")
    return "\n".join(parts) + "\n"


def _one_line(text: str) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text or "")
    return " ".join(cleaned.split())


def _render_questions_review(
    corpus: str,
    rows: List[dict],
    audited: List[dict],
    min_q: float,
) -> str:
    audit_by_id = {a["q_id"]: a for a in audited}
    qs = [a["quality"] for a in audited]
    bad = [a for a in audited if a["quality"] < min_q]
    parts = [
        f"# Regenerated GT Review - {corpus}\n",
        f"- Questions: **{len(rows)}**",
        f"- Mean quality: **{sum(qs)/len(qs):.3f}**",
        f"- Below threshold {min_q}: **{len(bad)}**",
        "\n## Questions",
    ]

    for idx, row in enumerate(rows, 1):
        audit = audit_by_id.get(row.get("q_id", ""), {})
        flags = [k for k, v in audit.get("flags", {}).items() if v]
        parts.extend(
            [
                f"\n### {idx}. {row.get('q_id', '<missing q_id>')}\n",
                f"**Quality:** {audit.get('quality', 0):.3f}  ",
                (
                    f"**Depth:** {row.get('difficulty_reasoning_depth', '?')} | "
                    f"**Distance:** {row.get('difficulty_semantic_distance', '?')}  "
                ),
                f"**Flags:** {', '.join(flags) if flags else 'none'}\n",
                f"**Question:** {_one_line(row.get('question', ''))}\n",
                f"**Gold answer:** {_one_line(row.get('gold_answer', ''))}\n",
                "**Required facts:**",
            ]
        )
        for fact in row.get("required_facts") or []:
            spans = fact.get("supporting_spans") or []
            chunk_id = spans[0].get("chunk_id", "<missing chunk>") if spans else "<missing chunk>"
            parts.append(
                f"- {fact.get('fact_id', '<missing fact_id>')} / {chunk_id}: "
                f"{_one_line(fact.get('text', ''))}"
            )

    return "\n".join(parts) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", required=True, help="path to GT JSONL")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-quality", type=float, default=0.5,
                   help="questions below this go into the 'bad' bucket")
    args = p.parse_args()

    with open(args.gt, encoding="utf-8") as f:
        gt = [json.loads(l) for l in f if l.strip()]
    print(f"loaded {len(gt)} questions")

    audited = [score_question(q) for q in gt]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "quality.json").write_text(json.dumps({
        "n": len(audited),
        "mean_quality": sum(a["quality"] for a in audited) / len(audited),
        "rows": audited,
    }, indent=2), encoding="utf-8")
    corpus = Path(args.gt).stem
    (out / "quality.md").write_text(_render_md(corpus, audited, args.min_quality),
                                      encoding="utf-8")
    (out / "questions_review.md").write_text(
        _render_questions_review(corpus, gt, audited, args.min_quality),
        encoding="utf-8",
    )
    print(
        f"wrote {out / 'quality.json'}, {out / 'quality.md'}, "
        f"and {out / 'questions_review.md'}"
    )
    print(f"mean quality: {sum(a['quality'] for a in audited)/len(audited):.3f}")
    print(f"below {args.min_quality}: {sum(1 for a in audited if a['quality'] < args.min_quality)} / {len(audited)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
