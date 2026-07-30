"""`python -m rag_gt.cli.score_audit` -- summarise a filled fact-audit CSV.

Reads the CSV produced by `cli/sample_facts_for_audit.py` after the auditor
has filled `well_formed?` and `span_correct?` (1 = pass, 0 = fail). Emits a
markdown summary suitable for inclusion in the paper.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import List


def _to_int(v: str) -> int | None:
    v = (v or "").strip()
    if v in ("1", "true", "True", "yes", "y"):
        return 1
    if v in ("0", "false", "False", "no", "n"):
        return 0
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        prog="rag-gt-score-audit",
        description="Summarise a filled fact-audit CSV.",
    )
    p.add_argument("--audit", default="data/eval_results/fact_audit/audit.csv")
    p.add_argument("--output", default="data/eval_results/fact_audit/summary.md")
    args = p.parse_args()

    with open(args.audit, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    n = len(rows)
    well_formed = [r for r in rows if _to_int(r.get("well_formed?")) is not None]
    span_correct = [r for r in rows if _to_int(r.get("span_correct?")) is not None]

    n_filled_wf = len(well_formed)
    n_filled_sc = len(span_correct)
    n_pass_wf = sum(_to_int(r.get("well_formed?")) or 0 for r in well_formed)
    n_pass_sc = sum(_to_int(r.get("span_correct?")) or 0 for r in span_correct)

    both = [
        r for r in rows
        if _to_int(r.get("well_formed?")) == 1
        and _to_int(r.get("span_correct?")) == 1
    ]

    role_counts: Counter[str] = Counter(r.get("role", "") for r in rows)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Fact corpus audit")
    lines.append("")
    lines.append(f"- Sampled facts: **{n}**")
    lines.append(f"- Rows scored on `well_formed?`: {n_filled_wf}")
    lines.append(f"- Rows scored on `span_correct?`: {n_filled_sc}")
    if n_filled_wf == 0 and n_filled_sc == 0:
        lines.append("")
        lines.append("**No rows have been audited yet.** Open the CSV, fill the "
                     "`well_formed?` and `span_correct?` columns with 1 or 0.")
    else:
        lines.append("")
        lines.append("## Quality")
        lines.append("")
        lines.append("| Dimension | Pass | Out of | Pass rate |")
        lines.append("|---|---:|---:|---:|")
        if n_filled_wf:
            lines.append(
                f"| Well-formed proposition | {n_pass_wf} | {n_filled_wf} | "
                f"{n_pass_wf / n_filled_wf:.1%} |"
            )
        if n_filled_sc:
            lines.append(
                f"| Supporting span correct | {n_pass_sc} | {n_filled_sc} | "
                f"{n_pass_sc / n_filled_sc:.1%} |"
            )
        if n_filled_wf and n_filled_sc:
            joint_denom = sum(
                1 for r in rows
                if _to_int(r.get("well_formed?")) is not None
                and _to_int(r.get("span_correct?")) is not None
            )
            lines.append(
                f"| Both | {len(both)} | {joint_denom} | "
                f"{(len(both) / joint_denom) if joint_denom else 0:.1%} |"
            )

    lines.append("")
    lines.append("## Sampled fact role distribution")
    lines.append("")
    lines.append("| role | count |")
    lines.append("|---|---:|")
    for role, count in role_counts.most_common():
        lines.append(f"| `{role}` | {count} |")

    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"[score_audit] wrote {args.output}")


if __name__ == "__main__":
    main()
