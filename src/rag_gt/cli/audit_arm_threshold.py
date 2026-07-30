"""Offline ARM threshold audit — re-evaluates recorded trace drops at different thresholds.

Usage:
    python -m rag_gt.cli.audit_arm_threshold --trace_path <path/to/run.trace.jsonl>

Reads arm_grounding_failed drop events and re-evaluates whether each blocked chain
would pass under max_np_overlap=0.35 vs 0.38, using recorded np_overlap/nli_score values.
Zero API calls; read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _chain_passes(steps: list[dict], max_np_overlap: float, nli_threshold: float) -> bool:
    return all(
        s["np_overlap"] <= max_np_overlap and s["nli_score"] >= nli_threshold
        for s in steps
    )


def _blocking_np_overlap(steps: list[dict]) -> float:
    for s in steps:
        if s["np_overlap"] > 0.35:
            return s["np_overlap"]
    return max(s["np_overlap"] for s in steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline ARM threshold audit")
    parser.add_argument("--trace_path", required=True, help="Path to .trace.jsonl file")
    parser.add_argument("--nli_threshold", type=float, default=0.55)
    args = parser.parse_args()

    trace_path = Path(args.trace_path)
    if not trace_path.exists():
        raise SystemExit(f"Trace file not found: {trace_path}")

    rows: list[dict] = []
    with trace_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "drop":
                continue
            if event.get("reason") != "arm_grounding_failed":
                continue

            data = event.get("data", {})
            trace = data.get("reasoning_trace", [])
            if not trace:
                continue

            chain = data.get("chain", {})
            chain_id = chain.get("chain_id", data.get("chain_id", "unknown"))
            fact_ids = chain.get("fact_ids", data.get("fact_ids", []))
            # Strip common doc-prefix so only the F-number part shows (e.g. F2023)
            short_ids = [fid.split("_")[-1] if "_" in fid else fid for fid in fact_ids]
            fact_label = "->".join(short_ids) if short_ids else "unknown"

            rows.append({
                "chain_id": chain_id,
                "fact_label": fact_label,
                "steps": trace,
            })

    if not rows:
        print("No arm_grounding_failed drop events found in trace.")
        return

    nli_thr = args.nli_threshold
    col_w = [24, 22, 16, 10, 26]
    header = (
        f"{'chain_id':<{col_w[0]}} | "
        f"{'fact_ids':<{col_w[1]}} | "
        f"{'current (0.35)':<{col_w[2]}} | "
        f"{'at 0.38':<{col_w[3]}} | "
        f"blocking step np_overlap"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        steps = row["steps"]
        pass_35 = _chain_passes(steps, 0.35, nli_thr)
        pass_38 = _chain_passes(steps, 0.38, nli_thr)
        blocking = _blocking_np_overlap(steps)
        print(
            f"{row['chain_id'][:col_w[0]]:<{col_w[0]}} | "
            f"{row['fact_label'][:col_w[1]]:<{col_w[1]}} | "
            f"{'PASS' if pass_35 else 'FAIL':<{col_w[2]}} | "
            f"{'PASS' if pass_38 else 'FAIL':<{col_w[3]}} | "
            f"{blocking:.4f}"
        )


if __name__ == "__main__":
    main()
