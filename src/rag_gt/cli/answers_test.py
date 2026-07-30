"""Smoke-test answer generator.

By default this does NOT produce real answers — it requires `--use_gold` so
that the resulting `answer_logs.jsonl` has meaningful content. The previous
default emitted a literal placeholder string that flowed into evaluation
metrics and corrupted them silently.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate answer logs (gold-as-prediction smoke test)"
    )
    p.add_argument("--gt", required=True, help="GT JSONL file")
    p.add_argument(
        "--output", default="answer_logs.jsonl", help="Output answer log file"
    )
    p.add_argument(
        "--use_gold",
        action="store_true",
        help="Use gold answers as predicted (smoke test).",
    )
    p.add_argument(
        "--abstain_only",
        action="store_true",
        help=(
            "Emit abstention rows instead of running a RAG system. Useful to "
            "establish a no-answer baseline."
        ),
    )
    args = p.parse_args()

    if not args.use_gold and not args.abstain_only:
        sys.stderr.write(
            "answers_test refuses to write placeholder predictions.\n"
            "Pass --use_gold to copy gold answers into the predicted field, "
            "or --abstain_only to emit empty abstention rows.\n"
        )
        sys.exit(2)

    with open(args.gt, "r", encoding="utf-8") as f:
        gt_questions = [json.loads(line) for line in f if line.strip()]

    results = []
    for q in gt_questions:
        if args.use_gold:
            predicted = q["gold_answer"]
            abstained = False
        else:
            predicted = ""
            abstained = True
        results.append(
            {
                "q_id": q["q_id"],
                "predicted_answer": predicted,
                "abstained": abstained,
            }
        )

    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Written {len(results)} answer logs to {args.output}")


if __name__ == "__main__":
    main()
