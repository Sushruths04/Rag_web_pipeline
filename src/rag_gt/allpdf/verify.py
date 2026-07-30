"""Per-stage verification harness.

Every pipeline stage runs a VerificationGate: a named set of boolean checks with
human-readable detail. The gate prints a report and records pass/fail so the
autonomous orchestrator can stop-and-inspect a failing stage before proceeding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class VerificationGate:
    stage: str
    checks: List[Check] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append(Check(name, bool(passed), detail))

    def expect(self, name: str, fn: Callable[[], bool], detail: str = "") -> None:
        try:
            ok = bool(fn())
        except Exception as e:  # noqa: BLE001
            ok = False
            detail = f"{detail} (raised {type(e).__name__}: {e})".strip()
        self.check(name, ok, detail)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def report(self) -> str:
        lines = [f"=== GATE [{self.stage}] {'PASS' if self.passed else 'FAIL'} ==="]
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}" + (f" — {c.detail}" if c.detail else ""))
        if self.metrics:
            lines.append("  metrics: " + json.dumps(self.metrics, default=str))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "checks": [vars(c) for c in self.checks],
            "metrics": self.metrics,
        }


def save_checkpoint(obj: Any, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return str(p)
