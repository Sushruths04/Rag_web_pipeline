"""Per-document yield controller for V16.2.

Enforces:
  - typed_mh quota: uncapped (it's the target category; attempt count is tracked)
  - untyped_mh accept cap: ≤ untyped_mh_cap_fraction of strict_target
  - fallback (fb) accept cap:  ≤ fb_cap_fraction of strict_target
  - attempt caps per category: fb and untyped_mh have hard attempt limits
  - halt when strict_target met AND typed_mh_share ≥ mh_floor_fraction
  - loud underyield flag when floor unmet at hard-stop

See docs/V16_2_FILTER_PLAN_20260519.md §4.6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

ChainCategory = Literal["typed_mh", "untyped_mh", "fb"]


@dataclass
class YieldControllerConfig:
    enabled: bool = True
    mh_floor_fraction: float = 0.60
    untyped_mh_cap_fraction: float = 0.10
    fb_cap_fraction: float = 0.40
    fb_attempt_multiplier: int = 3
    untyped_mh_attempt_multiplier: int = 3
    stop_when_target_met: bool = True
    emit_underyield_warning: bool = True

    @classmethod
    def from_dict(cls, raw: dict | None) -> "YieldControllerConfig":
        raw = raw or {}
        return cls(
            enabled=bool(raw.get("enabled", True)),
            mh_floor_fraction=float(raw.get("mh_floor_fraction", 0.60)),
            untyped_mh_cap_fraction=float(raw.get("untyped_mh_cap_fraction", 0.10)),
            fb_cap_fraction=float(raw.get("fb_cap_fraction", 0.40)),
            fb_attempt_multiplier=int(raw.get("fb_attempt_multiplier", 3)),
            untyped_mh_attempt_multiplier=int(raw.get("untyped_mh_attempt_multiplier", 3)),
            stop_when_target_met=bool(raw.get("stop_when_target_met", True)),
            emit_underyield_warning=bool(raw.get("emit_underyield_warning", True)),
        )


class _CategoryState:
    __slots__ = ("attempts", "accepted", "attempt_cap", "accept_cap")

    def __init__(self, attempt_cap: int, accept_cap: int) -> None:
        self.attempts = 0
        self.accepted = 0
        self.attempt_cap = attempt_cap  # 0 = uncapped
        self.accept_cap = accept_cap    # 0 = uncapped


class YieldController:
    """Per-document yield state machine.

    Typical usage::

        ctrl = YieldController(cfg=cfg, strict_target=budget.strict_target)
        for chain in ranked_chains:
            if ctrl.is_halted() or ctrl.is_hard_stopped(budget.attempt_cap):
                break
            if not ctrl.can_attempt(chain.category):
                continue
            ctrl.record_attempt(chain.category)
            # ... generate question, run QA-NLI, ARM ...
            if row_accepted:
                ctrl.record_accepted(chain.category)
    """

    def __init__(self, cfg: YieldControllerConfig, strict_target: int) -> None:
        self.cfg = cfg
        self.strict_target = max(1, strict_target)
        self._total_attempts = 0
        self._total_accepted = 0

        fb_accept_cap = math.ceil(strict_target * cfg.fb_cap_fraction)
        untyped_accept_cap = math.ceil(strict_target * cfg.untyped_mh_cap_fraction)

        self._states: dict[str, _CategoryState] = {
            "typed_mh": _CategoryState(attempt_cap=0, accept_cap=0),
            "untyped_mh": _CategoryState(
                attempt_cap=untyped_accept_cap * cfg.untyped_mh_attempt_multiplier,
                accept_cap=untyped_accept_cap,
            ),
            "fb": _CategoryState(
                attempt_cap=fb_accept_cap * cfg.fb_attempt_multiplier,
                accept_cap=fb_accept_cap,
            ),
        }

    # ---- public API ----

    def can_attempt(self, category: ChainCategory) -> bool:
        """True if this category has remaining attempt AND accept budget."""
        if not self.cfg.enabled:
            return True
        if not self.can_accept(category):
            return False
        st = self._states[category]
        if st.attempt_cap > 0 and st.attempts >= st.attempt_cap:
            return False
        return True

    def can_accept(self, category: ChainCategory) -> bool:
        """True if this category has remaining accept budget."""
        if not self.cfg.enabled:
            return True
        st = self._states[category]
        if st.accept_cap > 0 and st.accepted >= st.accept_cap:
            return False
        return True

    def record_attempt(self, category: ChainCategory) -> None:
        self._states[category].attempts += 1
        self._total_attempts += 1

    def record_accepted(self, category: ChainCategory) -> None:
        self._states[category].accepted += 1
        self._total_accepted += 1

    def is_halted(self) -> bool:
        """Target met AND typed_mh floor satisfied → stop cleanly."""
        if not self.cfg.enabled or not self.cfg.stop_when_target_met:
            return False
        if self._total_accepted < self.strict_target:
            return False
        return self._typed_mh_share() >= self.cfg.mh_floor_fraction

    def is_hard_stopped(self, attempt_cap: int) -> bool:
        """True when the DocBudget global attempt cap is exhausted."""
        return self._total_attempts >= attempt_cap

    @property
    def total_accepted(self) -> int:
        return self._total_accepted

    @property
    def total_attempts(self) -> int:
        return self._total_attempts

    def _typed_mh_share(self) -> float:
        if self._total_accepted == 0:
            return 0.0
        return self._states["typed_mh"].accepted / self._total_accepted

    def underyield(self) -> bool:
        """True if the multi-hop floor was not met when the run ended."""
        if self._total_accepted == 0:
            return self.strict_target > 0
        return self._typed_mh_share() < self.cfg.mh_floor_fraction

    def underyield_reason(self) -> str:
        if not self.underyield():
            return ""
        typed = self._states["typed_mh"].accepted
        total = self._total_accepted
        return (
            f"typed_mh={typed}/{total} ({self._typed_mh_share():.0%}) "
            f"< floor={self.cfg.mh_floor_fraction:.0%}"
        )

    def get_summary(self) -> dict:
        """Return dict for build_summary.json:cascade_stats.yield."""
        total = self._total_accepted
        st = self._states
        return {
            "attempts": self._total_attempts,
            "typed_mh": {
                "attempts": st["typed_mh"].attempts,
                "accepted": st["typed_mh"].accepted,
            },
            "untyped_mh": {
                "attempts": st["untyped_mh"].attempts,
                "accepted": st["untyped_mh"].accepted,
            },
            "fb": {
                "attempts": st["fb"].attempts,
                "accepted": st["fb"].accepted,
            },
            "typed_mh_share": self._typed_mh_share(),
            "untyped_mh_share": (st["untyped_mh"].accepted / total) if total else 0.0,
            "fb_share": (st["fb"].accepted / total) if total else 0.0,
            "strict_total": total,
            "strict_target": self.strict_target,
            "underyield": self.underyield(),
            "underyield_reason": self.underyield_reason(),
        }
