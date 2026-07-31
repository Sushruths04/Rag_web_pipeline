"""Tests for V16.2 per-document yield controller."""

from __future__ import annotations

import math

import pytest

from rag_gt.pipeline.yield_controller import YieldController, YieldControllerConfig


def _cfg(**overrides) -> YieldControllerConfig:
    defaults = {
        "enabled": True,
        "mh_floor_fraction": 0.60,
        "untyped_mh_cap_fraction": 0.10,
        "fb_cap_fraction": 0.40,
        "fb_attempt_multiplier": 3,
        "untyped_mh_attempt_multiplier": 3,
        "stop_when_target_met": True,
        "emit_underyield_warning": True,
    }
    defaults.update(overrides)
    return YieldControllerConfig(**defaults)


# ---------------------------------------------------------------------------
# Accept cap computation from fractions
# ---------------------------------------------------------------------------

def test_fb_accept_cap_correct() -> None:
    cfg = _cfg(fb_cap_fraction=0.40)
    ctrl = YieldController(cfg=cfg, strict_target=20)
    fb_accept_cap = math.ceil(20 * 0.40)  # 8
    # Exhaust the cap.
    for _ in range(fb_accept_cap):
        ctrl.record_attempt("fb")
        ctrl.record_accepted("fb")
    assert not ctrl.can_attempt("fb")


def test_untyped_mh_accept_cap_correct() -> None:
    cfg = _cfg(untyped_mh_cap_fraction=0.10)
    ctrl = YieldController(cfg=cfg, strict_target=20)
    cap = math.ceil(20 * 0.10)  # 2
    for _ in range(cap):
        ctrl.record_attempt("untyped_mh")
        ctrl.record_accepted("untyped_mh")
    assert not ctrl.can_attempt("untyped_mh")
    assert not ctrl.can_accept("untyped_mh")


def test_typed_mh_has_no_accept_cap() -> None:
    cfg = _cfg()
    ctrl = YieldController(cfg=cfg, strict_target=10)
    # Accept many typed_mh rows — should never hit a cap.
    for _ in range(100):
        ctrl.record_attempt("typed_mh")
        ctrl.record_accepted("typed_mh")
    assert ctrl.can_attempt("typed_mh")
    assert ctrl.can_accept("typed_mh")


# ---------------------------------------------------------------------------
# Attempt caps
# ---------------------------------------------------------------------------

def test_fb_attempt_cap_blocks_further_attempts() -> None:
    cfg = _cfg(fb_cap_fraction=0.40, fb_attempt_multiplier=3)
    ctrl = YieldController(cfg=cfg, strict_target=10)
    fb_accept_cap = math.ceil(10 * 0.40)  # 4
    fb_attempt_cap = fb_accept_cap * 3    # 12
    for _ in range(fb_attempt_cap):
        ctrl.record_attempt("fb")
    assert not ctrl.can_attempt("fb")


def test_untyped_mh_attempt_cap() -> None:
    cfg = _cfg(untyped_mh_cap_fraction=0.10, untyped_mh_attempt_multiplier=3)
    ctrl = YieldController(cfg=cfg, strict_target=20)
    cap = math.ceil(20 * 0.10) * 3  # 6
    for _ in range(cap):
        ctrl.record_attempt("untyped_mh")
    assert not ctrl.can_attempt("untyped_mh")


# ---------------------------------------------------------------------------
# Halt rule
# ---------------------------------------------------------------------------

def test_halted_when_target_met_and_floor_satisfied() -> None:
    cfg = _cfg(mh_floor_fraction=0.60, stop_when_target_met=True)
    ctrl = YieldController(cfg=cfg, strict_target=10)
    # Accept 6 typed_mh + 4 fb = 10 total, typed_mh_share = 0.60
    for _ in range(6):
        ctrl.record_attempt("typed_mh")
        ctrl.record_accepted("typed_mh")
    for _ in range(4):
        ctrl.record_attempt("fb")
        ctrl.record_accepted("fb")
    assert ctrl.is_halted()


def test_not_halted_if_floor_unmet() -> None:
    cfg = _cfg(mh_floor_fraction=0.60)
    ctrl = YieldController(cfg=cfg, strict_target=10)
    # Accept 3 typed_mh + 7 fb → typed_mh_share = 0.30 < 0.60
    for _ in range(3):
        ctrl.record_attempt("typed_mh")
        ctrl.record_accepted("typed_mh")
    for _ in range(7):
        ctrl.record_attempt("fb")
        ctrl.record_accepted("fb")
    assert not ctrl.is_halted()


def test_not_halted_if_target_unmet() -> None:
    cfg = _cfg(mh_floor_fraction=0.60)
    ctrl = YieldController(cfg=cfg, strict_target=10)
    for _ in range(6):
        ctrl.record_attempt("typed_mh")
        ctrl.record_accepted("typed_mh")
    # Only 6 accepted, target is 10.
    assert not ctrl.is_halted()


def test_halt_disabled_by_config() -> None:
    cfg = _cfg(stop_when_target_met=False)
    ctrl = YieldController(cfg=cfg, strict_target=5)
    for _ in range(5):
        ctrl.record_attempt("typed_mh")
        ctrl.record_accepted("typed_mh")
    assert not ctrl.is_halted()


# ---------------------------------------------------------------------------
# Hard stop (global attempt cap from DocBudget)
# ---------------------------------------------------------------------------

def test_is_hard_stopped() -> None:
    cfg = _cfg()
    ctrl = YieldController(cfg=cfg, strict_target=10)
    attempt_cap = 40
    for _ in range(attempt_cap):
        ctrl.record_attempt("typed_mh")
    assert ctrl.is_hard_stopped(attempt_cap)
    assert not ctrl.is_hard_stopped(attempt_cap + 1)


# ---------------------------------------------------------------------------
# Underyield detection
# ---------------------------------------------------------------------------

def test_underyield_fires_when_floor_unmet() -> None:
    cfg = _cfg(mh_floor_fraction=0.60)
    ctrl = YieldController(cfg=cfg, strict_target=10)
    for _ in range(2):
        ctrl.record_attempt("typed_mh")
        ctrl.record_accepted("typed_mh")
    for _ in range(8):
        ctrl.record_attempt("fb")
        ctrl.record_accepted("fb")
    assert ctrl.underyield()
    assert "typed_mh=" in ctrl.underyield_reason()


def test_no_underyield_when_floor_met() -> None:
    cfg = _cfg(mh_floor_fraction=0.60)
    ctrl = YieldController(cfg=cfg, strict_target=10)
    for _ in range(6):
        ctrl.record_attempt("typed_mh")
        ctrl.record_accepted("typed_mh")
    for _ in range(4):
        ctrl.record_attempt("fb")
        ctrl.record_accepted("fb")
    assert not ctrl.underyield()
    assert ctrl.underyield_reason() == ""


def test_underyield_true_when_zero_accepted_but_target_positive() -> None:
    cfg = _cfg()
    ctrl = YieldController(cfg=cfg, strict_target=5)
    assert ctrl.underyield()


# ---------------------------------------------------------------------------
# Summary dict
# ---------------------------------------------------------------------------

def test_summary_contains_required_keys() -> None:
    cfg = _cfg()
    ctrl = YieldController(cfg=cfg, strict_target=10)
    ctrl.record_attempt("typed_mh")
    ctrl.record_accepted("typed_mh")
    ctrl.record_attempt("fb")
    summary = ctrl.get_summary()
    for key in (
        "attempts", "typed_mh", "untyped_mh", "fb",
        "typed_mh_share", "untyped_mh_share", "fb_share",
        "strict_total", "strict_target", "underyield", "underyield_reason",
    ):
        assert key in summary, f"missing key: {key}"


def test_summary_shares_sum_to_one_approximately() -> None:
    cfg = _cfg()
    ctrl = YieldController(cfg=cfg, strict_target=10)
    for _ in range(6):
        ctrl.record_attempt("typed_mh")
        ctrl.record_accepted("typed_mh")
    for _ in range(2):
        ctrl.record_attempt("untyped_mh")
        ctrl.record_accepted("untyped_mh")
    for _ in range(2):
        ctrl.record_attempt("fb")
        ctrl.record_accepted("fb")
    s = ctrl.get_summary()
    total = s["typed_mh_share"] + s["untyped_mh_share"] + s["fb_share"]
    assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Disabled mode
# ---------------------------------------------------------------------------

def test_disabled_cfg_can_attempt_everything() -> None:
    cfg = _cfg(enabled=False)
    ctrl = YieldController(cfg=cfg, strict_target=5)
    for _ in range(1000):
        ctrl.record_attempt("fb")
        ctrl.record_accepted("fb")
    assert ctrl.can_attempt("fb")


def test_config_from_dict() -> None:
    raw = {
        "enabled": True,
        "mh_floor_fraction": 0.70,
        "untyped_mh_cap_fraction": 0.05,
        "fb_cap_fraction": 0.30,
        "fb_attempt_multiplier": 4,
        "untyped_mh_attempt_multiplier": 4,
        "stop_when_target_met": False,
        "emit_underyield_warning": False,
    }
    cfg = YieldControllerConfig.from_dict(raw)
    assert cfg.mh_floor_fraction == pytest.approx(0.70)
    assert cfg.fb_cap_fraction == pytest.approx(0.30)
    assert cfg.fb_attempt_multiplier == 4
    assert cfg.stop_when_target_met is False
