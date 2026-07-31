from rag_gt.allpdf.bridge_quality import JUNK_SURFACES, surface_ok

# The 9 junk bridge surfaces shipped in the round-2 dataset (F1). Each must
# either be rejected outright, or repaired (OCR-split fix) and re-tested --
# never silently shipped as-is.
SHIPPED_JUNK = [
    "backed up",
    "di erent",
    "less than",
    "less than 5",
    "n step",
    "nite number",
    "only in the limit",
    "some number",
    "than 0",
]

# Real domain concepts that must always pass untouched.
GOOD_SURFACES = [
    "Vickers microhardness",
    "ISO 15607",
    "Periodic verification",
    "JSON syntax",
    "change request",
    "object flows",
]


def test_shipped_junk_surfaces_reject_or_repair():
    for surface in SHIPPED_JUNK:
        ok, detail = surface_ok(surface)
        # Either rejected outright, or accepted only because it was repaired
        # into a different (non-junk) surface -- never accepted verbatim.
        assert (not ok) or (detail.strip().lower() != surface.strip().lower()), (
            f"{surface!r} must reject or repair, got ok={ok} detail={detail!r}"
        )


def test_shipped_junk_surfaces_all_seeded_in_module_list():
    normalized = {s.strip().lower() for s in SHIPPED_JUNK}
    assert normalized <= {s.strip().lower() for s in JUNK_SURFACES}


def test_good_surfaces_accept_unchanged():
    for surface in GOOD_SURFACES:
        ok, detail = surface_ok(surface)
        assert ok is True, f"{surface!r} should be accepted, got detail={detail!r}"
        assert detail == surface


def test_di_erent_repairs_to_different():
    ok, detail = surface_ok("di erent")
    assert ok is True
    assert detail == "different"


def test_nite_number_repairs_to_finite_number():
    ok, detail = surface_ok("nite number")
    assert ok is True
    assert detail == "finite number"


def test_exact_junk_surfaces_without_repair_pattern_are_rejected():
    for surface in ("backed up", "less than", "less than 5", "n step",
                     "only in the limit", "some number", "than 0"):
        ok, reason = surface_ok(surface)
        assert ok is False
        assert reason


def test_too_short_surface_rejected():
    ok, reason = surface_ok("abc")
    assert ok is False
    assert reason == "too_short"


def test_no_content_token_rejected():
    ok, reason = surface_ok("it is")
    assert ok is False


def test_empty_surface_rejected():
    ok, reason = surface_ok("")
    assert ok is False


def test_whitespace_and_case_are_normalized_for_junk_match():
    ok, reason = surface_ok("  LESS   THAN  ")
    assert ok is False
