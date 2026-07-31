from pathlib import Path

from rag_gt.profiling.folder_heuristic import infer_from_path


def test_iso_folder():
    assert infer_from_path(Path("inputs/iso/DIN_EN_ISO_9606.pdf")) == "ISO_STANDARD"


def test_iso_prefix_without_folder():
    assert infer_from_path(Path("workspace/DIN EN ISO 4136-ENG.pdf")) == "ISO_STANDARD"


def test_reports_folder():
    assert infer_from_path(Path("corpus/reports/Q3_2025.pdf")) == "REPORT"


def test_kt_prefix():
    assert infer_from_path(Path("anywhere/KT_team_handover.pdf")) == "KT_DOC"


def test_narrative_folder():
    assert infer_from_path(Path("library/fiction/chapter1.pdf")) == "NARRATIVE"


def test_ambiguous_returns_none():
    assert infer_from_path(Path("random.pdf")) is None
    assert infer_from_path(Path("corpus/misc/unlabeled.pdf")) is None


def test_case_insensitive_prefix():
    assert infer_from_path(Path("din en iso 3452-1.pdf")) == "ISO_STANDARD"
