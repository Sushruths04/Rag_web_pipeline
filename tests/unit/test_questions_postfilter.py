from rag_gt.generation.questions import _is_stitched


def test_detects_comma_and_stitching():
    assert _is_stitched("What is X, and how does Y work?") is True


def test_detects_two_question_marks():
    assert _is_stitched("What is X? What is Y?") is True


def test_accepts_clean_bridging_question():
    q = "Under what condition must a welder certified under ISO 9606-1 retake the test?"
    assert _is_stitched(q) is False


def test_detects_semicolon_branch():
    assert _is_stitched("What does A say; what does B imply?") is True
