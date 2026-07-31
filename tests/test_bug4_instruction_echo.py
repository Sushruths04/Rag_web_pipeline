"""BUG-4: recover the real question when the reasoning model echoes the prompt.

The model sometimes writes its framing deliberation then "Our question: ...".
_strip_instruction_echo must discard the echo and keep the clean question; a
normal output must be untouched.
"""
from rag_gt.generation.questions import _strip_instruction_echo, _parse_candidates


# The real leaked output observed on din_iso_15609 (pair was filtered post-hoc).
LEAKED = (
    'what is X and how does Y...\'). Combine the two facts into a single '
    'relationship question — one clause, not a run-on." Using "and" to list '
    'aspects within one question is okay as long as it\'s one clause. Our '
    'question: "What parameters define the maximum run width for manual and '
    'partly mechanized welding and the wire speed feed range for mechanized and '
    'automatic welding?"'
)


def test_strip_recovers_clean_question():
    cleaned = _strip_instruction_echo(LEAKED)
    assert cleaned.startswith("What parameters define the maximum run width")
    assert cleaned.endswith("?")
    assert "Combine the two facts" not in cleaned
    assert "Our question" not in cleaned


def test_parse_candidates_after_strip():
    cands = _parse_candidates(_strip_instruction_echo(LEAKED))
    assert cands and cands[0].startswith("What parameters define")


def test_normal_output_untouched():
    normal = "What pre-training task does BERT use to learn bidirectional context?"
    assert _strip_instruction_echo(normal) == normal
