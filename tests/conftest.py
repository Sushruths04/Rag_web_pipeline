"""Test configuration.

Adds src/ to sys.path so tests can import rag_gt without install, and keeps
live LLM tests opt-in by default.
"""

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def pytest_addoption(parser):
    parser.addoption(
        "--run-live-llm",
        action="store_true",
        default=False,
        help="Run tests marked live_llm.",
    )


def pytest_collection_modifyitems(config, items):
    markexpr = str(getattr(config.option, "markexpr", "") or "")
    wants_live = config.getoption("--run-live-llm") or "live_llm" in markexpr
    if wants_live:
        return

    skip_live = pytest.mark.skip(
        reason="live_llm tests are opt-in; use --run-live-llm or -m live_llm"
    )
    for item in items:
        if "live_llm" in item.keywords:
            item.add_marker(skip_live)
