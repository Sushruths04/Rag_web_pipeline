"""Fact extraction: roles + quality scoring + clause splitting."""

from rag_gt.facts.extraction import extract_candidate_facts
from rag_gt.facts.quality import structural_quality
from rag_gt.facts.roles import detect_role

__all__ = ["detect_role", "extract_candidate_facts", "structural_quality"]
