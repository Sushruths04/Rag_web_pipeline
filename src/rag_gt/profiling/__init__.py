"""Document profiling: Layer-0 folder heuristic + regex classifier."""

from rag_gt.profiling.folder_heuristic import infer_from_path
from rag_gt.profiling.profiler import profile_document

__all__ = ["infer_from_path", "profile_document"]
