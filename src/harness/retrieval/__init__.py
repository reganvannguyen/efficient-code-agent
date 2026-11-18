"""Swappable repository retrieval backends and candidate filtering."""

from harness.retrieval.backend import (
    CodeAwareRetrievalBackend,
    LexicalRetrievalBackend,
    NormalizedRetrievalMatch,
    RetrievalBackend,
    resolve_retrieval_backend,
)
from harness.retrieval.filtering import (
    BINARY_EXTENSIONS,
    EXCLUDED_DIRS,
    GENERATED_FILENAMES,
    GENERATED_SUFFIXES,
    MAX_CANDIDATE_FILE_BYTES,
    classify_candidate,
)

__all__ = [
    "BINARY_EXTENSIONS",
    "CodeAwareRetrievalBackend",
    "EXCLUDED_DIRS",
    "GENERATED_FILENAMES",
    "GENERATED_SUFFIXES",
    "LexicalRetrievalBackend",
    "MAX_CANDIDATE_FILE_BYTES",
    "NormalizedRetrievalMatch",
    "RetrievalBackend",
    "classify_candidate",
    "resolve_retrieval_backend",
]
