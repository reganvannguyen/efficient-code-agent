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
from harness.retrieval.python_symbols import (
    PYTHON_EXTENSIONS,
    PythonParseResult,
    PythonSymbol,
    extract_python_symbols_from_bytes,
    parse_python_file,
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
    "PYTHON_EXTENSIONS",
    "PythonParseResult",
    "PythonSymbol",
    "RetrievalBackend",
    "classify_candidate",
    "extract_python_symbols_from_bytes",
    "parse_python_file",
    "resolve_retrieval_backend",
]
