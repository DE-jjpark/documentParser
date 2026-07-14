"""Shared contract layer: the only module both engines may depend on."""

from document_parser.core.exceptions import (
    ChunkingFailedError,
    DocumentParserError,
    MissingDependencyError,
    ParsingFailedError,
    UnsupportedFormatError,
)
from document_parser.core.models import (
    Chunk,
    ChunkingConfig,
    DocumentElement,
    ElementType,
    ParsedDocument,
    Segment,
)

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "ChunkingFailedError",
    "DocumentElement",
    "DocumentParserError",
    "ElementType",
    "MissingDependencyError",
    "ParsedDocument",
    "ParsingFailedError",
    "Segment",
    "UnsupportedFormatError",
]
