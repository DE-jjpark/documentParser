"""document-parser: document parsing and chunking engines.

Everything re-exported here is the public, supported API. Anything imported
from submodules directly is internal and may change without notice.
"""

from document_parser.chunking import ChunkingEngine
from document_parser.core.exceptions import (
    ChunkingFailedError,
    DocumentParserError,
    MissingDependencyError,
    ParsingFailedError,
    UnsupportedFormatError,
)
from document_parser.core.models import (
    BBox,
    Chunk,
    ChunkingConfig,
    DocumentElement,
    ElementType,
    ParsedDocument,
    ParsingTier,
    Segment,
)
from document_parser.parsing import ParsingEngine
from document_parser.pipeline import IngestPipeline

__version__ = "0.1.0"

__all__ = [
    "BBox",
    "Chunk",
    "ChunkingConfig",
    "ChunkingEngine",
    "ChunkingFailedError",
    "DocumentElement",
    "DocumentParserError",
    "ElementType",
    "IngestPipeline",
    "MissingDependencyError",
    "ParsedDocument",
    "ParsingEngine",
    "ParsingFailedError",
    "ParsingTier",
    "Segment",
    "UnsupportedFormatError",
    "__version__",
]
