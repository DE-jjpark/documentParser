"""Typed exceptions raised by the public API.

Consumers can catch DocumentParserError to handle everything, or the specific
subclasses to map onto their own error responses.
"""


class DocumentParserError(Exception):
    """Base class for all errors raised by this library."""


class UnsupportedFormatError(DocumentParserError):
    """The document format is unknown or has no registered loader."""


class MissingDependencyError(DocumentParserError):
    """The format is supported but its optional extra is not installed."""


class ParsingFailedError(DocumentParserError):
    """The parsing engine failed to produce a ParsedDocument."""


class ChunkingFailedError(DocumentParserError):
    """The chunking engine failed to produce chunks."""
