"""Parsing engine: source document bytes -> ParsedDocument.

Public surface is ParsingEngine only; the LangGraph graph, its state and the
loaders are internal implementation details.
"""

from document_parser.parsing.engine import ParsingEngine

__all__ = ["ParsingEngine"]
