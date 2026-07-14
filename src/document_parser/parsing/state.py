"""LangGraph state for the parsing graph. Internal to the parsing engine."""

from typing import TypedDict

from document_parser.core.models import DocumentElement, ParsedDocument


class ParsingState(TypedDict, total=False):
    source: str
    data: bytes
    format: str
    elements: list[DocumentElement]
    document: ParsedDocument
