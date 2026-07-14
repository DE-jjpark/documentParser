"""LangGraph state for the chunking graph. Internal to the chunking engine."""

from typing import TypedDict

from document_parser.core.models import Chunk, ChunkingConfig, Segment


class ChunkingState(TypedDict, total=False):
    segments: list[Segment]
    config: ChunkingConfig
    pieces: list[Segment]
    chunks: list[Chunk]
