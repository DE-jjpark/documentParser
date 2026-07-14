"""Chunking engine: Segments -> Chunks.

Public surface is ChunkingEngine only; the LangGraph graph, its state and the
split strategies are internal implementation details. This package must stay
independent of the parsing engine — it may only import from core.
"""

from document_parser.chunking.engine import ChunkingEngine

__all__ = ["ChunkingEngine"]
