"""Public facade of the chunking engine."""

from document_parser.chunking.graph import build_chunking_graph
from document_parser.chunking.state import ChunkingState
from document_parser.core.exceptions import ChunkingFailedError, DocumentParserError
from document_parser.core.models import Chunk, ChunkingConfig, Segment


class ChunkingEngine:
    """Splits Segments into Chunks.

    The graph is compiled once at construction; instances are stateless and
    safe to share across concurrent requests.
    """

    def __init__(self) -> None:
        self._graph = build_chunking_graph().compile()

    async def achunk(
        self,
        segments: str | Segment | list[Segment],
        config: ChunkingConfig | None = None,
    ) -> list[Chunk]:
        """Chunk the given input. Preferred entrypoint for async consumers."""
        state = self._initial_state(segments, config)
        try:
            result = await self._graph.ainvoke(state)
        except DocumentParserError:
            raise
        except Exception as exc:
            raise ChunkingFailedError(f"chunking failed: {exc}") from exc
        return result["chunks"]

    def chunk(
        self,
        segments: str | Segment | list[Segment],
        config: ChunkingConfig | None = None,
    ) -> list[Chunk]:
        """Synchronous convenience wrapper around the same graph."""
        state = self._initial_state(segments, config)
        try:
            result = self._graph.invoke(state)
        except DocumentParserError:
            raise
        except Exception as exc:
            raise ChunkingFailedError(f"chunking failed: {exc}") from exc
        return result["chunks"]

    @staticmethod
    def _initial_state(
        segments: str | Segment | list[Segment], config: ChunkingConfig | None
    ) -> ChunkingState:
        if isinstance(segments, str):
            segments = [Segment(text=segments)]
        elif isinstance(segments, Segment):
            segments = [segments]
        return {"segments": segments, "config": config or ChunkingConfig()}
