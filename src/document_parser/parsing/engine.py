"""Public facade of the parsing engine."""

from pathlib import Path

from document_parser.core.exceptions import DocumentParserError, ParsingFailedError
from document_parser.core.models import ParsedDocument
from document_parser.parsing.graph import build_parsing_graph
from document_parser.parsing.state import ParsingState


class ParsingEngine:
    """Parses source documents into ParsedDocument objects.

    The graph is compiled once at construction; instances are stateless and
    safe to share across concurrent requests.
    """

    def __init__(self) -> None:
        self._graph = build_parsing_graph().compile()

    async def aparse(
        self,
        source: str | Path,
        *,
        data: bytes | None = None,
        format: str | None = None,
    ) -> ParsedDocument:
        """Parse a document. Preferred entrypoint for async consumers.

        ``source`` is a file path (read from disk when ``data`` is omitted) or
        just a name used for format detection and metadata when ``data`` is
        passed directly.
        """
        state = self._initial_state(source, data, format)
        try:
            result = await self._graph.ainvoke(state)
        except DocumentParserError:
            raise
        except Exception as exc:
            raise ParsingFailedError(f"parsing failed for {source}: {exc}") from exc
        return result["document"]

    def parse(
        self,
        source: str | Path,
        *,
        data: bytes | None = None,
        format: str | None = None,
    ) -> ParsedDocument:
        """Synchronous convenience wrapper around the same graph."""
        state = self._initial_state(source, data, format)
        try:
            result = self._graph.invoke(state)
        except DocumentParserError:
            raise
        except Exception as exc:
            raise ParsingFailedError(f"parsing failed for {source}: {exc}") from exc
        return result["document"]

    @staticmethod
    def _initial_state(source: str | Path, data: bytes | None, format: str | None) -> ParsingState:
        if data is None:
            path = Path(source)
            if not path.is_file():
                raise ParsingFailedError(f"file not found: {source}")
            data = path.read_bytes()
        state: ParsingState = {"source": str(source), "data": data}
        if format:
            state["format"] = format
        return state
