"""Public facade of the parsing engine."""

from pathlib import Path

from document_parser.core.exceptions import DocumentParserError, ParsingFailedError
from document_parser.core.models import ParsedDocument, ParsingTier
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
        tier: str | ParsingTier = ParsingTier.BALANCED,
    ) -> ParsedDocument:
        """Parse a document. Preferred entrypoint for async consumers.

        ``source`` is a file path (read from disk when ``data`` is omitted) or
        just a name used for format detection and metadata when ``data`` is
        passed directly. ``tier`` picks native-only ("fast", no AzureDI/VLM
        calls) vs the full pipeline ("balanced", default).
        """
        state = self._initial_state(source, data, format, tier)
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
        tier: str | ParsingTier = ParsingTier.BALANCED,
    ) -> ParsedDocument:
        """Synchronous convenience wrapper around the same graph."""
        state = self._initial_state(source, data, format, tier)
        try:
            result = self._graph.invoke(state)
        except DocumentParserError:
            raise
        except Exception as exc:
            raise ParsingFailedError(f"parsing failed for {source}: {exc}") from exc
        return result["document"]

    @staticmethod
    def _initial_state(
        source: str | Path, data: bytes | None, format: str | None, tier: str | ParsingTier
    ) -> ParsingState:
        if data is None:
            path = Path(source)
            if not path.is_file():
                raise ParsingFailedError(f"file not found: {source}")
            data = path.read_bytes()
        # ParsingTier(tier)가 잘못된 값이면 바로 ValueError -- 그래프 안까지
        # 들어가서야 실패하지 않도록 진입점에서 검증.
        state: ParsingState = {
            "source": str(source),
            "data": data,
            "tier": ParsingTier(tier).value,
        }
        if format:
            state["format"] = format
        return state
