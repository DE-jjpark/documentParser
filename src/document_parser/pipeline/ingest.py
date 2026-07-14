"""parse -> chunk composition. Engines stay unaware of each other."""

from pathlib import Path

from document_parser.chunking import ChunkingEngine
from document_parser.core.models import Chunk, ChunkingConfig, ParsedDocument, Segment
from document_parser.parsing import ParsingEngine


def document_to_segments(document: ParsedDocument) -> list[Segment]:
    """Convert the parsing contract into the chunking input contract."""
    segments: list[Segment] = []
    for element in document.elements:
        if not element.text.strip():
            continue
        metadata = {
            "source": document.source,
            "element_type": element.type.value,
            **({"page": element.page} if element.page is not None else {}),
            **element.metadata,
        }
        segments.append(Segment(text=element.text, metadata=metadata))
    return segments


class IngestPipeline:
    """End-to-end convenience: source document -> chunks."""

    def __init__(
        self,
        parsing_engine: ParsingEngine | None = None,
        chunking_engine: ChunkingEngine | None = None,
    ) -> None:
        self.parsing = parsing_engine or ParsingEngine()
        self.chunking = chunking_engine or ChunkingEngine()

    async def aingest(
        self,
        source: str | Path,
        *,
        data: bytes | None = None,
        format: str | None = None,
        config: ChunkingConfig | None = None,
    ) -> list[Chunk]:
        document = await self.parsing.aparse(source, data=data, format=format)
        return await self.chunking.achunk(document_to_segments(document), config)

    def ingest(
        self,
        source: str | Path,
        *,
        data: bytes | None = None,
        format: str | None = None,
        config: ChunkingConfig | None = None,
    ) -> list[Chunk]:
        document = self.parsing.parse(source, data=data, format=format)
        return self.chunking.chunk(document_to_segments(document), config)
