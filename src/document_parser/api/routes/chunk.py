from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from document_parser import Chunk, ChunkingConfig, ChunkingEngine, DocumentParserError, Segment
from document_parser.api.deps import get_chunking_engine, to_http_error

router = APIRouter(tags=["chunking"])


class ChunkRequest(BaseModel):
    segments: list[Segment]
    config: ChunkingConfig = Field(default_factory=ChunkingConfig)


@router.post("/chunk", response_model=list[Chunk])
async def chunk_segments(
    request: ChunkRequest,
    engine: Annotated[ChunkingEngine, Depends(get_chunking_engine)],
) -> list[Chunk]:
    try:
        return await engine.achunk(request.segments, request.config)
    except DocumentParserError as exc:
        raise to_http_error(exc) from exc
