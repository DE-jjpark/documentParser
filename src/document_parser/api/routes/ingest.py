from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile

from document_parser import Chunk, ChunkingConfig, DocumentParserError, IngestPipeline, ParsingTier
from document_parser.api.deps import get_pipeline, to_http_error

router = APIRouter(tags=["pipeline"])


@router.post("/ingest", response_model=list[Chunk])
async def ingest_document(
    file: UploadFile,
    pipeline: Annotated[IngestPipeline, Depends(get_pipeline)],
    tier: ParsingTier = ParsingTier.BALANCED,
    strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    data = await file.read()
    config = ChunkingConfig(strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    try:
        return await pipeline.aingest(
            file.filename or "upload", data=data, tier=tier, config=config
        )
    except DocumentParserError as exc:
        raise to_http_error(exc) from exc
