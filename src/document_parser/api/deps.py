from fastapi import HTTPException, Request

from document_parser import (
    ChunkingEngine,
    DocumentParserError,
    IngestPipeline,
    MissingDependencyError,
    ParsingEngine,
    UnsupportedFormatError,
)


def get_parsing_engine(request: Request) -> ParsingEngine:
    return request.app.state.parsing_engine


def get_chunking_engine(request: Request) -> ChunkingEngine:
    return request.app.state.chunking_engine


def get_pipeline(request: Request) -> IngestPipeline:
    return request.app.state.pipeline


def to_http_error(exc: DocumentParserError) -> HTTPException:
    if isinstance(exc, UnsupportedFormatError | MissingDependencyError):
        return HTTPException(status_code=415, detail=str(exc))
    return HTTPException(status_code=422, detail=str(exc))
