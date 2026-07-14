"""Run locally with: uvicorn document_parser.api.main:app --reload"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from document_parser import ChunkingEngine, IngestPipeline, ParsingEngine, __version__
from document_parser.api.routes import chunk, ingest, parse


@asynccontextmanager
async def lifespan(app: FastAPI):
    parsing_engine = ParsingEngine()
    chunking_engine = ChunkingEngine()
    app.state.parsing_engine = parsing_engine
    app.state.chunking_engine = chunking_engine
    app.state.pipeline = IngestPipeline(parsing_engine, chunking_engine)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="document-parser (local test)",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(parse.router, prefix="/v1")
    app.include_router(chunk.router, prefix="/v1")
    app.include_router(ingest.router, prefix="/v1")
    return app


app = create_app()
