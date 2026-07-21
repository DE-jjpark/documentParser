from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile

from document_parser import DocumentParserError, ParsedDocument, ParsingEngine, ParsingTier
from document_parser.api.deps import get_parsing_engine, to_http_error

router = APIRouter(tags=["parsing"])


@router.post("/parse", response_model=ParsedDocument)
async def parse_document(
    file: UploadFile,
    engine: Annotated[ParsingEngine, Depends(get_parsing_engine)],
    tier: ParsingTier = ParsingTier.BALANCED,
) -> ParsedDocument:
    data = await file.read()
    try:
        return await engine.aparse(file.filename or "upload", data=data, tier=tier)
    except DocumentParserError as exc:
        raise to_http_error(exc) from exc
