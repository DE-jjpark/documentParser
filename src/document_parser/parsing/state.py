"""LangGraph state for the parsing graph. Internal to the parsing engine."""

from typing import TypedDict

from document_parser.core.models import DocumentElement, ParsedDocument


class ParsingState(TypedDict, total=False):
    source: str
    data: bytes
    format: str
    # "fast" | "balanced" (ParsingTier) — extract 노드가 로더에 그대로 넘긴다.
    # 문자열로 두는 이유: format과 같은 취급(TypedDict라 pydantic 검증이 없어서
    # ParsingEngine.parse()의 진입점에서 이미 ParsingTier(tier)로 검증 끝난
    # 값만 여기 들어온다).
    tier: str
    elements: list[DocumentElement]
    document: ParsedDocument
