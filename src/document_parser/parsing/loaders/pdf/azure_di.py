"""Azure Document Intelligence 추출 경로.

다이어그램 메모: 크롭 크기와 무관하게 페이지당 과금이라 크롭 없이 페이지
전체를 한 번에 요청한다(VLM 경로와 다름).

지금은 고정 placeholder로 단락(short-circuit)시켜뒀다: in4u Azure DI
자격증명을 아직 못 써서, 실제 호출은 꺼두고(단순 미검증이 아니라 의도적으로
끔) 파싱이 실패하는 대신 계속 chunkable한 결과를 내게 했다. 실제
클라이언트(parsing.clients.azure_document_intelligence, 커밋 ec8c30d)는
그대로 남겨뒀다 — 자격증명이 준비되면 아래 placeholder를
`AzureDocumentIntelligenceClient` 호출로 바꾸면 된다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from document_parser.core.models import DocumentElement, ElementType

if TYPE_CHECKING:
    # 타입 힌트 전용 — layout.py 주석 참고.
    import pymupdf

_PLACEHOLDER_TEXT = "[Azure Document Intelligence not connected - placeholder]"


def extract_with_azure_di(page: pymupdf.Page, page_number: int) -> list[DocumentElement]:
    return [
        DocumentElement(
            type=ElementType.TEXT,
            text=_PLACEHOLDER_TEXT,
            page=page_number,
            metadata={"source": "azure_di", "stub": True},
        )
    ]
