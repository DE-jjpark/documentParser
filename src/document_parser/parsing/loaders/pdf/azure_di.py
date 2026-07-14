"""Azure Document Intelligence 추출 경로.

다이어그램 메모: 크롭 크기와 무관하게 페이지당 과금이라 크롭 없이 페이지
전체를 한 번에 요청한다(VLM 경로와 다름).

실제 in4u Document Intelligence 리소스로 연동 확인함(rnd-skep-commpf-di).
클라이언트가 돌려주는 bbox는 200dpi 렌더링 이미지 기준 픽셀 좌표라
coords.px_to_pt()로 변환해야 pymupdf 좌표계(포인트)와 맞는다 — layout.py의
PP-DocLayoutV2 좌표와 정확히 같은 이유로 같은 변환을 쓴다.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from document_parser.core.models import BBox, DocumentElement, ElementType
from document_parser.parsing.clients.azure_document_intelligence import (
    AzureDocumentIntelligenceClient,
)
from document_parser.parsing.loaders.pdf.coords import RENDER_DPI, px_to_pt

if TYPE_CHECKING:
    # 타입 힌트 전용 — layout.py 주석 참고.
    import pymupdf


@lru_cache(maxsize=1)
def _get_client() -> AzureDocumentIntelligenceClient:
    return AzureDocumentIntelligenceClient()


def extract_with_azure_di(page: pymupdf.Page, page_number: int) -> list[DocumentElement]:
    client = _get_client()
    pix = page.get_pixmap(dpi=RENDER_DPI)
    result = client.analyze_layout(pix.tobytes("png"))

    elements: list[DocumentElement] = []
    for paragraph in result.paragraphs:
        bboxes = [BBox(x0=b[0], y0=b[1], x1=b[2], y1=b[3]) for b in map(px_to_pt, paragraph.bboxes)]
        elements.append(
            DocumentElement(
                type=ElementType.TEXT,
                text=paragraph.text,
                page=page_number,
                bboxes=bboxes,
                metadata={"source": "azure_di"},
            )
        )
    return elements
