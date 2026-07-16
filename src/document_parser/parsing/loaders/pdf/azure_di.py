"""Azure Document Intelligence 추출 경로.

다이어그램 메모: 크롭 크기와 무관하게 페이지당 과금이라 크롭 없이 페이지
전체를 한 번에 요청한다(VLM 경로와 다름).

실제 in4u Document Intelligence 리소스로 연동 확인함(rnd-skep-commpf-di).
클라이언트가 돌려주는 bbox는 200dpi 렌더링 이미지 기준 픽셀 좌표라
coords.px_to_pt()로 변환해야 pymupdf 좌표계(포인트)와 맞는다 — layout.py의
PP-DocLayoutV2 좌표와 정확히 같은 이유로 같은 변환을 쓴다.

표 처리: DI가 페이지 전체에서 찾은 표(result.tables, 실제 행/열/병합 구조를
HTML로 가진 것)를 DetectedTable로 반환한다. PP-DocLayoutV2가 찾은 표 박스와는
독립적인 검출 결과라 서로 id가 없으므로, graph.py의 merge 노드가 bbox
겹침으로 어느 PaddleX 박스와 짝인지 판단해서 그 표 요소의 metadata["html"]에
채워 넣는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class DetectedTable:
    html: str
    bboxes: list[BBox] = field(default_factory=list)


@dataclass
class ContextParagraph:
    """DI가 찾은 문단 하나 — include_text=False라 TEXT 요소로는 안 만들어도
    (native가 이미 그 텍스트를 뽑고 있어서) 표 근처 문맥으로는 여전히 쓸모
    있다(청킹할 때 표 하나만 뚝 떼서 주는 것보다 주변 문단이 있으면 이해하기
    쉬움 — graph.py의 merge 노드가 표 bbox와 가까운 것들을 골라 표 요소의
    metadata["nearby_paragraphs"]에 붙인다)."""

    text: str
    bboxes: list[BBox] = field(default_factory=list)


@lru_cache(maxsize=1)
def _get_client() -> AzureDocumentIntelligenceClient:
    return AzureDocumentIntelligenceClient()


def extract_with_azure_di(
    page: pymupdf.Page, page_number: int, *, include_text: bool = True
) -> tuple[list[DocumentElement], list[DetectedTable], list[ContextParagraph]]:
    """페이지 전체를 DI로 분석한다.

    include_text=False면 문단(paragraphs) 기반 TEXT 요소는 안 만든다 — 텍스트
    레이어가 있어서 native가 이미 텍스트를 뽑고 있는 페이지에서, 표 구조만
    필요할 때 쓴다(불필요한 TEXT 중복 방지). 다만 문단 자체는 include_text와
    무관하게 항상 ContextParagraph로 반환한다 — 표 근처 문맥으로 붙일 때
    쓴다(nearby_paragraphs). 표(tables)도 include_text와 무관하게 항상 뽑는다.
    """
    client = _get_client()
    pix = page.get_pixmap(dpi=RENDER_DPI)
    result = client.analyze_layout(pix.tobytes("png"))

    context_paragraphs = [
        ContextParagraph(
            text=paragraph.text,
            bboxes=[
                BBox(x0=b[0], y0=b[1], x1=b[2], y1=b[3]) for b in map(px_to_pt, paragraph.bboxes)
            ],
        )
        for paragraph in result.paragraphs
    ]

    elements: list[DocumentElement] = []
    if include_text:
        for cp in context_paragraphs:
            elements.append(
                DocumentElement(
                    type=ElementType.TEXT,
                    text=cp.text,
                    page=page_number,
                    bboxes=cp.bboxes,
                    metadata={"source": "azure_di"},
                )
            )

    tables = [
        DetectedTable(
            html=table.html,
            bboxes=[BBox(x0=b[0], y0=b[1], x1=b[2], y1=b[3]) for b in map(px_to_pt, table.bboxes)],
        )
        for table in result.tables
    ]
    return elements, tables, context_paragraphs
