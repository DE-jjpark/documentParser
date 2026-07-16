"""VLM 그림 캡션 경로.

다이어그램 메모: AzureDI와 달리 먼저 크롭한 다음 VLM에 보낸다 — 페이지
전체보다 타이트하게 자른 영역이 정확도가 더 좋다.

실제 in4u Databricks AI Gateway(Claude Sonnet 4.6)로 연동 확인함 —
parsing.clients.vlm.VLMClient 참고. 크롭은 pymupdf의 get_pixmap(clip=...)로
하는데, clip은 포인트 좌표를 받으므로(픽셀 변환 불필요) box.bbox를 그대로
쓴다 — layout.py/azure_di.py에서 모델이 "돌려주는" 좌표를 변환해야 했던 것과는
반대 방향이라 혼동하지 말 것.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from document_parser.core.models import BBox, DocumentElement, ElementType
from document_parser.parsing.clients.vlm import VLMClient

if TYPE_CHECKING:
    # 타입 힌트 전용 — layout.py 주석 참고.
    import pymupdf

    from document_parser.parsing.loaders.pdf.layout import LayoutBox

_PROMPT = (
    "Describe the content of this figure concisely. If it's a chart, table-like "
    "image, or diagram, describe its structure and key values."
)


@lru_cache(maxsize=1)
def _get_client() -> VLMClient:
    return VLMClient()


def caption_figures(
    page: pymupdf.Page,
    page_number: int,
    boxes: list[LayoutBox],
) -> list[DocumentElement]:
    if not boxes:
        return []

    client = _get_client()
    elements: list[DocumentElement] = []
    for box in boxes:
        pix = page.get_pixmap(clip=box.bbox, dpi=200)
        caption = client.caption_image(pix.tobytes("png"), _PROMPT)
        x0, y0, x1, y1 = box.bbox
        elements.append(
            DocumentElement(
                type=ElementType.IMAGE,
                text=caption,
                page=page_number,
                bboxes=[BBox(x0=x0, y0=y0, x1=x1, y1=y1)],
                # layout_label: PP-DocLayoutV2의 25개 카테고리 중 실제 감지된 라벨
                # (예: "chart", "image", "seal" 등) — 그림이라는 것만 아는 게
                # 아니라 어떤 종류의 그림인지까지 남겨둔다.
                metadata={
                    "source": "vlm",
                    "layout_label": box.label,
                    "layout_cls_id": box.cls_id,
                    "layout_box_index": box.box_index,
                },
            )
        )
    return elements
