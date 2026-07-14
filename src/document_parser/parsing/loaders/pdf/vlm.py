"""VLM 그림 캡션 경로.

다이어그램 메모: AzureDI와 달리 먼저 크롭한 다음 VLM에 보낸다 — 페이지
전체보다 타이트하게 자른 영역이 정확도가 더 좋다.

지금은 고정 placeholder로 단락(short-circuit)시켜뒀다: VLM 호출은 이미지
1장당 비용이 발생해서, 자격증명 유무와 무관하게 이 경로가 실제로 필요해지기
전까지는 예산을 쓰지 않으려고 꺼둔 상태다. 실제 클라이언트(parsing.clients.vlm,
커밋 ec8c30d)는 그대로 남겨뒀다 — 준비되면 아래 placeholder를 `VLMClient`
호출로 바꾸면 된다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from document_parser.core.models import BBox, DocumentElement, ElementType

if TYPE_CHECKING:
    # 타입 힌트 전용 — layout.py 주석 참고.
    import pymupdf

    from document_parser.parsing.loaders.pdf.layout import LayoutBox

_PLACEHOLDER_CAPTION = "[image - VLM not connected, placeholder]"


def caption_figures(
    page: pymupdf.Page,
    page_number: int,
    boxes: list[LayoutBox],
) -> list[DocumentElement]:
    return [
        DocumentElement(
            type=ElementType.IMAGE,
            text=_PLACEHOLDER_CAPTION,
            page=page_number,
            bboxes=[BBox(x0=box.bbox[0], y0=box.bbox[1], x1=box.bbox[2], y1=box.bbox[3])],
            # layout_label: PP-DocLayoutV2의 25개 카테고리 중 실제 감지된 라벨
            # (예: "chart", "image", "seal" 등) — 그림이라는 것만 아는 게
            # 아니라 어떤 종류의 그림인지까지 남겨둔다.
            metadata={"source": "vlm", "stub": True, "layout_label": box.label},
        )
        for box in boxes
    ]
