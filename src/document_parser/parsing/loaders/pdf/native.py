"""텍스트 레이어가 있는 페이지의 네이티브 텍스트 추출.

레이아웃 분석(layout.py)이 실제 PP-DocLayoutV2 결과(``layout.boxes``)를 줬으면
그 영역별로 텍스트를 뽑아 25개 카테고리 중 실제 감지된 라벨을 그대로
``metadata["layout_label"]``에 남긴다 — 전엔 레이아웃 분석 결과를 라우팅
판단(has_figures)에만 쓰고 그 뒤엔 pymupdf 블록 단위로 처음부터 다시
추출하면서 라벨 정보를 통째로 버렸었다.

``layout.boxes``가 비어 있으면(휴리스틱 폴백이라 애초에 카테고리를 모르는
경우, 또는 극히 드물게 실제 모델이 영역을 하나도 못 찾은 경우) pymupdf
블록 단위 추출로 대체한다 — 텍스트 자체를 놓치지 않기 위한 안전망.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from document_parser.core.models import BBox, DocumentElement, ElementType

if TYPE_CHECKING:
    # 타입 힌트 전용 — layout.py 주석 참고.
    import pymupdf

    from document_parser.parsing.loaders.pdf.layout import LayoutBox, PageLayout

# PP-DocLayoutV2 라벨 -> ElementType. 대응하는 라벨이 없는 ElementType.LIST는
# 의도적으로 비어 있다 — PP-DocLayoutV2 25개 카테고리에 글머리 기호 목록에
# 해당하는 라벨이 없다(skep_parser 프로젝트의 DP-Bench 비교에서도 같은
# 한계를 확인함). 매핑에 없는 라벨은 ElementType.TEXT로 처리한다.
_LABEL_TO_TYPE = {
    "doc_title": ElementType.HEADING,
    "paragraph_title": ElementType.HEADING,
    "figure_title": ElementType.HEADING,
    "table": ElementType.TABLE,
}


def _label_to_element_type(label: str) -> ElementType:
    return _LABEL_TO_TYPE.get(label, ElementType.TEXT)


def extract_native(
    page: pymupdf.Page, page_number: int, layout: PageLayout
) -> list[DocumentElement]:
    if layout.text_boxes:
        return _extract_from_boxes(page, page_number, layout.text_boxes)
    return _extract_from_blocks(page, page_number)


def _extract_from_boxes(
    page: pymupdf.Page, page_number: int, boxes: list[LayoutBox]
) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    for box in boxes:
        text = page.get_text("text", clip=box.bbox).strip()
        if not text:
            continue
        x0, y0, x1, y1 = box.bbox
        elements.append(
            DocumentElement(
                type=_label_to_element_type(box.label),
                text=text,
                page=page_number,
                bboxes=[BBox(x0=x0, y0=y0, x1=x1, y1=y1)],
                metadata={"source": "native", "layout_label": box.label},
            )
        )
    return elements


def _extract_from_blocks(page: pymupdf.Page, page_number: int) -> list[DocumentElement]:
    """레이아웃 분석이 카테고리 정보를 못 줬을 때 쓰는 대체 경로 — 예전 구현
    그대로(pymupdf 블록 단위, 카테고리 구분 없이 전부 TEXT)."""
    elements: list[DocumentElement] = []
    for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks"):
        if block_type != 0:  # 1 = 이미지 블록; AzureDI+VLM 경로가 따로 처리
            continue
        text = text.strip()
        if not text:
            continue
        elements.append(
            DocumentElement(
                type=ElementType.TEXT,
                text=text,
                page=page_number,
                bboxes=[BBox(x0=x0, y0=y0, x1=x1, y1=y1)],
                metadata={"source": "native"},
            )
        )
    return elements
